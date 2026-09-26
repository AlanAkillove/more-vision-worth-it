"""Phase-0 lightweight predictability probe (plan §15 / §9 / M6).

Q2 (PASS) told us a *learned* value router has headroom over scalar uncertainty routing.
Before committing to a neural decision model we ask a cheaper question: can a plain
logistic-regression probe, looking only at the LOW-resolution (112) state, predict the
marginal-value target

    WC = 1  <=>  wrong at 112 but correct at 448

better than the best scalar uncertainty baseline? If not, the headroom Q2 exposed is not
readably LEARNABLE from cheap features and we should stop short of a Transformer.

Two probes (both multinomial-free binary logistic regression on P(WC)):
  Probe A : logits, confidence, entropy, margin, energy
  Probe B : z_112, logits, confidence, entropy, margin, energy

Leakage discipline: the shared linear backbone was fit on TRAIN, so its VAL logits /
embeddings are already out-of-sample. We still never fit a probe on the rows it is scored
on -- VAL is split into DISJOINT fit / eval folds (stratified on WC). The fit fold trains
the probe; every metric (AUROC / AUPRC / recall@rho / routing regret) is reported on the
held-out eval fold, for the probe AND for each scalar baseline, so comparisons are apples
to apples. This is an EXPLORATORY audit on a temporary VAL split (flagged in the report),
not a final result.

Bootstrap: we resample the eval fold with replacement and report 2.5 / 97.5 percentile CIs
for the probe-minus-baseline differences, so Q3 can require the gain to sit outside noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from mvwi.audit import outcomes as OC
from mvwi.audit import uncertainty as UN
from mvwi.audit.oracle import utility_vector
from mvwi.audit.routing import routing_metrics, top_rho_mask
from mvwi.features import extract as EX

# map probe.yaml feature names -> the low-res column stored on the pair frame
_SCALAR_COLS = {
    "confidence": "top1_confidence",
    "entropy": "entropy",
    "margin": "top1_top2_margin",
    "energy": "energy",
}


def _as_str_array(x):
    return np.asarray([str(v) for v in x])


def assemble_features(cfg, phash, pair, low):
    """Build per-sample feature blocks aligned row-for-row to `pair` (which is 112-order).

    Returns {block_name: ndarray}: 'logits' (n x C), 'z_112' (n x D), and each scalar as a
    length-n column. logits come from the outcome table's low-resolution rows; z_112 from
    the frozen-backbone feature cache; scalars are already on the pair frame.
    """
    df = OC.read_outcomes(cfg, phash, "val")
    lo = df[df["resolution"] == int(low)][["sample_id", "logits"]]
    lut = {str(s): lg for s, lg in zip(lo["sample_id"].to_numpy(), lo["logits"].to_numpy())}
    pids = _as_str_array(pair["sample_id"].to_numpy())
    missing = [s for s in pids if s not in lut]
    if missing:
        raise ValueError(f"{len(missing)} pair sample_ids have no low-res logits (e.g. {missing[:3]})")
    logits = np.stack([np.asarray(lut[s], dtype=np.float64) for s in pids]).astype(np.float64)

    f = EX.load_features(cfg, phash, low, "val")
    emb_lut = {str(s): e for s, e in zip(_as_str_array(f["sample_id"]), f["embedding"])}
    missing_emb = [s for s in pids if s not in emb_lut]
    if missing_emb:
        raise ValueError(f"{len(missing_emb)} pair sample_ids have no cached embedding (e.g. {missing_emb[:3]})")
    z = np.stack([np.asarray(emb_lut[s], dtype=np.float64) for s in pids]).astype(np.float64)

    blocks = {"logits": logits, "z_112": z}
    for name, col in _SCALAR_COLS.items():
        blocks[name] = pair[col].to_numpy(dtype=np.float64).reshape(-1, 1)
    return blocks


def make_matrix(blocks, feature_names):
    """Horizontal-concatenate the requested feature blocks, in config order."""
    cols = []
    for name in feature_names:
        if name not in blocks:
            raise ValueError(f"unknown probe feature {name!r} (available: {sorted(blocks)})")
        a = blocks[name]
        cols.append(a.reshape(len(a), -1))
    return np.hstack(cols)


def temp_val_split(pair, eval_fraction, seed):
    """Disjoint (fit_idx, eval_idx) stratified on WC. Returns integer positional indices."""
    y = pair["wc"].to_numpy()
    n = len(y)
    n_eval = int(round(eval_fraction * n))
    n_eval = max(1, min(n - 1, n_eval))
    ss = StratifiedShuffleSplit(n_splits=1, test_size=n_eval, random_state=int(seed))
    fit_idx, eval_idx = next(ss.split(np.zeros(n), y))
    if len(set(fit_idx.tolist()) & set(eval_idx.tolist())):
        raise ValueError("fit/eval folds overlap (leakage guard violated)")
    return fit_idx.astype(int), eval_idx.astype(int)


def fit_probe(X, y, pcfg):
    """Standardize + binary logistic regression on the WC label. Returns (scaler, model)."""
    scaler = StandardScaler() if pcfg.get("standardize", True) else None
    Xi = scaler.fit_transform(X) if scaler is not None else X
    model = LogisticRegression(
        solver=pcfg.get("solver", "lbfgs"),
        C=float(pcfg.get("C", 1.0)),
        max_iter=int(pcfg.get("max_iter", 2000)),
    )
    model.fit(Xi, y)
    return scaler, model


def predict_wc(scaler, model, X):
    Xi = scaler.transform(X) if scaler is not None else X
    return model.predict_proba(Xi)[:, 1]


def _safe_auroc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def escalation_policy_metrics(score, cl, ch, cost, ucfg, rhos):
    """Router metrics for ONE escalation score at each rho on the given rows.

    Higher score -> escalate. Returns per-rho wc_recall, final_accuracy and the *routing
    regret* (sequential-oracle escalation utility minus this policy's utility), all on the
    SAME rows so probe and scalar baselines are directly comparable.
    """
    out = {}
    u = utility_vector(cl, ch, ucfg)
    oracle_mask_base = u  # reused below per rho
    for rho in rhos:
        mask = top_rho_mask(score, rho)
        rm = routing_metrics(cl, ch, mask, cost)
        oracle_mask = top_rho_mask(oracle_mask_base, rho)
        oracle_util = float(u[oracle_mask].sum())
        policy_util = float(u[mask].sum())
        regret = oracle_util - policy_util
        rel = regret / oracle_util if oracle_util > 0 else float("nan")
        out[rho] = {
            "rho": rho,
            "wc_recall": rm["wc_recall"],
            "final_accuracy": rm["final_accuracy"],
            "escalation_utility": round(policy_util, 6),
            "oracle_utility": round(oracle_util, 6),
            "routing_regret": round(regret, 6),
            "relative_regret": None if rel != rel else round(rel, 6),
        }
    return out


def evaluate_policy_all(score, y, cl, ch, cost, ucfg, rhos):
    """AUROC / AUPRC + per-rho routing metrics for a single escalation score."""
    m = {
        "auroc": round(_safe_auroc(y, score), 5),
        "auprc": round(float(average_precision_score(y, score)), 5),
    }
    m["routing"] = escalation_policy_metrics(score, cl, ch, cost, ucfg, rhos)
    return m


def _bootstrap_once(scores_by_policy, y, cl, ch, cost, ucfg, rhos, rng):
    """One resample of eval rows -> compact metric snapshot per policy."""
    n = len(y)
    idx = rng.integers(0, n, size=n)
    yb, clb, chb = y[idx], cl[idx], ch[idx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        return None
    snap = {}
    for name, s in scores_by_policy.items():
        sb = s[idx]
        routing = escalation_policy_metrics(sb, clb, chb, cost, ucfg, rhos)
        snap[name] = {
            "auprc": float(average_precision_score(yb, sb)),
            "auroc": _safe_auroc(yb, sb),
            "wc_recall": {r: routing[r]["wc_recall"] for r in rhos},
            "mean_regret": float(np.mean([routing[r]["routing_regret"] for r in rhos])),
        }
    return snap


def bootstrap_differences(scores_by_policy, y, cl, ch, cost, ucfg, rhos, base_name, n, seed, ci):
    """Percentile CIs for (probe - base) on auprc, per-rho recall, and mean regret.

    `base_name` is the reference policy (best scalar baseline). Any policy key other than
    base_name is bootstrapped against it; CIs are stored under that policy's name.
    """
    rng = np.random.default_rng(int(seed))
    lo_q, hi_q = float(ci[0]), float(ci[1])
    accum = {}
    for _ in range(int(n)):
        snap = _bootstrap_once(scores_by_policy, y, cl, ch, cost, ucfg, rhos, rng)
        if snap is None:
            continue
        base = snap[base_name]
        for name in scores_by_policy:
            if name == base_name:
                continue
            d = accum.setdefault(name, {"auprc": [], "mean_regret": [], "wc_recall": {r: [] for r in rhos}})
            d["auprc"].append(snap[name]["auprc"] - base["auprc"])
            d["mean_regret"].append(snap[name]["mean_regret"] - base["mean_regret"])
            for r in rhos:
                d["wc_recall"][r].append(snap[name]["wc_recall"][r] - base["wc_recall"][r])

    def _ci(vals):
        a = np.asarray(vals, dtype=np.float64)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return {"low": None, "high": None}
        return {"low": round(float(np.percentile(a, lo_q)), 5),
                "high": round(float(np.percentile(a, hi_q)), 5)}

    out = {}
    for name, d in accum.items():
        out[name] = {
            "auprc_diff_ci": _ci(d["auprc"]),
            "mean_regret_diff_ci": _ci(d["mean_regret"]),
            "wc_recall_diff_ci": {str(r): _ci(d["wc_recall"][r]) for r in rhos},
            "n_effective": len(d["auprc"]),
        }
    return out


def curve_data(scores_by_policy, y):
    """ROC / PR curve points (long format) for every policy, for probe_curves.csv."""
    roc_rows, pr_rows = [], []
    for name, s in scores_by_policy.items():
        fpr, tpr, _ = roc_curve(y, s)
        for i in range(len(fpr)):
            roc_rows.append({"policy": name, "fpr": float(fpr[i]), "tpr": float(tpr[i])})
        prec, rec, _ = precision_recall_curve(y, s)
        for i in range(len(rec)):
            pr_rows.append({"policy": name, "recall": float(rec[i]), "precision": float(prec[i])})
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)
