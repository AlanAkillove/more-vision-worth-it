"""Uncertainty baselines for predicting WC (plan §11 / M5).

At the LOW resolution (112) we compute four scalar confidence / uncertainty signals and
ask how well each predicts the marginal-value target

    WC = 1  <=>  the sample is wrong at 112 but correct at 448 (recoverable escalation).

Each signal is turned into an ESCALATION score oriented so that "escalate to high
resolution" corresponds to a HIGHER score (uncertainty routers escalate the unsure):

    max_prob  -> 1 - top1_confidence     (less confident => escalate)
    entropy   -> entropy                  (more entropic  => escalate)
    margin    -> -(top1-top2 margin)      (smaller margin => escalate)
    energy    -> energy (-logsumexp)       (higher energy  => escalate)

We report AUROC and AUPRC (AUPRC matters more because WC can be imbalanced), the full
ROC / precision-recall curve points, and a binned estimate of P(WC | uncertainty bin).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# escalation-oriented transform of each stored low-res signal
SCORE_TRANSFORMS = {
    "max_prob": lambda d: 1.0 - d["top1_confidence"].to_numpy(),
    "entropy": lambda d: d["entropy"].to_numpy(),
    "margin": lambda d: -d["top1_top2_margin"].to_numpy(),
    "energy": lambda d: d["energy"].to_numpy(),
}


def build_pair_frame(df, low, high):
    """Join low- and high-resolution rows per sample_id into one analysis frame.

    Low-resolution signals are used for prediction/routing; high-resolution correctness
    defines the recoverable (WC) label. Guarantees exactly one row per sample_id.
    """
    avail = sorted(int(r) for r in df["resolution"].unique())
    sub_lo = df[df["resolution"] == int(low)]
    sub_hi = df[df["resolution"] == int(high)]
    if len(sub_lo) == 0 or len(sub_hi) == 0:
        raise ValueError(
            f"no outcome rows for resolution {low} or {high} (available resolutions: {avail})")
    lo = sub_lo[
        ["sample_id", "correct", "top1_confidence", "entropy", "top1_top2_margin", "energy"]
    ].rename(columns={"correct": "correct_low"}).copy()
    hi = sub_hi[
        ["sample_id", "correct", "top1_confidence"]
    ].rename(columns={"correct": "correct_high", "top1_confidence": "conf_high"}).copy()
    pair = lo.merge(hi, on="sample_id", how="inner")
    if pair.empty:
        raise ValueError(f"pair frame is empty for transition {low}->{high} (no shared sample_ids)")
    if pair["sample_id"].duplicated().any():
        raise ValueError("pair frame has duplicate sample_id (check outcomes uniqueness)")
    pair["wc"] = ((~pair["correct_low"].astype(bool)) & pair["correct_high"].astype(bool)).astype(int)
    pair["cw"] = (pair["correct_low"].astype(bool) & (~pair["correct_high"].astype(bool))).astype(int)
    return pair


def escalation_scores(pair):
    """Return {score_name: escalation_score_vector} aligned to pair rows (in order)."""
    return {name: fn(pair).astype(np.float64) for name, fn in SCORE_TRANSFORMS.items()}


def evaluate_scores(pair, score_names):
    """AUROC / AUPRC of each escalation score against the WC label."""
    y = pair["wc"].to_numpy()
    scores = escalation_scores(pair)
    out = {}
    for name in score_names:
        s = scores[name]
        try:
            auroc = float(roc_auc_score(y, s))
        except ValueError:
            auroc = float("nan")
        auprc = float(average_precision_score(y, s))
        out[name] = {
            "n": int(len(y)), "n_pos_wc": int(y.sum()),
            "base_rate": float(y.mean()) if len(y) else float("nan"),
            "auroc": round(auroc, 5), "auprc": round(auprc, 5),
        }
    return out


def curve_data(pair, score_names):
    """ROC and precision-recall curve points for each escalation score (long format)."""
    y = pair["wc"].to_numpy()
    scores = escalation_scores(pair)
    roc_rows, pr_rows = [], []
    for name in score_names:
        s = scores[name]
        fpr, tpr, _ = roc_curve(y, s)
        for i in range(len(fpr)):
            roc_rows.append({"score": name, "fpr": float(fpr[i]), "tpr": float(tpr[i])})
        prec, rec, _ = precision_recall_curve(y, s)
        for i in range(len(rec)):
            pr_rows.append({"score": name, "recall": float(rec[i]), "precision": float(prec[i])})
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def binned_p_wc(pair, score_col="top1_confidence", n_bins=10, by="quantile"):
    """Estimate P(WC | uncertainty bin). score_col is a stored low-res column name.

    Bins are on the RAW signal (e.g. confidence); the plotted/recorded x-axis keeps that
    raw meaning so high-confidence recoverable mass is visible at the confident end.
    """
    x = pair[score_col].to_numpy(dtype=np.float64)
    wc = pair["wc"].to_numpy()
    if by == "quantile":
        qs = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        qs[0] -= 1e-9
        edges = np.unique(qs)
    else:
        edges = np.linspace(x.min(), x.max() + 1e-9, n_bins + 1)
    if len(edges) < 2:  # zero-variance signal: a single degenerate bin
        return pd.DataFrame([{"signal": score_col, "bin": 0, "bin_lo": float(edges[0]),
                              "bin_hi": float(edges[0]), "n": int(len(x)),
                              "n_wc": int(wc.sum()), "p_wc": float(wc.mean()) if len(x) else float("nan")}])
    idx = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "signal": score_col, "bin": b,
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "n": n, "n_wc": int(wc[m].sum()) if n else 0,
            "p_wc": float(wc[m].mean()) if n else float("nan"),
        })
    return pd.DataFrame(rows)
