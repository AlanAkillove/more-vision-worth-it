"""CLI 09: lightweight predictability probe A/B + bootstrap + Q3 (plan §15 / §9 / M6).

Fits two logistic-regression probes (A: low-res state; B: + z_112 embedding) to predict
WC on a temporary, WC-stratified VAL fit/eval split, then scores both on the held-out eval
fold against the best scalar uncertainty baseline -- all on the SAME rows. The Q3 verdict
answers whether the marginal visual value Q2 exposed is cheaply LEARNABLE (GO for a
decision model) or not (NO-GO / weak predictability). Exploratory audit, not a final result.

Run it yourself (fast; a handful of LogisticRegression fits + bootstrap):
  conda activate deepminer
  python scripts/09_predictability_probe.py --config configs/probe.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from mvwi import config as C
from mvwi.audit import outcomes as OC
from mvwi.audit import probe as PR
from mvwi.audit import q3 as Q3
from mvwi.audit import uncertainty as UN
from mvwi.preprocess.version import compute_preproc_hash

# decision fields the probe reuses from analysis.yaml (probe.yaml does not redefine them)
_SHARED_FROM_ANALYSIS = ["main_transition", "uncertainty_scores", "escalation_rates", "cost", "utility"]


def _policy_metric_rows(name, m, is_base, boot_entry):
    """Tidy rows (one per scalar metric / per rho) for probe_metrics.csv."""
    rows = [
        {"policy": name, "role": "baseline" if is_base else "probe", "metric": "auroc", "rho": "",
         "value": m["auroc"], "diff_ci_low": None, "diff_ci_high": None},
        {"policy": name, "role": "baseline" if is_base else "probe", "metric": "auprc", "rho": "",
         "value": m["auprc"],
         "diff_ci_low": boot_entry["auprc_diff_ci"]["low"] if boot_entry else None,
         "diff_ci_high": boot_entry["auprc_diff_ci"]["high"] if boot_entry else None},
    ]
    for rho, r in sorted(m["routing"].items()):
        rows.append({"policy": name, "role": "baseline" if is_base else "probe",
                     "metric": "wc_recall", "rho": rho, "value": r["wc_recall"],
                     "diff_ci_low": (boot_entry["wc_recall_diff_ci"].get(str(rho)) or {}).get("low") if boot_entry else None,
                     "diff_ci_high": (boot_entry["wc_recall_diff_ci"].get(str(rho)) or {}).get("high") if boot_entry else None})
        rows.append({"policy": name, "role": "baseline" if is_base else "probe",
                     "metric": "final_accuracy", "rho": rho, "value": r["final_accuracy"],
                     "diff_ci_low": None, "diff_ci_high": None})
        rows.append({"policy": name, "role": "baseline" if is_base else "probe",
                     "metric": "routing_regret", "rho": rho, "value": r["routing_regret"],
                     "diff_ci_low": None, "diff_ci_high": None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="probe.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    ana = C.load_config("analysis.yaml")
    for k in _SHARED_FROM_ANALYSIS:
        cfg.setdefault(k, ana[k])

    logger = C.setup_logger("09_predictability_probe", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    tables = C.R(cfg["outputs"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    low, high = cfg["main_transition"]
    cost, ucfg = cfg["cost"], cfg["utility"]
    rhos = [float(r) for r in cfg["escalation_rates"]]
    score_names = cfg["uncertainty_scores"]
    pcfg = cfg["probe"]
    scfg = cfg["split"]
    bcfg = cfg["bootstrap"]

    df = OC.read_outcomes(cfg, phash, "val")
    pair = UN.build_pair_frame(df, low, high)
    blocks = PR.assemble_features(cfg, phash, pair, low)

    y_all = pair["wc"].to_numpy()
    cl_all = pair["correct_low"].to_numpy(dtype=bool)
    ch_all = pair["correct_high"].to_numpy(dtype=bool)
    fit_idx, eval_idx = PR.temp_val_split(pair, scfg["eval_fraction"], scfg["seed"])
    logger.info("probe split: fit=%d eval=%d (stratified on WC, mode=%s); eval WC=%d/%d",
                len(fit_idx), len(eval_idx), scfg["mode"], int(y_all[eval_idx].sum()), len(eval_idx))

    y = y_all[eval_idx]
    cl, ch = cl_all[eval_idx], ch_all[eval_idx]

    # fit each probe on the fit fold, score P(WC) on the disjoint eval fold
    scores_by_policy = {}
    probe_features = {"probe_a": cfg["probe_a_features"], "probe_b": cfg["probe_b_features"]}
    for pname, feats in probe_features.items():
        X = PR.make_matrix(blocks, feats)
        scaler, model = PR.fit_probe(X[fit_idx], y_all[fit_idx], pcfg)
        scores_by_policy[pname] = PR.predict_wc(scaler, model, X[eval_idx])
        logger.info("%s fitted on %d rows (%d features), evaluated on %d",
                    pname, len(fit_idx), X.shape[1], len(eval_idx))

    # scalar uncertainty baselines, on the SAME eval rows
    scalars = UN.escalation_scores(pair)
    for name in score_names:
        scores_by_policy[name] = np.asarray(scalars[name])[eval_idx]

    metrics = {k: PR.evaluate_policy_all(s, y, cl, ch, cost, ucfg, rhos)
               for k, s in tqdm(scores_by_policy.items(), desc="policies", unit="policy")}
    base_name = max(score_names, key=lambda n: metrics[n]["auprc"])
    logger.info("best scalar baseline (eval AUPRC): %s=%.4f", base_name, metrics[base_name]["auprc"])

    boot = PR.bootstrap_differences(scores_by_policy, y, cl, ch, cost, ucfg, rhos,
                                    base_name=base_name, n=bcfg["n"], seed=bcfg["seed"], ci=bcfg["ci"])

    probe_metrics = {p: metrics[p] for p in probe_features}
    q3res = Q3.evaluate_q3(probe_metrics, metrics[base_name], boot, rhos, cfg["q3"])
    q3res["best_scalar"] = {"name": base_name, **metrics[base_name]}
    q3res["split"] = {"mode": scfg["mode"], "n_fit": int(len(fit_idx)), "n_eval": int(len(eval_idx)),
                      "eval_fraction": float(scfg["eval_fraction"]), "stratify_on": scfg["stratify_on"]}

    for p in probe_features:
        c = q3res["comparisons"][p]
        logger.info("Q3 %s vs best_scalar(%s): auprc_gain=%s regret_reduction=%s recall_pts=%d/3 status=%s",
                    p, base_name, c["auprc_gain"], c["regret_reduction"],
                    c["n_points_meeting_recall_gate"], c["status"])
    logger.info("Q3 verdict=%s", q3res["verdict"])

    rows = []
    for name, m in metrics.items():
        entry = boot.get(name)
        rows.extend(_policy_metric_rows(name, m, is_base=(name == base_name), boot_entry=entry))
    pd.DataFrame(rows).to_csv(tables / "probe_metrics.csv", index=False)

    roc_df, pr_df = PR.curve_data(scores_by_policy, y)
    roc_df["fold"] = "val_eval"
    pr_df["fold"] = "val_eval"
    roc_df.to_csv(tables / "probe_curves.csv", index=False)
    pr_df.rename(columns={"recall": "pr_recall"}).assign(curve="pr").to_csv(
        tables / "probe_pr_curves.csv", index=False)

    C.update_metrics(cfg, "probe", {
        "preproc_hash": phash, "main_transition": [low, high],
        "target": cfg["target"], "features": probe_features,
        "split": q3res["split"], "best_scalar": {"name": base_name, **metrics[base_name]},
        "policies": metrics,
    })
    C.update_metrics(cfg, "q3", q3res)
    C.write_run_config(cfg, script="09_predictability_probe.py",
                       extra={"preproc_hash": phash, "q3_verdict": q3res["verdict"]})
    logger.info("predictability probe complete")


if __name__ == "__main__":
    main()
