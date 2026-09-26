"""CLI 12: single nonlinear learnability probe + unchanged Q3 gate (brief F3-F6).

Trains the FIXED tiny MLP on strictly-112-resolution state with 5-fold x 3-seed out-of-fold
prediction, re-runs the two Phase-0 logistic probes under the SAME OOF protocol, evaluates
every policy as an escalation router against the four scalar uncertainty baselines, and
applies the UNCHANGED Q3 gate (recall / regret / AUPRC / bootstrap-noise) versus the BEST
scalar baseline. This one test is allowed to terminate the project; NO-GO is reported as is.

Run it yourself (GPU recommended but fast; MLP over 15 fold-seed fits):
  conda activate deepminer
  python scripts/12_followup_nonlinear_probe.py --config configs/followup.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mvwi import config as C
from mvwi.audit import followup as FU
from mvwi.audit import mlp_probe as MLP
from mvwi.audit import outcomes as OC
from mvwi.audit import probe as PR
from mvwi.audit import q3 as Q3
from mvwi.audit import uncertainty as UN
from mvwi.preprocess.version import compute_preproc_hash

_SHARED = ["main_transition", "uncertainty_scores", "escalation_rates", "cost", "utility"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="followup.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    ana = C.load_config("analysis.yaml")
    for k in _SHARED:
        cfg.setdefault(k, ana[k])
    logger = C.setup_logger("12_followup_nonlinear_probe", cfg)
    C.set_seed(cfg["project"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    tables = C.R(cfg["outputs"]["tables_dir"]); tables.mkdir(parents=True, exist_ok=True)

    low, high = cfg["main_transition"]
    cost, ucfg = cfg["cost"], cfg["utility"]
    rhos = [float(r) for r in cfg["escalation_rates"]]
    score_names = cfg["uncertainty_scores"]
    cv_cfg, mcfg = cfg["cv"], cfg["mlp"]
    lcfg = cfg["logistic_probes"]
    bcfg = cfg["bootstrap"]

    df = OC.read_outcomes(cfg, phash, "val")
    pair = UN.build_pair_frame(df, low, high)
    blocks = PR.assemble_features(cfg, phash, pair, low)
    y = pair["wc"].to_numpy().astype(np.int64)
    cl = pair["correct_low"].to_numpy(dtype=bool)
    ch = pair["correct_high"].to_numpy(dtype=bool)
    logger.info("val pair n=%d WC=%d CW=%d; device=%s; input_dim=%d",
                len(pair), int(y.sum()), int(pair["cw"].sum()), device,
                PR.make_matrix(blocks, cfg["features"]).shape[1])

    # --- pooled out-of-fold scores for every learned policy ---
    X_mlp = PR.make_matrix(blocks, cfg["features"])
    oof_mlp, mlp_records = MLP.oof_mlp_probs(
        X_mlp, y, cv_cfg, mcfg, device=device, logger=logger,
        inner_fraction=float(cv_cfg["inner_val_fraction"]), inner_seed=int(cv_cfg["inner_seed"]))

    X_a = PR.make_matrix(blocks, lcfg["logit_only"])
    X_b = PR.make_matrix(blocks, lcfg["embedding_logit"])
    oof_a = FU.oof_logistic_probs(X_a, y, cv_cfg, lcfg)
    oof_b = FU.oof_logistic_probs(X_b, y, cv_cfg, lcfg)
    logger.info("logistic OOF probes re-run under the same 5-fold protocol")

    scalars = UN.escalation_scores(pair)
    scores_by_policy = {"mlp": oof_mlp, "logistic_logit_only": oof_a,
                        "logistic_embedding_logit": oof_b}
    for name in score_names:
        scores_by_policy[name] = np.asarray(scalars[name])

    # --- evaluate every policy as a budget-matched escalation router ---
    metrics = {k: FU.extended_policy_metrics(s, y, cl, ch, cost, ucfg, rhos)
               for k, s in tqdm(scores_by_policy.items(), desc="policies", unit="policy")}
    best_scalar = FU.pick_best_scalar(metrics, score_names)
    logger.info("best scalar baseline (pooled val AUPRC): %s=%.4f", best_scalar,
                metrics[best_scalar]["auprc"])
    logger.info("MLP AUPRC=%.4f AUROC=%.4f | logistic A=%.4f B=%.4f",
                metrics["mlp"]["auprc"], metrics["mlp"]["auroc"],
                metrics["logistic_logit_only"]["auprc"], metrics["logistic_embedding_logit"]["auprc"])

    boot = PR.bootstrap_differences(scores_by_policy, y, cl, ch, cost, ucfg, rhos,
                                    base_name=best_scalar, n=bcfg["n"], seed=bcfg["seed"],
                                    ci=bcfg["ci"])
    q3res = Q3.evaluate_q3({"mlp": metrics["mlp"]}, metrics[best_scalar], boot, rhos, cfg["q3"])
    q3res["best_scalar"] = {"name": best_scalar, **metrics[best_scalar]}
    q3res["mlp"] = {"auroc": metrics["mlp"]["auroc"], "auprc": metrics["mlp"]["auprc"]}
    q3res["cv"] = {"n_folds": int(cv_cfg["n_folds"]), "seeds": list(cv_cfg["seeds"]),
                   "input_dim": int(X_mlp.shape[1]), "features": cfg["features"],
                   "mean_best_epoch": round(float(np.mean([r["best_epoch"] for r in mlp_records])), 2),
                   "mean_inner_val_loss": round(float(np.mean([r["best_inner_val_loss"] for r in mlp_records])), 5)}

    c = q3res["comparisons"]["mlp"]
    logger.info("Q3-followup mlp vs best_scalar(%s): auprc_gain=%s regret_reduction=%s "
                "recall_pts=%d/3 gates=%s status=%s", best_scalar, c["auprc_gain"],
                c["regret_reduction"], c["n_points_meeting_recall_gate"], c["gates"], c["status"])
    verdict = q3res["verdict"]
    logger.info("Q3 FOLLOW-UP verdict=%s", verdict)

    # --- tables ---
    FU.routing_metrics_table(metrics, rhos).to_csv(tables / "followup_policy_metrics.csv", index=False)
    pd.DataFrame(mlp_records).to_csv(tables / "followup_mlp_training_log.csv", index=False)
    boot_rows = []
    for name, b in boot.items():
        boot_rows.append({"policy": name, "metric": "auprc_diff", "rho": "",
                          "ci_low": b["auprc_diff_ci"]["low"], "ci_high": b["auprc_diff_ci"]["high"],
                          "n_effective": b["n_effective"]})
        boot_rows.append({"policy": name, "metric": "mean_regret_diff", "rho": "",
                          "ci_low": b["mean_regret_diff_ci"]["low"], "ci_high": b["mean_regret_diff_ci"]["high"],
                          "n_effective": b["n_effective"]})
        for r in rhos:
            ci = b["wc_recall_diff_ci"].get(str(r), {})
            boot_rows.append({"policy": name, "metric": "wc_recall_diff", "rho": r,
                              "ci_low": ci.get("low"), "ci_high": ci.get("high"),
                              "n_effective": b["n_effective"]})
    pd.DataFrame(boot_rows).to_csv(tables / "followup_bootstrap_diff.csv", index=False)

    roc_df, pr_df = PR.curve_data(scores_by_policy, y)
    roc_df.to_csv(tables / "followup_roc_curves.csv", index=False)
    pr_df.to_csv(tables / "followup_pr_curves.csv", index=False)

    pd.DataFrame({"sample_id": pair["sample_id"].to_numpy(),
                  "wc": y, "cw": pair["cw"].to_numpy(),
                  "mlp_oof": oof_mlp, "logistic_logit_only_oof": oof_a,
                  "logistic_embedding_logit_oof": oof_b}).to_csv(
        tables / "followup_oof_predictions.csv", index=False)

    C.update_metrics(cfg, "mlp_probe", {
        "preproc_hash": phash, "main_transition": [low, high], "device": device,
        "cv": q3res["cv"], "training_log_summary": {
            "n_models": len(mlp_records),
            "median_epochs": int(np.median([r["epochs_run"] for r in mlp_records])),
            "mean_best_inner_val_loss": q3res["cv"]["mean_inner_val_loss"]},
        "policies": metrics, "best_scalar": {"name": best_scalar, **metrics[best_scalar]},
        "bootstrap": boot,
    })
    C.update_metrics(cfg, "q3_followup", q3res)
    C.write_run_config(cfg, script="12_followup_nonlinear_probe.py",
                       extra={"preproc_hash": phash, "q3_followup_verdict": verdict})
    logger.info("nonlinear follow-up probe complete")


if __name__ == "__main__":
    main()
