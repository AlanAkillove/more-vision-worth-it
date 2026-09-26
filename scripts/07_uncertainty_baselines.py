"""CLI 07: uncertainty baselines for predicting WC (plan §11 / M5).

At the LOW resolution of the main transition, evaluate max-prob / entropy / margin /
energy as escalation predictors of WC (wrong-at-low, correct-at-high): AUROC + AUPRC,
full ROC / precision-recall curve points, and P(WC | uncertainty bin).

Run it yourself (fast; per-score progress bar):
  conda activate deepminer
  python scripts/07_uncertainty_baselines.py --config configs/analysis.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from mvwi import config as C
from mvwi.audit import outcomes as OC
from mvwi.audit import uncertainty as UN
from mvwi.preprocess.version import compute_preproc_hash

PER_SAMPLE_COLS = ["sample_id", "top1_confidence", "entropy", "top1_top2_margin",
                   "energy", "correct_low", "correct_high", "wc", "cw"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="analysis.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("07_uncertainty_baselines", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    tables = C.R(cfg["outputs"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    df = OC.read_outcomes(cfg, phash, "val")
    low, high = cfg["main_transition"]
    pair = UN.build_pair_frame(df, low, high)
    scores = cfg["uncertainty_scores"]
    logger.info("pair frame: %d samples, transition %s->%s, WC=%d (%.2f%%)",
                len(pair), low, high, int(pair["wc"].sum()), 100 * pair["wc"].mean())

    metrics = {s: UN.evaluate_scores(pair, [s])[s] for s in tqdm(scores, desc="AUROC/AUPRC", unit="score")}
    for s in scores:
        logger.info("score %-8s AUROC=%.4f AUPRC=%.4f (base_rate=%.4f)",
                    s, metrics[s]["auroc"], metrics[s]["auprc"], metrics[s]["base_rate"])

    # per-sample score table (underlying data, never PNG-only)
    pair[PER_SAMPLE_COLS].to_csv(tables / "uncertainty_scores_val.csv", index=False)

    roc_df, pr_df = UN.curve_data(pair, scores)
    roc_df.to_csv(tables / "roc_curves_scalar.csv", index=False)
    pr_df.to_csv(tables / "pr_curves_scalar.csv", index=False)

    bins_frames = []
    for sig in tqdm(["top1_confidence", "entropy"], desc="binning", unit="signal"):
        bins_frames.append(UN.binned_p_wc(pair, sig, n_bins=10))
    pd.concat(bins_frames).to_csv(tables / "uncertainty_bins.csv", index=False)

    C.update_metrics(cfg, "uncertainty", {
        "preproc_hash": phash, "main_transition": [low, high],
        "n_wc": int(pair["wc"].sum()), "n": int(len(pair)),
        "scores": metrics,
    })
    C.write_run_config(cfg, script="07_uncertainty_baselines.py", extra={"preproc_hash": phash})
    logger.info("uncertainty baselines complete")


if __name__ == "__main__":
    main()
