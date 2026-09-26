"""CLI 11: follow-up calibration audit (brief F2).

Cross-fitted temperature scaling on the 112-resolution shared-head outputs, then the
before/after calibration report and the recalibrated high-confidence recoverable-error
counts and scalar baselines. Validation-only; no test access; nothing is fit on a sample
and evaluated on the same sample (each fold's temperature comes from the OTHER folds).

Run it yourself (fast; CPU 1-D temperature fits):
  conda activate deepminer
  python scripts/11_followup_calibration.py --config configs/followup.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

from mvwi import config as C
from mvwi.audit import calibration as CAL
from mvwi.audit import outcomes as OC
from mvwi.audit import uncertainty as UN
from mvwi.eval import metrics as M
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
    logger = C.setup_logger("11_followup_calibration", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    tables = C.R(cfg["outputs"]["tables_dir"]); tables.mkdir(parents=True, exist_ok=True)

    low, high = cfg["main_transition"]
    cal_cfg = cfg["calibration"]
    bins = int(cal_cfg["ece_bins"])

    logits, y_true, sid = CAL.load_val_state(cfg, phash, low)
    raw_probs = M.softmax(logits)
    logger.info("low-res val state: logits=%s y=%s (Acc=%.4f, mean maxprob=%.4f)",
                logits.shape, y_true.shape, M.accuracy(M.predicted_class(logits), y_true),
                float(M.top1_confidence(raw_probs).mean()))

    # WC / CW aligned to the logits rows (from the same outcome table)
    df = OC.read_outcomes(cfg, phash, "val")
    pair = UN.build_pair_frame(df, low, high)
    pmap = {str(s): (int(w), int(c), int(cl)) for s, w, c, cl in
            zip(pair["sample_id"], pair["wc"], pair["cw"], pair["correct_low"].astype(int))}
    wc = np.array([pmap[s][0] for s in sid], dtype=bool)
    cw = np.array([pmap[s][1] for s in sid], dtype=bool)
    correct_low = np.array([bool(pmap[s][2]) for s in sid])

    cal_probs, temps = CAL.oof_calibrated_probs(logits, y_true, cal_cfg)
    summary = CAL.calibration_summary(raw_probs, cal_probs, y_true, bins=bins)
    logger.info("temperatures per fold: %s", [round(t, 3) for t in temps])
    logger.info("calibration NLL %.4f->%.4f  ECE %.4f->%.4f  Brier %.4f->%.4f  conf %.3f->%.3f",
                summary["before"]["nll"], summary["after"]["nll"],
                summary["before"]["ece"], summary["after"]["ece"],
                summary["before"]["brier"], summary["after"]["brier"],
                summary["before"]["mean_confidence"], summary["after"]["mean_confidence"])

    # reliability curves (before vs after) as an underlying table
    rel = pd.concat([CAL.reliability_table(raw_probs, y_true, bins, "before"),
                     CAL.reliability_table(cal_probs, y_true, bins, "after")], ignore_index=True)
    rel["gap"] = (rel["accuracy"] - rel["mean_confidence"]).abs()
    rel.to_csv(tables / "reliability_112.csv", index=False)

    pd.DataFrame([
        {"metric": k, "before": summary["before"][k], "after": summary["after"][k]}
        for k in ("nll", "brier", "ece", "mean_confidence", "accuracy")
    ]).assign(resolution=low, n=len(y_true),
              mean_temperature=round(float(np.mean(temps)), 4)).to_csv(
        tables / "calibration_metrics.csv", index=False)

    # recalibrated high-confidence recoverable counts (raw vs calibrated max-prob)
    raw_conf = M.top1_confidence(raw_probs)
    cal_conf = M.top1_confidence(cal_probs)
    hc = []
    for thr in cal_cfg["high_conf_thresholds"]:
        r = CAL.high_conf_wc(raw_conf, wc, [thr]).iloc[0].to_dict()
        c = CAL.high_conf_wc(cal_conf, wc, [thr]).iloc[0].to_dict()
        hc.append({"threshold": float(thr), "raw_count": r["count"],
                   "raw_fraction_of_recoverable": r["fraction_of_recoverable"],
                   "cal_count": c["count"], "cal_fraction_of_recoverable": c["fraction_of_recoverable"],
                   "n_wc": int(wc.sum())})
    hc_df = pd.DataFrame(hc)
    hc_df.to_csv(tables / "high_conf_recoverable_calibrated.csv", index=False)

    # scalar baselines recomputed on calibrated probabilities (AUROC/AUPRC vs WC)
    from sklearn.metrics import average_precision_score, roc_auc_score
    cal_logits = logits / float(np.mean(temps))
    raw_scores = {"max_prob": 1.0 - raw_conf, "entropy": M.entropy(raw_probs),
                  "margin": -M.top1_top2_margin(raw_probs), "energy": M.energy_score(logits)}
    cal_scores = CAL.calibrated_scalar_escalation(cal_probs, cal_logits)
    yw = wc.astype(int)
    rows, sbcal = [], {}
    for name in tqdm(cfg["uncertainty_scores"], desc="scalar baselines", unit="score"):
        entry = {}
        for variant, sc in (("raw", raw_scores), ("calibrated", cal_scores)):
            rec = {"score": name, "variant": variant,
                   "auroc": round(float(roc_auc_score(yw, sc[name])), 5),
                   "auprc": round(float(average_precision_score(yw, sc[name])), 5)}
            rows.append(rec)
            entry[variant] = {"auroc": rec["auroc"], "auprc": rec["auprc"]}
        sbcal[name] = entry
    pd.DataFrame(rows).to_csv(tables / "uncertainty_scores_calibrated_val.csv", index=False)

    # per-sample calibrated uncertainty table (underlying data)
    pd.DataFrame({
        "sample_id": sid, "correct_low": correct_low,
        "wc": wc.astype(int), "cw": cw.astype(int),
        "raw_top1_confidence": raw_conf, "cal_top1_confidence": cal_conf,
        "raw_entropy": M.entropy(raw_probs), "cal_entropy": M.entropy(cal_probs),
    }).to_csv(tables / "calibration_scores_val.csv", index=False)

    C.update_metrics(cfg, "calibration", {
        "preproc_hash": phash, "resolution": low, "method": cal_cfg["method"],
        "n_folds": int(cal_cfg["n_folds"]), "ece_bins": bins,
        "temperatures": [round(float(t), 4) for t in temps],
        "mean_temperature": round(float(np.mean(temps)), 4),
        "summary": summary, "high_conf_calibrated": hc,
        "scalar_baselines_calibrated": sbcal,
    })
    C.write_run_config(cfg, script="11_followup_calibration.py", extra={"preproc_hash": phash})
    logger.info("calibration audit complete")


if __name__ == "__main__":
    main()
