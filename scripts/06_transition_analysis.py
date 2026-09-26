"""CLI 06: transition analysis, Q1 verdict, high-confidence recoverable errors (plan §8-§10 / M4).

Reads the validation outcome table produced by 05 and, for each configured resolution
transition (main analysis 112->448), reports CC/WC/WW/CW counts and fractions, applies the
pre-registered Q1 gate, and quantifies high-confidence recoverable errors.

Run it yourself:
  conda activate deepminer
  python scripts/06_transition_analysis.py --config configs/analysis.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mvwi import config as C
from mvwi.audit import high_conf as HC
from mvwi.audit import outcomes as OC
from mvwi.audit import q1 as Q1
from mvwi.audit import transitions as TR
from mvwi.preprocess.version import compute_preproc_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="analysis.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("06_transition_analysis", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    tables = C.R(cfg["outputs"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    df = OC.read_outcomes(cfg, phash, "val")
    logger.info("loaded outcomes: %d rows, resolutions=%s",
                len(df), sorted(df["resolution"].unique().tolist()))

    lo_main, hi_main = cfg["main_transition"]

    transition_summary = {}
    frames = []
    for low, high in cfg["transitions"]:
        s = TR.transition_stats(df, low, high)
        long_df = TR.stats_to_dataframe(low, high, s)
        pair_df = pd.DataFrame({
            "category": ["CC", "WC", "WW", "CW"],
            "count": [s["counts"][c] for c in ["CC", "WC", "WW", "CW"]],
            "fraction_of_all": [s["fraction_of_all"][c] for c in ["CC", "WC", "WW", "CW"]],
            "conditional": [
                s["conditional"]["CC_of_low_correct"],   # CC given low correct
                s["conditional"]["WC_of_low_wrong"],     # WC given low wrong
                s["conditional"]["WW_of_low_wrong"],     # WW given low wrong
                s["conditional"]["CW_of_low_correct"],   # CW given low correct
            ],
        })
        out_csv = tables / f"transitions_{low}-{high}.csv"
        pair_df.to_csv(out_csv, index=False)
        long_df.to_csv(tables / f"transitions_{low}-{high}_full.csv", index=False)
        frames.append(long_df)
        transition_summary[f"{low}->{high}"] = s
        logger.info("transition %s->%s: CC=%d WC=%d WW=%d CW=%d | r_rec=%.4f r_rec_error=%s acc %.4f->%.4f",
                    low, high, s["counts"]["CC"], s["counts"]["WC"], s["counts"]["WW"],
                    s["counts"]["CW"], s["r_rec"],
                    (None if s["r_rec_error"] != s["r_rec_error"] else round(s["r_rec_error"], 4)),
                    s["acc_low"], s["acc_high"])

    # Q1 on the MAIN transition
    q1res = Q1.evaluate_q1(transition_summary[f"{lo_main}->{hi_main}"], cfg["q1"])
    logger.info("Q1 verdict=%s | WC=%d r_rec=%.4f r_rec_error=%s acc_gain_pp=%.3f net_corr=%d",
                q1res["verdict"], q1res["WC"], q1res["r_rec"], q1res["r_rec_error"],
                q1res["acc_gain_pp"], q1res["net_correction"])

    # High-confidence recoverable errors on the MAIN transition
    hc_cfg = cfg["high_conf"]
    hc = HC.high_conf_recoverable(df, lo_main, hi_main, hc_cfg["thresholds"])
    strong = HC.strong_signal(hc, hc_cfg)
    hc_rows = []
    for thr, d in hc["by_threshold"].items():
        hc_rows.append({"threshold": d["threshold"], "count": d["count"],
                        "fraction_of_recoverable": d["fraction_of_recoverable"],
                        "fraction_of_all": d["fraction_of_all"]})
    pd.DataFrame(hc_rows).to_csv(tables / "high_conf_recoverable.csv", index=False)
    logger.info("high-conf recoverable (main %s->%s): %s | any_strong=%s",
                lo_main, hi_main, hc["by_threshold"], strong["any_strong_signal"])

    C.update_metrics(cfg, "transitions", {
        "preproc_hash": phash,
        "main_transition": [lo_main, hi_main],
        "summary": transition_summary,
    })
    C.update_metrics(cfg, "q1", q1res)
    C.update_metrics(cfg, "high_conf", {"main_transition": [lo_main, hi_main],
                                        "detail": hc, "strength": strong})
    C.write_run_config(cfg, script="06_transition_analysis.py",
                        extra={"preproc_hash": phash, "q1_verdict": q1res["verdict"]})
    logger.info("transition analysis complete; wrote tables to %s", tables)


if __name__ == "__main__":
    main()
