"""CLI 10: M7 figures + Phase-0 audit report (plan §16 / §10 / §18).

Renders the nine required figures from the underlying tables written by M2-M6 and then
assembles docs/phase0_report.md from metrics.json, printing the Q1 / Q2 / Q3 and OVERALL
verdicts. Idempotent: re-run any time the tables change.

Run it yourself (fast; reads cached tables only, no GPU):
  conda activate deepminer
  python scripts/10_make_figures.py --config configs/analysis.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvwi import config as C
from mvwi.report import figures as FG
from mvwi.report import report as RP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="analysis.yaml")
    ap.add_argument("--no-report", action="store_true", help="only build figures")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("10_make_figures", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    from mvwi.preprocess.version import compute_preproc_hash
    phash = compute_preproc_hash(preproc)

    figs_dir = C.R(cfg["outputs"]["figures_dir"])
    figs_dir.mkdir(parents=True, exist_ok=True)
    made = FG.make_all(cfg, logger)
    logger.info("figures complete (%d): %s", len(made), ", ".join(made))

    verdicts = None
    if not args.no_report:
        out, verdicts = RP.write_report(cfg, logger)
        logger.info("report: %s", out)

    C.write_run_config(cfg, script="10_make_figures.py",
                       extra={"preproc_hash": phash, "figures": len(made), "verdicts": verdicts})
    if verdicts:
        logger.info("Q1=%s Q2=%s Q3=%s OVERALL=%s",
                    verdicts["q1"], verdicts["q2"], verdicts["q3"], verdicts["overall"])
    logger.info("M7 complete")


if __name__ == "__main__":
    main()
