"""CLI 05: generate the validation outcome table (plan §7 / M3).

Applies the shared head (from 04) to every VAL sample at each resolution and writes one
row per (sample_id, resolution) to cache/outcomes/{preproc_hash}/outcomes_val.parquet.

Run it yourself:
  conda activate deepminer
  python scripts/05_generate_outcomes.py --config configs/classifier.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvwi import config as C
from mvwi.audit import outcomes as OC
from mvwi.models import linear_probe as LP
from mvwi.preprocess.version import compute_preproc_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="classifier.yaml")
    ap.add_argument("--split", default="val", help="Phase 0 audits only use val")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("05_generate_outcomes", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    resolutions = cfg["resolutions"]

    bundle = LP.load_classifier(cfg, phash)
    logger.info("loaded classifier preproc_hash=%s resolutions=%s",
                str(bundle["preproc_hash"]), bundle["resolutions"].tolist())

    df = OC.build_outcomes(cfg, phash, bundle, resolutions, split=args.split, logger=logger)
    op = OC.write_outcomes(cfg, phash, df, split=args.split)
    accs = OC.accuracy_by_resolution(df)
    logger.info("wrote %d outcome rows -> %s", len(df), op)

    C.update_metrics(cfg, "outcomes", {
        "preproc_hash": phash,
        "split": args.split,
        "n_rows": int(len(df)),
        "unique_sample_resolution": int(df[["sample_id", "resolution"]].drop_duplicates().shape[0]),
        "table": str(op),
        "accuracy_by_resolution": accs,
    })
    C.write_run_config(cfg, script="05_generate_outcomes.py",
                        extra={"preproc_hash": phash, "split": args.split})
    logger.info("outcome generation complete; accuracy_by_resolution=%s", accs)


if __name__ == "__main__":
    main()
