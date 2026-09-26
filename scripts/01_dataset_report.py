"""CLI 01: dataset validation report (plan §2 / M0)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvwi import config as C
from mvwi.data import dataset_report as dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("01_dataset_report", cfg)
    C.set_seed(cfg["project"]["seed"])

    banner = C.load_config("preprocessing.yaml")["preprocessing"].get("banner_remove_px", 20)
    report = dr.build_dataset_report(cfg, banner_px=banner)
    jp, md = dr.write_reports(cfg, report)

    # sanity vs known 2013b reference values (recorded, not fatal)
    checks = {
        "images_10000": report["n_images_on_disk"] == 10000,
        "variants_100": report["n_variants"] == 100,
        "train_3334": report["splits"]["train"]["n_samples"] == 3334,
        "val_3333": report["splits"]["val"]["n_samples"] == 3333,
        "test_3333": report["splits"]["test"]["n_samples"] == 3333,
        "no_split_overlap": all(v == 0 for v in report["split_overlap"].values()),
        "no_missing_images": all(d["missing_image_files"] == 0 for d in report["splits"].values()),
    }
    report["reference_checks"] = checks
    jp.write_text(__import__("json").dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("images=%s variants=%s train/val/test=%s/%s/%s overlap=%s",
                report["n_images_on_disk"], report["n_variants"],
                report["splits"]["train"]["n_samples"], report["splits"]["val"]["n_samples"],
                report["splits"]["test"]["n_samples"], report["split_overlap"])
    logger.info("reference_checks: %s", checks)
    logger.info("wrote %s and %s", jp, md)

    C.update_metrics(cfg, "dataset", {
        "n_images": report["n_images_on_disk"],
        "n_variants": report["n_variants"],
        "splits": {k: v["n_samples"] for k, v in report["splits"].items()},
        "reference_checks": checks,
    })
    C.write_run_config(cfg, script="01_dataset_report.py")

    if not all(checks.values()):
        logger.warning("dataset reference checks not all True: %s", checks)


if __name__ == "__main__":
    main()
