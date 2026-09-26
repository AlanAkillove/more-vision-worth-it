"""CLI 03: offline frozen-DINOv2 feature extraction (plan §5 / M2).

Examples (run these yourself to watch the tqdm progress bars):
  conda activate deepminer
  python scripts/03_extract_features.py --config configs/extract.yaml                 # full train+val, all 3 resolutions
  python scripts/03_extract_features.py --config configs/extract.yaml --res 448 --limit 64   # quick smoke on 448
  python scripts/03_extract_features.py --config configs/extract.yaml --cache-only     # verify caches, no backbone run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mvwi import config as C
from mvwi.backbone import dinov2
from mvwi.features import extract as EX
from mvwi.preprocess.version import compute_preproc_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="extract.yaml")
    ap.add_argument("--res", type=int, default=None, help="single resolution (default: all)")
    ap.add_argument("--split", default=None, help="single split (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="max samples per split (smoke test)")
    ap.add_argument("--cache-only", action="store_true", help="read caches, do not run backbone")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("03_extract_features", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    device = cfg["device"]["name"] if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high") if cfg["device"].get("allow_tf32") else None
    logger.info("preproc_hash=%s device=%s", phash, device)

    resolutions = [args.res] if args.res else cfg["resolutions"]
    splits = [args.split] if args.split else cfg["splits"]

    if args.cache_only:
        read = {}
        for res in resolutions:
            for split in splits:
                f = EX.load_features(cfg, phash, res, split)
                read[f"{res}/{split}"] = {
                    "n": int(f["embedding"].shape[0]),
                    "dim": int(f["embedding"].shape[1]),
                    "first_id": str(f["sample_id"][0]),
                }
        logger.info("cache-only readback OK: %s", read)
        C.update_metrics(cfg, "feature_cache_readback", {"preproc_hash": phash, "read": read})
        return

    model = dinov2.load_model(cfg, device, logger)
    by_res = {}
    for res in resolutions:
        for split in splits:
            by_res.setdefault(str(res), {})[split] = EX.extract_one(
                cfg, preproc, phash, res, split, model, device, logger, limit=args.limit)

    C.update_metrics(cfg, "extraction", {"preproc_hash": phash, "device": device, "by_resolution": by_res})
    C.write_run_config(cfg, script="03_extract_features.py", extra={"preproc_hash": phash, "limit": args.limit})
    logger.info("extraction complete for resolutions=%s splits=%s", resolutions, splits)


if __name__ == "__main__":
    main()
