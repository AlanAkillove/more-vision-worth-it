"""CLI 02: build canonical preprocessing, verify determinism + shared FOV, emit hash.

Plan §3 / M1. Produces outputs/phase0/preproc_check/ images for human FOV inspection and
records the preprocessing hash into metrics.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image

from mvwi import config as C
from mvwi.data.fgvc_aircraft import load_split, image_path
from mvwi.preprocess import canonical as pc
from mvwi.preprocess.version import compute_preproc_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="preprocessing.yaml")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("02_make_canonical", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = cfg["preprocessing"]
    phash = compute_preproc_hash(preproc)
    logger.info("preprocessing hash = %s", phash)

    images_dir = C.R(cfg["paths"]["images_dir"])
    recs = load_split(C.R(cfg["paths"]["raw_dir"]), "val")
    n = int(cfg.get("check", {}).get("n_samples", 4))
    out_dir = C.R(cfg["outputs"]["preproc_check_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    determinism_ok = True
    fov_reports = []
    for sid, name in recs[:n]:
        with Image.open(image_path(images_dir, sid)) as im:
            im = im.convert("RGB")
            ow, oh = im.width, im.height
            canonical, targets = pc.canonical_and_targets(im, preproc)
            # determinism: same tensor from same image twice
            t1 = pc.preprocess_to_tensor(im, preproc, 112)
            t2 = pc.preprocess_to_tensor(im, preproc, 112)
            determinism_ok &= bool(torch.equal(t1, t2))
            canonical.save(out_dir / f"{sid}_canonical.png")
            for res, img in targets.items():
                img.save(out_dir / f"{sid}_{res}.png")
            # side-by-side montage scaled to common height for eyeballing identical FOV
            th = 224
            tiles = [targets[r].resize((th, th)) for r in preproc["target_sizes"]]
            montage = Image.new("RGB", (th * len(tiles), th))
            for i, t in enumerate(tiles):
                montage.paste(t, (i * th, 0))
            montage.save(out_dir / f"{sid}_montage.png")
            fov_reports.append({
                "sample_id": sid, "orig_wh": [ow, oh],
                "canonical_side": max(ow, oh - preproc["banner_remove_px"]),
                "targets": list(preproc["target_sizes"]),
            })

    C.update_metrics(cfg, "preprocessing", {
        "preproc_hash": phash,
        "deterministic": determinism_ok,
        "banner_remove_px": preproc["banner_remove_px"],
        "pad_mode": preproc["pad_mode"],
        "pad_fill": preproc["pad_fill"],
        "interpolation": preproc["interpolation"],
        "antialias": preproc["antialias"],
        "target_sizes": preproc["target_sizes"],
        "normalize": preproc["normalize"],
        "fov_sample_check": fov_reports,
    })
    C.write_run_config(cfg, script="02_make_canonical.py", extra={"preproc_hash": phash})
    logger.info("determinism_ok=%s ; wrote %d samples to %s", determinism_ok, n, out_dir)
    if not determinism_ok:
        raise SystemExit("preprocessing is NOT deterministic")


if __name__ == "__main__":
    main()
