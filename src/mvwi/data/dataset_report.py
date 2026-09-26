"""M0 dataset validation report (plan §2). Verifies the actual on-disk FGVC-Aircraft
2013b version: image count, annotation files, variant count, train/val/test sizes,
image-path rule, and label mapping. Writes outputs/dataset_report.json + .md.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from mvwi.config import R
from mvwi.data import fgvc_aircraft as fga


def _size_stats(sizes):
    w = np.array([s[0] for s in sizes])
    h = np.array([s[1] for s in sizes])
    ar = w / h
    return {
        "count": int(len(sizes)),
        "width": {"min": int(w.min()), "median": int(np.median(w)), "max": int(w.max())},
        "height": {"min": int(h.min()), "median": int(np.median(h)), "max": int(h.max())},
        "aspect_ratio_w_over_h": {"min": round(float(ar.min()), 3),
                                   "median": round(float(np.median(ar)), 3),
                                   "max": round(float(ar.max()), 3)},
    }


def build_dataset_report(cfg, banner_px=20):
    data_dir = R(cfg["paths"]["raw_dir"])
    images_dir = R(cfg["paths"]["images_dir"])

    variant_names = fga.load_variants(data_dir)
    n2i = fga.name_to_idx(variant_names)

    disk_files = set(os_listdir(images_dir))
    disk_ids = {f[:-4] for f in disk_files if f.lower().endswith(".jpg")}

    report = {
        "dataset": "FGVC-Aircraft 2013b",
        "data_dir": str(data_dir),
        "images_dir": str(images_dir),
        "image_path_rule": "images/<sample_id>.jpg",
        "annotation_files_present": {},
        "n_images_on_disk": len(disk_ids),
        "n_variants": len(variant_names),
        "n_families": _line_count(data_dir / "families.txt"),
        "n_manufacturers": _line_count(data_dir / "manufacturers.txt"),
        "splits": {},
    }

    union_ids = set()
    per_split_ids = {}
    for split in fga.SPLITS:
        fpath = data_dir / fga.SPLIT_FILES[split]
        report["annotation_files_present"][fga.SPLIT_FILES[split]] = fpath.exists()
        recs = fga.load_split(data_dir, split)
        ids = [r[0] for r in recs]
        per_split_ids[split] = set(ids)
        union_ids.update(ids)
        labels = [n2i[r[1]] for r in recs]
        cnt = Counter(labels)
        missing = [i for i in ids if f"{i}.jpg" not in disk_files]
        report["splits"][split] = {
            "n_samples": len(recs),
            "n_classes_present": len(cnt),
            "min_per_class": int(min(cnt.values())),
            "max_per_class": int(max(cnt.values())),
            "missing_image_files": len(missing),
            "example_missing": missing[:5],
        }

    # overlap + coverage checks
    tr_va = len(per_split_ids["train"] & per_split_ids["val"])
    tr_te = len(per_split_ids["train"] & per_split_ids["test"])
    va_te = len(per_split_ids["val"] & per_split_ids["test"])
    report["split_overlap"] = {"train_val": tr_va, "train_test": tr_te, "val_test": va_te}
    report["union_of_splits"] = len(union_ids)
    report["disk_files_not_in_any_split"] = len(disk_ids - union_ids)
    report["split_ids_not_on_disk"] = len(union_ids - disk_ids)
    report["label_mapping"] = {"class_index_source": "variants.txt (order)",
                               "first_classes": variant_names[:5],
                               "n_classes": len(variant_names)}

    # image size stats over every val image + a train sample (full pass is header-only/cheap)
    sample_ids = sorted(per_split_ids["val"])
    sizes = []
    try:
        from tqdm import tqdm
        sample_ids = tqdm(sample_ids, desc="scan val image sizes", unit="img", dynamic_ncols=True)
    except Exception:
        pass
    for sid in sample_ids:
        if f"{sid}.jpg" in disk_files:
            with Image.open(fga.image_path(images_dir, sid)) as im:
                sizes.append((im.width, im.height))
    report["image_size_stats_val_full"] = _size_stats(sizes)

    report["banner_note"] = (
        f"FGVC-Aircraft images carry an approx {banner_px}px copyright banner at the bottom; "
        "removal is a preprocessing assumption (configs/preprocessing.yaml: banner_remove_px). "
        "Ground-truth bounding boxes are NOT used in the main preprocessing."
    )
    report["banner_remove_px_assumed"] = banner_px
    return report


def os_listdir(path):
    import os

    return os.listdir(path)


def _line_count(path):
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return sum(1 for ln in f if ln.strip() != "")


def write_reports(cfg, report):
    jp = R(cfg["outputs"]["dataset_report_json"])
    md = R(cfg["outputs"]["dataset_report_md"])
    jp.parent.mkdir(parents=True, exist_ok=True)
    md.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md.write_text(_to_md(report), encoding="utf-8")
    return jp, md


def _to_md(r):
    lines = ["# FGVC-Aircraft 2013b Dataset Report", ""]
    lines.append(f"- data_dir: `{r['data_dir']}`")
    lines.append(f"- images on disk: **{r['n_images_on_disk']}**")
    lines.append(f"- variants (classes): **{r['n_variants']}** | families: {r['n_families']} | manufacturers: {r['n_manufacturers']}")
    lines.append(f"- image path rule: `{r['image_path_rule']}`")
    lines.append("")
    lines.append("## Splits")
    lines.append("| split | n_samples | classes_present | min/max per class | missing files |")
    lines.append("|---|---|---|---|---|")
    for s, d in r["splits"].items():
        lines.append(f"| {s} | {d['n_samples']} | {d['n_classes_present']} | {d['min_per_class']}/{d['max_per_class']} | {d['missing_image_files']} |")
    lines.append("")
    o = r["split_overlap"]
    lines.append(f"- split overlaps: train∩val={o['train_val']}, train∩test={o['train_test']}, val∩test={o['val_test']}")
    lines.append(f"- union of splits = {r['union_of_splits']} | disk-not-in-split = {r['disk_files_not_in_any_split']} | split-not-on-disk = {r['split_ids_not_on_disk']}")
    ss = r["image_size_stats_val_full"]
    lines.append(f"- image size (val, w): {ss['width']} ; (h): {ss['height']} ; aspect: {ss['aspect_ratio_w_over_h']}")
    lines.append(f"- banner: {r['banner_note']}")
    return "\n".join(lines) + "\n"
