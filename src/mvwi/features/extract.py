"""Offline frozen-DINOv2 global-embedding extraction with a tqdm progress bar (plan §5 / M2).

Caches per (resolution, split) an .npz carrying every field required by plan §4.4 so the
backbone never needs to run again for downstream analysis.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mvwi.backbone import dinov2
from mvwi.config import R
from mvwi.data.fgvc_aircraft import (AircraftDataset, collate_fn, load_split,
                                      load_variants, name_to_idx)


def cache_path(cfg, preproc_hash, res, split):
    return R(cfg["cache_dir_template"].format(preproc_hash=preproc_hash, res=res, split=split))


def load_features(cfg, preproc_hash, res, split):
    z = np.load(cache_path(cfg, preproc_hash, res, split), allow_pickle=False)
    return {
        "sample_id": z["sample_id"],
        "embedding": z["embedding"],
        "ground_truth_label": z["ground_truth_label"],
        "original_width": z["original_width"],
        "original_height": z["original_height"],
        "class_names": z["class_names"],
    }


def extract_one(cfg, preproc, preproc_hash, res, split, model, device, logger, limit=None):
    data_dir = R(cfg["paths"]["raw_dir"])
    images_dir = R(cfg["paths"]["images_dir"])
    vnames = load_variants(data_dir)
    n2i = name_to_idx(vnames)

    records = load_split(data_dir, split)
    if limit:
        records = records[:limit]

    bs = int(cfg["batch_size"][str(res)])
    nw = int(cfg.get("num_workers", 0))
    ds = AircraftDataset(records, images_dir, n2i, res, preproc)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn,
                    pin_memory=(device == "cuda"), persistent_workers=(nw > 0))

    embs, ids, labels, ws, hs = [], [], [], [], []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for sids, lab, w, h, img in tqdm(dl, total=len(dl), unit="batch", dynamic_ncols=True,
                                     desc=f"extract res={res} {split}"):
        img = img.to(device, non_blocking=True)
        e = dinov2.global_embedding(model, img, device)
        embs.append(e.cpu().numpy())
        ids += list(sids)
        labels.append(lab.numpy())
        ws.append(w.numpy())
        hs.append(h.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    dt = max(time.time() - t0, 1e-6)

    emb = np.concatenate(embs, 0).astype(np.float32)
    labels = np.concatenate(labels)
    ws = np.concatenate(ws)
    hs = np.concatenate(hs)
    peak = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0.0

    cp = cache_path(cfg, preproc_hash, res, split)
    cp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cp,
        sample_id=np.array(ids),
        split=np.array(split),
        ground_truth_label=labels,
        resolution=np.array(int(res)),
        embedding=emb,
        original_width=ws,
        original_height=hs,
        preprocessing_version=np.array(preproc_hash),
        backbone_identifier=np.array(dinov2.identifier(cfg)),
        class_names=np.array(vnames),
    )
    size = cp.stat().st_size
    meta = {
        "n_samples": int(len(ids)),
        "runtime_sec": round(dt, 2),
        "peak_vram_mb": int(round(peak)),
        "images_per_sec": round(len(ids) / dt, 1),
        "cache_bytes": int(size),
        "cache_mb": round(size / 1024 / 1024, 2),
        "batch_size": bs,
        "num_workers": nw,
        "cache_path": str(cp),
    }
    logger.info("done res=%s split=%s -> %s", res, split, meta)
    return meta
