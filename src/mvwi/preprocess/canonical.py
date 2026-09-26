"""Canonical preprocessing (plan §3).

Single deterministic pipeline guaranteeing that 112/224/448 observe the SAME field of
view: original -> remove bottom banner -> pad-to-square (keep full image + aspect) ->
ONE canonical image -> independently resize THAT canonical to each target -> normalize.

Forbidden here by design: per-resolution ``Resize -> CenterCrop`` (would drift FOV),
and any use of ground-truth bounding boxes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_RESAMPLE = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "lanczos": Image.Resampling.LANCZOS,
}


def remove_banner(img: Image.Image, banner_px: int) -> Image.Image:
    if banner_px and banner_px > 0 and img.height > banner_px:
        return img.crop((0, 0, img.width, img.height - banner_px))
    return img


def make_canonical(img: Image.Image, p: dict) -> Image.Image:
    """De-banner then center-pad to a square that preserves the full image."""
    img = remove_banner(img, p.get("banner_remove_px", 0))
    w, h = img.width, img.height
    side = max(w, h)
    fill = int(p.get("pad_fill", 0))
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    # center placement (deterministic)
    off_x = (side - w) // 2
    off_y = (side - h) // 2
    canvas.paste(img, (off_x, off_y))
    return canvas


def resize_to_target(canonical: Image.Image, target: int, p: dict) -> Image.Image:
    interp = _RESAMPLE.get(str(p.get("interpolation", "bicubic")).lower(), Image.Resampling.BICUBIC)
    # PIL's BICUBIC downsampling is antialiased; explicit resize keeps FOV identical
    return canonical.resize((target, target), resample=interp)


def normalize_tensor(rgb_img: Image.Image, p: dict):
    import torch

    arr = np.asarray(rgb_img, dtype=np.float32) / 255.0  # HWC in [0,1]
    mean = np.array(p["normalize"]["mean"], dtype=np.float32)
    std = np.array(p["normalize"]["std"], dtype=np.float32)
    arr = (arr - mean) / std
    chw = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(np.ascontiguousarray(chw, dtype=np.float32))


def preprocess_to_tensor(img_rgb: Image.Image, p: dict, target: int):
    """End-to-end for a single image at a single resolution (used by the Dataset)."""
    canonical = make_canonical(img_rgb, p)
    resized = resize_to_target(canonical, target, p)
    return normalize_tensor(resized, p)


def canonical_and_targets(img_rgb: Image.Image, p: dict):
    """Return (canonical_pil, {target: pil}) for verification / figure checks."""
    canonical = make_canonical(img_rgb, p)
    outs = {t: resize_to_target(canonical, t, p) for t in p.get("target_sizes", [])}
    return canonical, outs
