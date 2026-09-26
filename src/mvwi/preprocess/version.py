"""Stable preprocessing hash from the pixel-affecting config fields (plan §3.7 / B.7)."""
from __future__ import annotations

import hashlib
import json

_HASH_KEYS = [
    "banner_remove_px",
    "pad_mode",
    "pad_fill",
    "pad_anchor",
    "canonical",
    "target_sizes",
    "interpolation",
    "antialias",
    "normalize",
]


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_preproc_hash(preproc_cfg: dict) -> str:
    subset = {k: preproc_cfg[k] for k in _HASH_KEYS if k in preproc_cfg}
    return hashlib.sha256(_canonical_json(subset).encode("utf-8")).hexdigest()[:12]
