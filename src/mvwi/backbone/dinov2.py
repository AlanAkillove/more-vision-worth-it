"""Frozen DINOv2 ViT-S/14 backbone (plan §4). Loaded from the vendored local source so
torch.hub works without GitHub API access. Always eval + frozen + inference_mode.
"""
from __future__ import annotations

import torch

from mvwi.config import R


def load_model(cfg, device, logger=None):
    b = cfg["backbone"]
    local_dir = str(R(b["local_dir"]))
    model = torch.hub.load(local_dir, b["name"], source="local", pretrained=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device != "cpu":
        model = model.to(device)
    if logger:
        logger.info("loaded backbone %s (embed_dim=%s, patch=%s) from %s",
                    b["name"], getattr(model, "embed_dim", "?"), b.get("patch_size"), local_dir)
    return model


def global_embedding(model, img_batch, device, amp_dtype="fp16"):
    """Return float32 global CLS embeddings [B, D] under inference_mode + autocast."""
    with torch.inference_mode():
        if device == "cuda" and amp_dtype:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model.forward_features(img_batch)["x_norm_clstoken"]
        else:
            out = model.forward_features(img_batch)["x_norm_clstoken"]
    return out.float()


def identifier(cfg):
    return cfg.get("identifier", cfg["backbone"]["name"])
