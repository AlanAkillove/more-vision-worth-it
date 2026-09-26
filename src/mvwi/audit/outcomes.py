"""Validation outcome generation (plan §7 / M3).

Applies the single shared linear head to every validation sample at each resolution and
records one row per (sample_id, resolution): ground truth, prediction, correctness and the
four uncertainty scores (max-prob confidence, entropy, top1-top2 margin, energy). Full
logit and probability vectors are kept alongside so downstream routing / probe analyses can
recompute any score without touching the backbone again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mvwi.config import R
from mvwi.eval import metrics as M
from mvwi.features import extract as EX
from mvwi.models import linear_probe as LP


def outcomes_path(cfg, preproc_hash, split="val"):
    return R(cfg["paths"]["outcomes_template"].format(preproc_hash=preproc_hash, split=split))


def build_outcomes(cfg, preproc_hash, bundle, resolutions, split="val", logger=None):
    rows = []
    for res in resolutions:
        f = EX.load_features(cfg, preproc_hash, res, split)
        sid = np.asarray(f["sample_id"])
        y = np.asarray(f["ground_truth_label"], dtype=np.int64)
        X = np.asarray(f["embedding"], dtype=np.float32)
        logits = LP.decision_logits(bundle, X)
        probs = M.softmax(logits)
        pred = logits.argmax(axis=1)
        conf = M.top1_confidence(probs)
        ent = M.entropy(probs)
        marg = M.top1_top2_margin(probs)
        ener = M.energy_score(logits)
        correct = (pred == y)
        for i in range(len(sid)):
            rows.append({
                "sample_id": str(sid[i]),
                "split": split,
                "resolution": int(res),
                "y_true": int(y[i]),
                "y_pred": int(pred[i]),
                "correct": bool(correct[i]),
                "top1_confidence": float(conf[i]),
                "entropy": float(ent[i]),
                "top1_top2_margin": float(marg[i]),
                "energy": float(ener[i]),
                "logits": logits[i].tolist(),
                "probs": probs[i].tolist(),
            })
        if logger:
            acc = M.accuracy(y, pred)
            logger.info("outcomes %s res=%s n=%d acc=%.4f", split, res, len(sid), acc)

    df = pd.DataFrame(rows)
    # enforce one row per (sample_id, resolution)
    dup = df.duplicated(subset=["sample_id", "resolution"]).sum()
    if dup:
        raise ValueError(f"outcome table has {dup} duplicated (sample_id, resolution) rows")
    return df


def write_outcomes(cfg, preproc_hash, df, split="val"):
    op = outcomes_path(cfg, preproc_hash, split)
    op.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(op, index=False)
    return op


def read_outcomes(cfg, preproc_hash, split="val"):
    return pd.read_parquet(outcomes_path(cfg, preproc_hash, split))


def accuracy_by_resolution(df):
    out = {}
    for res, g in df.groupby("resolution"):
        out[str(int(res))] = {
            "n": int(len(g)),
            "accuracy": float(M.accuracy(g["y_true"].values, g["y_pred"].values)),
            "mean_top1_confidence": float(g["top1_confidence"].mean()),
            "mean_entropy": float(g["entropy"].mean()),
        }
    return out
