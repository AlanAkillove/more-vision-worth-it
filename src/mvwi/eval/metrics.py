"""Evaluation metric helpers for Phase 0 audits.

Pure functions over logit / probability matrices so both the outcome generator (M3)
and every downstream analysis (M4-M6) share identical definitions of confidence,
entropy, margin and energy.
"""
from __future__ import annotations

import numpy as np


def softmax(logits):
    """Numerically-stable row-wise softmax over a [N, K] logit matrix."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def logsumexp(logits):
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(x - m).sum(axis=1)))


def predicted_class(logits):
    return np.asarray(logits).argmax(axis=1)


def top1_confidence(probs):
    return np.asarray(probs).max(axis=1)


def entropy(probs):
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def top1_top2_margin(probs):
    p = np.asarray(probs, dtype=np.float64)
    if p.shape[1] < 2:
        return p[:, 0].copy()
    part = np.partition(p, kth=-2, axis=1)[:, -2:]
    top = part[:, 1]
    second = part[:, 0]
    return top - second


def energy_score(logits, t=1.0):
    """Free-energy OOD score: -t * logsumexp(logits / t). Higher => less confident."""
    x = np.asarray(logits, dtype=np.float64) / t
    return -t * logsumexp(x)


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def score_row(logits, y_true):
    """Compute the full per-sample uncertainty/correctness record for one (sample,res)."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = softmax(logits)
    pred = predicted_class(logits)
    return {
        "probs": probs,
        "pred": pred,
        "correct": pred == np.asarray(y_true),
        "top1_confidence": top1_confidence(probs),
        "entropy": entropy(probs),
        "top1_top2_margin": top1_top2_margin(probs),
        "energy": energy_score(logits),
    }
