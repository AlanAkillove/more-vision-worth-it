"""Budget-matched routing (plan §12 / M5).

Beyond AUROC we ask the operating-point question directly: if a router escalates exactly
the top-rho fraction of samples (those it deems most worth continuing) to 448, what final
accuracy and recoverable (WC) recall does it achieve, and at what cost? Every escalation
policy -- a scalar uncertainty score or the oracle -- is evaluated through the SAME metric
function so their comparison is apples-to-apples under a matched high-resolution budget.
"""
from __future__ import annotations

import numpy as np


def top_rho_mask(score, rho):
    """Boolean mask selecting the top `rho` fraction by `score` (higher = escalate).

    Deterministic tie-breaking via a stable sort on descending score.
    """
    score = np.asarray(score, dtype=np.float64)
    n = len(score)
    k = int(round(rho * n))
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    if k >= n:
        mask[:] = True
        return mask
    order = np.argsort(-score, kind="stable")[:k]
    mask[order] = True
    return mask


def _r(v, nd=5):
    """Round a float for metrics.json and map NaN/inf to None (strict, valid JSON)."""
    v = float(v)
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return round(v, nd)


def routing_metrics(correct_low, correct_high, escalate, cost):
    """Shared evaluation of ANY escalation policy at whatever budget it actually used."""
    cl = np.asarray(correct_low, dtype=bool)
    ch = np.asarray(correct_high, dtype=bool)
    esc = np.asarray(escalate, dtype=bool)
    wc = (~cl) & ch          # recoverable by escalation
    cw = cl & (~ch)          # harmful escalation
    N = len(cl)
    n_esc = int(esc.sum())
    rho = n_esc / N if N else float("nan")

    final_correct = np.where(esc, ch, cl)
    final_acc = float(final_correct.mean()) if N else float("nan")

    wc_recovered = int((esc & wc).sum())
    wc_recall = wc_recovered / int(wc.sum()) if wc.sum() else float("nan")
    unnecessary = int((esc & (~wc)).sum())          # escalated without recovering
    unnecessary_rate = unnecessary / n_esc if n_esc else float("nan")
    cw_exposure = int((esc & cw).sum())             # escalated samples that got worse

    base_acc = float(cl.mean()) if N else float("nan")
    high_acc = float(ch.mean()) if N else float("nan")
    avg_cost = float(cost["C_112"] + rho * cost["C_448"] + cost.get("C_router", 0.0))

    return {
        "N": int(N), "n_escalated": n_esc, "invocation_rate": _r(rho),
        "n_wc_total": int(wc.sum()), "n_cw_total": int(cw.sum()),
        "base_acc_low": _r(base_acc), "high_res_acc": _r(high_acc),
        "final_accuracy": _r(final_acc),
        "wc_recovered": wc_recovered, "wc_recall": _r(wc_recall),
        "unnecessary_escalations": unnecessary,
        "unnecessary_escalation_rate": _r(unnecessary_rate),
        "cw_exposure": cw_exposure,
        "avg_visual_cost": _r(avg_cost),
    }
