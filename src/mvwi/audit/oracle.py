"""Sequential oracle + clairvoyant bound (plan §13 / M5).

Two DIFFERENT reference policies, deliberately kept apart:

* sequential oracle -- the honest headroom ceiling UNDER A BUDGET. Every sample first
  pays the 112 cost. Knowing each sample's true (low, high) outcome, the oracle spends a
  fixed escalation budget rho*N on the samples with the largest TRUE utility improvement
  of escalating (utility = correct_high - correct_low, i.e. +1 for WC, 0 for CC/WW,
  -1 for CW). This is the best any budget-respecting learned value router could do; the
  gap to the best scalar uncertainty router measures learnable headroom.

* clairvoyant bound -- an UPPER accuracy bound that ignores the budget: escalate EXACTLY
  the truly-recoverable samples (all WC, no CW) and nothing else. It uses however much
  cost that requires, so it is NOT budget-matched and must never be compared to routers
  at the same operating point. Reported only to bound total recoverable headroom.

The router's own cost (C_router) is ~0 in Phase 0 but the interface keeps it explicit;
the real sequential cost is C_policy = C_112 + I(continue)*C_448 + C_router.
"""
from __future__ import annotations

import numpy as np

from mvwi.audit.routing import routing_metrics, top_rho_mask


def utility_vector(correct_low, correct_high, ucfg):
    """True escalation utility per sample, driven by configs/analysis.yaml `utility`.

    type='error_to_correct' => +1 (WC, wrong->right), -1 (CW, right->wrong), 0 otherwise;
    `weight` scales the recovery value (kept explicit so sensitivity analysis is real, not
    a silent no-op). Unknown types fail fast rather than silently assume error_to_correct.
    """
    utype = str(ucfg.get("type", "error_to_correct"))
    if utype != "error_to_correct":
        raise ValueError(f"unsupported utility.type: {utype!r} (only 'error_to_correct' is defined)")
    w = float(ucfg.get("weight", 1.0))
    cl = np.asarray(correct_low, dtype=bool)
    ch = np.asarray(correct_high, dtype=bool)
    return w * (ch.astype(np.int64) - cl.astype(np.int64))   # +1 WC, 0 CC/WW, -1 CW


def sequential_oracle_mask(correct_low, correct_high, rho, ucfg=None):
    """Top-rho fraction of samples by TRUE escalation utility (deterministic ties)."""
    u = utility_vector(correct_low, correct_high, ucfg or {"type": "error_to_correct", "weight": 1.0})
    return top_rho_mask(u, rho)


def sequential_oracle(correct_low, correct_high, rho, cost, ucfg=None):
    cl = np.asarray(correct_low, dtype=bool)
    ch = np.asarray(correct_high, dtype=bool)
    m = sequential_oracle_mask(cl, ch, rho, ucfg)
    res = routing_metrics(cl, ch, m, cost)
    res["policy"] = f"sequential_oracle@rho={rho}"
    return res


def clairvoyant_bound(correct_low, correct_high, cost):
    """Unbudgeted reference: escalate exactly the recoverable samples, no others.

    This is an accuracy CEILING and a cost FLOOR for reaching that ceiling; it is NOT
    budget-matched and must never be compared to routers at a fixed operating point.
    """
    cl = np.asarray(correct_low, dtype=bool)
    ch = np.asarray(correct_high, dtype=bool)
    m = (~cl) & ch                     # escalate only true WC
    res = routing_metrics(cl, ch, m, cost)
    res["policy"] = "clairvoyant_bound_unbudgeted"
    return res
