"""Q2 pre-registered decision: is uncertainty already ~equal to marginal visual value? (plan §14 / M5)

At each main operating point (rho = 20/40/60%) we compare the BEST scalar uncertainty
router (highest final accuracy among the four signals) against the budget-matched
sequential oracle. A large, consistent oracle-over-scalar gap means the true marginal
value of extra vision is NOT captured by uncertainty, leaving headroom a learned value
router could exploit (GO for M6). If the oracle barely beats uncertainty at every point,
a learned router is unlikely to help (NO-GO).

Gates (from configs/analysis.yaml q2 block), expressed in percentage points:
  GO      : exists a point with acc_gap >= go_acc_gap*100  OR  recall_gap >= go_recall_gap*100
  NO-GO   : at EVERY point acc_gap < nogo_acc_gap*100  AND  recall_gap < go_recall_gap*100
  otherwise BORDERLINE.
"""
from __future__ import annotations


def pick_best_scalar(score_metrics_at_rho):
    """Best scalar router at one rho: max final_accuracy, tie-break on wc_recall then name."""
    best, best_key = None, None
    for name, m in sorted(score_metrics_at_rho.items()):
        acc = m["final_accuracy"] if m["final_accuracy"] is not None else -1.0
        rec = m["wc_recall"] if m["wc_recall"] is not None else -1.0
        key = (acc, rec)
        if best_key is None or key > best_key:
            best_key, best = key, (name, m)
    return best


def _pp(a, b):
    """percentage-point difference a-b, tolerant of None."""
    if a is None or b is None:
        return None
    return round((a - b) * 100.0, 4)


def evaluate_q2(scalar_by_rho, oracle_by_rho, q2cfg):
    per_point = []
    for rho in sorted(scalar_by_rho.keys()):
        best_name, best_m = pick_best_scalar(scalar_by_rho[rho])
        orc = oracle_by_rho[rho]
        acc_gap = _pp(orc["final_accuracy"], best_m["final_accuracy"])
        rec_gap = _pp(orc["wc_recall"], best_m["wc_recall"])
        per_point.append({
            "rho": rho,
            "best_scalar": best_name,
            "scalar_final_acc": best_m["final_accuracy"],
            "oracle_final_acc": orc["final_accuracy"],
            "acc_gap_pp": acc_gap,
            "scalar_wc_recall": best_m["wc_recall"],
            "oracle_wc_recall": orc["wc_recall"],
            "recall_gap_pp": rec_gap,
            "scalar_avg_cost": best_m["avg_visual_cost"],
            "oracle_avg_cost": orc["avg_visual_cost"],
        })

    go_acc = q2cfg["go_acc_gap"] * 100.0
    go_rec = q2cfg["go_recall_gap"] * 100.0
    nogo_acc = q2cfg["nogo_acc_gap"] * 100.0

    def g(v, d=0.0):
        return d if v is None else v

    is_go = any(g(p["acc_gap_pp"]) >= go_acc or g(p["recall_gap_pp"]) >= go_rec for p in per_point)
    is_nogo = all(g(p["acc_gap_pp"]) < nogo_acc and g(p["recall_gap_pp"]) < go_rec for p in per_point)

    if is_go:
        verdict = "PASS"          # GO for a learned value router
    elif is_nogo:
        verdict = "FAIL"          # NO-GO: uncertainty already ~= marginal visual value
    else:
        verdict = "BORDERLINE"

    return {"operating_points": per_point, "thresholds": dict(q2cfg), "verdict": verdict}
