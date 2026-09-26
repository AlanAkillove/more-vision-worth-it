"""Q3 pre-registered decision: is marginal visual value cheaply LEARNABLE? (plan §15 / §9.5 / M6)

Q2 (PASS) showed a scalar-vs-oracle gap, i.e. theoretical headroom. Q3 asks whether that
headroom is readable by a trivial logistic probe before we spend anything on a neural
decision model. A probe is judged against the BEST scalar uncertainty baseline, on the
SAME held-out eval fold, on four pre-registered GO signals (all must hold):

  1. recoverable (WC) recall gain >= recall_gain_pp at >= min_operating_points budgets;
  2. mean routing regret reduced by >= regret_reduction (relative to the baseline);
  3. AUPRC gain >= auprc_gain;
  4. the AUPRC gain is not clearly inside bootstrap noise (CI lower bound > 0).

PASS if either probe clears all four; BORDERLINE if a probe shows partial signal (>= 2 of
the four, including the AUPRC gate); otherwise FAIL (NO-GO -- weak predictability, do not
escalate to a Transformer). Thresholds come from configs/probe.yaml `q3` and are applied
mechanically.
"""
from __future__ import annotations

import numpy as np


def _mean_regret(policy, rhos):
    vals = [policy["routing"][r]["routing_regret"] for r in rhos]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _pp(a, b):
    if a is None or b is None:
        return None
    return (a - b) * 100.0


def evaluate_probe_vs_base(name, probe, base, boot_entry, rhos, q3cfg):
    per_point = []
    n_recall_ok = 0
    for r in rhos:
        gain_pp = _pp(probe["routing"][r]["wc_recall"], base["routing"][r]["wc_recall"])
        ok = gain_pp is not None and gain_pp >= q3cfg["recall_gain_pp"]
        n_recall_ok += int(ok)
        per_point.append({
            "rho": r,
            "probe_recall": probe["routing"][r]["wc_recall"],
            "base_recall": base["routing"][r]["wc_recall"],
            "recall_gain_pp": None if gain_pp is None else round(gain_pp, 4),
            "recall_gain_ci": boot_entry["wc_recall_diff_ci"].get(str(r)) if boot_entry else None,
            "meets_recall_gate": bool(ok),
        })

    base_regret = _mean_regret(base, rhos)
    probe_regret = _mean_regret(probe, rhos)
    if base_regret is None or base_regret <= 0 or probe_regret is None:
        regret_reduction = None
    else:
        regret_reduction = (base_regret - probe_regret) / base_regret

    auprc_gain = probe["auprc"] - base["auprc"]
    ci_low = boot_entry["auprc_diff_ci"]["low"] if boot_entry else None

    gates = {
        "recall_gate": n_recall_ok >= int(q3cfg["min_operating_points"]),
        "regret_gate": regret_reduction is not None and regret_reduction >= q3cfg["regret_reduction"],
        "auprc_gate": auprc_gain >= q3cfg["auprc_gain"],
        "noise_gate": ci_low is not None and ci_low > 0.0,
    }
    n_gates = sum(bool(v) for v in gates.values())
    status = "GO" if n_gates == 4 else ("PARTIAL" if (n_gates >= 2 and gates["auprc_gate"]) else "NO-GO")

    return {
        "vs_base": "best_scalar",
        "operating_points": per_point,
        "n_points_meeting_recall_gate": n_recall_ok,
        "base_mean_regret": None if base_regret is None else round(base_regret, 6),
        "probe_mean_regret": None if probe_regret is None else round(probe_regret, 6),
        "regret_reduction": None if regret_reduction is None else round(float(regret_reduction), 5),
        "auprc_gain": round(float(auprc_gain), 5),
        "auprc_diff_ci": boot_entry["auprc_diff_ci"] if boot_entry else None,
        "mean_regret_diff_ci": boot_entry["mean_regret_diff_ci"] if boot_entry else None,
        "gates": gates,
        "status": status,
    }


def evaluate_q3(probe_metrics, base_metrics, boot, rhos, q3cfg):
    """probe_metrics: {name: policy_metrics}; base_metrics: best-scalar policy_metrics."""
    comparisons = {}
    for name, probe in probe_metrics.items():
        entry = boot.get(name) if boot else None
        comparisons[name] = evaluate_probe_vs_base(name, probe, base_metrics, entry, rhos, q3cfg)

    any_go = any(c["status"] == "GO" for c in comparisons.values())
    any_partial = any(c["status"] == "PARTIAL" for c in comparisons.values())
    verdict = "PASS" if any_go else ("BORDERLINE" if any_partial else "FAIL")

    return {
        "comparisons": comparisons,
        "thresholds": dict(q3cfg),
        "operating_rhos": list(rhos),
        "verdict": verdict,
    }
