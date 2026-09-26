"""Q1 pre-registered GO / NO-GO decision (plan §9 / M4).

Answers: does the recoverable-error phenomenon (WC under the MAIN transition) exist at a
scale that justifies a learned value router? Thresholds come from configs/analysis.yaml and
are applied mechanically -- never tuned after seeing the numbers.

Decision rule (all four GO criteria must hold for PASS; hard floors trigger FAIL;
anything in between is BORDERLINE):
  GO  : WC >= wc_go_min  AND  WC/N >= wc_frac_go  AND  WC/N(low wrong) >= rec_err_go
        AND (acc_gain_pp >= acc_gain_pp  OR  net_correction > 0)
  NO-GO: WC < wc_border_min  OR  WC/N < wc_frac_nogo
  otherwise BORDERLINE.
"""
from __future__ import annotations


def evaluate_q1(s, q1cfg):
    WC = s["counts"]["WC"]
    CW = s["counts"]["CW"]
    N = s["N"]
    frac_all = s["r_rec"]
    frac_err = s["r_rec_error"]
    acc_gain = s["acc_gain_pp"]

    crit = {
        "wc_count_ge_min": WC >= q1cfg["wc_go_min"],
        "wc_frac_all_ge_min": frac_all >= q1cfg["wc_frac_go"],
        "wc_frac_err_ge_min": frac_err >= q1cfg["rec_err_go"],
        "improvement_meaningful": (acc_gain >= q1cfg["acc_gain_pp"]) or (s["net_correction"] > 0),
    }
    hard_nogo = (WC < q1cfg["wc_border_min"]) or (frac_all < q1cfg["wc_frac_nogo"])
    all_go = all(crit.values())

    if hard_nogo:
        verdict = "FAIL"          # NO-GO
    elif all_go:
        verdict = "PASS"          # GO
    else:
        verdict = "BORDERLINE"

    return {
        "main_transition": f"{s['low']}->{s['high']}",
        "WC": int(WC), "CW": int(CW), "N": int(N),
        "r_rec": round(float(frac_all), 5),
        "r_rec_error": None if frac_err != frac_err else round(float(frac_err), 5),  # NaN guard
        "acc_low": round(float(s["acc_low"]), 5),
        "acc_high": round(float(s["acc_high"]), 5),
        "acc_gain_pp": round(float(acc_gain), 4),
        "net_correction": int(s["net_correction"]),
        "criteria": crit,
        "thresholds": dict(q1cfg),
        "verdict": verdict,
    }
