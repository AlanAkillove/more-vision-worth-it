"""Transition analysis (plan §8 / M4).

For a low->high resolution pair, every validation sample falls into one of four cells:

  CC = low correct, high correct
  WC = low wrong, high correct   (recoverable error  -- the phenomenon we care about)
  WW = low wrong, high wrong
  CW = low correct, high wrong   (harmful escalation)

All statistics are computed on the set of sample_ids present at BOTH resolutions, so the
low/high correctness flags describe the same physical image under two visual budgets.
"""
from __future__ import annotations

import pandas as pd


def _correctness_map(df, res):
    g = df[df["resolution"] == int(res)]
    return g.set_index("sample_id")["correct"].astype(bool)


def _conf_map(df, res):
    g = df[df["resolution"] == int(res)]
    return g.set_index("sample_id")["top1_confidence"].astype(float)


def transition_stats(df, low, high):
    lo = _correctness_map(df, low)
    hi = _correctness_map(df, high)
    common = lo.index.intersection(hi.index)
    lo_v = lo.loc[common].to_numpy()
    hi_v = hi.loc[common].to_numpy()
    N = int(len(common))
    if N == 0:
        raise ValueError(f"no shared samples between res={low} and res={high}")

    low_correct, high_correct = lo_v, hi_v
    CC = int((low_correct & high_correct).sum())
    WC = int((~low_correct & high_correct).sum())
    WW = int((~low_correct & ~high_correct).sum())
    CW = int((low_correct & ~high_correct).sum())
    n_low_wrong = int((~low_correct).sum())
    n_low_correct = int(low_correct.sum())
    acc_low = float(low_correct.mean())
    acc_high = float(high_correct.mean())

    def _safe(num, den):
        return float(num / den) if den else float("nan")

    return {
        "low": int(low), "high": int(high), "N": N,
        "counts": {"CC": CC, "WC": WC, "WW": WW, "CW": CW},
        "fraction_of_all": {"CC": _safe(CC, N), "WC": _safe(WC, N),
                             "WW": _safe(WW, N), "CW": _safe(CW, N)},
        "acc_low": acc_low, "acc_high": acc_high,
        "acc_gain_pp": (acc_high - acc_low) * 100.0,
        "n_low_correct": n_low_correct, "n_low_wrong": n_low_wrong,
        "conditional": {
            "WC_of_low_wrong": _safe(WC, n_low_wrong),
            "WW_of_low_wrong": _safe(WW, n_low_wrong),
            "CC_of_low_correct": _safe(CC, n_low_correct),
            "CW_of_low_correct": _safe(CW, n_low_correct),
        },
        "r_rec": _safe(WC, N),                 # recoverable fraction of all samples
        "r_rec_error": _safe(WC, n_low_wrong),  # recoverable fraction of low-res errors
        "net_correction": WC - CW,
    }


def stats_to_dataframe(low, high, s):
    """Tidy long table: one row per reported metric for this transition."""
    rows = []
    tname = f"{low}->{high}"
    for cat in ["CC", "WC", "WW", "CW"]:
        rows.append({"transition": tname, "metric": f"count_{cat}", "value": s["counts"][cat]})
        rows.append({"transition": tname, "metric": f"frac_all_{cat}",
                     "value": s["fraction_of_all"][cat]})
    for name, val in s["conditional"].items():
        rows.append({"transition": tname, "metric": f"cond_{name}", "value": val})
    for k in ["N", "acc_low", "acc_high", "acc_gain_pp", "n_low_wrong",
              "r_rec", "r_rec_error", "net_correction"]:
        rows.append({"transition": tname, "metric": k, "value": s[k]})
    return pd.DataFrame(rows)
