"""High-confidence recoverable errors (plan §10 / M4).

The strongest evidence for the research question is a sample the model gets WRONG at low
resolution yet is very CONFIDENT about, and which the higher resolution then corrects
(WC with high low-res max-probability). Pure uncertainty routing would never escalate
these, so their existence is exactly what a learned marginal-value model could exploit.

We report WC samples whose low-resolution top-1 confidence exceeds each threshold
(default 0.8 and 0.9), as an absolute count, as a fraction of all recoverable errors, and
as a fraction of all samples.
"""
from __future__ import annotations

from mvwi.audit.transitions import _conf_map, _correctness_map


def high_conf_recoverable(df, low, high, thresholds):
    lo_correct = _correctness_map(df, low)
    hi_correct = _correctness_map(df, high)
    lo_conf = _conf_map(df, low)
    common = lo_correct.index.intersection(hi_correct.index).intersection(lo_conf.index)

    lo_c = lo_correct.loc[common].to_numpy()
    hi_c = hi_correct.loc[common].to_numpy()
    conf = lo_conf.loc[common].to_numpy()

    wc_mask = (~lo_c) & hi_c
    N = int(len(common))
    n_wc = int(wc_mask.sum())

    per_thr = {}
    for thr in thresholds:
        sel = wc_mask & (conf >= thr)
        count = int(sel.sum())
        per_thr[str(thr)] = {
            "threshold": float(thr),
            "count": count,
            "fraction_of_recoverable": float(count / n_wc) if n_wc else 0.0,
            "fraction_of_all": float(count / N) if N else 0.0,
        }
    return {
        "transition": f"{low}->{high}",
        "N": N,
        "n_recoverable_WC": n_wc,
        "by_threshold": per_thr,
    }


def strong_signal(hc, hcfg):
    """Pre-registered engineering criterion: count >= count_strong AND
    fraction_of_recoverable >= frac_strong, at any configured threshold."""
    checks = []
    for thr, d in hc["by_threshold"].items():
        meets = (d["count"] >= hcfg["count_strong"]
                 and d["fraction_of_recoverable"] >= hcfg["frac_strong"])
        checks.append({"threshold": float(thr), "meets_strong": bool(meets), **d})
    any_strong = any(c["meets_strong"] for c in checks)
    return {"any_strong_signal": any_strong, "per_threshold": checks}
