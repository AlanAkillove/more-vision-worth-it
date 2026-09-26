"""Orchestration for the Phase-0 single follow-up test (brief F3-F6).

Holds everything that ties the nonlinear probe to the SAME evaluation the Phase-0 gate
used, so the comparison is apples-to-apples on all 3333 validation rows:

* 5-fold out-of-fold (OOF) pooled scores for the MLP, and for the two Phase-0 logistic
  probes re-run under the same protocol, plus the four scalar uncertainty baselines;
* evaluation of every policy as an escalation router (AUROC / AUPRC, and per-rho final
  accuracy, WC recall, unnecessary-escalation rate, CW exposure, routing regret, gap to the
  sequential oracle);
* the CW safety diagnostics the brief insists on (P(escalate|CW) and share of escalations
  that are CW), kept separate from WC recall so harmful escalation is never hidden;
* bootstrap CIs for MLP-minus-best-scalar differences;
* the UNCHANGED Q3 gate, reused from mvwi.audit.q3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mvwi.audit.oracle import sequential_oracle, utility_vector
from mvwi.audit.routing import routing_metrics, top_rho_mask


def safe_auroc(y, s):
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def oof_logistic_probs(X, y, cv_cfg, lcfg):
    """5-fold OOF P(WC) from a standardized logistic regression (re-run of Phase-0 probes)."""
    n = len(y)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_splits=int(cv_cfg["n_folds"]), shuffle=True, random_state=int(1337))
    for tr, te in skf.split(X, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])          # train-fold statistics only
        Xte = sc.transform(X[te])
        clf = LogisticRegression(solver="lbfgs", C=float(lcfg.get("C", 1.0)),
                                 max_iter=int(lcfg.get("max_iter", 2000)))
        clf.fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return oof


def extended_policy_metrics(score, y, cl, ch, cost, ucfg, rhos):
    """Router metrics for one escalation score, with CW exposure and oracle gap per rho."""
    cl_b = np.asarray(cl, dtype=bool)
    ch_b = np.asarray(ch, dtype=bool)
    wc = (~cl_b) & ch_b
    cw = cl_b & (~ch_b)
    n_cw = int(cw.sum())
    out = {"auroc": round(safe_auroc(y, score), 5),
           "auprc": round(float(average_precision_score(y, score)), 5)}
    u = utility_vector(cl_b, ch_b, ucfg)
    routing = {}
    for rho in rhos:
        mask = top_rho_mask(score, rho)
        rm = routing_metrics(cl_b, ch_b, mask, cost)
        orc = sequential_oracle(cl_b, ch_b, rho, cost, ucfg)
        n_esc = rm["n_escalated"]
        cw_esc = int((mask & cw).sum())
        oracle_util = float(u[top_rho_mask(u, rho)].sum())
        policy_util = float(u[mask].sum())
        regret = oracle_util - policy_util
        rel = regret / oracle_util if oracle_util > 0 else None
        routing[rho] = {
            "rho": rho,
            "wc_recall": rm["wc_recall"],
            "final_accuracy": rm["final_accuracy"],
            "unnecessary_escalation_rate": rm["unnecessary_escalation_rate"],
            "escalation_utility": round(policy_util, 6),
            "oracle_utility": round(oracle_util, 6),
            "routing_regret": round(regret, 6),
            "relative_regret": None if rel is None or rel != rel else round(rel, 6),
            "p_escalate_given_cw": round(cw_esc / n_cw, 5) if n_cw else None,
            "cw_share_of_escalated": round(cw_esc / n_esc, 5) if n_esc else None,
            "wc_recall_gap_to_oracle": _diff(orc["wc_recall"], rm["wc_recall"]),
            "acc_gap_to_oracle": _diff(orc["final_accuracy"], rm["final_accuracy"]),
            "oracle_wc_recall": orc["wc_recall"],
            "oracle_final_accuracy": orc["final_accuracy"],
        }
    out["routing"] = routing
    return out


def _diff(a, b):
    if a is None or b is None:
        return None
    return round(a - b, 5)


def pick_best_scalar(metrics, score_names):
    best = max(score_names, key=lambda n: (metrics[n]["auprc"], n))
    return best


def routing_metrics_table(metrics_by_policy, rhos):
    """Tidy long table of per-policy / per-rho routing diagnostics for CSV output."""
    rows = []
    for name, m in metrics_by_policy.items():
        rows.append({"policy": name, "metric": "auroc", "rho": "", "value": m["auroc"]})
        rows.append({"policy": name, "metric": "auprc", "rho": "", "value": m["auprc"]})
        for rho in rhos:
            r = m["routing"][rho]
            for k in ("final_accuracy", "wc_recall", "unnecessary_escalation_rate",
                      "routing_regret", "relative_regret", "p_escalate_given_cw",
                      "cw_share_of_escalated", "wc_recall_gap_to_oracle",
                      "oracle_wc_recall"):
                rows.append({"policy": name, "metric": k, "rho": rho, "value": r[k]})
    return pd.DataFrame(rows)
