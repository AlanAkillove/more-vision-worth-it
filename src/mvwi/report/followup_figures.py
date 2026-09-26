"""Figures for the Phase-0 follow-up report (brief F8). Rendered from the tables written by
scripts 11 / 12 so every picture is backed by its underlying data (plan §10.2). English
labels only (plan §10.7).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

from mvwi import config as C                # noqa: E402

POLICY_COLOR = {"mlp": "C3", "logistic_logit_only": "C4", "logistic_embedding_logit": "C5",
                "max_prob": "C0", "entropy": "C1", "margin": "C2", "energy": "C7"}
POLICY_LABEL = {"mlp": "MLP (nonlinear, OOF)", "logistic_logit_only": "logistic A (logits)",
                "logistic_embedding_logit": "logistic B (+z_112)",
                "max_prob": "max_prob", "entropy": "entropy", "margin": "margin",
                "energy": "energy"}


def _paths(cfg):
    tables = C.R(cfg["outputs"]["tables_dir"])
    figs = C.R(cfg["outputs"]["figures_dir"]); figs.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(C.R(cfg["outputs"]["metrics"]).read_text(encoding="utf-8"))
    return tables, figs, metrics


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fu01_calibration(cfg):
    tables, figs, _ = _paths(cfg)
    rel = pd.read_csv(tables / "reliability_112.csv")
    cal = pd.read_csv(tables / "calibration_metrics.csv").set_index("metric")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, tag in zip(axes, ["before", "after"]):
        d = rel[rel["tag"] == tag].dropna()
        cx = (d["bin_lo"] + d["bin_hi"]) / 2
        ax.bar(cx, d["n"] / d["n"].sum(), width=0.06, color="lightgray", label="count")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        ax.plot(cx, d["accuracy"], "o-", color="C3", label="accuracy")
        ax.set_xlabel("predicted confidence"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{'raw (T=1)' if tag=='before' else 'calibrated (OOF temperature)'}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy / share")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        f"FU01 calibration before/after  "
        f"(NLL {cal.loc['nll','before']:.3f}->{cal.loc['nll','after']:.3f}, "
        f"ECE {cal.loc['ece','before']:.3f}->{cal.loc['ece','after']:.3f}, "
        f"Brier {cal.loc['brier','before']:.3f}->{cal.loc['brier','after']:.3f})")
    _save(fig, figs / "fu01_calibration_before_after.png")


def fu02_high_conf_calibrated(cfg):
    tables, figs, _ = _paths(cfg)
    df = pd.read_csv(tables / "high_conf_recoverable_calibrated.csv")
    x = np.arange(len(df)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, df["raw_count"], w, color="C0", label="raw max-prob")
    ax.bar(x + w / 2, df["cal_count"], w, color="C3", label="calibrated max-prob")
    for i, (r, c) in enumerate(zip(df["raw_count"], df["cal_count"])):
        ax.annotate(str(int(r)), (i - w / 2, r), ha="center", va="bottom", fontsize=9)
        ax.annotate(str(int(c)), (i + w / 2, c), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [f"p>= {t:g}" for t in df["threshold"]])
    ax.set_ylabel("# recoverable (WC)")
    ax.set_title("FU02 high-confidence recoverable errors: raw vs calibrated")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, figs / "fu02_high_conf_recoverable_calibrated.png")


def fu03_pr_curves(cfg):
    tables, figs, metrics = _paths(cfg)
    pr = pd.read_csv(tables / "followup_pr_curves.csv")
    pol = metrics["mlp_probe"]["policies"]
    base = metrics["mlp_probe"]["best_scalar"]["name"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ["mlp", "logistic_embedding_logit", "logistic_logit_only"] + list(
            C.load_config("analysis.yaml")["uncertainty_scores"]):
        s = pr[pr["policy"] == name].sort_values("recall")
        if s.empty:
            continue
        ax.plot(s["recall"], s["precision"],
                color=POLICY_COLOR.get(name, "C0"),
                lw=2.2 if name == "mlp" else 1.3,
                ls="--" if name in ("max_prob", "entropy", "margin", "energy") else "-",
                label=f"{POLICY_LABEL.get(name, name)} (AUPRC {pol[name]['auprc']:.3f})")
    ax.axhline(pol[base]["auprc"], color="C0", ls=":", lw=1,
               label=f"best scalar ({base}) AUPRC")
    ax.set_xlabel("Recall (fraction of WC escalated)"); ax.set_ylabel("Precision")
    ax.set_title("FU03 precision-recall: MLP vs logistic vs scalar baselines (pooled OOF)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, figs / "fu03_mlp_pr_vs_baselines.png")


def _pivot(tables, metric):
    df = pd.read_csv(tables / "followup_policy_metrics.csv")
    d = df[df["metric"] == metric].copy()
    d["rho"] = d["rho"].astype(float)
    return d


def _operating_scatter(cfg, figs, metric, ylabel, title, fname, extra_policies=("mlp",
                        "logistic_embedding_logit", "logistic_logit_only")):
    tables, _, metrics = _paths(cfg)
    d = _pivot(tables, metric)
    oracle = _pivot(tables, "oracle_wc_recall") if metric == "wc_recall" else None
    fig, ax = plt.subplots(figsize=(7, 4.6))
    # scalar baselines (dashed) + learned policies (solid)
    for name in metrics["mlp_probe"]["policies"]:
        s = d[d["policy"] == name].dropna().sort_values("rho")
        if s.empty:
            continue
        learned = name in extra_policies
        ax.plot(s["rho"], s["value"], marker="o",
                color=POLICY_COLOR.get(name, "C0"),
                lw=2.2 if name == "mlp" else 1.3, ls="-" if learned else "--",
                alpha=0.9 if learned else 0.7, label=POLICY_LABEL.get(name, name))
    ax.set_xlabel("high-resolution invocation rate rho"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)
    leg = ax.legend(fontsize=8)
    ax.add_artist(leg)
    _save(fig, figs / fname)


def fu04_wc_recall(cfg):
    _operating_scatter(cfg, C.R(C.load_config("analysis.yaml")["outputs"]["figures_dir"]),
                       "wc_recall", "WC (recoverable) recall",
                       "FU04 recoverable recall vs invocation rate (pooled OOF)",
                       "fu04_wc_recall_vs_invocation.png")


def fu05_final_accuracy(cfg):
    _operating_scatter(cfg, C.R(C.load_config("analysis.yaml")["outputs"]["figures_dir"]),
                       "final_accuracy", "final accuracy",
                       "FU05 final accuracy vs invocation rate (pooled OOF)",
                       "fu05_final_accuracy_vs_invocation.png")


def fu06_cw_escalation(cfg):
    tables, figs, _ = _paths(cfg)
    d = _pivot(tables, "p_escalate_given_cw")
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name in d["policy"].unique():
        s = d[d["policy"] == name].sort_values("rho")
        ax.plot(s["rho"], s["value"], marker="o", color=POLICY_COLOR.get(name, "C0"),
                lw=2.2 if name == "mlp" else 1.3, ls="-" if name in
                ("mlp", "logistic_embedding_logit", "logistic_logit_only") else "--",
                label=POLICY_LABEL.get(name, name))
    ax.set_xlabel("high-resolution invocation rate rho")
    ax.set_ylabel("P(escalate | CW)  (harmful-escalation exposure)")
    ax.set_title("FU06 harmful-escalation exposure vs invocation rate (lower is safer)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, figs / "fu06_cw_escalation_vs_invocation.png")


def fu07_routing_regret(cfg):
    _operating_scatter(cfg, C.R(C.load_config("analysis.yaml")["outputs"]["figures_dir"]),
                       "routing_regret", "routing regret (utility vs sequential oracle)",
                       "FU07 routing regret vs invocation rate (pooled OOF)",
                       "fu07_routing_regret_vs_invocation.png")


def fu08_bootstrap(cfg):
    tables, figs, _ = _paths(cfg)
    b = pd.read_csv(tables / "followup_bootstrap_diff.csv")
    mlp = b[b["policy"] == "mlp"]
    rows = [mlp[mlp["metric"] == "auprc_diff"].iloc[0]]
    for r in sorted(mlp[mlp["metric"] == "wc_recall_diff"]["rho"].unique()):
        rows.append(mlp[(mlp["metric"] == "wc_recall_diff") & (mlp["rho"] == r.astype(float))].iloc[0])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    labels = ["\u0394AUPRC"] + [f"\u0394WC recall\n@rho={r:g}" for r in sorted(
        mlp[mlp["metric"] == "wc_recall_diff"]["rho"].unique())]
    lows = [x["ci_low"] for x in rows]; highs = [x["ci_high"] for x in rows]
    centers = [(l + h) / 2 for l, h in zip(lows, highs)]
    y = np.arange(len(rows))
    ax.errorbar(centers, y, xerr=[np.array(centers) - np.array(lows),
                                 np.array(highs) - np.array(centers)],
                fmt="o", color="C3", capsize=4)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(y, labels); ax.set_xlabel("MLP - best scalar (95% bootstrap CI)")
    ax.set_title("FU08 bootstrap difference vs best scalar (CI covering 0 = noise)")
    ax.grid(axis="x", alpha=0.3)
    _save(fig, figs / "fu08_bootstrap_comparison.png")


ALL = [fu01_calibration, fu02_high_conf_calibrated, fu03_pr_curves, fu04_wc_recall,
       fu05_final_accuracy, fu06_cw_escalation, fu07_routing_regret, fu08_bootstrap]


def make_all(cfg, logger=None):
    made = []
    for fn in ALL:
        fn(cfg); made.append(fn.__name__)
        if logger:
            logger.info("figure %s done", fn.__name__)
    return made
