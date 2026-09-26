"""Figure generation for M7 (plan §16 / §10.1). Nine required PNGs, each rendered from the
underlying tables in outputs/phase0/tables/ so the picture can never drift from the data
(plan §10.7: "图统一从 tables 读取"). Every figure keeps its source table on disk too
(plan §10.2), re-deriving only the small helper tables figures need but M2-M6 did not write.

Labels are English (plan §10.7: matplotlib 中文字体缺失风险 -> 英文标签).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")                      # headless, font-independent
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                          # noqa: E402
import pandas as pd                         # noqa: E402

from mvwi import config as C                # noqa: E402

CATEGORY_ORDER = ["CC", "WC", "WW", "CW"]
RESOLUTIONS = [112, 224, 448]
SCORE_STYLE = {"max_prob": "C0", "entropy": "C1", "margin": "C2", "energy": "C3",
               "probe_a": "C4", "probe_b": "C5", "sequential_oracle": "C7"}


def _paths(cfg):
    tables = C.R(cfg["outputs"]["tables_dir"])
    figs = C.R(cfg["outputs"]["figures_dir"])
    figs.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(C.R(cfg["outputs"]["metrics"]).read_text(encoding="utf-8"))
    return tables, figs, metrics


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- helper tables
def ensure_accuracy_by_resolution(cfg):
    tables, _, metrics = _paths(cfg)
    acc = metrics["outcomes"]["accuracy_by_resolution"]
    rows = [{"resolution": int(res), "n": acc[str(res)]["n"],
             "accuracy": acc[str(res)]["accuracy"],
             "mean_top1_confidence": acc[str(res)]["mean_top1_confidence"],
             "mean_entropy": acc[str(res)]["mean_entropy"]} for res in RESOLUTIONS]
    df = pd.DataFrame(rows)
    df.to_csv(tables / "accuracy_by_resolution.csv", index=False)
    return df


def ensure_transition_matrix(cfg):
    tables, _, metrics = _paths(cfg)
    low, high = cfg["main_transition"]
    counts = metrics["transitions"]["summary"][f"{low}->{high}"]["counts"]
    mat = np.array([[counts["CC"], counts["CW"]],
                    [counts["WC"], counts["WW"]]], dtype=float)
    df = pd.DataFrame(mat,
                      index=["low_correct", "low_wrong"],
                      columns=["high_correct", "high_wrong"])
    df.to_csv(tables / f"transition_matrix_{low}-{high}.csv")
    return df


def ensure_scalar_vs_oracle(cfg):
    """Tidy best-scalar vs oracle per rho (both accuracy and WC-recall), for figure f09."""
    tables, _, metrics = _paths(cfg)
    rc = pd.read_csv(tables / "routing_curves.csv")
    oc = pd.read_csv(tables / "oracle_curves.csv")
    rows = []
    for rho, g in rc.groupby("target_rho"):
        best = g.loc[g["final_accuracy"].idxmax()]
        o = oc[oc["target_rho"] == rho].iloc[0]
        for metric in ("final_accuracy", "wc_recall"):
            rows.append({"rho": float(rho), "metric": metric, "best_scalar": best["policy"],
                         "scalar_value": float(best[metric]), "oracle_value": float(o[metric]),
                         "gap": float(o[metric]) - float(best[metric])})
    df = pd.DataFrame(rows)
    df.to_csv(tables / "scalar_vs_oracle.csv", index=False)
    return df


# --------------------------------------------------------------------------------- figures
def f01_accuracy_vs_resolution(cfg):
    tables, figs, _ = _paths(cfg)
    df = ensure_accuracy_by_resolution(cfg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["resolution"], df["accuracy"], "o-", color="C0", label="accuracy")
    ax.plot(df["resolution"], df["mean_top1_confidence"], "s--", color="C3",
            label="mean top-1 confidence")
    for _, r in df.iterrows():
        ax.annotate(f"{r['accuracy']:.3f}", (r["resolution"], r["accuracy"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xticks(RESOLUTIONS)
    ax.set_xlabel("Input resolution (px)")
    ax.set_ylabel("Value")
    ax.set_title("F01 Accuracy vs resolution (val, shared linear head)")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    _save(fig, figs / "f01_accuracy_vs_resolution.png")


def f02_transition_matrix(cfg):
    tables, figs, _ = _paths(cfg)
    low, high = cfg["main_transition"]
    df = ensure_transition_matrix(cfg)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(df.values, cmap="viridis")
    total = df.values.sum()
    for i in range(2):
        for j in range(2):
            v = df.values[i, j]
            ax.text(j, i, f"{int(v)}\n{v / total:.1%}", ha="center", va="center",
                    color="white" if v > total / 3 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["correct @high", "wrong @high"])
    ax.set_yticks([0, 1], ["correct @low", "wrong @low"])
    ax.set_title(f"F02 {low}->{high} transition matrix (count, share of N={int(total)})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    _save(fig, figs / "f02_transition_matrix.png")


def f03_transition_counts(cfg):
    tables, figs, metrics = _paths(cfg)
    low, high = cfg["main_transition"]
    counts = metrics["transitions"]["summary"][f"{low}->{high}"]["counts"]
    labels = {"CC": "CC correct/correct", "WC": "WC recoverable (target)",
              "WW": "WW wrong/wrong", "CW": "CW harmful-escalation"}
    vals = [counts[c] for c in CATEGORY_ORDER]
    colors = ["C2", "C1", "C4", "C3"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([labels[c] for c in CATEGORY_ORDER], vals, color=colors)
    for b, c in zip(bars, CATEGORY_ORDER):
        ax.annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Sample count")
    ax.set_title(f"F03 transition counts {low}->{high} (WC is the marginal-value target)")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, figs / "f03_transition_counts.png")


def f04_confidence_distribution(cfg):
    tables, figs, _ = _paths(cfg)
    df = pd.read_csv(tables / "uncertainty_scores_val.csv")
    rec = df[df["wc"] == 1]["top1_confidence"]
    non = df[df["wc"] == 0]["top1_confidence"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 25)
    ax.hist(non, bins=bins, alpha=0.6, label=f"not recoverable (n={len(non)})", color="C0")
    ax.hist(rec, bins=bins, alpha=0.6, label=f"recoverable WC (n={len(rec)})", color="C1")
    ax.set_xlabel("top-1 confidence @ low resolution")
    ax.set_ylabel("count")
    ax.set_title("F04 confidence distribution: recoverable vs not")
    ax.legend()
    _save(fig, figs / "f04_confidence_distribution.png")


def f05_p_recoverable_by_conf_bin(cfg):
    tables, figs, _ = _paths(cfg)
    df = pd.read_csv(tables / "uncertainty_bins.csv")
    df = df[df["signal"] == "top1_confidence"].copy()
    df["center"] = (df["bin_lo"] + df["bin_hi"]) / 2
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["center"], df["p_wc"], "o-", color="C1")
    ax.axhline(df["n_wc"].sum() / df["n"].sum(), ls=":", color="gray",
               label="base rate")
    ax.set_xlabel("top-1 confidence bin center @ low resolution")
    ax.set_ylabel("P(WC | bin)")
    ax.set_title("F05 P(recoverable | confidence bin)")
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, figs / "f05_p_recoverable_by_conf_bin.png")


def f06_pr_curves_scalar(cfg):
    tables, figs, metrics = _paths(cfg)
    df = pd.read_csv(tables / "pr_curves_scalar.csv")
    auroc = {k: v["auroc"] for k, v in metrics["uncertainty"]["scores"].items()}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    base = metrics["uncertainty"]["scores"]["max_prob"]["base_rate"]
    for name in metrics.get("uncertainty", {}).get("scores", {}):
        s = df[df["score"] == name].sort_values("recall")
        ax.plot(s["recall"], s["precision"], color=SCORE_STYLE.get(name, "C0"),
                label=f"{name} (AUROC {auroc[name]:.3f})")
    ax.axhline(base, ls=":", color="gray", label=f"base rate {base:.3f}")
    ax.set_xlabel("Recall (fraction of WC escalated)")
    ax.set_ylabel("Precision")
    ax.set_title("F06 precision-recall: scalar uncertainty baselines")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, figs / "f06_pr_curves_scalar_baselines.png")


def _operating_curves(cfg):
    tables, _, _ = _paths(cfg)
    rc = pd.read_csv(tables / "routing_curves.csv")
    oc = pd.read_csv(tables / "oracle_curves.csv")
    oc = oc.assign(policy="sequential_oracle")
    return pd.concat([rc, oc], ignore_index=True)


def f07_accuracy_vs_invocation(cfg):
    tables, figs, _ = _paths(cfg)
    allp = _operating_curves(cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol, g in allp.groupby("policy"):
        g = g.sort_values("invocation_rate")
        ax.plot(g["invocation_rate"], g["final_accuracy"], "o-",
                color=SCORE_STYLE.get(pol, None), label=pol, alpha=0.85)
    ax.set_xlabel("High-resolution invocation rate rho")
    ax.set_ylabel("Final accuracy")
    ax.set_title("F07 accuracy vs invocation rate (budget-matched routers + oracle)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, figs / "f07_accuracy_vs_invocation_rate.png")


def f08_recall_vs_invocation(cfg):
    tables, figs, _ = _paths(cfg)
    allp = _operating_curves(cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for pol, g in allp.groupby("policy"):
        g = g.sort_values("invocation_rate")
        ax.plot(g["invocation_rate"], g["wc_recall"], "o-",
                color=SCORE_STYLE.get(pol, None), label=pol, alpha=0.85)
    ax.set_xlabel("High-resolution invocation rate rho")
    ax.set_ylabel("Recoverable (WC) recall")
    ax.set_title("F08 recoverable recall vs invocation rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, figs / "f08_recoverable_recall_vs_invocation_rate.png")


def f09_scalar_vs_oracle(cfg):
    tables, figs, _ = _paths(cfg)
    df = ensure_scalar_vs_oracle(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, metric in zip(axes, ["final_accuracy", "wc_recall"]):
        d = df[df["metric"] == metric]
        rhos = sorted(d["rho"].unique())
        x = np.arange(len(rhos))
        sc = [float(d[(d["rho"] == r)]["scalar_value"].iloc[0]) for r in rhos]
        orc = [float(d[(d["rho"] == r)]["oracle_value"].iloc[0]) for r in rhos]
        ax.bar(x - 0.18, sc, width=0.36, color="C0", label="best scalar")
        ax.bar(x + 0.18, orc, width=0.36, color="C7", label="sequential oracle")
        for i, (s, o) in enumerate(zip(sc, orc)):
            ax.annotate(f"+{(o - s) * 100:.1f}pp", (i, max(s, o)), ha="center",
                        va="bottom", fontsize=8, color="C7")
        ax.set_xticks(x, [f"{r:.0%}" for r in rhos])
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Value")
    fig.suptitle("F09 best scalar router vs budget-matched oracle (Q2 headroom)")
    axes[-1].legend(loc="lower right")
    _save(fig, figs / "f09_scalar_vs_oracle.png")


ALL_FIGURES = [f01_accuracy_vs_resolution, f02_transition_matrix, f03_transition_counts,
               f04_confidence_distribution, f05_p_recoverable_by_conf_bin,
               f06_pr_curves_scalar, f07_accuracy_vs_invocation,
               f08_recall_vs_invocation, f09_scalar_vs_oracle]


def make_all(cfg, logger=None):
    made = []
    for fn in ALL_FIGURES:
        fn(cfg)
        made.append(fn.__name__)
        if logger:
            logger.info("figure %s done", fn.__name__)
    return made
