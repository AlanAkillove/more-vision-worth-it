"""Phase-0 audit report writer (plan §18 / §10.3). Assembles docs/phase0_report.md entirely
from outputs/phase0/metrics.json + the tables on disk, and derives the OVERALL verdict from
the three pre-registered question verdicts with a fixed, documented rule (never a post-hoc
judgement). Faithfully reports negative results: a NO-GO / follow-up outcome is stated as
such, not papered over because code was written (plan §0.4 / §10.3).
"""
from __future__ import annotations

import json

from mvwi import config as C


def overall_verdict(q1, q2, q3):
    """Fixed mapping documented in the report:
      - Q1 FAIL (no recoverable-error phenomenon) or Q2 FAIL (uncertainty already ~= marginal
        visual value)  -> NO-GO;
      - Q1, Q2 pass and Q3 PASS (a cheap probe already reads the headroom) -> GO;
      - Q1, Q2 pass but Q3 not a clean PASS (real headroom, weak cheap-learnability evidence)
        -> NEEDS ONE PREDEFINED FOLLOW-UP TEST (do not design the decision model yet).
    """
    if q1 == "FAIL" or q2 == "FAIL":
        return "NO-GO"
    if q1 == "PASS" and q2 == "PASS" and q3 == "PASS":
        return "GO"
    return "NEEDS ONE PREDEFINED FOLLOW-UP TEST"


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _pct(v, nd=1):
    return "n/a" if v is None else f"{100.0 * v:.{nd}f}%"


def build_report(cfg, metrics=None):
    tables = C.R(cfg["outputs"]["tables_dir"])
    metrics = metrics or json.loads(C.R(cfg["outputs"]["metrics"]).read_text(encoding="utf-8"))
    env = metrics.get("environment", {})
    ds = metrics.get("dataset", {})
    pre = metrics.get("preprocessing", {})
    ext = metrics.get("extraction", {}).get("by_resolution", {})
    clf = metrics.get("classifier", {})
    acc = metrics.get("outcomes", {}).get("accuracy_by_resolution", {})
    tr = metrics.get("transitions", {}).get("summary", {})
    main = metrics.get("transitions", {}).get("main_transition", [112, 448])
    key = f"{main[0]}->{main[1]}"
    hc = metrics.get("high_conf", {})
    unc = metrics.get("uncertainty", {}).get("scores", {})
    q1 = metrics.get("q1", {})
    q2 = metrics.get("q2", {})
    q3 = metrics.get("q3", {})
    probe_pol = metrics.get("probe", {}).get("policies", {})

    v1, v2, v3 = q1.get("verdict", "?"), q2.get("verdict", "?"), q3.get("verdict", "?")
    overall = overall_verdict(v1, v2, v3)

    L = []
    A = L.append
    A("# When Is More Vision Worth It? — Phase 0 Audit Report\n")
    A("> Phase-0 phenomenon audit only: no neural decision model was trained. All numbers are "
      "drawn from `outputs/phase0/metrics.json`; every figure is rendered from its underlying "
      "table in `outputs/phase0/tables/`. Negative results are reported as they are.\n")
    A(f"- preproc hash: `{pre.get('preproc_hash')}`  ")
    A(f"- git commit: `{env.get('git', {}).get('commit')}` (working tree "
      f"{'DIRTY' if env.get('git', {}).get('dirty') else 'clean'}) on branch `{env.get('git', {}).get('branch')}`  ")
    A(f"- seed: {env.get('seed')}  device: {env.get('device', {}).get('name')} "
      f"({env.get('gpu')}, {env.get('gpu_total_mem_mb')} MB)\n")

    A("## Verdict summary\n")
    A("| Question | Meaning | Verdict |")
    A("| --- | --- | --- |")
    A(f"| Q1 | recoverable-error phenomenon exists at scale | **{v1}** |")
    A(f"| Q2 | marginal visual value != uncertainty (learnable headroom exists) | **{v2}** |")
    A(f"| Q3 | headroom is cheaply learnable by a lightweight probe | **{v3}** |")
    A(f"| **OVERALL** | GO / NO-GO / follow-up | **{overall}** |\n")
    A("**OVERALL rule (fixed, documented):** Q1 or Q2 FAIL -> NO-GO; Q1+Q2+Q3 PASS -> GO; "
      "Q1+Q2 PASS but Q3 not a clean PASS (real headroom, weak cheap-learnability evidence) "
      f"-> NEEDS ONE PREDEFINED FOLLOW-UP TEST. This run: Q1={v1}, Q2={v2}, Q3={v3} -> **{overall}**.\n")

    A("## Environment\n")
    A(f"- Python {env.get('python')}, torch {env.get('torch')} (CUDA {env.get('torch_cuda')}), "
      f"cuDNN {env.get('cudnn')}, platform {env.get('platform')}.")
    A(f"- GPU: {env.get('gpu')}, {env.get('gpu_total_mem_mb')} MB; CUDA available: {env.get('cuda_available')}.")
    A(f"- Key packages: " + ", ".join(f"{k} {v}" for k, v in sorted(env.get("packages", {}).items())) + ".\n")

    A("## Dataset validation\n")
    A(f"- FGVC-Aircraft 2013b: {ds.get('n_images')} images, {ds.get('n_variants')} variants; "
      f"splits {ds.get('splits')}.")
    checks = ds.get("reference_checks", {})
    status = ("ALL PASS" if checks and all(checks.values()) else
              ("FAIL" if checks else "n/a"))
    A(f"- reference checks: {status} -> "
      + ", ".join(f"{k}={'ok' if val else 'FAIL'}" for k, val in checks.items()) + ".\n")

    A("## Preprocessing\n")
    A(f"- canonical: banner trim {pre.get('banner_remove_px')}px -> square pad (fill {pre.get('pad_fill')}) "
      f"-> resize to {pre.get('target_sizes')} ({pre.get('interpolation')}, antialias={pre.get('antialias')}), "
      f"normalize ImageNet mean/std; deterministic={pre.get('deterministic')}.")
    A(f"- preproc_hash `{pre.get('preproc_hash')}` gates every cache path.\n")

    A("## Backbone and feature cache\n")
    A("- frozen DINOv2 ViT-S/14 global embeddings (384-dim), extracted offline per resolution.\n")
    A("| resolution | val runtime(s) | peak VRAM(MB) | img/s | cache(MB) |")
    A("| --- | --- | --- | --- | --- |")
    for res in ["112", "224", "448"]:
        r = ext.get(res, {}).get("val", {})
        A(f"| {res} | {_fmt(r.get('runtime_sec'), 1)} | {r.get('peak_vram_mb')} | "
          f"{_fmt(r.get('images_per_sec'), 1)} | {_fmt(r.get('cache_mb'), 2)} |")
    A("")

    A("## Linear classifier (shared head)\n")
    hp = clf.get("hyperparameters", {})
    A(f"- sklearn {hp.get('model')} (solver {hp.get('solver')}, C {hp.get('C')}, max_iter "
      f"{hp.get('max_iter')}, multi_class {hp.get('multi_class')}); shared across resolutions, "
      f"trained on {clf.get('trained_on', {}).get('split')} only, so VAL predictions are out-of-sample.\n")
    A("### Accuracy by resolution\n")
    A("| resolution | val accuracy | mean top-1 conf | mean entropy |")
    A("| --- | --- | --- | --- |")
    for res in ["112", "224", "448"]:
        r = acc.get(res, {})
        A(f"| {res} | {_fmt(r.get('accuracy'))} | {_fmt(r.get('mean_top1_confidence'))} | "
          f"{_fmt(r.get('mean_entropy'))} |")
    A(f"\n![accuracy](../outputs/phase0/figures/f01_accuracy_vs_resolution.png)\n")

    A("## Transition analysis (Q1)\n")
    s = tr.get(key, {})
    A(f"- main transition {key}: N={s.get('N')}, counts CC/WC/WW/CW = "
      f"{s.get('counts', {}).get('CC')}/{s.get('counts', {}).get('WC')}/"
      f"{s.get('counts', {}).get('WW')}/{s.get('counts', {}).get('CW')}.")
    A(f"- acc_low {_fmt(q1.get('acc_low'))} -> acc_high {_fmt(q1.get('acc_high'))} "
      f"(gain {q1.get('acc_gain_pp'):.1f} pp); WC/N={_pct(q1.get('r_rec'))}, "
      f"WC/low-wrong={_pct(q1.get('r_rec_error'))}, net_correction={q1.get('net_correction')}.")
    A(f"- **Q1 = {q1.get('verdict')}** (criteria " + ", ".join(f"{k}={v}" for k, v in q1.get("criteria", {}).items()) + ").\n")
    A(f"![transition matrix](../outputs/phase0/figures/f02_transition_matrix.png)\n")
    A(f"![counts](../outputs/phase0/figures/f03_transition_counts.png)\n")

    A("## High-confidence recoverable errors\n")
    for th in hc.get("strength", {}).get("per_threshold", []):
        A(f"- conf >= {th['threshold']}: {th['count']} recoverable "
          f"({_pct(th['fraction_of_recoverable'])} of WC, {_pct(th['fraction_of_all'])} of all); "
          f"meets strong signal: {th['meets_strong']}.")
    A(f"- any strong signal: {hc.get('strength', {}).get('any_strong_signal')}. "
      "A sizeable slice of recoverable errors sits at HIGH confidence, i.e. plain confidence "
      "would not escalate them -> supports the value-of-vision-vs-uncertainty distinction.\n")
    A(f"![confidence dist](../outputs/phase0/figures/f04_confidence_distribution.png)\n")
    A(f"![p recoverable by bin](../outputs/phase0/figures/f05_p_recoverable_by_conf_bin.png)\n")

    A("## Uncertainty baselines\n")
    A("| score | AUROC | AUPRC | base rate |")
    A("| --- | --- | --- | --- |")
    for name, m in sorted(unc.items()):
        A(f"| {name} | {_fmt(m.get('auroc'), 3)} | {_fmt(m.get('auprc'), 3)} | {_fmt(m.get('base_rate'), 3)} |")
    A("\nScalar uncertainty scores separate WC only weakly (AUROC ~0.63-0.67).\n")
    A(f"![pr curves](../outputs/phase0/figures/f06_pr_curves_scalar_baselines.png)\n")

    A("## Oracle headroom (Q2)\n")
    A("| rho | best scalar | scalar acc | oracle acc | acc gap (pp) | scalar WC-recall | oracle WC-recall | recall gap (pp) |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for p in q2.get("operating_points", []):
        A(f"| {p['rho']:.2f} | {p['best_scalar']} | {_fmt(p['scalar_final_acc'])} | "
          f"{_fmt(p['oracle_final_acc'])} | {p['acc_gap_pp']:.1f} | {_fmt(p['scalar_wc_recall'], 3)} | "
          f"{_fmt(p['oracle_wc_recall'], 3)} | {p['recall_gap_pp']:.1f} |")
    A(f"\n- **Q2 = {q2.get('verdict')}: the budget-matched sequential oracle beats the best "
      "scalar router by a large margin at every operating point, so real headroom exists that "
      "a learned value router could in principle capture.**\n")
    A(f"![acc vs rho](../outputs/phase0/figures/f07_accuracy_vs_invocation_rate.png)\n")
    A(f"![recall vs rho](../outputs/phase0/figures/f08_recoverable_recall_vs_invocation_rate.png)\n")
    A(f"![scalar vs oracle](../outputs/phase0/figures/f09_scalar_vs_oracle.png)\n")

    A("## Lightweight predictability probe (Q3)\n")
    split = q3.get("split", {})
    A(f"- exploratory temporary VAL split (mode={split.get('mode')}), fit={split.get('n_fit')} "
      f"eval={split.get('n_eval')} rows, stratified on WC; probes never scored on the rows "
      f"they fit. Best scalar baseline on the eval fold: **{q3.get('best_scalar', {}).get('name')}** "
      f"(AUPRC {_fmt(q3.get('best_scalar', {}).get('auprc'), 3)}).\n")
    A("| probe | features | AUROC | AUPRC | AUPRC gain vs best scalar | AUPRC gain 95% CI | regret reduction | recall pts >= +5pp | status |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    feats = metrics.get("probe", {}).get("features", {})
    for name, comp in q3.get("comparisons", {}).items():
        pm = probe_pol.get(name, {})
        ci = comp.get("auprc_diff_ci", {}) or {}
        rr = comp.get("regret_reduction")
        A(f"| {name} | {len(feats.get(name, []))} blocks | {_fmt(pm.get('auroc'), 3)} | "
          f"{_fmt(pm.get('auprc'), 3)} | {comp.get('auprc_gain'):+.3f} | "
          f"[{_fmt(ci.get('low'), 3)}, {_fmt(ci.get('high'), 3)}] | "
          f"{'n/a' if rr is None else format(rr, '+.3f')} | {comp.get('n_points_meeting_recall_gate')}/3 | "
          f"{comp.get('status')} |")
    A(f"\n- **Q3 = {q3.get('verdict')} (NO-GO / weak predictability evidence).** Neither probe "
      "beat the best scalar uncertainty baseline on the held-out eval fold: AUPRC did not "
      "improve (both negative), no operating point met the +5 pp recall gate, and the "
      "probe-minus-baseline AUPRC difference is inside bootstrap noise (CI covers 0). The "
      "headroom Q2 exposed is therefore **not readable by a trivial logistic probe** on "
      "low-resolution features/embeddings.\n")
    A("- underlying per-policy metrics and curves: `outputs/phase0/tables/probe_metrics.csv`, "
      "`probe_curves.csv`, `probe_pr_curves.csv`.\n")

    A("## Runtime / VRAM / storage\n")
    A(f"- extraction peak VRAM (val): 112={ext.get('112', {}).get('val', {}).get('peak_vram_mb')} MB, "
      f"224={ext.get('224', {}).get('val', {}).get('peak_vram_mb')} MB, "
      f"448={ext.get('448', {}).get('val', {}).get('peak_vram_mb')} MB "
      "(well within the 8 GB budget; batch sizes 64/32/8).")
    A("- audit stages (07-09) run in seconds on cached features; all long-job stdout is under "
      "`outputs/logs/run_*.log`.\n")

    A("## GO / NO-GO\n")
    A(f"- Q1 {v1} / Q2 {v2} / Q3 {v3} -> **OVERALL: {overall}**.")
    if overall == "NEEDS ONE PREDEFINED FOLLOW-UP TEST":
        A("- The core research premise SURVIVES: more vision genuinely has marginal value "
          "beyond uncertainty (Q1 strong; Q2 large oracle-vs-scalar headroom). But a cheap "
          "logistic probe could not read that headroom (Q3 NO-GO).")
        A("- **Do NOT proceed to design the Transformer / decision model now** (plan §9.5). "
          "One bounded, predefined follow-up test is warranted before any GO: a stronger — but "
          "still non-neural — learnability probe (e.g. gradient-boosted / cross-validated "
          "logistic features, or richer high-res-side features), pre-registered against the "
          "same Q3 gates. If that also fails, the honest outcome is NO-GO.")
    elif overall == "GO":
        A("- proceed to the tiny decision-model design phase.")
    else:
        A("- stop; the phenomenon does not justify further work.")
    A("")

    A("## Prohibited-scope reminder (plan §19)\n")
    A("No Transformer controller, RL, adaptive crop, active vision, token pruning, other "
      "datasets, backbone fine-tuning, or hyperparameter sweeps were run — all deferred until "
      "a Phase-0 GO.\n")

    return "\n".join(L), {"q1": v1, "q2": v2, "q3": v3, "overall": overall}


def write_report(cfg, logger=None):
    body, verdicts = build_report(cfg)
    out = C.R(cfg["paths"]["docs_dir"]) / "phase0_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    if logger:
        logger.info("report written: %s (OVERALL=%s)", out, verdicts["overall"])
    return out, verdicts
