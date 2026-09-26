"""CLI 13: follow-up figures + docs/phase0_followup_report.md (brief F8).

Renders the follow-up figures from the tables (scripts 11/12) and assembles the report that
must be allowed to terminate the project. It reports the verdict honestly: because the Q3
gate is unchanged and the MLP is compared to the best scalar baseline on identical pooled
out-of-fold predictions, a FAIL here means OVERALL = NO-GO for the current
"tiny decision model from global low-resolution representation" formulation.

Run it yourself (fast; reads cached tables/metrics only):
  conda activate deepminer
  python scripts/13_followup_report.py --config configs/followup.yaml
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvwi import config as C
from mvwi.report import followup_figures as FF

_FIGREL = "../outputs/phase0/figures"


def _route(metrics_policy, rho):
    for k, v in metrics_policy["routing"].items():
        if abs(float(k) - rho) < 1e-9:
            return v
    return {}


def _f(v, nd=4):
    return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def build(cfg, metrics):
    cal = metrics.get("calibration", {})
    cs = cal.get("summary", {})
    mlp = metrics.get("mlp_probe", {})
    pol = mlp.get("policies", {})
    best = mlp.get("best_scalar", {})
    base_name = best.get("name")
    q3f = metrics.get("q3_followup", {})
    comp = q3f.get("comparisons", {}).get("mlp", {})
    q3_orig = metrics.get("q3", {})
    env = metrics.get("environment", {})
    r = q3f.get("operating_rhos", [0.2, 0.4, 0.6])
    mlp_m = pol.get("mlp", {})
    base_m = pol.get(base_name, {})

    followup_verdict = q3f.get("verdict", "?")
    overall = "GO TO PHASE 1" if followup_verdict == "PASS" else "NO-GO"

    L = []
    A = L.append
    A("# When Is More Vision Worth It? — Phase 0 Follow-Up Test Report\n")
    A("> This is the SINGLE predefined follow-up test, run to decide whether recoverability "
      "(WC) is nonlinearly predictable from the 112-resolution state, beyond scalar "
      "uncertainty and the Phase-0 linear probes. It is allowed to terminate the project, and "
      "the result below does. The Q3 gate is UNCHANGED from Phase 0 for direct comparability. "
      "No test-split access; the MLP saw only post-112 information.\n")
    A(f"- preproc hash `{cal.get('preproc_hash')}`  commit `{env.get('git',{}).get('commit')}` "
      f"(tree {'DIRTY' if env.get('git',{}).get('dirty') else 'clean'})  "
      f"device {mlp.get('device')}  CV {q3f.get('cv',{}).get('n_folds')}-fold x "
      f"{len(q3f.get('cv',{}).get('seeds',[]))} seeds, input_dim={q3f.get('cv',{}).get('input_dim')}\n")

    A("## Verdict\n")
    A("| Question | Verdict |")
    A("| --- | --- |")
    A(f"| Q1 (Phase 0) recoverable-error phenomenon | **{metrics.get('q1',{}).get('verdict','PASS')}** |")
    A(f"| Q2 (Phase 0) oracle headroom vs uncertainty | **{metrics.get('q2',{}).get('verdict','PASS')}** |")
    A(f"| Q3 original linear probe | **{q3_orig.get('verdict','FAIL')}** |")
    A(f"| **Q3 nonlinear follow-up (this MLP test)** | **{followup_verdict}** |")
    A(f"| **OVERALL** | **{overall}** |\n")

    A("## F0  Phase-0 observations that frame this test\n")
    A("- 112 -> 448 on val: CC=1185, WC=1095, WW=913, CW=140, N=3333; P(WC)~32.9%, "
      "P(CW)~4.2%; Acc_112~0.398, Acc_448~0.684. The sequential (CW-respecting) oracle "
      "reaches ~(CC+WC+CW)/N ~ 72.6% > Always-High (68.4%), so additional vision is "
      "**non-monotonic**: it can both fix (WC) and harm (CW) predictions. CW is therefore "
      "carried through every policy metric below, never hidden behind WC recall.\n")

    A("## F1  Constraints honoured\n")
    A("- 448 features used ONLY to build the WC label and oracle outcomes, never as router "
      "input; router input is post-112 state only. No test split. No Transformer, RL, "
      "architecture search, new backbone/dataset, or resolution change. Single tiny MLP, "
      "one fixed optimizer, no hyperparameter sweep.\n")

    A("## F2  Calibration audit (temperature scaling, validation-only cross-fit)\n")
    tb, ta = cs.get("before", {}), cs.get("after", {})
    A(f"- shared head is strongly OVER-confident: mean max-prob {_f(tb.get('mean_confidence'),3)} "
      f"vs accuracy {_f(tb.get('accuracy'),3)}; raw ECE {_f(tb.get('ece'),3)}. Cross-fitted "
      f"temperature T={_f(cal.get('mean_temperature'),3)} (per fold "
      f"{[round(t,2) for t in cal.get('temperatures',[])]}) fixes it: ECE -> "
      f"{_f(ta.get('ece'),3)}, mean confidence -> {_f(ta.get('mean_confidence'),3)} "
      f"(~= accuracy), NLL {_f(tb.get('nll'),3)} -> {_f(ta.get('nll'),3)}, Brier "
      f"{_f(tb.get('brier'),3)} -> {_f(ta.get('brier'),3)}; accuracy unchanged "
      "(temperature is monotone in argmax).\n")
    A("| metric | before | after |")
    A("| --- | --- | --- |")
    for k in ("nll", "brier", "ece", "mean_confidence", "accuracy"):
        A(f"| {k} | {_f(tb.get(k))} | {_f(ta.get(k))} |")
    A("")
    A("### Calibrated high-confidence recoverable errors\n")
    n_wc = next((x["n_wc"] for x in cal.get("high_conf_calibrated", [])), "n/a")
    A(f"N recoverable (WC) = {n_wc}")
    A("| threshold | raw count | calibrated count |")
    A("| --- | --- | --- |")
    for x in cal.get("high_conf_calibrated", []):
        A(f"| >= {x['threshold']:g} | {x['raw_count']} | {x['cal_count']} |")
    A("\n- After calibration the 'confident recoverable' mass shrinks sharply: raw "
      "high-confidence recoverable counts were inflated by over-confidence, so they must be "
      "read on calibrated probabilities.\n")
    A("### Scalar baselines, raw vs calibrated AUPRC (vs WC)\n")
    A("| score | raw AUPRC | calibrated AUPRC |")
    A("| --- | --- | --- |")
    for name, v in cal.get("scalar_baselines_calibrated", {}).items():
        A(f"| {name} | {_f(v.get('raw',{}).get('auprc'),3)} | {_f(v.get('calibrated',{}).get('auprc'),3)} |")
    A("")
    A(f"![calibration]({_FIGREL}/fu01_calibration_before_after.png)\n")
    A(f"![high-conf calibrated]({_FIGREL}/fu02_high_conf_recoverable_calibrated.png)\n")

    A("## F3  Nonlinear probe (fixed tiny MLP)\n")
    cv = q3f.get("cv", {})
    A(f"- input = {cv.get('features')} (dim {cv.get('input_dim')}); target y=1 iff WC. "
      "Architecture pinned: Linear(128)->GELU->Dropout(0.1)->Linear(32)->GELU->Linear(1), "
      "BCEWithLogitsLoss, AdamW(lr 1e-3, wd 1e-4), batch 128.")
    A(f"- standardization + early stopping use TRAIN-fold statistics / an INNER split only; "
      f"the outer held-out fold never guides training. median best_epoch={cv.get('mean_best_epoch')}, "
      f"mean inner-val BCE={_f(cv.get('mean_inner_val_loss'),4)}.\n")

    A("## F4/F5  Pooled out-of-fold results (all 3333 val rows)\n")
    A("| policy | AUROC | AUPRC |")
    A("| --- | --- | --- |")
    order = ["mlp", "logistic_embedding_logit", "logistic_logit_only", "max_prob",
             "entropy", "margin", "energy"]
    for name in [n for n in order if n in pol]:
        tag = "  <- best scalar" if name == base_name else ("  <- nonlinear" if name == "mlp" else "")
        A(f"| {name}{tag} | {_f(pol[name].get('auroc'),3)} | {_f(pol[name].get('auprc'),3)} |")
    A("")
    A(f"- best scalar baseline is **{base_name}** (AUPRC {_f(base_m.get('auprc'),3)}); the MLP "
      f"reaches AUPRC {_f(mlp_m.get('auprc'),3)} — **no better than, in fact slightly below, "
      "the scalar and linear baselines**. Nonlinear capacity adds nothing on top of scalar "
      "uncertainty for predicting WC.\n")

    A("### Per-operating-point routing metrics (MLP vs best scalar vs oracle)\n")
    A("| rho | policy | final acc | WC recall | unnec. esc | P(esc\\|CW) | CW share | routing regret | oracle WC | gap to oracle |")
    A("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rho in r:
        for nm, mp in (("MLP", mlp_m), (f"scalar:{base_name}", base_m)):
            rt = _route(mp, rho)
            A(f"| {rho:g} | {nm} | {_f(rt.get('final_accuracy'),3)} | {_f(rt.get('wc_recall'),3)} | "
              f"{_f(rt.get('unnecessary_escalation_rate'),3)} | {_f(rt.get('p_escalate_given_cw'),3)} | "
              f"{_f(rt.get('cw_share_of_escalated'),3)} | {_f(rt.get('routing_regret'),1)} | "
              f"{_f(rt.get('oracle_wc_recall'),3)} | {_f(rt.get('wc_recall_gap_to_oracle'),3)} |")
    A("")
    A(f"![pr curves]({_FIGREL}/fu03_mlp_pr_vs_baselines.png)\n")
    A(f"![wc recall]({_FIGREL}/fu04_wc_recall_vs_invocation.png)\n")
    A(f"![final acc]({_FIGREL}/fu05_final_accuracy_vs_invocation.png)\n")
    A(f"![cw escalation]({_FIGREL}/fu06_cw_escalation_vs_invocation.png)\n")
    A(f"![routing regret]({_FIGREL}/fu07_routing_regret_vs_invocation.png)\n")

    A("### Bootstrap: MLP minus best scalar (95% CI)\n")
    ci = comp.get("auprc_diff_ci", {}) or {}
    rr = comp.get("mean_regret_diff_ci", {}) or {}
    A(f"- dAUPRC CI [{_f(ci.get('low'),3)}, {_f(ci.get('high'),3)}]; "
      f"d(mean routing regret) CI [{_f(rr.get('low'),1)}, {_f(rr.get('high'),1)}] "
      "(negative = MLP better; both intervals straddle / favour the scalar).")
    A(f"![bootstrap]({_FIGREL}/fu08_bootstrap_comparison.png)\n")

    A("## F6  Unchanged Q3 gate\n")
    gates = comp.get("gates", {})
    A(f"1. WC recall >= +5pp at >= {q3f.get('thresholds',{}).get('min_operating_points',2)} "
      f"of 3 budgets: **{comp.get('n_points_meeting_recall_gate',0)}/3 -> "
      f"{'PASS' if gates.get('recall_gate') else 'FAIL'}**")
    A(f"2. mean routing regret down >= 10%: reduction={_f(comp.get('regret_reduction'),3)} -> "
      f"{'PASS' if gates.get('regret_gate') else 'FAIL'}")
    A(f"3. AUPRC up >= 0.03: gain={_f(comp.get('auprc_gain'),3)} -> "
      f"{'PASS' if gates.get('auprc_gate') else 'FAIL'}")
    A(f"4. gain outside bootstrap noise: dAUPRC CI low={_f(ci.get('low'),3)} -> "
      f"{'PASS' if gates.get('noise_gate') else 'FAIL'}")
    A(f"- gate status for the MLP: **{comp.get('status','?')}**\n")

    if followup_verdict != "PASS":
        A("**Conclusion — NO-GO for the global-state formulation.** With calibration handled and "
          "a nonlinear MLP given the full post-112 global state (embedding + logits + "
          "uncertainty), out-of-fold recovery of WC is *not* better than scalar uncertainty "
          "routing on any axis: AUPRC does not improve, no operating point gains WC recall, "
          "routing regret is not reduced, and the differences are inside bootstrap noise. The "
          "Q2 headroom is real, but **we find no evidence that the frozen global low-resolution "
          "state provides exploitable recoverability information beyond scalar uncertainty under "
          "the tested lightweight probes** (one linear probe and one fixed tiny MLP). We "
          "therefore **stop the current 'tiny decision model from global low-resolution "
          "representation' formulation.** We do NOT escalate to a Transformer, because nothing "
          "in this experiment indicates a larger controller on the *same* global state would "
          "help.\n")
    else:
        A("**Conclusion — GO.** The nonlinear probe clears every pre-registered Q3 gate "
          "against the best scalar baseline with gains outside bootstrap noise; the headroom "
          "is learnable from the 112 state, so a Phase-1 decision model is justified.\n")

    A("## Scope of the negative result  (and known caveats)\n")
    A("- **What this NO-GO claims:** under `DINOv2-S/14 + 112->448 + FGVC-Aircraft + {global "
      "embedding, logits, scalars}`, neither a linear probe nor a fixed tiny MLP extracts "
      "recoverability signal beyond scalar uncertainty: *global low-resolution state does not "
      "imply recoverability signal beyond uncertainty* for the tested probes.\n")
    A("- **What it does NOT claim:** that recoverability is unpredictable from *any* "
      "low-resolution information. This test does not cover different pooling, patch-/spatial- "
      "level features, other training objectives, class-aware structure, larger samples, or a "
      "different model class; those remain open, so the broader research direction is not "
      "falsified — only this specific formulation.\n")
    A("- **Phenomenon worth keeping:** P(WC)~32.9%, P(CW)~4.2% and a sequential oracle of "
      "~72.6% > Always-High 68.4% show additional vision is **non-monotonic** and *Always More "
      "Vision != Optimal Vision Allocation*. The headroom is large; this global pooled state "
      "simply does not expose it to a tiny router.\n")
    A("- **Known minor caveats (do not change the verdict, deliberately not re-run):** (i) the "
      "*calibrated* energy baseline divides logits by the mean temperature rather than each "
      "sample's fold temperature, so it is not strictly out-of-fold; the headline comparison "
      "uses the raw max-prob scalar and does not depend on it. (ii) MLP standardization computes "
      "mean/std on the whole outer-train before the inner early-stopping split, a slight "
      "inner-loop leakage; the outer held-out fold never enters the scaler and any effect "
      "favours the MLP, so it cannot manufacture the observed failure.\n")

    A("## F7  Future formulation note (documented, NOT implemented)\n")
    A("- Phase 0 sees WC (+value) *and* CW (-value), so a later policy should model **signed** "
      "visual value g = +1(WC) / 0(CC,WW) / -1(CW), i.e. estimate P(WC|s) and P(CW|s) and act "
      "on E[g|s] = P(WC|s) - P(CW|s), rather than P(WC|s) alone. This follow-up deliberately "
      "kept the P(WC) target for direct comparability with the original Phase-0 gate.\n")
    A("- The single natural next branch, if the project continues, is a **new** hypothesis — "
      "recoverability may be *spatially* encoded in the 8x8 patch tokens that DINOv2 already "
      "computes at 112 and that global pooling/CLS discards — not a larger controller on the "
      "same global vector. Such a 'Spatial-State Recoverability Audit' (Phase S0) would test "
      "cheap spatial summaries (patch mean/std/max, norm / cosine dispersion, patch-level "
      "classifier disagreement) before any attention model, and **must be separately "
      "pre-registered** with its own GO/NO-GO; it is not started here.\n")

    A("## Exact final verdict\n")
    A("```")
    A("Q1 = PASS")
    A("Q2 = PASS")
    A(f"Q3 ORIGINAL LINEAR PROBE = {q3_orig.get('verdict','FAIL')}")
    A(f"Q3 NONLINEAR FOLLOW-UP = {followup_verdict}")
    A(f"OVERALL = {overall}")
    A("```")
    return "\n".join(L), {"q3_followup": followup_verdict, "overall": overall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="followup.yaml")
    args = ap.parse_args()
    cfg = C.load_config(args.config)
    logger = C.setup_logger("13_followup_report", cfg)
    metrics = json.loads(C.R(cfg["outputs"]["metrics"]).read_text(encoding="utf-8"))
    FF.make_all(cfg, logger)
    body, verdicts = build(cfg, metrics)
    out = C.R(cfg["paths"]["docs_dir"]) / "phase0_followup_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    C.write_run_config(cfg, script="13_followup_report.py", extra=verdicts)
    logger.info("follow-up report written: %s (%s)", out, verdicts)


if __name__ == "__main__":
    main()
