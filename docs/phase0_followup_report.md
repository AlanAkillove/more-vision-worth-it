# When Is More Vision Worth It? — Phase 0 Follow-Up Test Report

> This is the SINGLE predefined follow-up test, run to decide whether recoverability (WC) is nonlinearly predictable from the 112-resolution state, beyond scalar uncertainty and the Phase-0 linear probes. It is allowed to terminate the project, and the result below does. The Q3 gate is UNCHANGED from Phase 0 for direct comparability. No test-split access; the MLP saw only post-112 information.

- preproc hash `30d9c89bb7dd`  commit `e4f5a1e1529a3cd096390d1ccdb8eaed85534e24` (tree DIRTY)  device cpu  CV 5-fold x 3 seeds, input_dim=488

## Verdict

| Question | Verdict |
| --- | --- |
| Q1 (Phase 0) recoverable-error phenomenon | **PASS** |
| Q2 (Phase 0) oracle headroom vs uncertainty | **PASS** |
| Q3 original linear probe | **FAIL** |
| **Q3 nonlinear follow-up (this MLP test)** | **FAIL** |
| **OVERALL** | **NO-GO** |

## F0  Phase-0 observations that frame this test

- 112 -> 448 on val: CC=1185, WC=1095, WW=913, CW=140, N=3333; P(WC)~32.9%, P(CW)~4.2%; Acc_112~0.398, Acc_448~0.684. The sequential (CW-respecting) oracle reaches ~(CC+WC+CW)/N ~ 72.6% > Always-High (68.4%), so additional vision is **non-monotonic**: it can both fix (WC) and harm (CW) predictions. CW is therefore carried through every policy metric below, never hidden behind WC recall.

## F1  Constraints honoured

- 448 features used ONLY to build the WC label and oracle outcomes, never as router input; router input is post-112 state only. No test split. No Transformer, RL, architecture search, new backbone/dataset, or resolution change. Single tiny MLP, one fixed optimizer, no hyperparameter sweep.

## F2  Calibration audit (temperature scaling, validation-only cross-fit)

- shared head is strongly OVER-confident: mean max-prob 0.692 vs accuracy 0.398; raw ECE 0.295. Cross-fitted temperature T=2.293 (per fold [2.29, 2.3, 2.31, 2.28, 2.29]) fixes it: ECE -> 0.032, mean confidence -> 0.373 (~= accuracy), NLL 3.297 -> 2.393, Brier 0.856 -> 0.746; accuracy unchanged (temperature is monotone in argmax).

| metric | before | after |
| --- | --- | --- |
| nll | 3.2967 | 2.3925 |
| brier | 0.8564 | 0.7460 |
| ece | 0.2949 | 0.0317 |
| mean_confidence | 0.6924 | 0.3735 |
| accuracy | 0.3975 | 0.3975 |

### Calibrated high-confidence recoverable errors

N recoverable (WC) = 1095
| threshold | raw count | calibrated count |
| --- | --- | --- |
| >= 0.8 | 253.0 | 7.0 |
| >= 0.9 | 124.0 | 1.0 |

- After calibration the 'confident recoverable' mass shrinks sharply: raw high-confidence recoverable counts were inflated by over-confidence, so they must be read on calibrated probabilities.

### Scalar baselines, raw vs calibrated AUPRC (vs WC)

| score | raw AUPRC | calibrated AUPRC |
| --- | --- | --- |
| energy | 0.423 | 0.412 |
| entropy | 0.447 | 0.427 |
| margin | 0.455 | 0.456 |
| max_prob | 0.459 | 0.447 |

![calibration](../outputs/phase0/figures/fu01_calibration_before_after.png)

![high-conf calibrated](../outputs/phase0/figures/fu02_high_conf_recoverable_calibrated.png)

## F3  Nonlinear probe (fixed tiny MLP)

- input = ['z_112', 'logits', 'confidence', 'entropy', 'margin', 'energy'] (dim 488); target y=1 iff WC. Architecture pinned: Linear(128)->GELU->Dropout(0.1)->Linear(32)->GELU->Linear(1), BCEWithLogitsLoss, AdamW(lr 1e-3, wd 1e-4), batch 128.
- standardization + early stopping use TRAIN-fold statistics / an INNER split only; the outer held-out fold never guides training. median best_epoch=1.73, mean inner-val BCE=0.5947.

## F4/F5  Pooled out-of-fold results (all 3333 val rows)

| policy | AUROC | AUPRC |
| --- | --- | --- |
| mlp  <- nonlinear | 0.645 | 0.440 |
| logistic_embedding_logit | 0.639 | 0.446 |
| logistic_logit_only | 0.659 | 0.440 |
| max_prob  <- best scalar | 0.673 | 0.459 |
| entropy | 0.668 | 0.447 |
| margin | 0.671 | 0.455 |
| energy | 0.633 | 0.423 |

- best scalar baseline is **max_prob** (AUPRC 0.459); the MLP reaches AUPRC 0.440 — **no better than, in fact slightly below, the scalar and linear baselines**. Nonlinear capacity adds nothing on top of scalar uncertainty for predicting WC.

### Per-operating-point routing metrics (MLP vs best scalar vs oracle)

| rho | policy | final acc | WC recall | unnec. esc | P(esc\|CW) | CW share | routing regret | oracle WC | gap to oracle |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2 | MLP | 0.483 | 0.280 | 0.540 | 0.164 | 0.034 | 383.0 | 0.609 | 0.329 |
| 0.2 | scalar:max_prob | 0.487 | 0.292 | 0.520 | 0.157 | 0.033 | 369.0 | 0.609 | 0.317 |
| 0.4 | MLP | 0.560 | 0.528 | 0.566 | 0.264 | 0.028 | 554.0 | 1.000 | 0.472 |
| 0.4 | scalar:max_prob | 0.566 | 0.555 | 0.544 | 0.336 | 0.035 | 534.0 | 1.000 | 0.445 |
| 0.6 | MLP | 0.620 | 0.745 | 0.592 | 0.543 | 0.038 | 355.0 | 1.000 | 0.255 |
| 0.6 | scalar:max_prob | 0.624 | 0.769 | 0.579 | 0.614 | 0.043 | 339.0 | 1.000 | 0.231 |

![pr curves](../outputs/phase0/figures/fu03_mlp_pr_vs_baselines.png)

![wc recall](../outputs/phase0/figures/fu04_wc_recall_vs_invocation.png)

![final acc](../outputs/phase0/figures/fu05_final_accuracy_vs_invocation.png)

![cw escalation](../outputs/phase0/figures/fu06_cw_escalation_vs_invocation.png)

![routing regret](../outputs/phase0/figures/fu07_routing_regret_vs_invocation.png)

### Bootstrap: MLP minus best scalar (95% CI)

- dAUPRC CI [-0.046, 0.007]; d(mean routing regret) CI [-8.0, 43.0] (negative = MLP better; both intervals straddle / favour the scalar).
![bootstrap](../outputs/phase0/figures/fu08_bootstrap_comparison.png)

## F6  Unchanged Q3 gate

1. WC recall >= +5pp at >= 2 of 3 budgets: **0/3 -> FAIL**
2. mean routing regret down >= 10%: reduction=-0.040 -> FAIL
3. AUPRC up >= 0.03: gain=-0.020 -> FAIL
4. gain outside bootstrap noise: dAUPRC CI low=-0.046 -> FAIL
- gate status for the MLP: **NO-GO**

**Conclusion — NO-GO.** With calibration handled and a nonlinear MLP given the full 112 representation (embedding + logits + uncertainty), out-of-fold recovery of WC is *not* better than scalar uncertainty routing on any axis: AUPRC does not improve, no operating point gains WC recall, routing regret is not reduced, and the differences are inside bootstrap noise. The Q2 headroom is real but is **not learnable from the global low-resolution state** by a model this size. We therefore **stop the current 'tiny decision model from global low-resolution representation' formulation.** We do NOT escalate to a Transformer because the MLP failed — the failure is about the *input representation*, not model class.

## F7  Future formulation note (documented, NOT implemented)

- Phase 0 sees WC (+value) *and* CW (-value), so a later policy should model **signed** visual value g = +1(WC) / 0(CC,WW) / -1(CW), i.e. estimate P(WC|s) and P(CW|s) and act on E[g|s] = P(WC|s) - P(CW|s), rather than P(WC|s) alone. This follow-up deliberately kept the P(WC) target for direct comparability with the original Phase-0 gate.

## Exact final verdict

```
Q1 = PASS
Q2 = PASS
Q3 ORIGINAL LINEAR PROBE = FAIL
Q3 NONLINEAR FOLLOW-UP = FAIL
OVERALL = NO-GO
```