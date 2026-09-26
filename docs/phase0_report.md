# When Is More Vision Worth It? — Phase 0 Audit Report

> Phase-0 phenomenon audit only: no neural decision model was trained. All numbers are drawn from `outputs/phase0/metrics.json`; every figure is rendered from its underlying table in `outputs/phase0/tables/`. Negative results are reported as they are.

- preproc hash: `30d9c89bb7dd`  
- git commit: `e4f5a1e1529a3cd096390d1ccdb8eaed85534e24` (working tree DIRTY) on branch `main`  
- seed: 1337  device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MB)

## Verdict summary

| Question | Meaning | Verdict |
| --- | --- | --- |
| Q1 | recoverable-error phenomenon exists at scale | **PASS** |
| Q2 | marginal visual value != uncertainty (learnable headroom exists) | **PASS** |
| Q3 | headroom is cheaply learnable by a lightweight probe | **FAIL** |
| **OVERALL** | GO / NO-GO / follow-up | **NEEDS ONE PREDEFINED FOLLOW-UP TEST** |

**OVERALL rule (fixed, documented):** Q1 or Q2 FAIL -> NO-GO; Q1+Q2+Q3 PASS -> GO; Q1+Q2 PASS but Q3 not a clean PASS (real headroom, weak cheap-learnability evidence) -> NEEDS ONE PREDEFINED FOLLOW-UP TEST. This run: Q1=PASS, Q2=PASS, Q3=FAIL -> **NEEDS ONE PREDEFINED FOLLOW-UP TEST**.

## Environment

- Python 3.10.19, torch 2.5.1+cu121 (CUDA 12.1), cuDNN 90100, platform Windows-10-10.0.26200-SP0.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MB; CUDA available: True.
- Key packages: PIL 12.0.0, joblib 1.5.3, matplotlib 3.10.8, numpy 1.26.4, pandas 2.3.3, pyarrow 23.0.0, sklearn 1.7.2, tqdm 4.67.3, yaml 6.0.3.

## Dataset validation

- FGVC-Aircraft 2013b: 10000 images, 100 variants; splits {'test': 3333, 'train': 3334, 'val': 3333}.
- reference checks: ALL PASS -> images_10000=ok, no_missing_images=ok, no_split_overlap=ok, test_3333=ok, train_3334=ok, val_3333=ok, variants_100=ok.

## Preprocessing

- canonical: banner trim 20px -> square pad (fill 0) -> resize to [112, 224, 448] (bicubic, antialias=True), normalize ImageNet mean/std; deterministic=True.
- preproc_hash `30d9c89bb7dd` gates every cache path.

## Backbone and feature cache

- frozen DINOv2 ViT-S/14 global embeddings (384-dim), extracted offline per resolution.

| resolution | val runtime(s) | peak VRAM(MB) | img/s | cache(MB) |
| --- | --- | --- | --- | --- |
| 112 | 147.2 | 151 | 22.6 | 5.03 |
| 224 | 161.0 | 208 | 20.7 | 5.03 |
| 448 | 33.9 | 207 | 98.4 | 5.03 |

## Linear classifier (shared head)

- sklearn logistic_regression (solver lbfgs, C 1.0, max_iter 2000, multi_class multinomial); shared across resolutions, trained on train only, so VAL predictions are out-of-sample.

### Accuracy by resolution

| resolution | val accuracy | mean top-1 conf | mean entropy |
| --- | --- | --- | --- |
| 112 | 0.3975 | 0.6924 | 0.9498 |
| 224 | 0.6220 | 0.7831 | 0.6538 |
| 448 | 0.6841 | 0.7994 | 0.6068 |

![accuracy](../outputs/phase0/figures/f01_accuracy_vs_resolution.png)

## Transition analysis (Q1)

- main transition 112->448: N=3333, counts CC/WC/WW/CW = 1185/1095/913/140.
- acc_low 0.3975 -> acc_high 0.6841 (gain 28.7 pp); WC/N=32.9%, WC/low-wrong=54.5%, net_correction=955.
- **Q1 = PASS** (criteria improvement_meaningful=True, wc_count_ge_min=True, wc_frac_all_ge_min=True, wc_frac_err_ge_min=True).

![transition matrix](../outputs/phase0/figures/f02_transition_matrix.png)

![counts](../outputs/phase0/figures/f03_transition_counts.png)

## High-confidence recoverable errors

- conf >= 0.8: 253 recoverable (23.1% of WC, 7.6% of all); meets strong signal: True.
- conf >= 0.9: 124 recoverable (11.3% of WC, 3.7% of all); meets strong signal: True.
- any strong signal: True. A sizeable slice of recoverable errors sits at HIGH confidence, i.e. plain confidence would not escalate them -> supports the value-of-vision-vs-uncertainty distinction.

![confidence dist](../outputs/phase0/figures/f04_confidence_distribution.png)

![p recoverable by bin](../outputs/phase0/figures/f05_p_recoverable_by_conf_bin.png)

## Uncertainty baselines

| score | AUROC | AUPRC | base rate |
| --- | --- | --- | --- |
| energy | 0.633 | 0.423 | 0.329 |
| entropy | 0.668 | 0.447 | 0.329 |
| margin | 0.671 | 0.455 | 0.329 |
| max_prob | 0.673 | 0.459 | 0.329 |

Scalar uncertainty scores separate WC only weakly (AUROC ~0.63-0.67).

![pr curves](../outputs/phase0/figures/f06_pr_curves_scalar_baselines.png)

## Oracle headroom (Q2)

| rho | best scalar | scalar acc | oracle acc | acc gap (pp) | scalar WC-recall | oracle WC-recall | recall gap (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.20 | entropy | 0.4873 | 0.5977 | 11.0 | 0.289 | 0.609 | 32.1 |
| 0.40 | max_prob | 0.5659 | 0.7261 | 16.0 | 0.555 | 1.000 | 44.5 |
| 0.60 | entropy | 0.6259 | 0.7261 | 10.0 | 0.769 | 1.000 | 23.1 |

- **Q2 = PASS: the budget-matched sequential oracle beats the best scalar router by a large margin at every operating point, so real headroom exists that a learned value router could in principle capture.**

![acc vs rho](../outputs/phase0/figures/f07_accuracy_vs_invocation_rate.png)

![recall vs rho](../outputs/phase0/figures/f08_recoverable_recall_vs_invocation_rate.png)

![scalar vs oracle](../outputs/phase0/figures/f09_scalar_vs_oracle.png)

## Lightweight predictability probe (Q3)

- exploratory temporary VAL split (mode=temporary_val_split), fit=2000 eval=1333 rows, stratified on WC; probes never scored on the rows they fit. Best scalar baseline on the eval fold: **max_prob** (AUPRC 0.479).

| probe | features | AUROC | AUPRC | AUPRC gain vs best scalar | AUPRC gain 95% CI | regret reduction | recall pts >= +5pp | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probe_a | 5 blocks | 0.673 | 0.466 | -0.014 | [-0.050, 0.023] | -0.040 | 0/3 | NO-GO |
| probe_b | 6 blocks | 0.649 | 0.458 | -0.022 | [-0.067, 0.024] | -0.078 | 0/3 | NO-GO |

- **Q3 = FAIL (NO-GO / weak predictability evidence).** Neither probe beat the best scalar uncertainty baseline on the held-out eval fold: AUPRC did not improve (both negative), no operating point met the +5 pp recall gate, and the probe-minus-baseline AUPRC difference is inside bootstrap noise (CI covers 0). The headroom Q2 exposed is therefore **not readable by a trivial logistic probe** on low-resolution features/embeddings.

- underlying per-policy metrics and curves: `outputs/phase0/tables/probe_metrics.csv`, `probe_curves.csv`, `probe_pr_curves.csv`.

## Runtime / VRAM / storage

- extraction peak VRAM (val): 112=151 MB, 224=208 MB, 448=207 MB (well within the 8 GB budget; batch sizes 64/32/8).
- audit stages (07-09) run in seconds on cached features; all long-job stdout is under `outputs/logs/run_*.log`.

## GO / NO-GO

- Q1 PASS / Q2 PASS / Q3 FAIL -> **OVERALL: NEEDS ONE PREDEFINED FOLLOW-UP TEST**.
- The core research premise SURVIVES: more vision genuinely has marginal value beyond uncertainty (Q1 strong; Q2 large oracle-vs-scalar headroom). But a cheap logistic probe could not read that headroom (Q3 NO-GO).
- **Do NOT proceed to design the Transformer / decision model now** (plan §9.5). One bounded, predefined follow-up test is warranted before any GO: a stronger — but still non-neural — learnability probe over the SAME post-112 low-resolution state (e.g. gradient-boosted / cross-validated features). High-res (448) features are supervision/oracle only and can never be router inputs before escalation, per the causal routing constraint. Pre-register against the same Q3 gates; if that also fails, the honest outcome is NO-GO.

## Prohibited-scope reminder (plan §19)

No Transformer controller, RL, adaptive crop, active vision, token pruning, other datasets, backbone fine-tuning, or hyperparameter sweeps were run — all deferred until a Phase-0 GO.
