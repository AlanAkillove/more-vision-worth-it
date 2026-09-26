# More vision worth it

**When Is More Vision Worth It?** studies whether a lightweight decision model can predict when additional visual evidence will actually improve a visual prediction, rather than merely detecting uncertain or difficult samples.

**Key words**: adaptive-vision, dynamic-inference, fine-grained-recognition, uncertainty, calibration, budget-aware-inference, dinov2

## Overview

This project asks whether a lightweight decision model can predict the **marginal value of additional visual evidence** for budget-aware visual recognition. The goal is to decide *when spending more visual budget is actually worth it*, and to distinguish that question from the simpler one of *how uncertain a current prediction is*.

### Uncertainty vs. marginal visual value

These are deliberately separate concepts:

- **Uncertainty** = *how unsure the current prediction is*. A model output may have high uncertainty for many reasons (ambiguity, distribution shift, hard samples) that more visual evidence cannot necessarily fix.
- **Marginal visual value** = *whether additional visual evidence would actually change/correct the prediction*. This is the quantity of interest: it is about the potential gain from spending more budget, not merely the presence of doubt.

A high-uncertainty sample is not automatically a sample where more vision helps, and vice versa. This repository is organized around measuring and predicting that distinction.

## Repository structure

```
more-vision-worth-it/
├── src/mvwi/        # Core Python package (import name: mvwi)
├── configs/         # YAML configuration files
├── scripts/         # Entry-point scripts (data prep, feature extraction, experiments)
├── docs/            # Plans, environment report, dataset notes
├── tests/           # Test suite
├── requirements.txt # Pinned core dependencies
└── pyproject.toml   # Package metadata
```

## Environment

The project runs in the conda environment `deepminer` (Python 3.10, PyTorch built for CUDA 12.1, NVIDIA RTX 4060 Laptop GPU):

```powershell
conda activate deepminer
```

See `docs/environment_report.md` for the verified environment details and `requirements.txt` for pinned versions. (Tip: `conda run -n deepminer python ...` works in non-interactive shells.)

## Status

**Phase 0 — complete, with a scoped NO-GO.** The project ran a pre-registered Oracle Headroom audit plus a single pre-defined nonlinear follow-up test, and reached **NO-GO for global-state tiny routing**.

| Question | Verdict |
| --- | --- |
| Q1 — recoverable-error phenomenon exists | **PASS** (at 112→448: WC ≈ 32.9%, CW ≈ 4.2%; additional vision is non-monotonic) |
| Q2 — oracle headroom over uncertainty | **PASS** (sequential oracle ≈ 72.6% > Always-High 68.4%) |
| Q3 — recoverability readable from global 112 state | **FAIL** (a linear probe and a fixed tiny MLP both ≤ best scalar uncertainty) |

Headline finding: *Always More Vision ≠ Optimal Vision Allocation*. The oracle headroom is real, but the frozen **global** low-resolution state does not expose it to a lightweight router. This is reported honestly as a *scoped* negative result: it does **not** claim recoverability is unpredictable from all low-resolution information (spatial / patch-level state is untested). See `docs/phase0_report.md` and `docs/phase0_followup_report.md`.

## Roadmap

- **Phase 0** — inference-only audit on DINOv2 ViT-S/14: frozen feature extraction, offline caching, transition/oracle analysis, calibration audit, and linear + nonlinear learnability probes. Status: **complete — NO-GO for the global-state formulation**; no further rescue of this formulation is planned.
- **Phase S0 (proposed, not started)** — *Spatial-State Recoverability Audit*: test whether the 8×8 patch tokens DINOv2 already computes at 112 encode recoverability that global pooling/CLS discards. This is a **new, separately pre-registered hypothesis** (cheap spatial summaries before any attention model), not a continuation or a larger controller for Phase 0.

