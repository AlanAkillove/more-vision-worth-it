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

## Roadmap

- **Phase 0** — inference-only setup on DINOv2 ViT-S/14: frozen feature extraction, offline feature caching, and a linear probe baseline. Status: **planned / not yet executed**; see `docs/phase0_plan.md`.

