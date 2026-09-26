# Environment Report — Phase 0 Setup

Conda env: **deepminer** (`E:\conda\envs\deepminer`)
Verification date: 2026-09-26
Status: **All required dependencies present. Nothing was installed or upgraded.**

## System / Runtime

| Item | Value |
| --- | --- |
| OS | Windows 11 |
| Python | 3.10.19 (conda-forge, MSC v.1944 64-bit) |
| Python executable | `E:\conda\envs\deepminer\python.exe` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| GPU memory | 8.0 GB |
| GPU compute capability | (8, 9) |
| NVIDIA driver | 572.61 |
| CUDA (torch build) | 12.1 (`torch.version.cuda`) |
| `torch.cuda.is_available()` | True |
| cuDNN | 9.1.0 (`cudnn.version() == 90100`) |

## Core Packages (verified present, pinned in `requirements.txt`)

| Package | Version | Note |
| --- | --- | --- |
| torch | 2.5.1+cu121 | CUDA 12.1 build |
| torchvision | 0.20.1+cu121 | matched to torch 2.5.1 |
| numpy | 1.26.4 | |
| pandas | 2.3.3 | |
| scikit-learn | 1.7.2 | |
| matplotlib | 3.10.8 | |
| pyyaml | 6.0.3 | |
| tqdm | 4.67.3 | |
| pillow | 12.0.0 | |
| pyarrow | 23.0.0 | |

## GPU Smoke Test — PASS

| Test | Result |
| --- | --- |
| 1000x1000 `torch.randn` matmul on CUDA | OK (result mean = -0.0010, `torch.cuda.synchronize()` clean) |
| fp16 conv under `torch.inference_mode()` | OK (output `(1, 16, 64, 64)`, dtype `float16`) |
| fp16 conv under `torch.autocast(device_type="cuda")` + `inference_mode()` | OK (output dtype `float16`) |

Conclusion: GPU inference, half precision, and AMP are all functional. The environment is ready for Phase 0 (frozen DINOv2 ViT-S/14 feature extraction with mixed-precision GPU inference and offline feature caching).

## Installed / Upgraded

- **Nothing.** All 10 required packages were already present at the versions listed above.
- No package was upgraded; `torch`/`torchvision` CUDA 12.1 wheels were already installed and match the driver.
