"""Environment provenance capture (plan §17 reproducibility)."""
from __future__ import annotations

import importlib
import os
import platform


def capture_env_lite():
    """Small env snapshot used inside run_config.yaml."""
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # pragma: no cover
        info["torch_error"] = repr(e)
    return info


def capture_env(seed=None):
    import torch

    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gpu_mem_mb = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        if torch.cuda.is_available()
        else None
    )
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": cudnn,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "gpu_total_mem_mb": gpu_mem_mb,
        "seed": seed,
        "packages": {},
    }
    for pkg in ["numpy", "pandas", "sklearn", "matplotlib", "yaml", "tqdm", "PIL", "pyarrow", "joblib"]:
        try:
            m = importlib.import_module(pkg)
            info["packages"][pkg] = getattr(m, "__version__", "?")
        except Exception:
            info["packages"][pkg] = None
    return info
