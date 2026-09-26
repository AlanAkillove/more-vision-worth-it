"""Config loading, path resolution, seeding, metrics/run-config IO, logging.

All experiment scripts read a YAML config (merged over configs/base.yaml) and
emit outputs/phase0/metrics.json + outputs/phase0/run_config.yaml per plan §17.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _deep_merge(base, over):
    out = copy.deepcopy(base) if isinstance(base, dict) else {}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(name_or_path):
    p = Path(name_or_path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    cand = CONFIGS_DIR / name_or_path
    if not cand.exists():
        cand = CONFIGS_DIR / (str(name_or_path) + ".yaml")
    return cand


def load_config(name_or_path, with_base=True):
    """Load a config file (by name in configs/ or a path), deep-merged over base.yaml."""
    p = _resolve_config_path(name_or_path)
    cfg = load_yaml(p)
    if with_base and p.name != "base.yaml":
        cfg = _deep_merge(load_yaml(CONFIGS_DIR / "base.yaml"), cfg)
    cfg["_config_path"] = str(p)
    return cfg


def R(*parts):
    """Resolve a path relative to the project root; absolute paths pass through."""
    q = Path(*parts) if len(parts) > 1 else Path(str(parts[0]))
    return q if q.is_absolute() else (PROJECT_ROOT / q)


def set_seed(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_stamp():
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(name, cfg, filename=None):
    log_dir = R(cfg["outputs"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"run_{run_stamp()}_{name}.log"
    logfile = log_dir / fname
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(cfg.get("logging", {}).get("level", "INFO")).upper()))
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.info("log file: %s", logfile)
    return logger


def read_metrics(cfg):
    mp = R(cfg["outputs"]["metrics"])
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return {}


def update_metrics(cfg, section, value):
    """Incrementally merge one top-level section into metrics.json."""
    mp = R(cfg["outputs"]["metrics"])
    mp.parent.mkdir(parents=True, exist_ok=True)
    data = read_metrics(cfg)
    data[section] = _deep_merge(data.get(section, {}), value) if isinstance(value, dict) else value
    mp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def capture_git(cfg):
    if not cfg.get("git", {}).get("capture_commit", True):
        return {}

    def _run(cmd):
        try:
            return subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def _to_plain(obj):
    """Round-trip through JSON so only native types reach the YAML SafeDumper
    (numpy/torch string or number subclasses are otherwise unrepresentable)."""
    return json.loads(json.dumps(obj, default=str))


def write_run_config(cfg, script, extra=None):
    """Append this run's config snapshot + provenance to run_config.yaml."""
    from mvwi import env_report

    rc_path = R(cfg["outputs"]["run_config"])
    rc_path.parent.mkdir(parents=True, exist_ok=True)
    doc = load_yaml(rc_path) if rc_path.exists() else {"runs": []}
    snap = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    git = capture_git(cfg)
    entry = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "config_path": cfg.get("_config_path"),
        "seed": cfg.get("project", {}).get("seed"),
        "git": git,
        "env": env_report.capture_env_lite(),
        "config": snap,
    }
    if extra:
        entry["extra"] = extra
    doc["runs"] = doc.get("runs", []) + [entry]
    doc = _to_plain(doc)
    with open(rc_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    if cfg.get("git", {}).get("warn_if_dirty", True) and git.get("dirty"):
        logging.getLogger(__name__).warning("working tree is DIRTY at commit %s", git.get("commit"))
    return entry
