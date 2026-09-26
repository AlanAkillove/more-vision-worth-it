"""CLI 00: environment capture (plan §17). Writes outputs/logs/env_report.json and
merges the 'environment' section into metrics.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvwi import config as C
from mvwi import env_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="base.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    seed = cfg["project"]["seed"]
    logger = C.setup_logger("00_env_check", cfg)
    C.set_seed(seed)

    env = env_report.capture_env(seed=seed)
    env["git"] = C.capture_git(cfg)
    env["device"] = cfg.get("device")

    out = C.R(cfg["outputs"]["logs_dir"]) / "env_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2, sort_keys=True), encoding="utf-8")

    C.update_metrics(cfg, "environment", env)
    C.write_run_config(cfg, script="00_env_check.py")

    logger.info("python=%s torch=%s cuda=%s gpu=%s", env["python"], env["torch"],
                env["cuda_available"], env["gpu"])
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
