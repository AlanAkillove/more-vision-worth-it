"""CLI 04: train the shared linear classifier (plan §6 / M3).

Reads the offline DINOv2 feature caches (produced by 03) for the TRAIN split across all
three resolutions, fits ONE shared head h: R^384 -> R^K on their joint, persists it as a
pickle-free .npz, and records train/val accuracy per resolution into metrics.json.

Run it yourself to watch progress:
  conda activate deepminer
  python scripts/04_train_classifier.py --config configs/classifier.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mvwi import config as C
from mvwi.eval import metrics as M
from mvwi.features import extract as EX
from mvwi.models import linear_probe as LP
from mvwi.preprocess.version import compute_preproc_hash


def _acc_split(bundle, cfg, phash, resolutions, split):
    out = {}
    for res in resolutions:
        f = EX.load_features(cfg, phash, res, split)
        y = f["ground_truth_label"]
        logits = LP.decision_logits(bundle, f["embedding"])
        pred = logits.argmax(axis=1)
        out[str(int(res))] = {"n": int(len(y)), "accuracy": round(M.accuracy(y, pred), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="classifier.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("04_train_classifier", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    resolutions = cfg["resolutions"]
    logger.info("preproc_hash=%s resolutions=%s", phash, resolutions)

    bundle = LP.train_shared_classifier(cfg, phash, resolutions, logger)
    cp = LP.save_classifier(cfg, phash, bundle)
    logger.info("saved classifier artifact -> %s", cp)

    train_acc = _acc_split(bundle, cfg, phash, resolutions, "train")
    val_acc = _acc_split(bundle, cfg, phash, resolutions, "val")
    for tag, accs in (("train", train_acc), ("val", val_acc)):
        logger.info("%s accuracy by resolution: %s", tag, accs)

    C.update_metrics(cfg, "classifier", {
        "preproc_hash": phash,
        "shared_head": True,
        "trained_on": {"split": "train", "resolutions": resolutions,
                        "n_rows_joint": int(sum(v["n"] for v in train_acc.values()))},
        "hyperparameters": bundle["hp"],
        "train_accuracy_by_resolution": train_acc,
        "val_accuracy_by_resolution": val_acc,
        "artifact": str(cp),
    })
    C.write_run_config(cfg, script="04_train_classifier.py", extra={"preproc_hash": phash})
    logger.info("classifier training complete")


if __name__ == "__main__":
    main()
