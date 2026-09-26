"""Shared linear classifier over frozen-DINOv2 embeddings (plan §6 / M3).

One and only one head h: R^384 -> R^K is trained on the JOINT of the three resolution
embedding sets from the train split {z_112, z_224, z_448}. The same head must serve all
three visual budgets; we never train a per-resolution classifier. Standardisation is fit
strictly on train data to avoid leakage into the validation audit.

The trained head is persisted as a plain ``.npz`` of float arrays (coef_/intercept_ plus
the standardizer statistics). This is pickle-free (no arbitrary-object deserialization)
and reproducible across scikit-learn versions: the multinomial lbfgs head's logits are
recomputed directly as z = (X - mean)/scale @ coef_.T + intercept_, and softmax(z) equals
the estimator's predict_proba.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mvwi.config import R
from mvwi.features import extract as EX


def classifier_path(cfg, preproc_hash):
    return R(cfg["paths"]["classifier_template"].format(preproc_hash=preproc_hash))


def load_joint_embeddings(cfg, preproc_hash, resolutions, split):
    """Concatenate embeddings across resolutions for one split.

    Returns (X, y, sample_id, res_tag, class_names). Each physical sample contributes
    len(resolutions) rows, one per visual budget, all sharing the same label.
    """
    xs, ys, ids, tags = [], [], [], []
    class_names = None
    for res in resolutions:
        f = EX.load_features(cfg, preproc_hash, res, split)
        xs.append(np.asarray(f["embedding"], dtype=np.float32))
        ys.append(np.asarray(f["ground_truth_label"], dtype=np.int64))
        ids.append(np.asarray(f["sample_id"]))
        tags.append(np.full(len(f["sample_id"]), int(res)))
        if class_names is None:
            class_names = list(f["class_names"])
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sid = np.concatenate(ids, axis=0)
    tag = np.concatenate(tags, axis=0)
    return X, y, sid, tag, class_names


def train_shared_classifier(cfg, preproc_hash, resolutions, logger):
    """Fit the shared head on the joint train features. Returns a pickle-free bundle dict."""
    mc = cfg["classifier"]
    X, y, sid, tag, class_names = load_joint_embeddings(cfg, preproc_hash, resolutions, "train")
    logger.info("joint train matrix X=%s (rows=%d, dim=%d); %d rows/sample across %d resolutions",
                X.shape, X.shape[0], X.shape[1], len(resolutions), len(resolutions))

    scaler = None
    if mc.get("standardize", {}).get("enable", True):
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)  # fit ONLY on train -> no val leakage
    else:
        X_fit = X

    hp = {
        "model": "logistic_regression",
        "solver": mc.get("solver", "lbfgs"),
        "C": float(mc.get("C", 1.0)),
        "max_iter": int(mc.get("max_iter", 1000)),
        "multi_class": mc.get("multi_class", "multinomial"),
        "class_weight": mc.get("class_weight", None),
        "n_jobs": int(mc.get("n_jobs", -1)),
        "seed": int(cfg["project"]["seed"]),
    }
    # NOTE: for scikit-learn >= 1.5 the `multi_class` argument is deprecated and lbfgs is
    # already multinomial (softmax over all classes) by default, so we record the intended
    # setting for reproducibility but do NOT pass it to the estimator (avoids the warning).
    clf = LogisticRegression(
        solver=hp["solver"], C=hp["C"], max_iter=hp["max_iter"],
        class_weight=hp["class_weight"], n_jobs=hp["n_jobs"], random_state=hp["seed"],
    )
    clf.fit(X_fit, y)
    logger.info("trained LogisticRegression %s (n_iter=%d)", hp, int(clf.n_iter_[0]))

    bundle = {
        "coef_": np.asarray(clf.coef_, dtype=np.float64),
        "intercept_": np.asarray(clf.intercept_, dtype=np.float64),
        "classes_": np.asarray(clf.classes_, dtype=np.int64),
        "mean_": (np.asarray(scaler.mean_, dtype=np.float64) if scaler is not None
                  else np.zeros(0, dtype=np.float64)),
        "scale_": (np.asarray(scaler.scale_, dtype=np.float64) if scaler is not None
                   else np.zeros(0, dtype=np.float64)),
        "class_names": np.array(class_names),
        "resolutions": np.array(resolutions, dtype=np.int64),
        "preproc_hash": np.array(preproc_hash),
        "standardize": np.array(1 if scaler is not None else 0, dtype=np.int64),
        "n_classes": np.array(int(len(class_names)), dtype=np.int64),
        "hp": hp,
    }
    return bundle


def save_classifier(cfg, preproc_hash, bundle):
    """Persist the head as a pickle-free .npz (hp dict flattened to string arrays)."""
    cp = classifier_path(cfg, preproc_hash)
    cp.parent.mkdir(parents=True, exist_ok=True)
    arr = {k: v for k, v in bundle.items() if k != "hp"}
    hp = bundle.get("hp", {})
    arr["hp_keys"] = np.array(list(hp.keys()), dtype=object).astype(str)
    arr["hp_vals"] = np.array([str(v) for v in hp.values()], dtype=object).astype(str)
    np.savez(cp, **arr)  # allow_pickle stays default False; every value is a plain array
    return cp


def load_classifier(cfg, preproc_hash):
    cp = classifier_path(cfg, preproc_hash)
    z = np.load(cp, allow_pickle=False)
    b = {k: z[k] for k in z.files}
    if "hp_keys" in z.files:
        b["hp"] = _coerce_hp(z["hp_keys"].tolist(), z["hp_vals"].tolist())
    return b


_HP_INT_KEYS = {"max_iter", "n_jobs", "seed"}
_HP_FLOAT_KEYS = {"C"}


def _coerce_hp(keys, vals):
    """Rebuild the hyperparameter dict with native types from its stringified .npz form."""
    hp = {}
    for k, v in zip(keys, vals):
        if v == "None":
            hp[k] = None
        elif k in _HP_INT_KEYS:
            hp[k] = int(float(v))
        elif k in _HP_FLOAT_KEYS:
            hp[k] = float(v)
        else:
            hp[k] = v
    return hp


def _is_standardized(bundle):
    s = bundle.get("standardize")
    if s is None:
        return False
    return int(np.asarray(s).reshape(-1)[0]) == 1


def decision_logits(bundle, X):
    """Raw per-class logits from the shared head, recomputed with pure numpy."""
    Xd = np.asarray(X, dtype=np.float64)
    if _is_standardized(bundle):
        mean = np.asarray(bundle["mean_"], dtype=np.float64)
        scale = np.asarray(bundle["scale_"], dtype=np.float64)
        Xd = (Xd - mean) / scale
    coef = np.asarray(bundle["coef_"], dtype=np.float64)
    intercept = np.asarray(bundle["intercept_"], dtype=np.float64)
    return Xd @ coef.T + intercept
