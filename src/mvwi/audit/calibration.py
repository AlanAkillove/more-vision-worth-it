"""Calibration audit for the Phase-0 follow-up (brief F2).

Phase 0 exposed a strongly OVER-confident shared head: Acc_112 ~= 0.398 but mean max-prob
~= 0.69. So raw "high-confidence recoverable" counts are not yet calibrated evidence.

We fix this with validation-only cross-fitted TEMPERATURE SCALING (Guo et al. 2017): a
single scalar T > 0 fit by minimising the held-out-class NLL of softmax(logits / T). The
key discipline is that T for a fold is fit ONLY on the other folds and applied to the
held-out fold, so every calibrated probability below is out-of-fold (never fit on the same
sample it is scored on). Temperature scaling is monotone in the logits, so argmax accuracy
is unchanged -- it only reshapes confidence, which is exactly what we need to re-read the
high-confidence recoverable-error counts and the confidence-based scalar baselines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from mvwi.audit import outcomes as OC
from mvwi.eval import metrics as M


def load_val_state(cfg, phash, low):
    """Return (logits [n,K], y_true [n], sample_id [n]) for the low-resolution val rows."""
    df = OC.read_outcomes(cfg, phash, "val")
    lo = df[df["resolution"] == int(low)].copy()
    logits = np.stack(lo["logits"].to_numpy()).astype(np.float64)
    y_true = lo["y_true"].to_numpy(dtype=np.int64)
    sid = lo["sample_id"].astype(str).to_numpy()
    return logits, y_true, sid


def _nll_from_logits(logits, y, T):
    x = torch.as_tensor(logits, dtype=torch.float64) / T
    return torch.nn.functional.cross_entropy(x, torch.as_tensor(y, dtype=torch.long))


def fit_temperature(logits, y, opt_cfg, device="cpu"):
    """Fit a single scalar temperature T on ONE fold (minimise NLL of softmax(logits/T))."""
    x = torch.as_tensor(logits, dtype=torch.float64).to(device)
    t = torch.as_tensor(y, dtype=torch.long).to(device)
    log_T = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)  # T=1 init
    opt = torch.optim.Adam([log_T], lr=float(opt_cfg.get("lr", 0.05)))
    for _ in range(int(opt_cfg.get("max_iter", 300))):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_T.exp(), t)
        loss.backward()
        opt.step()
    return float(log_T.exp().detach().cpu().item())


def oof_calibrated_probs(logits, y_true, cal_cfg, device="cpu"):
    """Cross-fitted out-of-fold calibrated probabilities + per-fold temperatures."""
    n_folds = int(cal_cfg["n_folds"])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(1337))
    probs = np.zeros_like(logits)
    temps = []
    for tr, te in skf.split(logits, y_true):
        T = fit_temperature(logits[tr], y_true[tr], cal_cfg.get("optimizer", {}), device)
        temps.append(T)
        probs[te] = M.softmax(logits[te] / T)
    return probs, temps


def nll(probs, y):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.log(p[np.arange(len(y)), np.asarray(y)]).mean())


def brier(probs, y):
    """Multiclass Brier score: mean squared distance to the one-hot target."""
    p = np.asarray(probs, dtype=np.float64)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), np.asarray(y)] = 1.0
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def ece(probs, y, bins=15):
    """Expected calibration error over confidence bins (max-prob as confidence)."""
    conf = M.top1_confidence(probs)
    correct = (M.predicted_class(probs) == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    N = len(conf)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_table(probs, y, bins=15, tag=""):
    conf = M.top1_confidence(probs)
    correct = (M.predicted_class(probs) == np.asarray(y)).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        rows.append({"tag": tag, "bin": b, "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
                     "n": n, "mean_confidence": float(conf[m].mean()) if n else float("nan"),
                     "accuracy": float(correct[m].mean()) if n else float("nan")})
    return pd.DataFrame(rows)


def calibration_summary(raw_probs, cal_probs, y, bins=15):
    def block(probs):
        return {"nll": round(nll(probs, y), 5), "brier": round(brier(probs, y), 5),
                "ece": round(ece(probs, y, bins), 5),
                "mean_confidence": round(float(M.top1_confidence(probs).mean()), 5),
                "accuracy": round(M.accuracy(M.predicted_class(probs), y), 5)}
    return {"before": block(raw_probs), "after": block(cal_probs)}


def calibrated_scalar_escalation(cal_probs, cal_logits_over_T):
    """Escalation-oriented uncertainty scores recomputed from CALIBRATED probabilities.

    cal_logits_over_T are the temperature-scaled logits (logits / T), so energy uses the
    calibrated energy. Returns {max_prob, entropy, margin, energy} higher => escalate, same
    orientation as uncertainty.SCORE_TRANSFORMS but on calibrated quantities.
    """
    conf = M.top1_confidence(cal_probs)
    ent = M.entropy(cal_probs)
    marg = M.top1_top2_margin(cal_probs)
    ener = M.energy_score(cal_logits_over_T)
    return {"max_prob": 1.0 - conf, "entropy": ent, "margin": -marg, "energy": ener}


def high_conf_wc(cal_maxprob, wc, thresholds):
    """Number / fraction of recoverable (WC) samples the CALIBRATED model is confident about."""
    cal_maxprob = np.asarray(cal_maxprob)
    wc = np.asarray(wc, dtype=bool)
    n_wc = int(wc.sum())
    rows = []
    for thr in thresholds:
        mask = wc & (cal_maxprob >= thr)
        rows.append({"threshold": float(thr), "count": int(mask.sum()),
                     "fraction_of_recoverable": round(float(mask.sum() / n_wc), 5) if n_wc else None,
                     "fraction_of_all": round(float(mask.sum() / len(wc)), 5)})
    return pd.DataFrame(rows)
