"""Single nonlinear learnability probe (brief F3): a FIXED tiny MLP, no architecture search.

Question it answers: does the frozen 112-resolution representation carry NONLINEAR
information about recoverability (WC) that scalar uncertainty and the Phase-0 linear
probes could not extract? Input is strictly post-112-forward-pass state (z_112, 112
logits, max-prob, entropy, margin, energy); the high-resolution outcome is used only to
form the WC label, never as an input.

Architecture is pinned by the brief:
    input -> Linear(128) -> GELU -> Dropout(0.1) -> Linear(32) -> GELU -> Linear(1)
trained with BCEWithLogitsLoss, AdamW(lr=1e-3, wd=1e-4), batch ~128, early stopping on an
INNER validation split carved from the outer TRAIN folds only -- the outer held-out fold is
never used for early stopping or model selection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 32), dropout=0.1):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _bce():
    return nn.BCEWithLogitsLoss()


def _epoch(model, loader, opt, device, train=True):
    crit = _bce()
    tot, nb = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        if train:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                loss = crit(model(xb), yb)
        tot += float(loss.item()); nb += 1
    return tot / max(nb, 1)


def _inner_split(Xtr, ytr, fraction, seed):
    ss = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=int(seed))
    a, b = next(ss.split(Xtr, ytr))
    return a, b


def _train_with_inner(Xtr, ytr, Xte, mcfg, seed, device, inner_idx=None, inner_fraction=0.15,
                      inner_seed=99):
    torch.manual_seed(int(seed)); np.random.seed(int(seed) + 1000)
    if inner_idx is None:
        ia, ib = _inner_split(Xtr, ytr, inner_fraction, inner_seed + int(seed))
    else:
        ia, ib = inner_idx
    model = TinyMLP(Xtr.shape[1], tuple(mcfg["hidden"]), float(mcfg["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(mcfg["lr"]),
                           weight_decay=float(mcfg["weight_decay"]))
    bs = int(mcfg["batch_size"])
    tr_dl = DataLoader(TensorDataset(torch.as_tensor(Xtr[ia], dtype=torch.float32),
                                     torch.as_tensor(ytr[ia], dtype=torch.float32)),
                       batch_size=bs, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.as_tensor(Xtr[ib], dtype=torch.float32),
                                     torch.as_tensor(ytr[ib], dtype=torch.float32)),
                       batch_size=512, shuffle=False)
    te_t = torch.as_tensor(Xte, dtype=torch.float32).to(device)

    best = float("inf"); best_state = None; best_epoch = 0; bad = 0
    patience = int(mcfg["patience"]); min_delta = float(mcfg.get("min_delta", 1e-4))
    for epoch in range(int(mcfg["max_epochs"])):
        model.train(); _epoch(model, tr_dl, opt, device, train=True)
        model.eval(); vloss = _epoch(model, va_dl, opt, device, train=False)
        if vloss < best - min_delta:
            best = vloss; best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(te_t)).detach().cpu().numpy().astype(np.float64)
    return probs, {"best_epoch": int(best_epoch), "best_inner_val_loss": round(float(best), 6),
                   "epochs_run": int(epoch + 1)}


def oof_mlp_probs(X, y, cv_cfg, mcfg, device="cpu", logger=None, inner_fraction=0.15,
                  inner_seed=99):
    """5-fold OOF pooled probabilities, averaged over the configured seeds per fold.

    Returns (oof_probs[n], records[list of per fold/seed diagnostics]).
    """
    n = len(y)
    oof = np.zeros(n)
    skf = StratifiedKFold(n_splits=int(cv_cfg["n_folds"]), shuffle=True, random_state=int(1337))
    records = []
    seeds = list(cv_cfg["seeds"])
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        # standardize with TRAIN-fold statistics only, then hand to every seed of this fold
        Xtr_raw, Xte_raw = X[tr], X[te]
        mu = Xtr_raw.mean(axis=0, keepdims=True)
        sd = Xtr_raw.std(axis=0, keepdims=True); sd[sd < 1e-8] = 1.0
        Xtr = (Xtr_raw - mu) / sd
        Xte = (Xte_raw - mu) / sd
        ytr = y[tr].astype(np.float32)
        fold_probs = []
        for s in seeds:
            ia, ib = _inner_split(Xtr, ytr, inner_fraction, inner_seed + s)
            probs, diag = _train_with_inner(Xtr, ytr, Xte, mcfg, s, device,
                                            inner_idx=(ia, ib))
            fold_probs.append(probs)
            records.append({"fold": fi, "seed": int(s), **diag})
            if logger:
                logger.info("mlp fold=%d seed=%d epochs=%s best_inner=%.4f",
                            fi, s, diag["epochs_run"], diag["best_inner_val_loss"])
        oof[te] = np.mean(fold_probs, axis=0)
    return oof, records
