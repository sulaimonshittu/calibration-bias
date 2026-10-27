"""Experiment 2: adapt the HAM10000-trained model on DDI, then measure calibration by FST group."""
import shutil
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .calibration import apply_temperature, fit_temperature
from .datasets import ArrayDataset
from .model import SkinLesionClassifier
from .train import fit, predict, set_seed

IN_SCOPE = ("I-II", "V-VI")


def make_folds(df, n_splits=5, seed=0):
    """Stratified (FST group x class) K-fold over in-scope images with a rotating validation fold.

    Per fold: test = one part, validation = the next part, training = the remaining parts
    (about 60/20/20 of in-scope images). Out-of-scope images (FST III-IV) are always in training.
    Every in-scope image is tested exactly once per call."""
    in_scope = df["fst"].isin(IN_SCOPE).to_numpy()
    scope_idx = np.flatnonzero(in_scope)
    always_train = np.flatnonzero(~in_scope)
    strata = (df["fst"].astype(str) + "_" + df["label"].astype(str)).to_numpy()[scope_idx]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    parts = [scope_idx[test] for _, test in skf.split(scope_idx, strata)]
    folds = []
    for k in range(n_splits):
        val_k = (k + 1) % n_splits
        rest = [parts[j] for j in range(n_splits) if j not in (k, val_k)]
        folds.append({"fold": k, "test": parts[k], "val": parts[val_k],
                      "train": np.concatenate([always_train, *rest])})
    return folds


def finetune_fold(images, labels, fold, ham_ckpt, work_dir, device, seed, epochs=30, patience=5,
                  peak_lr=1e-4, weight_decay=1e-4, dropout=0.5, batch_size=32):
    set_seed(seed)
    model = SkinLesionClassifier(dropout=dropout, pretrained=False)
    model.load_state_dict(torch.load(ham_ckpt, map_location="cpu"))

    def loader(idx, train):
        ds = ArrayDataset(images[idx], labels[idx], train=train)
        return DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=train)

    hist = fit(model, loader(fold["train"], True), loader(fold["val"], False), device, work_dir,
               epochs=epochs, patience=patience, peak_lr=peak_lr,
               weight_decay=weight_decay, verbose=False)
    best = hist.loc[hist["val_loss"].idxmin()]
    model.load_state_dict(torch.load(Path(work_dir) / "best.pt", map_location=device))
    val_logits, val_labels = predict(model, loader(fold["val"], False), device)
    test_logits, _ = predict(model, loader(fold["test"], False), device)
    T = fit_temperature(val_logits, val_labels)
    shutil.rmtree(work_dir, ignore_errors=True)
    return {"test_logits": test_logits,
            "test_logits_ts": apply_temperature(test_logits, T).astype(np.float32),
            "T": T, "best_epoch": int(best["epoch"]), "val_auroc": float(best["val_auroc"])}


def run_repetition(df, images, labels, ham_ckpt, work_dir, device, rep_seed, **cfg):
    parts = {k: [] for k in ("index", "logits", "logits_ts", "fold")}
    T, best_epoch, val_auroc = [], [], []
    for fd in make_folds(df, seed=rep_seed):
        r = finetune_fold(images, labels, fd, ham_ckpt, work_dir, device,
                          seed=rep_seed * 10 + fd["fold"], **cfg)
        parts["index"].append(fd["test"])
        parts["logits"].append(r["test_logits"])
        parts["logits_ts"].append(r["test_logits_ts"])
        parts["fold"].append(np.full(len(fd["test"]), fd["fold"]))
        T.append(r["T"]); best_epoch.append(r["best_epoch"]); val_auroc.append(r["val_auroc"])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out.update(T=np.array(T), best_epoch=np.array(best_epoch), val_auroc=np.array(val_auroc))
    return out