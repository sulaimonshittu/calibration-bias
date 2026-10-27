"""Training utilities: seeding, LR schedule, optimiser, prediction and the fit loop."""
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lr_at(step, total_steps, warmup_steps, peak_lr, start_frac=0.1, end_frac=0.01):
    """Linear warm-up from start_frac*peak to peak, then cosine decay to end_frac*peak."""
    if step < warmup_steps:
        return peak_lr * (start_frac + (1 - start_frac) * step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * (end_frac + (1 - end_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_optimizer(model, peak_lr, weight_decay):
    """AdamW with weight decay only on weight matrices/kernels (not biases or norm params)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=peak_lr,
    )


@torch.no_grad()
def predict(model, loader, device):
    """Return (logits, labels) as numpy arrays, in loader order."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        logits.append(model(batch["image"].to(device, non_blocking=True)).float().cpu())
        labels.append(batch["label"])
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def summarise_predictions(logits, labels):
    t_logits, t_labels = torch.from_numpy(logits), torch.from_numpy(labels).long()
    probs = torch.softmax(t_logits, dim=1)[:, 1].numpy()
    return {
        "loss": F.cross_entropy(t_logits, t_labels).item(),
        "acc": float((logits.argmax(1) == labels).mean()),
        "auroc": float(roc_auc_score(labels, probs)),
    }


def fit(model, train_loader, valid_loader, device, out_dir, epochs=15, patience=5,
        peak_lr=3e-4, weight_decay=1e-4, warmup_frac=0.1, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = build_optimizer(model, peak_lr, weight_decay)
    total = epochs * len(train_loader)
    warm = max(1, int(warmup_frac * total))
    step, best, bad, history = 0, float("inf"), 0, []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        run_loss, run_correct, seen = 0.0, 0, 0
        for batch in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warm, peak_lr)
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            step += 1
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val = summarise_predictions(*predict(model, valid_loader, device))
        row = {"epoch": epoch + 1, "lr_end": opt.param_groups[0]["lr"],
               "train_loss": run_loss / seen, "train_acc": run_correct / seen,
               "val_loss": val["loss"], "val_acc": val["acc"], "val_auroc": val["auroc"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        if verbose:
            print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))

        if val["loss"] < best:
            best, bad = val["loss"], 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping: no validation-loss improvement for {patience} epochs.")
                break

    hist = pd.DataFrame(history)
    hist.to_csv(out_dir / "history.csv", index=False)
    return hist