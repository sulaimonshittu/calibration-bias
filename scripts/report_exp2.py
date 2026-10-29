"""Experiment 2 analysis. Reads outputs/exp2/rep*.npz (local only); prints aggregate results."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from calibration_bias.bootstrap import bootstrap_ci, bootstrap_diff_ci  # noqa: E402
from calibration_bias.calibration import (  # noqa: E402
    brier, confidence_and_correct, ece, mce, overconfidence_rate, reliability_table)

DIR = root / "outputs" / "exp2"
GROUPS = ("I-II", "V-VI")
N_BOOT = 1000
EPOCH_CAP = 30  # must match epochs in the Kaggle config

reps = [np.load(f) for f in sorted(DIR.glob("rep*.npz"))]
R = len(reps)
print(f"{R} repetitions loaded")

# Align every repetition to one image order (sorted by DDI_ID).
orders = [np.argsort(r["ddi_id"]) for r in reps]
ids0 = reps[0]["ddi_id"][orders[0]]
labels = reps[0]["labels"][orders[0]]
fst = reps[0]["fst"][orders[0]]
for r, o in zip(reps, orders):
    assert (r["ddi_id"][o] == ids0).all() and (r["labels"][o] == labels).all()
raw = np.stack([r["logits"][o] for r, o in zip(reps, orders)], axis=1)        # (N, R, 2)
scaled = np.stack([r["logits_ts"][o] for r, o in zip(reps, orders)], axis=1)  # (N, R, 2)
A, B = fst == "I-II", fst == "V-VI"

BASE = {
    "AUROC": lambda y, z: roc_auc_score(y, softmax(z, axis=1)[:, 1]),
    "ECE": lambda y, z: ece(*confidence_and_correct(z, y)),
    "MCE": lambda y, z: mce(*confidence_and_correct(z, y)),
    "Brier": lambda y, z: brier(y, softmax(z, axis=1)[:, 1]),
    "Overconfidence rate (>0.70)": lambda y, z: overconfidence_rate(*confidence_and_correct(z, y)),
    "Mean confidence": lambda y, z: confidence_and_correct(z, y)[0].mean(),
    "Accuracy": lambda y, z: confidence_and_correct(z, y)[1].mean(),
}


def over_reps(fn):
    """Apply a per-repetition metric to z of shape (n, R, 2), then average over repetitions."""
    return lambda y, z: float(np.nanmean([fn(y, z[:, r, :]) for r in range(z.shape[1])]))


def table(z, fns, label_fmt="{}"):
    rows = []
    for name, fn in fns.items():
        row = {"metric": label_fmt.format(name)}
        for g, m in (("I-II", A), ("V-VI", B)):
            est, lo, hi = bootstrap_ci(fn, labels[m], z[m], n_boot=N_BOOT, seed=0)
            row[g] = f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]" if "after" in label_fmt else f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
        est, lo, hi = bootstrap_diff_ci(fn, labels[A], z[A], labels[B], z[B], n_boot=N_BOOT, seed=0)
        row["V-VI minus I-II"] = f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)


print("\n1. Adapted model before Temperature Scaling: estimate [95% CI]")
print("   (image-level bootstrap, 1,000 resamples, metric averaged over repetitions)")
print(table(raw, {k: over_reps(v) for k, v in BASE.items()}))

print("\n2. After Temperature Scaling (T fitted per fold on its validation part)")
print(table(scaled, {k: over_reps(v) for k, v in BASE.items() if k != "AUROC"}))

print("\n3. Change due to Temperature Scaling (after minus before)")
both = np.stack([raw, scaled], axis=1)  # (N, 2, R, 2)


def ts_change(name):
    fn = over_reps(BASE[name])
    return lambda y, z: fn(y, z[:, 1]) - fn(y, z[:, 0])


print(table(both, {n: ts_change(n) for n in ("ECE", "Brier", "Overconfidence rate (>0.70)")},
            label_fmt="{}: after minus before"))

T = np.concatenate([r["T"] for r in reps])
best = np.concatenate([r["best_epoch"] for r in reps])
vauc = np.concatenate([r["val_auroc"] for r in reps])
print(f"\n4. Fine-tuning runs: {len(T)}")
print(f"   T: mean {T.mean():.2f}, SD {T.std():.2f}, range {T.min():.2f} to {T.max():.2f}")
print(f"   best epoch: median {np.median(best):.0f}; at the {EPOCH_CAP}-epoch cap in "
      f"{int((best >= EPOCH_CAP).sum())} of {len(best)} runs")
print(f"   validation AUROC: mean {vauc.mean():.3f}, SD {vauc.std():.3f}")

print("\n5. Missed malignancies, adapted model before TS (mean over repetitions)")
for g, m in (("I-II", A), ("V-VI", B)):
    n_fn, n_fn_hi = [], []
    for r in range(R):
        conf, correct = confidence_and_correct(raw[m][:, r, :], labels[m])
        fn_mask = ~correct & (labels[m] == 1)
        n_fn.append(fn_mask.sum())
        n_fn_hi.append((fn_mask & (conf > 0.70)).sum())
    print(f"   FST {g}: malignant {int(labels[m].sum())} | false negatives {np.mean(n_fn):.1f}"
          f" | of which confidence > 0.70: {np.mean(n_fn_hi):.1f}")

fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
for i, (tag, z) in enumerate((("before TS", raw), ("after TS", scaled))):
    for j, (g, m) in enumerate((("I-II", A), ("V-VI", B))):
        conf, correct = confidence_and_correct(z[m].reshape(-1, 2), np.repeat(labels[m], R))
        t = reliability_table(conf, correct)
        t = t[t["n"] > 0]
        mid = (t["bin_lo"] + t["bin_hi"]) / 2
        ax = axes[i, j]
        ax.bar(mid, t["accuracy"], width=0.08, edgecolor="k", alpha=0.8)
        ax.plot([0, 1], [0, 1], "k--")
        ax.set(xlim=(0.45, 1), ylim=(0, 1.05), title=f"FST {g}, {tag}")
for ax in axes[1]:
    ax.set_xlabel("Confidence")
for ax in axes[:, 0]:
    ax.set_ylabel("Accuracy")
fig.tight_layout()
fig.savefig(DIR / "exp2_reliability.png", dpi=200)
print("\nsaved", DIR / "exp2_reliability.png")