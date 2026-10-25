"""Experiment 1: calibration of the zero-shot HAM10000-trained model on in-scope DDI.
Reads outputs/run1/ddi_logits.npz (local only) and prints aggregate results."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import StratifiedShuffleSplit

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from calibration_bias.bootstrap import bootstrap_ci, bootstrap_diff_ci  # noqa: E402
from calibration_bias.calibration import (  # noqa: E402
    apply_temperature, brier, confidence_and_correct, ece, fit_temperature, mce,
    overconfidence_rate, reliability_table)

RUN = root / "outputs" / "run1"
GROUPS = ("I-II", "V-VI")
d = np.load(RUN / "ddi_logits.npz")
logits, labels, fst = d["logits"], d["labels"], d["fst"]
Y = {g: labels[fst == g] for g in GROUPS}
Z = {g: logits[fst == g] for g in GROUPS}

METRICS = {
    "ECE": lambda y, z: ece(*confidence_and_correct(z, y)),
    "MCE": lambda y, z: mce(*confidence_and_correct(z, y)),
    "Brier": lambda y, z: brier(y, softmax(z, axis=1)[:, 1]),
    "Overconfidence rate (>0.70)": lambda y, z: overconfidence_rate(*confidence_and_correct(z, y)),
    "Mean confidence": lambda y, z: confidence_and_correct(z, y)[0].mean(),
    "Accuracy": lambda y, z: confidence_and_correct(z, y)[1].mean(),
}

print("A. Zero-shot calibration by group, estimate [95% bootstrap CI, 1,000 resamples]")
rows = []
for name, fn in METRICS.items():
    row = {"metric": name}
    for g in GROUPS:
        est, lo, hi = bootstrap_ci(fn, Y[g], Z[g], n_boot=1000, seed=0)
        row[g] = f"{est:.3f} [{lo:.3f}, {hi:.3f}]"
    est, lo, hi = bootstrap_diff_ci(fn, Y["I-II"], Z["I-II"], Y["V-VI"], Z["V-VI"],
                                    n_boot=1000, seed=0)
    row["V-VI minus I-II"] = f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\nB. Error counts")
for g in GROUPS:
    conf, correct = confidence_and_correct(Z[g], Y[g])
    wrong = ~correct
    fn_mask = wrong & (Y[g] == 1)
    print(f"FST {g}: errors {int(wrong.sum())} | false negatives {int(fn_mask.sum())}"
          f" | errors with conf>0.70 {int((wrong & (conf > 0.70)).sum())}"
          f" | false negatives with conf>0.70 {int((fn_mask & (conf > 0.70)).sum())}")

print("\nC. Temperature Scaling: 20 repeated stratified 50/50 splits of in-scope DDI")
print("   T fitted on one half (both groups pooled), ECE measured per group on the other half")
strata = np.array([f"{g}_{y}" for g, y in zip(fst, labels)])
splitter = StratifiedShuffleSplit(n_splits=20, test_size=0.5, random_state=0)
recs = []
for fit_idx, ev_idx in splitter.split(np.zeros(len(labels)), strata):
    T = fit_temperature(logits[fit_idx], labels[fit_idx])
    rec = {"T": T}
    for g in GROUPS:
        m = ev_idx[fst[ev_idx] == g]
        scaled = apply_temperature(logits[m], T)
        assert (scaled.argmax(1) == logits[m].argmax(1)).all()  # predictions unchanged
        rec[f"ECE {g} before"] = METRICS["ECE"](labels[m], logits[m])
        rec[f"ECE {g} after"] = METRICS["ECE"](labels[m], scaled)
    rec["gap before (V-VI minus I-II)"] = rec["ECE V-VI before"] - rec["ECE I-II before"]
    rec["gap after (V-VI minus I-II)"] = rec["ECE V-VI after"] - rec["ECE I-II after"]
    recs.append(rec)
print(pd.DataFrame(recs).agg(["mean", "std", "min", "max"]).T.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, g in zip(axes, GROUPS):
    t = reliability_table(*confidence_and_correct(Z[g], Y[g]))
    t = t[t["n"] > 0]
    mid = (t["bin_lo"] + t["bin_hi"]) / 2
    ax.bar(mid, t["accuracy"], width=0.08, edgecolor="k", alpha=0.8)
    for x, acc, n in zip(mid, t["accuracy"], t["n"]):
        ax.text(x, acc + 0.02, f"n={n}", ha="center", fontsize=7)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlim=(0.45, 1), ylim=(0, 1.1), xlabel="Confidence", title=f"FST {g} (n={len(Y[g])})")
axes[0].set_ylabel("Accuracy")
fig.tight_layout()
fig.savefig(RUN / "exp1_reliability.png", dpi=200)
print("\nsaved", RUN / "exp1_reliability.png")