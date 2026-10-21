from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

run = Path(__file__).resolve().parents[1] / "outputs" / "run1"
h = pd.read_csv(run / "history.csv")
best = int(h.loc[h["val_loss"].idxmin(), "epoch"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h["epoch"], h["train_loss"], label="train")
ax[0].plot(h["epoch"], h["val_loss"], label="validation")
ax[0].axvline(best, ls="--", c="grey", label=f"best epoch ({best})")
ax[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
ax[0].legend()

ax[1].plot(h["epoch"], h["val_auroc"], label="validation AUROC")
ax[1].plot(h["epoch"], h["val_acc"], label="validation accuracy")
ax[1].axvline(best, ls="--", c="grey")
ax[1].set(xlabel="Epoch", title="Validation metrics")
ax[1].legend()

fig.tight_layout()
fig.savefig(run / "training_curves.png", dpi=200)
print("saved", run / "training_curves.png")