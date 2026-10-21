"""Run the fine-tuned model on in-scope DDI images (CPU, local) and print decision-gate metrics."""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from calibration_bias.bootstrap import bootstrap_ci  # noqa: E402
from calibration_bias.datasets import LesionDataset  # noqa: E402
from calibration_bias.ddi import load_ddi  # noqa: E402
from calibration_bias.model import SkinLesionClassifier  # noqa: E402
from calibration_bias.train import predict  # noqa: E402

RUN = root / "outputs" / "run1"
GROUPS = ("I-II", "V-VI")


def main():
    df = load_ddi(root / "data" / "ddi")
    df = df[df["fst"].isin(GROUPS)].reset_index(drop=True)
    print("In-scope images:", len(df))
    print(df.groupby("fst")["label"].agg(n="size", malignant="sum").to_string())

    model = SkinLesionClassifier(dropout=0.5, pretrained=False)
    model.load_state_dict(torch.load(RUN / "best.pt", map_location="cpu"))
    loader = DataLoader(LesionDataset(df, train=False), batch_size=16, shuffle=False, num_workers=0)
    logits, labels = predict(model, loader, torch.device("cpu"))
    assert (labels == df["label"].to_numpy()).all()

    np.savez(RUN / "ddi_logits.npz", logits=logits, labels=labels,
             ddi_id=df["DDI_ID"].to_numpy(), fst=df["fst"].to_numpy().astype(str),
             disease=df["disease"].to_numpy().astype(str))

    p_mal = torch.softmax(torch.from_numpy(logits), dim=1)[:, 1].numpy()
    print("\nReference: HAM10000 test AUROC 0.9136")
    for g in GROUPS:
        m = (df["fst"] == g).to_numpy()
        y, p = labels[m], p_mal[m]
        auc, lo, hi = bootstrap_ci(roc_auc_score, y, p, n_boot=1000, seed=0)
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        print(f"\nFST {g}: n={len(y)}, malignant={int(y.sum())}")
        print(f"  AUROC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  accuracy {(pred == y).mean():.3f} (always-benign baseline {1 - y.mean():.3f})"
              f" | sensitivity {tp / (tp + fn):.3f} | specificity {tn / (tn + fp):.3f}")
        print(f"  p(malignant): mean {p.mean():.3f} | median {np.median(p):.3f}"
              f" | 90th pct {np.percentile(p, 90):.3f}"
              f" | predicted malignant at 0.5: {100 * pred.mean():.1f}%")


if __name__ == "__main__":
    main()