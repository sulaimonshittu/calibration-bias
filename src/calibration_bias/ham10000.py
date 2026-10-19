"""HAM10000 loading, binary label mapping and lesion-level splitting."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Binary mapping from Chapter 3, Table 3.1.
MALIGNANT_DX = {"mel", "bcc", "akiec"}
BENIGN_DX = {"nv", "bkl", "df", "vasc"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dx_to_label(dx: str) -> int:
    if dx in MALIGNANT_DX:
        return 1
    if dx in BENIGN_DX:
        return 0
    raise ValueError(f"Unknown HAM10000 diagnosis code: {dx!r}")


def load_metadata(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].map(dx_to_label)
    return df


def build_image_index(root) -> tuple[dict[str, Path], int]:
    """Map image_id -> file path. Returns (index, number of duplicate files skipped)."""
    index, duplicates = {}, 0
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in index:
                duplicates += 1
            else:
                index[path.stem] = path
    return index, duplicates


def split_by_lesion(df, seed=42, val_frac=0.15, test_frac=0.15):
    """Split at lesion level, stratified by diagnosis, so no lesion spans two splits."""
    lesions = df.groupby("lesion_id")["dx"].agg(["first", "nunique"]).reset_index()
    if (lesions["nunique"] != 1).any():
        raise ValueError("Some lesion_ids have more than one dx; cannot stratify by lesion.")

    holdout = val_frac + test_frac
    train_l, rest = train_test_split(
        lesions, test_size=holdout, stratify=lesions["first"], random_state=seed
    )
    val_l, test_l = train_test_split(
        rest, test_size=test_frac / holdout, stratify=rest["first"], random_state=seed
    )
    return {
        name: df[df["lesion_id"].isin(part["lesion_id"])].reset_index(drop=True)
        for name, part in (("train", train_l), ("valid", val_l), ("test", test_l))
    }


def check_splits(splits, df):
    """Raise if any lesion or image appears in two splits, or images are lost."""
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(splits[a]["lesion_id"]) & set(splits[b]["lesion_id"]):
                raise AssertionError(f"Lesion overlap between {a} and {b}")
            if set(splits[a]["image_id"]) & set(splits[b]["image_id"]):
                raise AssertionError(f"Image overlap between {a} and {b}")
    if sum(len(s) for s in splits.values()) != len(df):
        raise AssertionError("Images lost or duplicated by the split")


def summarise(splits) -> pd.DataFrame:
    rows = []
    for name, part in splits.items():
        malignant = int((part["label"] == 1).sum())
        rows.append({
            "split": name,
            "images": len(part),
            "lesions": part["lesion_id"].nunique(),
            "benign": len(part) - malignant,
            "malignant": malignant,
            "malignant_pct": round(100 * malignant / len(part), 1),
        })
    return pd.DataFrame(rows)