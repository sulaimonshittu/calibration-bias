"""Loading the DDI evaluation set. DDI stays local and is never committed."""
from pathlib import Path

import pandas as pd

FST_GROUPS = {12: "I-II", 34: "III-IV", 56: "V-VI"}


def load_ddi(root) -> pd.DataFrame:
    root = Path(root)
    meta = pd.read_csv(next(root.rglob("ddi_metadata.csv")))
    meta = meta.drop(columns=[c for c in meta.columns if c.startswith("Unnamed")])
    files = {p.name: p for p in root.rglob("*.png")}
    meta["path"] = meta["DDI_file"].map(lambda name: str(files[name]))
    meta["label"] = meta["malignant"].astype(int)
    meta["fst"] = meta["skin_tone"].map(FST_GROUPS)
    if meta["fst"].isna().any():
        raise ValueError("Unexpected skin_tone codes in DDI metadata")
    return meta