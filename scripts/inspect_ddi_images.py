from pathlib import Path
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parents[1] / "data" / "ddi"
meta = pd.read_csv(next(root.rglob("ddi_metadata.csv")))
files = {p.name: p for p in root.rglob("*.png")}

rows = []
for _, r in meta.iterrows():
    with Image.open(files[r["DDI_file"]]) as im:
        w, h = im.size
        rows.append({"skin_tone": r["skin_tone"], "malignant": r["malignant"],
                     "w": w, "h": h, "aspect": w / h, "mode": im.mode})
df = pd.DataFrame(rows)

print(df[["w", "h", "aspect"]].describe().round(2).to_string())
print("\nModes:", df["mode"].value_counts().to_dict())
print("\nMedian size per FST group:")
print(df.groupby("skin_tone")[["w", "h", "aspect"]].median().round(2).to_string())
print("\nShare wider than tall by >20%:", round((df["aspect"] > 1.2).mean(), 3))
print("Share taller than wide by >20%:", round((df["aspect"] < 1 / 1.2).mean(), 3))