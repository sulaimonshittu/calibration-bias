from pathlib import Path
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parents[1] / "data" / "ddi"
meta = pd.read_csv(next(root.rglob("ddi_metadata.csv")))
files = {p.name: p for p in root.rglob("*.png")}

rows = []
for _, r in meta.iterrows():
    with Image.open(files[r["DDI_file"]]) as im:
        alpha_min = im.getchannel("A").getextrema()[0] if im.mode == "RGBA" else None
        rows.append({"skin_tone": r["skin_tone"], "malignant": r["malignant"],
                     "w": im.width, "h": im.height, "rgba": im.mode == "RGBA",
                     "alpha_min": alpha_min})
df = pd.DataFrame(rows)
df["area_kpx"] = df["w"] * df["h"] / 1000
df["min_side"] = df[["w", "h"]].min(axis=1)

print("RGBA share per FST group:\n", df.groupby("skin_tone")["rgba"].mean().round(3).to_string())
rgba = df[df["rgba"]]
print("\nRGBA images with any transparency:", int((rgba["alpha_min"] < 255).sum()), "of", len(rgba))
print("\nShare with shorter side < 224 px, per group:\n",
      df.assign(small=df["min_side"] < 224).groupby("skin_tone")["small"].mean().round(3).to_string())
print("\nImages with shorter side < 100 px:", int((df["min_side"] < 100).sum()))
print("\nMedian area (kpx) by group and class:\n",
      df.groupby(["skin_tone", "malignant"])["area_kpx"].median().round(0).to_string())