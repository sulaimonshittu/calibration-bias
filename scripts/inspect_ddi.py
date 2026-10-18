import sys, platform
from pathlib import Path
import pandas as pd

print(sys.version)
print(platform.platform())

root = Path(__file__).resolve().parents[1] / "data" / "ddi"
print("\nTop level of data/ddi:")
for p in sorted(root.iterdir()):
    print("  ", p.name)

images = [p for p in root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
print(f"\nImage files found: {len(images)}")

csvs = list(root.rglob("*.csv"))
print("CSV files:", [c.name for c in csvs])

df = pd.read_csv(csvs[0])
print("\nShape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3).to_string())

for col in df.columns:
    if df[col].nunique() <= 12:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts(dropna=False).to_string())

if {"skin_tone", "malignant"} <= set(df.columns):
    print("\nSkin tone x malignant:")
    print(pd.crosstab(df["skin_tone"], df["malignant"]))