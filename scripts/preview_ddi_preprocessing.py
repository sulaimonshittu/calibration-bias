import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from calibration_bias.preprocess import IMAGE_SIZE, prepare_image  # noqa: E402

ddi = root / "data" / "ddi"
meta = pd.read_csv(next(ddi.rglob("ddi_metadata.csv")))
files = {p.name: p for p in ddi.rglob("*.png")}

random.seed(1)
picks = random.sample(list(meta["DDI_file"]), 12)
sheet = Image.new("RGB", (IMAGE_SIZE * 6, IMAGE_SIZE * 2))
for k, name in enumerate(picks):
    with Image.open(files[name]) as im:
        sheet.paste(prepare_image(im), ((k % 6) * IMAGE_SIZE, (k // 6) * IMAGE_SIZE))

out = root.parent / "outputs"
out.mkdir(exist_ok=True)
sheet.save(out / "ddi_preprocessing_preview.png")
print("saved", out / "ddi_preprocessing_preview.png")