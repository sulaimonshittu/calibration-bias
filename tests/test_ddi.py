import pandas as pd
from PIL import Image

from calibration_bias.ddi import load_ddi


def test_load_ddi_maps_groups_and_labels(tmp_path):
    rows = [{"Unnamed: 0": i, "DDI_ID": i + 1, "DDI_file": f"{i + 1:06d}.png",
             "skin_tone": tone, "malignant": mal, "disease": "x"}
            for i, (tone, mal) in enumerate([(12, True), (34, False), (56, False)])]
    pd.DataFrame(rows).to_csv(tmp_path / "ddi_metadata.csv", index=False)
    for r in rows:
        Image.new("RGB", (8, 8)).save(tmp_path / r["DDI_file"])
    df = load_ddi(tmp_path)
    assert list(df["fst"]) == ["I-II", "III-IV", "V-VI"]
    assert list(df["label"]) == [1, 0, 0]
    assert "Unnamed: 0" not in df.columns