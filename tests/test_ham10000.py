import pandas as pd
import pytest

from calibration_bias.ham10000 import check_splits, dx_to_label, split_by_lesion

DX = ["nv", "nv", "nv", "bkl", "mel", "bcc", "akiec", "df", "vasc"]


def make_df(n_lesions=450):
    rows = []
    for i in range(n_lesions):
        k = i % len(DX)
        for j in range(1 + k % 3):
            rows.append({"lesion_id": f"L{i}", "image_id": f"L{i}_{j}", "dx": DX[k]})
    df = pd.DataFrame(rows)
    df["label"] = df["dx"].map(dx_to_label)
    return df


def test_label_mapping_matches_table_3_1():
    got = {dx: dx_to_label(dx) for dx in ["nv", "bkl", "df", "vasc", "mel", "bcc", "akiec"]}
    assert got == {"nv": 0, "bkl": 0, "df": 0, "vasc": 0, "mel": 1, "bcc": 1, "akiec": 1}


def test_unknown_dx_raises():
    with pytest.raises(ValueError):
        dx_to_label("xyz")


def test_no_lesion_or_image_overlap_between_splits():
    df = make_df()
    check_splits(split_by_lesion(df, seed=1), df)


def test_split_is_deterministic():
    df = make_df()
    a, b = split_by_lesion(df, seed=7), split_by_lesion(df, seed=7)
    for name in a:
        assert list(a[name]["image_id"]) == list(b[name]["image_id"])


def test_fractions_roughly_respected():
    df = make_df()
    splits = split_by_lesion(df, seed=3)
    total = df["lesion_id"].nunique()
    assert splits["train"]["lesion_id"].nunique() / total == pytest.approx(0.70, abs=0.02)
    assert splits["valid"]["lesion_id"].nunique() / total == pytest.approx(0.15, abs=0.02)


def test_class_mix_preserved_across_splits():
    df = make_df()
    overall = df["label"].mean()
    for part in split_by_lesion(df, seed=3).values():
        assert part["label"].mean() == pytest.approx(overall, abs=0.05)


def test_lesion_with_two_diagnoses_raises():
    df = make_df()
    extra = pd.DataFrame([{"lesion_id": "L0", "image_id": "L0_extra", "dx": "mel", "label": 1}])
    with pytest.raises(ValueError):
        split_by_lesion(pd.concat([df, extra], ignore_index=True))