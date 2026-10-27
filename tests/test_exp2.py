import numpy as np
import pandas as pd
import pytest

from calibration_bias.exp2 import IN_SCOPE, make_folds


def make_df():
    rows = []
    for g, n_b, n_m in [("I-II", 40, 15), ("III-IV", 30, 10), ("V-VI", 40, 15)]:
        rows += [{"fst": g, "label": 0}] * n_b + [{"fst": g, "label": 1}] * n_m
    return pd.DataFrame(rows)


def test_parts_are_disjoint_and_cover_everything():
    df = make_df()
    for fd in make_folds(df, seed=1):
        a, b, c = set(fd["train"]), set(fd["val"]), set(fd["test"])
        assert not (a & b or a & c or b & c)
        assert a | b | c == set(range(len(df)))


def test_every_in_scope_image_is_tested_exactly_once():
    df = make_df()
    counts = np.zeros(len(df), int)
    for fd in make_folds(df, seed=2):
        counts[fd["test"]] += 1
    in_scope = df["fst"].isin(IN_SCOPE).to_numpy()
    assert (counts[in_scope] == 1).all() and (counts[~in_scope] == 0).all()


def test_out_of_scope_images_only_in_training():
    df = make_df()
    out = set(np.flatnonzero(~df["fst"].isin(IN_SCOPE)))
    for fd in make_folds(df, seed=3):
        assert not out & (set(fd["val"]) | set(fd["test"]))
        assert out <= set(fd["train"])


def test_class_mix_is_stratified_within_each_group():
    df = make_df()
    for fd in make_folds(df, seed=4):
        test = df.iloc[fd["test"]]
        for g in IN_SCOPE:
            assert test[test["fst"] == g]["label"].mean() == pytest.approx(15 / 55, abs=0.06)


def test_seed_controls_the_split():
    df = make_df()
    a, b, c = (make_folds(df, seed=s) for s in (5, 5, 6))
    assert all((x["test"] == y["test"]).all() for x, y in zip(a, b))
    assert any(not np.array_equal(x["test"], z["test"]) for x, z in zip(a, c))