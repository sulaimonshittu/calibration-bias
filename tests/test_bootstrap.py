import numpy as np
from sklearn.metrics import roc_auc_score

from calibration_bias.bootstrap import bootstrap_ci, bootstrap_diff_ci

def test_perfect_separation_gives_degenerate_interval():
    y = np.array([0] * 50 + [1] * 50)
    s = np.array([0.1] * 50 + [0.9] * 50)
    est, lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=200)
    assert est == lo == hi == 1.0


def test_chance_scores_give_interval_containing_half():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 300), rng.random(300)
    _, lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=300)
    assert lo < 0.5 < hi


def test_same_seed_gives_same_interval():
    rng = np.random.default_rng(2)
    y, s = rng.integers(0, 2, 200), rng.random(200)
    assert bootstrap_ci(roc_auc_score, y, s, n_boot=100, seed=5) == \
           bootstrap_ci(roc_auc_score, y, s, n_boot=100, seed=5)


def test_diff_ci_for_identical_groups_includes_zero():
    rng = np.random.default_rng(3)
    s, y = rng.normal(0, 1, 200), np.zeros(200)
    est, lo, hi = bootstrap_diff_ci(lambda y, s: s.mean(), y, s, y, s, n_boot=300)
    assert est == 0.0 and lo < 0 < hi


def test_diff_ci_excludes_zero_for_clear_shift():
    rng = np.random.default_rng(4)
    s_a, s_b, y = rng.normal(0, 1, 200), rng.normal(1, 1, 200), np.zeros(200)
    _, lo, _ = bootstrap_diff_ci(lambda y, s: s.mean(), y, s_a, y, s_b, n_boot=300)
    assert lo > 0