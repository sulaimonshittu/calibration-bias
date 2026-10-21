import numpy as np
from sklearn.metrics import roc_auc_score

from calibration_bias.bootstrap import bootstrap_ci


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