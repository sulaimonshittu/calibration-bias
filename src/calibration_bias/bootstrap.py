"""Percentile bootstrap confidence intervals."""
import numpy as np


def bootstrap_ci(metric_fn, y_true, y_score, n_boot=1000, alpha=0.05, seed=0):
    """Resample (y_true, y_score) pairs with replacement; return (estimate, lower, upper).
    Resamples on which the metric is undefined (for example a single class) are skipped."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            values.append(metric_fn(y_true[idx], y_score[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric_fn(y_true, y_score)), float(lo), float(hi)