"""Percentile bootstrap confidence intervals."""
import numpy as np


def _interval(values, alpha):
    lo, hi = np.nanpercentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci(metric_fn, y_true, y_score, n_boot=1000, alpha=0.05, seed=0):
    """Resample rows with replacement; return (estimate, lower, upper).
    Resamples where the metric is undefined (ValueError) are skipped; NaNs are ignored."""
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
    lo, hi = _interval(values, alpha)
    return float(metric_fn(y_true, y_score)), lo, hi


def bootstrap_diff_ci(metric_fn, y_a, s_a, y_b, s_b, n_boot=1000, alpha=0.05, seed=0):
    """CI for metric(group b) - metric(group a); each group is resampled independently."""
    y_a, s_a, y_b, s_b = (np.asarray(x) for x in (y_a, s_a, y_b, s_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(y_a), len(y_a))
        ib = rng.integers(0, len(y_b), len(y_b))
        try:
            diffs.append(metric_fn(y_b[ib], s_b[ib]) - metric_fn(y_a[ia], s_a[ia]))
        except ValueError:
            continue
    lo, hi = _interval(diffs, alpha)
    return float(metric_fn(y_b, s_b) - metric_fn(y_a, s_a)), lo, hi