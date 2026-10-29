"""Calibration metrics and Temperature Scaling (Chapter 3, Sections 3.10-3.11)."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def confidence_and_correct(logits, labels):
    """Top-label confidence, and whether the predicted class equals the label."""
    probs = softmax(np.asarray(logits, dtype=float), axis=1)
    return probs.max(axis=1), probs.argmax(axis=1) == np.asarray(labels)


def _bin_index(conf, n_bins):
    # Bins are (lo, hi], as in Guo et al. (2017).
    return np.clip(np.ceil(np.asarray(conf) * n_bins).astype(int) - 1, 0, n_bins - 1)


def reliability_table(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = _bin_index(conf, n_bins)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({"bin_lo": b / n_bins, "bin_hi": (b + 1) / n_bins, "n": int(m.sum()),
                     "mean_conf": conf[m].mean() if m.any() else np.nan,
                     "accuracy": correct[m].mean() if m.any() else np.nan})
    return pd.DataFrame(rows)


def _binned_sums(conf, correct, n_bins):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    idx = _bin_index(conf, n_bins)
    return (np.bincount(idx, minlength=n_bins),
            np.bincount(idx, weights=conf, minlength=n_bins),
            np.bincount(idx, weights=correct, minlength=n_bins))


def ece(conf, correct, n_bins=10):
    n, sum_conf, sum_acc = _binned_sums(conf, correct, n_bins)
    return float(np.abs(sum_acc - sum_conf).sum() / n.sum())


def mce(conf, correct, n_bins=10):
    n, sum_conf, sum_acc = _binned_sums(conf, correct, n_bins)
    filled = n > 0
    return float((np.abs(sum_acc[filled] - sum_conf[filled]) / n[filled]).max())


def brier(labels, p_positive):
    return float(np.mean((np.asarray(p_positive, float) - np.asarray(labels, float)) ** 2))


def overconfidence_rate(conf, correct, threshold=0.70):
    """Share of misclassified predictions made with confidence above threshold (NaN if no errors)."""
    wrong = ~np.asarray(correct, bool)
    if not wrong.any():
        return float("nan")
    return float((np.asarray(conf)[wrong] > threshold).mean())


def fit_temperature(logits, labels):
    """Single scalar T > 0 minimising negative log-likelihood (Guo et al., 2017)."""
    logits, labels = np.asarray(logits, float), np.asarray(labels, int)
    rows = np.arange(len(labels))

    def nll(log_t):
        return -log_softmax(logits / np.exp(log_t), axis=1)[rows, labels].mean()

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def apply_temperature(logits, T):
    return np.asarray(logits, float) / T