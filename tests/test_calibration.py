import numpy as np
import pytest

from calibration_bias.calibration import (
    apply_temperature, brier, confidence_and_correct, ece, fit_temperature, mce,
    overconfidence_rate, reliability_table)


def test_ece_zero_when_confidence_matches_accuracy():
    conf = np.full(1000, 0.8)
    correct = np.array([True] * 800 + [False] * 200)
    assert ece(conf, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_single_bin_known_value():
    conf = np.full(100, 0.9)
    correct = np.array([True] * 50 + [False] * 50)
    assert ece(conf, correct) == pytest.approx(0.4)
    assert mce(conf, correct) == pytest.approx(0.4)


def test_ece_and_mce_with_two_bins():
    conf = np.array([0.6] * 10 + [0.95] * 10)
    correct = np.array([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5)
    assert ece(conf, correct) == pytest.approx(0.275)   # 0.5*0.10 + 0.5*0.45
    assert mce(conf, correct) == pytest.approx(0.45)


def test_brier_known_values():
    y = np.array([0, 1, 1, 0])
    assert brier(y, y) == 0.0
    assert brier(y, 1 - y) == 1.0
    assert brier(y, np.full(4, 0.5)) == pytest.approx(0.25)


def test_overconfidence_rate():
    conf = np.array([0.9, 0.8, 0.6, 0.99, 0.95])
    correct = np.array([False, False, False, True, True])
    assert overconfidence_rate(conf, correct) == pytest.approx(2 / 3)
    assert np.isnan(overconfidence_rate(conf, np.ones(5, dtype=bool)))


def test_reliability_table_counts_sum_to_n():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 200)
    assert reliability_table(conf, rng.random(200) < conf)["n"].sum() == 200


def _overconfident_data(n=5000, factor=3.0, seed=0):
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 1.5, n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    return np.stack([np.zeros(n), score * factor], axis=1), labels


def test_temperature_recovers_known_scale():
    logits, labels = _overconfident_data(factor=3.0)
    assert fit_temperature(logits, labels) == pytest.approx(3.0, rel=0.15)


def test_temperature_scaling_keeps_predictions_and_lowers_ece():
    logits, labels = _overconfident_data()
    scaled = apply_temperature(logits, fit_temperature(logits, labels))
    assert (scaled.argmax(1) == logits.argmax(1)).all()
    assert ece(*confidence_and_correct(scaled, labels)) < ece(*confidence_and_correct(logits, labels))