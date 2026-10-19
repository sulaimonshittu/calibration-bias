import numpy as np
import pytest

from calibration_bias.model import SkinLesionClassifier
from calibration_bias.train import build_optimizer, lr_at, summarise_predictions


def test_lr_schedule_shape():
    total, warm, peak = 1000, 100, 3e-4
    assert lr_at(0, total, warm, peak) == pytest.approx(0.1 * peak)
    assert lr_at(warm, total, warm, peak) == pytest.approx(peak)
    assert lr_at(total, total, warm, peak) == pytest.approx(0.01 * peak)
    assert 0.01 * peak < lr_at(550, total, warm, peak) < peak


def test_optimizer_groups_cover_all_params_once():
    m = SkinLesionClassifier(pretrained=False)
    opt = build_optimizer(m, 1e-3, 1e-4)
    grouped = [p for g in opt.param_groups for p in g["params"]]
    assert len(grouped) == len(list(m.parameters()))
    assert len({id(p) for p in grouped}) == len(grouped)


def test_weight_decay_only_on_matrices():
    m = SkinLesionClassifier(pretrained=False)
    decay, no_decay = build_optimizer(m, 1e-3, 1e-4).param_groups
    assert decay["weight_decay"] == 1e-4 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])


def test_summary_for_perfect_and_inverted_classifier():
    labels = np.array([0, 0, 1, 1])
    good = np.array([[3., 0.], [2., 0.], [0., 2.], [0., 3.]], dtype=np.float32)
    s = summarise_predictions(good, labels)
    assert s["acc"] == 1.0 and s["auroc"] == 1.0
    assert summarise_predictions(-good, labels)["acc"] == 0.0