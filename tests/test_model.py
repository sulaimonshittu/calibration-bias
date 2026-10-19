import torch

from calibration_bias.model import SkinLesionClassifier


def test_forward_shape_is_batch_by_two_logits():
    m = SkinLesionClassifier(pretrained=False).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 2)


def test_backbone_outputs_2048_features():
    m = SkinLesionClassifier(pretrained=False).eval()
    out = m.backbone(pixel_values=torch.randn(1, 3, 224, 224)).pooler_output
    assert out.shape == (1, 2048, 1, 1)


def test_head_parameter_count():
    m = SkinLesionClassifier(pretrained=False)
    assert sum(p.numel() for p in m.head.parameters()) == 557_698


def test_eval_mode_is_deterministic():
    m = SkinLesionClassifier(pretrained=False, dropout=0.5).eval()
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(m(x), m(x))


def test_train_mode_dropout_is_stochastic():
    m = SkinLesionClassifier(pretrained=False, dropout=0.5).train()
    x = torch.randn(4, 3, 224, 224)
    assert not torch.equal(m(x), m(x))