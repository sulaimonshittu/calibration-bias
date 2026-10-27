import numpy as np
import torch
from PIL import Image

from calibration_bias.datasets import ArrayDataset, cache_images


def test_cache_images_shape_and_dtype(tmp_path):
    for i, size in enumerate([(30, 80), (120, 50)]):
        Image.new("RGB", size, (i * 50, 10, 10)).save(tmp_path / f"{i}.png")
    arr = cache_images([tmp_path / "0.png", tmp_path / "1.png"])
    assert arr.shape == (2, 224, 224, 3) and arr.dtype == np.uint8


def test_array_dataset_item_shape_and_label():
    imgs = np.random.default_rng(0).integers(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    item = ArrayDataset(imgs, np.array([0, 1, 0]))[1]
    assert tuple(item["image"].shape) == (3, 224, 224) and item["label"] == 1


def test_training_mode_applies_random_augmentation():
    imgs = np.random.default_rng(1).integers(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    ds = ArrayDataset(imgs, np.array([0]), train=True)
    assert not torch.equal(ds[0]["image"], ds[0]["image"])