"""PyTorch dataset and transforms for lesion images."""
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocess import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, PAD_FILL, prepare_image

_to_tensor = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)

# Training-time augmentation (Chapter 3, Section 3.6), applied after padding/resizing.
_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10, fill=PAD_FILL),
    transforms.ColorJitter(brightness=0.1, contrast=(0.9, 1.0), hue=0.05),
])


class LesionDataset(Dataset):
    """Expects a DataFrame with 'path' and 'label' columns; other columns are ignored."""

    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:
            img = prepare_image(img)
        if self.train:
            img = _augment(img)
        return {"image": _to_tensor(img), "label": self.labels[i], "index": i}

def cache_images(paths) -> np.ndarray:
    """Preprocess every image once and keep it in memory as uint8, shape (N, 224, 224, 3)."""
    out = np.empty((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            out[i] = np.asarray(prepare_image(img))
    return out


class ArrayDataset(Dataset):
    """In-memory dataset of preprocessed uint8 images with integer labels."""

    def __init__(self, images: np.ndarray, labels, train: bool = False):
        self.images, self.labels, self.train = images, np.asarray(labels), train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.train:
            img = _augment(img)
        return {"image": _to_tensor(img), "label": int(self.labels[i]), "index": i}