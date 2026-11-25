"""PyTorch Dataset over the preprocessed PNG cache (decisions D8, D9).

    train=True : random augmentation (train split only) -> normalize
    train=False: no augmentation (val/test)            -> normalize

Images are stored as 1-channel 224x224 PNGs (src/03_preprocess.py). The single channel is
replicated to 3 because ImageNet-pretrained backbones expect RGB input, then standardized
with ImageNet statistics so inputs match the distribution the pretrained filters were trained on.
"""
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import InterpolationMode, v2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLASSES = ("NORMAL", "PNEUMONIA")  # label index = position in this tuple

# D8: only label-preserving AND realistic transformations. No flips, no blur, no erasing.
TRAIN_AUGMENT = v2.Compose([
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR, fill=0),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
])


def normalize(img_uint8: torch.Tensor) -> torch.Tensor:
    """1xHxW uint8 -> 3xHxW float32 standardized with ImageNet statistics (D9)."""
    x = img_uint8.float().div(255.0).expand(3, -1, -1)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize, for visualization: returns HxW float in [0, 1]."""
    return (x * IMAGENET_STD + IMAGENET_MEAN)[0].clamp(0, 1)


class ChestXrayDataset(Dataset):
    def __init__(self, splits: pd.DataFrame, split: str, image_root: Path, train: bool = False):
        df = splits[splits["split"] == split]
        if df.empty:
            raise ValueError(f"no images for split '{split}'")
        self.paths = [image_root / Path(p).with_suffix(".png") for p in df["relpath"]]
        self.labels = torch.tensor([CLASSES.index(lab) for lab in df["label"]])
        self.augment = TRAIN_AUGMENT if train else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = read_image(str(self.paths[i]), mode=ImageReadMode.GRAY)  # 1xHxW uint8
        if self.augment is not None:
            img = self.augment(img)
        return normalize(img), self.labels[i]
