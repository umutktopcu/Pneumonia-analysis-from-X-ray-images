"""Model definitions. build_model(name) is the single entry point used by train.py."""
import torch
from torch import nn


class SmallCNN(nn.Module):
    """Baseline CNN trained from scratch (~0.39 M parameters; ResNet-18 has ~11.7 M).

    4 blocks of conv3x3 -> BatchNorm -> ReLU -> 2x2 max-pool (224 -> 112 -> 56 -> 28 -> 14),
    then global average pooling, dropout and one linear unit that outputs a logit for PNEUMONIA.
    """

    def __init__(self, channels=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, in_ch = [], 3
        for out_ch in channels:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),  # bias redundant before BN
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(self.pool(self.features(x)), 1)
        return self.head(x).squeeze(1)  # shape (batch,) of logits


def build_model(name: str) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN()
    raise ValueError(f"unknown model: {name}")
