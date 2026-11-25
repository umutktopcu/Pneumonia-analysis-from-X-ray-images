"""Model definitions. build_model(name) is the single entry point used by train.py."""
import torch
from torch import nn
from torchvision import models


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


class PretrainedClassifier(nn.Module):
    """ImageNet-pretrained torchvision backbone whose 1000-class head is replaced by one logit."""

    def __init__(self, backbone: nn.Module, head_attr: str):
        super().__init__()
        self.backbone = backbone
        self.head_attr = head_attr  # name of the replaced final layer, e.g. "fc" or "classifier"

    def head_parameters(self):
        return getattr(self.backbone, self.head_attr).parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


def _pretrained(name: str, pretrained: bool = True) -> PretrainedClassifier:
    """pretrained=False builds the same architecture with random weights (used when a fine-tuned
    checkpoint is loaded right after, and for the Grad-CAM randomization sanity check)."""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return PretrainedClassifier(m, "fc")
    if name == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        return PretrainedClassifier(m, "classifier")
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)  # classifier = Dropout + Linear
        return PretrainedClassifier(m, "classifier")
    raise ValueError(f"unknown model: {name}")


def build_model(name: str, freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    """freeze_backbone=True -> feature extraction: only the new head is trained."""
    if name == "small_cnn":
        if freeze_backbone:
            raise ValueError("small_cnn has no pretrained backbone to freeze")
        return SmallCNN()
    model = _pretrained(name, pretrained)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.head_parameters():
            param.requires_grad = True
    return model
