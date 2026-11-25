"""Model zoo.

``resnet50_scratch`` is a faithful PyTorch port of the original notebook's hand-built
Keras ResNet50 (identity / convolutional bottleneck blocks, Glorot-uniform init,
no pretrained weights). It is kept as the baseline.

Any other name is created through ``timm`` (e.g. ``resnet18``, ``resnet34``,
``efficientnet_b0``, ``convnext_tiny``), optionally with ImageNet weights —
transfer learning is by far the biggest single improvement on this dataset.
"""

from __future__ import annotations

import timm
import torch
from torch import nn

from . import NUM_CLASSES


class IdentityBlock(nn.Module):
    """Bottleneck block whose input and output shapes match (shortcut = identity)."""

    def __init__(self, in_ch: int, filters: tuple[int, int, int], f: int = 3):
        super().__init__()
        f1, f2, f3 = filters
        assert in_ch == f3
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, f1, 1, bias=False), nn.BatchNorm2d(f1), nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, f, padding=f // 2, bias=False), nn.BatchNorm2d(f2), nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, 1, bias=False), nn.BatchNorm2d(f3),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + x)


class ConvBlock(nn.Module):
    """Bottleneck block with a projection shortcut (used when shape changes)."""

    def __init__(self, in_ch: int, filters: tuple[int, int, int], f: int = 3, s: int = 2):
        super().__init__()
        f1, f2, f3 = filters
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, f1, 1, stride=s, bias=False), nn.BatchNorm2d(f1), nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, f, padding=f // 2, bias=False), nn.BatchNorm2d(f2), nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, 1, bias=False), nn.BatchNorm2d(f3),
        )
        self.shortcut = nn.Sequential(nn.Conv2d(in_ch, f3, 1, stride=s, bias=False), nn.BatchNorm2d(f3))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


def _stage(in_ch, filters, n_identity, s):
    return nn.Sequential(ConvBlock(in_ch, filters, s=s), *[IdentityBlock(filters[2], filters) for _ in range(n_identity)])


class ResNet50Scratch(nn.Module):
    """CONV -> BN -> RELU -> MAXPOOL -> [CONV + ID*2] -> [CONV + ID*3] -> [CONV + ID*5] -> [CONV + ID*2] -> POOL -> FC"""

    def __init__(self, num_classes: int = NUM_CLASSES, drop_rate: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.stage2 = _stage(64, (64, 64, 256), 2, s=1)
        self.stage3 = _stage(256, (128, 128, 512), 3, s=2)
        self.stage4 = _stage(512, (256, 256, 1024), 5, s=2)
        self.stage5 = _stage(1024, (512, 512, 2048), 2, s=2)
        # Global pooling instead of the original fixed 2x2 AveragePooling2D, so any input size works
        # (for 64x64 inputs the two are identical).
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(2048, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)  # == Keras glorot_uniform
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        return self.stage5(self.stage4(self.stage3(self.stage2(self.stem(x)))))

    def forward(self, x):
        return self.fc(self.dropout(torch.flatten(self.pool(self.forward_features(x)), 1)))


def build_model(arch: str, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                drop_rate: float = 0.0) -> nn.Module:
    if arch == "resnet50_scratch":
        return ResNet50Scratch(num_classes, drop_rate)
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)


def model_norm_stats(model: nn.Module):
    """Mean/std the backbone was pretrained with (ImageNet defaults for the scratch model)."""
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(cfg.get("std", (0.229, 0.224, 0.225)))


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature block, used for Grad-CAM."""
    for name in ("stage5", "layer4", "stages", "blocks", "features"):
        layer = getattr(model, name, None)
        if layer is not None:
            return layer[-1] if isinstance(layer, nn.Sequential) else layer
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1]
