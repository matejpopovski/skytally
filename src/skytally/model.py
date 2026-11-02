"""Anchor-free detector: ImageNet ResNet backbone, FPN, and a light shared head."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34

from skytally import NUM_CLASSES
from skytally.assign import pyramid_levels

BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, (64, 128, 256, 512)),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, (64, 128, 256, 512)),
}
LOG_SIZE_RANGE = (-3.0, 7.0)
PRIOR_PROBABILITY = 0.01


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "resnet18"
    min_level: int = 2
    max_level: int = 5
    channels: int = 96
    head_depth: int = 2
    num_classes: int = NUM_CLASSES
    pretrained: bool = True
    frozen_stages: int = 1

    @property
    def levels(self):
        return pyramid_levels(self.min_level, self.max_level)

    def to_dict(self) -> dict:
        return asdict(self)


def separable(channels: int) -> nn.Sequential:
    """Depthwise then pointwise convolution, without normalisation or activation.

    Stride 4 has 16 times the points of stride 16, so the head's cost is dominated by P2.
    At 96 channels a separable layer takes about an eighth of the multiplies of a 3x3 one.
    """
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.Conv2d(channels, channels, 1, bias=False),
    )


class FPN(nn.Module):
    """Top-down feature pyramid."""

    def __init__(self, in_channels: tuple[int, ...], channels: int, min_level: int) -> None:
        super().__init__()
        used = in_channels[min_level - 2 :]
        self.lateral = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in used)
        self.smooth = nn.ModuleList(
            nn.Sequential(separable(channels), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
            for _ in used
        )
        self.min_level = min_level

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        features = features[self.min_level - 2 :]
        laterals = [conv(f) for conv, f in zip(self.lateral, features, strict=True)]
        out = [laterals[-1]]
        for lateral in reversed(laterals[:-1]):
            out.append(lateral + F.interpolate(out[-1], scale_factor=2.0, mode="nearest"))
        out.reverse()
        return [conv(f) for conv, f in zip(self.smooth, out, strict=True)]


class Tower(nn.Module):
    """Convolutions shared across pyramid levels, with BatchNorm private to each level.

    Levels have different feature statistics, so they cannot share running averages.
    GroupNorm would avoid that, but on the MPS backend it runs through a slow Python
    fallback instead of a native kernel (``scripts/bench_norm_mps.py`` measures the cost).
    """

    def __init__(self, channels: int, depth: int, num_levels: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(separable(channels) for _ in range(depth))
        self.norms = nn.ModuleList(
            nn.ModuleList(nn.BatchNorm2d(channels) for _ in range(num_levels)) for _ in range(depth)
        )

    def forward(self, x: Tensor, level: int) -> Tensor:
        for conv, norms in zip(self.convs, self.norms, strict=True):
            x = F.relu(norms[level](conv(x)), inplace=True)
        return x


class Head(nn.Module):
    """Classification, box and IoU-quality predictions.

    The quality branch predicts the IoU of the box it will output. FCOS's centerness is
    defined by a point's distance to the box edges, which turns to noise on objects a few
    cells wide and is undefined for a point outside its object.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        c = config.channels
        num_levels = len(config.levels)
        self.cls_tower = Tower(c, config.head_depth, num_levels)
        self.box_tower = Tower(c, config.head_depth, num_levels)
        self.cls = nn.Conv2d(c, config.num_classes, 1)
        self.box = nn.Conv2d(c, 4, 1)
        self.quality = nn.Conv2d(c, 1, 1)

        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.normal_(self.box.weight, std=0.01)
        nn.init.normal_(self.quality.weight, std=0.01)
        prior = torch.tensor(PRIOR_PROBABILITY)
        nn.init.constant_(self.cls.bias, float(-torch.log((1 - prior) / prior)))
        nn.init.zeros_(self.box.bias)
        nn.init.zeros_(self.quality.bias)

    def forward(self, feature: Tensor, level: int) -> tuple[Tensor, Tensor, Tensor]:
        cls_feature = self.cls_tower(feature, level)
        box_feature = self.box_tower(feature, level)
        return self.cls(cls_feature), self.box(box_feature), self.quality(box_feature)


class Detector(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        build, weights, channels = BACKBONES[config.backbone]
        net = build(weights=weights if config.pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])
        self.fpn = FPN(channels, config.channels, config.min_level)
        self.head = Head(config)
        self.strides = tuple(spec.stride for spec in config.levels)

        # Backpropagating through the high-resolution stages is expensive, and their
        # ImageNet filters are generic edges that a short run would barely improve.
        frozen = [self.stem, *self.stages[: config.frozen_stages]]
        for part in frozen:
            part.requires_grad_(False)

    def features(self, images: Tensor) -> list[Tensor]:
        with torch.no_grad():
            x = self.stem(images)
            frozen = self.stages[: self.config.frozen_stages]
            outputs = []
            for stage in frozen:
                x = stage(x)
                outputs.append(x)
        for stage in self.stages[self.config.frozen_stages :]:
            x = stage(x)
            outputs.append(x)
        return self.fpn(outputs)

    def forward(self, images: Tensor) -> list[tuple[Tensor, Tensor, Tensor]]:
        """Per-level ``(cls, box, quality)`` maps, finest level first."""
        return [self.head(f, i) for i, f in enumerate(self.features(images))]

    def freeze_backbone_norm(self) -> None:
        """Keep ImageNet BatchNorm statistics: batches of crops are too small to refit them."""
        for part in (self.stem, self.stages):
            for module in part.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()


def grid_points(height: int, width: int, stride: int, device, dtype=torch.float32) -> Tensor:
    """(H * W, 2) xy centres of a level's cells in input pixels."""
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1)


def decode_boxes(raw: Tensor, points: Tensor, strides: Tensor) -> Tensor:
    """Map (..., 4) centre-offset and log-size predictions to xyxy boxes.

    Size is exponential, so a box can never invert, and tiny sizes stay representable
    with fine relative precision.
    """
    center = points + raw[..., :2] * strides
    half = 0.5 * strides * raw[..., 2:].clamp(*LOG_SIZE_RANGE).exp()
    return torch.cat([center - half, center + half], dim=-1)


def flatten_outputs(
    outputs: list[tuple[Tensor, Tensor, Tensor]], strides: tuple[int, ...]
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Concatenate levels into (B, N, ...) tensors with their points and strides."""
    device = outputs[0][0].device
    cls_list, box_list, quality_list, point_list, stride_list = [], [], [], [], []
    for (cls, box, quality), stride in zip(outputs, strides, strict=True):
        height, width = cls.shape[-2:]
        cls_list.append(cls.flatten(2).transpose(1, 2))
        box_list.append(box.flatten(2).transpose(1, 2))
        quality_list.append(quality.flatten(1))
        point_list.append(grid_points(height, width, stride, device))
        stride_list.append(torch.full((height * width, 1), float(stride), device=device))
    return (
        torch.cat(cls_list, dim=1),
        torch.cat(box_list, dim=1),
        torch.cat(quality_list, dim=1),
        torch.cat(point_list),
        torch.cat(stride_list),
    )
