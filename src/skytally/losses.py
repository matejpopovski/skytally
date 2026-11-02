"""Detection losses, written without boolean indexing.

Selecting positives with a mask needs ``nonzero`` on the MPS backend, which stalls the
command queue. Every term here is computed for all points and multiplied by a weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from skytally.assign import BACKGROUND, IGNORE
from skytally.boxes import aligned_giou, aligned_iou
from skytally.model import decode_boxes


def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    """Element-wise focal loss (Lin et al., 2017) for multi-label targets in {0, 1}."""
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    agreement = probability * targets + (1 - probability) * (1 - targets)
    weight = alpha * targets + (1 - alpha) * (1 - targets)
    return weight * (1 - agreement) ** gamma * ce


@dataclass(frozen=True)
class LossWeights:
    box: float = 2.0
    quality: float = 1.0


def detection_loss(
    cls_logits: Tensor,
    box_raw: Tensor,
    quality_logits: Tensor,
    points: Tensor,
    strides: Tensor,
    target_classes: Tensor,
    target_boxes: Tensor,
    normalizer: Tensor | float,
    weights: LossWeights | None = None,
) -> dict[str, Tensor]:
    """Sum of losses, each normalised by the (smoothed) number of positive points."""
    weights = weights or LossWeights()
    positive = target_classes >= 0
    used = (target_classes != IGNORE).to(cls_logits.dtype)
    positive_f = positive.to(cls_logits.dtype)

    # One-hot through scatter on (C + 1) classes, so background falls into a dropped slot.
    num_classes = cls_logits.shape[-1]
    slots = target_classes.clamp(min=BACKGROUND).masked_fill(~positive, num_classes)
    one_hot = torch.zeros((*target_classes.shape, num_classes + 1), device=cls_logits.device)
    one_hot.scatter_(-1, slots.unsqueeze(-1), 1.0)
    focal = sigmoid_focal_loss(cls_logits, one_hot[..., :num_classes])
    loss_cls = (focal.sum(dim=-1) * used).sum() / normalizer

    decoded = decode_boxes(box_raw, points, strides)
    giou = aligned_giou(decoded, target_boxes)
    loss_box = ((1 - giou) * positive_f).sum() / normalizer

    with torch.no_grad():
        iou = aligned_iou(decoded, target_boxes)
    bce = F.binary_cross_entropy_with_logits(quality_logits, iou, reduction="none")
    loss_quality = (bce * positive_f).sum() / normalizer

    total = loss_cls + weights.box * loss_box + weights.quality * loss_quality
    return {"total": total, "cls": loss_cls, "box": loss_box, "quality": loss_quality}
