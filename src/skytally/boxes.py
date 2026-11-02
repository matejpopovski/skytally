"""Box geometry and non-maximum suppression.

Evaluation and post-processing run in NumPy on the CPU, because they are branchy and
sized by the data. Training losses (``aligned_iou``, ``aligned_giou``) are in PyTorch and
avoid boolean indexing, which forces a synchronisation on the MPS backend.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in ``a`` (N, 4) and ``b`` (M, 4), both xyxy. Returns (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS. Returns kept indices, highest score first.

    A box is suppressed when its IoU with a kept box is strictly above the threshold,
    which matches ``torchvision.ops.nms``. Ties in score keep the earlier box.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0) * (y1 - y0)

    keep: list[int] = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        w = np.clip(np.minimum(x1[best], x1[rest]) - np.maximum(x0[best], x0[rest]), 0.0, None)
        h = np.clip(np.minimum(y1[best], y1[rest]) - np.maximum(y0[best], y0[rest]), 0.0, None)
        inter = w * h
        union = areas[best] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Per-class NMS. Boxes of different classes never suppress each other."""
    labels = np.asarray(labels)
    kept = []
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        kept.append(idx[nms(boxes[idx], scores[idx], iou_threshold)])
    if not kept:
        return np.empty(0, dtype=np.int64)
    merged = np.concatenate(kept)
    return merged[np.argsort(-np.asarray(scores)[merged], kind="stable")]


def aligned_iou(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise IoU of two (..., 4) xyxy tensors."""
    return _aligned_overlap(a, b, eps)[0]


def aligned_giou(a: Tensor, b: Tensor, eps: float = 1e-7) -> Tensor:
    """Element-wise generalised IoU of two (..., 4) xyxy tensors."""
    iou, union = _aligned_overlap(a, b, eps)
    lt = torch.minimum(a[..., :2], b[..., :2])
    rb = torch.maximum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0.0)
    hull = wh[..., 0] * wh[..., 1]
    return iou - (hull - union) / (hull + eps)


def _aligned_overlap(a: Tensor, b: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0.0) * (a[..., 3] - a[..., 1]).clamp(min=0.0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0.0) * (b[..., 3] - b[..., 1]).clamp(min=0.0)
    union = area_a + area_b - inter
    return inter / (union + eps), union
