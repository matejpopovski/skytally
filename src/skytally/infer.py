"""Whole-image inference: network, candidate extraction, class-wise NMS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from skytally.boxes import batched_nms
from skytally.checkpoint import load_checkpoint
from skytally.data import letterbox_size, to_tensor
from skytally.model import Detector, ModelConfig, decode_boxes, grid_points


@dataclass(frozen=True)
class InferConfig:
    long_side: int = 1536
    score_threshold: float = 0.01
    topk_per_level: int = 4000
    nms_iou: float = 0.6
    max_dets: int = 500


@dataclass
class Candidates:
    """Pre-NMS detections in original image pixels, xyxy."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray


def best_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_detector(path: Path, device: torch.device) -> Detector:
    """Build a detector from a checkpoint, preferring the EMA weights."""
    state = load_checkpoint(path)
    config = ModelConfig(**{**state["model_config"], "pretrained": False})
    model = Detector(config)
    model.load_state_dict(state.get("ema") or state["model"])
    return model.to(device).eval()


def prepare(image: Image.Image, long_side: int) -> tuple[Tensor, float]:
    width, height = image.size
    pad_w, pad_h, scale = letterbox_size(width, height, long_side)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BILINEAR)
    tensor = to_tensor(np.asarray(resized))
    return F.pad(tensor, (0, pad_w - tensor.shape[2], 0, pad_h - tensor.shape[1])), scale


@torch.inference_mode()
def network_candidates(
    model: Detector, batch: Tensor, config: InferConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the network on a one-image batch. Returns (boxes, scores, labels) on the device."""
    outputs = model(batch)
    all_boxes, all_scores, all_labels = [], [], []
    for (cls, box, quality), stride in zip(outputs, model.strides, strict=True):
        height, width = cls.shape[-2:]
        num_classes = cls.shape[1]
        # Geometric mean, so that a box needs both the right class and a good fit.
        scores = (cls.sigmoid() * quality.sigmoid()).sqrt()[0].flatten(1).T.flatten()
        top = scores.topk(min(config.topk_per_level, scores.numel()))
        point_index = top.indices // num_classes
        points = grid_points(height, width, stride, batch.device)
        raw = box[0].flatten(1).T[point_index]
        all_boxes.append(
            decode_boxes(raw, points[point_index], torch.tensor(float(stride), device=batch.device))
        )
        all_scores.append(top.values)
        all_labels.append(top.indices % num_classes)
    return torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)


def to_candidates(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    scale: float,
    size: tuple[int, int],
    config: InferConfig,
) -> Candidates:
    """Move to the CPU, threshold, and map back to original image coordinates."""
    boxes_np = boxes.cpu().numpy().astype(np.float64) / scale
    scores_np = scores.cpu().numpy().astype(np.float64)
    labels_np = labels.cpu().numpy()
    keep = scores_np >= config.score_threshold
    boxes_np = boxes_np[keep]
    width, height = size
    boxes_np[:, [0, 2]] = boxes_np[:, [0, 2]].clip(0, width)
    boxes_np[:, [1, 3]] = boxes_np[:, [1, 3]].clip(0, height)
    return Candidates(boxes_np, scores_np[keep], labels_np[keep])


def suppress(candidates: Candidates, config: InferConfig) -> Candidates:
    keep = batched_nms(candidates.boxes, candidates.scores, candidates.labels, config.nms_iou)[
        : config.max_dets
    ]
    return Candidates(candidates.boxes[keep], candidates.scores[keep], candidates.labels[keep])


def detect(
    model: Detector, image: Image.Image, config: InferConfig, device: torch.device
) -> Candidates:
    """Full pipeline for one PIL image."""
    tensor, scale = prepare(image, config.long_side)
    boxes, scores, labels = network_candidates(model, tensor[None].to(device), config)
    return suppress(to_candidates(boxes, scores, labels, scale, image.size, config), config)
