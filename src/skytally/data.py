"""VisDrone-DET loading, training crops and the evaluation view of the annotations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from skytally.assign import LevelSpec, TargetGrid, assign_targets
from skytally.coco_map import ImageTruth

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IGNORED_REGION = 0
OTHERS = 11
# Objects narrower than this after resizing are unlearnable, and treating them as
# background would teach the model to reject real objects, so they become ignore regions.
MIN_VISIBLE_SIDE = 2.0
MIN_VISIBLE_FRACTION = 0.4


@dataclass
class Annotations:
    """One image's labels in xyxy pixels. ``ignore`` boxes get no class label."""

    boxes: np.ndarray
    labels: np.ndarray
    ignore: np.ndarray


def parse_annotation(text: str) -> Annotations:
    """Parse VisDrone's ``x,y,w,h,score,category,truncation,occlusion`` lines.

    Categories 1 to 10 are objects. Category 0 marks an ignored region and 11 is "others".
    A score of 0 means the box is excluded from evaluation. All of those become ignore
    boxes, because the imagery there holds unlabelled objects.
    """
    boxes, labels, ignore = [], [], []
    for line in text.splitlines():
        fields = [f for f in line.strip().split(",") if f != ""]
        if len(fields) < 6:
            continue
        x, y, w, h, score, category = (int(float(v)) for v in fields[:6])
        if w <= 0 or h <= 0:
            continue
        box = (x, y, x + w, y + h)
        if category in (IGNORED_REGION, OTHERS) or score == 0:
            ignore.append(box)
        else:
            boxes.append(box)
            labels.append(category - 1)
    return Annotations(
        np.array(boxes, dtype=np.float32).reshape(-1, 4),
        np.array(labels, dtype=np.int64),
        np.array(ignore, dtype=np.float32).reshape(-1, 4),
    )


def load_split(root: Path) -> tuple[list[Path], list[Annotations]]:
    images = sorted((root / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"no images under {root}, run scripts/download_visdrone.py")
    annotations = [
        parse_annotation((root / "annotations" / f"{p.stem}.txt").read_text()) for p in images
    ]
    return images, annotations


def to_truth(ann: Annotations, num_classes: int) -> ImageTruth:
    """COCO view of the labels. Each ignore box is a crowd box in every class.

    pycocotools matches crowd boxes per category, so a region that should forgive any
    detection has to be repeated for each one.
    """
    boxes = _xyxy_to_xywh(ann.boxes)
    ignore = _xyxy_to_xywh(ann.ignore)
    n = len(ignore)
    return ImageTruth(
        np.concatenate([boxes, np.tile(ignore, (num_classes, 1))]),
        np.concatenate([ann.labels, np.repeat(np.arange(num_classes), n)]),
        np.concatenate([np.zeros(len(boxes), bool), np.ones(n * num_classes, bool)]),
    )


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = np.array(boxes, dtype=np.float64)
    out[:, 2:] -= out[:, :2]
    return out


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB to normalised CHW float."""
    scaled = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(scaled.transpose(2, 0, 1).copy())


@dataclass(frozen=True)
class CropConfig:
    size: int = 640
    scale_min: float = 0.75
    scale_max: float = 1.5
    object_centered: float = 0.75
    flip: float = 0.5
    color_jitter: float = 0.25


def random_crop(
    image: Image.Image, ann: Annotations, config: CropConfig, rng: np.random.Generator
) -> tuple[np.ndarray, Annotations]:
    """Sample a square crop at a random scale, biased towards labelled objects.

    A uniform crop spends many samples on rooftops and fields. Boxes that the crop cuts
    down to a sliver move to the ignore list instead of becoming background.
    """
    scale = float(np.exp(rng.uniform(np.log(config.scale_min), np.log(config.scale_max))))
    window = config.size / scale
    width, height = image.size

    if len(ann.boxes) and rng.random() < config.object_centered:
        box = ann.boxes[rng.integers(len(ann.boxes))]
        cx = (box[0] + box[2]) / 2 + rng.uniform(-0.3, 0.3) * window
        cy = (box[1] + box[3]) / 2 + rng.uniform(-0.3, 0.3) * window
    else:
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)

    x0 = _place_window(cx - window / 2, window, width)
    y0 = _place_window(cy - window / 2, window, height)
    region = (x0, y0, x0 + window, y0 + window)

    pixels = _resample(image, region, config.size)
    boxes, labels, ignored = _crop_boxes(ann, region, scale, config.size)

    if rng.random() < config.flip:
        pixels = pixels[:, ::-1]
        boxes = _flip_boxes(boxes, config.size)
        ignored = _flip_boxes(ignored, config.size)
    if config.color_jitter > 0:
        pixels = _jitter(pixels, config.color_jitter, rng)
    return pixels, Annotations(boxes, labels, ignored)


def _place_window(start: float, window: float, extent: float) -> float:
    if window >= extent:
        return (extent - window) / 2
    return float(np.clip(start, 0.0, extent - window))


def _resample(image: Image.Image, region: tuple[float, ...], size: int) -> np.ndarray:
    x0, y0, x1, y1 = region
    width, height = image.size
    inner = (max(x0, 0.0), max(y0, 0.0), min(x1, width), min(y1, height))
    scale = size / (x1 - x0)
    # Rounding the destination rectangle keeps the sampled source box aligned to it.
    left = round((inner[0] - x0) * scale)
    top = round((inner[1] - y0) * scale)
    out_w = max(1, round((inner[2] - inner[0]) * scale))
    out_h = max(1, round((inner[3] - inner[1]) * scale))
    resized = image.resize((out_w, out_h), Image.Resampling.BILINEAR, box=inner)
    canvas = np.full((size, size, 3), 124, dtype=np.uint8)
    pasted = np.asarray(resized)[: size - top, : size - left]
    canvas[top : top + pasted.shape[0], left : left + pasted.shape[1]] = pasted
    return canvas


def _crop_boxes(
    ann: Annotations, region: tuple[float, ...], scale: float, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.array(region[:2] * 2, dtype=np.float32)
    boxes = (ann.boxes - origin) * scale
    clipped = np.clip(boxes, 0, size)
    wh = clipped[:, 2:] - clipped[:, :2]
    full_area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    visible = np.prod(np.clip(wh, 0, None), axis=1) / np.maximum(full_area, 1e-6)
    present = visible > 0
    usable = (visible >= MIN_VISIBLE_FRACTION) & (wh.min(axis=1) >= MIN_VISIBLE_SIDE)

    regions = np.clip((ann.ignore - origin) * scale, 0, size)
    regions = regions[(regions[:, 2] > regions[:, 0]) & (regions[:, 3] > regions[:, 1])]
    ignored = np.concatenate([regions, clipped[present & ~usable]])
    return clipped[usable], ann.labels[usable], ignored


def _flip_boxes(boxes: np.ndarray, size: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = size - boxes[:, 2]
    out[:, 2] = size - boxes[:, 0]
    return out


def _jitter(pixels: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    gain = rng.uniform(1 - amount, 1 + amount)
    contrast = rng.uniform(1 - amount, 1 + amount)
    tint = rng.uniform(1 - amount / 3, 1 + amount / 3, 3)
    out = pixels.astype(np.float32)
    mean = out.mean()
    out = (out - mean) * contrast + mean * gain
    return np.clip(out * tint, 0, 255).astype(np.uint8)


@dataclass
class TrainingSamples(Dataset):
    """Crops plus assigned targets, computed in the workers.

    Items are addressed by ``(seed, epoch, index)`` and each one draws from a generator
    seeded by that key. Augmentation therefore does not depend on which worker serves
    the item, or on restarts, which is what lets a resumed run reproduce the original.
    """

    images: list[Path]
    annotations: list[Annotations]
    levels: tuple[LevelSpec, ...]
    crop: CropConfig = field(default_factory=CropConfig)

    def __post_init__(self) -> None:
        self.grid = TargetGrid(self.crop.size, self.crop.size, self.levels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, key: tuple[int, int, int]) -> tuple[torch.Tensor, ...]:
        seed, epoch, index = key
        rng = np.random.default_rng((seed, epoch, index))
        with Image.open(self.images[index]) as handle:
            image = handle.convert("RGB")
        pixels, ann = random_crop(image, self.annotations[index], self.crop, rng)
        targets = assign_targets(ann.boxes, ann.labels, ann.ignore, self.grid)
        return (
            to_tensor(pixels),
            torch.from_numpy(targets.classes),
            torch.from_numpy(targets.boxes),
        )


class EpochSampler(Sampler):
    """Deterministic per-epoch shuffle that can start from any position."""

    def __init__(self, length: int, seed: int, epoch: int = 0, start: int = 0) -> None:
        self.length = length
        self.seed = seed
        self.epoch = epoch
        self.start = start

    def __iter__(self):
        order = np.random.default_rng((self.seed, self.epoch)).permutation(self.length)
        for index in order[self.start :]:
            yield (self.seed, self.epoch, int(index))

    def __len__(self) -> int:
        return self.length - self.start


def letterbox_size(
    width: int, height: int, long_side: int, divisor: int = 32
) -> tuple[int, int, float]:
    """Network input size for a whole image: scaled to ``long_side``, rounded up to ``divisor``."""
    scale = long_side / max(width, height)
    new_w = round(width * scale)
    new_h = round(height * scale)
    pad_w = (new_w + divisor - 1) // divisor * divisor
    pad_h = (new_h + divisor - 1) // divisor * divisor
    return pad_w, pad_h, scale
