"""Anchor-free target assignment.

Each object goes to exactly one pyramid level, chosen by its longer side. The positive
points are those inside the object and within ``center_radius`` strides of its centre.

Plain FCOS drops an object when no grid point lands inside it, which is a common event
for sub-stride objects. Here such an object is matched to its nearest grid point instead,
so every labelled object contributes to the loss. The box head predicts a centre offset
and a log size rather than distances to the four edges, because a matched point may sit
outside the object, where edge distances would be negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BACKGROUND = -1
IGNORE = -2


@dataclass(frozen=True)
class LevelSpec:
    """A pyramid level and the object sizes (longer side, in pixels) it is responsible for."""

    stride: int
    min_size: float
    max_size: float


def pyramid_levels(min_level: int, max_level: int = 5) -> tuple[LevelSpec, ...]:
    """Levels P<min_level>..P<max_level>. The finest level takes all small objects.

    The size bands double with stride, so each object spans a similar number of cells
    on its level.
    """
    levels = []
    for level in range(min_level, max_level + 1):
        stride = 2**level
        low = 0.0 if level == min_level else 8.0 * stride
        high = float("inf") if level == max_level else 16.0 * stride
        levels.append(LevelSpec(stride, low, high))
    return tuple(levels)


@dataclass
class Targets:
    """Per-point targets, levels concatenated finest first."""

    classes: np.ndarray  # int64 (N): class index, BACKGROUND or IGNORE
    boxes: np.ndarray  # float32 (N, 4): xyxy of the assigned object


class TargetGrid:
    """Point layout for one input size."""

    def __init__(self, height: int, width: int, levels: tuple[LevelSpec, ...]) -> None:
        self.levels = levels
        self.shapes = [(height // spec.stride, width // spec.stride) for spec in levels]
        self.counts = [rows * cols for rows, cols in self.shapes]
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)]).astype(int)
        self.num_points = int(self.offsets[-1])

    def level_for(self, size: np.ndarray) -> np.ndarray:
        """Index of the level responsible for each object size."""
        bounds = np.array([spec.max_size for spec in self.levels[:-1]])
        return np.searchsorted(bounds, size, side="right")


def assign_targets(
    boxes: np.ndarray,
    labels: np.ndarray,
    ignore: np.ndarray,
    grid: TargetGrid,
    center_radius: float = 1.5,
    force_match: bool = True,
) -> Targets:
    """Assign objects to points. Where objects overlap, the smallest one wins the point."""
    classes = [np.full(shape, BACKGROUND, dtype=np.int64) for shape in grid.shapes]
    owners = [np.full(shape, -1, dtype=np.int64) for shape in grid.shapes]

    for region in ignore:
        for level, spec in enumerate(grid.levels):
            rows, cols = _covered(region, spec.stride, grid.shapes[level])
            classes[level][rows, cols] = IGNORE

    if len(boxes):
        wh = boxes[:, 2:] - boxes[:, :2]
        target_level = grid.level_for(wh.max(axis=1))
        for g in np.argsort(-(wh[:, 0] * wh[:, 1]), kind="stable"):
            level = target_level[g]
            stride = grid.levels[level].stride
            shape = grid.shapes[level]
            cx, cy = (boxes[g, :2] + boxes[g, 2:]) / 2
            reach = center_radius * stride
            core = (
                max(boxes[g, 0], cx - reach),
                max(boxes[g, 1], cy - reach),
                min(boxes[g, 2], cx + reach),
                min(boxes[g, 3], cy + reach),
            )
            rows, cols = _covered(core, stride, shape)
            empty = rows.start >= rows.stop or cols.start >= cols.stop
            if empty:
                if not force_match:
                    continue
                row = int(np.clip(cy // stride, 0, shape[0] - 1))
                col = int(np.clip(cx // stride, 0, shape[1] - 1))
                rows, cols = slice(row, row + 1), slice(col, col + 1)
            classes[level][rows, cols] = labels[g]
            owners[level][rows, cols] = g

    flat_classes = np.concatenate([c.ravel() for c in classes])
    flat_owners = np.concatenate([o.ravel() for o in owners])
    flat_boxes = np.zeros((grid.num_points, 4), dtype=np.float32)
    positive = flat_owners >= 0
    if positive.any():
        flat_boxes[positive] = boxes[flat_owners[positive]]
    return Targets(flat_classes, flat_boxes)


def _covered(box, stride: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    """Grid cells whose centre, at ``(i + 0.5) * stride``, lies strictly inside ``box``."""

    def span(low: float, high: float, limit: int) -> slice:
        first = int(np.floor(low / stride - 0.5)) + 1
        last = int(np.ceil(high / stride - 0.5))
        return slice(max(first, 0), min(last, limit))

    return span(box[1], box[3], shape[0]), span(box[0], box[2], shape[1])


def coverage(boxes: np.ndarray, grid: TargetGrid, center_radius: float = 1.5) -> np.ndarray:
    """For each object, the number of points plain centre sampling would give it."""
    counts = np.zeros(len(boxes), dtype=np.int64)
    if not len(boxes):
        return counts
    wh = boxes[:, 2:] - boxes[:, :2]
    target_level = grid.level_for(wh.max(axis=1))
    for g, box in enumerate(boxes):
        level = target_level[g]
        stride = grid.levels[level].stride
        cx, cy = (box[:2] + box[2:]) / 2
        reach = center_radius * stride
        core = (
            max(box[0], cx - reach),
            max(box[1], cy - reach),
            min(box[2], cx + reach),
            min(box[3], cy + reach),
        )
        rows, cols = _covered(core, stride, grid.shapes[level])
        counts[g] = max(rows.stop - rows.start, 0) * max(cols.stop - cols.start, 0)
    return counts
