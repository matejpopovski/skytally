"""COCO-style average precision, written from the definition.

The semantics follow pycocotools ``COCOeval`` for bounding boxes, including the details
that decide the numbers:

* 101-point interpolation over the precision envelope.
* Greedy matching in descending score order, preferring non-ignored ground truth.
* A crowd box can absorb many detections, and its IoU is intersection over detection
  area. VisDrone's ignored regions are mapped onto this rule.
* A detection left unmatched is discounted when its area is outside the evaluated range.

``tests/test_map_crosscheck.py`` holds this to agreement with pycocotools on random
scenes, and ``scripts/crosscheck_map.py`` does the same on the real validation
predictions.

One structural difference: greedy matching is prefix-stable, so the match for the top
``k`` detections is the first ``k`` entries of the match for the top ``K > k``. Matching
therefore runs once per area range at the largest ``max_dets``, then is truncated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

COCO_AREAS: dict[str, tuple[float, float]] = {
    "all": (0.0, 1e10),
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}
# COCO's "small" contains most of VisDrone, so a tiny bucket shows where the loss is.
VISDRONE_AREAS: dict[str, tuple[float, float]] = {"tiny": (0.0, 16.0**2), **COCO_AREAS}


@dataclass
class ImageTruth:
    """Ground truth for one image. Boxes are xywh, as in COCO annotations."""

    boxes: np.ndarray
    labels: np.ndarray
    crowd: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if self.crowd is None:
            self.crowd = np.zeros(len(self.labels), dtype=bool)
        self.crowd = np.asarray(self.crowd, dtype=bool).reshape(-1)


@dataclass
class ImageDetections:
    """Detections for one image. Boxes are xywh."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)


@dataclass(frozen=True)
class EvalParams:
    iou_thresholds: np.ndarray = field(default_factory=lambda: np.linspace(0.5, 0.95, 10))
    recall_thresholds: np.ndarray = field(default_factory=lambda: np.linspace(0.0, 1.0, 101))
    areas: Mapping[str, tuple[float, float]] = field(default_factory=lambda: dict(COCO_AREAS))
    max_dets: tuple[int, ...] = (1, 10, 100)


@dataclass
class MapResult:
    """Tensors shaped like ``COCOeval.eval``, with -1 marking undefined entries.

    ``precision`` is (iou, recall, class, area, max_dets), ``recall`` is (iou, class, area,
    max_dets).
    """

    params: EvalParams
    class_ids: np.ndarray
    precision: np.ndarray
    recall: np.ndarray

    def ap(
        self,
        iou: float | None = None,
        area: str = "all",
        max_dets: int | None = None,
        class_id: int | None = None,
    ) -> float:
        """Mean precision over the selected slice, or -1 when no ground truth exists."""
        selected = self.precision[
            ...,
            self._area_index(area),
            self._max_det_index(max_dets),
        ]
        selected = selected[:, :, self._class_mask(class_id)]
        if iou is not None:
            selected = selected[self._iou_mask(iou)]
        return _mean_defined(selected)

    def ar(
        self,
        iou: float | None = None,
        area: str = "all",
        max_dets: int | None = None,
        class_id: int | None = None,
    ) -> float:
        selected = self.recall[
            ...,
            self._area_index(area),
            self._max_det_index(max_dets),
        ]
        selected = selected[:, self._class_mask(class_id)]
        if iou is not None:
            selected = selected[self._iou_mask(iou)]
        return _mean_defined(selected)

    def summary(self) -> dict[str, float]:
        """Headline numbers at the largest ``max_dets``."""
        out = {
            "AP": self.ap(),
            "AP50": self.ap(iou=0.5),
            "AP75": self.ap(iou=0.75),
        }
        for name in self.params.areas:
            if name != "all":
                out[f"AP_{name}"] = self.ap(area=name)
        for k in self.params.max_dets:
            out[f"AR_{k}"] = self.ar(max_dets=k)
        for name in self.params.areas:
            if name != "all":
                out[f"AR_{name}"] = self.ar(area=name)
        return out

    def _area_index(self, area: str) -> int:
        return list(self.params.areas).index(area)

    def _max_det_index(self, max_dets: int | None) -> int:
        if max_dets is None:
            return len(self.params.max_dets) - 1
        return self.params.max_dets.index(max_dets)

    def _class_mask(self, class_id: int | None) -> np.ndarray:
        if class_id is None:
            return np.ones(len(self.class_ids), dtype=bool)
        return self.class_ids == class_id

    def _iou_mask(self, iou: float) -> np.ndarray:
        return np.isclose(self.params.iou_thresholds, iou)


def _mean_defined(values: np.ndarray) -> float:
    defined = values[values > -1]
    return float(defined.mean()) if defined.size else -1.0


def _xywh_iou(dets: np.ndarray, gts: np.ndarray, crowd: np.ndarray) -> np.ndarray:
    """IoU matrix (D, G). Against a crowd box the denominator is the detection's own area."""
    lt = np.maximum(dets[:, None, :2], gts[None, :, :2])
    rb = np.minimum(dets[:, None, :2] + dets[:, None, 2:], gts[None, :, :2] + gts[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    det_area = dets[:, 2] * dets[:, 3]
    gt_area = gts[:, 2] * gts[:, 3]
    union = det_area[:, None] + gt_area[None, :] - inter
    union = np.where(crowd[None, :], det_area[:, None], union)
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _match(
    iou: np.ndarray,
    crowd: np.ndarray,
    ignored: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy matching for one image and class, with ground truth ordered non-ignored first.

    Returns ``matched`` and ``match_ignored`` (T, D): whether each detection found a
    ground truth at each threshold, and whether that ground truth is ignored.
    """
    num_dets, num_gts = iou.shape
    matched = np.zeros((len(thresholds), num_dets), dtype=bool)
    match_ignored = np.zeros((len(thresholds), num_dets), dtype=bool)
    if num_dets == 0 or num_gts == 0:
        return matched, match_ignored

    first_ignored = int(np.count_nonzero(~ignored))
    # Most low-scoring detections overlap nothing; skipping them saves most of the runtime.
    candidates = np.flatnonzero(iou.max(axis=1) >= thresholds.min() - 1e-10)
    for t, threshold in enumerate(thresholds):
        limit = min(threshold, 1 - 1e-10)
        taken = np.zeros(num_gts, dtype=bool)
        for d in candidates:
            hits = (iou[d] >= limit) & (~taken | crowd)
            # A real ground truth always beats an ignored one, whatever the IoU.
            pool = hits[:first_ignored]
            offset = 0
            if not pool.any():
                pool = hits[first_ignored:]
                offset = first_ignored
                if not pool.any():
                    continue
            values = np.where(pool, iou[d, offset : offset + len(pool)], -1.0)
            # pycocotools keeps scanning on equal IoU, so ties resolve to the last index.
            g = offset + len(values) - 1 - int(np.argmax(values[::-1]))
            taken[g] = True
            matched[t, d] = True
            match_ignored[t, d] = ignored[g]
    return matched, match_ignored


@dataclass
class _Record:
    scores: np.ndarray
    matched: np.ndarray
    ignored: np.ndarray
    num_truth: int


def _evaluate_image_class(
    gt_boxes: np.ndarray,
    gt_crowd: np.ndarray,
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    params: EvalParams,
) -> list[_Record]:
    max_k = max(params.max_dets)
    order = np.argsort(-det_scores, kind="stable")[:max_k]
    det_boxes = det_boxes[order]
    det_scores = det_scores[order]
    det_area = det_boxes[:, 2] * det_boxes[:, 3]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]

    records = []
    for low, high in params.areas.values():
        ignored = gt_crowd | (gt_area < low) | (gt_area > high)
        gt_order = np.argsort(ignored, kind="stable")
        iou = _xywh_iou(det_boxes, gt_boxes[gt_order], gt_crowd[gt_order])
        matched, match_ignored = _match(
            iou, gt_crowd[gt_order], ignored[gt_order], params.iou_thresholds
        )
        outside = (det_area < low) | (det_area > high)
        records.append(
            _Record(
                scores=det_scores,
                matched=matched,
                ignored=match_ignored | (~matched & outside),
                num_truth=int(np.count_nonzero(~ignored)),
            )
        )
    return records


def _accumulate(
    records: Sequence[_Record], max_det: int, recall_thresholds: np.ndarray, num_iou: int
) -> tuple[np.ndarray, np.ndarray]:
    num_truth = sum(r.num_truth for r in records)
    precision = np.full((num_iou, len(recall_thresholds)), -1.0)
    recall = np.full(num_iou, -1.0)
    if num_truth == 0:
        return precision, recall

    scores = np.concatenate([r.scores[:max_det] for r in records])
    order = np.argsort(-scores, kind="mergesort")
    matched = np.concatenate([r.matched[:, :max_det] for r in records], axis=1)[:, order]
    ignored = np.concatenate([r.ignored[:, :max_det] for r in records], axis=1)[:, order]

    tp = np.cumsum(matched & ~ignored, axis=1, dtype=np.float64)
    fp = np.cumsum(~matched & ~ignored, axis=1, dtype=np.float64)
    eps = np.spacing(1)
    for t in range(num_iou):
        rc = tp[t] / num_truth
        pr = tp[t] / (fp[t] + tp[t] + eps)
        recall[t] = rc[-1] if rc.size else 0.0
        envelope = np.maximum.accumulate(pr[::-1])[::-1]
        where = np.searchsorted(rc, recall_thresholds, side="left")
        row = np.zeros(len(recall_thresholds))
        valid = where < len(envelope)
        row[valid] = envelope[where[valid]]
        precision[t] = row
    return precision, recall


def evaluate(
    truths: Sequence[ImageTruth],
    detections: Sequence[ImageDetections],
    params: EvalParams | None = None,
    class_ids: Sequence[int] | None = None,
) -> MapResult:
    """Evaluate paired per-image ground truth and detections."""
    if len(truths) != len(detections):
        raise ValueError(f"{len(truths)} ground-truth images but {len(detections)} prediction sets")
    params = params or EvalParams()
    if class_ids is None:
        seen = [t.labels for t in truths] + [d.labels for d in detections]
        class_ids = np.unique(np.concatenate(seen)) if seen else np.empty(0, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)

    num_iou = len(params.iou_thresholds)
    num_rec = len(params.recall_thresholds)
    num_area = len(params.areas)
    num_k = len(params.max_dets)
    shape = (num_iou, num_rec, len(class_ids), num_area, num_k)
    precision = np.full(shape, -1.0)
    recall = np.full((num_iou, len(class_ids), num_area, num_k), -1.0)

    for c, class_id in enumerate(class_ids):
        per_area: list[list[_Record]] = [[] for _ in range(num_area)]
        for truth, det in zip(truths, detections, strict=True):
            gt_mask = truth.labels == class_id
            det_mask = det.labels == class_id
            if not gt_mask.any() and not det_mask.any():
                continue
            records = _evaluate_image_class(
                truth.boxes[gt_mask],
                truth.crowd[gt_mask],
                det.boxes[det_mask],
                det.scores[det_mask],
                params,
            )
            for a, record in enumerate(records):
                per_area[a].append(record)
        for a in range(num_area):
            for m, k in enumerate(params.max_dets):
                precision[:, :, c, a, m], recall[:, c, a, m] = _accumulate(
                    per_area[a], k, params.recall_thresholds, num_iou
                )
    return MapResult(params, class_ids, precision, recall)
