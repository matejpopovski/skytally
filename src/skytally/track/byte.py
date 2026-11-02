"""ByteTrack-style association (Zhang et al., 2022), tuned for small fast objects.

The core idea is unchanged: a low-confidence detection is usually a real object that is
partly hidden or blurred, so it is only discarded after it has failed to continue an
existing track. Three settings exist for tiny objects:

* ``buffer`` grows boxes for matching only, following the buffered IoU of C-BIoU (Yang
  et al., 2023). A 12-pixel car moving 12 pixels per frame has no overlap with its own
  previous box, so plain IoU cannot link it once the velocity estimate lags. Too much
  buffer reaches into the neighbouring lane, so the setting is measured, not assumed.
* ``lost_buffer`` widens the search for a track per frame it has gone unseen, because
  its predicted position gets less certain the longer it coasts. Healthy tracks are not
  affected, so occlusion recovery does not cost lane discipline.
* Kalman noise has a pixel floor, see ``skytally.track.kalman``.
* ``use_kalman=False`` and ``use_low_scores=False`` exist for the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from skytally.boxes import pairwise_iou
from skytally.track.kalman import BoxFilter, KalmanConfig


@dataclass(frozen=True)
class TrackerConfig:
    # Read off the detector's vehicle curve (results/operating_point.json): best F1 at 0.45,
    # 0.91 precision at 0.5, and no recall left to gain below 0.3.
    high_score: float = 0.45
    low_score: float = 0.3
    new_track_score: float = 0.5
    match_iou: float = 0.15
    low_match_iou: float = 0.25
    max_lost: int = 20
    min_hits: int = 2
    buffer: float = 0.25
    lost_buffer: float = 0.1
    use_kalman: bool = True
    use_low_scores: bool = True
    kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass
class Detection:
    """Detections for a frame, xyxy."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)

    def __len__(self) -> int:
        return len(self.scores)

    def select(self, mask: np.ndarray) -> Detection:
        return Detection(self.boxes[mask], self.scores[mask], self.labels[mask])


@dataclass
class Track:
    track_id: int
    filter: BoxFilter
    label: int
    score: float
    hits: int = 1
    lost: int = 0
    last_box: np.ndarray | None = None
    label_votes: dict[int, float] = field(default_factory=dict)
    use_filter: bool = True

    @property
    def box(self) -> np.ndarray:
        return self.filter.box if self.use_filter else self.last_box

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.box
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])

    @property
    def stable_label(self) -> int:
        """Score-weighted vote: a van seen 30 times is not relabelled by one 'car' frame."""
        return max(self.label_votes, key=self.label_votes.get) if self.label_votes else self.label


def _to_cxcywh(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]]
    )


def expand(boxes: np.ndarray, buffer: float | np.ndarray) -> np.ndarray:
    """Grow each box by ``buffer`` times its width and height on every side."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    wh = boxes[:, 2:] - boxes[:, :2]
    grow = np.asarray(buffer, dtype=np.float64).reshape(-1, 1) * wh
    return np.concatenate([boxes[:, :2] - grow, boxes[:, 2:] + grow], axis=1)


def buffered_iou(track_boxes: np.ndarray, det_boxes: np.ndarray, buffers: np.ndarray) -> np.ndarray:
    """IoU after growing a track and its candidate detections by that track's buffer.

    Both sides must grow together: a large box around a small one has a low IoU even when
    it contains it, so growing only the track would reject the match it was meant to find.
    """
    iou = np.zeros((len(track_boxes), len(det_boxes)))
    for buffer in np.unique(buffers):
        rows = buffers == buffer
        iou[rows] = pairwise_iou(expand(track_boxes[rows], buffer), expand(det_boxes, buffer))
    return iou


def match(
    track_boxes: np.ndarray,
    det_boxes: np.ndarray,
    min_iou: float,
    buffers: float | np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Optimal assignment on buffered IoU. Returns matches, free tracks, free detections."""
    num_tracks, num_dets = len(track_boxes), len(det_boxes)
    if num_tracks == 0 or num_dets == 0:
        return [], list(range(num_tracks)), list(range(num_dets))
    iou = buffered_iou(track_boxes, det_boxes, np.broadcast_to(buffers, (num_tracks,)))
    rows, cols = linear_sum_assignment(-iou)
    pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if iou[r, c] >= min_iou]
    matched_tracks = {r for r, _ in pairs}
    matched_dets = {c for _, c in pairs}
    return (
        [(int(r), int(c)) for r, c in pairs],
        [t for t in range(num_tracks) if t not in matched_tracks],
        [d for d in range(num_dets) if d not in matched_dets],
    )


class ByteTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: Detection) -> list[Track]:
        """Advance one frame. Returns confirmed tracks that were updated on this frame."""
        config = self.config
        for track in self.tracks:
            if config.use_kalman:
                track.filter.predict()

        high = detections.select(detections.scores >= config.high_score)
        low_mask = (detections.scores >= config.low_score) & (detections.scores < config.high_score)
        low = detections.select(low_mask)

        confirmed = [t for t in self.tracks if t.hits >= config.min_hits]
        tentative = [t for t in self.tracks if t.hits < config.min_hits]

        matches, free, free_high = self._associate(confirmed, high, config.match_iou)
        for t, d in matches:
            self._apply(confirmed[t], high, d)

        remaining = [confirmed[t] for t in free]
        if config.use_low_scores:
            # Only tracks seen last frame: a long-lost track grabbing a low score is a
            # common source of identity swaps.
            active = [t for t in remaining if t.lost == 0]
            low_matches, _, _ = self._associate(active, low, config.low_match_iou)
            for t, d in low_matches:
                self._apply(active[t], low, d)
                remaining.remove(active[t])

        leftovers = high.select(np.isin(np.arange(len(high)), free_high))
        tentative_matches, free_tentative, free_new = self._associate(
            tentative, leftovers, config.match_iou
        )
        for t, d in tentative_matches:
            self._apply(tentative[t], leftovers, d)

        for track in remaining:
            track.lost += 1
            track.filter.hold_size()
        dropped = {id(tentative[t]) for t in free_tentative}
        self.tracks = [t for t in self.tracks if id(t) not in dropped and t.lost <= config.max_lost]

        for d in free_new:
            if leftovers.scores[d] >= config.new_track_score:
                self._spawn(leftovers, d)

        return [t for t in self.tracks if t.lost == 0 and t.hits >= config.min_hits]

    def _associate(self, tracks: list[Track], detections: Detection, min_iou: float):
        boxes = np.array([self._reference_box(t) for t in tracks]).reshape(-1, 4)
        config = self.config
        lost = np.array([t.lost for t in tracks], dtype=np.float64)
        return match(boxes, detections.boxes, min_iou, config.buffer + config.lost_buffer * lost)

    def _reference_box(self, track: Track) -> np.ndarray:
        return track.box if self.config.use_kalman else track.last_box

    def _apply(self, track: Track, detections: Detection, index: int) -> None:
        box = detections.boxes[index]
        if self.config.use_kalman:
            track.filter.update(_to_cxcywh(box))
        track.last_box = box
        track.score = float(detections.scores[index])
        track.label = int(detections.labels[index])
        track.label_votes[track.label] = track.label_votes.get(track.label, 0.0) + track.score
        track.hits += 1
        track.lost = 0

    def _spawn(self, detections: Detection, index: int) -> None:
        box = detections.boxes[index]
        label = int(detections.labels[index])
        score = float(detections.scores[index])
        self.tracks.append(
            Track(
                self._next_id,
                BoxFilter(_to_cxcywh(box), self.config.kalman),
                label,
                score,
                last_box=box,
                label_votes={label: score},
                use_filter=self.config.use_kalman,
            )
        )
        self._next_id += 1
