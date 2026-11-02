"""Count tracks crossing a virtual line, by direction and class.

A crossing needs a track to be seen clearly on one side and then clearly on the other,
with a dead band around the line. Without the dead band, box jitter around the line reads
as a stream of crossings. A track counts at most once per direction, so a jittering or
turning vehicle cannot inflate the tally.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CrossingEvent:
    frame: int
    track_id: int
    direction: str
    label: int


@dataclass
class LineCounter:
    """Line from ``start`` to ``end``. "forward" means crossing towards the left normal."""

    start: tuple[float, float]
    end: tuple[float, float]
    dead_band: float = 4.0
    margin: float = 10.0
    events: list[CrossingEvent] = field(default_factory=list)
    _side: dict[int, int] = field(default_factory=dict)
    _counted: set[tuple[int, str]] = field(default_factory=set)

    def __post_init__(self) -> None:
        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        self._origin = start
        self._length = float(np.linalg.norm(end - start))
        if self._length == 0:
            raise ValueError("counting line has zero length")
        self._along = (end - start) / self._length
        self._normal = np.array([self._along[1], -self._along[0]])

    def signed_distance(self, point: np.ndarray) -> float:
        return float(np.dot(np.asarray(point) - self._origin, self._normal))

    def _within_span(self, point: np.ndarray) -> bool:
        projection = float(np.dot(np.asarray(point) - self._origin, self._along))
        return -self.margin <= projection <= self._length + self.margin

    def update(self, frame: int, tracks) -> list[CrossingEvent]:
        """Feed one frame's tracks; each needs ``track_id``, ``center`` and ``stable_label``."""
        new_events = []
        for track in tracks:
            center = track.center
            distance = self.signed_distance(center)
            if abs(distance) <= self.dead_band:
                continue
            side = 1 if distance > 0 else -1
            previous = self._side.get(track.track_id)
            self._side[track.track_id] = side
            if previous is None or previous == side or not self._within_span(center):
                continue
            direction = "forward" if side > 0 else "backward"
            key = (track.track_id, direction)
            if key in self._counted:
                continue
            self._counted.add(key)
            event = CrossingEvent(frame, track.track_id, direction, track.stable_label)
            self.events.append(event)
            new_events.append(event)
        return new_events

    def totals(self) -> Counter:
        """Counts keyed by ``(direction, label)``."""
        return Counter((e.direction, e.label) for e in self.events)

    def direction_totals(self) -> Counter:
        return Counter(e.direction for e in self.events)
