"""Constant-velocity Kalman filter over box centre and size.

ByteTrack scales its process and measurement noise with box height alone. On a 12-pixel
car that gives a measurement sigma under a pixel, which is less than the detector's real
jitter, so the filter would chase noise. A floor in pixels keeps the sigmas honest at the
small end while leaving large boxes on the usual proportional scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATE = 8
MEASURE = 4


@dataclass(frozen=True)
class KalmanConfig:
    position_weight: float = 1 / 20
    velocity_weight: float = 1 / 160
    min_position_std: float = 1.5
    min_velocity_std: float = 0.5


class BoxFilter:
    """State is (cx, cy, w, h) and their per-frame rates."""

    _transition = np.eye(STATE)
    _transition[range(MEASURE), range(MEASURE, STATE)] = 1.0
    _observe = np.eye(MEASURE, STATE)

    def __init__(self, cxcywh: np.ndarray, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self.mean = np.concatenate([np.asarray(cxcywh, dtype=np.float64), np.zeros(MEASURE)])
        size = float(self.mean[2:4].max())
        std_position = self._position_std(size) * 2
        std_velocity = self._velocity_std(size) * 10
        self.covariance = np.diag([std_position**2] * MEASURE + [std_velocity**2] * MEASURE)

    def _position_std(self, size: float) -> float:
        return max(self.config.position_weight * size, self.config.min_position_std)

    def _velocity_std(self, size: float) -> float:
        return max(self.config.velocity_weight * size, self.config.min_velocity_std)

    def predict(self) -> None:
        size = float(self.mean[2:4].max())
        noise = np.diag(
            [self._position_std(size) ** 2] * MEASURE + [self._velocity_std(size) ** 2] * MEASURE
        )
        self.mean = self._transition @ self.mean
        self.covariance = self._transition @ self.covariance @ self._transition.T + noise
        # A shrinking box can extrapolate through zero during a long coast.
        self.mean[2:4] = np.maximum(self.mean[2:4], 1.0)

    def update(self, cxcywh: np.ndarray) -> None:
        size = float(self.mean[2:4].max())
        noise = np.eye(MEASURE) * self._position_std(size) ** 2
        innovation_cov = self._observe @ self.covariance @ self._observe.T + noise
        cross = self.covariance @ self._observe.T
        gain = np.linalg.solve(innovation_cov.T, cross.T).T
        self.mean = self.mean + gain @ (np.asarray(cxcywh) - self._observe @ self.mean)
        # Joseph form stays symmetric positive definite through long runs.
        identity = np.eye(STATE) - gain @ self._observe
        self.covariance = identity @ self.covariance @ identity.T + gain @ noise @ gain.T

    @property
    def box(self) -> np.ndarray:
        """Current estimate as xyxy."""
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def velocity(self) -> np.ndarray:
        return self.mean[4:6].copy()
