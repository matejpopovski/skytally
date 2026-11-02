"""Vehicle motion on a straight multi-lane road viewed from above."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# VisDrone class indices, so simulated labels line up with the detector's.
CAR, VAN, TRUCK, BUS = 3, 4, 5, 8
VEHICLE_SHAPES = {CAR: (24.0, 11.0), VAN: (29.0, 13.0), TRUCK: (44.0, 15.0), BUS: (54.0, 15.0)}
VEHICLE_MIX = {CAR: 0.74, VAN: 0.16, TRUCK: 0.06, BUS: 0.04}
MIN_GAP_PX = 10.0


@dataclass(frozen=True)
class SceneConfig:
    width: int = 1280
    height: int = 720
    fps: float = 10.0
    seconds: float = 60.0
    seed: int = 0
    road_angle_deg: float = -12.0
    lanes_per_direction: int = 2
    lane_width: float = 22.0
    median_width: float = 14.0
    arrivals_per_minute: float = 26.0
    speed_px_per_s: tuple[float, float] = (60.0, 110.0)
    size_scale: float = 1.0
    # Band along the road, in px from its start, that hides traffic (a bridge or tree line).
    occluder: tuple[float, float] | None = None
    camera_jitter_px: float = 0.6
    line_position: float = 0.5
    warmup_seconds: float = 8.0


@dataclass
class Vehicle:
    vehicle_id: int
    label: int
    lane: int
    direction: int
    length: float
    width: float
    position: float
    speed: float
    desired_speed: float


@dataclass
class ObjectState:
    """A vehicle in one frame."""

    vehicle_id: int
    label: int
    box: np.ndarray
    center: np.ndarray
    heading: float
    length: float
    width: float
    visible: bool


@dataclass
class Sequence:
    config: SceneConfig
    frames: list[list[ObjectState]]
    line: tuple[tuple[float, float], tuple[float, float]]
    crossings: list[tuple[int, int, str, int]] = field(default_factory=list)

    def true_counts(self) -> dict[str, int]:
        totals = {"forward": 0, "backward": 0}
        for _, _, direction, _ in self.crossings:
            totals[direction] += 1
        return totals


def road_axes(config: SceneConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Road origin, unit vector along it, unit normal, and length through the frame."""
    angle = np.deg2rad(config.road_angle_deg)
    along = np.array([np.cos(angle), np.sin(angle)])
    normal = np.array([-along[1], along[0]])
    center = np.array([config.width / 2, config.height / 2])
    half = np.hypot(config.width, config.height) / 2 + 80
    return center - along * half, along, normal, 2 * half


def lane_offset(config: SceneConfig, direction: int, lane: int) -> float:
    """Signed lateral distance from the road centre line."""
    near_centre = config.median_width / 2 + config.lane_width * (lane + 0.5)
    return direction * near_centre


def _spawn(config, rng, next_id, traffic, road_length) -> Vehicle | None:
    direction = int(rng.choice([1, -1]))
    lane = int(rng.integers(config.lanes_per_direction))
    start = 0.0
    same_lane = [v for v in traffic if v.lane == lane and v.direction == direction]
    if same_lane:
        tail = min(v.position - v.length / 2 for v in same_lane)
        if tail < 80.0:
            return None
    label = int(rng.choice(list(VEHICLE_MIX), p=list(VEHICLE_MIX.values())))
    length, width = VEHICLE_SHAPES[label]
    desired = float(rng.uniform(*config.speed_px_per_s))
    if label in (TRUCK, BUS):
        desired *= 0.85
    return Vehicle(
        next_id,
        label,
        lane,
        direction,
        length * config.size_scale,
        width * config.size_scale,
        start,
        desired,
        desired,
    )


def _advance(traffic: list[Vehicle], dt: float) -> None:
    """Keep each vehicle behind its leader. Speeds relax towards the desired value."""
    for vehicle in traffic:
        leaders = [
            v
            for v in traffic
            if v.lane == vehicle.lane
            and v.direction == vehicle.direction
            and v.position > vehicle.position
        ]
        target = vehicle.desired_speed
        if leaders:
            leader = min(leaders, key=lambda v: v.position)
            gap = leader.position - leader.length / 2 - vehicle.position - vehicle.length / 2
            if gap < MIN_GAP_PX + vehicle.speed * 0.8:
                target = min(target, leader.speed * max(gap / (MIN_GAP_PX + 1), 0.0))
        vehicle.speed += (target - vehicle.speed) * min(1.0, dt * 1.5)
        vehicle.position += vehicle.speed * dt


def simulate(config: SceneConfig) -> Sequence:
    """Run the traffic model. Crossings are read from true positions, never estimated."""
    rng = np.random.default_rng(config.seed)
    origin, along, normal, road_length = road_axes(config)
    dt = 1.0 / config.fps
    total_frames = round(config.seconds * config.fps)
    warmup = round(config.warmup_seconds * config.fps)
    arrival_probability = config.arrivals_per_minute / 60.0 * dt

    center_world = np.array([config.width / 2, config.height / 2])
    span = (config.median_width / 2 + config.lanes_per_direction * config.lane_width) * 1.15
    line_center = center_world + along * (config.line_position - 0.5) * config.width * 0.9
    line = (tuple(line_center - normal * span), tuple(line_center + normal * span))
    # Orientation of the line, used to label which way a vehicle crossed.
    line_at = line_center

    traffic: list[Vehicle] = []
    frames: list[list[ObjectState]] = []
    crossings: list[tuple[int, int, str, int]] = []
    previous_side: dict[int, float] = {}
    next_id = 1
    forward_normal = _line_normal(line)

    for frame in range(-warmup, total_frames):
        if rng.random() < arrival_probability:
            vehicle = _spawn(config, rng, next_id, traffic, road_length)
            if vehicle is not None:
                traffic.append(vehicle)
                next_id += 1
        _advance(traffic, dt)
        traffic = [v for v in traffic if v.position < road_length]
        if frame < 0:
            for vehicle in traffic:
                center = _center(vehicle, config, origin, along, normal, road_length)
                previous_side[vehicle.vehicle_id] = float(np.dot(center - line_at, forward_normal))
            continue

        shake = rng.normal(0, config.camera_jitter_px, 2) if config.camera_jitter_px else 0.0
        states = []
        for vehicle in traffic:
            center = _center(vehicle, config, origin, along, normal, road_length)
            side = float(np.dot(center - line_at, forward_normal))
            before = previous_side.get(vehicle.vehicle_id)
            if before is not None and before * side < 0:
                direction = "forward" if side > 0 else "backward"
                crossings.append((frame, vehicle.vehicle_id, direction, vehicle.label))
            previous_side[vehicle.vehicle_id] = side

            drawn = center + shake
            state = _state(vehicle, drawn, along, config)
            if _on_screen(state, config):
                states.append(state)
        frames.append(states)
    return Sequence(config, frames, line, crossings)


def _line_normal(line) -> np.ndarray:
    start, end = np.asarray(line[0]), np.asarray(line[1])
    along = (end - start) / np.linalg.norm(end - start)
    return np.array([along[1], -along[0]])


def _center(vehicle, config, origin, along, normal, road_length) -> np.ndarray:
    position = vehicle.position if vehicle.direction > 0 else road_length - vehicle.position
    return origin + along * position + normal * lane_offset(config, vehicle.direction, vehicle.lane)


def _state(vehicle: Vehicle, center: np.ndarray, along: np.ndarray, config) -> ObjectState:
    heading = float(np.arctan2(along[1], along[0]))
    if vehicle.direction < 0:
        heading += np.pi
    half_w = abs(np.cos(heading)) * vehicle.length / 2 + abs(np.sin(heading)) * vehicle.width / 2
    half_h = abs(np.sin(heading)) * vehicle.length / 2 + abs(np.cos(heading)) * vehicle.width / 2
    box = np.array([center[0] - half_w, center[1] - half_h, center[0] + half_w, center[1] + half_h])
    visible = True
    if config.occluder is not None:
        low, high = config.occluder
        progress = float(np.dot(center - np.array([config.width / 2, config.height / 2]), along))
        progress += config.width / 2
        visible = not (low <= progress <= high)
    return ObjectState(
        vehicle.vehicle_id,
        vehicle.label,
        box,
        center,
        heading,
        vehicle.length,
        vehicle.width,
        visible,
    )


def _on_screen(state: ObjectState, config: SceneConfig) -> bool:
    x0, y0, x1, y1 = state.box
    return x1 > 0 and y1 > 0 and x0 < config.width and y0 < config.height
