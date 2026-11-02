import numpy as np
import pytest

from skytally.sim.noise import DetectorModel, corrupt
from skytally.sim.traffic import SceneConfig, simulate
from skytally.track.byte import ByteTracker
from skytally.track.counter import LineCounter


def short_config(**kwargs):
    defaults = dict(seconds=30.0, seed=1, arrivals_per_minute=30.0)
    return SceneConfig(**{**defaults, **kwargs})


def test_simulation_is_deterministic():
    a, b = simulate(short_config()), simulate(short_config())
    assert a.crossings == b.crossings
    assert len(a.frames) == len(b.frames) == 300
    first_a = a.frames[100][0]
    first_b = b.frames[100][0]
    np.testing.assert_array_equal(first_a.box, first_b.box)


def test_seed_changes_traffic():
    assert simulate(short_config()).crossings != simulate(short_config(seed=2)).crossings


def test_vehicles_cross_the_line_in_both_directions():
    sequence = simulate(short_config(seconds=120.0))
    counts = sequence.true_counts()
    assert counts["forward"] > 0 and counts["backward"] > 0


def test_each_vehicle_crosses_at_most_once():
    sequence = simulate(short_config(seconds=120.0))
    ids = [vehicle for _, vehicle, _, _ in sequence.crossings]
    assert len(ids) == len(set(ids))


def test_vehicles_in_a_lane_never_overlap():
    sequence = simulate(short_config(seconds=60.0, arrivals_per_minute=60.0))
    for states in sequence.frames[::10]:
        centers = np.array([s.center for s in states]).reshape(-1, 2)
        if len(centers) < 2:
            continue
        distance = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        np.fill_diagonal(distance, np.inf)
        assert distance.min() > 8.0


def test_occluder_hides_vehicles_in_its_band():
    sequence = simulate(short_config(occluder=(500.0, 640.0)))
    states = [s for frame in sequence.frames for s in frame]
    assert any(not s.visible for s in states)
    assert any(s.visible for s in states)
    detections = corrupt(sequence, DetectorModel(false_positives_per_frame=0.0), seed=0)
    for frame, det in zip(sequence.frames, detections, strict=True):
        visible = [s for s in frame if s.visible]
        assert len(det) <= len(visible)


def test_counter_recovers_true_count_from_clean_detections():
    """With no detector noise, the pipeline must reproduce the simulator's count exactly."""
    sequence = simulate(short_config(seconds=90.0, camera_jitter_px=0.0))
    clean = DetectorModel(
        high_recall=1.0,
        any_recall=1.0,
        hard_state_entry=0.0,
        center_noise_px=0.0,
        size_noise=0.0,
        false_positives_per_frame=0.0,
        wrong_label_rate=0.0,
    )
    tracker = ByteTracker()
    counter = LineCounter(*sequence.line)
    for i, detections in enumerate(corrupt(sequence, clean, seed=0)):
        counter.update(i, tracker.update(detections))
    expected = sequence.true_counts()
    assert sum(expected.values()) > 5
    totals = counter.direction_totals()
    assert totals["forward"] == pytest.approx(expected["forward"], abs=1)
    assert totals["backward"] == pytest.approx(expected["backward"], abs=1)
