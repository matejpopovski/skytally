from types import SimpleNamespace

import numpy as np
import pytest

from skytally.track.byte import ByteTracker, Detection, TrackerConfig, expand, match
from skytally.track.counter import LineCounter
from skytally.track.kalman import BoxFilter
from skytally.track.metrics import FrameBoxes, evaluate_tracking


def box_at(cx, cy, w=12, h=8):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def frame(*items):
    """Detections from ``(box, score)`` pairs."""
    boxes = [b for b, _ in items]
    scores = [s for _, s in items]
    return Detection(np.array(boxes).reshape(-1, 4), scores, [3] * len(items))


def test_kalman_learns_velocity_and_predicts_forward():
    f = BoxFilter(np.array([0.0, 0, 12, 8]))
    for t in range(1, 15):
        f.predict()
        f.update(np.array([5.0 * t, 2.0 * t, 12, 8]))
    assert f.velocity == pytest.approx([5, 2], abs=0.3)
    x = f.mean[0]
    f.predict()
    assert f.mean[0] == pytest.approx(x + 5, abs=0.3)


def test_kalman_noise_has_a_pixel_floor_for_tiny_boxes():
    f = BoxFilter(np.array([0.0, 0, 8, 8]))
    assert f._position_std(8.0) == f.config.min_position_std
    assert f._position_std(200.0) == pytest.approx(10.0)


def test_kalman_covariance_stays_symmetric_positive_definite():
    f = BoxFilter(np.array([0.0, 0, 12, 8]))
    rng = np.random.default_rng(0)
    for _ in range(300):
        f.predict()
        f.update(f.mean[:4] + rng.normal(0, 2, 4))
    assert np.allclose(f.covariance, f.covariance.T, atol=1e-6)
    assert np.linalg.eigvalsh(f.covariance).min() > 0


def test_expand_grows_each_side_by_fraction_of_size():
    assert expand(np.array([[10, 10, 20, 30.0]]), 0.5).tolist() == [[5, 0, 25, 40]]


def test_buffer_links_a_box_with_no_plain_overlap():
    a = np.array([box_at(0, 0)])
    b = np.array([box_at(16, 0)])
    assert match(a, b, 0.15, 0.0)[0] == []
    assert match(a, b, 0.15, 0.5)[0] == [(0, 0)]


def test_assignment_is_global_rather_than_greedy():
    tracks = np.array([box_at(0, 0, 10, 10), box_at(6, 0, 10, 10)])
    dets = np.array([box_at(3, 0, 10, 10), box_at(-4, 0, 10, 10)])
    matches, _, _ = match(tracks, dets, 0.1, 0.0)
    assert sorted(matches) == [(0, 1), (1, 0)]


def test_track_keeps_identity_across_constant_motion():
    tracker = ByteTracker()
    ids = set()
    for t in range(30):
        tracks = tracker.update(frame((box_at(10 + 6 * t, 50), 0.8)))
        ids |= {track.track_id for track in tracks}
    assert ids == {1}


def test_low_score_detection_rescues_a_track():
    tracker = ByteTracker(TrackerConfig(max_lost=0))
    for t in range(5):
        tracker.update(frame((box_at(10 + 2 * t, 50), 0.8)))
    survivors = tracker.update(frame((box_at(20, 50), 0.2)))
    assert [t.track_id for t in survivors] == [1]


def test_low_score_detection_is_ignored_when_the_second_pass_is_off():
    tracker = ByteTracker(TrackerConfig(max_lost=0, use_low_scores=False))
    for t in range(5):
        tracker.update(frame((box_at(10 + 2 * t, 50), 0.8)))
    assert tracker.update(frame((box_at(20, 50), 0.2))) == []


def test_low_score_detection_never_starts_a_track():
    tracker = ByteTracker()
    for _ in range(5):
        assert tracker.update(frame((box_at(30, 30), 0.2))) == []
    assert tracker.tracks == []


def test_track_coasts_through_a_gap_and_reacquires():
    tracker = ByteTracker(TrackerConfig(max_lost=10))
    for t in range(8):
        tracker.update(frame((box_at(10 + 8 * t, 50), 0.8)))
    for _ in range(5):
        tracker.update(Detection(np.empty((0, 4)), [], []))
    tracks = tracker.update(frame((box_at(10 + 8 * 13, 50), 0.8)))
    assert [t.track_id for t in tracks] == [1]


def test_lost_track_expires():
    tracker = ByteTracker(TrackerConfig(max_lost=3))
    for _t in range(4):
        tracker.update(frame((box_at(10, 50), 0.8)))
    for _ in range(5):
        tracker.update(Detection(np.empty((0, 4)), [], []))
    assert tracker.tracks == []


def test_one_frame_false_positive_is_never_reported():
    tracker = ByteTracker()
    assert tracker.update(frame((box_at(30, 30), 0.9))) == []
    tracker.update(Detection(np.empty((0, 4)), [], []))
    assert tracker.tracks == []


def test_crossing_paths_keep_their_identities():
    tracker = ByteTracker()
    first, second = None, None
    for t in range(40):
        a, b = 10 + 4 * t, 170 - 4 * t
        tracks = tracker.update(frame((box_at(a, 40), 0.8), (box_at(b, 60), 0.8)))
        by_row = {round(track.center[1] / 20): track.track_id for track in tracks}
        if t == 5:
            first, second = by_row[2], by_row[3]
        if t > 5:
            assert by_row[2] == first and by_row[3] == second


def walker(track_id, x, y=50.0, label=3):
    return SimpleNamespace(track_id=track_id, center=np.array([x, y]), stable_label=label)


def vertical_line(**kwargs):
    return LineCounter((100.0, 0.0), (100.0, 200.0), **kwargs)


def test_counter_counts_one_crossing_with_direction():
    counter = vertical_line()
    for f, x in enumerate(range(80, 125, 5)):
        counter.update(f, [walker(1, float(x))])
    assert sum(counter.totals().values()) == 1
    forward = counter.events[0].direction
    other = vertical_line()
    for f, x in enumerate(range(120, 75, -5)):
        other.update(f, [walker(2, float(x))])
    assert other.events[0].direction != forward


def test_jitter_on_the_line_does_not_count():
    counter = vertical_line(dead_band=4.0)
    for f, x in enumerate([90, 102, 98, 103, 97, 102, 98]):
        counter.update(f, [walker(1, float(x))])
    assert not counter.events


def test_jitter_across_the_line_counts_once_per_direction():
    counter = vertical_line(dead_band=2.0)
    for f, x in enumerate([90, 110, 90, 110, 90, 110]):
        counter.update(f, [walker(1, float(x))])
    assert sorted(e.direction for e in counter.events) == ["backward", "forward"]


def test_track_lost_at_the_line_still_counts_on_reappearing():
    counter = vertical_line()
    counter.update(0, [walker(1, 90.0)])
    counter.update(1, [])
    counter.update(2, [walker(1, 120.0)])
    assert len(counter.events) == 1


def test_new_track_past_the_line_is_not_counted():
    counter = vertical_line()
    counter.update(0, [walker(5, 120.0)])
    counter.update(1, [walker(5, 130.0)])
    assert not counter.events


def test_crossing_outside_the_segment_is_ignored():
    counter = vertical_line(margin=5.0)
    counter.update(0, [walker(1, 90.0, y=500.0)])
    counter.update(1, [walker(1, 110.0, y=500.0)])
    assert not counter.events


def test_counts_group_by_label():
    counter = vertical_line()
    counter.update(0, [walker(1, 90.0, label=3), walker(2, 90.0, y=80, label=8)])
    counter.update(1, [walker(1, 110.0, label=3), walker(2, 110.0, y=80, label=8)])
    totals = counter.totals()
    assert {label for _, label in totals} == {3, 8}


def test_zero_length_line_is_rejected():
    with pytest.raises(ValueError, match="zero length"):
        LineCounter((1.0, 1.0), (1.0, 1.0))


def fb(ids, boxes):
    return FrameBoxes(np.array(ids, dtype=int), np.array(boxes, dtype=float).reshape(-1, 4))


def test_perfect_tracking_scores_one():
    truth = [fb([1, 2], [box_at(10, 10), box_at(50, 50)]) for _ in range(5)]
    scores = evaluate_tracking(truth, truth)
    assert scores.mota == 1.0
    assert scores.idf1 == 1.0
    assert scores.id_switches == 0


def test_identity_switch_is_counted_once_and_lowers_idf1():
    box = [box_at(10, 10)]
    truth = [fb([1], box) for _ in range(10)]
    pred = [fb([7 if t < 5 else 8], box) for t in range(10)]
    scores = evaluate_tracking(truth, pred)
    assert scores.id_switches == 1
    assert scores.mota == pytest.approx(0.9)
    assert scores.idf1 == pytest.approx(0.5)


def test_misses_and_false_positives_enter_mota():
    truth = [fb([1], [box_at(10, 10)]), fb([1], [box_at(10, 10)])]
    pred = [fb([1], [box_at(10, 10)]), fb([9], [box_at(100, 100)])]
    scores = evaluate_tracking(truth, pred)
    assert scores.false_negatives == 1
    assert scores.false_positives == 1
    assert scores.mota == pytest.approx(0.0)


def test_gap_does_not_count_as_a_switch():
    box = [box_at(10, 10)]
    truth = [fb([1], box) for _ in range(3)]
    pred = [fb([7], box), fb([], []), fb([7], box)]
    scores = evaluate_tracking(truth, pred)
    assert scores.id_switches == 0
    assert scores.false_negatives == 1
