import numpy as np
import pytest

from skytally.assign import (
    BACKGROUND,
    IGNORE,
    TargetGrid,
    assign_targets,
    coverage,
    pyramid_levels,
)


def grid(size=64, min_level=2):
    return TargetGrid(size, size, pyramid_levels(min_level))


def boxes(*rows):
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def empty():
    return np.empty((0, 4), dtype=np.float32)


def level_map(array, g, level):
    start, stop = g.offsets[level], g.offsets[level + 1]
    return array[start:stop].reshape(g.shapes[level])


def test_levels_have_doubling_size_bands_and_finest_takes_everything_small():
    levels = pyramid_levels(2)
    assert [spec.stride for spec in levels] == [4, 8, 16, 32]
    assert levels[0].min_size == 0
    assert levels[0].max_size == levels[1].min_size == 64
    assert levels[-1].max_size == float("inf")


def test_level_selection_by_longer_side():
    g = grid(640)
    assert g.level_for(np.array([10.0, 63.9, 64.0, 200.0, 600.0])).tolist() == [0, 0, 1, 2, 3]


def test_no_objects_is_all_background():
    g = grid()
    targets = assign_targets(empty(), np.empty(0, np.int64), empty(), g)
    assert (targets.classes == BACKGROUND).all()


def test_small_object_positive_points_are_inside_and_labelled():
    g = grid()
    box = boxes((10, 10, 20, 22))
    targets = assign_targets(box, np.array([3]), empty(), g)
    classes = level_map(targets.classes, g, 0)
    rows, cols = np.nonzero(classes == 3)
    xs, ys = (cols + 0.5) * 4, (rows + 0.5) * 4
    assert len(xs) > 0
    assert (xs > 10).all() and (xs < 20).all() and (ys > 10).all() and (ys < 22).all()
    positive = targets.classes >= 0
    np.testing.assert_array_equal(targets.boxes[positive], np.repeat(box, positive.sum(), axis=0))


def test_center_radius_limits_positive_area_on_large_box():
    g = grid()
    targets = assign_targets(boxes((0, 0, 40, 40)), np.array([1]), empty(), g, center_radius=1.0)
    classes = level_map(targets.classes, g, 0)
    rows, cols = np.nonzero(classes == 1)
    assert np.abs((cols + 0.5) * 4 - 20).max() < 4
    assert np.abs((rows + 0.5) * 4 - 20).max() < 4


def test_object_between_grid_points_is_force_matched_to_nearest():
    g = TargetGrid(64, 64, pyramid_levels(3))
    # 1-pixel object whose interval contains no stride-8 cell centre.
    box = boxes((9, 9, 10, 10))
    forced = assign_targets(box, np.array([2]), empty(), g)
    assert np.count_nonzero(forced.classes == 2) == 1
    assert level_map(forced.classes, g, 0)[1, 1] == 2


def test_force_match_can_be_disabled():
    g = TargetGrid(64, 64, pyramid_levels(3))
    plain = assign_targets(boxes((9, 9, 10, 10)), np.array([2]), empty(), g, force_match=False)
    assert (plain.classes == BACKGROUND).all()


def test_smaller_object_wins_contested_points():
    g = grid()
    large = (8, 8, 40, 40)
    small = (16, 16, 26, 26)
    targets = assign_targets(boxes(large, small), np.array([0, 1]), empty(), g)
    classes = level_map(targets.classes, g, 0)
    # Point at (18, 18) lies in both boxes.
    assert classes[4, 4] == 1


def test_ignore_regions_mark_every_level_and_objects_override_them():
    g = grid()
    region = boxes((0, 0, 64, 64))
    targets = assign_targets(boxes((10, 10, 20, 20)), np.array([5]), region, g)
    assert (targets.classes[targets.classes < 0] == IGNORE).all()
    assert np.count_nonzero(targets.classes == 5) > 0


def test_box_outside_grid_keeps_nearest_point_in_bounds():
    g = grid()
    targets = assign_targets(boxes((70, 70, 71, 71)), np.array([0]), empty(), g)
    assert np.count_nonzero(targets.classes == 0) == 1


@pytest.mark.parametrize("seed", range(10))
def test_slice_assignment_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    g = TargetGrid(128, 128, pyramid_levels(2))
    xy = rng.uniform(0, 110, (6, 2))
    wh = rng.uniform(5, 18, (6, 2))
    b = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    labels = np.arange(6)
    targets = assign_targets(b, labels, empty(), g, force_match=False)

    expected = np.full(g.shapes[0], BACKGROUND)
    for i in np.argsort(-(wh[:, 0] * wh[:, 1]), kind="stable"):
        cx, cy = (b[i, :2] + b[i, 2:]) / 2
        for row in range(g.shapes[0][0]):
            for col in range(g.shapes[0][1]):
                x, y = (col + 0.5) * 4, (row + 0.5) * 4
                inside = b[i, 0] < x < b[i, 2] and b[i, 1] < y < b[i, 3]
                near = abs(x - cx) < 6 and abs(y - cy) < 6
                if inside and near:
                    expected[row, col] = i
    np.testing.assert_array_equal(level_map(targets.classes, g, 0), expected)


def test_coverage_counts_agree_with_assignment():
    g = TargetGrid(128, 128, pyramid_levels(3))
    b = boxes((10, 10, 40, 40), (50.5, 50.5, 51.5, 51.5))
    counts = coverage(b, g)
    assert counts[0] > 0
    assert counts[1] == 0
    single = assign_targets(b[:1], np.array([0]), empty(), g)
    assert counts[0] == np.count_nonzero(single.classes == 0)
