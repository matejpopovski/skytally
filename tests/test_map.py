"""Hand-computed cases for the mAP implementation."""

import numpy as np
import pytest

from skytally.coco_map import EvalParams, ImageDetections, ImageTruth, evaluate


def truth(boxes, labels=None, crowd=None):
    labels = [0] * len(boxes) if labels is None else labels
    return ImageTruth(np.array(boxes, dtype=float).reshape(-1, 4), labels, crowd)


def dets(boxes, scores, labels=None):
    labels = [0] * len(boxes) if labels is None else labels
    return ImageDetections(np.array(boxes, dtype=float).reshape(-1, 4), scores, labels)


def test_perfect_detections_score_one():
    gt = [[10, 10, 20, 20], [50, 50, 30, 30]]
    result = evaluate([truth(gt)], [dets(gt, [0.9, 0.8])])
    assert result.ap() == pytest.approx(1.0)
    assert result.ap(iou=0.5) == pytest.approx(1.0)
    assert result.ar() == pytest.approx(1.0)


def test_no_detections_scores_zero():
    result = evaluate([truth([[0, 0, 10, 10]])], [dets([], [])], class_ids=[0])
    assert result.ap() == 0.0
    assert result.ar() == 0.0


def test_no_ground_truth_is_undefined_not_zero():
    result = evaluate([truth([])], [dets([[0, 0, 10, 10]], [0.9])], class_ids=[0])
    assert result.ap() == -1.0


def test_higher_ranked_false_positive_gives_half_precision_plateau():
    # Rank 1 is a false positive, rank 2 a hit. Precision at any recall above 0 is 0.5,
    # and the recall threshold of exactly 0 also lands on rank 2.
    gt = [[10, 10, 20, 20]]
    predictions = [[100, 100, 20, 20], [10, 10, 20, 20]]
    result = evaluate([truth(gt)], [dets(predictions, [0.9, 0.8])])
    assert result.ap(iou=0.5) == pytest.approx(0.5)


def test_lower_ranked_false_positive_does_not_change_ap():
    gt = [[10, 10, 20, 20]]
    predictions = [[10, 10, 20, 20], [100, 100, 20, 20]]
    result = evaluate([truth(gt)], [dets(predictions, [0.9, 0.8])])
    assert result.ap(iou=0.5) == pytest.approx(1.0)


def test_half_recall_with_perfect_precision_averages_recall_grid():
    # Recall reaches 0.5 with precision 1. Grid points 0.00..0.50 hit, 51 of 101.
    gt = [[0, 0, 10, 10], [100, 100, 10, 10]]
    result = evaluate([truth(gt)], [dets([[0, 0, 10, 10]], [0.9])])
    assert result.ap(iou=0.5) == pytest.approx(51 / 101)
    assert result.ar(iou=0.5) == pytest.approx(0.5)


def test_precision_envelope_uses_later_higher_precision():
    # Hit, miss, hit over two ground truths: precision 1, 1/2, 2/3 at recall .5, .5, 1.
    # The envelope lifts recall in (0.5, 1] to 2/3.
    gt = [[0, 0, 10, 10], [100, 100, 10, 10]]
    predictions = [[0, 0, 10, 10], [200, 200, 10, 10], [100, 100, 10, 10]]
    result = evaluate([truth(gt)], [dets(predictions, [0.9, 0.8, 0.7])])
    assert result.ap(iou=0.5) == pytest.approx((51 * 1.0 + 50 * (2 / 3)) / 101)


def test_duplicate_detection_is_a_false_positive():
    gt = [[0, 0, 10, 10]]
    result = evaluate([truth(gt)], [dets([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8])])
    # Recall hits 1 on the first detection, so the duplicate does not lower AP here...
    assert result.ap(iou=0.5) == pytest.approx(1.0)
    # ...but ranked above a second object it does.
    gt = [[0, 0, 10, 10], [100, 100, 10, 10]]
    predictions = [[0, 0, 10, 10], [0, 0, 10, 10], [100, 100, 10, 10]]
    result = evaluate([truth(gt)], [dets(predictions, [0.9, 0.8, 0.7])])
    assert result.ap(iou=0.5) == pytest.approx((51 + 50 * (2 / 3)) / 101)


def test_iou_threshold_boundary_is_inclusive():
    # Detection overlaps exactly half of a 2:1 union: IoU is 0.5 exactly.
    gt = [[0, 0, 20, 10]]
    det = [[0, 0, 10, 10]]
    result = evaluate([truth(gt)], [dets(det, [0.9])])
    assert result.ap(iou=0.5) == pytest.approx(1.0)
    assert result.ap(iou=0.55) == 0.0


def test_ap_averages_iou_thresholds():
    # IoU of 0.8 passes .50 through .80: 7 of the 10 thresholds.
    gt = [[0, 0, 100, 10]]
    det = [[0, 0, 80, 10]]
    result = evaluate([truth(gt)], [dets(det, [0.9])])
    assert result.ap() == pytest.approx(0.7)


def test_classes_are_matched_separately_and_averaged():
    gt = [[0, 0, 10, 10], [50, 50, 10, 10]]
    # Class 1 detection sits on class 0's object, so class 1 gets nothing.
    predictions = [[0, 0, 10, 10], [0, 0, 10, 10]]
    result = evaluate([truth(gt, labels=[0, 1])], [dets(predictions, [0.9, 0.8], labels=[0, 1])])
    assert result.ap(iou=0.5, class_id=0) == pytest.approx(1.0)
    assert result.ap(iou=0.5, class_id=1) == 0.0
    assert result.ap(iou=0.5) == pytest.approx(0.5)


def test_detections_across_images_are_ranked_jointly():
    images = [truth([[0, 0, 10, 10]]), truth([[0, 0, 10, 10]])]
    predictions = [dets([[50, 50, 10, 10]], [0.9]), dets([[0, 0, 10, 10]], [0.8])]
    result = evaluate(images, predictions)
    # False positive from image 1 outranks the hit in image 2, and image 1 is missed.
    assert result.ap(iou=0.5) == pytest.approx(51 * 0.5 / 101)


def test_crowd_region_absorbs_many_detections_without_penalty():
    gt = [[0, 0, 100, 100], [200, 200, 10, 10]]
    crowd = [True, False]
    predictions = [[10, 10, 5, 5], [30, 30, 5, 5], [60, 60, 5, 5], [200, 200, 10, 10]]
    result = evaluate([truth(gt, crowd=crowd)], [dets(predictions, [0.9, 0.8, 0.7, 0.6])])
    assert result.ap(iou=0.5) == pytest.approx(1.0)


def test_crowd_iou_uses_detection_area():
    # 60% of the detection lies in the crowd box, which passes .50 and .55 and .60.
    gt = [[0, 0, 100, 100], [500, 500, 10, 10]]
    predictions = [[-40, 0, 100, 100], [500, 500, 10, 10]]
    result = evaluate([truth(gt, crowd=[True, False])], [dets(predictions, [0.9, 0.8])])
    # Above IoU 0.6 the first detection is a false positive ranked above the hit.
    expected = [1.0, 1.0, 1.0] + [0.5] * 7
    assert result.ap() == pytest.approx(np.mean(expected))


def test_crowd_does_not_count_towards_recall():
    gt = [[0, 0, 100, 100]]
    result = evaluate([truth(gt, crowd=[True])], [dets([[0, 0, 100, 100]], [0.9])], class_ids=[0])
    assert result.ap() == -1.0


def test_real_ground_truth_beats_better_overlapping_crowd():
    gt = [[0, 0, 20, 20], [0, 0, 10, 10]]
    # Exact match on the crowd box, but IoU 0.5 clear on the real one.
    predictions = [[0, 0, 15, 10]]
    result = evaluate([truth(gt, crowd=[True, False])], [dets(predictions, [0.9])])
    assert result.ap(iou=0.5) == pytest.approx(1.0)


def test_area_range_ignores_outside_truth_and_discounts_unmatched_detections():
    params = EvalParams()
    small_gt = [0, 0, 10, 10]
    large_gt = [100, 100, 200, 200]
    stray_large = [400, 400, 200, 200]
    gt = [small_gt, large_gt]
    predictions = [stray_large, large_gt, small_gt]
    result = evaluate([truth(gt)], [dets(predictions, [0.9, 0.8, 0.7])], params)
    # In "small" the large detections are all discounted, leaving a clean hit.
    assert result.ap(iou=0.5, area="small") == pytest.approx(1.0)
    # In "large" the stray false positive ranks first and the small hit is discounted.
    assert result.ap(iou=0.5, area="large") == pytest.approx(0.5)


def test_max_dets_truncates_by_score():
    gt = [[0, 0, 10, 10], [100, 100, 10, 10], [200, 200, 10, 10]]
    params = EvalParams(max_dets=(1, 2, 3))
    result = evaluate([truth(gt)], [dets(gt, [0.3, 0.2, 0.1])], params)
    assert result.ar(iou=0.5, max_dets=1) == pytest.approx(1 / 3)
    assert result.ar(iou=0.5, max_dets=2) == pytest.approx(2 / 3)
    assert result.ar(iou=0.5, max_dets=3) == pytest.approx(1.0)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="prediction sets"):
        evaluate([truth([])], [])
