import numpy as np
import pytest
import torch
import torchvision

from skytally.boxes import aligned_giou, aligned_iou, batched_nms, nms, pairwise_iou


def test_pairwise_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(pairwise_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_pairwise_iou_degenerate_box_is_zero_not_nan():
    iou = pairwise_iou(np.array([[3, 3, 3, 3.0]]), np.array([[3, 3, 3, 3.0]]))
    assert iou[0, 0] == 0.0


def test_nms_suppresses_overlap_and_keeps_distant_box():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60.0]])
    scores = np.array([0.9, 0.8, 0.7])
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_nms_threshold_is_exclusive():
    # IoU of these two boxes is exactly 0.5.
    boxes = np.array([[0, 0, 3, 1], [1, 0, 4, 1.0]])
    assert nms(boxes, np.array([0.9, 0.8]), 0.5).tolist() == [0, 1]
    assert nms(boxes, np.array([0.9, 0.8]), 0.49).tolist() == [0]


def test_nms_empty_input():
    assert nms(np.empty((0, 4)), np.empty(0), 0.5).size == 0
    assert batched_nms(np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int), 0.5).size == 0


def test_batched_nms_keeps_overlapping_boxes_of_different_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10.0]])
    assert batched_nms(boxes, np.array([0.5, 0.9]), np.array([0, 1]), 0.5).tolist() == [1, 0]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("threshold", [0.3, 0.6])
def test_nms_agrees_with_torchvision(seed, threshold):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 200, (400, 2))
    wh = rng.uniform(4, 40, (400, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = rng.uniform(0, 1, 400)
    expected = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), threshold)
    assert nms(boxes, scores, threshold).tolist() == expected.tolist()


def test_aligned_iou_and_giou_known_values():
    a = torch.tensor([[0.0, 0, 10, 10], [0, 0, 10, 10]])
    b = torch.tensor([[0.0, 0, 10, 10], [20, 0, 30, 10]])
    assert aligned_iou(a, b).tolist() == pytest.approx([1.0, 0.0], abs=1e-5)
    # Hull is 30x10 and the union covers 200, so GIoU is 0 - 100/300.
    assert aligned_giou(a, b).tolist() == pytest.approx([1.0, -1 / 3], abs=1e-5)


def test_aligned_giou_matches_torchvision_on_random_boxes():
    torch.manual_seed(0)
    xy = torch.rand(64, 2) * 100
    wh = torch.rand(64, 2) * 40 + 1
    a = torch.cat([xy, xy + wh], dim=1).double()
    b = a + torch.randn(64, 4).double() * 2
    b[:, 2:] = torch.maximum(b[:, 2:], b[:, :2] + 0.5)
    ours = aligned_giou(a, b, eps=1e-12)
    theirs = torchvision.ops.generalized_box_iou(a, b).diagonal()
    assert torch.allclose(ours, theirs, atol=1e-6)
