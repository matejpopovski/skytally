import numpy as np
import pytest
import torch
import torchvision
from torch.nn import functional as F

from skytally.assign import BACKGROUND, IGNORE, TargetGrid, assign_targets
from skytally.boxes import pairwise_iou
from skytally.infer import InferConfig, network_candidates, suppress, to_candidates
from skytally.losses import detection_loss, sigmoid_focal_loss
from skytally.model import Detector, ModelConfig, decode_boxes, flatten_outputs, grid_points


def small_config(min_level=2):
    return ModelConfig(min_level=min_level, channels=32, head_depth=1, pretrained=False)


def test_grid_points_are_cell_centres():
    points = grid_points(2, 3, 4, "cpu")
    assert points[:4].tolist() == [[2, 2], [6, 2], [10, 2], [2, 6]]


def test_decode_zero_prediction_is_stride_sized_box_at_point():
    box = decode_boxes(torch.zeros(1, 4), torch.tensor([[10.0, 20.0]]), torch.tensor([[4.0]]))
    assert box.tolist() == [[8.0, 18.0, 12.0, 22.0]]


def test_decode_never_inverts_a_box():
    raw = torch.randn(100, 4) * 50
    box = decode_boxes(raw, torch.zeros(100, 2), torch.full((100, 1), 8.0))
    assert (box[:, 2] > box[:, 0]).all() and (box[:, 3] > box[:, 1]).all()


@pytest.mark.parametrize(("min_level", "points"), [(2, 8500), (3, 2100)])
def test_output_layout_matches_target_grid(min_level, points):
    config = small_config(min_level)
    model = Detector(config).eval()
    with torch.no_grad():
        outputs = model(torch.zeros(1, 3, 320, 320))
    grid = TargetGrid(320, 320, config.levels)
    assert grid.num_points == sum(o[0].shape[-2] * o[0].shape[-1] for o in outputs)
    cls, box, quality, pts, strides = flatten_outputs(outputs, model.strides)
    assert cls.shape == (1, grid.num_points, 10)
    assert box.shape == (1, grid.num_points, 4)
    assert quality.shape == (1, grid.num_points)
    assert pts.shape == (grid.num_points, 2)
    assert strides.shape == (grid.num_points, 1)
    assert grid.num_points == points


def test_flattened_points_align_with_assigned_cells():
    """A point's coordinate in the flat tensor must be the cell the assigner labelled."""
    config = small_config()
    grid = TargetGrid(320, 320, config.levels)
    b = np.array([[100, 60, 112, 70.0]], dtype=np.float32)
    targets = assign_targets(b, np.array([1]), np.empty((0, 4), np.float32), grid)
    model = Detector(config).eval()
    with torch.no_grad():
        _, _, _, points, _ = flatten_outputs(model(torch.zeros(1, 3, 320, 320)), model.strides)
    positives = points[torch.from_numpy(targets.classes >= 0)]
    assert len(positives) > 0
    assert (positives[:, 0] > 100).all() and (positives[:, 0] < 112).all()
    assert (positives[:, 1] > 60).all() and (positives[:, 1] < 70).all()


def test_frozen_stages_receive_no_gradient_and_later_stages_do():
    model = Detector(ModelConfig(channels=16, head_depth=1, pretrained=False, frozen_stages=1))
    outputs = model(torch.zeros(1, 3, 64, 64))
    sum(o.sum() for level in outputs for o in level).backward()
    assert all(p.grad is None for p in model.stem.parameters())
    assert all(p.grad is None for p in model.stages[0].parameters())
    assert any(p.grad is not None for p in model.stages[1].parameters())


def test_classification_head_starts_at_prior():
    model = Detector(small_config()).eval()
    with torch.no_grad():
        cls = model(torch.zeros(1, 3, 320, 320))[0][0]
    assert torch.sigmoid(cls).mean().item() == pytest.approx(0.01, abs=0.005)


def test_focal_loss_matches_torchvision():
    logits = torch.randn(64, 10)
    targets = (torch.rand(64, 10) > 0.8).float()
    expected = torchvision.ops.sigmoid_focal_loss(logits, targets, 0.25, 2.0)
    assert torch.allclose(sigmoid_focal_loss(logits, targets), expected, atol=1e-6)


def test_focal_loss_downweights_easy_negatives():
    easy = sigmoid_focal_loss(torch.tensor([-5.0]), torch.tensor([0.0]))
    hard = sigmoid_focal_loss(torch.tensor([2.0]), torch.tensor([0.0]))
    ce_ratio = F.binary_cross_entropy_with_logits(
        torch.tensor([2.0]), torch.tensor([0.0])
    ) / F.binary_cross_entropy_with_logits(torch.tensor([-5.0]), torch.tensor([0.0]))
    assert hard / easy > 100 * ce_ratio


def make_loss_inputs():
    num_points, num_classes = 6, 3
    cls = torch.zeros(1, num_points, num_classes, requires_grad=True)
    box = torch.zeros(1, num_points, 4, requires_grad=True)
    quality = torch.zeros(1, num_points, requires_grad=True)
    points = torch.tensor([[float(i * 8 + 4), 4.0] for i in range(num_points)])
    strides = torch.full((num_points, 1), 8.0)
    target_boxes = torch.zeros(1, num_points, 4)
    target_boxes[0, 0] = torch.tensor([0.0, 0.0, 8.0, 8.0])
    target_classes = torch.tensor([[1, BACKGROUND, IGNORE, BACKGROUND, BACKGROUND, BACKGROUND]])
    return cls, box, quality, points, strides, target_classes, target_boxes


def test_ignored_points_receive_no_gradient():
    cls, box, quality, points, strides, classes, boxes = make_loss_inputs()
    loss = detection_loss(cls, box, quality, points, strides, classes, boxes, 1.0)
    loss["total"].backward()
    assert cls.grad[0, 2].abs().sum() == 0
    assert cls.grad[0, 1].abs().sum() > 0


def test_box_loss_only_touches_positive_points():
    cls, box, quality, points, strides, classes, boxes = make_loss_inputs()
    loss = detection_loss(cls, box, quality, points, strides, classes, boxes, 1.0)
    loss["box"].backward(retain_graph=True)
    assert box.grad[0, 1:].abs().sum() == 0
    assert quality.grad is None or quality.grad[0, 1:].abs().sum() == 0


def test_perfect_prediction_has_zero_box_loss():
    cls, box, quality, points, strides, classes, boxes = make_loss_inputs()
    loss = detection_loss(cls, box, quality, points, strides, classes, boxes, 1.0)
    assert loss["box"].item() == pytest.approx(0.0, abs=1e-4)


def test_model_overfits_one_scene_on_cpu():
    """End-to-end: targets, losses and post-processing fit a trivially learnable image."""
    torch.manual_seed(0)
    config = small_config()
    model = Detector(config)
    grid = TargetGrid(160, 160, config.levels)
    image = torch.full((1, 3, 160, 160), -1.5)
    objects = np.array([[40, 40, 56, 52], [100, 90, 120, 106]], dtype=np.float32)
    for x0, y0, x1, y1 in objects.astype(int):
        image[..., y0:y1, x0:x1] = 1.5
    labels = np.array([3, 3])
    targets = assign_targets(objects, labels, np.empty((0, 4), np.float32), grid)
    target_classes = torch.from_numpy(targets.classes)[None]
    target_boxes = torch.from_numpy(targets.boxes)[None]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    model.freeze_backbone_norm()
    for _ in range(150):
        cls, box, quality, points, strides = flatten_outputs(model(image), model.strides)
        losses = detection_loss(
            cls, box, quality, points, strides, target_classes, target_boxes, 10.0
        )
        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

    model.eval()
    infer = InferConfig(long_side=160, score_threshold=0.2)
    b, s, c = network_candidates(model, image, infer)
    found = suppress(to_candidates(b, s, c, 1.0, (160, 160), infer), infer)
    assert set(found.labels.tolist()) == {3}
    assert pairwise_iou(found.boxes, objects).max(axis=0).min() > 0.5
