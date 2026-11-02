"""Our evaluator and pycocotools must agree on the same predictions."""

import contextlib
import io

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from skytally.coco_map import (
    VISDRONE_AREAS,
    EvalParams,
    ImageDetections,
    ImageTruth,
    evaluate,
)


def random_scene(seed, images=25, classes=4):
    """Boxes from tiny to large, with noisy duplicates, strays and crowd regions."""
    rng = np.random.default_rng(seed)
    truths, detections = [], []
    for _ in range(images):
        n = rng.integers(0, 25)
        size = np.exp(rng.uniform(np.log(4), np.log(200), (n, 2)))
        xy = rng.uniform(0, 600, (n, 2))
        boxes = np.concatenate([xy, size], axis=1).round(2)
        labels = rng.integers(0, classes, n)
        crowd = rng.random(n) < 0.08

        keep = rng.random(n) < 0.8
        noisy = boxes[keep].copy()
        noisy[:, :2] += rng.normal(0, 2, (keep.sum(), 2))
        noisy[:, 2:] *= rng.uniform(0.75, 1.25, (keep.sum(), 2))
        dup = noisy[rng.random(len(noisy)) < 0.2]
        dup_labels = labels[keep][: len(dup)]
        stray_n = rng.integers(0, 10)
        stray = np.concatenate(
            [rng.uniform(0, 600, (stray_n, 2)), rng.uniform(4, 100, (stray_n, 2))], axis=1
        )
        det_boxes = np.concatenate([noisy, dup, stray]).round(3)
        det_labels = np.concatenate([labels[keep], dup_labels, rng.integers(0, classes, stray_n)])
        # Rounded scores create ties, which exercise the stable sort.
        scores = rng.random(len(det_boxes)).round(2)
        truths.append(ImageTruth(boxes, labels, crowd))
        detections.append(ImageDetections(det_boxes, scores, det_labels))
    return truths, detections


def run_pycocotools(truths, detections, params, classes):
    images, annotations, results = [], [], []
    ann_id = 1
    for image_id, (truth, det) in enumerate(zip(truths, detections, strict=True), start=1):
        images.append({"id": image_id, "width": 2000, "height": 2000})
        for box, label, crowd in zip(truth.boxes, truth.labels, truth.crowd, strict=True):
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "area": float(box[2] * box[3]),
                    "iscrowd": int(crowd),
                }
            )
            ann_id += 1
        for box, score, label in zip(det.boxes, det.scores, det.labels, strict=True):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(score),
                }
            )
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": str(c)} for c in classes],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        predictions = coco.loadRes(results)
        evaluator = COCOeval(coco, predictions, "bbox")
        evaluator.params.catIds = list(classes)
        evaluator.params.imgIds = [i["id"] for i in images]
        evaluator.params.maxDets = list(params.max_dets)
        evaluator.params.areaRng = [list(v) for v in params.areas.values()]
        evaluator.params.areaRngLbl = list(params.areas)
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


@pytest.mark.parametrize("seed", range(8))
def test_full_precision_and_recall_tensors_match(seed):
    truths, detections = random_scene(seed)
    classes = [0, 1, 2, 3]
    params = EvalParams()
    ours = evaluate(truths, detections, params, classes)
    theirs = run_pycocotools(truths, detections, params, classes)
    np.testing.assert_allclose(ours.precision, theirs.eval["precision"], atol=1e-12)
    np.testing.assert_allclose(ours.recall, theirs.eval["recall"], atol=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_visdrone_protocol_with_tiny_bucket_and_500_dets(seed):
    truths, detections = random_scene(seed + 100, images=15)
    classes = [0, 1, 2, 3]
    params = EvalParams(areas=VISDRONE_AREAS, max_dets=(1, 10, 100, 500))
    ours = evaluate(truths, detections, params, classes)
    theirs = run_pycocotools(truths, detections, params, classes)
    np.testing.assert_allclose(ours.precision, theirs.eval["precision"], atol=1e-12)
    np.testing.assert_allclose(ours.recall, theirs.eval["recall"], atol=1e-12)
