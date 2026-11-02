#!/usr/bin/env python3
"""Score the same validation predictions with our evaluator and with pycocotools.

Reads the pre-NMS cache written by ``skytally.evaluate --candidates``, so both evaluators
see identical detections. Writes the largest disagreement over every entry of the
precision and recall tensors, plus each evaluator's runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from skytally import NUM_CLASSES
from skytally.data import load_split, to_truth
from skytally.evaluate import VISDRONE_PARAMS, score, to_coco_detections
from skytally.infer import InferConfig, suppress


def as_coco(truths, detections):
    images, annotations, results = [], [], []
    ann_id = 1
    for image_id, (truth, det) in enumerate(zip(truths, detections, strict=True), start=1):
        images.append({"id": image_id, "width": 2000, "height": 1500})
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
        for box, value, label in zip(det.boxes, det.scores, det.labels, strict=True):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(value),
                }
            )
    categories = [{"id": c, "name": str(c)} for c in range(NUM_CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--output", type=Path, default=Path("results/map_crosscheck.json"))
    args = parser.parse_args()

    _, annotations = load_split(args.data_dir / "val")
    config = InferConfig()
    detections = [suppress(c, config) for c in pickle.loads(args.candidates.read_bytes())]

    started = time.perf_counter()
    ours = score(annotations, detections)
    ours_seconds = time.perf_counter() - started

    truths = [to_truth(a, NUM_CLASSES) for a in annotations]
    dataset, results = as_coco(truths, [to_coco_detections(d) for d in detections])
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(results), "bbox")
        evaluator.params.imgIds = [i["id"] for i in dataset["images"]]
        evaluator.params.catIds = list(range(NUM_CLASSES))
        evaluator.params.maxDets = list(VISDRONE_PARAMS.max_dets)
        evaluator.params.areaRng = [list(v) for v in VISDRONE_PARAMS.areas.values()]
        evaluator.params.areaRngLbl = list(VISDRONE_PARAMS.areas)
        evaluator.evaluate()
        evaluator.accumulate()
    theirs_seconds = time.perf_counter() - started

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    defined = precision > -1
    their_ap = float(precision[..., 1, -1][precision[..., 1, -1] > -1].mean())
    payload = {
        "images": len(annotations),
        "detections": len(results),
        "ap_ours": ours.ap(),
        "ap_pycocotools": their_ap,
        "ap50_ours": ours.ap(iou=0.5),
        "ap50_pycocotools": float(precision[0, :, :, 1, -1][precision[0, :, :, 1, -1] > -1].mean()),
        "max_abs_difference_precision": float(np.abs(ours.precision - precision).max()),
        "max_abs_difference_recall": float(np.abs(ours.recall - recall).max()),
        "tensor_entries_compared": int(defined.size + recall.size),
        "seconds_ours": ours_seconds,
        "seconds_pycocotools": theirs_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
