"""Evaluate a checkpoint on VisDrone val with the VisDrone-style COCO protocol.

``--candidates`` caches pre-NMS output, so NMS settings can be swept without the GPU.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image

from skytally import CLASS_NAMES, NUM_CLASSES
from skytally.coco_map import VISDRONE_AREAS, EvalParams, ImageDetections, MapResult, evaluate
from skytally.data import load_split, to_truth
from skytally.infer import (
    Candidates,
    InferConfig,
    best_device,
    load_detector,
    network_candidates,
    prepare,
    suppress,
    to_candidates,
)

VISDRONE_PARAMS = EvalParams(areas=VISDRONE_AREAS, max_dets=(1, 10, 100, 500))


def _load(path: Path, long_side: int):
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    tensor, scale = prepare(image, long_side)
    return tensor, scale, image.size


def run_network(model, paths, config, device, on_image=None) -> list[Candidates]:
    """Candidates for every image, with decoding on threads so the GPU never waits on JPEGs."""
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for tensor, scale, size in pool.map(lambda p: _load(p, config.long_side), paths):
            boxes, scores, labels = network_candidates(model, tensor[None].to(device), config)
            results.append(to_candidates(boxes, scores, labels, scale, size, config))
            if on_image is not None:
                on_image()
    return results


def to_coco_detections(detections: Candidates) -> ImageDetections:
    boxes = detections.boxes.copy()
    boxes[:, 2:] -= boxes[:, :2]
    return ImageDetections(boxes, detections.scores, detections.labels)


def score(annotations, detections: list[Candidates]) -> MapResult:
    truths = [to_truth(a, NUM_CLASSES) for a in annotations]
    coco = [to_coco_detections(d) for d in detections]
    return evaluate(truths, coco, VISDRONE_PARAMS, class_ids=range(NUM_CLASSES))


def report(result: MapResult) -> dict:
    per_class = {
        name: {
            "AP": result.ap(class_id=i),
            "AP50": result.ap(iou=0.5, class_id=i),
            "AR": result.ar(class_id=i),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    return {"overall": result.summary(), "per_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--long-side", type=int, default=1536)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--candidates", type=Path, help="pickle cache of pre-NMS output")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    images, annotations = load_split(args.data_dir / "val")
    config = InferConfig(long_side=args.long_side, nms_iou=args.nms_iou)
    device = best_device()

    if args.candidates and args.candidates.exists():
        candidates = pickle.loads(args.candidates.read_bytes())
    else:
        model = load_detector(args.checkpoint, device)
        started = time.perf_counter()
        candidates = run_network(model, images, config, device)
        print(f"network pass: {time.perf_counter() - started:.1f} s for {len(images)} images")
        if args.candidates:
            args.candidates.parent.mkdir(parents=True, exist_ok=True)
            args.candidates.write_bytes(pickle.dumps(candidates))

    detections = [suppress(c, config) for c in candidates]
    started = time.perf_counter()
    result = score(annotations, detections)
    print(f"evaluation: {time.perf_counter() - started:.1f} s")

    payload = {
        "checkpoint": args.checkpoint.name,
        "protocol": "COCO AP with ignored regions as crowd boxes, max 500 detections per image",
        "infer_config": asdict(replace(config)),
        "device": str(device),
        **report(result),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
