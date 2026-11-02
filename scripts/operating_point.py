#!/usr/bin/env python3
"""Measure the detector's vehicle precision and recall across score thresholds.

The tracker's thresholds are read off this curve, and the simulated detector stand-in
takes its rates from it, so the tracker ablation runs at a measured detection quality.
Class is ignored within the vehicle group, because a tracker only needs a box. Unlabelled
real objects count as false positives here, which makes the clutter figure pessimistic.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from skytally import CLASS_NAMES
from skytally.boxes import pairwise_iou
from skytally.data import load_split
from skytally.infer import InferConfig, suppress
from skytally.track.byte import TrackerConfig

VEHICLES = [CLASS_NAMES.index(n) for n in ("car", "van", "truck", "bus")]
MATCH_IOU = 0.3
THRESHOLDS = (0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--output", type=Path, default=Path("results/operating_point.json"))
    args = parser.parse_args()

    images, annotations = load_split(args.data_dir / "val")
    candidates = pickle.loads(args.candidates.read_bytes())
    config = InferConfig()
    tracker = TrackerConfig()

    objects = 0
    megapixels = 0.0
    hits = dict.fromkeys(THRESHOLDS, 0)
    true_positives = dict.fromkeys(THRESHOLDS, 0)
    strays = dict.fromkeys(THRESHOLDS, 0)
    for path, ann, raw in zip(images, annotations, candidates, strict=True):
        with Image.open(path) as handle:
            megapixels += handle.size[0] * handle.size[1] / 1e6
        found = suppress(raw, config)
        keep = np.isin(found.labels, VEHICLES)
        boxes, scores = found.boxes[keep], found.scores[keep]
        truth = ann.boxes[np.isin(ann.labels, VEHICLES)]
        objects += len(truth)
        iou = pairwise_iou(truth, boxes)
        ignored = np.zeros(len(boxes), dtype=bool)
        if len(ann.ignore) and len(boxes):
            ignored = pairwise_iou(boxes, ann.ignore).max(axis=1) >= 0.1
        for threshold in THRESHOLDS:
            selected = scores >= threshold
            overlap = iou[:, selected]
            good = overlap.max(axis=0, initial=0.0) >= MATCH_IOU
            hits[threshold] += int((overlap.max(axis=1, initial=0.0) >= MATCH_IOU).sum())
            true_positives[threshold] += int(good.sum())
            strays[threshold] += int((~good & ~ignored[selected]).sum())

    curve = []
    for threshold in THRESHOLDS:
        recall = hits[threshold] / objects
        precision = true_positives[threshold] / max(
            true_positives[threshold] + strays[threshold], 1
        )
        curve.append(
            {
                "score": threshold,
                "recall": recall,
                "precision": precision,
                "f1": 2 * recall * precision / max(recall + precision, 1e-9),
                "false_positives_per_megapixel": strays[threshold] / megapixels,
            }
        )

    at = {row["score"]: row for row in curve}
    high, low = at[tracker.high_score], at[tracker.low_score]
    payload = {
        "classes": [CLASS_NAMES[i] for i in VEHICLES],
        "match_iou": MATCH_IOU,
        "objects": objects,
        "high_score": tracker.high_score,
        "low_score": tracker.low_score,
        "recall_at_high_score": high["recall"],
        "recall_at_low_score": low["recall"],
        "clutter_per_megapixel_above_high": high["false_positives_per_megapixel"],
        "clutter_per_megapixel_low_band": low["false_positives_per_megapixel"]
        - high["false_positives_per_megapixel"],
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "curve"}, indent=2))


if __name__ == "__main__":
    main()
