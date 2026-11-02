#!/usr/bin/env python3
"""Measure what makes VisDrone hard: object size, density, and grid coverage by stride.

"Unreachable" objects are those for which plain FCOS centre sampling finds no grid point,
measured at native resolution on the annotated coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skytally import CLASS_NAMES
from skytally.assign import TargetGrid, coverage, pyramid_levels
from skytally.data import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--output", type=Path, default=Path("results/dataset_stats.json"))
    args = parser.parse_args()

    summary = {}
    for split in ("train", "val"):
        _, annotations = load_split(args.data_dir / split)
        boxes = np.concatenate([a.boxes for a in annotations])
        labels = np.concatenate([a.labels for a in annotations])
        wh = boxes[:, 2:] - boxes[:, :2]
        area = wh[:, 0] * wh[:, 1]
        root_area = np.sqrt(area)
        counts_per_image = np.array([len(a.boxes) for a in annotations])

        reach = {}
        for min_level in (2, 3):
            levels = pyramid_levels(min_level)
            # Coordinates are absolute, so the grid must span the largest image.
            grid = TargetGrid(4096, 4096, levels)
            points = np.concatenate([coverage(a.boxes, grid) for a in annotations])
            tiny = area < 16**2
            reach[f"stride_{levels[0].stride}"] = {
                "unreachable_fraction": float((points == 0).mean()),
                "unreachable_fraction_tiny": float((points[tiny] == 0).mean()),
                "median_points_per_object": float(np.median(points)),
            }

        summary[split] = {
            "images": len(annotations),
            "objects": len(boxes),
            "ignore_regions": int(sum(len(a.ignore) for a in annotations)),
            "objects_per_image_mean": float(counts_per_image.mean()),
            "objects_per_image_max": int(counts_per_image.max()),
            "sqrt_area_median_px": float(np.median(root_area)),
            "fraction_area_below_16_sq": float((area < 16**2).mean()),
            "fraction_area_below_32_sq": float((area < 32**2).mean()),
            "fraction_area_below_96_sq": float((area < 96**2).mean()),
            "objects_per_class": {
                name: int((labels == i).sum()) for i, name in enumerate(CLASS_NAMES)
            },
            "grid_coverage": reach,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
