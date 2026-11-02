#!/usr/bin/env python3
"""Sweep the NMS IoU threshold on cached candidates. Needs no GPU."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from skytally.data import load_split
from skytally.evaluate import score
from skytally.infer import InferConfig, suppress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--output", type=Path, default=Path("results/nms_sweep.json"))
    args = parser.parse_args()

    _, annotations = load_split(args.data_dir / "val")
    candidates = pickle.loads(args.candidates.read_bytes())
    rows = []
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
        config = InferConfig(nms_iou=threshold)
        result = score(annotations, [suppress(c, config) for c in candidates])
        rows.append(
            {
                "nms_iou": threshold,
                "AP": result.ap(),
                "AP50": result.ap(iou=0.5),
                "AP75": result.ap(iou=0.75),
                "AR_500": result.ar(max_dets=500),
            }
        )
        print(rows[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
