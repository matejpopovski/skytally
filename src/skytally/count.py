"""Count traffic across a line in a folder of ordered frames.

    python -m skytally.count --checkpoint runs/p2/best.pt --frames clip/ \
        --line 620,300 650,420 --fps 10 --output counts.json

Frames are read in filename order. The output carries totals, per-event records and the
tracker settings used. The score thresholds suit VisDrone-like imagery. Footage that looks
different needs them recalibrated on a labelled clip, because a detector's confidence
shifts with the domain (``scripts/run_e2e_sim.py`` measures how much this matters).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from skytally import CLASS_NAMES, VEHICLE_CLASSES
from skytally.boxes import nms
from skytally.infer import Candidates, InferConfig, best_device, detect, load_detector
from skytally.track.byte import ByteTracker, Detection, TrackerConfig
from skytally.track.counter import LineCounter
from skytally.viz import Overlay

FRAME_SUFFIXES = {".jpg", ".jpeg", ".png"}


MERGE_IOU = 0.3


def to_detection(candidates: Candidates) -> Detection:
    """Tracker input, restricted to vehicles: a person near the line is not traffic."""
    keep = np.isin(candidates.labels, VEHICLE_CLASSES)
    return Detection(candidates.boxes[keep], candidates.scores[keep], candidates.labels[keep])


def merge_classes(detection: Detection, iou_threshold: float = MERGE_IOU) -> Detection:
    """Keep one box per object across classes.

    NMS runs per class because that is what mAP rewards, but it leaves a "car" and a "van"
    box on one vehicle. A tracker follows both, so the vehicle crosses the line twice.
    """
    if not len(detection):
        return detection
    kept = nms(detection.boxes, detection.scores, iou_threshold)
    return detection.select(np.isin(np.arange(len(detection)), kept))


def parse_point(text: str) -> tuple[float, float]:
    x, y = text.split(",")
    return float(x), float(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--line", type=parse_point, nargs=2, required=True, metavar="X,Y")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--long-side", type=int, default=1536)
    parser.add_argument("--high-score", type=float, default=TrackerConfig.high_score)
    parser.add_argument("--low-score", type=float, default=TrackerConfig.low_score)
    parser.add_argument("--new-track-score", type=float, default=TrackerConfig.new_track_score)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path, help="write annotated frames here")
    args = parser.parse_args()

    paths = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in FRAME_SUFFIXES)
    if not paths:
        raise SystemExit(f"no frames in {args.frames}")

    device = best_device()
    model = load_detector(args.checkpoint, device)
    infer = InferConfig(long_side=args.long_side, score_threshold=0.05)
    settings = TrackerConfig(
        high_score=args.high_score,
        low_score=args.low_score,
        new_track_score=args.new_track_score,
    )
    tracker = ByteTracker(settings)
    counter = LineCounter(*args.line)
    overlay = Overlay(tuple(args.line))
    if args.overlay_dir:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    for frame, path in enumerate(paths):
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        tracks = tracker.update(merge_classes(to_detection(detect(model, image, infer, device))))
        counter.update(frame, tracks)
        if args.overlay_dir:
            drawn = overlay.draw(np.asarray(image), frame, args.fps, tracks, counter)
            drawn.save(args.overlay_dir / f"{frame:06d}.jpg", quality=90)

    payload = {
        "frames": len(paths),
        "line": list(args.line),
        "totals": dict(counter.direction_totals()),
        "by_class": {
            f"{direction}/{CLASS_NAMES[label]}": value
            for (direction, label), value in sorted(counter.totals().items())
        },
        "events": [asdict(e) for e in counter.events],
        "tracker": {k: v for k, v in asdict(settings).items() if k != "kalman"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["totals"]))


if __name__ == "__main__":
    main()
