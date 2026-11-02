#!/usr/bin/env python3
"""Tracker ablation on simulated detections. SIMULATOR MEASUREMENT, not real footage.

Each variant runs on the same sequences, with the same corrupted detections, over several
seeds. The detector stand-in can take its recall from ``--operating-point``, which holds
rates measured on VisDrone val for the trained detector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np

from skytally.sim.noise import DetectorModel, corrupt
from skytally.sim.traffic import SceneConfig, simulate
from skytally.track.byte import ByteTracker, TrackerConfig
from skytally.track.counter import LineCounter
from skytally.track.metrics import FrameBoxes, evaluate_tracking

VARIANTS = {
    "iou_only": TrackerConfig(use_kalman=False, use_low_scores=False, buffer=0.0, lost_buffer=0.0),
    "kalman": TrackerConfig(use_low_scores=False, buffer=0.0, lost_buffer=0.0),
    "kalman_low_scores": TrackerConfig(buffer=0.0, lost_buffer=0.0),
    "plus_buffer": TrackerConfig(lost_buffer=0.0),
    "plus_lost_search": TrackerConfig(),
}
SCENARIOS = {
    "nominal": SceneConfig(),
    "fast_traffic": SceneConfig(speed_px_per_s=(130.0, 200.0)),
    "dense": SceneConfig(arrivals_per_minute=70.0),
    "occluded_line": SceneConfig(occluder=(590.0, 690.0)),
}


def truth_frames(sequence) -> list[FrameBoxes]:
    frames = []
    for states in sequence.frames:
        visible = [s for s in states if s.visible]
        frames.append(
            FrameBoxes(
                np.array([s.vehicle_id for s in visible], dtype=np.int64),
                np.array([s.box for s in visible]).reshape(-1, 4),
            )
        )
    return frames


def run(sequence, detections, config: TrackerConfig) -> dict:
    tracker = ByteTracker(config)
    counter = LineCounter(*sequence.line)
    predicted = []
    for frame, frame_detections in enumerate(detections):
        tracks = tracker.update(frame_detections)
        counter.update(frame, tracks)
        predicted.append(
            FrameBoxes(
                np.array([t.track_id for t in tracks], dtype=np.int64),
                np.array([t.box for t in tracks]).reshape(-1, 4),
            )
        )
    scores = evaluate_tracking(truth_frames(sequence), predicted)
    truth = sequence.true_counts()
    counted = counter.direction_totals()
    error = sum(abs(truth[k] - counted[k]) for k in truth)
    return {
        "mota": scores.mota,
        "idf1": scores.idf1,
        "id_switches": scores.id_switches,
        "true_crossings": sum(truth.values()),
        "counted": sum(counted.values()),
        "count_error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=100)
    parser.add_argument("--operating-point", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/tracker_sim.json"))
    args = parser.parse_args()

    model = DetectorModel()
    if args.operating_point:
        rates = json.loads(args.operating_point.read_text())
        megapixels = SceneConfig().width * SceneConfig().height / 1e6
        model = replace(
            model,
            high_recall=rates["recall_at_high_score"],
            any_recall=rates["recall_at_low_score"],
            false_positives_per_frame=rates["clutter_per_megapixel_low_band"] * megapixels,
            persistent_false_positives_per_frame=rates["clutter_per_megapixel_above_high"]
            * megapixels,
        )

    table: dict[str, dict] = {}
    for scenario, base in SCENARIOS.items():
        collected: dict[str, list[dict]] = {name: [] for name in VARIANTS}
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            sequence = simulate(replace(base, seed=seed))
            detections = corrupt(sequence, model, seed=seed + 1000)
            for name, config in VARIANTS.items():
                collected[name].append(run(sequence, detections, config))
        table[scenario] = {}
        for name, runs in collected.items():
            true_total = sum(r["true_crossings"] for r in runs)
            table[scenario][name] = {
                "mota": mean(r["mota"] for r in runs),
                "idf1": mean(r["idf1"] for r in runs),
                "id_switches_per_run": mean(r["id_switches"] for r in runs),
                "true_crossings": true_total,
                "count_absolute_error": sum(r["count_error"] for r in runs),
                "count_error_percent": 100
                * sum(r["count_error"] for r in runs)
                / max(true_total, 1),
            }
            print(scenario, name, table[scenario][name], flush=True)

    payload = {
        "label": "simulator measurement",
        "seeds": args.seeds,
        "seconds_per_run": SceneConfig().seconds,
        "fps": SceneConfig().fps,
        "detector_model": model.__dict__,
        "results": table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
