#!/usr/bin/env python3
"""Whole pipeline on rendered scenes: detector, tracker, counter, against simulator truth.

SIMULATOR MEASUREMENT. The detector is the one trained on VisDrone, and these frames are
a domain it has never seen. No VisDrone pixels are involved, so the frames written here
can be published.

The same detections go through three variants: VisDrone thresholds, thresholds chosen on
separate labelled rendered clips, and those chosen thresholds without the cross-class merge.
Thresholds are picked by IDF1 rather than count error, because over- and under-counting
can cancel. The evaluation clips are never used to choose anything.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from skytally.boxes import pairwise_iou
from skytally.count import merge_classes, to_detection
from skytally.infer import InferConfig, best_device, detect, load_detector
from skytally.sim.render import SceneRenderer
from skytally.sim.traffic import SceneConfig, road_axes, simulate
from skytally.track.byte import ByteTracker, TrackerConfig
from skytally.track.counter import LineCounter
from skytally.track.metrics import FrameBoxes, evaluate_tracking
from skytally.viz import Overlay

CALIBRATION_SEEDS = (50, 51)
EVALUATION_SEEDS = (0, 1, 2, 3, 4)
GRID = tuple(round(0.1 + 0.05 * i, 2) for i in range(11))


def detections_for(model, device, config: SceneConfig, args) -> tuple[list, float]:
    """Detector output for every frame of a clip, cached because it is the slow part."""
    path = args.cache_dir / f"seed_{config.seed}_{args.long_side}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    sequence = simulate(config)
    renderer = SceneRenderer(config, seed=config.seed)
    infer = InferConfig(long_side=args.long_side, score_threshold=0.1)
    results = []
    spent = 0.0
    for states in sequence.frames:
        image = Image.fromarray(renderer.render(states))
        started = time.perf_counter()
        results.append(to_detection(detect(model, image, infer, device)))
        spent += time.perf_counter() - started
    payload = (results, 1000 * spent / len(sequence.frames))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload))
    return payload


def settings_for(high: float) -> TrackerConfig:
    """The offsets used on VisDrone, applied around a different high threshold."""
    return TrackerConfig(
        high_score=high,
        low_score=round(max(high - 0.15, GRID[0]), 2),
        new_track_score=round(high + 0.05, 2),
    )


def calibrate(clips: list[tuple]) -> dict:
    """Pick the high threshold with the best IDF1 on labelled calibration clips."""
    rows = []
    for high in GRID:
        runs = [track_clip(s, d, settings_for(high)) for s, d in clips]
        rows.append(
            {"high_score": high, **{k: v for k, v in summarise(runs).items() if k != "runs"}}
        )
    return {
        "best_high_score": max(rows, key=lambda row: row["idf1_mean"])["high_score"],
        "grid": rows,
    }


def recall_by_lane(clips: list[tuple], score: float) -> dict:
    """Detection recall per lane at ``score``, on calibration clips only."""
    found: dict[str, int] = {}
    total: dict[str, int] = {}
    for sequence, detections in clips:
        config = sequence.config
        for states, detection in zip(sequence.frames, detections, strict=True):
            boxes = detection.boxes[detection.scores >= score]
            for state in states:
                _, _, normal, _ = road_axes(config)
                offset = float(np.dot(state.center - (config.width / 2, config.height / 2), normal))
                heading = "rightward" if np.cos(state.heading) > 0 else "leftward"
                position = "outer" if abs(offset) > config.lane_width * 1.25 else "inner"
                key = f"{heading}_{position}"
                total[key] = total.get(key, 0) + 1
                if len(boxes) and pairwise_iou(state.box[None], boxes).max() >= 0.3:
                    found[key] = found.get(key, 0) + 1
    return {key: found.get(key, 0) / total[key] for key in sorted(total)}


def track_clip(sequence, detections, settings: TrackerConfig) -> dict:
    tracker = ByteTracker(settings)
    counter = LineCounter(*sequence.line)
    truth_frames, predicted = [], []
    for frame, (states, detection) in enumerate(zip(sequence.frames, detections, strict=True)):
        tracks = tracker.update(detection)
        counter.update(frame, tracks)
        visible = [s for s in states if s.visible]
        truth_frames.append(
            FrameBoxes(
                np.array([s.vehicle_id for s in visible], dtype=np.int64),
                np.array([s.box for s in visible]).reshape(-1, 4),
            )
        )
        predicted.append(
            FrameBoxes(
                np.array([t.track_id for t in tracks], dtype=np.int64),
                np.array([t.box for t in tracks]).reshape(-1, 4),
            )
        )
    scores = evaluate_tracking(truth_frames, predicted)
    truth = sequence.true_counts()
    counted = counter.direction_totals()
    return {
        "seed": sequence.config.seed,
        "mota": scores.mota,
        "idf1": scores.idf1,
        "id_switches": scores.id_switches,
        "true_counts": truth,
        "pipeline_counts": {k: counted[k] for k in truth},
        "count_error": sum(abs(truth[k] - counted[k]) for k in truth),
    }


def summarise(runs: list[dict]) -> dict:
    true_total = sum(sum(r["true_counts"].values()) for r in runs)
    return {
        "true_crossings": true_total,
        "pipeline_crossings": sum(sum(r["pipeline_counts"].values()) for r in runs),
        "count_absolute_error": sum(r["count_error"] for r in runs),
        "count_error_percent": 100 * sum(r["count_error"] for r in runs) / max(true_total, 1),
        "mota_mean": float(np.mean([r["mota"] for r in runs])),
        "idf1_mean": float(np.mean([r["idf1"] for r in runs])),
        "id_switches_mean": float(np.mean([r["id_switches"] for r in runs])),
        "runs": runs,
    }


def save_gif(frames: list[Image.Image], path: Path, milliseconds: int, width: int) -> None:
    """Hold pixels that barely changed, so sensor noise does not defeat frame differencing."""
    height = round(frames[0].height * width / frames[0].width)
    small = [f.resize((width, height), Image.Resampling.LANCZOS) for f in frames]
    palette = small[len(small) // 2].quantize(colors=48, method=Image.Quantize.MEDIANCUT)
    held = []
    previous = None
    for frame in small:
        rgb = np.asarray(frame, dtype=np.int16)
        indices = np.array(frame.quantize(palette=palette, dither=Image.Dither.NONE))
        if previous is not None:
            still = np.abs(rgb - previous[0]).sum(axis=2) < 48
            indices[still] = previous[1][still]
            rgb = np.where(still[..., None], previous[0], rgb)
        previous = (rgb, indices)
        image = Image.fromarray(indices.astype(np.uint8), mode="P")
        image.putpalette(palette.getpalette())
        held.append(image)
    held[0].save(
        path,
        save_all=True,
        append_images=held[1:],
        duration=milliseconds,
        loop=0,
        optimize=False,
    )


def write_figures(sequence, detections, settings: TrackerConfig, args) -> None:
    """Replay a clip with tracking on, drawing the overlay over the real output."""
    config = sequence.config
    renderer = SceneRenderer(config, seed=config.seed)
    tracker = ByteTracker(settings)
    counter = LineCounter(*sequence.line)
    overlay = Overlay(sequence.line, caption="SYNTHETIC SCENE")
    frames = []
    last = args.gif_start + args.gif_frames * args.gif_step
    for frame, (states, detection) in enumerate(zip(sequence.frames, detections, strict=True)):
        if frame >= last:
            break
        pixels = renderer.render(states)
        tracks = tracker.update(detection)
        counter.update(frame, tracks)
        if frame >= args.gif_start and (frame - args.gif_start) % args.gif_step == 0:
            frames.append(overlay.draw(pixels, frame, config.fps, tracks, counter))
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    milliseconds = round(1000 * args.gif_step / config.fps)
    save_gif(frames, args.figure_dir / "pipeline.gif", milliseconds, args.gif_width)
    frames[len(frames) // 2].save(args.figure_dir / "pipeline.png", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--long-side", type=int, default=1536)
    parser.add_argument("--gif-start", type=int, default=120)
    parser.add_argument("--gif-frames", type=int, default=32)
    parser.add_argument("--gif-step", type=int, default=2)
    parser.add_argument("--gif-width", type=int, default=800)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/e2e_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/e2e_sim.json"))
    args = parser.parse_args()

    device = best_device()
    model = load_detector(args.checkpoint, device)
    base = SceneConfig(seconds=args.seconds, arrivals_per_minute=40.0)

    def clip(seed: int) -> tuple:
        config = replace(base, seed=seed)
        detections, milliseconds = detections_for(model, device, config, args)
        return simulate(config), detections, milliseconds

    def merged(detections: list) -> list:
        return [merge_classes(d) for d in detections]

    calibration_clips = []
    for seed in CALIBRATION_SEEDS:
        sequence, detections, _ = clip(seed)
        calibration_clips.append((sequence, merged(detections)))
    calibrated = calibrate(calibration_clips)
    evaluation = [clip(seed) for seed in EVALUATION_SEEDS]

    domain_settings = settings_for(calibrated["best_high_score"])
    payload = {
        "label": "simulator measurement: rendered synthetic scenes, trained VisDrone detector",
        "scene": {k: v for k, v in base.__dict__.items() if k != "seed"},
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "calibration": calibrated,
        "recall_by_lane_on_calibration_clips": recall_by_lane(
            calibration_clips, domain_settings.high_score
        ),
        "detect_ms_per_frame": float(np.mean([e[2] for e in evaluation])),
        "calibrated_tracker": {
            k: getattr(domain_settings, k) for k in ("high_score", "low_score", "new_track_score")
        },
        "visdrone_thresholds": summarise(
            [track_clip(s, merged(d), TrackerConfig()) for s, d, _ in evaluation]
        ),
        "calibrated_thresholds": summarise(
            [track_clip(s, merged(d), domain_settings) for s, d, _ in evaluation]
        ),
        "calibrated_without_merge": summarise(
            [track_clip(s, d, domain_settings) for s, d, _ in evaluation]
        ),
    }
    write_figures(evaluation[0][0], merged(evaluation[0][1]), domain_settings, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for name in ("visdrone_thresholds", "calibrated_thresholds", "calibrated_without_merge"):
        print(name, {k: v for k, v in payload[name].items() if k != "runs"})
    print(payload["calibrated_tracker"], payload["recall_by_lane_on_calibration_clips"])


if __name__ == "__main__":
    main()
