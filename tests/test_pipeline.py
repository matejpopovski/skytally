"""The parts between detector output and the published overlay and counts."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from skytally import count
from skytally.checkpoint import save_checkpoint
from skytally.infer import Candidates, InferConfig, suppress
from skytally.model import Detector, ModelConfig
from skytally.sim.render import SceneRenderer, paint_background
from skytally.sim.traffic import SceneConfig, simulate
from skytally.track.byte import ByteTracker, Detection
from skytally.track.counter import LineCounter
from skytally.viz import Overlay


def small_scene(**kwargs):
    return SceneConfig(width=320, height=180, seconds=6.0, warmup_seconds=3.0, **kwargs)


def test_background_depends_only_on_seed():
    config = small_scene()
    a = np.asarray(paint_background(config, 1))
    b = np.asarray(paint_background(config, 1))
    c = np.asarray(paint_background(config, 2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_render_draws_vehicles_at_their_positions():
    config = small_scene(camera_jitter_px=0.0, arrivals_per_minute=120.0)
    sequence = simulate(config)
    frame = next(f for f in sequence.frames if f)
    renderer = SceneRenderer(config, seed=0)
    drawn = renderer.render(frame).astype(np.float32)
    empty = renderer.render([]).astype(np.float32)
    assert drawn.shape == (180, 320, 3)
    state = next(s for s in frame if 20 < s.center[0] < 300 and 20 < s.center[1] < 160)
    x, y = state.center.astype(int)
    assert (
        np.abs(drawn[y - 3 : y + 3, x - 3 : x + 3] - empty[y - 3 : y + 3, x - 3 : x + 3]).mean() > 8
    )


def test_occluder_hides_vehicles_in_the_render():
    config = small_scene(camera_jitter_px=0.0, arrivals_per_minute=120.0, occluder=(0.0, 320.0))
    sequence = simulate(config)
    frame = next(f for f in sequence.frames if f)
    renderer = SceneRenderer(config, seed=0)
    drawn = renderer.render(frame).astype(np.float32)
    empty = renderer.render([]).astype(np.float32)
    assert np.abs(drawn - empty).mean() < 4


def test_overlay_marks_the_frame_without_resizing_it():
    tracker = ByteTracker()
    box = np.array([[60.0, 40.0, 84.0, 52.0]])
    for _ in range(3):
        tracks = tracker.update(Detection(box, [0.9], [3]))
    counter = LineCounter((10.0, 0.0), (10.0, 100.0))
    overlay = Overlay(((10.0, 0.0), (10.0, 100.0)), caption="SYNTHETIC SCENE")
    pixels = np.full((180, 320, 3), 60, dtype=np.uint8)
    out = np.asarray(overlay.draw(pixels, 5, 10.0, tracks, counter))
    assert out.shape == pixels.shape
    assert np.abs(out[46, 60].astype(int) - 60).sum() > 30


def test_suppress_enforces_max_detections_and_keeps_best():
    boxes = np.array([[i * 20.0, 0, i * 20.0 + 10, 10] for i in range(6)])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    candidates = Candidates(boxes, scores, np.zeros(6, dtype=np.int64))
    kept = suppress(candidates, InferConfig(max_dets=2))
    assert kept.scores.tolist() == [0.9, 0.8]


def test_merge_keeps_one_box_when_two_classes_claim_a_vehicle():
    box = [100.0, 100.0, 124.0, 112.0]
    nearby = [101.0, 100.0, 125.0, 112.0]
    far = [300.0, 100.0, 324.0, 112.0]
    detection = Detection(np.array([box, nearby, far]), [0.4, 0.6, 0.5], [3, 4, 3])
    merged = count.merge_classes(detection)
    assert sorted(merged.scores.tolist()) == [0.5, 0.6]
    assert 4 in merged.labels


def test_merge_leaves_adjacent_vehicles_alone():
    a = [0.0, 0.0, 24.0, 12.0]
    b = [24.0, 0.0, 48.0, 12.0]
    assert len(count.merge_classes(Detection(np.array([a, b]), [0.5, 0.5], [3, 3]))) == 2


def test_merge_passes_empty_detections_through():
    assert len(count.merge_classes(Detection(np.empty((0, 4)), [], []))) == 0


def test_to_detection_drops_non_vehicle_classes():
    boxes = np.array([[0.0, 0, 5, 5], [10.0, 0, 15, 5]])
    detection = count.to_detection(Candidates(boxes, np.array([0.9, 0.9]), np.array([0, 3])))
    assert detection.labels.tolist() == [3]


def test_parse_point():
    assert count.parse_point("12.5,40") == (12.5, 40.0)


@pytest.mark.parametrize("frames", [0, 3])
def test_count_command_runs_on_a_folder_of_frames(tmp_path, monkeypatch, frames):
    config = ModelConfig(channels=16, head_depth=1, pretrained=False)
    checkpoint = tmp_path / "model.pt"
    model = Detector(config)
    save_checkpoint(
        checkpoint, {"model": model.state_dict(), "ema": None, "model_config": config.to_dict()}
    )
    clip = tmp_path / "clip"
    clip.mkdir()
    for i in range(frames):
        Image.fromarray(np.zeros((64, 96, 3), dtype=np.uint8)).save(clip / f"{i}.png")
    output = tmp_path / "counts.json"
    overlays = tmp_path / "overlay"
    argv = [
        "count",
        "--checkpoint",
        str(checkpoint),
        "--frames",
        str(clip),
        "--line",
        "10,0",
        "10,64",
        "--long-side",
        "96",
        "--output",
        str(output),
        "--overlay-dir",
        str(overlays),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(count, "best_device", lambda: torch.device("cpu"))
    if frames == 0:
        with pytest.raises(SystemExit, match="no frames"):
            count.main()
        return
    count.main()
    payload = json.loads(Path(output).read_text())
    assert payload["frames"] == frames
    assert payload["events"] == []
    assert len(list(overlays.iterdir())) == frames
