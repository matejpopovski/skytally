import math
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image

from skytally.checkpoint import load_checkpoint, save_checkpoint, to_cpu
from skytally.model import ModelConfig
from skytally.train import (
    EXIT_BUDGET_REACHED,
    TrainConfig,
    learning_rate_scale,
    stream_batches,
    train,
)


def test_to_cpu_recurses_and_copies():
    tensor = torch.ones(2)
    out = to_cpu({"a": [tensor], "b": (tensor, 3), "c": "x"})
    assert out["a"][0] is not tensor
    assert out["b"][1] == 3
    tensor += 1
    assert out["a"][0].tolist() == [1, 1]


def test_round_trip(tmp_path):
    path = tmp_path / "ck.pt"
    save_checkpoint(path, {"step": 7, "weights": {"w": torch.arange(3.0)}})
    state = load_checkpoint(path)
    assert state["step"] == 7
    assert state["weights"]["w"].tolist() == [0, 1, 2]


def test_crash_during_write_keeps_previous_checkpoint(tmp_path):
    path = tmp_path / "ck.pt"
    save_checkpoint(path, {"step": 1})

    def truncated_write(obj, f):
        f.write(b"partial")
        raise OSError("disk went away")

    with (
        mock.patch("skytally.checkpoint.torch.save", truncated_write),
        pytest.raises(OSError, match="disk went away"),
    ):
        save_checkpoint(path, {"step": 2})
    assert load_checkpoint(path)["step"] == 1


def test_a_second_save_replaces_the_first_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "ck.pt"
    save_checkpoint(path, {"step": 1})
    save_checkpoint(path, {"step": 2})
    assert load_checkpoint(path)["step"] == 2
    assert [p.name for p in tmp_path.iterdir()] == ["ck.pt"]


def test_learning_rate_schedule_shape():
    assert learning_rate_scale(0, 100, 10) == pytest.approx(0.1)
    assert learning_rate_scale(9, 100, 10) == pytest.approx(1.0)
    assert learning_rate_scale(10, 100, 10) == pytest.approx(1.0)
    assert learning_rate_scale(100, 100, 10) == pytest.approx(0.01)
    assert learning_rate_scale(55, 100, 10) == pytest.approx(
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi / 2))
    )


def write_tiny_visdrone(root):
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "annotations").mkdir()
        for i in range(6):
            pixels = rng.integers(0, 255, (96, 128, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / split / "images" / f"{i:03d}.jpg")
            (root / split / "annotations" / f"{i:03d}.txt").write_text(
                "20,30,16,12,1,4,0,0\n50,40,20,18,1,1,0,0\n"
            )


def tiny_config(root, name, total_steps):
    return TrainConfig(
        name=name,
        model=ModelConfig(channels=16, head_depth=1, pretrained=False),
        crop=TrainConfig().crop.__class__(size=64),
        data_dir=str(root),
        out_dir=str(root / "runs"),
        batch_size=2,
        total_steps=total_steps,
        warmup_steps=2,
        workers=0,
        checkpoint_every_s=1e9,
    )


def test_interrupted_run_resumes_to_identical_weights(tmp_path):
    """Stopping at step 3 and resuming must equal an uninterrupted run, bit for bit."""
    write_tiny_visdrone(tmp_path)
    device = torch.device("cpu")

    assert train(tiny_config(tmp_path, "straight", 6), device) == 0

    interrupted = tiny_config(tmp_path, "split", 6)
    interrupted.stop_at_step = 3
    assert train(interrupted, device) == EXIT_BUDGET_REACHED
    assert load_checkpoint(tmp_path / "runs/split/last.pt")["step"] == 3
    assert train(tiny_config(tmp_path, "split", 6), device) == 0

    straight = load_checkpoint(tmp_path / "runs/straight/last.pt")
    split = load_checkpoint(tmp_path / "runs/split/last.pt")
    assert split["step"] == straight["step"] == 6
    for section in ("model", "ema"):
        for key, value in straight[section].items():
            assert torch.equal(value, split[section][key]), f"{section}/{key} diverged"


def test_data_stream_reproduces_the_same_batches_after_a_restart(tmp_path):
    write_tiny_visdrone(tmp_path)
    from skytally.data import TrainingSamples, load_split

    config = tiny_config(tmp_path, "stream", 10)
    images, annotations = load_split(tmp_path / "train")
    dataset = TrainingSamples(images, annotations, config.model.levels, config.crop)
    device = torch.device("cpu")

    first = stream_batches(dataset, config, 0, device)
    original = [next(first) for _ in range(5)]
    restarted = stream_batches(dataset, config, 3, device)
    again = [next(restarted) for _ in range(2)]
    assert [step for step, _ in again] == [3, 4]
    for (_, a), (_, b) in zip(original[3:], again, strict=True):
        for x, y in zip(a, b, strict=True):
            assert torch.equal(x, y)


def test_stream_rebuilds_loader_after_worker_failure(tmp_path):
    write_tiny_visdrone(tmp_path)
    from skytally import train as train_module
    from skytally.data import TrainingSamples, load_split

    config = tiny_config(tmp_path, "rebuild", 10)
    images, annotations = load_split(tmp_path / "train")
    dataset = TrainingSamples(images, annotations, config.model.levels, config.crop)
    real = train_module.DataLoader
    built = []

    class FailsOnce:
        def __init__(self, *args, **kwargs):
            self.inner = real(*args, **kwargs)
            built.append(self)

        def __iter__(self):
            iterator = iter(self.inner)
            if len(built) == 1:
                yield next(iterator)
                raise RuntimeError("DataLoader timed out after 60 seconds")
            yield from iterator

    with mock.patch.object(train_module, "DataLoader", FailsOnce):
        stream = stream_batches(dataset, config, 0, torch.device("cpu"))
        steps = [next(stream)[0] for _ in range(3)]
    assert steps == [0, 1, 2]
    assert len(built) == 2
