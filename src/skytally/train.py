"""Resumable training loop.

Three failure modes shaped this file:

* A hung GPU command or data worker. ``faulthandler`` arms a C-level timer that needs
  neither the GIL nor the event loop; it dumps every stack and exits, and
  ``skytally.supervise`` restarts from the last checkpoint.
* A stalled or dead data worker. The loader has a timeout, and on expiry it is rebuilt
  at the same step. Sampling is a pure function of the step, so no sample is lost.
* A crash during a save. See ``skytally.checkpoint``.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from skytally.checkpoint import load_checkpoint, save_checkpoint
from skytally.data import CropConfig, EpochSampler, TrainingSamples, load_split
from skytally.evaluate import run_network, score
from skytally.infer import InferConfig, best_device, suppress
from skytally.losses import detection_loss
from skytally.model import Detector, ModelConfig, flatten_outputs

EXIT_BUDGET_REACHED = 3


@dataclass
class TrainConfig:
    name: str = "p2"
    model: ModelConfig = field(default_factory=ModelConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    data_dir: str = "data/visdrone"
    out_dir: str = "runs"
    seed: int = 0
    batch_size: int = 10
    total_steps: int = 10_000
    warmup_steps: int = 500
    lr: float = 3e-4
    backbone_lr_scale: float = 0.25
    weight_decay: float = 1e-4
    grad_clip: float = 10.0
    ema_decay: float = 0.999
    workers: int = 4
    worker_timeout_s: float = 120.0
    stall_timeout_s: float = 300.0
    checkpoint_every_s: float = 600.0
    eval_every_steps: int = 0
    eval_stride: int = 5
    eval_long_side: int = 1024
    max_wall_minutes: float = 0.0
    stop_at_step: int = 0
    memory_fraction: float = 0.25

    @classmethod
    def load(cls, path: Path) -> TrainConfig:
        raw = json.loads(Path(path).read_text())
        raw["model"] = ModelConfig(**raw.get("model", {}))
        raw["crop"] = CropConfig(**raw.get("crop", {}))
        return cls(**raw)


def collate(items: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
    """Default collate, kept top level so spawned workers can import it."""
    return tuple(torch.stack(column) for column in zip(*items, strict=True))


def stream_batches(
    dataset: TrainingSamples, config: TrainConfig, start_step: int, device: torch.device
) -> Iterator[tuple[int, tuple[torch.Tensor, ...]]]:
    """Yield ``(step, batch)`` forever, rebuilding the loader if a worker stalls or dies."""
    steps_per_epoch = len(dataset) // config.batch_size
    step = start_step
    sampler = EpochSampler(len(dataset), config.seed)
    loader = None
    iterator = None

    def build():
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            drop_last=True,
            collate_fn=collate,
            num_workers=config.workers,
            # Forking a process that has touched MPS can deadlock; spawn starts clean.
            multiprocessing_context="spawn" if config.workers else None,
            persistent_workers=config.workers > 0,
            prefetch_factor=2 if config.workers else None,
            timeout=config.worker_timeout_s if config.workers else 0,
        )

    while True:
        if iterator is None:
            epoch, position = divmod(step, steps_per_epoch)
            sampler.epoch, sampler.start = epoch, position * config.batch_size
            loader = loader or build()
            iterator = iter(loader)
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = None
            continue
        except RuntimeError as error:
            print(f"data loader failed at step {step}: {error}; rebuilding", flush=True)
            loader = None
            iterator = None
            continue
        yield step, batch
        step += 1


class ExponentialAverage:
    """Exponential moving average of model parameters, ramped up over early steps."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        decay = min(self.decay, (1 + step) / (10 + step))
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].lerp_(value.detach(), 1 - decay)
            else:
                self.shadow[key].copy_(value)


def learning_rate_scale(step: int, total: int, warmup: int) -> float:
    """Linear warmup, then cosine decay to 1% of the base rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def parameter_groups(model: Detector, config: TrainConfig) -> list[dict]:
    groups: dict[tuple[bool, bool], list] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = name.startswith(("stem", "stages"))
        decays = parameter.ndim > 1
        groups.setdefault((is_backbone, decays), []).append(parameter)
    return [
        {
            "params": params,
            "lr": config.lr * (config.backbone_lr_scale if is_backbone else 1.0),
            "weight_decay": config.weight_decay if decays else 0.0,
        }
        for (is_backbone, decays), params in groups.items()
    ]


def memory_gb() -> float:
    """Resident memory of this process and its workers, plus GPU driver allocations."""
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    if torch.backends.mps.is_available():
        total += torch.mps.driver_allocated_memory()
    return total / 1e9


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(config: TrainConfig, device: torch.device, resume: bool = True) -> int:
    run_dir = Path(config.out_dir) / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    log_path = run_dir / "metrics.jsonl"
    seed_everything(config.seed)

    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(config.memory_fraction)

    data_root = Path(config.data_dir)
    images, annotations = load_split(data_root / "train")
    dataset = TrainingSamples(images, annotations, config.model.levels, config.crop)

    model = Detector(config.model).to(device)
    ema = ExponentialAverage(model, config.ema_decay)
    optimizer = torch.optim.AdamW(parameter_groups(model, config))
    base_rates = [group["lr"] for group in optimizer.param_groups]

    start_step = 0
    normalizer = 100.0 * config.batch_size
    wall_used = 0.0
    if resume and checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path)
        model.load_state_dict(state["model"])
        ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        normalizer = state["normalizer"]
        wall_used = state["wall_seconds"]
        print(f"resumed at step {start_step}", flush=True)

    def checkpoint(step: int) -> None:
        faulthandler.dump_traceback_later(config.stall_timeout_s, exit=True)
        save_checkpoint(
            checkpoint_path,
            {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.shadow,
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer,
                "wall_seconds": wall_used + time.perf_counter() - started,
                "model_config": config.model.to_dict(),
            },
        )

    def validate(step: int) -> dict:
        eval_model = Detector(replace_pretrained(config.model)).to(device).eval()
        eval_model.load_state_dict(ema.shadow)
        infer = InferConfig(long_side=config.eval_long_side)
        val_images, val_annotations = load_split(data_root / "val")
        paths = val_images[:: config.eval_stride]

        def rearm() -> None:
            faulthandler.dump_traceback_later(config.stall_timeout_s, exit=True)

        candidates = run_network(eval_model, paths, infer, device, on_image=rearm)
        result = score(
            val_annotations[:: config.eval_stride], [suppress(c, infer) for c in candidates]
        )
        del eval_model
        if device.type == "mps":
            torch.mps.empty_cache()
        return {"step": step, "val": result.summary()}

    started = time.perf_counter()
    last_checkpoint = started
    meter_time = started
    meter_images = 0

    model.train()
    model.freeze_backbone_norm()
    batches = stream_batches(dataset, config, start_step, device)
    with log_path.open("a", buffering=1) as log:
        for step, (images_batch, target_classes, target_boxes) in batches:
            if step >= config.total_steps:
                break
            faulthandler.dump_traceback_later(config.stall_timeout_s, exit=True)

            scale = learning_rate_scale(step, config.total_steps, config.warmup_steps)
            for group, base in zip(optimizer.param_groups, base_rates, strict=True):
                group["lr"] = base * scale

            positives = int((target_classes >= 0).sum())
            normalizer = 0.99 * normalizer + 0.01 * max(positives, 1)

            images_batch = images_batch.to(device)
            target_classes = target_classes.to(device)
            target_boxes = target_boxes.to(device)
            cls, box, quality, points, strides = flatten_outputs(model(images_batch), model.strides)
            losses = detection_loss(
                cls, box, quality, points, strides, target_classes, target_boxes, normalizer
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            ema.update(model, step)

            # Reading the loss each step is also what keeps the MPS queue from running ahead.
            total = losses["total"].item()
            if not math.isfinite(total):
                raise FloatingPointError(f"non-finite loss {total} at step {step}")
            meter_images += images_batch.shape[0]

            done = step + 1
            if done % 50 == 0 or done == config.total_steps:
                now = time.perf_counter()
                row = {
                    "step": done,
                    "loss": total,
                    "cls": losses["cls"].item(),
                    "box": losses["box"].item(),
                    "quality": losses["quality"].item(),
                    "grad_norm": float(grad_norm),
                    "lr": optimizer.param_groups[0]["lr"],
                    "images_per_s": meter_images / (now - meter_time),
                    "memory_gb": memory_gb(),
                    "wall_seconds": wall_used + now - started,
                }
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                meter_time, meter_images = now, 0

            if config.eval_every_steps and done % config.eval_every_steps == 0:
                row = validate(done)
                log.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                model.train()
                model.freeze_backbone_norm()
                meter_time, meter_images = time.perf_counter(), 0

            now = time.perf_counter()
            if now - last_checkpoint > config.checkpoint_every_s or done == config.total_steps:
                checkpoint(done)
                last_checkpoint = time.perf_counter()

            minutes = (wall_used + time.perf_counter() - started) / 60
            out_of_time = config.max_wall_minutes and minutes > config.max_wall_minutes
            if out_of_time or done == config.stop_at_step:
                checkpoint(done)
                print(f"stopping early at step {done}", flush=True)
                faulthandler.cancel_dump_traceback_later()
                return EXIT_BUDGET_REACHED

    faulthandler.cancel_dump_traceback_later()
    (run_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print("training finished", flush=True)
    return 0


def replace_pretrained(config: ModelConfig) -> ModelConfig:
    """Same architecture without fetching ImageNet weights."""
    return ModelConfig(**{**config.to_dict(), "pretrained": False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    if args.total_steps:
        config.total_steps = args.total_steps
    if args.name:
        config.name = args.name
    device = torch.device(args.device) if args.device else best_device()
    faulthandler.enable()
    sys.exit(train(config, device, resume=not args.no_resume))


if __name__ == "__main__":
    main()
