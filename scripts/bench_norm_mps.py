#!/usr/bin/env python3
"""Measure what the head's normalisation layer costs on the MPS backend.

Runs the same tower with GroupNorm, BatchNorm and no normalisation on a P2-sized feature
map and reports training steps per second and driver memory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn


def block(kind: str, channels: int) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.Conv2d(channels, channels, 1, bias=False),
    ]
    if kind == "groupnorm":
        layers.append(nn.GroupNorm(16, channels))
    elif kind == "batchnorm":
        layers.append(nn.BatchNorm2d(channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def measure(kind: str, device: torch.device, batch: int, size: int, steps: int) -> dict:
    tower = nn.Sequential(*(block(kind, 96) for _ in range(4))).to(device)
    optimizer = torch.optim.SGD(tower.parameters(), lr=1e-4)
    features = torch.randn(batch, 96, size, size, device=device)
    timings = []
    for _ in range(steps):
        started = time.perf_counter()
        loss = tower(features).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss.item()
        timings.append(time.perf_counter() - started)
    timings = sorted(timings[2:])
    result = {
        "norm": kind,
        "step_ms_median": timings[len(timings) // 2] * 1000,
        "driver_gb": torch.mps.driver_allocated_memory() / 1e9,
    }
    del tower, optimizer, features, loss
    torch.mps.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/norm_mps.json"))
    args = parser.parse_args()

    device = torch.device("mps")
    rows = [
        measure(kind, device, args.batch, args.size, args.steps)
        for kind in ("groupnorm", "batchnorm", "none")
    ]
    payload = {
        "hardware": "Apple M3 Max, 36 GB",
        "torch": torch.__version__,
        "feature_map": [args.batch, 96, args.size, args.size],
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
