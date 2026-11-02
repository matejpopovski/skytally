#!/usr/bin/env python3
"""Per-image latency by input size, split into network and post-processing time.

The GPU queue is drained around the network call, because MPS returns from ``forward``
before the work has finished. Timings use real validation images. The machine may be
running other work, so the output records the 1-minute load average.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from PIL import Image

from skytally.data import load_split
from skytally.infer import (
    InferConfig,
    best_device,
    load_detector,
    network_candidates,
    prepare,
    suppress,
    to_candidates,
)


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[int(fraction * (len(values) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[960, 1280, 1536, 1920, 2400])
    parser.add_argument("--images", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("results/latency.json"))
    args = parser.parse_args()

    device = best_device()
    model = load_detector(args.checkpoint, device)
    paths, _ = load_split(args.data_dir / "val")
    paths = paths[:: max(1, len(paths) // args.images)][: args.images]

    rows = []
    for long_side in args.sizes:
        config = InferConfig(long_side=long_side)
        network, post = [], []
        for path in paths:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
            tensor, scale = prepare(image, long_side)
            batch = tensor[None].to(device)
            started = time.perf_counter()
            boxes, scores, labels = network_candidates(model, batch, config)
            if device.type == "mps":
                torch.mps.synchronize()
            middle = time.perf_counter()
            suppress(to_candidates(boxes, scores, labels, scale, image.size, config), config)
            ended = time.perf_counter()
            network.append((middle - started) * 1000)
            post.append((ended - middle) * 1000)
        rows.append(
            {
                "long_side": long_side,
                "input": list(tensor.shape[1:]),
                "network_ms_median": statistics.median(network[3:]),
                "network_ms_p95": percentile(network[3:], 0.95),
                "postprocess_ms_median": statistics.median(post[3:]),
            }
        )
        print(rows[-1], flush=True)

    payload = {
        "device": str(device),
        "hardware": "Apple M3 Max, 36 GB",
        "images": len(paths),
        "load_average_1min": os.getloadavg()[0],
        "model": args.checkpoint.parent.name,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
