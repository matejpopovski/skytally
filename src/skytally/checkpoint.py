"""Crash-safe checkpoints.

Saving directly from MPS tensors has been seen to deadlock, and a process killed in the
middle of ``torch.save`` leaves a truncated file. So every tensor is copied to the CPU
first (after waiting for the GPU queue to drain), written to a temporary file in the same
directory, flushed to disk, and only then renamed over the previous checkpoint. A reader
sees the old checkpoint or the new one, never a partial one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def to_cpu(value: Any) -> Any:
    """Recursively copy every tensor in a nested structure to CPU memory."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(to_cpu(v) for v in value)
    return value


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    payload = to_cpu(state)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load to the CPU. ``weights_only`` refuses anything but tensors and plain containers."""
    return torch.load(path, map_location="cpu", weights_only=True)
