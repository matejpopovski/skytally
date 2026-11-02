"""A detector stand-in: corrupts true boxes the way a real detector does.

Misses are not independent per frame. A real detector fails on the same object for a run
of frames while its appearance stays hard, so each object carries a two-state Markov
chain. Independent misses would make every tracker look better than it is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skytally.sim.traffic import Sequence
from skytally.track.byte import Detection


@dataclass(frozen=True)
class DetectorModel:
    high_recall: float = 0.65
    any_recall: float = 0.85
    hard_state_entry: float = 0.04
    hard_state_exit: float = 0.25
    hard_recall_factor: float = 0.35
    center_noise_px: float = 1.2
    size_noise: float = 0.10
    false_positives_per_frame: float = 1.0
    wrong_label_rate: float = 0.08


def corrupt(sequence: Sequence, model: DetectorModel, seed: int) -> list[Detection]:
    rng = np.random.default_rng(seed)
    config = sequence.config
    hard: dict[int, bool] = {}
    output = []
    for states in sequence.frames:
        boxes, scores, labels = [], [], []
        for state in states:
            if not state.visible:
                continue
            now_hard = hard.get(state.vehicle_id, False)
            flip = model.hard_state_exit if now_hard else model.hard_state_entry
            if rng.random() < flip:
                now_hard = not now_hard
            hard[state.vehicle_id] = now_hard

            factor = model.hard_recall_factor if now_hard else 1.0
            draw = rng.random()
            if draw < model.high_recall * factor:
                score = rng.uniform(0.35, 0.9)
            elif draw < model.any_recall * factor:
                score = rng.uniform(0.1, 0.35)
            else:
                continue

            center = (state.box[:2] + state.box[2:]) / 2 + rng.normal(0, model.center_noise_px, 2)
            size = (state.box[2:] - state.box[:2]) * np.exp(rng.normal(0, model.size_noise, 2))
            boxes.append(np.concatenate([center - size / 2, center + size / 2]))
            scores.append(score)
            label = state.label
            if rng.random() < model.wrong_label_rate:
                label = int(rng.choice([3, 4, 5, 8]))
            labels.append(label)

        for _ in range(rng.poisson(model.false_positives_per_frame)):
            size = rng.uniform(10, 30, 2)
            corner = rng.uniform((0, 0), (config.width - 30, config.height - 30))
            boxes.append(np.concatenate([corner, corner + size]))
            scores.append(rng.uniform(0.1, 0.45))
            labels.append(3)
        output.append(Detection(np.array(boxes).reshape(-1, 4), scores, labels))
    return output
