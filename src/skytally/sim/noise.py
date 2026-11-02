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
    """Marginal recall per band, and clutter per frame, as a tracker would see them."""

    high_score: float = 0.45
    low_score: float = 0.3
    high_recall: float = 0.75
    any_recall: float = 0.9
    hard_state_entry: float = 0.04
    hard_state_exit: float = 0.25
    hard_recall_factor: float = 0.35
    center_noise_px: float = 1.2
    size_noise: float = 0.10
    false_positives_per_frame: float = 1.0
    persistent_false_positives_per_frame: float = 1.0
    site_fire_rate: float = 0.5
    wrong_label_rate: float = 0.08

    def base_recall_factor(self) -> float:
        """Average recall multiplier, so the marginal rates stay equal to the measured ones."""
        hard = self.hard_state_entry / (self.hard_state_entry + self.hard_state_exit)
        return (1 - hard) + hard * self.hard_recall_factor


def corrupt(sequence: Sequence, model: DetectorModel, seed: int) -> list[Detection]:
    """Turn true boxes into a detection stream.

    Clutter comes in two kinds. Random low-score noise appears anywhere. Persistent clutter
    sits at fixed sites and fires repeatedly with a high score, like a rooftop unit that
    looks like a car from the air. Persistent false positives are the ones that form
    tracks, so leaving them out would flatter the tracker.
    """
    rng = np.random.default_rng(seed)
    config = sequence.config
    hard: dict[int, bool] = {}
    base = model.base_recall_factor()
    high = min(model.high_recall / base, 1.0)
    any_band = min(model.any_recall / base, 1.0)

    num_sites = round(model.persistent_false_positives_per_frame / model.site_fire_rate)
    corners = rng.uniform((0, 0), (config.width - 30, config.height - 30), (num_sites, 2))
    sizes = rng.uniform(10, 30, (num_sites, 2))
    sites = np.concatenate([corners, corners + sizes], axis=1)

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
            if draw < high * factor:
                score = rng.uniform(model.high_score, model.high_score + 0.3)
            elif draw < any_band * factor:
                score = rng.uniform(model.low_score, model.high_score)
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
            scores.append(rng.uniform(model.low_score, model.high_score))
            labels.append(3)

        fired = rng.random(num_sites) < model.site_fire_rate
        for site in sites[fired]:
            boxes.append(site + rng.normal(0, model.center_noise_px, 4))
            scores.append(rng.uniform(model.high_score, model.high_score + 0.2))
            labels.append(3)
        output.append(Detection(np.array(boxes).reshape(-1, 4), scores, labels))
    return output
