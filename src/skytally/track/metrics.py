"""CLEAR MOT and identity metrics for tracker evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from skytally.boxes import pairwise_iou


@dataclass
class FrameBoxes:
    """Boxes and identities for one frame."""

    ids: np.ndarray
    boxes: np.ndarray


@dataclass
class TrackingScores:
    mota: float
    idf1: float
    id_switches: int
    false_negatives: int
    false_positives: int
    num_truth: int
    num_predicted: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def evaluate_tracking(
    truth: Sequence[FrameBoxes], predicted: Sequence[FrameBoxes], iou_threshold: float = 0.3
) -> TrackingScores:
    """MOTA (Bernardin and Stiefelhagen, 2008) and IDF1 (Ristani et al., 2016).

    MOTA matches frame by frame and keeps an existing truth-to-track link while it still
    overlaps. IDF1 finds the single best truth-to-track pairing over the whole sequence.
    The default threshold of 0.3 reflects how much a tiny box's IoU drops per pixel of error.
    """
    if len(truth) != len(predicted):
        raise ValueError("truth and predictions cover different frame counts")
    links: dict[int, int] = {}
    misses = false_positives = switches = 0
    num_truth = num_predicted = 0
    shared: dict[tuple[int, int], int] = {}
    truth_ids: set[int] = set()
    pred_ids: set[int] = set()

    for gt, pred in zip(truth, predicted, strict=True):
        num_truth += len(gt.ids)
        num_predicted += len(pred.ids)
        truth_ids.update(gt.ids.tolist())
        pred_ids.update(pred.ids.tolist())
        iou = pairwise_iou(gt.boxes, pred.boxes)
        rows, cols = np.nonzero(iou >= iou_threshold)
        for r, c in zip(rows, cols, strict=True):
            key = (int(gt.ids[r]), int(pred.ids[c]))
            shared[key] = shared.get(key, 0) + 1

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        gt_index = {int(v): i for i, v in enumerate(gt.ids)}
        pred_index = {int(v): i for i, v in enumerate(pred.ids)}
        for gt_id, pred_id in links.items():
            g, p = gt_index.get(gt_id), pred_index.get(pred_id)
            if g is not None and p is not None and iou[g, p] >= iou_threshold:
                matched_gt.add(g)
                matched_pred.add(p)

        free_gt = [i for i in range(len(gt.ids)) if i not in matched_gt]
        free_pred = [i for i in range(len(pred.ids)) if i not in matched_pred]
        if free_gt and free_pred:
            sub = iou[np.ix_(free_gt, free_pred)]
            sub_rows, sub_cols = linear_sum_assignment(-sub)
            for r, c in zip(sub_rows, sub_cols, strict=True):
                if sub[r, c] < iou_threshold:
                    continue
                g, p = free_gt[r], free_pred[c]
                gt_id, pred_id = int(gt.ids[g]), int(pred.ids[p])
                if gt_id in links and links[gt_id] != pred_id:
                    switches += 1
                links[gt_id] = pred_id
                matched_gt.add(g)
                matched_pred.add(p)
        misses += len(gt.ids) - len(matched_gt)
        false_positives += len(pred.ids) - len(matched_pred)

    gt_list = sorted(truth_ids)
    pred_list = sorted(pred_ids)
    id_true_positives = 0
    if gt_list and pred_list:
        overlap = np.zeros((len(gt_list), len(pred_list)), dtype=np.int64)
        gt_pos = {v: i for i, v in enumerate(gt_list)}
        pred_pos = {v: i for i, v in enumerate(pred_list)}
        for (gt_id, pred_id), count in shared.items():
            overlap[gt_pos[gt_id], pred_pos[pred_id]] = count
        rows, cols = linear_sum_assignment(-overlap)
        id_true_positives = int(overlap[rows, cols].sum())

    denominator = num_truth + num_predicted
    return TrackingScores(
        mota=1 - (misses + false_positives + switches) / max(num_truth, 1),
        idf1=2 * id_true_positives / denominator if denominator else 0.0,
        id_switches=switches,
        false_negatives=misses,
        false_positives=false_positives,
        num_truth=num_truth,
        num_predicted=num_predicted,
    )
