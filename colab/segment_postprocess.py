"""
Segment post-processing and segment-aware evaluation utilities.

Improves region-level F1 by closing short gaps and filtering spurious short
predicted IDR segments before segment matching.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from colab.biological_utility import compute_segment_metrics, intervals_from_binary


def postprocess_binary(
    binary: np.ndarray,
    min_len: int = 5,
    max_gap: int = 3,
) -> np.ndarray:
    """
    Post-process a binary disorder prediction sequence.

    1. Close short gaps (runs of 0) between predicted disorder segments.
    2. Remove predicted disorder segments shorter than min_len.
    """
    binary = np.asarray(binary, dtype=np.int8).ravel().copy()
    if len(binary) == 0:
        return binary

    if max_gap > 0:
        binary = _close_gaps(binary, max_gap=max_gap)

    if min_len > 1:
        binary = _remove_short_runs(binary, min_len=min_len, value=1)

    return binary


def _close_gaps(binary: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill runs of 0 shorter than max_gap when sandwiched between 1s."""
    out = binary.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != 0:
            i += 1
            continue
        gap_start = i
        while i < n and out[i] == 0:
            i += 1
        gap_end = i
        gap_len = gap_end - gap_start
        has_left = gap_start > 0 and out[gap_start - 1] == 1
        has_right = gap_end < n and out[gap_end] == 1
        if has_left and has_right and gap_len <= max_gap:
            out[gap_start:gap_end] = 1
    return out


def _remove_short_runs(binary: np.ndarray, min_len: int, value: int = 1) -> np.ndarray:
    """Zero out runs of `value` shorter than min_len."""
    out = binary.copy()
    for start, end in intervals_from_binary(out == value, min_len=1):
        if end - start < min_len:
            out[start:end] = 1 - value
    return out


def probs_to_postprocessed_binary(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_len: int = 5,
    max_gap: int = 3,
) -> np.ndarray:
    """Threshold probabilities then apply segment post-processing."""
    binary = (np.asarray(probs, dtype=np.float32) >= threshold).astype(np.int8)
    return postprocess_binary(binary, min_len=min_len, max_gap=max_gap)


def segment_f1_for_protein(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    min_region_len: int = 5,
    postprocess_min_len: int = 5,
    postprocess_max_gap: int = 3,
    iou_threshold: float = 0.5,
    apply_postprocess: bool = True,
) -> float:
    """Macro segment F1 for one protein (optionally with post-processing)."""
    labels = np.asarray(labels, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float32)

    if apply_postprocess:
        preds = probs_to_postprocessed_binary(
            probs,
            threshold=threshold,
            min_len=postprocess_min_len,
            max_gap=postprocess_max_gap,
        )
    else:
        preds = (probs >= threshold).astype(np.int8)

    metrics = compute_segment_metrics(
        labels,
        preds,
        iou_threshold=iou_threshold,
        min_region_len=min_region_len,
    )
    return metrics.segment_f1


def pooled_segment_f1(
    proteins: list,
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    min_region_len: int = 5,
    postprocess_min_len: int = 5,
    postprocess_max_gap: int = 3,
    iou_threshold: float = 0.5,
    apply_postprocess: bool = True,
) -> float:
    """
    Mean segment F1 across proteins (matches biological_utility aggregation).

    `probs` and `labels` must be concatenated in the same order as `proteins`.
    """
    probs = np.asarray(probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    offset = 0
    f1_scores: list[float] = []

    for p in proteins:
        length = p["length"]
        p_probs = probs[offset:offset + length]
        p_labels = labels[offset:offset + length]
        offset += length

        f1_scores.append(segment_f1_for_protein(
            p_labels,
            p_probs,
            threshold=threshold,
            min_region_len=min_region_len,
            postprocess_min_len=postprocess_min_len,
            postprocess_max_gap=postprocess_max_gap,
            iou_threshold=iou_threshold,
            apply_postprocess=apply_postprocess,
        ))

    if offset != len(probs):
        raise ValueError(
            f"Prediction length mismatch: expected {offset}, got {len(probs)}",
        )

    return float(np.mean(f1_scores)) if f1_scores else 0.0


def composite_early_stop_score(
    auc: float,
    ap: float,
    segment_f1: float,
    auc_weight: float = 0.5,
    ap_weight: float = 0.3,
    segment_weight: float = 0.2,
) -> float:
    """Weighted composite score for checkpoint selection."""
    return auc_weight * auc + ap_weight * ap + segment_weight * segment_f1
