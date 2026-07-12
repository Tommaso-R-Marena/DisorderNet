"""
Biological utility evaluation for DisorderNet (Phase 1).

Goes beyond residue-level AUC with:
  - Functional term enrichment (binding, PTMs, condensates, …)
  - Region/segment metrics (IoU, F1, MDR, boundary error)
  - Transition-zone performance (disorder↔order boundaries)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from colab.disordernet_gpu import FUNCTIONAL_TERM_GROUPS


# ---------------------------------------------------------------------------
# Interval utilities
# ---------------------------------------------------------------------------
Interval = tuple[int, int]  # half-open [start, end)


def intervals_from_binary(binary: np.ndarray, min_len: int = 1) -> list[Interval]:
    """Extract contiguous runs of 1s as half-open intervals."""
    binary = np.asarray(binary, dtype=np.int8).ravel()
    intervals: list[Interval] = []
    start: Optional[int] = None
    for i, v in enumerate(binary):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                intervals.append((start, i))
            start = None
    if start is not None and len(binary) - start >= min_len:
        intervals.append((start, len(binary)))
    return intervals


def interval_iou(a: Interval, b: Interval) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def _best_match_iou(pred: Interval, truth_intervals: list[Interval]) -> float:
    if not truth_intervals:
        return 0.0
    return max(interval_iou(pred, t) for t in truth_intervals)


def _best_match_boundary_error(pred: Interval, truth_intervals: list[Interval]) -> float:
    if not truth_intervals:
        return float("nan")
    best_t = max(truth_intervals, key=lambda t: interval_iou(pred, t))
    return abs(pred[0] - best_t[0]) + abs(pred[1] - best_t[1])


# ---------------------------------------------------------------------------
# Region-level metrics
# ---------------------------------------------------------------------------
@dataclass
class SegmentMetrics:
    segment_precision: float
    segment_recall: float
    segment_f1: float
    mean_segment_iou: float
    mdr_recall: float
    mean_boundary_error: float
    n_true_segments: int
    n_pred_segments: int


def compute_segment_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iou_threshold: float = 0.5,
    min_region_len: int = 5,
) -> SegmentMetrics:
    """
    Segment-level precision/recall/F1 via IoU matching (greedy, CAID-style).

    MDR recall: fraction of true disordered segments (length >= min_region_len)
    matched by any predicted segment with IoU >= iou_threshold.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    true_segs = intervals_from_binary(y_true, min_len=min_region_len)
    pred_segs = intervals_from_binary(y_pred, min_len=1)

    if not pred_segs and not true_segs:
        return SegmentMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0, 0)
    if not pred_segs:
        return SegmentMetrics(0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), len(true_segs), 0)
    if not true_segs:
        return SegmentMetrics(0.0, 0.0, 0.0, 0.0, 1.0, float("nan"), 0, len(pred_segs))

    matched_true = set()
    matched_pred = set()
    ious: list[float] = []
    boundary_errors: list[float] = []

    for pi, pred in enumerate(pred_segs):
        best_iou = 0.0
        best_ti = -1
        for ti, truth in enumerate(true_segs):
            iou = interval_iou(pred, truth)
            if iou > best_iou:
                best_iou = iou
                best_ti = ti
        if best_iou >= iou_threshold and best_ti >= 0:
            matched_pred.add(pi)
            matched_true.add(best_ti)
            ious.append(best_iou)
            boundary_errors.append(_best_match_boundary_error(pred, [true_segs[best_ti]]))

    seg_prec = len(matched_pred) / len(pred_segs)
    seg_rec = len(matched_true) / len(true_segs)
    seg_f1 = (
        2 * seg_prec * seg_rec / (seg_prec + seg_rec)
        if (seg_prec + seg_rec) > 0 else 0.0
    )
    mdr_rec = len(matched_true) / len(true_segs)
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_be = float(np.nanmean(boundary_errors)) if boundary_errors else float("nan")

    return SegmentMetrics(
        segment_precision=seg_prec,
        segment_recall=seg_rec,
        segment_f1=seg_f1,
        mean_segment_iou=mean_iou,
        mdr_recall=mdr_rec,
        mean_boundary_error=mean_be,
        n_true_segments=len(true_segs),
        n_pred_segments=len(pred_segs),
    )


# ---------------------------------------------------------------------------
# Functional enrichment
# ---------------------------------------------------------------------------
def _mask_from_regions(length: int, regions: list, term_filter: Optional[set] = None) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for reg in regions:
        term = reg.get("term_norm") or reg.get("term_name", "").strip().lower()
        if term_filter is not None and term not in term_filter:
            continue
        start = reg.get("start", 0) - 1
        end = min(reg.get("end", 0), length)
        if start < end:
            mask[start:end] = True
    return mask


def compute_functional_enrichment(
    y_pred: np.ndarray,
    probs: np.ndarray,
    functional_regions: list,
    disorder_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    For each functional term group:
      - recall@function: P(predicted disorder | annotated with function)
      - precision@IDR:   P(function annotation | predicted disorder)
      - enrichment:      recall@function / global disorder rate
    """
    y_pred = np.asarray(y_pred, dtype=np.int8)
    probs = np.asarray(probs, dtype=np.float32)
    disorder_labels = np.asarray(disorder_labels, dtype=np.int8)
    length = len(y_pred)

    pred_mask = y_pred.astype(bool)
    disorder_rate = float(disorder_labels.mean()) if length else 0.0

    results: dict = {}
    for group_name, term_set in FUNCTIONAL_TERM_GROUPS.items():
        func_mask = _mask_from_regions(length, functional_regions, term_set)
        n_func = int(func_mask.sum())
        if n_func == 0:
            continue

        n_pred_and_func = int((pred_mask & func_mask).sum())
        n_pred = int(pred_mask.sum())
        recall_func = n_pred_and_func / n_func
        precision_idr = n_pred_and_func / n_pred if n_pred > 0 else 0.0
        enrichment = recall_func / disorder_rate if disorder_rate > 0 else 0.0

        results[group_name] = {
            "n_residues": n_func,
            "recall_at_function": float(recall_func),
            "precision_at_idr": float(precision_idr),
            "enrichment_vs_disorder_rate": float(enrichment),
        }

    # Top individual DisProt terms (excluding pure disorder labels)
    term_counts: dict[str, dict] = {}
    for reg in functional_regions:
        term = reg.get("term_norm") or ""
        if not term or term in {"disorder", "flexible linker", "flexible n-terminal tail",
                                "flexible c-terminal tail", "pre-molten globule",
                                "molten globule", "entropic chain"}:
            continue
        if term not in term_counts:
            term_counts[term] = {"n": 0, "hits": 0}
        s, e = reg["start"] - 1, min(reg["end"], length)
        for i in range(s, e):
            term_counts[term]["n"] += 1
            if pred_mask[i]:
                term_counts[term]["hits"] += 1

    top_terms = sorted(
        (
            {
                "term": t,
                "n_residues": v["n"],
                "recall_at_function": v["hits"] / v["n"] if v["n"] else 0.0,
            }
            for t, v in term_counts.items()
        ),
        key=lambda x: x["n_residues"],
        reverse=True,
    )[:15]

    results["_top_individual_terms"] = top_terms
    return results


def compute_transition_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    transition_mask: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Residue metrics restricted to disorder↔order transition zones."""
    transition_mask = np.asarray(transition_mask, dtype=bool)
    if transition_mask.sum() < 10:
        return {"n_residues": int(transition_mask.sum()), "insufficient_data": True}

    probs_t = probs[transition_mask]
    labels_t = labels[transition_mask]
    preds_t = (probs_t >= threshold).astype(int)

    return {
        "n_residues": int(transition_mask.sum()),
        "insufficient_data": False,
        "auc": float(roc_auc_score(labels_t, probs_t)),
        "ap": float(average_precision_score(labels_t, probs_t)),
        "f1": float(f1_score(labels_t, preds_t, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_t, preds_t)),
        "disorder_fraction": float(labels_t.mean()),
    }


# ---------------------------------------------------------------------------
# Align CV predictions to proteins
# ---------------------------------------------------------------------------
def align_fold_predictions(
    proteins: list,
    fold_results: list,
    n_folds: int = 5,
) -> list[dict]:
    """
    Map each protein to its out-of-fold predictions (pooled CV).

    Returns list of dicts with keys: id, labels, probs, preds, protein, fold.
    """
    gkf = GroupKFold(n_splits=n_folds)
    groups = np.arange(len(proteins))
    aligned: list[dict] = []

    for fold_idx, (_, val_idx) in enumerate(gkf.split(groups, groups=groups)):
        if fold_idx >= len(fold_results):
            break
        val_proteins = [proteins[i] for i in val_idx]
        val_probs = fold_results[fold_idx]["val_probs"]
        val_labels = fold_results[fold_idx]["val_labels"]
        offset = 0
        for p in val_proteins:
            L = p["length"]
            aligned.append({
                "id": p["id"],
                "fold": fold_idx + 1,
                "labels": np.asarray(val_labels[offset:offset + L], dtype=np.float32),
                "probs": np.asarray(val_probs[offset:offset + L], dtype=np.float32),
                "protein": p,
            })
            offset += L
        assert offset == len(val_probs), "Prediction length mismatch in fold alignment"

    return aligned


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def run_biological_utility_report(
    proteins: list,
    fold_results: list,
    threshold: Optional[float] = None,
    n_folds: int = 5,
    min_region_len: int = 5,
    iou_threshold: float = 0.5,
    apply_postprocess: bool = True,
    postprocess_min_len: int = 5,
    postprocess_max_gap: int = 3,
) -> dict:
    """
    Compute pooled biological-utility metrics across all CV folds.
    """
    if threshold is None:
        threshold = 0.5

    from colab.segment_postprocess import probs_to_postprocessed_binary

    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    all_seg: list[SegmentMetrics] = []
    func_agg: dict[str, list] = {g: [] for g in FUNCTIONAL_TERM_GROUPS}
    transition_probs: list = []
    transition_labels: list = []
    transition_masks_n = 0

    for item in aligned:
        p = item["protein"]
        labels = item["labels"]
        probs = item["probs"]
        preds = (
            probs_to_postprocessed_binary(
                probs,
                threshold=threshold,
                min_len=postprocess_min_len,
                max_gap=postprocess_max_gap,
            )
            if apply_postprocess
            else (probs >= threshold).astype(np.int8)
        )

        all_seg.append(compute_segment_metrics(
            labels, preds, iou_threshold=iou_threshold, min_region_len=min_region_len,
        ))

        func = compute_functional_enrichment(
            preds, probs, p.get("functional_regions", []), labels, threshold=threshold,
        )
        for group_name in FUNCTIONAL_TERM_GROUPS:
            if group_name in func:
                func_agg[group_name].append(func[group_name])

        tmask = np.asarray(p.get("transition_mask", [0] * len(labels)), dtype=bool)
        if tmask.any():
            transition_probs.append(probs[tmask])
            transition_labels.append(labels[tmask])
            transition_masks_n += int(tmask.sum())

    # Pooled segment metrics (mean across proteins)
    seg_summary = {
        "segment_precision": float(np.mean([s.segment_precision for s in all_seg])),
        "segment_recall": float(np.mean([s.segment_recall for s in all_seg])),
        "segment_f1": float(np.mean([s.segment_f1 for s in all_seg])),
        "mean_segment_iou": float(np.mean([s.mean_segment_iou for s in all_seg])),
        "mdr_recall": float(np.mean([s.mdr_recall for s in all_seg])),
        "mean_boundary_error": float(np.nanmean([s.mean_boundary_error for s in all_seg])),
        "n_proteins": len(all_seg),
        "total_true_segments": int(sum(s.n_true_segments for s in all_seg)),
        "total_pred_segments": int(sum(s.n_pred_segments for s in all_seg)),
        "min_region_len": min_region_len,
        "iou_threshold": iou_threshold,
        "apply_postprocess": apply_postprocess,
        "postprocess_min_len": postprocess_min_len,
        "postprocess_max_gap": postprocess_max_gap,
    }

    # Pooled functional enrichment (weighted by n_residues)
    functional_summary: dict = {}
    for group_name, entries in func_agg.items():
        if not entries:
            continue
        total_n = sum(e["n_residues"] for e in entries)
        functional_summary[group_name] = {
            "n_residues": total_n,
            "recall_at_function": float(sum(
                e["recall_at_function"] * e["n_residues"] for e in entries
            ) / total_n),
            "precision_at_idr": float(np.mean([e["precision_at_idr"] for e in entries])),
            "enrichment_vs_disorder_rate": float(sum(
                e["enrichment_vs_disorder_rate"] * e["n_residues"] for e in entries
            ) / total_n),
        }

    # Aggregate top individual terms across proteins
    term_hits: dict[str, dict] = {}
    for item in aligned:
        p = item["protein"]
        preds = (item["probs"] >= threshold).astype(np.int8)
        func = compute_functional_enrichment(
            preds, item["probs"], p.get("functional_regions", []), item["labels"],
            threshold=threshold,
        )
        for tinfo in func.get("_top_individual_terms", []):
            t = tinfo["term"]
            if t not in term_hits:
                term_hits[t] = {"n": 0, "hits": 0}
            term_hits[t]["n"] += tinfo["n_residues"]
            term_hits[t]["hits"] += int(tinfo["recall_at_function"] * tinfo["n_residues"])

    top_terms = sorted(
        [
            {
                "term": t,
                "n_residues": v["n"],
                "recall_at_function": v["hits"] / v["n"] if v["n"] else 0.0,
            }
            for t, v in term_hits.items()
        ],
        key=lambda x: x["n_residues"],
        reverse=True,
    )[:15]

    # Transition zone metrics
    if transition_probs:
        tp = np.concatenate(transition_probs)
        tl = np.concatenate(transition_labels)
        preds_t = (tp >= threshold).astype(int)
        transition_summary = {
            "n_residues": int(transition_masks_n),
            "insufficient_data": False,
            "auc": float(roc_auc_score(tl, tp)),
            "ap": float(average_precision_score(tl, tp)),
            "f1": float(f1_score(tl, preds_t, zero_division=0)),
            "mcc": float(matthews_corrcoef(tl, preds_t)),
            "disorder_fraction": float(tl.mean()),
        }
    else:
        transition_summary = {"n_residues": 0, "insufficient_data": True}

    report = {
        "threshold": float(threshold),
        "segment_metrics": seg_summary,
        "functional_enrichment": functional_summary,
        "top_functional_terms": top_terms,
        "transition_zones": transition_summary,
    }
    return report


def print_biological_utility_report(report: dict) -> None:
    """Pretty-print biological utility report to stdout."""
    print(f"\n{'═' * 64}")
    print(" BIOLOGICAL UTILITY REPORT (Phase 1)")
    print(f"{'═' * 64}")
    print(f"  Threshold: {report['threshold']:.3f}")

    seg = report["segment_metrics"]
    print(f"\n── Region / segment metrics ({seg['n_proteins']} proteins) ──")
    print(f"  Segment precision : {seg['segment_precision']:.4f}")
    print(f"  Segment recall    : {seg['segment_recall']:.4f}")
    print(f"  Segment F1        : {seg['segment_f1']:.4f}")
    print(f"  Mean segment IoU  : {seg['mean_segment_iou']:.4f}")
    print(f"  MDR recall        : {seg['mdr_recall']:.4f}  "
          f"(IoU≥{seg['iou_threshold']}, min len {seg['min_region_len']})")
    be = seg["mean_boundary_error"]
    print(f"  Boundary error    : {be:.2f} residues" if not np.isnan(be) else "  Boundary error    : N/A")
    print(f"  True / pred segs  : {seg['total_true_segments']} / {seg['total_pred_segments']}")

    func = report["functional_enrichment"]
    if func:
        print(f"\n── Functional enrichment (grouped DisProt terms) ──")
        print(f"  {'Group':<32} {'N':>7} {'Recall@Fn':>10} {'Prec@IDR':>10} {'Enrich':>8}")
        print(f"  {'─' * 62}")
        for name, m in sorted(func.items(), key=lambda x: -x[1]["n_residues"]):
            print(
                f"  {name:<32} {m['n_residues']:>7,} "
                f"{m['recall_at_function']:>10.3f} {m['precision_at_idr']:>10.3f} "
                f"{m['enrichment_vs_disorder_rate']:>8.2f}x"
            )

    top = report.get("top_functional_terms", [])
    if top:
        print(f"\n── Top individual functional terms ──")
        for t in top[:8]:
            print(f"  {t['term']:<40} n={t['n_residues']:>5,}  "
                  f"recall@function={t['recall_at_function']:.3f}")

    tr = report["transition_zones"]
    print(f"\n── Disorder↔order transition zones ──")
    if tr.get("insufficient_data"):
        print(f"  Insufficient transition annotations (n={tr.get('n_residues', 0)})")
    else:
        print(f"  Residues : {tr['n_residues']:,}")
        print(f"  AUC      : {tr['auc']:.4f}")
        print(f"  AP       : {tr['ap']:.4f}")
        print(f"  F1       : {tr['f1']:.4f}")
        print(f"  MCC      : {tr['mcc']:.4f}")

    print(f"{'═' * 64}")


def save_biological_utility_report(report: dict, path: str = "biological_utility_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
