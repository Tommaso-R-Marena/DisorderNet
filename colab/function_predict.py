"""
Disorder → Function: multi-label IDR functional class prediction.

Predicts DisProt functional term groups (binding, PTM regulation, condensates, …)
at residue resolution, trained jointly with disorder detection.

Labels come from ``functional_regions`` already parsed in ``process_disprot``.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# Stable ordered names — must match FUNCTIONAL_TERM_GROUPS keys in disordernet_gpu
FUNCTION_GROUP_NAMES: tuple[str, ...] = (
    "protein binding",
    "nucleic acid binding",
    "post-translational regulation",
    "condensate / assembly",
    "lipid / small molecule binding",
)
N_FUNCTION_GROUPS: int = len(FUNCTION_GROUP_NAMES)


def _functional_term_groups() -> dict:
    from colab.disordernet_gpu import FUNCTIONAL_TERM_GROUPS
    return FUNCTIONAL_TERM_GROUPS


def build_function_labels(
    length: int,
    functional_regions: list,
    group_names: tuple[str, ...] = FUNCTION_GROUP_NAMES,
) -> np.ndarray:
    """
    Multi-hot residue labels (L, G) for functional term groups.

    A residue is positive for a group if any overlapping DisProt region matches
    that group's term set.
    """
    labels = np.zeros((length, len(group_names)), dtype=np.float32)
    if length <= 0 or not functional_regions:
        return labels

    name_to_idx = {n: i for i, n in enumerate(group_names)}
    for group_name, term_set in _functional_term_groups().items():
        gi = name_to_idx.get(group_name)
        if gi is None:
            continue
        for reg in functional_regions:
            term = (reg.get("term_norm") or "").strip().lower()
            if term not in term_set:
                continue
            start = int(reg.get("start", 0)) - 1
            end = min(int(reg.get("end", 0)), length)
            if start < end:
                labels[start:end, gi] = 1.0
    return labels


def function_supervise_mask(
    disorder_labels: np.ndarray,
    function_labels: np.ndarray,
    disordered_only: bool = True,
) -> np.ndarray:
    """
    Residues that contribute to function loss.

    When ``disordered_only``:
      disordered residues OR any positive function annotation
    (negatives drawn from IDRs; ordered residues without function skipped).
    """
    disorder_labels = np.asarray(disorder_labels, dtype=np.float32).ravel()
    function_labels = np.asarray(function_labels, dtype=np.float32)
    L = min(len(disorder_labels), function_labels.shape[0])
    if not disordered_only:
        return np.ones(L, dtype=bool)
    any_func = function_labels[:L].max(axis=1) > 0.5
    return (disorder_labels[:L] > 0.5) | any_func


def summarize_function_label_coverage(proteins: list) -> dict:
    """Dataset-level stats for function multilabel coverage."""
    totals = {g: 0 for g in FUNCTION_GROUP_NAMES}
    n_with_any = 0
    n_residues = 0
    n_func_residues = 0
    for p in proteins:
        fl = build_function_labels(p["length"], p.get("functional_regions", []))
        n_residues += p["length"]
        if fl.any():
            n_with_any += 1
        n_func_residues += int((fl.max(axis=1) > 0.5).sum())
        for gi, g in enumerate(FUNCTION_GROUP_NAMES):
            totals[g] += int(fl[:, gi].sum())
    return {
        "n_proteins": len(proteins),
        "n_proteins_with_function": n_with_any,
        "n_residues": n_residues,
        "n_function_residues": n_func_residues,
        "residues_per_group": totals,
        "groups": list(FUNCTION_GROUP_NAMES),
    }


class FunctionMultiLabelHead(nn.Module):
    """Lightweight multi-scale CNN → per-residue multi-label logits (B, L, G)."""

    def __init__(
        self,
        in_dim: int,
        n_groups: int = N_FUNCTION_GROUPS,
        mid: int = 192,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, mid, kernel_size=5, padding=2, dilation=1),
                nn.BatchNorm1d(mid),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(in_dim, mid, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm1d(mid),
                nn.GELU(),
            ),
        ])
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(mid * 2, mid, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(mid, n_groups, kernel_size=1),
        )
        self.skip = nn.Conv1d(in_dim, n_groups, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, L, C)
        returns: (B, L, G) logits
        """
        xc = x.permute(0, 2, 1)
        feat = torch.cat([b(xc) for b in self.branches], dim=1)
        logits = self.readout(feat) + self.skip(xc)
        logits = logits.permute(0, 2, 1)
        if pad_mask is not None:
            logits = logits.masked_fill(~pad_mask.unsqueeze(-1), 0.0)
        return logits


def compute_function_pos_weight(
    proteins: list,
    device: torch.device,
    group_names: tuple[str, ...] = FUNCTION_GROUP_NAMES,
    max_weight: float = 50.0,
) -> torch.Tensor:
    """Per-group pos_weight from training set residue counts."""
    pos = np.zeros(len(group_names), dtype=np.float64)
    neg = np.zeros(len(group_names), dtype=np.float64)
    for p in proteins:
        fl = build_function_labels(p["length"], p.get("functional_regions", []), group_names)
        dis = np.asarray(p["labels"], dtype=np.float32)
        mask = function_supervise_mask(dis, fl, disordered_only=True)
        if not mask.any():
            continue
        sub = fl[mask]
        pos += sub.sum(axis=0)
        neg += (1.0 - sub).sum(axis=0)
    weights = neg / np.maximum(pos, 1.0)
    weights = np.clip(weights, 1.0, max_weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def function_multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_mask: torch.Tensor,
    supervise_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Masked multi-label BCE.

    logits/labels: (B, L, G)
    pad_mask / supervise_mask: (B, L) bool
    """
    valid = pad_mask & supervise_mask
    if not valid.any():
        return logits.new_zeros(())
    flat_logits = logits[valid]
    flat_labels = labels[valid]
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(
            flat_logits, flat_labels, pos_weight=pos_weight, reduction="mean",
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            flat_logits, flat_labels, reduction="mean",
        )
    return loss


def stack_batch_function_labels(
    batch_ids: list[str],
    proteins_by_id: dict[str, dict],
    max_seq: int,
    device: torch.device,
    group_names: tuple[str, ...] = FUNCTION_GROUP_NAMES,
    disordered_only: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (B, L, G) function labels + (B, L) supervise mask for a collated batch.
    """
    g = len(group_names)
    labels = torch.zeros(len(batch_ids), max_seq, g, dtype=torch.float32, device=device)
    supervise = torch.zeros(len(batch_ids), max_seq, dtype=torch.bool, device=device)
    for i, pid in enumerate(batch_ids):
        p = proteins_by_id.get(pid)
        if p is None:
            continue
        fl = build_function_labels(p["length"], p.get("functional_regions", []), group_names)
        dis = np.asarray(p["labels"], dtype=np.float32)
        mask = function_supervise_mask(dis, fl, disordered_only=disordered_only)
        L = min(max_seq, fl.shape[0], len(mask))
        labels[i, :L] = torch.from_numpy(fl[:L])
        supervise[i, :L] = torch.from_numpy(mask[:L])
    return labels, supervise


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    if y_true.min() == y_true.max() or len(y_true) < 10:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=np.int8)
    if y_true.sum() == 0 or len(y_true) < 10:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def compute_function_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_names: tuple[str, ...] = FUNCTION_GROUP_NAMES,
    threshold: float = 0.5,
) -> dict:
    """
    Residue-level multi-label metrics.

    y_true / y_prob: (N, G)
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_prob = y_prob.reshape(-1, 1)
    n_groups = min(y_true.shape[1], len(group_names))
    per_group: dict = {}
    aucs: list[float] = []
    aps: list[float] = []
    f1s: list[float] = []

    for gi in range(n_groups):
        name = group_names[gi]
        yt = y_true[:, gi]
        yp = y_prob[:, gi]
        pred = (yp >= threshold).astype(np.int8)
        auc = _safe_auc(yt, yp)
        ap = _safe_ap(yt, yp)
        f1 = float(f1_score(yt.astype(int), pred, zero_division=0)) if yt.sum() > 0 else None
        entry = {
            "n_positives": int(yt.sum()),
            "n_residues": int(len(yt)),
            "prevalence": float(yt.mean()) if len(yt) else 0.0,
            "auc": auc,
            "ap": ap,
            "f1": f1,
        }
        per_group[name] = entry
        if auc is not None:
            aucs.append(auc)
        if ap is not None:
            aps.append(ap)
        if f1 is not None:
            f1s.append(f1)

    # Micro: flatten all groups
    yt_flat = y_true[:, :n_groups].ravel()
    yp_flat = y_prob[:, :n_groups].ravel()
    pred_flat = (yp_flat >= threshold).astype(int)
    micro = {
        "auc": _safe_auc(yt_flat, yp_flat),
        "ap": _safe_ap(yt_flat, yp_flat),
        "f1": float(f1_score(yt_flat.astype(int), pred_flat, zero_division=0)),
        "n_positives": int(yt_flat.sum()),
        "n_pairs": int(len(yt_flat)),
    }

    return {
        "groups": list(group_names[:n_groups]),
        "per_group": per_group,
        "macro_auc": float(np.mean(aucs)) if aucs else None,
        "macro_ap": float(np.mean(aps)) if aps else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "micro": micro,
        "n_groups_with_auc": len(aucs),
        "threshold": threshold,
    }


def align_function_oof(
    proteins: list,
    fold_results: list,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Concatenate OOF function labels/probs when stored on fold results.

    Returns (y_true, y_prob, protein_ids for each val protein in fold order).
    """
    if not any(
        fr.get("val_function_probs") is not None and fr.get("val_function_labels") is not None
        for fr in fold_results
    ):
        return (
            np.zeros((0, N_FUNCTION_GROUPS), dtype=np.float32),
            np.zeros((0, N_FUNCTION_GROUPS), dtype=np.float32),
            [],
        )

    from colab.cv_splits import get_cv_splits

    splits = get_cv_splits(proteins, n_folds)
    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    protein_ids: list[str] = []

    for fold_idx, (_, val_idx) in enumerate(splits):
        if fold_idx >= len(fold_results):
            break
        fr = fold_results[fold_idx]
        func_probs = fr.get("val_function_probs")
        func_labels = fr.get("val_function_labels")
        if func_probs is None or func_labels is None:
            continue
        y_prob_parts.append(np.asarray(func_probs, dtype=np.float32))
        y_true_parts.append(np.asarray(func_labels, dtype=np.float32))
        for i in val_idx:
            protein_ids.append(proteins[i]["id"])

    if not y_true_parts:
        return (
            np.zeros((0, N_FUNCTION_GROUPS), dtype=np.float32),
            np.zeros((0, N_FUNCTION_GROUPS), dtype=np.float32),
            [],
        )
    return np.concatenate(y_true_parts), np.concatenate(y_prob_parts), protein_ids


def split_function_oof_by_lengths(
    y_prob: np.ndarray,
    protein_ids: list[str],
    lengths_by_id: dict[str, int],
) -> dict[str, np.ndarray]:
    """
    Slice flat OOF function probs into per-protein (L, G) maps.

    Requires ``sum(lengths) == len(y_prob)`` — the contract after full-sequence
    (pad-mask) function export in ``eval_epoch``.
    """
    if y_prob is None or len(y_prob) == 0:
        return {}
    y = np.asarray(y_prob, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, N_FUNCTION_GROUPS)
    expected = sum(int(lengths_by_id[pid]) for pid in protein_ids)
    if expected != len(y):
        raise ValueError(
            f"Function OOF length {len(y)} ≠ sum of protein lengths {expected}"
        )
    out: dict[str, np.ndarray] = {}
    offset = 0
    for pid in protein_ids:
        L = int(lengths_by_id[pid])
        out[pid] = y[offset:offset + L]
        offset += L
    return out


def run_function_prediction_report(
    proteins: list,
    fold_results: list,
    n_folds: int = 5,
    threshold: float = 0.5,
) -> dict:
    """Pooled OOF multi-label function metrics + coverage."""
    coverage = summarize_function_label_coverage(proteins)
    y_true, y_prob, _ = align_function_oof(proteins, fold_results, n_folds=n_folds)
    if len(y_true) == 0:
        return {
            "enabled": False,
            "insufficient_data": True,
            "reason": "No val_function_probs in fold results — train with use_function_head=True",
            "label_coverage": coverage,
        }
    metrics_all = compute_function_metrics(y_true, y_prob, threshold=threshold)

    # Also restrict to disordered residues (matches training supervise prior)
    metrics_disordered = None
    try:
        from colab.biological_utility import align_fold_predictions
        aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
        dis_parts = [item["labels"] for item in aligned]
        if dis_parts:
            dis = np.concatenate([np.asarray(d, dtype=np.float32).ravel() for d in dis_parts])
            if len(dis) == len(y_true):
                m = dis > 0.5
                if int(m.sum()) >= 10:
                    metrics_disordered = compute_function_metrics(
                        y_true[m], y_prob[m], threshold=threshold,
                    )
    except Exception:
        metrics_disordered = None

    return {
        "enabled": True,
        "insufficient_data": False,
        "label_coverage": coverage,
        "metrics": metrics_all,
        "metrics_on_disordered": metrics_disordered,
        "n_oof_residues": int(y_true.shape[0]),
        "use_case": "Disorder → function (multi-label IDR roles)",
        "note": (
            "metrics = all OOF residues (aligned to disorder stream); "
            "metrics_on_disordered = disordered residues only"
        ),
    }


def print_function_report(report: dict) -> None:
    print(f"\n{'═' * 60}")
    print(" Disorder → Function (multi-label)")
    print(f"{'═' * 60}")
    if report.get("insufficient_data") or not report.get("enabled"):
        print(f"  {report.get('reason', 'insufficient data')}")
        cov = report.get("label_coverage", {})
        if cov:
            print(
                f"  Label coverage: {cov.get('n_proteins_with_function', 0)}/"
                f"{cov.get('n_proteins', 0)} proteins, "
                f"{cov.get('n_function_residues', 0):,} function residues"
            )
        return
    m = report["metrics"]
    print(
        f"  OOF residues: {report['n_oof_residues']:,}  │  "
        f"macro AUC={m.get('macro_auc')}  macro AP={m.get('macro_ap')}  "
        f"micro F1={m['micro'].get('f1')}"
    )
    for name, g in m.get("per_group", {}).items():
        print(
            f"    {name:32s}  n+={g['n_positives']:6d}  "
            f"AUC={g['auc']}  AP={g['ap']}  F1={g['f1']}"
        )


def save_function_report(report: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def predict_protein_functions(
    disorder_probs: np.ndarray,
    function_probs: np.ndarray,
    sequence: str,
    protein_id: str = "query",
    disorder_threshold: float = 0.5,
    function_threshold: float = 0.5,
    group_names: tuple[str, ...] = FUNCTION_GROUP_NAMES,
    min_region_len: int = 5,
) -> dict:
    """
    Export IDR functional roles for a single protein (novel use case).

    Only reports function calls inside predicted disordered segments.
    """
    from colab.biological_utility import intervals_from_binary

    n = min(len(sequence), len(disorder_probs), len(function_probs))
    dis_mask = disorder_probs[:n] >= disorder_threshold
    segs = intervals_from_binary(dis_mask.astype(np.int8), min_len=min_region_len)
    regions: list[dict] = []
    for start, end in segs:
        slice_probs = function_probs[start:end]
        mean_p = slice_probs.mean(axis=0)
        max_p = slice_probs.max(axis=0)
        roles = []
        for gi, name in enumerate(group_names):
            if gi >= mean_p.shape[0]:
                break
            if mean_p[gi] >= function_threshold or max_p[gi] >= function_threshold + 0.15:
                roles.append({
                    "group": name,
                    "mean_prob": float(mean_p[gi]),
                    "max_prob": float(max_p[gi]),
                })
        roles.sort(key=lambda r: -r["mean_prob"])
        regions.append({
            "start": start + 1,
            "end": end,
            "length": end - start,
            "mean_disorder_prob": float(disorder_probs[start:end].mean()),
            "predicted_roles": roles,
        })
    return {
        "protein_id": protein_id,
        "length": n,
        "n_idr_segments": len(regions),
        "idr_function_regions": regions,
        "use_case": "Annotate functional roles of predicted IDRs",
    }
