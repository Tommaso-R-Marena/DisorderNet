"""
GPU + v6 CPU ensemble for DisorderNet Colab CV predictions.

Blends out-of-fold GPU probabilities with a lightweight v6-style CPU model
(features_fast + gradient boosting). Blend weight w is tuned on OOF residues.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from colab.cv_splits import get_cv_splits

from colab.biological_utility import align_fold_predictions
from colab.inference_fusion import compute_pooled_metrics, write_fused_probs_to_fold_results
from features_fast import compute_features_fast


def build_v6_features(sequence: str) -> np.ndarray:
    """Per-residue v6-style physics features (no ESM dependency)."""
    return compute_features_fast(sequence).astype(np.float32)


def _stack_protein_features(
    proteins: list,
    verbose: bool = False,
    desc: str = "features",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return (X, y, protein_lengths) for all proteins."""
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    lengths: list[int] = []

    iterator = proteins
    if verbose and len(proteins) > 3:
        iterator = tqdm(proteins, desc=f"  {desc}", leave=False)

    for p in iterator:
        feats = build_v6_features(p["sequence"])
        labels = np.asarray(p["labels"], dtype=np.float32)
        length = min(len(labels), feats.shape[0])
        chunks_x.append(feats[:length])
        chunks_y.append(labels[:length])
        lengths.append(length)

    return (
        np.nan_to_num(np.vstack(chunks_x), nan=0.0, posinf=0.0, neginf=0.0),
        np.concatenate(chunks_y),
        lengths,
    )


def _train_v6_lite_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> HistGradientBoostingClassifier:
    """Train a lightweight v6 proxy on one fold."""
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    weight_ratio = min(neg / max(pos, 1), 8.0)
    sample_weight = np.where(y_train > 0.5, weight_ratio, 1.0).astype(np.float32)

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=120,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    model.fit(X_train, y_train.astype(np.int8), sample_weight=sample_weight)
    return model


def run_v6_lite_oof(
    proteins: list,
    n_folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Run lightweight v6-style OOF predictions aligned with GPU CV folds.

    Returns (oof_probs, oof_labels, per_fold_metadata).
    """
    splits = get_cv_splits(proteins, n_folds)
    oof_probs_by_id: dict[str, np.ndarray] = {}
    oof_labels_by_id: dict[str, np.ndarray] = {}
    fold_meta: list[dict] = []
    n_res = sum(p["length"] for p in proteins)

    if verbose:
        print(
            f"\n{'─' * 60}\n"
            f" v6-lite OOF  │  {len(proteins)} proteins  │  {n_res:,} residues  │  "
            f"{n_folds} folds\n"
            f"{'─' * 60}",
            flush=True,
        )

    cv_t0 = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_t0 = time.time()
        train_proteins = [proteins[i] for i in train_idx]
        val_proteins = [proteins[i] for i in val_idx]

        if verbose:
            print(
                f"\n  Fold {fold_idx + 1}/{n_folds}  "
                f"(train={len(train_proteins)}, val={len(val_proteins)})",
                flush=True,
            )

        if verbose:
            print("    → stacking train features…", flush=True)
        X_train, y_train, _ = _stack_protein_features(
            train_proteins, verbose=verbose, desc="train features",
        )
        if verbose:
            print(
                f"    → stacking val features ({len(val_proteins)} proteins)…",
                flush=True,
            )
        X_val, y_val, val_lengths = _stack_protein_features(
            val_proteins, verbose=verbose, desc="val features",
        )

        if verbose:
            print(
                f"    → training HGB ({X_train.shape[0]:,} train residues)…",
                flush=True,
            )
        model = _train_v6_lite_fold(X_train, y_train, seed=seed + fold_idx)

        if verbose:
            print(f"    → predicting val ({X_val.shape[0]:,} residues)…", flush=True)
        val_probs = model.predict_proba(X_val)[:, 1].astype(np.float32)

        val_offset = 0
        for p in val_proteins:
            length = p["length"]
            oof_probs_by_id[p["id"]] = val_probs[val_offset:val_offset + length].copy()
            oof_labels_by_id[p["id"]] = y_val[val_offset:val_offset + length].copy()
            val_offset += length

        val_auc = (
            float(roc_auc_score(y_val, val_probs))
            if len(np.unique(y_val)) > 1 else None
        )
        fold_meta.append({
            "fold": fold_idx + 1,
            "n_train": len(train_proteins),
            "n_val": len(val_proteins),
            "val_auc": val_auc,
            "val_lengths": val_lengths,
        })

        if verbose:
            auc_s = f"{val_auc:.4f}" if val_auc is not None else "n/a"
            elapsed = time.time() - fold_t0
            total = time.time() - cv_t0
            eta = (total / (fold_idx + 1)) * (n_folds - fold_idx - 1)
            print(
                f"    ✓ fold {fold_idx + 1} done  val_AUC={auc_s}  "
                f"[{elapsed:.0f}s fold, {total / 60:.1f}m elapsed, ~{eta / 60:.1f}m left]",
                flush=True,
            )

    oof_probs = np.concatenate([oof_probs_by_id[p["id"]] for p in proteins])
    oof_labels = np.concatenate([oof_labels_by_id[p["id"]] for p in proteins])

    if verbose:
        pooled_auc = float(roc_auc_score(oof_labels, oof_probs))
        print(
            f"\n  v6-lite OOF complete  pooled_AUC={pooled_auc:.4f}  "
            f"total={(time.time() - cv_t0) / 60:.1f} min\n",
            flush=True,
        )

    return oof_probs, oof_labels, fold_meta


def load_v6_probs_cache(path: str) -> Optional[dict[str, np.ndarray]]:
    """Load per-protein v6 probabilities from JSON cache."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {k: np.asarray(v, dtype=np.float32) for k, v in data.get("probs_by_id", {}).items()}


def save_v6_probs_cache(probs_by_id: dict[str, np.ndarray], path: str) -> str:
    with open(path, "w") as f:
        json.dump(
            {"probs_by_id": {k: v.tolist() for k, v in probs_by_id.items()}},
            f,
            indent=2,
        )
    return path


def aligned_probs_from_oof(
    proteins: list,
    oof_probs: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map concatenated OOF probs to protein id."""
    by_id: dict[str, np.ndarray] = {}
    offset = 0
    for p in proteins:
        length = p["length"]
        by_id[p["id"]] = oof_probs[offset:offset + length].copy()
        offset += length
    return by_id


def find_optimal_blend_weight(
    labels: np.ndarray,
    gpu_probs: np.ndarray,
    v6_probs: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> dict:
    """Grid-search GPU/v6 blend weight maximizing pooled AUC."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 21)

    labels = np.asarray(labels, dtype=np.int8)
    gpu_probs = np.asarray(gpu_probs, dtype=np.float32)
    v6_probs = np.asarray(v6_probs, dtype=np.float32)

    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return {"best_weight": 0.5, "best_auc": None, "insufficient_data": True, "curve": []}

    best_w = 0.5
    best_auc = -1.0
    curve: list[dict] = []

    for w in grid:
        blended = (1.0 - w) * gpu_probs + w * v6_probs
        auc = float(roc_auc_score(labels, blended))
        ap = float(average_precision_score(labels, blended))
        curve.append({"weight": float(w), "auc": auc, "ap": ap})
        if auc > best_auc:
            best_auc = auc
            best_w = float(w)

    return {
        "best_weight": best_w,
        "best_auc": best_auc,
        "curve": curve,
    }


def blend_aligned_predictions(
    aligned: list[dict],
    v6_probs_by_id: dict[str, np.ndarray],
    weight: float,
) -> list[dict]:
    """Blend GPU probs with v6 probs per protein."""
    w = float(weight)
    updated: list[dict] = []

    for item in aligned:
        pid = item["id"]
        gpu_probs = np.asarray(item["probs"], dtype=np.float32)
        if pid not in v6_probs_by_id:
            updated.append(item)
            continue

        v6_probs = np.asarray(v6_probs_by_id[pid], dtype=np.float32)
        if len(v6_probs) != len(gpu_probs):
            updated.append(item)
            continue

        blended = (1.0 - w) * gpu_probs + w * v6_probs
        new_item = dict(item)
        new_item["probs"] = blended.astype(np.float32)
        new_item["gpu_probs"] = gpu_probs
        new_item["v6_probs"] = v6_probs
        updated.append(new_item)

    return updated


def apply_gpu_v6_ensemble(
    proteins: list,
    fold_results: list,
    n_folds: int = 5,
    weight: Optional[float] = None,
    v6_probs_by_id: Optional[dict[str, np.ndarray]] = None,
    v6_cache_path: str = "v6_oof_probs_cache.json",
    run_v6_if_missing: bool = True,
    seed: int = 42,
    use_v6_pro: bool = False,
) -> tuple[dict, list, dict[str, np.ndarray]]:
    """
    Optimize blend weight (if needed), ensemble GPU + v6 OOF predictions.

    Returns (report, updated_fold_results, v6_probs_by_id).
    """
    print("\nGPU + v6 ensemble starting…", flush=True)
    before = compute_pooled_metrics(fold_results)
    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)

    if v6_probs_by_id is None:
        v6_probs_by_id = load_v6_probs_cache(v6_cache_path)

    v6_meta: dict = {"source": "cache" if v6_probs_by_id else "lite_cv"}

    if v6_probs_by_id is not None:
        print(
            f"  v6 cache hit: {v6_cache_path} ({len(v6_probs_by_id)} proteins) — skipping OOF",
            flush=True,
        )
    elif run_v6_if_missing:
        if use_v6_pro:
            from colab.v6_pro_ensemble import get_v6_pro_oof_probs
            pro_cache = v6_cache_path.replace(".json", "_pro.json")
            print(
                f"  v6 cache miss — running v6-pro OOF (first time: often 20–60 min)…\n"
                f"  cache will be saved to {pro_cache}",
                flush=True,
            )
            v6_probs_by_id = get_v6_pro_oof_probs(
                proteins, n_folds=n_folds, seed=seed, cache_path=pro_cache,
                verbose=True,
            )
            v6_meta.update({"source": "v6_pro", "cache": pro_cache})
        else:
            print(
                f"  v6 cache miss — running v6-lite OOF (first time: often 15–45 min)…\n"
                f"  cache will be saved to {v6_cache_path}",
                flush=True,
            )
            oof_probs, oof_labels, fold_meta = run_v6_lite_oof(
                proteins, n_folds=n_folds, seed=seed, verbose=True,
            )
            v6_probs_by_id = aligned_probs_from_oof(proteins, oof_probs)
            save_v6_probs_cache(v6_probs_by_id, v6_cache_path)
            v6_meta.update({
                "source": "lite_cv",
                "lite_cv_auc": float(roc_auc_score(oof_labels, oof_probs)),
                "fold_meta": fold_meta,
            })
            print(f"  saved v6 cache → {v6_cache_path}", flush=True)
    elif v6_probs_by_id is None:
        raise ValueError("v6 probabilities unavailable and run_v6_if_missing=False")

    # Build aligned v6 array for weight search
    gpu_concat, v6_concat, labels_concat = [], [], []
    for item in aligned:
        pid = item["id"]
        if pid not in v6_probs_by_id:
            continue
        v6p = np.asarray(v6_probs_by_id[pid], dtype=np.float32)
        if len(v6p) != len(item["probs"]):
            continue
        gpu_concat.append(item["probs"])
        v6_concat.append(v6p)
        labels_concat.append(item["labels"])

    if gpu_concat:
        gpu_all = np.concatenate(gpu_concat)
        v6_all = np.concatenate(v6_concat)
        labels_all = np.concatenate(labels_concat)
    else:
        gpu_all = before["all_probs"]
        v6_all = gpu_all.copy()
        labels_all = before["all_labels"]

    if weight is None:
        print("  Tuning GPU/v6 blend weight on OOF residues…", flush=True)
        search = find_optimal_blend_weight(labels_all, gpu_all, v6_all)
        weight = search["best_weight"]
        weight_search = search
        print(f"  Best v6 blend weight: {weight:.2f}", flush=True)
    else:
        weight_search = {"best_weight": float(weight), "provided": True}

    aligned_blended = blend_aligned_predictions(aligned, v6_probs_by_id, weight)
    fold_results_ensembled = write_fused_probs_to_fold_results(
        proteins, fold_results, aligned_blended, n_folds=n_folds,
    )
    for fr in fold_results_ensembled:
        fr["ensembled"] = True
        fr["ensemble_weight"] = float(weight)

    after = compute_pooled_metrics(fold_results_ensembled)

    report = {
        "ensemble_weight": float(weight),
        "weight_search": weight_search,
        "v6_meta": v6_meta,
        "before": {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}},
        "after": {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}},
        "delta_auc_pooled": after["auc"] - before["auc"],
        "delta_ap_pooled": after["ap"] - before["ap"],
    }
    return report, fold_results_ensembled, v6_probs_by_id


def print_ensemble_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" GPU + v6 CPU ENSEMBLE")
    print(f"{'═' * 64}")
    print(f"  Blend weight (v6) : {report['ensemble_weight']:.2f}")
    v6 = report.get("v6_meta", {})
    print(f"  v6 source         : {v6.get('source', 'unknown')}")
    if v6.get("lite_cv_auc") is not None:
        print(f"  v6 lite CV AUC    : {v6['lite_cv_auc']:.4f}")

    b, a = report["before"], report["after"]
    print(f"\n── Pooled CV (all residues) ──")
    print(f"  Before : AUC={b['pooled']['auc']:.4f}  AP={b['pooled']['ap']:.4f}")
    print(f"  After  : AUC={a['pooled']['auc']:.4f}  AP={a['pooled']['ap']:.4f}")
    print(f"  Δ AUC  : {report['delta_auc_pooled']:+.4f}  "
          f"Δ AP: {report['delta_ap_pooled']:+.4f}")
    print(f"{'═' * 64}")


def save_ensemble_report(report: dict, path: str = "gpu_v6_ensemble_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
