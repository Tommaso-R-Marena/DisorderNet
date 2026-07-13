"""
Full-strength v6-style OOF ensemble for GPU CV alignment.

Uses features_fast + dual gradient boosting (LightGBM + XGBoost when available,
HistGradientBoosting fallback). Stronger than v6-lite HGB-only proxy.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from colab.cv_splits import get_cv_splits
from colab.ensemble_v6 import (
    _stack_protein_features,
    aligned_probs_from_oof,
    build_v6_features,
    load_v6_probs_cache,
    save_v6_probs_cache,
)


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except ImportError:
        return False


def _train_v6_pro_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    verbose: bool = False,
) -> object:
    """Train dual GBDT ensemble or strong HGB fallback."""
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    weight_ratio = min(neg / max(pos, 1), 10.0)
    sample_weight = np.where(y_train > 0.5, weight_ratio, 1.0).astype(np.float32)

    models: list = []

    if _has_lightgbm():
        if verbose:
            print("      training LightGBM…", flush=True)
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        lgb_model.fit(X_train, y_train.astype(np.int8), sample_weight=sample_weight)
        models.append(("lgb", lgb_model))

    if _has_xgboost():
        if verbose:
            print("      training XGBoost…", flush=True)
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
            eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train.astype(np.int8), sample_weight=sample_weight)
        models.append(("xgb", xgb_model))

    if not models:
        if verbose:
            print("      training HGB (install lightgbm+xgboost for v6-pro)…", flush=True)
        hgb = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=180,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        hgb.fit(X_train, y_train.astype(np.int8), sample_weight=sample_weight)
        models.append(("hgb", hgb))

    return models


def _predict_v6_pro(models: list, X: np.ndarray) -> np.ndarray:
    preds = []
    for _, model in models:
        if hasattr(model, "predict_proba"):
            preds.append(model.predict_proba(X)[:, 1])
        else:
            preds.append(model.predict(X).astype(np.float32))
    return np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)


def run_v6_pro_oof(
    proteins: list,
    n_folds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Run full-strength v6-style OOF on GPU protein list.

    Returns (oof_probs, oof_labels, per_fold_metadata).
    """
    splits = get_cv_splits(proteins, n_folds)
    oof_probs_by_id: dict[str, np.ndarray] = {}
    oof_labels_by_id: dict[str, np.ndarray] = {}
    fold_meta: list[dict] = []

    backends = []
    if _has_lightgbm():
        backends.append("lgb")
    if _has_xgboost():
        backends.append("xgb")
    if not backends:
        backends.append("hgb")

    n_res = sum(p["length"] for p in proteins)
    if verbose:
        print(
            f"\n{'─' * 60}\n"
            f" v6-pro OOF  │  {len(proteins)} proteins  │  {n_res:,} residues  │  "
            f"backends={'+'.join(backends)}\n"
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
            print("    → stacking train features…", flush=True)
        X_train, y_train, _ = _stack_protein_features(
            train_proteins, verbose=verbose, desc="train features",
        )
        if verbose:
            print(f"    → stacking val features…", flush=True)
        X_val, y_val, val_lengths = _stack_protein_features(
            val_proteins, verbose=verbose, desc="val features",
        )

        if verbose:
            print(f"    → training GBDT ({X_train.shape[0]:,} train residues)…", flush=True)
        models = _train_v6_pro_fold(X_train, y_train, seed=seed + fold_idx, verbose=verbose)

        if verbose:
            print(f"    → predicting val ({X_val.shape[0]:,} residues)…", flush=True)
        val_probs = _predict_v6_pro(models, X_val)

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
            "backends": backends,
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
            f"\n  v6-pro OOF complete  pooled_AUC={pooled_auc:.4f}  "
            f"total={(time.time() - cv_t0) / 60:.1f} min\n",
            flush=True,
        )

    return oof_probs, oof_labels, fold_meta


def get_v6_pro_oof_probs(
    proteins: list,
    n_folds: int = 5,
    seed: int = 42,
    cache_path: str = "v6_pro_oof_probs_cache.json",
    force_recompute: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Load or compute v6-pro OOF probabilities keyed by protein id."""
    if not force_recompute:
        cached = load_v6_probs_cache(cache_path)
        if cached and len(cached) >= len(proteins) * 0.95:
            if verbose:
                print(f"  v6-pro cache hit: {cache_path} ({len(cached)} proteins)", flush=True)
            return cached

    oof_probs, _, _ = run_v6_pro_oof(
        proteins, n_folds=n_folds, seed=seed, verbose=verbose,
    )
    by_id = aligned_probs_from_oof(proteins, oof_probs)
    save_v6_probs_cache(by_id, cache_path)
    if verbose:
        print(f"  saved v6-pro cache → {cache_path}", flush=True)
    return by_id
