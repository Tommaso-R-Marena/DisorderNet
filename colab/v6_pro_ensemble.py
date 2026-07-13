"""
Full-strength v6-style OOF ensemble for GPU CV alignment.

Uses features_fast + dual gradient boosting (LightGBM + XGBoost when available,
HistGradientBoosting fallback). Stronger than v6-lite HGB-only proxy.
"""

from __future__ import annotations

import json
import os
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
) -> object:
    """Train dual GBDT ensemble or strong HGB fallback."""
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    weight_ratio = min(neg / max(pos, 1), 10.0)
    sample_weight = np.where(y_train > 0.5, weight_ratio, 1.0).astype(np.float32)

    models: list = []

    if _has_lightgbm():
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

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_proteins = [proteins[i] for i in train_idx]
        val_proteins = [proteins[i] for i in val_idx]

        X_train, y_train, _ = _stack_protein_features(train_proteins)
        X_val, y_val, val_lengths = _stack_protein_features(val_proteins)

        models = _train_v6_pro_fold(X_train, y_train, seed=seed + fold_idx)
        val_probs = _predict_v6_pro(models, X_val)

        val_offset = 0
        for p in val_proteins:
            length = p["length"]
            oof_probs_by_id[p["id"]] = val_probs[val_offset:val_offset + length].copy()
            oof_labels_by_id[p["id"]] = y_val[val_offset:val_offset + length].copy()
            val_offset += length

        fold_meta.append({
            "fold": fold_idx + 1,
            "n_train": len(train_proteins),
            "n_val": len(val_proteins),
            "backends": backends,
            "val_auc": float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else None,
            "val_lengths": val_lengths,
        })

    oof_probs = np.concatenate([oof_probs_by_id[p["id"]] for p in proteins])
    oof_labels = np.concatenate([oof_labels_by_id[p["id"]] for p in proteins])
    return oof_probs, oof_labels, fold_meta


def get_v6_pro_oof_probs(
    proteins: list,
    n_folds: int = 5,
    seed: int = 42,
    cache_path: str = "v6_pro_oof_probs_cache.json",
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Load or compute v6-pro OOF probabilities keyed by protein id."""
    if not force_recompute:
        cached = load_v6_probs_cache(cache_path)
        if cached and len(cached) >= len(proteins) * 0.95:
            return cached

    oof_probs, _, meta = run_v6_pro_oof(proteins, n_folds=n_folds, seed=seed)
    by_id = aligned_probs_from_oof(proteins, oof_probs)
    save_v6_probs_cache(by_id, cache_path)
    return by_id
