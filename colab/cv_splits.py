"""
Shared cross-validation splits for DisorderNet (Tier 1 rigor).

All modules that align OOF predictions must use the same GroupKFold splits on
proteins sorted deterministically by DisProt ID. This prevents silent fold
misalignment when protein list order changes between notebook cells.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from sklearn.model_selection import GroupKFold

# TrainConfig fields that affect splits, training, or evaluation comparability.
_FINGERPRINT_KEYS = (
    "seed",
    "min_seq_len",
    "max_seq_len",
    "min_disorder",
    "min_order",
    "batch_size",
    "accum_steps",
    "num_epochs",
    "lr_lora",
    "lr_head",
    "patience",
    "lora_layers",
    "lora_rank",
    "lora_alpha",
    "lora_on_k",
    "use_focal_loss",
    "focal_gamma",
    "boundary_weight",
    "boundary_radius",
    "use_physico_features",
    "physico_dim",
    "esm_fusion_layers",
    "n_folds",
    "use_segment_early_stop",
    "auc_score_weight",
    "ap_score_weight",
    "segment_score_weight",
    "use_hallucination_weighting",
    "hallucination_weight",
    "high_plddt_threshold",
    "early_stop_mode",
    "head_type",
    "use_dice_loss",
    "dice_loss_weight",
    "use_ema",
    "compact_checkpoints",
)


def sort_proteins_deterministic(proteins: list) -> list:
    """Stable protein order for reproducible GroupKFold assignments."""
    return sorted(proteins, key=lambda p: p["id"])


def get_cv_splits(proteins: list, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """GroupKFold splits using one group per protein index."""
    groups = np.arange(len(proteins))
    gkf = GroupKFold(n_splits=n_folds)
    return list(gkf.split(groups, groups=groups))


def get_fold_val_protein_ids(proteins: list, n_folds: int) -> list[list[str]]:
    """Validation protein IDs per fold (for resume / audit fingerprints)."""
    return [
        [proteins[i]["id"] for i in val_idx]
        for _, val_idx in get_cv_splits(proteins, n_folds)
    ]


def config_fingerprint(cfg: Any) -> str:
    """Hash of hyperparameters that must match when resuming CV."""
    raw = asdict(cfg)
    subset = {k: raw[k] for k in _FINGERPRINT_KEYS if k in raw}
    # JSON-serialize device/dtype if present in overrides
    for k, v in list(subset.items()):
        if hasattr(v, "type"):
            subset[k] = str(v)
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def proteins_fingerprint(proteins: list) -> str:
    """Short hash of ordered protein IDs + lengths."""
    parts = [f"{p['id']}:{p['length']}" for p in proteins]
    blob = "|".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
