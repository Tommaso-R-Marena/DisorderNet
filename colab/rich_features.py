"""
Rich per-residue features (features_fast, 162-dim) for GPU training.

Precomputes and caches numpy features in DisProtDataset; encodes with a small MLP.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from features_fast import compute_features_fast

RICH_FEATURE_DIM = 162


def compute_rich_features(sequence: str) -> np.ndarray:
    """Return (L, 162) float32 features for a protein sequence."""
    feats = compute_features_fast(sequence)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class RichFeatureEncoder(nn.Module):
    """Project full v6-style physics features into embedding space."""

    def __init__(self, in_dim: int = RICH_FEATURE_DIM, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(128, out_dim),
            nn.GELU(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, L, in_dim)"""
        return self.net(feats)
