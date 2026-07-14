"""
Structure-aware features for DisorderNet — train-time pLDDT channel.

Novel vs sequence-only SOTA (ESMDisPred): learns joint sequence+structure
disorder signal instead of post-hoc pLDDT fusion only.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from colab.af_plddt import plddt_to_disorder_score


def build_plddt_feature_tensor(
    plddt: Optional[np.ndarray],
    length: int,
) -> torch.Tensor:
    """
    Per-residue structure features: [inverse_plddt_score, valid_mask].

    Missing pLDDT → zeros (model learns to rely on sequence).
    """
    out = torch.zeros(length, 2, dtype=torch.float32)
    if plddt is None:
        return out
    arr = np.asarray(plddt, dtype=np.float32)
    n = min(length, len(arr))
    valid = ~np.isnan(arr[:n])
    if valid.any():
        scores = plddt_to_disorder_score(arr[:n])
        out[:n, 0] = torch.from_numpy(np.where(valid, scores, 0.0))
        out[:n, 1] = torch.from_numpy(valid.astype(np.float32))
    return out


class PlddtFeatureEncoder(nn.Module):
    """Lightweight 1D conv over pLDDT + validity mask."""

    def __init__(self, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, out_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

    def forward(self, plddt_feats: torch.Tensor) -> torch.Tensor:
        """plddt_feats: (B, L, 2) → (B, L, out_dim)."""
        x = plddt_feats.permute(0, 2, 1)
        return self.net(x).permute(0, 2, 1)
