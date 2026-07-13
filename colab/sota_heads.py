"""
SOTA prediction head: multi-scale CNN features + lightweight Transformer encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DisorderSOTAHead(nn.Module):
    """
    Multi-scale dilated CNN → Transformer encoder → per-residue logits.

    Transformer uses padding mask (True = valid residue) for variable-length batches.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        d_model: int = 256,
        n_transformer_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.12,
    ):
        super().__init__()
        mid = 384
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_dim, mid, kernel_size=7, padding=3, dilation=1),
                nn.BatchNorm1d(mid),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(in_dim, mid, kernel_size=5, padding=6, dilation=3),
                nn.BatchNorm1d(mid),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv1d(in_dim, mid, kernel_size=3, padding=8, dilation=8),
                nn.BatchNorm1d(mid),
                nn.GELU(),
            ),
        ])
        self.to_transformer = nn.Sequential(
            nn.Conv1d(mid * 3, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_transformer_layers)
        self.readout = nn.Conv1d(d_model, 1, kernel_size=1)
        self.skip = nn.Conv1d(in_dim, 1, kernel_size=1)
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
        x: (B, L, C) embeddings
        pad_mask: (B, L) bool, True = valid residue
        """
        xc = x.permute(0, 2, 1)
        parts = [b(xc) for b in self.branches]
        feat = self.to_transformer(torch.cat(parts, dim=1))
        feat = feat.permute(0, 2, 1)
        key_padding_mask = None
        if pad_mask is not None:
            key_padding_mask = ~pad_mask
        feat = self.transformer(feat, src_key_padding_mask=key_padding_mask)
        logits = self.readout(feat.permute(0, 2, 1)).squeeze(1)
        skip = self.skip(xc).squeeze(1)
        return logits + skip
