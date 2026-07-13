"""
Monte Carlo dropout test-time augmentation for DisorderNet GPU inference.

Averages multiple stochastic forward passes (dropout on head/LoRA active) to
stabilize predictions. Typically +0.003–0.015 AUC at ~N× inference cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def mc_dropout_forward_logits(
    model: nn.Module,
    tokens: torch.Tensor,
    aa_idx: torch.Tensor | None,
    mask: torch.Tensor,
    rich_feats: torch.Tensor | None,
    n_passes: int,
    forward_fn,
    **forward_kw,
) -> torch.Tensor:
    """
    Average logits over MC dropout passes.

    forward_fn: callable(model, tokens, aa_idx, mask, rich_feats=..., **kw) -> logits
    """
    if n_passes <= 1:
        return forward_fn(model, tokens, aa_idx, mask, rich_feats=rich_feats, **forward_kw)

    was_training = model.training
    model.train()
    accum = None
    for _ in range(n_passes):
        logits = forward_fn(model, tokens, aa_idx, mask, rich_feats=rich_feats, **forward_kw)
        accum = logits if accum is None else accum + logits
    model.train(was_training)
    return accum / n_passes


def mc_dropout_predict_probs(
    model: nn.Module,
    tokens: torch.Tensor,
    aa_idx: torch.Tensor | None,
    mask: torch.Tensor,
    rich_feats: torch.Tensor | None,
    n_passes: int,
    forward_fn,
    **forward_kw,
) -> torch.Tensor:
    """Return sigmoid-averaged probabilities (B, L)."""
    logits = mc_dropout_forward_logits(
        model, tokens, aa_idx, mask, rich_feats, n_passes, forward_fn, **forward_kw,
    )
    return torch.sigmoid(logits)
