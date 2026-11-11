"""
Monte Carlo dropout test-time augmentation for DisorderNet GPU inference.

Averages multiple stochastic forward passes (dropout on head/LoRA active) to
stabilize predictions. Typically +0.003–0.015 AUC at ~N× inference cost.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


# Public dropout classes, resolved defensively so a torch version that drops or
# renames one of them degrades to "skip it" rather than raising at import time.
_DROPOUT_TYPES = tuple(
    cls for cls in (
        getattr(nn, name, None) for name in (
            "Dropout", "Dropout1d", "Dropout2d", "Dropout3d",
            "AlphaDropout", "FeatureAlphaDropout",
        )
    )
    if isinstance(cls, type)
)


@contextlib.contextmanager
def dropout_only_train_mode(model: nn.Module):
    """Enable dropout for MC sampling while every other layer stays in eval mode.

    A blanket ``model.train()`` would also switch the head's BatchNorm layers to
    batch statistics *and* let them update their running mean/var — under
    ``torch.no_grad()`` that still mutates the buffers, so a TTA inference pass
    would silently corrupt the trained checkpoint. Only dropout modules need to
    be stochastic here.
    """
    was_training = model.training
    model.eval()
    toggled = [
        m for m in model.modules()
        if isinstance(m, _DROPOUT_TYPES) and getattr(m, "p", 0.0) > 0
    ]
    for m in toggled:
        m.train()
    try:
        yield
    finally:
        for m in toggled:
            m.eval()
        model.train(was_training)


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

    accum = None
    with dropout_only_train_mode(model):
        for _ in range(n_passes):
            logits = forward_fn(model, tokens, aa_idx, mask, rich_feats=rich_feats, **forward_kw)
            accum = logits if accum is None else accum + logits
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
    """Return sigmoid-averaged probabilities (B, L) as float32.

    Callers convert these to numpy, and ``Tensor.numpy()`` has no bfloat16
    support — under bf16 autocast this raised
    ``TypeError: Got unsupported ScalarType BFloat16`` only once the fold soup
    actually ran with TTA enabled. Casting here keeps the guarantee at the
    source rather than at each call site.
    """
    logits = mc_dropout_forward_logits(
        model, tokens, aa_idx, mask, rich_feats, n_passes, forward_fn, **forward_kw,
    )
    return torch.sigmoid(logits).float()
