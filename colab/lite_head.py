"""DisorderNet-Lite: a frozen-PLM, low-capacity head for the small-data regime.

Why a different architecture
----------------------------
The measurements from the 650M homology-split run point one way:

  ESM-2 650M + LoRA (69.9M trainable)   pooled AUC 0.7454
  v6 physics GBDT                        pooled AUC 0.7804
  AlphaFold pLDDT alone                  pooled AUC 0.7906

A 650M protein language model losing to a gradient-boosted tree on hand-crafted
features is not what ESM-2 can do — published ESM2-650M+LoRA disorder models sit
near 0.88. The bottleneck is not the backbone, it is the fit between capacity and
data:

  * 69.9M trainable parameters (LoRA r=128 over 20 layers, plus FFN, out_proj,
    and two unfrozen ESM tail layers)
  * ~1M evidenced residues from 2,340 proteins
  * ten simultaneously-active regularisers (focal gamma=3, dice, tversky,
    R-drop, v6 distillation, boundary x4, hallucination weighting, label
    smoothing, EMA, SWA, MC-dropout TTA), every hyperparameter tuned while the
    evaluation was leaking

The screen showed the failure directly: train loss 0.069 against validation AUC
0.66. A GBDT wins in that regime because it is sample-efficient, not because
physics features beat language-model features.

Design, one decision per measurement
------------------------------------
1. **Freeze the backbone.** Learn a scalar mixture over its layers instead
   (33 parameters for ESM-2 650M). Different layers encode different things and
   the useful depth for disorder is an empirical question, not a fixed choice.
   Precedent: SETH (frozen ProtT5 + CNN) reaches 0.830 on CAID — above this
   project's 0.8155 — with no fine-tuning at all.
2. **Small dilated CNN head.** Disorder is local-to-medium range; dilations
   1/2/4/8 cover ~30 residues of context with ~1-3M parameters instead of 70M.
3. **Keep an explicit physics channel.** v6 alone (0.7804) beat the LoRA model,
   and ensembling gained +0.056 — that signal is real and cheap.
4. **Plain weighted BCE.** Every extra loss term is a hyperparameter fitted to a
   broken evaluation. Start from the simplest objective that can work and add a
   term back only when it beats the measured ~0.023 AUC noise floor.

What this is not
----------------
This is not expected to reach 0.895. Its purpose is to test the specific
hypothesis that capacity/data mismatch — not the backbone — is what costs this
project ~0.08 AUC against comparable published models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

# Dilations 1/2/4/8 over 4 residual blocks give a 61-residue receptive field —
# comfortably wider than a typical IDR, without a transformer's parameter cost.
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)


def receptive_field(dilations: Sequence[int], n_blocks: int) -> int:
    """Residues visible to one output position.

    Each block applies two dilated kernel-3 convolutions, so it widens the field
    by ``2 * dilation`` on each side.
    """
    span = sum(2 * dilations[i % len(dilations)] for i in range(n_blocks))
    return 1 + 2 * span


class ScalarMix(nn.Module):
    """Learned softmax-weighted mixture of frozen layer representations.

    One weight per layer plus a global scale — 33 parameters for ESM-2 650M.
    Which depth carries disorder signal is an empirical question; fixing it by
    hand (or concatenating every layer, which multiplies the head's input width)
    both throw information away. The learned weights are also interpretable: they
    say where in the backbone the signal actually lives.
    """

    def __init__(self, n_layers: int, do_layer_norm: bool = False):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.ones(1))
        self.do_layer_norm = do_layer_norm

    def forward(self, layers: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(layers) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} layers, got {len(layers)}")
        w = torch.softmax(self.weights, dim=0)
        out = None
        for i, h in enumerate(layers):
            if self.do_layer_norm:
                h = nn.functional.layer_norm(h, h.shape[-1:])
            term = w[i] * h
            out = term if out is None else out + term
        return self.gamma * out

    def layer_weights(self) -> torch.Tensor:
        """Normalised mixture weights, for reporting where the signal sits."""
        return torch.softmax(self.weights.detach(), dim=0)


class DilatedResidualBlock(nn.Module):
    """Residual 1-D conv block with dilation, GroupNorm and GELU.

    GroupNorm rather than BatchNorm: batches here are a handful of proteins of
    wildly differing length, so batch statistics are unstable and — as this
    project found the hard way — BatchNorm running statistics are easy to lose
    across a checkpoint round-trip.
    """

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        h = self.drop(self.act(self.norm1(self.conv1(x))))
        h = self.drop(self.act(self.norm2(self.conv2(h))))
        return x + h


class LiteDisorderHead(nn.Module):
    """The Lite trunk, in the shape the existing pipeline expects.

    ``DisorderNetGPU`` already performs layer fusion (``ESMLayerFusion`` is a
    softmax scalar mix) and already concatenates physics/pLDDT channels, so a
    head plugged in there receives fused ``(B, L, C)`` features and returns
    per-residue logits. This is that head; :class:`DisorderNetLite` is the
    standalone equivalent that owns its own mixing.

    Against ``DisorderCNNHead`` the differences are deliberate: GroupNorm instead
    of BatchNorm (batches are a few proteins of very different lengths), residual
    blocks instead of parallel branches, and a 1x1 bottleneck before the trunk so
    width scales with ``hidden`` rather than with the backbone's embedding size.
    """

    def __init__(
        self,
        in_dim: int = 1280,
        dropout: float = 0.1,
        hidden: int = 256,
        n_blocks: int = 4,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        dil = list(dilations) if dilations is not None else list(DEFAULT_DILATIONS)
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(hidden, dil[i % len(dil)], dropout)
            for i in range(n_blocks)
        )
        self.out = nn.Conv1d(hidden, 1, 1)
        # A 1x1 skip keeps a direct linear path from features to logit, so the
        # head degrades to logistic regression rather than to noise if the
        # convolutional trunk fails to learn on this much data.
        self.skip = nn.Conv1d(in_dim, 1, 1)
        self.receptive_field = receptive_field(dil, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, C) -> (B, L)
        x = x.transpose(1, 2)
        h = self.proj(x)
        for blk in self.blocks:
            h = blk(h)
        return (self.out(h) + self.skip(x)).squeeze(1)


class DisorderNetLite(nn.Module):
    """Frozen PLM features -> scalar mix -> physics concat -> dilated CNN -> logit.

    ``esm_backbone`` is used strictly as a feature extractor and is never
    unfrozen. Trainable parameters are the mixture weights, an input projection,
    the CNN stack and a linear output — order 1-3M rather than 70M.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers_mixed: int,
        physics_dim: int = 0,
        plddt_dim: int = 0,
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.1,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.mix = ScalarMix(n_layers_mixed)
        in_dim = embed_dim + physics_dim + plddt_dim
        self.physics_dim = physics_dim
        self.plddt_dim = plddt_dim
        self.proj = nn.Conv1d(in_dim, hidden, 1)
        dil = list(dilations) if dilations is not None else list(DEFAULT_DILATIONS)
        # Cycle the dilation schedule if more blocks than dilations are asked for.
        self.blocks = nn.ModuleList(
            DilatedResidualBlock(hidden, dil[i % len(dil)], dropout) for i in range(n_blocks)
        )
        self.out = nn.Conv1d(hidden, 1, 1)
        self.receptive_field = receptive_field(dil, n_blocks)

    def forward(
        self,
        layer_hiddens: Sequence[torch.Tensor],   # each (B, L, D)
        physics: Optional[torch.Tensor] = None,  # (B, L, P)
        plddt: Optional[torch.Tensor] = None,    # (B, L, K)
    ) -> torch.Tensor:
        x = self.mix(layer_hiddens)                      # (B, L, D)
        parts = [x]
        if self.physics_dim:
            if physics is None:
                raise ValueError("physics features required (physics_dim > 0)")
            parts.append(physics)
        if self.plddt_dim:
            if plddt is None:
                raise ValueError("pLDDT features required (plddt_dim > 0)")
            parts.append(plddt)
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        x = x.transpose(1, 2)                            # (B, C, L)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x).squeeze(1)                    # (B, L) logits

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> dict:
        return {
            "architecture": "DisorderNetLite",
            "n_trainable": self.n_trainable(),
            "receptive_field_residues": self.receptive_field,
            "n_layers_mixed": self.mix.n_layers,
            "physics_dim": self.physics_dim,
            "plddt_dim": self.plddt_dim,
            "rationale": (
                "Frozen backbone with a low-capacity head. The 650M+LoRA "
                "configuration trains 69.9M parameters on ~1M residues from "
                "2,340 proteins and is beaten by a GBDT on physics features "
                "(0.7454 vs 0.7804); this trades capacity for sample efficiency."
            ),
        }


def freeze_backbone(esm_backbone: nn.Module) -> int:
    """Freeze every backbone parameter. Returns the number frozen.

    Explicit rather than implied: the checkpoint defect in this project came
    from ESM tail weights being trained but never saved, and a frozen backbone
    removes that failure mode entirely — the only trainable state is the head.
    """
    n = 0
    for p in esm_backbone.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n += 1
    esm_backbone.eval()
    return n
