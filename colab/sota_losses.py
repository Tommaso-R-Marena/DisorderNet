"""
SOTA-oriented training losses for DisorderNet GPU.

Combines focal/BCE with soft Dice (region-friendly), Tversky, R-Drop, and
optional v6 teacher distillation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Differentiable soft Dice loss on flattened valid residues."""
    probs = torch.sigmoid(logits)
    labels = labels.float()
    intersection = (probs * labels).sum()
    denom = probs.sum() + labels.sum()
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice


def _masked_probs_and_labels(logits, labels, mask):
    """Sigmoid probabilities and labels with padded positions zeroed out.

    Zeroing lets every per-sequence statistic be a row sum, so the whole batch
    is a handful of kernels instead of one Python iteration per sequence.
    """
    m = mask.to(logits.dtype)
    return torch.sigmoid(logits) * m, labels.to(logits.dtype) * m


def batch_mean_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Per-sequence soft Dice averaged over batch (ignores padding)."""
    probs, y = _masked_probs_and_labels(logits, labels, mask)
    intersection = (probs * y).sum(dim=1)
    denom = probs.sum(dim=1) + y.sum(dim=1)
    per_seq = 1.0 - (2.0 * intersection + smooth) / (denom + smooth)

    valid = mask.sum(dim=1) >= 2
    if not bool(valid.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return per_seq[valid].mean()


def apply_label_smoothing(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """Binary label smoothing: 0→ε/2, 1→1-ε/2."""
    if smoothing <= 0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def batch_mean_tversky_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Per-sequence Tversky loss (alpha=FN weight, beta=FP weight)."""
    probs, y = _masked_probs_and_labels(logits, labels, mask)
    # Padding is zero in both probs and y, so it drops out of every term:
    #   fn = sum (1-p)*y = sum y - p*y      fp = sum (1-y)*p = sum p - p*y
    tp = (probs * y).sum(dim=1)
    fn = y.sum(dim=1) - tp
    fp = probs.sum(dim=1) - tp
    per_seq = 1.0 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)

    valid = mask.sum(dim=1) >= 2
    if not bool(valid.any()):
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return per_seq[valid].mean()


def rdrop_symmetric_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL between two dropout views (R-Drop consistency term)."""
    log_pa = F.log_softmax(torch.stack([logits_a, -logits_a], dim=-1), dim=-1)[..., 0]
    log_pb = F.log_softmax(torch.stack([logits_b, -logits_b], dim=-1), dim=-1)[..., 0]
    pa = log_pa.exp()
    pb = log_pb.exp()
    kl_ab = (pa * (log_pa - log_pb))[mask].mean()
    kl_ba = (pb * (log_pb - log_pa))[mask].mean()
    return 0.5 * (kl_ab + kl_ba)


def v6_distillation_loss(
    logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """BCE distillation from v6 OOF teacher probabilities (soft targets)."""
    student_logits = logits[mask] / temperature
    teacher = teacher_probs[mask].clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy_with_logits(student_logits, teacher)


def composite_disorder_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
    sample_weight: torch.Tensor | None,
    cfg,
) -> torch.Tensor:
    """
    Focal/BCE + optional Dice + boundary/hallucination sample weights.

    When mask is 2D (B, L), Dice is computed per sequence; focal/BCE on valid residues.
    """
    flat_logits = logits[mask]
    flat_labels = labels[mask]
    if cfg.label_smoothing > 0:
        flat_labels = apply_label_smoothing(flat_labels, cfg.label_smoothing)

    # One per-residue pass, then reduce. (This used to compute the focal/BCE
    # term twice whenever sample weights were supplied — i.e. on every training
    # step — and throw the unweighted copy away.)
    per_residue = nn.functional.binary_cross_entropy_with_logits(
        flat_logits, flat_labels, pos_weight=pos_weight, reduction="none",
    )
    if cfg.use_focal_loss:
        probs = torch.sigmoid(flat_logits)
        pt = torch.where(flat_labels > 0.5, probs, 1.0 - probs)
        per_residue = per_residue * ((1.0 - pt) ** cfg.focal_gamma)

    if sample_weight is None:
        base_loss = per_residue.mean()
    else:
        sw = sample_weight[mask]
        base_loss = (per_residue * sw).sum() / sw.sum().clamp(min=1.0)

    total = base_loss
    if getattr(cfg, "use_dice_loss", False) and logits.dim() == 2:
        dice = batch_mean_dice_loss(logits, labels, mask)
        w = getattr(cfg, "dice_loss_weight", 0.25)
        total = total + w * dice

    if getattr(cfg, "use_tversky_loss", False) and logits.dim() == 2:
        tversky = batch_mean_tversky_loss(
            logits, labels, mask,
            alpha=getattr(cfg, "tversky_alpha", 0.3),
            beta=getattr(cfg, "tversky_beta", 0.7),
        )
        total = total + getattr(cfg, "tversky_weight", 0.15) * tversky

    return total
