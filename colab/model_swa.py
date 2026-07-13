"""
Stochastic Weight Averaging (SWA) for DisorderNet fold training.

Averages trainable weights over the last epochs for a flatter minimum.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class ModelSWA:
    """Running average of trainable parameters (Polyak-Ruppert style)."""

    def __init__(self, model: nn.Module):
        self.n_averaged = 0
        self.swa_state: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.swa_state[name] = param.data.detach().clone()

    def update(self, model: nn.Module) -> None:
        self.n_averaged += 1
        n = self.n_averaged
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.swa_state:
                continue
            self.swa_state[name].add_((param.data - self.swa_state[name]) / n)

    def apply_swa(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap model weights to SWA average; return backup for restore."""
        backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if name in self.swa_state:
                backup[name] = param.data.detach().clone()
                param.data.copy_(self.swa_state[name])
        return backup

    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])

    @property
    def ready(self) -> bool:
        return self.n_averaged > 0
