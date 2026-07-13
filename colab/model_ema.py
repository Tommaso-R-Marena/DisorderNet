"""
Exponential moving average (EMA) of trainable weights for SOTA training stability.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class ModelEMA:
    """Track EMA of trainable parameters; apply for eval / checkpoint selection."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.detach().clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, shadow: dict[str, torch.Tensor]) -> None:
        self.shadow = copy.deepcopy(shadow)
