"""
Compact fold checkpoints — trainable weights only (~50–150 MB vs ~2.5 GB full state).

Saves LoRA adapters, fusion module, physico encoder, and prediction head.
Frozen ESM-2 650M backbone is reloaded from fair-esm on inference.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn

TRAINABLE_KEY_MARKERS = (
    "lora_A",
    "lora_B",
    "head.",
    "function_head.",
    "layer_fusion.",
    "physico.",
    "rich_encoder.",
    "plddt_encoder.",
)


def is_trainable_key(key: str) -> bool:
    return any(marker in key for marker in TRAINABLE_KEY_MARKERS)


def extract_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Subset of state_dict containing only trainable adapter/head weights."""
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if is_trainable_key(k)
    }


def save_compact_checkpoint(
    path: str,
    model: nn.Module,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Save trainable weights + metadata JSON-serializable fields."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 1,
        "format": "compact_trainable",
        "trainable": extract_trainable_state_dict(model),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_compact_checkpoint(
    path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load trainable weights into a built DisorderNetGPU (ESM backbone must exist)."""
    payload = torch.load(path, map_location=device or "cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("format") == "compact_trainable":
        trainable = payload["trainable"]
        meta = payload.get("metadata", {})
    else:
        # Legacy full state_dict fallback
        trainable = {k: v for k, v in payload.items() if is_trainable_key(k)}
        meta = {}
    missing, unexpected = model.load_state_dict(trainable, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys in compact checkpoint: {unexpected[:5]}")
    return {"missing_keys": missing, "metadata": meta, "n_tensors": len(trainable)}


def checkpoint_size_mb(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    return os.path.getsize(path) / (1024 ** 2)
