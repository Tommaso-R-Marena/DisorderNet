"""
Compact fold checkpoints — trainable weights only (~50–150 MB vs ~2.5 GB full state).

Saves LoRA adapters, fusion module, physico encoder, and prediction head.
Frozen ESM-2 650M backbone is reloaded from fair-esm on inference.
"""

from __future__ import annotations

import os
import tempfile
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
    """Every tensor needed to reproduce this model's predictions.

    Selection is driven by ``requires_grad``, not by a name allowlist. The
    allowlist (lora_A/lora_B/head./…) silently omitted anything trained outside
    those namespaces — in particular ``unfreeze_last_layers``, which the ultra
    profile sets to 2 and which fine-tunes
    ``esm.layers.{N}.self_attn_layer_norm`` and ``final_layer_norm``. Those
    names match no marker, so the tuned LayerNorm weights were never saved and
    silently reverted to pretrained values on reload. LayerNorm rescales the
    whole residual stream, so the restored model was not the trained model:
    measured Spearman 0.387 against the predictions it was saved from, and
    pooled AUC 0.5835 against 0.7651.

    Buffers (BatchNorm running statistics) are included for any module that owns
    a trainable parameter — in eval mode those statistics determine the output
    as surely as the weights do.
    """
    trainable_params = {n for n, p in model.named_parameters() if p.requires_grad}

    # Module prefixes that own at least one trained parameter; their buffers
    # (e.g. BatchNorm running_mean/var) are part of the trained state.
    trained_prefixes = {n.rsplit(".", 1)[0] for n in trainable_params if "." in n}

    out: dict[str, torch.Tensor] = {}
    buffer_names = {n for n, _ in model.named_buffers()}
    for k, v in model.state_dict().items():
        keep = k in trainable_params or is_trainable_key(k)
        if not keep and k in buffer_names:
            prefix = k.rsplit(".", 1)[0]
            keep = prefix in trained_prefixes
        if keep:
            out[k] = v.detach().cpu().clone()
    return out


def atomic_torch_save(payload: Any, path: str) -> str:
    """Write a checkpoint so it is either complete or absent, never partial.

    A job killed mid-write leaves a truncated file at the destination path. That
    happened here: a lite_frozen replicate was killed during its fold-4 write and
    left a 0-byte fold4_best.pt, which any resume or soup step would then find by
    existence check and try to load. Writing to a sibling temp file and renaming
    makes the publish atomic on POSIX — readers see the old file or the new one.

    The fsync matters on a shared filesystem: without it the rename can land
    before the data, so a node failure yields an intact-looking name over
    unwritten blocks.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Includes KeyboardInterrupt/SystemExit: a partial temp file is litter,
        # and leaving it behind in the checkpoint directory is its own hazard.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def save_compact_checkpoint(
    path: str,
    model: nn.Module,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Save trainable weights + metadata JSON-serializable fields."""
    payload = {
        "version": 1,
        "format": "compact_trainable",
        "trainable": extract_trainable_state_dict(model),
        "metadata": metadata or {},
    }
    return atomic_torch_save(payload, path)


def load_compact_checkpoint(
    path: str,
    model: nn.Module,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load trainable weights into a built DisorderNetGPU (ESM backbone must exist)."""
    # A truncated checkpoint from a killed job otherwise surfaces as an opaque
    # unpickling error several frames deep, or — worse — as an empty state dict
    # that loads without complaint.
    if os.path.getsize(path) == 0:
        raise RuntimeError(
            f"Checkpoint {path} is empty — the writing job was killed mid-save. "
            "Delete it and re-run that fold; do not treat it as a completed fold."
        )
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

    # A *missing* trainable tensor is as damaging as an unexpected one and far
    # harder to notice: the model still runs and still emits plausible
    # probabilities, just without the weights that were actually trained.
    #
    # The test is requires_grad, not a name allowlist. Checking names was how the
    # unfreeze_last_layers LayerNorm weights went unnoticed — they were absent
    # from the checkpoint AND from the allowlist, so nothing flagged them, and
    # the reloaded model scored 0.5835 against the 0.7651 it was saved from.
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    absent = sorted(expected - set(trainable))
    if absent:
        raise RuntimeError(
            f"{os.path.basename(path)} is missing {len(absent)} tensor(s) that "
            f"this model trains, so it cannot reproduce its own predictions. "
            f"First few: {absent[:5]}. "
            "A checkpoint written before the compact-format fix will not contain "
            "fine-tuned ESM tail weights (unfreeze_last_layers) and must be "
            "regenerated by re-running CV."
        )

    trained_missing = [k for k in missing if is_trainable_key(k)]
    if trained_missing:
        raise RuntimeError(
            f"{len(trained_missing)} tensors in {os.path.basename(path)} had no "
            f"destination in the model — it would run partially restored. "
            f"First few: {trained_missing[:5]}. "
            "Usual cause: LoRA adapters not injected into the backbone before load."
        )
    return {"missing_keys": missing, "metadata": meta, "n_tensors": len(trainable)}


def checkpoint_size_mb(path: str) -> float:
    if not os.path.isfile(path):
        return 0.0
    return os.path.getsize(path) / (1024 ** 2)
