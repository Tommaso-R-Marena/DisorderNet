"""
ESM-2 backbone registry for DisorderNet GPU.

Supports fair-esm checkpoints from 8M through 3B with Colab VRAM-aware
batch presets and profile recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class BackboneSpec:
    """Metadata for one ESM-2 checkpoint."""

    key: str
    fair_name: str
    embed_dim: int
    n_layers: int
    params_m: int
    min_vram_gb: float
    recommended_gpu: str
    # (batch_size, accum_steps) for A100 40GB — setup scales down for smaller VRAM
    batch_a100_40: tuple[int, int]
    notes: str


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "8M": BackboneSpec(
        key="8M",
        fair_name="esm2_t6_8M_UR50D",
        embed_dim=320,
        n_layers=6,
        params_m=8,
        min_vram_gb=8,
        recommended_gpu="T4 / any GPU",
        batch_a100_40=(8, 2),
        notes="Debug / smoke tests only — too weak for SOTA.",
    ),
    "35M": BackboneSpec(
        key="35M",
        fair_name="esm2_t12_35M_UR50D",
        embed_dim=480,
        n_layers=12,
        params_m=35,
        min_vram_gb=12,
        recommended_gpu="T4 16GB+",
        batch_a100_40=(8, 2),
        notes="Fast iteration; below 650M capacity.",
    ),
    "150M": BackboneSpec(
        key="150M",
        fair_name="esm2_t30_150M_UR50D",
        embed_dim=640,
        n_layers=30,
        params_m=150,
        min_vram_gb=16,
        recommended_gpu="V100 / L4",
        batch_a100_40=(6, 3),
        notes="Good mid-tier step before 650M.",
    ),
    "650M": BackboneSpec(
        key="650M",
        fair_name="esm2_t33_650M_UR50D",
        embed_dim=1280,
        n_layers=33,
        params_m=650,
        min_vram_gb=20,
        recommended_gpu="A100 40GB / L4",
        batch_a100_40=(6, 3),
        notes="Current default — verified 0.817 GPU baseline.",
    ),
    "3B": BackboneSpec(
        key="3B",
        fair_name="esm2_t36_3B_UR50D",
        embed_dim=2560,
        n_layers=36,
        params_m=3000,
        min_vram_gb=38,
        recommended_gpu="A100 40GB+ (High RAM)",
        batch_a100_40=(2, 8),
        notes="Primary backbone upgrade — target +0.03–0.06 AUC vs 650M.",
    ),
}


def get_backbone_spec(key: str) -> BackboneSpec:
    k = key.upper().replace("ESM2_", "").replace("_", "")
    aliases = {
        "650": "650M", "650M": "650M", "ESM2650M": "650M",
        "3B": "3B", "3000M": "3B", "ESM23B": "3B",
        "150M": "150M", "35M": "35M", "8M": "8M",
    }
    norm = aliases.get(k, k)
    if norm not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{key}'. Choose: {list(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[norm]


def load_esm_backbone(
    device: torch.device,
    backbone: str = "650M",
    use_gradient_checkpointing: bool = True,
) -> tuple[torch.nn.Module, Any, Any, BackboneSpec]:
    """Load ESM-2 via fair-esm. Returns (model, alphabet, batch_converter, spec)."""
    import esm

    spec = get_backbone_spec(backbone)
    loader = getattr(esm.pretrained, spec.fair_name)

    print(f"Loading ESM-2 {spec.key} ({spec.fair_name})…")
    t0 = __import__("time").time()
    model, alphabet = loader()
    model = model.to(device)
    model.eval()

    if use_gradient_checkpointing and hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)

    converter = alphabet.get_batch_converter()
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"  Loaded in {__import__('time').time() - t0:.1f}s  |  "
        f"{params_m:.0f}M params  |  dim={spec.embed_dim}  layers={spec.n_layers}",
        flush=True,
    )
    return model, alphabet, converter, spec


def auto_batch_for_backbone(
    vram_gb: float,
    backbone: str,
) -> tuple[int, int]:
    """VRAM-aware (batch, accum) scaled from A100-40GB backbone preset."""
    spec = get_backbone_spec(backbone)
    base_bs, base_acc = spec.batch_a100_40
    scale = min(1.0, vram_gb / 40.0)
    if scale >= 0.95:
        return base_bs, base_acc
    if scale >= 0.7:
        return max(1, base_bs - 1), min(16, base_acc + 2)
    if scale >= 0.45:
        return max(1, base_bs // 2), min(16, base_acc * 2)
    return 1, min(16, base_acc * 3)


def apply_backbone_to_config(cfg: Any, backbone: str) -> Any:
    """Set embed_dim and batch presets on TrainConfig from backbone key."""
    spec = get_backbone_spec(backbone)
    cfg.esm_backbone = spec.key
    cfg.esm_embed_dim = spec.embed_dim
    if hasattr(cfg, "vram_gb") and cfg.vram_gb > 0:
        bs, acc = auto_batch_for_backbone(cfg.vram_gb, spec.key)
        cfg.batch_size = bs
        cfg.accum_steps = acc
    return cfg


def print_backbone_playbook(vram_gb: Optional[float] = None) -> None:
    """Print upgrade guidance for the user."""
    print(f"\n{'═' * 64}")
    print(" BACKBONE UPGRADE PLAYBOOK")
    print(f"{'═' * 64}")
    if vram_gb:
        print(f"  Detected VRAM: {vram_gb:.1f} GB")
    print("""
  Step 1 — Quick Screen (2–3h, 650M)
    DisorderNet_Colab_QuickScreen.ipynb  →  verdict HIGH/MODERATE?

  Step 2 — Full ultra on 650M (18–24h)
    DisorderNet_Colab_Pro.ipynb  →  QUALITY_PROFILE="ultra"
    If stacked AUC ≥ 0.87 → backbone is the bottleneck; go Step 3.

  Step 3 — Upgrade to ESM-2 3B (recommended)
    Colab: Runtime → A100 40GB + High RAM
    Set ESM_BACKBONE = "3B" and QUALITY_PROFILE = "ultra3b"
    pip install lightgbm xgboost  (v6-pro ensemble)
    Run Quick Screen with SCREEN_BACKBONE="3B" first (~4–6h paradigm mode)

  Step 4 — Optional multi-seed (Cell 7e)
    Two full CV runs: seed=42 and seed=43 → blend OOF

  Do NOT use T4/V100 for 3B — OOM likely.
  ESMDisPred 0.895 is CAID3; our target is DisProt CV ≥ 0.90.
""")
    print(f"{'─' * 64}")
    print(f"  {'Key':<6} {'Params':>8} {'Dim':>6} {'Min VRAM':>10}  GPU")
    for spec in BACKBONE_REGISTRY.values():
        print(
            f"  {spec.key:<6} {spec.params_m:>6}M {spec.embed_dim:>6} "
            f"{spec.min_vram_gb:>8.0f}GB  {spec.recommended_gpu}"
        )
    print(f"{'═' * 64}\n")
