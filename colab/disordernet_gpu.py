"""
DisorderNet GPU — Colab-ready training utilities.

Designed for Google Colab Pro (A100 / L4 / T4 / V100) with:
  - Pinned, compatible dependency versions
  - Modern torch.amp mixed precision (no deprecated cuda.amp APIs)
  - DisProt region filtering by disorder-related term names
  - Pre-tokenized dataset cache, TF32, cudnn benchmark
  - tqdm + live metric updates, checkpoint resume
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterator, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from colab.cv_splits import (
    config_fingerprint,
    get_cv_splits,
    get_fold_val_protein_ids,
    proteins_fingerprint,
    sort_proteins_deterministic,
)
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# DisProt disorder term whitelist (CAID-style: only experimentally verified IDRs)
# ---------------------------------------------------------------------------
DISORDER_TERMS = frozenset({
    "disorder",
    "flexible linker",
    "flexible n-terminal tail",
    "flexible c-terminal tail",
    "pre-molten globule",
    "molten globule",
    "entropic chain",
})

# Terms that explicitly indicate order — never label as disordered
ORDER_TERMS = frozenset({
    "order",
    "order to disorder",
})

# Disorder↔order transition regions (evaluated separately)
TRANSITION_TERMS = frozenset({
    "disorder to order",
    "order to disorder",
})

# Functional terms tracked for biological-utility enrichment (non-disorder labels)
FUNCTIONAL_TERM_GROUPS = {
    "protein binding": frozenset({"protein binding"}),
    "nucleic acid binding": frozenset({
        "dna binding", "rna binding", "nucleic acid binding",
    }),
    "post-translational regulation": frozenset({
        "phosphorylation display site",
        "molecular function regulator",
        "molecular function inhibitor activity",
        "molecular function activator activity",
    }),
    "condensate / assembly": frozenset({
        "molecular condensate scaffold activity",
        "self-assembly",
        "amyloid fibril formation",
    }),
    "lipid / small molecule binding": frozenset({
        "lipid binding", "small molecule binding", "metal ion binding",
        "calcium ion binding", "ion binding",
    }),
}

# Amino acid alphabet for lightweight physicochemical features (matches v6 CPU path)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {a: i for i, a in enumerate(AA_ALPHABET)}
# Kyte-Doolittle hydro, charge proxy, disorder propensity (Uversky-style scale subset)
_AA_HYDRO = torch.tensor(
    [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6,
     -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3],
    dtype=torch.float32,
)
_AA_CHARGE = torch.tensor(
    [0, 0, -1, -1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    dtype=torch.float32,
)
_AA_DISPROP = torch.tensor(
    [0.06, -0.02, 0.192, 0.736, -0.697, 0.166, 0.303, -0.486, 0.586, -0.326,
     -0.397, 0.007, 0.987, 0.318, 0.18, 0.341, 0.059, -0.121, -0.884, -0.510],
    dtype=torch.float32,
)


def _label_boundary_mask(labels: list[int], radius: int = 2) -> list[float]:
    """Weight residues near disorder↔order transitions (improves segment F1)."""
    n = len(labels)
    if n == 0:
        return []
    lab = np.asarray(labels, dtype=np.int8)
    core = np.zeros(n, dtype=np.bool_)
    for i in range(1, n):
        if lab[i] != lab[i - 1]:
            lo = max(0, i - radius)
            hi = min(n, i + radius)
            core[lo:hi] = True
    return core.astype(np.float32).tolist()


def _hallucination_weight_mask(
    labels: list[int],
    plddt: Optional[np.ndarray],
    cfg: TrainConfig,
) -> list[float]:
    """Upweight disordered residues with high AF pLDDT (hallucinated order)."""
    n = len(labels)
    if not cfg.use_hallucination_weighting or plddt is None or len(plddt) != n:
        return [1.0] * n

    weights: list[float] = []
    threshold = cfg.high_plddt_threshold
    hall_w = cfg.hallucination_weight
    for i, lab in enumerate(labels):
        if lab == 1 and not np.isnan(plddt[i]) and plddt[i] >= threshold:
            weights.append(hall_w)
        else:
            weights.append(1.0)
    return weights


def _load_plddt_for_protein(protein: dict, cache_dir: str) -> Optional[np.ndarray]:
    """Load cached AF pLDDT for training sample weights (cache-only)."""
    from colab.af_plddt import load_cached_plddt

    acc = protein.get("uniprot_acc", "")
    if not acc:
        return None
    return load_cached_plddt(acc, protein["sequence"], cache_dir=cache_dir)


def merge_plddt_for_training(
    plddt_af2: Optional[dict[str, np.ndarray]] = None,
    plddt_af3: Optional[dict[str, np.ndarray]] = None,
    prefer_af3: bool = True,
) -> dict[str, np.ndarray]:
    """Merge AF2/AF3 pLDDT caches for hallucination weighting (AF3 preferred)."""
    from colab.inference_fusion import build_combined_plddt_map

    combined, _ = build_combined_plddt_map(
        plddt_af2 or {},
        plddt_af3,
        prefer="af3" if prefer_af3 else "af2",
    )
    return combined


def build_plddt_cache_for_training(
    proteins: list,
    cache_dir: str = "af_plddt_cache",
    sleep_s: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Pre-fetch AF pLDDT before CV so hallucination weighting can use the cache.

    Safe to call even if cache already populated; only fetches missing entries.
    """
    from colab.af_plddt import fetch_plddt_batch

    existing = {
        p["id"]: plddt
        for p in proteins
        if (plddt := _load_plddt_for_protein(p, cache_dir)) is not None
    }
    missing = [p for p in proteins if p["id"] not in existing and p.get("uniprot_acc")]
    n_cached = len(existing)
    if missing:
        print(
            f"  pLDDT cache: {n_cached} hit, {len(missing)} to fetch "
            f"({len(proteins) - n_cached - len(missing)} without UniProt)"
        )
        fetched = fetch_plddt_batch(missing, cache_dir=cache_dir, sleep_s=sleep_s)
        existing.update(fetched)
        print(f"  pLDDT fetch done: {len(existing)}/{len(proteins)} proteins cached")
    elif n_cached:
        print(f"  pLDDT cache warm: {n_cached}/{len(proteins)} proteins (no fetch needed)")
    return existing


@dataclass
class TrainConfig:
    """Hyperparameters with Colab-friendly defaults."""

    seed: int = 42
    min_seq_len: int = 20
    max_seq_len: int = 1022
    min_disorder: int = 3
    min_order: int = 3

    batch_size: int = 4
    accum_steps: int = 4
    num_epochs: int = 20
    lr_lora: float = 8e-5
    lr_head: float = 4e-4
    weight_decay: float = 1e-2
    patience: int = 6
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.08

    lora_layers: int = 12
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_on_k: bool = True
    head_dropout: float = 0.12

    # Loss & features (v2 performance defaults)
    use_focal_loss: bool = True
    focal_gamma: float = 2.0
    boundary_weight: float = 2.5
    boundary_radius: int = 2
    use_physico_features: bool = True
    physico_dim: int = 32
    esm_fusion_layers: int = 4

    n_folds: int = 5
    num_workers: int = 2
    checkpoint_dir: str = "checkpoints"
    data_cache: str = "disprot_raw.json"

    # Performance toggles
    use_gradient_checkpointing: bool = True
    deterministic: bool = False  # False → cudnn.benchmark + TF32 for speed

    # Segment-aware early stopping & post-processing
    use_segment_early_stop: bool = True
    auc_score_weight: float = 0.5
    ap_score_weight: float = 0.3
    segment_score_weight: float = 0.2
    segment_min_region_len: int = 5
    segment_postprocess_min_len: int = 5
    segment_postprocess_max_gap: int = 3

    # Checkpoint selection: "composite" (AUC+AP+SegF1) or "auc" (AUC-only, rigor-first)
    early_stop_mode: str = "composite"

    # SOTA track (profile "sota")
    head_type: str = "cnn"  # "cnn" | "sota" (CNN + Transformer)
    use_dice_loss: bool = False
    dice_loss_weight: float = 0.25
    label_smoothing: float = 0.0
    use_ema: bool = False
    ema_decay: float = 0.999
    compact_checkpoints: bool = False

    # Advanced SOTA training (profile "sota")
    use_rdrop: bool = False
    rdrop_weight: float = 0.5
    use_tversky_loss: bool = False
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_weight: float = 0.15
    use_swa: bool = False
    swa_start_frac: float = 0.75
    use_v6_distill: bool = False
    v6_distill_weight: float = 0.15
    v6_distill_temperature: float = 2.0

    # Ultra / radical SOTA track
    use_rich_features: bool = False  # full 162-dim features_fast stream
    fusion_type: str = "softmax"  # "softmax" | "attention"
    lora_on_out_proj: bool = False
    lora_on_ffn: bool = False
    unfreeze_last_layers: int = 0  # fine-tune tail LayerNorm + FFN
    lr_esm_tail: float = 1e-5

    # Inference-time boosts
    use_mc_dropout_tta: bool = False
    mc_dropout_tta_passes: int = 6

    # AF hallucination hard-negative weighting (disordered + high pLDDT)
    use_hallucination_weighting: bool = True
    hallucination_weight: float = 3.0
    high_plddt_threshold: float = 70.0
    af_plddt_cache_dir: str = "af_plddt_cache"

    # Set automatically by setup_environment()
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    amp_dtype: torch.dtype = torch.float16
    pin_memory: bool = False
    gpu_name: str = "cpu"
    vram_gb: float = 0.0

    def effective_batch(self) -> int:
        return self.batch_size * self.accum_steps

    @classmethod
    def from_profile(cls, profile: str = "balanced", **overrides) -> "TrainConfig":
        """
        Training quality presets.

        balanced — default v2 (rank 16, 20 epochs)
        max      — higher capacity (rank 32, 25 epochs, larger physico stream)
        sota     — SOTA push: rank 64, Transformer head, Dice+EMA, compact ckpt
        ultra    — maximum push: rich features, attn fusion, FFN LoRA, v6-pro stack
        screen   — fast paradigm check (~2h): 8 epochs, CNN head, 10 LoRA layers
        screen_plus — paradigm fidelity (~4h): mini-ultra on subset
        """
        presets: dict[str, dict] = {
            "balanced": {},
            "max": {
                "lora_rank": 32,
                "lora_alpha": 64,
                "num_epochs": 25,
                "patience": 8,
                "lr_lora": 5e-5,
                "lr_head": 3e-4,
                "boundary_weight": 3.0,
                "physico_dim": 48,
                "focal_gamma": 2.5,
            },
            "sota": {
                "lora_rank": 64,
                "lora_alpha": 128,
                "lora_layers": 16,
                "num_epochs": 30,
                "patience": 10,
                "lr_lora": 3e-5,
                "lr_head": 2e-4,
                "boundary_weight": 3.5,
                "physico_dim": 64,
                "focal_gamma": 2.5,
                "esm_fusion_layers": 8,
                "head_type": "sota",
                "use_dice_loss": True,
                "dice_loss_weight": 0.3,
                "label_smoothing": 0.02,
                "use_ema": True,
                "ema_decay": 0.999,
                "compact_checkpoints": True,
                "auc_score_weight": 0.45,
                "ap_score_weight": 0.20,
                "segment_score_weight": 0.35,
                "hallucination_weight": 3.5,
                "use_rdrop": True,
                "rdrop_weight": 0.5,
                "use_tversky_loss": True,
                "tversky_weight": 0.12,
                "use_swa": True,
                "swa_start_frac": 0.70,
                "use_v6_distill": True,
                "v6_distill_weight": 0.12,
            },
            "ultra": {
                "lora_rank": 128,
                "lora_alpha": 256,
                "lora_layers": 20,
                "num_epochs": 35,
                "patience": 12,
                "lr_lora": 2e-5,
                "lr_head": 1.5e-4,
                "lr_esm_tail": 8e-6,
                "boundary_weight": 4.0,
                "physico_dim": 96,
                "focal_gamma": 3.0,
                "esm_fusion_layers": 12,
                "head_type": "sota",
                "use_dice_loss": True,
                "dice_loss_weight": 0.35,
                "label_smoothing": 0.03,
                "use_ema": True,
                "ema_decay": 0.9995,
                "compact_checkpoints": True,
                "auc_score_weight": 0.40,
                "ap_score_weight": 0.15,
                "segment_score_weight": 0.45,
                "hallucination_weight": 4.0,
                "use_rdrop": True,
                "rdrop_weight": 0.6,
                "use_tversky_loss": True,
                "tversky_weight": 0.15,
                "use_swa": True,
                "swa_start_frac": 0.65,
                "use_v6_distill": True,
                "v6_distill_weight": 0.15,
                "use_rich_features": True,
                "fusion_type": "attention",
                "lora_on_out_proj": True,
                "lora_on_ffn": True,
                "unfreeze_last_layers": 2,
                "use_mc_dropout_tta": True,
                "mc_dropout_tta_passes": 6,
            },
            "screen": {
                "lora_rank": 32,
                "lora_alpha": 64,
                "lora_layers": 10,
                "num_epochs": 8,
                "patience": 3,
                "lr_lora": 5e-5,
                "lr_head": 3e-4,
                "esm_fusion_layers": 4,
                "head_type": "cnn",
                "use_physico_features": True,
                "physico_dim": 48,
                "compact_checkpoints": True,
                "early_stop_mode": "auc",
                "use_v6_distill": False,
                "use_rdrop": False,
                "use_swa": False,
                "use_ema": False,
                "use_hallucination_weighting": False,
            },
            "screen_plus": {
                "lora_rank": 64,
                "lora_alpha": 128,
                "lora_layers": 12,
                "num_epochs": 10,
                "patience": 4,
                "lr_lora": 3e-5,
                "lr_head": 2e-4,
                "esm_fusion_layers": 6,
                "head_type": "sota",
                "physico_dim": 64,
                "use_dice_loss": True,
                "dice_loss_weight": 0.25,
                "use_ema": True,
                "compact_checkpoints": True,
                "early_stop_mode": "auc",
                "use_rich_features": True,
                "fusion_type": "attention",
                "use_v6_distill": False,
                "use_rdrop": False,
                "use_swa": False,
                "use_hallucination_weighting": False,
            },
        }
        if profile not in presets:
            raise ValueError(
                f"Unknown profile '{profile}'. Choose: {list(presets)}"
            )
        obj = cls(**{**presets[profile], **overrides})
        obj._profile_name = profile  # type: ignore[attr-defined]
        return obj


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def install_dependencies(quiet: bool = True) -> None:
    """Install pinned packages compatible with Colab's pre-installed PyTorch."""
    flag = "-q" if quiet else ""
    pkgs = [
        "fair-esm==2.0.0",
        "scikit-learn>=1.3,<2",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "requests>=2.31",
        "biopython>=1.83",
        "tqdm>=4.66",
        "pandas>=2.0",
    ]
    cmd = [sys.executable, "-m", "pip", "install", flag, *pkgs]
    cmd = [c for c in cmd if c]  # drop empty -q when not quiet
    print("Installing packages...")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"Packages installed in {time.time() - t0:.0f}s")


def _auto_batch_size(vram_gb: float) -> tuple[int, int]:
    """Return (batch_size, accum_steps) tuned to available VRAM."""
    if vram_gb >= 70:      # A100 80GB, RTX 6000 Ada
        return 8, 2
    if vram_gb >= 35:      # A100 40GB
        return 6, 3
    if vram_gb >= 20:      # L4, A10
        return 4, 4
    if vram_gb >= 14:      # T4, V100 16GB
        return 2, 8
    return 1, 16           # fallback


def setup_environment(cfg: TrainConfig) -> TrainConfig:
    """GPU detection, seeding, and performance flags. Mutates and returns cfg."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU detected. In Colab: Runtime → Change runtime type → "
            "GPU (A100 or L4 recommended)."
        )

    props = torch.cuda.get_device_properties(0)
    cfg.gpu_name = torch.cuda.get_device_name(0)
    cfg.vram_gb = props.total_memory / (1024 ** 3)
    cfg.device = torch.device("cuda")
    cfg.pin_memory = True

    # bfloat16 on Ampere+ (A100, L4); float16 elsewhere
    major, _ = torch.cuda.get_device_capability(0)
    cfg.amp_dtype = torch.bfloat16 if major >= 8 else torch.float16

    bs, accum = _auto_batch_size(cfg.vram_gb)
    if cfg.batch_size == 4 and cfg.accum_steps == 4:
        cfg.batch_size = bs
        cfg.accum_steps = accum

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'=' * 55}")
    print(f"  Environment : {'Google Colab' if _in_colab() else 'local'}")
    print(f"  GPU         : {cfg.gpu_name}")
    print(f"  VRAM        : {cfg.vram_gb:.1f} GB")
    print(f"  CUDA        : {torch.version.cuda}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  AMP dtype   : {cfg.amp_dtype}")
    print(f"  Batch × accum = {cfg.batch_size} × {cfg.accum_steps} "
          f"(effective {cfg.effective_batch()})")
    if _in_colab() and cfg.num_workers != 0:
        cfg.num_workers = 0
        print("  DataLoader workers: 0 (Colab-safe; avoids multiprocessing hangs)")
    print(f"{'=' * 55}")
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_disprot(cache_path: str = "disprot_raw.json") -> list:
    """Fetch DisProt via REST API with local JSON cache (versioned metadata wrapper)."""
    if os.path.exists(cache_path):
        print(f"Loading cached DisProt from '{cache_path}'...")
        with open(cache_path) as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            meta = payload.get("meta", {})
            n = meta.get("n_entries", len(payload["data"]))
            fetched = meta.get("fetched_at", "unknown date")
            sha = meta.get("content_sha256", "")[:12]
            print(f"  Cache meta: {n} entries  fetched={fetched}  sha={sha}")
            return payload["data"]
        return payload

    base_url = "https://disprot.org/api/search"
    params = {
        "release": "current",
        "show_ambiguous": "false",
        "show_obsolete": "false",
        "format": "json",
        "page": 0,
        "per_page": 100,
    }

    all_entries: list = []
    print("Downloading DisProt database...")
    t0 = time.time()

    pbar = tqdm(desc="DisProt pages", unit="page")
    while True:
        try:
            resp = requests.get(base_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            print(f"  Request error page {params['page']}: {exc}. Retrying...")
            time.sleep(5)
            continue

        entries = data.get("data", [])
        if not entries:
            break

        all_entries.extend(entries)
        total = data.get("total", len(all_entries))
        pbar.set_postfix(entries=len(all_entries), total=total)
        pbar.update(1)

        if len(all_entries) >= total:
            break
        params["page"] += 1
        time.sleep(0.2)

    pbar.close()
    elapsed = time.time() - t0
    print(f"Downloaded {len(all_entries)} entries in {elapsed:.0f}s")

    meta = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_entries": len(all_entries),
        "api_release": params["release"],
        "content_sha256": hashlib.sha256(
            json.dumps(all_entries, sort_keys=True).encode()
        ).hexdigest(),
    }
    with open(cache_path, "w") as f:
        json.dump({"meta": meta, "data": all_entries}, f)
    print(f"  Saved DisProt cache (sha256={meta['content_sha256'][:12]}...)")
    return all_entries


def get_disprot_cache_meta(cache_path: str = "disprot_raw.json") -> Optional[dict]:
    """Return DisProt cache metadata if present (legacy flat caches return None)."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(payload, dict) and "meta" in payload:
        return dict(payload["meta"])
    return None


def _normalize_term(term_name: Optional[str]) -> str:
    if not term_name:
        return ""
    return term_name.strip().lower()


def _region_is_disorder(term_name: Optional[str]) -> bool:
    t = _normalize_term(term_name)
    if not t or t in ORDER_TERMS:
        return False
    return t in DISORDER_TERMS


def _region_is_transition(term_name: Optional[str]) -> bool:
    return _normalize_term(term_name) in TRANSITION_TERMS


def process_disprot(
    entries: list,
    cfg: TrainConfig,
) -> tuple[list, Counter]:
    """Parse DisProt entries into protein dicts with binary disorder labels."""
    proteins: list = []
    skipped: Counter = Counter()

    for entry in entries:
        seq = entry.get("sequence", "")
        if not seq:
            skipped["no_sequence"] += 1
            continue
        if len(seq) < cfg.min_seq_len:
            skipped["too_short"] += 1
            continue
        if len(seq) > cfg.max_seq_len:
            skipped["too_long"] += 1
            continue

        labels = [0] * len(seq)
        transition_mask = [0] * len(seq)
        functional_regions: list = []
        has_disorder_term = False

        for region in entry.get("regions", []):
            term_raw = region.get("term_name") or region.get("type") or ""
            term = _normalize_term(term_raw)
            start = region.get("start", 0)
            end = region.get("end", 0)
            if not (start and end):
                continue

            s0 = start - 1
            e0 = min(end, len(seq))

            if term and term not in ORDER_TERMS:
                functional_regions.append({
                    "start": int(start),
                    "end": int(end),
                    "term_name": term_raw.strip() if term_raw else term,
                    "term_norm": term,
                })

            if _region_is_transition(term_raw):
                for i in range(s0, e0):
                    transition_mask[i] = 1

            if not _region_is_disorder(term_raw):
                continue

            has_disorder_term = True
            for i in range(s0, e0):
                labels[i] = 1

        if not has_disorder_term:
            skipped["no_disorder_annotation"] += 1
            continue

        n_dis = sum(labels)
        n_ord = len(seq) - n_dis
        if n_dis < cfg.min_disorder:
            skipped["too_few_disorder"] += 1
            continue
        if n_ord < cfg.min_order:
            skipped["too_few_order"] += 1
            continue

        proteins.append({
            "id": entry.get("disprot_id", ""),
            "uniprot_acc": (entry.get("acc") or "").strip(),
            "organism": (entry.get("organism") or entry.get("taxon_name") or "").strip(),
            "sequence": seq,
            "labels": labels,
            "length": len(seq),
            "n_dis": n_dis,
            "frac_dis": n_dis / len(seq),
            "functional_regions": functional_regions,
            "transition_mask": transition_mask,
            "n_functional_regions": len(functional_regions),
        })

    proteins = sort_proteins_deterministic(proteins)
    return proteins, skipped


def print_dataset_summary(proteins: list, skipped: Counter) -> tuple[int, int, float]:
    total_res = sum(p["length"] for p in proteins)
    total_dis = sum(p["n_dis"] for p in proteins)
    frac_dis = total_dis / max(total_res, 1)

    print(f"\n{'─' * 55}")
    print(f"  Proteins accepted : {len(proteins):,}")
    print(f"  Residues total    : {total_res:,}")
    print(f"  Disordered        : {total_dis:,}  ({100 * frac_dis:.1f}%)")
    print(f"  Ordered           : {total_res - total_dis:,}  ({100 * (1 - frac_dis):.1f}%)")
    if frac_dis > 0:
        print(f"  Class imbalance   : 1 : {(1 - frac_dis) / frac_dis:.1f}")
    print(f"{'─' * 55}")
    print(f"  Skipped: {dict(skipped)}")

    fracs = [p["frac_dis"] for p in proteins]
    if fracs:
        print(
            f"\n  Disorder fraction: mean={np.mean(fracs):.3f}  "
            f"median={np.median(fracs):.3f}  "
            f"min={np.min(fracs):.3f}  max={np.max(fracs):.3f}"
        )
    return total_res, total_dis, frac_dis


def print_training_config_summary(cfg: TrainConfig, proteins: Optional[list] = None) -> None:
    """Print active training features and hyperparameters for Colab progress logs."""
    print(f"\n{'─' * 55}")
    print("  Training configuration")
    print(f"{'─' * 55}")
    print(f"  Profile            : {getattr(cfg, '_profile_name', 'custom')}")
    print(f"  LoRA rank / layers : {cfg.lora_rank} / {cfg.lora_layers}")
    print(f"  Epochs / patience  : {cfg.num_epochs} / {cfg.patience}")
    print(f"  Batch × accum      : {cfg.batch_size} × {cfg.accum_steps} "
          f"(effective {cfg.effective_batch()})")
    print(f"  Physico features   : {cfg.use_physico_features} (dim={cfg.physico_dim})")
    print(f"  Focal loss         : {cfg.use_focal_loss} (γ={cfg.focal_gamma})")
    print(f"  Segment early-stop : {cfg.use_segment_early_stop} "
          f"({cfg.auc_score_weight}·AUC + {cfg.ap_score_weight}·AP + "
          f"{cfg.segment_score_weight}·SegF1)")
    print(f"  Early-stop mode    : {cfg.early_stop_mode}")
    print(f"  Head / SOTA        : {cfg.head_type}  dice={cfg.use_dice_loss}  ema={cfg.use_ema}")
    print(f"  Rich features      : {cfg.use_rich_features}  fusion={cfg.fusion_type}")
    print(f"  LoRA FFN/out       : {cfg.lora_on_ffn}/{cfg.lora_on_out_proj}  "
          f"unfreeze_tail={cfg.unfreeze_last_layers}")
    print(f"  Compact ckpt       : {cfg.compact_checkpoints}")
    print(f"  Hallucination wt   : {cfg.use_hallucination_weighting} "
          f"(×{cfg.hallucination_weight} @ pLDDT≥{cfg.high_plddt_threshold})")
    if proteins is not None:
        print(f"  Proteins / residues: {len(proteins):,} / "
              f"{sum(p['length'] for p in proteins):,}")
        print(f"  CV folds           : {cfg.n_folds}")
    print(f"{'─' * 55}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for nn.Linear (frozen base + trainable A·B)."""

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(in_f, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_dropout = nn.Dropout(dropout)

        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class PhysicoFeatureEncoder(nn.Module):
    """Lightweight sequence physics (v6-style) fused with ESM embeddings."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.register_buffer("hydro", _AA_HYDRO)
        self.register_buffer("charge", _AA_CHARGE)
        self.register_buffer("disprop", _AA_DISPROP)
        self.net = nn.Sequential(
            nn.Conv1d(23, 64, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(64, out_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )

    def forward(self, aa_idx: torch.Tensor) -> torch.Tensor:
        """aa_idx: (B, L) integer indices 0–19."""
        aa_idx = aa_idx.clamp(0, 19)
        onehot = torch.nn.functional.one_hot(aa_idx, num_classes=20).float()
        props = torch.stack([
            self.hydro[aa_idx],
            self.charge[aa_idx],
            self.disprop[aa_idx],
        ], dim=-1)
        x = torch.cat([onehot, props], dim=-1).permute(0, 2, 1)
        return self.net(x).permute(0, 2, 1)


class ESMLayerFusion(nn.Module):
    """Learned fusion of the last N ESM layer representations."""

    def __init__(self, n_layers: int, dim: int = 1280):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, layer_stack: torch.Tensor) -> torch.Tensor:
        """layer_stack: (B, L, D, N_layers)."""
        w = torch.softmax(self.weights, dim=0)
        fused = (layer_stack * w.view(1, 1, 1, -1)).sum(dim=-1)
        return self.norm(fused)


class ESMAttentionFusion(nn.Module):
    """Context-dependent attention over ESM layer stack (stronger than scalar softmax)."""

    def __init__(self, n_layers: int, dim: int = 1280):
        super().__init__()
        hidden = max(dim // 8, 64)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_layers),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, layer_stack: torch.Tensor) -> torch.Tensor:
        """layer_stack: (B, L, D, N_layers)."""
        context = layer_stack[..., -1]
        scores = self.gate(context)
        w = torch.softmax(scores, dim=-1)
        fused = (layer_stack * w.unsqueeze(-2)).sum(dim=-1)
        return self.norm(fused)


class DisorderCNNHead(nn.Module):
    """Multi-scale dilated 1D CNN — sharper IDR boundaries than single-kernel stack."""

    def __init__(self, in_dim: int = 1280, dropout: float = 0.12):
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
        self.merge = nn.Sequential(
            nn.Conv1d(mid * 3, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 1, kernel_size=1),
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        parts = [b(x) for b in self.branches]
        merged = self.merge(torch.cat(parts, dim=1))
        return (merged + self.skip(x)).squeeze(1)


class DisorderCNNHeadLegacy(nn.Module):
    """Previous 3-layer CNN (kept for reference / tests)."""

    def __init__(self, in_dim: int = 1280, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 512, kernel_size=15, padding=7),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(512, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 1, kernel_size=5, padding=2),
        )
        self.skip = nn.Conv1d(in_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        return (self.net(x) + self.skip(x)).squeeze(1)


class DisorderNetGPU(nn.Module):
    """ESM-2 650M + LoRA (Q/V/K/out/FFN, last N layers) + disorder head."""

    def __init__(
        self,
        esm_model: nn.Module,
        cfg: TrainConfig,
        verbose: bool = True,
    ):
        super().__init__()
        self.esm = esm_model
        self.cfg = cfg

        for p in self.esm.parameters():
            p.requires_grad = False

        if cfg.use_gradient_checkpointing:
            if hasattr(self.esm, "set_gradient_checkpointing"):
                self.esm.set_gradient_checkpointing(True)

        n_layers = len(self.esm.layers)
        start = n_layers - cfg.lora_layers
        self._lora_modules: list[LoRALinear] = []
        proj_names = ["q_proj", "v_proj"]
        if cfg.lora_on_k:
            proj_names.append("k_proj")
        if cfg.lora_on_out_proj:
            proj_names.append("out_proj")

        for layer_idx in range(start, n_layers):
            layer = self.esm.layers[layer_idx]
            attn = layer.self_attn
            for proj_name in proj_names:
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, nn.Linear):
                    lora = LoRALinear(
                        proj, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
                    )
                    setattr(attn, proj_name, lora)
                    self._lora_modules.append(lora)

            if cfg.lora_on_ffn:
                for ffn_name in ("fc1", "fc2"):
                    ffn = getattr(layer, ffn_name, None)
                    if isinstance(ffn, nn.Linear):
                        lora = LoRALinear(
                            ffn, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
                        )
                        setattr(layer, ffn_name, lora)
                        self._lora_modules.append(lora)

        self._esm_tail_params: list[nn.Parameter] = []
        if cfg.unfreeze_last_layers > 0:
            tail_start = n_layers - cfg.unfreeze_last_layers
            for layer_idx in range(tail_start, n_layers):
                layer = self.esm.layers[layer_idx]
                for mod_name in ("self_attn_layer_norm", "final_layer_norm"):
                    mod = getattr(layer, mod_name, None)
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad = True
                            self._esm_tail_params.append(p)
                if not cfg.lora_on_ffn:
                    for ffn_name in ("fc1", "fc2"):
                        mod = getattr(layer, ffn_name, None)
                        if isinstance(mod, nn.Linear):
                            for p in mod.parameters():
                                p.requires_grad = True
                                self._esm_tail_params.append(p)

        fusion_n = min(cfg.esm_fusion_layers, n_layers)
        self._fusion_layer_ids = list(range(n_layers - fusion_n, n_layers))
        if cfg.fusion_type == "attention":
            self.layer_fusion = ESMAttentionFusion(fusion_n, dim=1280)
        else:
            self.layer_fusion = ESMLayerFusion(fusion_n, dim=1280)

        self.use_rich = cfg.use_rich_features
        self.use_physico = cfg.use_physico_features and not self.use_rich
        extra_dim = 0
        self.rich_encoder = None
        self.physico = None

        if self.use_rich:
            from colab.rich_features import RichFeatureEncoder
            self.rich_encoder = RichFeatureEncoder(out_dim=cfg.physico_dim)
            extra_dim = cfg.physico_dim
        elif self.use_physico:
            self.physico = PhysicoFeatureEncoder(cfg.physico_dim)
            extra_dim = cfg.physico_dim

        head_in = 1280 + extra_dim

        self.head_type = cfg.head_type
        if cfg.head_type == "sota":
            from colab.sota_heads import DisorderSOTAHead
            self.head = DisorderSOTAHead(
                in_dim=head_in,
                dropout=cfg.head_dropout,
                n_transformer_layers=3 if cfg.use_rich_features else 2,
            )
        else:
            self.head = DisorderCNNHead(in_dim=head_in, dropout=cfg.head_dropout)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            lora_projs = len(self._lora_modules)
            print(
                f"  LoRA layers {start}–{n_layers - 1} ({lora_projs} projections)"
            )
            print(
                f"  Fusion={cfg.fusion_type}  rich={self.use_rich}  "
                f"unfreeze_tail={cfg.unfreeze_last_layers}"
            )
            print(
                f"  Parameters: {total / 1e6:.1f}M total, "
                f"{trainable / 1e6:.2f}M trainable ({100 * trainable / total:.2f}%)"
            )

    def train(self, mode: bool = True) -> "DisorderNetGPU":
        """Keep frozen ESM in eval mode (LayerNorm) while LoRA dropout is active."""
        super().train(mode)
        self.esm.eval()
        return self

    def get_lora_params(self) -> Iterator[nn.Parameter]:
        for m in self._lora_modules:
            yield m.lora_A
            yield m.lora_B

    def get_head_params(self) -> Iterator[nn.Parameter]:
        params = list(self.head.parameters())
        params.extend(self.layer_fusion.parameters())
        if self.physico is not None:
            params.extend(self.physico.parameters())
        if self.rich_encoder is not None:
            params.extend(self.rich_encoder.parameters())
        return iter(params)

    def get_esm_tail_params(self) -> Iterator[nn.Parameter]:
        return iter(self._esm_tail_params)

    def forward(
        self,
        tokens: torch.Tensor,
        aa_idx: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rich_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = self.esm(tokens, repr_layers=self._fusion_layer_ids, return_contacts=False)
        layer_hiddens = [
            out["representations"][i][:, 1:-1, :] for i in self._fusion_layer_ids
        ]
        stack = torch.stack(layer_hiddens, dim=-1)
        embeddings = self.layer_fusion(stack)

        if self.rich_encoder is not None:
            if rich_feats is None:
                raise ValueError("rich_feats required when use_rich_features=True")
            embeddings = torch.cat([embeddings, self.rich_encoder(rich_feats)], dim=-1)
        elif self.physico is not None:
            if aa_idx is None:
                raise ValueError("aa_idx required when use_physico_features=True")
            embeddings = torch.cat([embeddings, self.physico(aa_idx)], dim=-1)

        if self.head_type == "sota":
            if pad_mask is None:
                pad_mask = torch.ones(
                    embeddings.shape[0], embeddings.shape[1],
                    dtype=torch.bool, device=embeddings.device,
                )
            return self.head(embeddings, pad_mask=pad_mask)
        return self.head(embeddings)


def load_esm_model(device: torch.device):
    """Load ESM-2 650M and batch converter."""
    import esm

    print("Loading ESM-2 650M...")
    t0 = time.time()
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    converter = alphabet.get_batch_converter()
    elapsed = time.time() - t0
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {elapsed:.1f}s  |  {params_m:.0f}M parameters")
    return model, alphabet, converter


# ---------------------------------------------------------------------------
# Dataset (pre-tokenized cache)
# ---------------------------------------------------------------------------
class DisProtDataset(Dataset):
    """Pre-tokenizes all proteins once to avoid repeated batch_converter calls."""

    def __init__(
        self,
        proteins: list,
        batch_converter,
        cache: Optional[dict] = None,
        boundary_radius: int = 2,
        cfg: Optional[TrainConfig] = None,
        plddt_by_id: Optional[dict[str, np.ndarray]] = None,
    ):
        self.proteins = proteins
        self._cache = cache if cache is not None else {}
        self.boundary_radius = boundary_radius
        self.cfg = cfg
        self.plddt_by_id = plddt_by_id or {}

        missing = [p for p in proteins if p["id"] not in self._cache]
        if missing:
            for p in tqdm(missing, desc="Tokenizing", leave=False):
                _, _, tokens = batch_converter([(p["id"], p["sequence"])])
                labels = p["labels"]
                boundary = _label_boundary_mask(labels, radius=self.boundary_radius)
                trans = p.get("transition_mask") or [0] * len(labels)
                is_boundary = [
                    1.0 if (boundary[i] or trans[i]) else 0.0 for i in range(len(labels))
                ]

                plddt = self.plddt_by_id.get(p["id"])
                if plddt is None and self.cfg is not None:
                    plddt = _load_plddt_for_protein(p, self.cfg.af_plddt_cache_dir)

                hall_weights = (
                    _hallucination_weight_mask(labels, plddt, self.cfg)
                    if self.cfg is not None else [1.0] * len(labels)
                )
                boundary_weight = self.cfg.boundary_weight if self.cfg is not None else 1.0
                sample_weight = [
                    (1.0 + (boundary_weight - 1.0) * is_boundary[i]) * hall_weights[i]
                    for i in range(len(labels))
                ]

                aa_idx = [AA_TO_IDX.get(c, 0) for c in p["sequence"]]
                entry = {
                    "tokens": tokens.squeeze(0),
                    "labels": torch.tensor(labels, dtype=torch.float32),
                    "aa_idx": torch.tensor(aa_idx, dtype=torch.long),
                    "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
                }
                if self.cfg is not None and self.cfg.use_rich_features:
                    from colab.rich_features import compute_rich_features
                    rich = compute_rich_features(p["sequence"])
                    length = min(len(labels), rich.shape[0])
                    entry["rich_feats"] = torch.tensor(rich[:length], dtype=torch.float32)
                self._cache[p["id"]] = entry

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int):
        p = self.proteins[idx]
        item = self._cache[p["id"]]
        labels = item["labels"]
        mask = torch.ones(labels.shape[0], dtype=torch.bool)
        rich = item.get("rich_feats")
        return (
            item["tokens"],
            labels,
            mask,
            item["aa_idx"],
            item["sample_weight"],
            rich,
            p["id"],
        )


def disprot_collate(batch):
    tokens_list, labels_list, mask_list, aa_list, weight_list, rich_list, ids = zip(*batch)
    max_tok = max(t.shape[0] for t in tokens_list)
    max_seq = max_tok - 2
    pad_idx = 1

    tokens_padded = torch.full((len(batch), max_tok), pad_idx, dtype=torch.long)
    labels_padded = torch.zeros(len(batch), max_seq, dtype=torch.float32)
    mask_padded = torch.zeros(len(batch), max_seq, dtype=torch.bool)
    aa_padded = torch.zeros(len(batch), max_seq, dtype=torch.long)
    weight_padded = torch.ones(len(batch), max_seq, dtype=torch.float32)
    rich_padded = None
    if any(r is not None for r in rich_list):
        from colab.rich_features import RICH_FEATURE_DIM
        rich_padded = torch.zeros(len(batch), max_seq, RICH_FEATURE_DIM, dtype=torch.float32)
        for i, rich in enumerate(rich_list):
            if rich is None:
                continue
            lr = rich.shape[0]
            rich_padded[i, :lr] = rich

    for i, (tok, lab, msk, aa, wt) in enumerate(
        zip(tokens_list, labels_list, mask_list, aa_list, weight_list)
    ):
        lt, ls = tok.shape[0], lab.shape[0]
        tokens_padded[i, :lt] = tok
        labels_padded[i, :ls] = lab
        mask_padded[i, :ls] = msk
        aa_padded[i, :ls] = aa
        weight_padded[i, :ls] = wt

    return tokens_padded, labels_padded, mask_padded, aa_padded, weight_padded, rich_padded, list(ids)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _compute_pos_weight(proteins: list, device: torch.device) -> torch.Tensor:
    total_pos = sum(p["n_dis"] for p in proteins)
    total_neg = sum(p["length"] - p["n_dis"] for p in proteins)
    pw = min(total_neg / max(total_pos, 1), 12.0)
    return torch.tensor([pw], device=device)


def _sequence_pad_mask(tokens: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
    """Valid residue positions (B, L) from token batch and collate mask."""
    return seq_mask


def _forward_logits(
    model: nn.Module,
    tokens: torch.Tensor,
    aa_idx: Optional[torch.Tensor],
    mask: torch.Tensor,
    rich_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    use_physico = getattr(model, "use_physico", False)
    aa = aa_idx if use_physico else None
    if getattr(model, "head_type", "cnn") == "sota":
        return model(tokens, aa_idx=aa, pad_mask=mask, rich_feats=rich_feats)
    return model(tokens, aa_idx=aa, rich_feats=rich_feats)


def _disorder_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: Optional[torch.Tensor],
    sample_weight: Optional[torch.Tensor],
    cfg: TrainConfig,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Focal or BCE loss; delegates to SOTA composite loss when Dice enabled."""
    if getattr(cfg, "use_dice_loss", False) and mask is not None and logits.dim() == 2:
        from colab.sota_losses import composite_disorder_loss
        return composite_disorder_loss(
            logits, labels, mask, pos_weight, sample_weight, cfg,
        )

    if mask is not None and logits.dim() == 2:
        logits = logits[mask]
        labels = labels[mask]
        if sample_weight is not None:
            sample_weight = sample_weight[mask]

    if cfg.use_focal_loss:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probs, 1.0 - probs)
        loss = bce * ((1.0 - pt) ** cfg.focal_gamma)
    else:
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight, reduction="none",
        )

    if sample_weight is not None:
        loss = loss * sample_weight

    return loss.mean()


def _warmup_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@torch.inference_mode()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    pos_weight: Optional[torch.Tensor] = None,
    cfg: Optional[TrainConfig] = None,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    for tokens, labels, mask, aa_idx, sample_weight, rich_feats, _ in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        aa_idx = aa_idx.to(device, non_blocking=True)
        sample_weight = sample_weight.to(device, non_blocking=True)
        rich_t = None
        if rich_feats is not None:
            rich_t = rich_feats.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = _forward_logits(
                model, tokens, aa_idx if getattr(model, "use_physico", False) else None, mask,
                rich_feats=rich_t,
            )

        if cfg is not None:
            loss = _disorder_loss(
                logits, labels, pos_weight, sample_weight, cfg, mask=mask,
            )
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], labels[mask], pos_weight=pos_weight,
            )
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits)
        all_probs.append(probs[mask].float().cpu().numpy())
        all_labels.append(labels[mask].cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs >= 0.5).astype(int)
    labels_int = all_labels.astype(int)

    return {
        "loss": total_loss / max(n_batches, 1),
        "auc": roc_auc_score(all_labels, all_probs),
        "ap": average_precision_score(all_labels, all_probs),
        "f1": f1_score(labels_int, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels_int, preds),
        "probs": all_probs,
        "labels": all_labels,
    }


def _pad_teacher_probs(
    batch_ids: list[str],
    proteins_by_id: dict[str, dict],
    teacher_by_id: dict[str, np.ndarray],
    max_seq: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Pad per-residue v6 teacher probabilities for a batch."""
    if not teacher_by_id:
        return None
    padded = torch.zeros(len(batch_ids), max_seq, dtype=torch.float32, device=device)
    has_any = False
    for i, pid in enumerate(batch_ids):
        if pid not in teacher_by_id:
            continue
        teacher = teacher_by_id[pid]
        p = proteins_by_id.get(pid)
        if p is None:
            continue
        length = min(len(teacher), p["length"], max_seq)
        if length > 0:
            padded[i, :length] = torch.from_numpy(teacher[:length])
            has_any = True
    return padded if has_any else None


def train_fold(
    fold_idx: int,
    proteins_train: list,
    proteins_val: list,
    esm_backbone: nn.Module,
    batch_converter,
    token_cache: dict,
    cfg: TrainConfig,
    plddt_by_id: Optional[dict[str, np.ndarray]] = None,
    v6_teacher_by_id: Optional[dict[str, np.ndarray]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Train one CV fold. Returns metrics dict including best checkpoint path."""
    device, amp_dtype = cfg.device, cfg.amp_dtype
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f" FOLD {fold_idx + 1}  |  train={len(proteins_train)}  val={len(proteins_val)}")
    print(f"{'═' * 60}")

    fold_model = DisorderNetGPU(esm_backbone, cfg, verbose=True).to(device)

    train_ds = DisProtDataset(
        proteins_train, batch_converter, token_cache,
        boundary_radius=cfg.boundary_radius, cfg=cfg, plddt_by_id=plddt_by_id,
    )
    val_ds = DisProtDataset(
        proteins_val, batch_converter, token_cache,
        boundary_radius=cfg.boundary_radius, cfg=cfg, plddt_by_id=plddt_by_id,
    )

    loader_kw = dict(
        batch_size=cfg.batch_size,
        collate_fn=disprot_collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    train_gen = torch.Generator()
    train_gen.manual_seed(cfg.seed + fold_idx * 1000)
    train_dl = DataLoader(
        train_ds, shuffle=True, drop_last=False, generator=train_gen, **loader_kw,
    )
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    lora_params = list(fold_model.get_lora_params())
    head_params = list(fold_model.get_head_params())
    esm_tail_params = list(fold_model.get_esm_tail_params())
    param_groups = [
        {"params": lora_params, "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay},
        {"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
    ]
    if esm_tail_params:
        param_groups.append({
            "params": esm_tail_params,
            "lr": cfg.lr_esm_tail,
            "weight_decay": cfg.weight_decay * 0.1,
        })
    optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = math.ceil(len(train_dl) / cfg.accum_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_frac)
    scheduler = _warmup_cosine_scheduler(optimizer, total_steps, warmup_steps)

    pos_weight = _compute_pos_weight(proteins_train, device)
    scaler = GradScaler(device="cuda", enabled=(amp_dtype == torch.float16))

    best_score = -1.0
    best_auc = -1.0
    best_ap = 0.0
    best_state: Optional[dict] = None
    best_probs = best_labels = None
    best_epoch = 0
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"fold{fold_idx + 1}_best.pt")
    patience_counter = 0
    history: list = []
    global_step = 0
    fold_t0 = time.time()

    ema = None
    if cfg.use_ema:
        from colab.model_ema import ModelEMA
        ema = ModelEMA(fold_model, decay=cfg.ema_decay)

    swa = None
    swa_start_epoch = int(cfg.num_epochs * cfg.swa_start_frac) if cfg.use_swa else cfg.num_epochs + 1
    if cfg.use_swa:
        from colab.model_swa import ModelSWA
        swa = ModelSWA(fold_model)

    proteins_by_id = {p["id"]: p for p in proteins_train + proteins_val}
    use_distill = cfg.use_v6_distill and bool(v6_teacher_by_id)

    for epoch in range(cfg.num_epochs):
        fold_model.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.time()

        pbar = tqdm(
            train_dl,
            desc=f"Fold {fold_idx + 1} Ep {epoch + 1}/{cfg.num_epochs}",
            leave=False,
        )

        for step, (tokens, labels, mask, aa_idx, sample_weight, rich_feats, batch_ids) in enumerate(pbar):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            aa_idx = aa_idx.to(device, non_blocking=True)
            sample_weight = sample_weight.to(device, non_blocking=True)
            rich_t = None
            if rich_feats is not None:
                rich_t = rich_feats.to(device, non_blocking=True)
            max_seq = mask.shape[1]
            teacher_probs = None
            if use_distill and v6_teacher_by_id is not None:
                teacher_probs = _pad_teacher_probs(
                    list(batch_ids), proteins_by_id, v6_teacher_by_id, max_seq, device,
                )

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                aa = aa_idx if fold_model.use_physico else None
                if cfg.use_rdrop:
                    logits_a = _forward_logits(fold_model, tokens, aa, mask, rich_feats=rich_t)
                    logits_b = _forward_logits(fold_model, tokens, aa, mask, rich_feats=rich_t)
                    loss_a = _disorder_loss(
                        logits_a, labels, pos_weight, sample_weight, cfg, mask=mask,
                    )
                    loss_b = _disorder_loss(
                        logits_b, labels, pos_weight, sample_weight, cfg, mask=mask,
                    )
                    from colab.sota_losses import rdrop_symmetric_kl
                    cons = rdrop_symmetric_kl(logits_a, logits_b, mask)
                    loss = 0.5 * (loss_a + loss_b) + cfg.rdrop_weight * cons
                    logits = logits_a
                else:
                    logits = _forward_logits(fold_model, tokens, aa, mask, rich_feats=rich_t)
                    loss = _disorder_loss(
                        logits, labels, pos_weight, sample_weight, cfg, mask=mask,
                    )

                if use_distill and teacher_probs is not None:
                    from colab.sota_losses import v6_distillation_loss
                    d_loss = v6_distillation_loss(
                        logits, teacher_probs, mask, temperature=cfg.v6_distill_temperature,
                    )
                    loss = loss + cfg.v6_distill_weight * d_loss

                loss = loss / cfg.accum_steps

            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(train_dl):
                if amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in fold_model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        [p for p in fold_model.parameters() if p.requires_grad],
                        cfg.max_grad_norm,
                    )
                    optimizer.step()

                if ema is not None:
                    ema.update(fold_model)

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            train_loss += loss.item() * cfg.accum_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * cfg.accum_steps:.4f}")

        avg_train_loss = train_loss / max(n_batches, 1)
        if swa is not None and epoch >= swa_start_epoch:
            swa.update(fold_model)

        swa_backup = None
        if ema is not None:
            ema.apply_shadow(fold_model)
        elif swa is not None and swa.ready and epoch >= swa_start_epoch:
            swa_backup = swa.apply_swa(fold_model)

        val_metrics = eval_epoch(fold_model, val_dl, device, amp_dtype, pos_weight, cfg)

        if ema is not None:
            ema.restore(fold_model)
        elif swa_backup is not None:
            swa.restore(fold_model, swa_backup)

        from colab.segment_postprocess import composite_early_stop_score, pooled_segment_f1

        seg_f1 = 0.0
        if cfg.use_segment_early_stop:
            seg_f1 = pooled_segment_f1(
                proteins_val,
                val_metrics["probs"],
                val_metrics["labels"],
                min_region_len=cfg.segment_min_region_len,
                postprocess_min_len=cfg.segment_postprocess_min_len,
                postprocess_max_gap=cfg.segment_postprocess_max_gap,
            )
        val_metrics["segment_f1"] = seg_f1

        epoch_time = time.time() - epoch_t0
        elapsed = time.time() - fold_t0
        eta_h = (elapsed / (epoch + 1)) * (cfg.num_epochs - epoch - 1) / 3600

        row = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_ap": val_metrics["ap"],
            "val_f1": val_metrics["f1"],
            "val_mcc": val_metrics["mcc"],
            "val_segment_f1": seg_f1,
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        if len(optimizer.param_groups) > 2:
            row["lr_esm_tail"] = optimizer.param_groups[2]["lr"]
        history.append(row)

        marker = ""
        if cfg.early_stop_mode == "auc":
            composite = val_metrics["auc"]
        elif cfg.use_segment_early_stop:
            composite = composite_early_stop_score(
                val_metrics["auc"],
                val_metrics["ap"],
                seg_f1,
                auc_weight=cfg.auc_score_weight,
                ap_weight=cfg.ap_score_weight,
                segment_weight=cfg.segment_score_weight,
            )
        else:
            composite = 0.7 * val_metrics["auc"] + 0.3 * val_metrics["ap"]
        if composite > best_score:
            best_score = composite
            best_auc = val_metrics["auc"]
            best_ap = val_metrics["ap"]
            best_epoch = epoch + 1
            if ema is not None:
                ema.apply_shadow(fold_model)
                best_state = copy.deepcopy(fold_model.state_dict())
                ema.restore(fold_model)
            elif swa is not None and swa.ready and epoch >= swa_start_epoch:
                swa_backup = swa.apply_swa(fold_model)
                best_state = copy.deepcopy(fold_model.state_dict())
                swa.restore(fold_model, swa_backup)
            else:
                best_state = copy.deepcopy(fold_model.state_dict())
            best_probs = val_metrics["probs"]
            best_labels = val_metrics["labels"]
            patience_counter = 0
            meta = {
                "fold": fold_idx + 1,
                "best_auc": best_auc,
                "best_ap": best_ap,
                "best_epoch": best_epoch,
                "profile": getattr(cfg, "_profile_name", "custom"),
            }
            if cfg.compact_checkpoints:
                from colab.compact_checkpoint import save_compact_checkpoint
                if ema is not None:
                    ema.apply_shadow(fold_model)
                save_compact_checkpoint(ckpt_path, fold_model, metadata=meta)
                if ema is not None:
                    ema.restore(fold_model)
            else:
                torch.save(best_state, ckpt_path)
            marker = " ◀ BEST"
        else:
            patience_counter += 1

        msg = (
            f"  Ep {epoch + 1:2d}/{cfg.num_epochs}  "
            f"loss={avg_train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"AUC={val_metrics['auc']:.4f}  AP={val_metrics['ap']:.4f}  "
            f"F1={val_metrics['f1']:.4f}  SegF1={seg_f1:.4f}  "
            f"[{epoch_time:.0f}s, ETA {eta_h:.1f}h]{marker}"
        )
        print(msg)
        if on_epoch_end:
            on_epoch_end(row)

        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={cfg.patience})")
            break

    total_time = time.time() - fold_t0
    print(
        f"\n  Fold {fold_idx + 1} done in {total_time / 3600:.2f}h  "
        f"│  Best AUC={best_auc:.4f}  AP={best_ap:.4f}"
    )

    if best_state is not None:
        fold_model.load_state_dict(best_state)

    del train_dl, val_dl, fold_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx + 1,
        "best_auc": best_auc,
        "best_ap": best_ap,
        "best_epoch": best_epoch,
        "early_stop_mode": cfg.early_stop_mode,
        "history": history,
        "val_probs": best_probs,
        "val_labels": best_labels,
        "ckpt_path": ckpt_path if best_state is not None else None,
        "total_time": total_time,
    }


def _serialize_fold_result(result: dict) -> dict:
    """JSON-safe fold result (numpy arrays → lists)."""
    out = dict(result)
    if out.get("val_probs") is not None:
        out["val_probs"] = np.asarray(out["val_probs"]).tolist()
    if out.get("val_labels") is not None:
        out["val_labels"] = np.asarray(out["val_labels"]).tolist()
    return out


def _deserialize_fold_result(data: dict) -> dict:
    """Restore fold result from cv_progress.json."""
    out = dict(data)
    if out.get("val_probs") is not None:
        out["val_probs"] = np.asarray(out["val_probs"], dtype=np.float32)
    if out.get("val_labels") is not None:
        out["val_labels"] = np.asarray(out["val_labels"], dtype=np.float32)
    return out


def save_cv_progress(
    path: str,
    fold_results: list,
    cfg: TrainConfig,
    proteins: list,
    disprot_meta: Optional[dict] = None,
) -> None:
    """Persist completed folds so Colab can resume after disconnect."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": 2,
        "n_proteins": len(proteins),
        "protein_ids": [p["id"] for p in proteins],
        "proteins_fingerprint": proteins_fingerprint(proteins),
        "fold_val_ids": get_fold_val_protein_ids(proteins, cfg.n_folds),
        "config_fingerprint": config_fingerprint(cfg),
        "n_folds": cfg.n_folds,
        "seed": cfg.seed,
        "disprot_meta": disprot_meta,
        "fold_results": [_serialize_fold_result(r) for r in fold_results],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


def load_cv_progress(
    path: str,
    cfg: TrainConfig,
    proteins: list,
    disprot_meta: Optional[dict] = None,
) -> list:
    """Load completed folds if progress file matches current run."""
    if not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    version = payload.get("version", 1)
    if version >= 2:
        if payload.get("protein_ids") != [p["id"] for p in proteins]:
            print("  ⚠ cv_progress.json protein ID list mismatch — ignoring saved folds")
            return []
        if payload.get("proteins_fingerprint") != proteins_fingerprint(proteins):
            print("  ⚠ cv_progress.json protein fingerprint mismatch — ignoring saved folds")
            return []
        if payload.get("config_fingerprint") != config_fingerprint(cfg):
            print("  ⚠ cv_progress.json config fingerprint mismatch — ignoring saved folds")
            return []
        saved_meta = payload.get("disprot_meta")
        if disprot_meta and saved_meta and saved_meta.get("content_sha256") != disprot_meta.get(
            "content_sha256"
        ):
            print("  ⚠ cv_progress.json DisProt snapshot mismatch — ignoring saved folds")
            return []
    elif (
        payload.get("n_proteins") != len(proteins)
        or payload.get("n_folds") != cfg.n_folds
        or payload.get("seed") != cfg.seed
    ):
        print(
            "  ⚠ cv_progress.json mismatch (proteins/folds/seed changed) — ignoring saved folds"
        )
        return []

    if payload.get("n_folds") != cfg.n_folds or payload.get("seed") != cfg.seed:
        print("  ⚠ cv_progress.json folds/seed mismatch — ignoring saved folds")
        return []

    return [_deserialize_fold_result(r) for r in payload.get("fold_results", [])]


def infer_resume_fold(
    path: str,
    cfg: TrainConfig,
    proteins: list,
    disprot_meta: Optional[dict] = None,
) -> int:
    """Return next fold index (0-based) from saved CV progress."""
    return len(load_cv_progress(path, cfg, proteins, disprot_meta))


def run_cross_validation(
    proteins: list,
    esm_backbone: nn.Module,
    batch_converter,
    cfg: TrainConfig,
    resume_from_fold: int = 0,
    plddt_by_id: Optional[dict[str, np.ndarray]] = None,
    prefetch_af_plddt: bool = False,
    on_fold_complete: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
) -> tuple[list, dict]:
    """
    Run N-fold GroupKFold CV. Returns (fold_results, summary_dict).

    resume_from_fold: skip folds < this index (0-based) for Colab reconnect.
    Completed folds are loaded from checkpoints/cv_progress.json when resuming.
    prefetch_af_plddt: fetch AF pLDDT cache before training (hallucination weights).
    """
    if plddt_by_id is None and cfg.use_hallucination_weighting and prefetch_af_plddt:
        print("Pre-fetching AF pLDDT for hallucination hard-negative weighting...")
        plddt_by_id = build_plddt_cache_for_training(proteins, cache_dir=cfg.af_plddt_cache_dir)
        n_cached = len(plddt_by_id)
        print(f"  pLDDT available for {n_cached}/{len(proteins)} proteins")

    token_cache: dict = {}
    splits = get_cv_splits(proteins, cfg.n_folds)

    progress_path = os.path.join(cfg.checkpoint_dir, "cv_progress.json")
    disprot_meta = get_disprot_cache_meta(cfg.data_cache)
    fold_results: list = []
    if resume_from_fold > 0:
        loaded = load_cv_progress(progress_path, cfg, proteins, disprot_meta)
        if len(loaded) >= resume_from_fold:
            fold_results = loaded[:resume_from_fold]
            print(
                f"  Resumed {len(fold_results)} fold(s) from {progress_path} "
                f"(AUCs: {[round(r['best_auc'], 4) for r in fold_results]})"
            )
        elif loaded:
            print(
                f"  ⚠ cv_progress has {len(loaded)} fold(s) but resume_from_fold="
                f"{resume_from_fold} — using saved folds and continuing"
            )
            fold_results = loaded
            resume_from_fold = len(loaded)
        else:
            print(
                f"  ⚠ No cv_progress.json at {progress_path}; "
                f"folds 1–{resume_from_fold} will be missing from pooled metrics"
            )

    cv_t0 = time.time()

    print(f"\nStarting {cfg.n_folds}-fold cross-validation on {len(proteins)} proteins")
    print(f"  Effective batch size: {cfg.effective_batch()}")
    print_training_config_summary(cfg, proteins)
    if plddt_by_id:
        print(f"  AF pLDDT cache     : {len(plddt_by_id)} proteins (hallucination weighting)")
    elif cfg.use_hallucination_weighting:
        print("  AF pLDDT cache     : none (hallucination weights = boundary only)")

    v6_teacher_by_id: Optional[dict[str, np.ndarray]] = None
    if cfg.use_v6_distill:
        print("Pre-computing v6 OOF teacher probabilities for distillation...")
        profile = getattr(cfg, "_profile_name", "custom")
        if profile == "ultra":
            from colab.v6_pro_ensemble import get_v6_pro_oof_probs
            v6_teacher_by_id = get_v6_pro_oof_probs(
                proteins, n_folds=cfg.n_folds, seed=cfg.seed,
                cache_path="v6_pro_oof_probs_cache.json",
            )
        else:
            from colab.ensemble_v6 import aligned_probs_from_oof, run_v6_lite_oof
            oof_probs, _, _ = run_v6_lite_oof(proteins, n_folds=cfg.n_folds, seed=cfg.seed)
            v6_teacher_by_id = aligned_probs_from_oof(proteins, oof_probs)
        print(f"  v6 teacher probs for {len(v6_teacher_by_id)} proteins")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        if fold_idx < resume_from_fold:
            print(f"  Skipping fold {fold_idx + 1} (resume_from_fold={resume_from_fold})")
            continue

        proteins_train = [proteins[i] for i in train_idx]
        proteins_val = [proteins[i] for i in val_idx]

        result = train_fold(
            fold_idx=fold_idx,
            proteins_train=proteins_train,
            proteins_val=proteins_val,
            esm_backbone=esm_backbone,
            batch_converter=batch_converter,
            token_cache=token_cache,
            cfg=cfg,
            plddt_by_id=plddt_by_id,
            v6_teacher_by_id=v6_teacher_by_id,
            on_epoch_end=on_epoch_end,
        )
        fold_results.append(result)
        save_cv_progress(progress_path, fold_results, cfg, proteins, disprot_meta)

        if on_fold_complete:
            on_fold_complete(result)

        aucs = [r["best_auc"] for r in fold_results]
        print(
            f"\n  ─── Running mean AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  "
            f"[{(time.time() - cv_t0) / 3600:.1f}h elapsed] ───"
        )

    all_probs = np.concatenate([r["val_probs"] for r in fold_results])
    all_labels = np.concatenate([r["val_labels"] for r in fold_results])

    pooled_auc = roc_auc_score(all_labels, all_probs)
    pooled_ap = average_precision_score(all_labels, all_probs)
    fold_aucs = [r["best_auc"] for r in fold_results]
    fold_aps = [r["best_ap"] for r in fold_results]
    total_cv_h = (time.time() - cv_t0) / 3600

    summary = {
        "pooled_auc": float(pooled_auc),
        "pooled_ap": float(pooled_ap),
        "fold_aucs": [float(a) for a in fold_aucs],
        "fold_aps": [float(a) for a in fold_aps],
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "mean_ap": float(np.mean(fold_aps)),
        "total_cv_hours": float(total_cv_h),
        "config": {
            k: (str(v) if isinstance(v, (torch.device, torch.dtype)) else v)
            for k, v in asdict(cfg).items()
        },
        "proteins_fingerprint": proteins_fingerprint(proteins),
        "config_fingerprint": config_fingerprint(cfg),
        "disprot_meta": disprot_meta,
    }

    _print_cv_summary(fold_results, summary)
    return fold_results, summary


def _print_cv_summary(fold_results: list, summary: dict) -> None:
    print(f"\n{'═' * 60}")
    print(" CROSS-VALIDATION COMPLETE")
    print(f"{'═' * 60}")
    print(f"{'Fold':>5} {'AUC':>8} {'AP':>8} {'Time(h)':>9}")
    print(f"{'─' * 35}")
    for r in fold_results:
        print(
            f"{r['fold']:>5} {r['best_auc']:>8.4f} {r['best_ap']:>8.4f} "
            f"{r['total_time'] / 3600:>9.2f}"
        )
    print(f"{'─' * 35}")
    print(
        f"{'Mean':>5} {summary['mean_auc']:>8.4f} {summary['mean_ap']:>8.4f}"
    )
    print(f"{'Std':>5} {summary['std_auc']:>8.4f}")
    print(f"{'─' * 35}")
    print(f"{'Pooled':>5} {summary['pooled_auc']:>8.4f} {summary['pooled_ap']:>8.4f}")
    print(f"{'═' * 60}")
    print(f"  Total CV time: {summary['total_cv_hours']:.2f}h")
