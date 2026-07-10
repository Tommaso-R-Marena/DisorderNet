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
from sklearn.model_selection import GroupKFold
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
    num_epochs: int = 15
    lr_lora: float = 1e-4
    lr_head: float = 5e-4
    weight_decay: float = 1e-2
    patience: int = 4
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.1

    lora_layers: int = 8
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    head_dropout: float = 0.10

    n_folds: int = 5
    num_workers: int = 2
    checkpoint_dir: str = "checkpoints"
    data_cache: str = "disprot_raw.json"

    # Performance toggles
    use_gradient_checkpointing: bool = True
    deterministic: bool = False  # False → cudnn.benchmark + TF32 for speed

    # Set automatically by setup_environment()
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    amp_dtype: torch.dtype = torch.float16
    pin_memory: bool = False
    gpu_name: str = "cpu"
    vram_gb: float = 0.0

    def effective_batch(self) -> int:
        return self.batch_size * self.accum_steps


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
    print(f"{'=' * 55}")
    return cfg


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_disprot(cache_path: str = "disprot_raw.json") -> list:
    """Fetch DisProt via REST API with local JSON cache."""
    if os.path.exists(cache_path):
        print(f"Loading cached DisProt from '{cache_path}'...")
        with open(cache_path) as f:
            return json.load(f)

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

    with open(cache_path, "w") as f:
        json.dump(all_entries, f)
    return all_entries


def _region_is_disorder(term_name: Optional[str]) -> bool:
    if not term_name:
        return False
    t = term_name.strip().lower()
    if t in ORDER_TERMS:
        return False
    return t in DISORDER_TERMS


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
        has_disorder_term = False

        for region in entry.get("regions", []):
            term = region.get("term_name") or region.get("type") or ""
            if not _region_is_disorder(term):
                continue
            has_disorder_term = True
            start = region.get("start", 0)
            end = region.get("end", 0)
            if start and end:
                for i in range(start - 1, min(end, len(seq))):
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
            "sequence": seq,
            "labels": labels,
            "length": len(seq),
            "n_dis": n_dis,
            "frac_dis": n_dis / len(seq),
        })

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


class DisorderCNNHead(nn.Module):
    """3-layer 1D CNN head with residual skip."""

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
        return (self.net(x) + self.skip(x)).squeeze(1)


class DisorderNetGPU(nn.Module):
    """ESM-2 650M + LoRA (Q/V, last N layers) + CNN disorder head."""

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
            # fair-esm exposes this on the ESM2 model class
            if hasattr(self.esm, "set_gradient_checkpointing"):
                self.esm.set_gradient_checkpointing(True)

        n_layers = len(self.esm.layers)
        start = n_layers - cfg.lora_layers
        self._lora_modules: list[LoRALinear] = []

        for layer_idx in range(start, n_layers):
            attn = self.esm.layers[layer_idx].self_attn
            for proj_name in ("q_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if isinstance(proj, nn.Linear):
                    lora = LoRALinear(
                        proj, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout
                    )
                    setattr(attn, proj_name, lora)
                    self._lora_modules.append(lora)

        self.head = DisorderCNNHead(in_dim=1280, dropout=cfg.head_dropout)

        if verbose:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(
                f"  LoRA layers {start}–{n_layers - 1} "
                f"({len(self._lora_modules)} projections)"
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
        return self.head.parameters()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out = self.esm(tokens, repr_layers=[33], return_contacts=False)
        embeddings = out["representations"][33][:, 1:-1, :]
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

    def __init__(self, proteins: list, batch_converter, cache: Optional[dict] = None):
        self.proteins = proteins
        self._cache = cache if cache is not None else {}

        missing = [p for p in proteins if p["id"] not in self._cache]
        if missing:
            for p in tqdm(missing, desc="Tokenizing", leave=False):
                _, _, tokens = batch_converter([(p["id"], p["sequence"])])
                self._cache[p["id"]] = {
                    "tokens": tokens.squeeze(0),
                    "labels": torch.tensor(p["labels"], dtype=torch.float32),
                }

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int):
        p = self.proteins[idx]
        item = self._cache[p["id"]]
        labels = item["labels"]
        mask = torch.ones(labels.shape[0], dtype=torch.bool)
        return item["tokens"], labels, mask, p["id"]


def disprot_collate(batch):
    tokens_list, labels_list, mask_list, ids = zip(*batch)
    max_tok = max(t.shape[0] for t in tokens_list)
    max_seq = max_tok - 2
    pad_idx = 1

    tokens_padded = torch.full((len(batch), max_tok), pad_idx, dtype=torch.long)
    labels_padded = torch.zeros(len(batch), max_seq, dtype=torch.float32)
    mask_padded = torch.zeros(len(batch), max_seq, dtype=torch.bool)

    for i, (tok, lab, msk) in enumerate(zip(tokens_list, labels_list, mask_list)):
        lt, ls = tok.shape[0], lab.shape[0]
        tokens_padded[i, :lt] = tok
        labels_padded[i, :ls] = lab
        mask_padded[i, :ls] = msk

    return tokens_padded, labels_padded, mask_padded, list(ids)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _compute_pos_weight(proteins: list, device: torch.device) -> torch.Tensor:
    total_pos = sum(p["n_dis"] for p in proteins)
    total_neg = sum(p["length"] - p["n_dis"] for p in proteins)
    pw = min(total_neg / max(total_pos, 1), 10.0)
    return torch.tensor([pw], device=device)


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
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    for tokens, labels, mask, _ in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(tokens)

        loss = criterion(logits[mask], labels[mask]).mean()
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


def train_fold(
    fold_idx: int,
    proteins_train: list,
    proteins_val: list,
    esm_backbone: nn.Module,
    batch_converter,
    token_cache: dict,
    cfg: TrainConfig,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Train one CV fold. Returns metrics dict including best checkpoint path."""
    device, amp_dtype = cfg.device, cfg.amp_dtype
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"\n{'═' * 60}")
    print(f" FOLD {fold_idx + 1}  |  train={len(proteins_train)}  val={len(proteins_val)}")
    print(f"{'═' * 60}")

    fold_model = DisorderNetGPU(esm_backbone, cfg, verbose=True).to(device)

    train_ds = DisProtDataset(proteins_train, batch_converter, token_cache)
    val_ds = DisProtDataset(proteins_val, batch_converter, token_cache)

    loader_kw = dict(
        batch_size=cfg.batch_size,
        collate_fn=disprot_collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    lora_params = list(fold_model.get_lora_params())
    head_params = list(fold_model.get_head_params())
    optimizer = AdamW(
        [
            {"params": lora_params, "lr": cfg.lr_lora, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = math.ceil(len(train_dl) / cfg.accum_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_frac)
    scheduler = _warmup_cosine_scheduler(optimizer, total_steps, warmup_steps)

    pos_weight = _compute_pos_weight(proteins_train, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    scaler = GradScaler(device="cuda", enabled=(amp_dtype == torch.float16))

    best_auc = -1.0
    best_ap = 0.0
    best_state: Optional[dict] = None
    best_probs = best_labels = None
    ckpt_path = os.path.join(cfg.checkpoint_dir, f"fold{fold_idx + 1}_best.pt")
    patience_counter = 0
    history: list = []
    global_step = 0
    fold_t0 = time.time()

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

        for step, (tokens, labels, mask, _) in enumerate(pbar):
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = fold_model(tokens)
                loss = criterion(logits[mask], labels[mask]).mean() / cfg.accum_steps

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

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            train_loss += loss.item() * cfg.accum_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * cfg.accum_steps:.4f}")

        avg_train_loss = train_loss / max(n_batches, 1)
        val_metrics = eval_epoch(fold_model, val_dl, device, amp_dtype, pos_weight)
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
            "lr_lora": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
        }
        history.append(row)

        marker = ""
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_ap = val_metrics["ap"]
            best_state = copy.deepcopy(fold_model.state_dict())
            best_probs = val_metrics["probs"]
            best_labels = val_metrics["labels"]
            patience_counter = 0
            torch.save(best_state, ckpt_path)
            marker = " ◀ BEST"
        else:
            patience_counter += 1

        msg = (
            f"  Ep {epoch + 1:2d}/{cfg.num_epochs}  "
            f"loss={avg_train_loss:.4f}  val_loss={val_metrics['loss']:.4f}  "
            f"AUC={val_metrics['auc']:.4f}  AP={val_metrics['ap']:.4f}  "
            f"F1={val_metrics['f1']:.4f}  "
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
        "history": history,
        "val_probs": best_probs,
        "val_labels": best_labels,
        "ckpt_path": ckpt_path if best_state is not None else None,
        "total_time": total_time,
    }


def run_cross_validation(
    proteins: list,
    esm_backbone: nn.Module,
    batch_converter,
    cfg: TrainConfig,
    resume_from_fold: int = 0,
    on_fold_complete: Optional[Callable[[dict], None]] = None,
    on_epoch_end: Optional[Callable[[dict], None]] = None,
) -> tuple[list, dict]:
    """
    Run N-fold GroupKFold CV. Returns (fold_results, summary_dict).

    resume_from_fold: skip folds < this index (0-based) for Colab reconnect.
    """
    token_cache: dict = {}
    gkf = GroupKFold(n_splits=cfg.n_folds)
    groups = np.arange(len(proteins))

    fold_results: list = []
    cv_t0 = time.time()

    print(f"\nStarting {cfg.n_folds}-fold cross-validation on {len(proteins)} proteins")
    print(f"  Effective batch size: {cfg.effective_batch()}")

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(groups, groups=groups)):
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
            on_epoch_end=on_epoch_end,
        )
        fold_results.append(result)

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
