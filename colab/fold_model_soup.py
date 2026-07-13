"""
5-fold model soup: average predictions from fold checkpoints on OOF proteins.

Modes:
  held_out   — each fold checkpoint predicts only its validation proteins (rigorous OOF)
  full_soup  — all fold models predict every validation protein (deployment-style;
               inflates CV metrics — flagged in report)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from colab.compact_checkpoint import load_compact_checkpoint
from colab.cv_splits import get_cv_splits
from colab.disordernet_gpu import (
    DisorderNetGPU,
    DisProtDataset,
    TrainConfig,
    _forward_logits,
    disprot_collate,
)
from colab.inference_fusion import compute_pooled_metrics, write_fused_probs_to_fold_results
from colab.biological_utility import align_fold_predictions


@torch.inference_mode()
def _predict_proteins(
    model: DisorderNetGPU,
    proteins: list,
    batch_converter,
    token_cache: dict,
    cfg: TrainConfig,
    plddt_by_id: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Run inference; return protein_id -> per-residue probabilities."""
    if not proteins:
        return {}
    ds = DisProtDataset(
        proteins, batch_converter, token_cache,
        boundary_radius=cfg.boundary_radius, cfg=cfg, plddt_by_id=plddt_by_id,
    )
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, collate_fn=disprot_collate,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
    )
    device = cfg.device
    amp_dtype = cfg.amp_dtype
    out: dict[str, np.ndarray] = {}

    for tokens, labels, mask, aa_idx, _, ids in dl:
        tokens = tokens.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        aa_idx = aa_idx.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda",
        ):
            logits = _forward_logits(
                model, tokens, aa_idx if model.use_physico else None, mask,
            )
        probs = torch.sigmoid(logits).float().cpu().numpy()
        mask_np = mask.cpu().numpy()
        for i, pid in enumerate(ids):
            m = mask_np[i]
            out[pid] = probs[i][m].astype(np.float32)
    return out


def _load_fold_model(
    ckpt_path: str,
    esm_backbone: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
) -> DisorderNetGPU:
    fold_model = DisorderNetGPU(esm_backbone, cfg, verbose=False).to(device)
    try:
        load_compact_checkpoint(ckpt_path, fold_model, device=device)
    except Exception:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(payload, dict) and "trainable" in payload:
            fold_model.load_state_dict(payload["trainable"], strict=False)
        else:
            fold_model.load_state_dict(payload, strict=False)
    fold_model.eval()
    return fold_model


def run_fold_model_soup(
    proteins: list,
    esm_backbone: torch.nn.Module,
    batch_converter,
    cfg: TrainConfig,
    fold_results: list,
    plddt_by_id: Optional[dict] = None,
    checkpoint_dir: Optional[str] = None,
    mode: str = "held_out",
) -> tuple[dict, list]:
    """
    Ensemble fold checkpoints into updated OOF predictions.

    Returns (report, updated_fold_results).
    """
    if mode not in ("held_out", "full_soup"):
        raise ValueError(f"Unknown soup mode '{mode}'. Use held_out or full_soup.")

    ckpt_dir = checkpoint_dir or cfg.checkpoint_dir
    n_folds = cfg.n_folds
    splits = get_cv_splits(proteins, n_folds)
    device = cfg.device

    prob_sum: dict[str, np.ndarray] = {
        p["id"]: np.zeros(p["length"], dtype=np.float64) for p in proteins
    }
    prob_count: dict[str, int] = {p["id"]: 0 for p in proteins}

    token_cache: dict = {}
    models_loaded = 0

    for model_idx in range(n_folds):
        ckpt_path = os.path.join(ckpt_dir, f"fold{model_idx + 1}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"  ⚠ Missing {ckpt_path} — soup uses {models_loaded} models")
            break

        fold_model = _load_fold_model(ckpt_path, esm_backbone, cfg, device)
        models_loaded += 1

        if mode == "held_out":
            _, val_idx = splits[model_idx]
            val_proteins = [proteins[i] for i in val_idx]
            preds = _predict_proteins(
                fold_model, val_proteins, batch_converter, token_cache, cfg, plddt_by_id,
            )
            for pid, pr in preds.items():
                prob_sum[pid] += pr
                prob_count[pid] += 1
        else:
            for _, val_idx in tqdm(splits, desc=f"Soup model {model_idx + 1}", leave=False):
                val_proteins = [proteins[i] for i in val_idx]
                preds = _predict_proteins(
                    fold_model, val_proteins, batch_converter, token_cache, cfg, plddt_by_id,
                )
                for pid, pr in preds.items():
                    prob_sum[pid] += pr
                    prob_count[pid] += 1

        del fold_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if models_loaded == 0:
        return {"skipped": True, "reason": "no checkpoints"}, fold_results

    prob_avg = {
        pid: (prob_sum[pid] / max(prob_count[pid], 1)).astype(np.float32)
        for pid in prob_sum
        if prob_count[pid] > 0
    }

    aligned = align_fold_predictions(proteins, fold_results, n_folds=n_folds)
    aligned_soup = []
    for item in aligned:
        pid = item["id"]
        if pid in prob_avg:
            new_item = dict(item)
            new_item["probs"] = prob_avg[pid]
            new_item["model_soup"] = True
            aligned_soup.append(new_item)
        else:
            aligned_soup.append(item)

    before = compute_pooled_metrics(fold_results)
    fold_results_soup = write_fused_probs_to_fold_results(
        proteins, fold_results, aligned_soup, n_folds=n_folds,
    )
    for fr in fold_results_soup:
        fr["model_soup"] = True
        fr["soup_n_models"] = models_loaded
        fr["soup_mode"] = mode

    after = compute_pooled_metrics(fold_results_soup)
    report = {
        "n_models_averaged": models_loaded,
        "mode": mode,
        "cv_leakage_warning": mode == "full_soup",
        "before": {"pooled": {k: before[k] for k in ("auc", "ap", "n_residues")}},
        "after": {"pooled": {k: after[k] for k in ("auc", "ap", "n_residues")}},
        "delta_auc_pooled": after["auc"] - before["auc"],
        "delta_ap_pooled": after["ap"] - before["ap"],
        "method": "fold_checkpoint_ensemble",
    }
    return report, fold_results_soup


def print_fold_soup_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" FOLD CHECKPOINT ENSEMBLE")
    print(f"{'═' * 64}")
    if report.get("skipped"):
        print(f"  Skipped: {report.get('reason')}")
        return
    print(f"  Mode            : {report.get('mode', 'held_out')}")
    if report.get("cv_leakage_warning"):
        print("  ⚠ full_soup may inflate CV AUC (models trained on some val proteins)")
    print(f"  Models loaded   : {report['n_models_averaged']}")
    b, a = report["before"]["pooled"], report["after"]["pooled"]
    print(f"  Before : AUC={b['auc']:.4f}  AP={b['ap']:.4f}")
    print(f"  After  : AUC={a['auc']:.4f}  AP={a['ap']:.4f}")
    print(f"  Δ AUC  : {report['delta_auc_pooled']:+.4f}")
    print(f"{'═' * 64}")
