"""
Batch FASTA inference — score arbitrary proteomes with trained DisorderNet checkpoints.

Novel use case: proteome-scale disorder screening on Rockfish (not limited to DisProt CV).
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from colab.caid3_eval import write_caid_prediction_file
from colab.disordernet_gpu import (
    AA_TO_IDX,
    DisorderNetGPU,
    TrainConfig,
    _forward_logits,
    disprot_collate,
)
from colab.fold_model_soup import _load_fold_model, _predict_proteins


def parse_fasta(path: str) -> list[dict]:
    """Parse FASTA into protein dicts compatible with DisProtDataset."""
    proteins: list[dict] = []
    with open(path) as f:
        header = None
        seq_parts: list[str] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_parts).upper()
                    proteins.append({
                        "id": header.split()[0],
                        "sequence": seq,
                        "length": len(seq),
                        "labels": [0] * len(seq),
                        "n_dis": 0,
                    })
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seq = "".join(seq_parts).upper()
            proteins.append({
                "id": header.split()[0],
                "sequence": seq,
                "length": len(seq),
                "labels": [0] * len(seq),
                "n_dis": 0,
            })
    return proteins


class _InferenceDataset(torch.utils.data.Dataset):
    """Minimal dataset for inference (no labels required)."""

    def __init__(self, proteins: list, batch_converter, token_cache: dict, cfg: TrainConfig):
        self.proteins = proteins
        self._cache = token_cache
        self.cfg = cfg
        for p in proteins:
            if p["id"] not in self._cache:
                _, _, tokens = batch_converter([(p["id"], p["sequence"])])
                entry = {
                    "tokens": tokens.squeeze(0),
                    "labels": torch.zeros(len(p["sequence"]), dtype=torch.float32),
                    "aa_idx": torch.tensor(
                        [AA_TO_IDX.get(c, 0) for c in p["sequence"]],
                        dtype=torch.long,
                    ),
                    "sample_weight": torch.ones(len(p["sequence"]), dtype=torch.float32),
                }
                if cfg.use_rich_features:
                    from colab.rich_features import compute_rich_features
                    rich = compute_rich_features(p["sequence"])
                    entry["rich_feats"] = torch.tensor(rich, dtype=torch.float32)
                if cfg.use_plddt_features:
                    from colab.structure_encoder import build_plddt_feature_tensor
                    entry["plddt_feats"] = build_plddt_feature_tensor(None, len(p["sequence"]))
                self._cache[p["id"]] = entry

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int):
        p = self.proteins[idx]
        item = self._cache[p["id"]]
        labels = item["labels"]
        mask = torch.ones(labels.shape[0], dtype=torch.bool)
        return (
            item["tokens"], labels, mask, item["aa_idx"],
            item["sample_weight"], item.get("rich_feats"), item.get("plddt_feats"), p["id"],
        )


def load_fold_ensemble_models(
    checkpoint_dir: str,
    esm_backbone: torch.nn.Module,
    cfg: TrainConfig,
    device: torch.device,
    n_folds: int = 5,
) -> list[DisorderNetGPU]:
    """Load all compact fold checkpoints from a CV run."""
    models: list[DisorderNetGPU] = []
    for fold in range(1, n_folds + 1):
        for name in (f"fold_{fold}_compact.pt", f"fold_{fold}_best.pt"):
            path = os.path.join(checkpoint_dir, name)
            if os.path.isfile(path):
                models.append(_load_fold_model(path, esm_backbone, cfg, device))
                break
    if not models:
        raise FileNotFoundError(f"No fold checkpoints in {checkpoint_dir}")
    return models


def predict_fasta_batch(
    fasta_path: str,
    esm_backbone: torch.nn.Module,
    batch_converter,
    cfg: TrainConfig,
    checkpoint_dir: str,
    plddt_by_id: Optional[dict[str, np.ndarray]] = None,
    use_tta: bool = True,
    tta_passes: int = 6,
    n_folds: int = 5,
    return_function: bool = False,
):
    """
    Score a FASTA file with fold-model soup (average of all fold checkpoints).

    If return_function=True and models have a function head, returns
    (disorder_by_id, function_by_id); otherwise a single disorder dict.
    """
    from colab.fold_model_soup import _predict_proteins_multitask

    proteins = parse_fasta(fasta_path)
    if plddt_by_id:
        for p in proteins:
            if p["id"] in plddt_by_id:
                p["_plddt"] = plddt_by_id[p["id"]]

    models = load_fold_ensemble_models(checkpoint_dir, esm_backbone, cfg, cfg.device, n_folds)
    token_cache: dict = {}
    accum: dict[str, list[np.ndarray]] = {}
    accum_fn: dict[str, list[np.ndarray]] = {}
    # Only change return type when the caller asks — cfg.use_function_head alone
    # must not turn a dict into a tuple (breaks CAID3 / legacy callers).
    want_fn = bool(return_function)

    for model in models:
        dis, fn = _predict_proteins_multitask(
            model, proteins, batch_converter, token_cache, cfg,
            plddt_by_id=plddt_by_id,
            use_tta=use_tta and cfg.use_mc_dropout_tta and not want_fn,
            tta_passes=tta_passes or cfg.mc_dropout_tta_passes,
            return_function=want_fn,
        )
        for pid, arr in dis.items():
            accum.setdefault(pid, []).append(arr)
        for pid, arr in fn.items():
            accum_fn.setdefault(pid, []).append(arr)

    disorder = {
        pid: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
        for pid, v in accum.items()
    }
    if not want_fn:
        return disorder
    function = {
        pid: np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
        for pid, v in accum_fn.items()
    }
    return disorder, function


def export_predictions(
    proteins: list[dict],
    preds_by_id: dict[str, np.ndarray],
    out_dir: str,
    formats: tuple[str, ...] = ("caid", "tsv"),
) -> dict:
    """Write predictions in CAID and/or TSV format."""
    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, list[str]] = {"caid": [], "tsv": []}

    for p in proteins:
        pid = p["id"]
        if pid not in preds_by_id:
            continue
        scores = preds_by_id[pid]
        safe = re.sub(r"[^\w\-.]", "_", pid)
        if "caid" in formats:
            caid_path = os.path.join(out_dir, f"{safe}.caid")
            write_caid_prediction_file(caid_path, pid, p["sequence"], scores)
            paths["caid"].append(caid_path)
        if "tsv" in formats:
            tsv_path = os.path.join(out_dir, f"{safe}.tsv")
            with open(tsv_path, "w") as f:
                f.write("position\tresidue\tscore\n")
                for i, aa in enumerate(p["sequence"][: len(scores)]):
                    f.write(f"{i + 1}\t{aa}\t{scores[i]:.4f}\n")
            paths["tsv"].append(tsv_path)

    manifest = {
        "n_proteins": len(proteins),
        "n_scored": len(preds_by_id),
        "out_dir": out_dir,
        "paths": paths,
    }
    with open(os.path.join(out_dir, "predict_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
