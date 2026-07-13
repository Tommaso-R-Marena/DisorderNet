"""
CAID3 benchmark evaluation harness — fair comparison vs ESMDisPred (0.895).

Loads CAID reference FASTA (labels on second line), scores predictions,
exports .caid submission format, and reports CAID-standard metrics.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from colab.caid_reporting import compute_caid_metrics, compute_f1_max

# CAID3 Disorder-PDB reference (public CAID challenge data)
CAID3_DISORDER_PDB_URL = (
    "https://raw.githubusercontent.com/BioComputingUP/CAID/master/demo-data/"
    "references/disorder_pdb.fasta"
)


def parse_caid_reference_fasta(path: str) -> list[dict]:
    """
    Parse CAID reference FASTA.

    Format: header line, sequence line, optional label line (0/1/-).
    If labels are absent on line 3, they may appear in header as labels=...
    """
    proteins: list[dict] = []
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        header = lines[i][1:]
        i += 1
        if i >= len(lines):
            break
        sequence = lines[i].upper()
        i += 1
        labels_str: Optional[str] = None
        if i < len(lines) and not lines[i].startswith(">"):
            candidate = lines[i]
            if re.fullmatch(r"[01\-]+", candidate):
                labels_str = candidate
                i += 1
        if labels_str is None:
            m = re.search(r"labels=([01\-]+)", header)
            if m:
                labels_str = m.group(1)
        if labels_str is None:
            continue
        pid = header.split()[0]
        n = min(len(sequence), len(labels_str))
        labels: list[int] = []
        eval_mask: list[bool] = []
        for j in range(n):
            c = labels_str[j]
            if c == "-":
                eval_mask.append(False)
            else:
                eval_mask.append(True)
                labels.append(1 if c == "1" else 0)
        proteins.append({
            "id": pid,
            "sequence": sequence[:n],
            "length": n,
            "labels": labels,
            "eval_mask": eval_mask[:n],
            "caid_header": header,
        })
    return proteins


def fetch_caid3_reference(cache_path: str = "caid3_disorder_pdb.fasta") -> str:
    """Download CAID3 Disorder-PDB reference if missing."""
    if os.path.isfile(cache_path):
        return cache_path
    import requests

    print(f"Downloading CAID3 reference from {CAID3_DISORDER_PDB_URL}…")
    resp = requests.get(CAID3_DISORDER_PDB_URL, timeout=120)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(resp.text)
    print(f"  Saved {cache_path} ({len(resp.text):,} bytes)")
    return cache_path


def write_caid_prediction_file(
    path: str,
    protein_id: str,
    sequence: str,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> None:
    """
    Write one .caid prediction file (CAID submission format).

    Columns: position, residue, score, binary_state
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    scores = np.asarray(scores, dtype=np.float32)
    n = min(len(sequence), len(scores))
    if threshold is None:
        threshold = 0.5
    with open(path, "w") as f:
        f.write(f">{protein_id}\n")
        for i in range(n):
            aa = sequence[i]
            sc = float(scores[i])
            state = 1 if sc >= threshold else 0
            f.write(f"{i + 1}\t{aa}\t{sc:.4f}\t{state}\n")


def export_caid_predictions_dir(
    proteins: list[dict],
    preds_by_id: dict[str, np.ndarray],
    out_dir: str,
    threshold: Optional[float] = None,
) -> list[str]:
    """Export all predictions to a directory of .caid files."""
    paths: list[str] = []
    os.makedirs(out_dir, exist_ok=True)
    for p in proteins:
        pid = p["id"]
        if pid not in preds_by_id:
            continue
        safe = re.sub(r"[^\w\-.]", "_", pid)
        path = os.path.join(out_dir, f"{safe}.caid")
        write_caid_prediction_file(path, pid, p["sequence"], preds_by_id[pid], threshold)
        paths.append(path)
    return paths


def evaluate_caid_predictions(
    reference_proteins: list[dict],
    preds_by_id: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> dict:
    """
    Score predictions against CAID reference (eval_mask respected).
    """
    all_labels: list[float] = []
    all_probs: list[float] = []
    per_protein: list[dict] = []
    n_missing = 0

    for p in reference_proteins:
        pid = p["id"]
        if pid not in preds_by_id:
            n_missing += 1
            continue
        probs = np.asarray(preds_by_id[pid], dtype=np.float32)
        labels = np.asarray(p["labels"], dtype=np.int8)
        mask = np.asarray(p.get("eval_mask", [True] * len(labels)), dtype=bool)
        n = min(len(probs), len(labels), len(mask))
        if n == 0:
            continue
        lab = labels[:n][mask[:n]]
        prb = probs[:n][mask[:n]]
        if len(lab) < 5 or len(np.unique(lab)) < 2:
            continue
        all_labels.extend(lab.tolist())
        all_probs.extend(prb.tolist())
        m = compute_caid_metrics(lab, prb, threshold=threshold)
        per_protein.append({"id": pid, **m})

    if len(all_labels) < 10 or len(np.unique(all_labels)) < 2:
        return {
            "insufficient_data": True,
            "n_reference": len(reference_proteins),
            "n_scored": len(per_protein),
            "n_missing_predictions": n_missing,
        }

    labels_arr = np.asarray(all_labels, dtype=np.int8)
    probs_arr = np.asarray(all_probs, dtype=np.float32)
    pooled = compute_caid_metrics(labels_arr, probs_arr, threshold=threshold)
    f1m = compute_f1_max(labels_arr, probs_arr)

    return {
        "insufficient_data": False,
        "benchmark": "CAID3_disorder_pdb",
        "n_reference": len(reference_proteins),
        "n_scored": len(per_protein),
        "n_missing_predictions": n_missing,
        "pooled": {
            **pooled,
            "f1_max": f1m["f1_max"],
            "threshold_at_f1_max": f1m["threshold_at_f1_max"],
        },
        "per_protein": per_protein[:20],
        "esmdispred_reference_auc": 0.895,
        "delta_vs_esmdispred": float(pooled["auc"]) - 0.895 if pooled.get("auc") else None,
    }


def save_caid3_eval_report(report: dict, path: str = "caid3_eval_report.json") -> str:
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def print_caid3_eval_report(report: dict) -> None:
    print(f"\n{'═' * 64}")
    print(" CAID3 BENCHMARK EVALUATION (Disorder-PDB)")
    print(f"{'═' * 64}")
    if report.get("insufficient_data"):
        print(f"  Insufficient data — scored {report.get('n_scored', 0)} proteins")
        print(f"{'═' * 64}")
        return
    p = report["pooled"]
    print(f"  Proteins scored : {report['n_scored']}/{report['n_reference']}")
    print(f"  Pooled AUC      : {p.get('auc', 0):.4f}")
    print(f"  Pooled AP       : {p.get('ap', 0):.4f}")
    print(f"  F1_max          : {p.get('f1_max', 0):.4f}")
    print(f"  MCC @ F1_max    : {p.get('mcc_at_f1_max', 0):.4f}")
    delta = report.get("delta_vs_esmdispred")
    if delta is not None:
        print(f"  vs ESMDisPred   : {delta:+.4f}  (ref 0.895, CAID3 protocol)")
    print(f"{'═' * 64}")
