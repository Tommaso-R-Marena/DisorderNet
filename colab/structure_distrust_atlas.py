"""
Structure distrust atlas — proteome-scale post-AF/Boltz export (paper pillar 1).

Distinguishes:
  - **proxy distrust flags**: DN predicts disorder ∩ high pLDDT  (deployment triage;
    NOT scientific rescue)
  - **labeled hallucination / rescue**: independent DisProt (or other) labels

The atlas is the public-facing artifact for the claim that DisorderNet is the
default post-structure distrust layer.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from colab.af_plddt import plddt_to_disorder_score
from colab.biological_utility import intervals_from_binary
from colab.hallucination_benchmark import (
    PROTOCOL_VERSION,
    compare_distrust_baselines,
)


ATLAS_VERSION = "1.0.0"


def load_plddt_cache_for_proteins(
    proteins: list[dict],
    cache_dir: str,
) -> dict[str, np.ndarray]:
    """
    Load pLDDT arrays from a cache directory for atlas-only CPU export.

    Tries UniProt accession JSON (AF-style) then protein_id stem JSON
    (``{\"plddt\": [...]}`` or Boltz-style caches).
    """
    from colab.af_plddt import load_cached_plddt

    out: dict[str, np.ndarray] = {}
    if not cache_dir or not os.path.isdir(cache_dir):
        return out
    for p in proteins:
        pid = p["id"]
        seq = p["sequence"]
        acc = (p.get("uniprot_acc") or "").upper()
        arr = None
        if acc:
            arr = load_cached_plddt(acc, seq, cache_dir=cache_dir)
        if arr is None:
            for stem in (pid, acc, pid.replace("|", "_")):
                if not stem:
                    continue
                path = os.path.join(cache_dir, f"{stem}.json")
                if not os.path.isfile(path):
                    # case-insensitive scan of small caches
                    continue
                try:
                    with open(path) as f:
                        cached = json.load(f)
                    if "plddt" in cached:
                        arr = np.asarray(cached["plddt"], dtype=np.float32)
                        break
                except (OSError, json.JSONDecodeError, TypeError):
                    continue
        if arr is not None:
            out[pid] = arr
    return out


def compute_protein_distrust_row(
    *,
    protein_id: str,
    sequence: str,
    disorder_probs: np.ndarray,
    plddt: np.ndarray,
    labels: Optional[np.ndarray] = None,
    uniprot_acc: str = "",
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
) -> dict:
    """Per-protein atlas row with clear proxy vs labeled fields."""
    n = min(len(sequence), len(disorder_probs), len(plddt))
    dis = np.asarray(disorder_probs, dtype=np.float32).ravel()[:n]
    pld = np.asarray(plddt, dtype=np.float32).ravel()[:n]
    valid = ~np.isnan(pld)

    proxy_mask = valid & (dis >= disorder_threshold) & (pld >= high_plddt_threshold)
    proxy_regions = [
        (s, e) for s, e in intervals_from_binary(proxy_mask.astype(np.int8), min_len=3)
    ][:30]

    high_plddt = valid & (pld >= high_plddt_threshold)
    row: dict = {
        "atlas_version": ATLAS_VERSION,
        "protein_id": protein_id,
        "uniprot_acc": uniprot_acc or None,
        "length": n,
        "structure_source": structure_source,
        "n_valid_plddt": int(valid.sum()),
        "mean_plddt": float(np.nanmean(pld)) if valid.any() else None,
        "disorder_fraction_pred": float((dis >= disorder_threshold).mean()) if n else 0.0,
        "proxy_distrust": {
            "definition": "disorder_pred_AND_high_plddt",
            "n_residues": int(proxy_mask.sum()),
            "fraction_of_chain": float(proxy_mask.mean()) if n else 0.0,
            "fraction_of_high_plddt": (
                float(proxy_mask.sum() / high_plddt.sum()) if int(high_plddt.sum()) else 0.0
            ),
            "regions": proxy_regions,
            "note": (
                "Deployment triage flag — NOT an independent hallucination rescue rate"
            ),
        },
        "labeled": None,
        "action": (
            "prefer_disordernet_over_structure_confidence"
            if int(proxy_mask.sum()) > 0
            else "structure_confidence_mostly_aligned"
        ),
    }

    if labels is not None:
        lab = np.asarray(labels, dtype=np.int8).ravel()
        if len(lab) >= n:
            lab = lab[:n]
            labeled_metrics = compare_distrust_baselines(
                lab, dis, pld,
                disorder_threshold=disorder_threshold,
                high_plddt_threshold=high_plddt_threshold,
            )
            hall = labeled_metrics.get("hallucination_rescue") or {}
            halluc_mask = (
                valid & (lab == 1) & (pld >= high_plddt_threshold)
            )
            rescued_mask = halluc_mask & (dis >= disorder_threshold)
            row["labeled"] = {
                "definition": "independent_labels_AND_high_plddt",
                "n_hallucinated": int(hall.get("n_hallucinated", halluc_mask.sum())),
                "n_rescued": int(hall.get("n_rescued", rescued_mask.sum())),
                "rescue_rate": hall.get("rescue_rate"),
                "hallucination_rate": hall.get("hallucination_rate"),
                "baselines": {
                    "disordernet_auc": (labeled_metrics.get("disordernet") or {}).get("auc"),
                    "plddt_inverse_auc": (
                        (labeled_metrics.get("plddt_inverse_baseline") or {}).get("auc")
                    ),
                    "delta_auc_dn_minus_plddt": labeled_metrics.get("delta_auc_dn_minus_plddt"),
                },
                "regions": [
                    (s, e)
                    for s, e in intervals_from_binary(halluc_mask.astype(np.int8), min_len=3)
                ][:30],
            }
    return row


def build_structure_distrust_atlas(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    plddt_by_id: dict[str, np.ndarray],
    *,
    labels_by_id: Optional[dict[str, np.ndarray]] = None,
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
    structure_source: str = "boltz2",
) -> dict:
    """
    Proteome-scale structure distrust atlas.

    Always emits proxy distrust flags for deployment. When ``labels_by_id`` is
    provided (DisProt CV / curated sets), also emits independent rescue metrics.
    """
    labels_by_id = labels_by_id or {}
    rows: list[dict] = []
    for p in proteins:
        pid = p["id"]
        if pid not in disorder_probs_by_id or pid not in plddt_by_id:
            continue
        rows.append(
            compute_protein_distrust_row(
                protein_id=pid,
                sequence=p["sequence"],
                disorder_probs=disorder_probs_by_id[pid],
                plddt=plddt_by_id[pid],
                labels=labels_by_id.get(pid),
                uniprot_acc=p.get("uniprot_acc", ""),
                disorder_threshold=disorder_threshold,
                high_plddt_threshold=high_plddt_threshold,
                structure_source=structure_source,
            )
        )

    rows.sort(
        key=lambda r: -float((r.get("proxy_distrust") or {}).get("n_residues", 0)),
    )

    n_proxy = sum(1 for r in rows if (r.get("proxy_distrust") or {}).get("n_residues", 0) > 0)
    sum_proxy = sum(int((r.get("proxy_distrust") or {}).get("n_residues", 0)) for r in rows)
    labeled_rows = [r for r in rows if r.get("labeled")]
    n_halluc = sum(int((r["labeled"] or {}).get("n_hallucinated", 0)) for r in labeled_rows)
    n_rescued = sum(int((r["labeled"] or {}).get("n_rescued", 0)) for r in labeled_rows)

    # Optional pooled labeled baseline on concatenated residues
    pooled_baselines = None
    if labeled_rows and labels_by_id:
        ys, ps, plds = [], [], []
        for p in proteins:
            pid = p["id"]
            if pid not in labels_by_id or pid not in disorder_probs_by_id or pid not in plddt_by_id:
                continue
            lab = np.asarray(labels_by_id[pid], dtype=np.int8).ravel()
            dis = np.asarray(disorder_probs_by_id[pid], dtype=np.float32).ravel()
            pld = np.asarray(plddt_by_id[pid], dtype=np.float32).ravel()
            n = min(len(lab), len(dis), len(pld), len(p["sequence"]))
            ys.append(lab[:n])
            ps.append(dis[:n])
            plds.append(pld[:n])
        if ys:
            pooled_baselines = compare_distrust_baselines(
                np.concatenate(ys),
                np.concatenate(ps),
                np.concatenate(plds),
                disorder_threshold=disorder_threshold,
                high_plddt_threshold=high_plddt_threshold,
            )

    return {
        "atlas_version": ATLAS_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "thesis": (
            "After Boltz/AF, DisorderNet is the default post-structure distrust "
            "layer for IDR-aware interpretation"
        ),
        "structure_source": structure_source,
        "n_proteins": len(rows),
        "n_proteins_with_proxy_distrust": n_proxy,
        "total_proxy_distrust_residues": sum_proxy,
        "proxy_distrust_protein_fraction": (
            round(n_proxy / len(rows), 4) if rows else 0.0
        ),
        "labeled_evaluation": {
            "n_proteins_with_labels": len(labeled_rows),
            "total_hallucinated_residues": n_halluc,
            "total_rescued_residues": n_rescued,
            "overall_rescue_rate": (
                round(float(n_rescued / n_halluc), 4) if n_halluc else None
            ),
            "pooled_baselines": pooled_baselines,
            "note": "Rescue rates only meaningful on this labeled subset",
        },
        "thresholds": {
            "disorder_threshold": disorder_threshold,
            "high_plddt_threshold": high_plddt_threshold,
        },
        "non_claims": [
            "proxy_flags_are_not_independent_rescue",
            "not_full_md_ensembles",
            "not_alphafold_replacement",
        ],
        "top_distrust_proteins": [
            {
                "protein_id": r["protein_id"],
                "uniprot_acc": r.get("uniprot_acc"),
                "n_proxy_distrust": (r.get("proxy_distrust") or {}).get("n_residues"),
                "n_hallucinated_labeled": (r.get("labeled") or {}).get("n_hallucinated"),
                "action": r.get("action"),
            }
            for r in rows[:30]
        ],
        "proteins": rows,
    }


def estimate_downstream_mask_utility(
    labels: np.ndarray,
    disorder_probs: np.ndarray,
    plddt: np.ndarray,
    *,
    disorder_threshold: float = 0.5,
    high_plddt_threshold: float = 70.0,
) -> dict:
    """
    Computational downstream utility of the distrust mask (labeled only).

    Asks: among high-pLDDT residues, does masking those where DN predicts
    disorder enrich for true DisProt disorder vs masking randomly / by
    inverse-pLDDT alone?
    """
    labels = np.asarray(labels, dtype=np.int8).ravel()
    probs = np.asarray(disorder_probs, dtype=np.float32).ravel()
    plddt = np.asarray(plddt, dtype=np.float32).ravel()
    n = min(len(labels), len(probs), len(plddt))
    labels, probs, plddt = labels[:n], probs[:n], plddt[:n]
    valid = ~np.isnan(plddt)
    high = valid & (plddt >= high_plddt_threshold)
    if int(high.sum()) < 10:
        return {"enabled": False, "insufficient_data": True}

    y_high = labels[high]
    dn_mask = probs[high] >= disorder_threshold
    inv = plddt_to_disorder_score(plddt[high])
    # Match DN mask size for fair comparison
    k = int(dn_mask.sum())
    if k < 5:
        return {"enabled": False, "insufficient_data": True, "reason": "empty_dn_mask"}
    plddt_mask = np.zeros_like(dn_mask)
    order = np.argsort(-inv)
    plddt_mask[order[:k]] = True

    def _prec(mask: np.ndarray) -> Optional[float]:
        if int(mask.sum()) == 0:
            return None
        return float(y_high[mask].mean())

    dn_prec = _prec(dn_mask)
    pl_prec = _prec(plddt_mask)
    base_rate = float(y_high.mean())
    return {
        "enabled": True,
        "n_high_plddt": int(high.sum()),
        "n_masked": k,
        "base_disorder_rate_in_high_plddt": round(base_rate, 4),
        "precision_dn_distrust_mask": None if dn_prec is None else round(dn_prec, 4),
        "precision_plddt_size_matched_mask": None if pl_prec is None else round(pl_prec, 4),
        "lift_vs_base": (
            None if dn_prec is None or base_rate <= 0
            else round(float(dn_prec / base_rate), 4)
        ),
        "lift_vs_plddt_mask": (
            None if dn_prec is None or pl_prec in (None, 0)
            else round(float(dn_prec / pl_prec), 4)
        ),
        "note": (
            "Higher precision = mask better enriches true disorder among "
            "high-pLDDT residues (computational utility proxy)"
        ),
    }


def export_distrust_atlas_jsonl(atlas: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for row in atlas.get("proteins") or []:
            f.write(json.dumps(row) + "\n")
    return path


def export_distrust_atlas_tsv(atlas: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(
            "protein_id\tuniprot_acc\tlength\tn_proxy_distrust\t"
            "proxy_fraction\tn_hallucinated_labeled\tn_rescued_labeled\taction\n"
        )
        for r in atlas.get("proteins") or []:
            lab = r.get("labeled") or {}
            px = r.get("proxy_distrust") or {}
            f.write(
                f"{r['protein_id']}\t{r.get('uniprot_acc') or ''}\t{r.get('length', 0)}\t"
                f"{px.get('n_residues', 0)}\t{px.get('fraction_of_chain', 0)}\t"
                f"{lab.get('n_hallucinated', '')}\t{lab.get('n_rescued', '')}\t"
                f"{r.get('action', '')}\n"
            )
    return path


def save_distrust_atlas(atlas: dict, path: str) -> str:
    # Embed proteins in a separate JSONL for large proteomes; report keeps summary
    summary = {k: v for k, v in atlas.items() if k != "proteins"}
    summary["n_proteins_embedded"] = 0
    summary["note"] = "Full per-protein rows in accompanying JSONL/TSV"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def export_structure_distrust_atlas_bundle(
    atlas: dict,
    out_dir: str,
    *,
    include_proteins_in_json: bool = False,
) -> dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "structure_distrust_atlas_report.json")
    if include_proteins_in_json:
        with open(report_path, "w") as f:
            json.dump(atlas, f, indent=2)
    else:
        save_distrust_atlas(atlas, report_path)
    paths = {
        "report": report_path,
        "jsonl": export_distrust_atlas_jsonl(
            atlas, os.path.join(out_dir, "structure_distrust_atlas.jsonl"),
        ),
        "tsv": export_distrust_atlas_tsv(
            atlas, os.path.join(out_dir, "structure_distrust_atlas.tsv"),
        ),
    }
    return paths


def print_distrust_atlas(atlas: dict) -> None:
    print(f"\n{'═' * 60}")
    print(" Structure distrust atlas (post-AF/Boltz default layer)")
    print(f"{'═' * 60}")
    print(f"  atlas={atlas.get('atlas_version')}  proteins={atlas.get('n_proteins', 0)}")
    print(
        f"  proxy distrust proteins={atlas.get('n_proteins_with_proxy_distrust')}  "
        f"residues={atlas.get('total_proxy_distrust_residues')}"
    )
    lab = atlas.get("labeled_evaluation") or {}
    if lab.get("n_proteins_with_labels"):
        print(
            f"  labeled: proteins={lab.get('n_proteins_with_labels')}  "
            f"halluc={lab.get('total_hallucinated_residues')}  "
            f"rescue_rate={lab.get('overall_rescue_rate')}"
        )
        pb = lab.get("pooled_baselines") or {}
        if pb.get("enabled"):
            print(
                f"  matched AUC Δ(DN−pLDDT)={pb.get('delta_auc_dn_minus_plddt')}"
            )
    print("  non-claims:", ", ".join(atlas.get("non_claims") or []))
