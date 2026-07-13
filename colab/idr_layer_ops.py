"""
Operational helpers for the IDR biology layer (v1.4+).

Caching, landscape stats, markdown reports, JSONL compare — keep
``idr_biology_layer.py`` focused on per-protein / proteome composition.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np


def sequence_fingerprint(sequence: str) -> str:
    return hashlib.sha1((sequence or "").encode("utf-8")).hexdigest()[:16]


def layer_record_cache_key(
    *,
    protein_id: str,
    sequence: str,
    layer_version: str,
    disorder_threshold: float,
    function_threshold: float,
    high_plddt_threshold: float,
    has_function: bool,
    has_plddt: bool,
    has_variance: bool,
    has_partners: bool,
    has_ligands: bool,
) -> str:
    payload = "|".join([
        protein_id,
        sequence_fingerprint(sequence),
        layer_version,
        f"{disorder_threshold:.4f}",
        f"{function_threshold:.4f}",
        f"{high_plddt_threshold:.1f}",
        str(int(has_function)),
        str(int(has_plddt)),
        str(int(has_variance)),
        str(int(has_partners)),
        str(int(has_ligands)),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_cached_layer_record(cache_dir: str, key: str) -> Optional[dict]:
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{key}.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_cached_layer_record(cache_dir: str, key: str, record: dict) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w") as f:
        json.dump(record, f)
    return path


def proteome_landscape_summary(records: list[dict]) -> dict:
    """Compact proteome composition / role frequency summary."""
    if not records:
        return {"n_proteins": 0, "insufficient_data": True}

    fracs = [float(r.get("disorder_fraction", 0.0)) for r in records]
    bins = {
        "mostly_ordered_<0.1": sum(1 for x in fracs if x < 0.1),
        "mixed_0.1_0.4": sum(1 for x in fracs if 0.1 <= x < 0.4),
        "idr_rich_0.4_0.7": sum(1 for x in fracs if 0.4 <= x < 0.7),
        "mostly_disordered_>=0.7": sum(1 for x in fracs if x >= 0.7),
    }
    role_counts: Counter = Counter()
    cue_counts: Counter = Counter()
    for r in records:
        for seg in r.get("idr_segments", []):
            for role in seg.get("predicted_roles", []):
                role_counts[role["group"]] += 1
                for ev in role.get("evidence") or []:
                    cue_counts[ev] += 1
            for tag in (seg.get("sequence_cues") or {}).get("cue_tags", []):
                cue_counts[f"seq:{tag}"] += 1

    return {
        "n_proteins": len(records),
        "mean_disorder_fraction": round(float(np.mean(fracs)), 4),
        "median_disorder_fraction": round(float(np.median(fracs)), 4),
        "disorder_fraction_bins": bins,
        "role_call_frequency": dict(role_counts.most_common()),
        "evidence_frequency": dict(cue_counts.most_common(20)),
        "n_proteins_with_roles": sum(1 for r in records if r.get("n_role_assignments", 0) > 0),
        "n_proteins_with_hallucination": sum(
            1 for r in records
            if (r.get("hallucination") or {}).get("n_hallucinated", 0) > 0
        ),
    }


def write_idr_layer_markdown(report: dict, path: str) -> str:
    """Human-readable Markdown summary for Rockfish / lab notebooks."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        f"# DisorderNet IDR Biology Layer",
        "",
        f"- **version**: `{report.get('layer_version')}`",
        f"- **proteins**: {report.get('n_proteins', 0)}",
        f"- **mean disorder fraction**: {report.get('mean_disorder_fraction', 0):.3f}",
        f"- **role assignments**: {report.get('total_role_assignments', 0)}",
        f"- **hallucinated residues**: {report.get('total_hallucinated_residues', 0)}",
        "",
        "## Thesis",
        "",
        str(report.get("thesis", "")),
        "",
        "## Non-goals",
        "",
    ]
    for ng in report.get("non_goals") or []:
        lines.append(f"- `{ng}`")

    land = report.get("landscape") or {}
    if land and not land.get("insufficient_data"):
        lines += ["", "## Proteome landscape", ""]
        lines.append(f"- mean / median disorder: "
                     f"{land.get('mean_disorder_fraction')} / {land.get('median_disorder_fraction')}")
        bins = land.get("disorder_fraction_bins") or {}
        for k, v in bins.items():
            lines.append(f"- {k}: {v}")
        roles = land.get("role_call_frequency") or {}
        if roles:
            lines += ["", "### Role call frequency", ""]
            for name, n in list(roles.items())[:10]:
                lines.append(f"- {name}: {n}")

    rv = report.get("role_validation") or {}
    if rv.get("enabled"):
        micro = rv.get("micro") or {}
        lines += [
            "", "## Role validation (vs DisProt)", "",
            f"- precision={micro.get('precision')}  recall={micro.get('recall')}  "
            f"F1={micro.get('f1')}",
            f"- proteins with annotations: {rv.get('n_proteins_with_annotations')}",
        ]

    thr = report.get("thresholds") or {}
    tune = report.get("function_threshold_tuning") or {}
    if thr or tune:
        lines += ["", "## Thresholds", ""]
        lines.append(f"- disorder={thr.get('disorder_threshold')}  "
                     f"function={thr.get('function_threshold')}")
        if tune.get("enabled"):
            lines.append(
                f"- OOF-tuned function threshold={tune.get('threshold')}  "
                f"({tune.get('metric')}={tune.get('best_score')})"
            )

    top = report.get("top_priority_proteins") or []
    if top:
        lines += ["", "## Top priority proteins", "",
                  "| protein | score | roles | halluc | reasons |",
                  "|---|---:|---:|---:|---|"]
        for row in top[:15]:
            reasons = ", ".join(row.get("reasons") or [])
            lines.append(
                f"| {row.get('protein_id')} | {row.get('triage_score')} | "
                f"{row.get('n_role_assignments')} | {row.get('n_hallucinated')} | {reasons} |"
            )

    lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _iter_jsonl(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode, encoding="utf-8") as f:  # type: ignore[arg-type]
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compare_idr_layer_jsonl(path_a: str, path_b: str) -> dict:
    """
    Diff two IDR layer JSONL exports (e.g. before/after threshold change).

    Accuracy-neutral QA tool — ranks proteins by triage / role / hallucination deltas.
    """
    a = {rec["protein_id"]: rec for rec in _iter_jsonl(path_a)}
    b = {rec["protein_id"]: rec for rec in _iter_jsonl(path_b)}
    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    deltas: list[dict] = []
    for pid in shared:
        ra, rb = a[pid], b[pid]
        sa = float((ra.get("triage") or {}).get("score", 0.0))
        sb = float((rb.get("triage") or {}).get("score", 0.0))
        deltas.append({
            "protein_id": pid,
            "triage_delta": round(sb - sa, 4),
            "roles_a": ra.get("n_role_assignments", 0),
            "roles_b": rb.get("n_role_assignments", 0),
            "roles_delta": int(rb.get("n_role_assignments", 0)) - int(ra.get("n_role_assignments", 0)),
            "halluc_a": (ra.get("hallucination") or {}).get("n_hallucinated", 0),
            "halluc_b": (rb.get("hallucination") or {}).get("n_hallucinated", 0),
        })
    deltas.sort(key=lambda d: -abs(d["triage_delta"]))

    return {
        "path_a": path_a,
        "path_b": path_b,
        "n_a": len(a),
        "n_b": len(b),
        "n_shared": len(shared),
        "only_in_a": only_a[:50],
        "only_in_b": only_b[:50],
        "largest_triage_deltas": deltas[:30],
        "mean_abs_triage_delta": (
            round(float(np.mean([abs(d["triage_delta"]) for d in deltas])), 4)
            if deltas else 0.0
        ),
    }


def export_disorder_caid_bundle(
    proteins: list[dict],
    disorder_probs_by_id: dict[str, np.ndarray],
    out_dir: str,
    *,
    threshold: float = 0.5,
) -> list[str]:
    """Write CAID-format disorder predictions next to the layer bundle."""
    from colab.caid3_eval import write_caid_prediction_file

    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    for p in proteins:
        pid = p["id"]
        if pid not in disorder_probs_by_id:
            continue
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in pid)
        path = os.path.join(out_dir, f"{safe}.caid")
        write_caid_prediction_file(
            path, pid, p["sequence"], disorder_probs_by_id[pid], threshold=threshold,
        )
        paths.append(path)
    return paths
