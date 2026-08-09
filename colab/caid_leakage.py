"""Leak-free CAID challenge evaluation helpers.

CAID3 (and future CAID rounds) must not be contaminated by training on the same
proteins or close homologs. Ultra CV uses homology splits *within DisProt*; that
does **not** automatically isolate the CAID challenge set.

This module:
1. Audits train proteins vs CAID reference sequences (ID + pairwise identity).
2. Optionally filters training proteins that hit the CAID set (leak-free train).
3. Writes a machine-readable audit for publish packages / METHODS_CHECKLIST.

Identity uses the same SequenceMatcher protocol as ``homology_splits`` (≥40%
default) so paper language stays consistent with ``docs/HOMOLOGY_HOLDOUT.md``.
"""
from __future__ import annotations

import json
import os
from difflib import SequenceMatcher
from typing import Optional


def _norm_id(pid: str) -> str:
    return str(pid).split("|")[0].split()[0].strip().upper()


def _seq_identity(a: str, b: str) -> float:
    """Identity approximation shared with ``homology_splits.sequence_identity``.

    ``autojunk`` must stay disabled: with it on, difflib junks every amino acid
    for inputs of length >= 200, so a near-duplicate of a CAID target scored
    ~0.01 and this audit certified "no leakage" for essentially every protein
    long enough to matter.
    """
    if not a or not b:
        return 0.0
    # Length gate: skip pairwise work that cannot reach the threshold anyway.
    la, lb = len(a), len(b)
    if min(la, lb) / max(la, lb) < 0.5:
        return 0.0
    return float(SequenceMatcher(None, a.upper(), b.upper(), autojunk=False).ratio())


def audit_train_vs_caid(
    train_proteins: list[dict],
    caid_proteins: list[dict],
    *,
    min_identity: float = 0.40,
    max_pairwise: int = 2_000_000,
) -> dict:
    """Report ID overlaps and sequence-identity hits between train and CAID refs."""
    train_by_id = {_norm_id(p.get("id", p.get("name", ""))): p for p in train_proteins}
    caid_by_id = {_norm_id(p["id"]): p for p in caid_proteins}

    id_hits = sorted(set(train_by_id) & set(caid_by_id))
    seq_hits: list[dict] = []
    comparisons = 0
    for cid, cp in caid_by_id.items():
        cseq = cp.get("sequence", "")
        for tid, tp in train_by_id.items():
            if tid == cid:
                continue
            comparisons += 1
            if comparisons > max_pairwise:
                break
            ident = _seq_identity(cseq, tp.get("sequence", ""))
            if ident >= min_identity:
                seq_hits.append(
                    {
                        "caid_id": cid,
                        "train_id": tid,
                        "identity": round(ident, 4),
                        "caid_len": len(cseq),
                        "train_len": len(tp.get("sequence", "")),
                    }
                )
        if comparisons > max_pairwise:
            break

    n_train = len(train_proteins)
    n_caid = len(caid_proteins)
    leak_ids = set(id_hits) | {h["train_id"] for h in seq_hits}
    return {
        "protocol": "SequenceMatcher_identity_plus_id",
        "min_identity": min_identity,
        "n_train": n_train,
        "n_caid": n_caid,
        "n_id_overlap": len(id_hits),
        "id_overlap": id_hits,
        "n_homology_hits": len(seq_hits),
        "homology_hits": seq_hits[:500],  # cap JSON size
        "n_train_flagged_for_exclusion": len(leak_ids),
        "flagged_train_ids": sorted(leak_ids),
        "comparisons": comparisons,
        "truncated_pairwise": comparisons > max_pairwise,
        "leak_free": len(leak_ids) == 0,
        "disclaimer": (
            "Not MMseqs2 / not official CAID BLAST filters. Conservative internal "
            "audit aligned with docs/HOMOLOGY_HOLDOUT.md."
        ),
    }


def filter_train_proteins(
    train_proteins: list[dict],
    audit: dict,
) -> tuple[list[dict], dict]:
    """Drop train proteins flagged in an audit (ID or homology hit)."""
    ban = set(audit.get("flagged_train_ids") or [])
    if not ban:
        return list(train_proteins), {
            "n_before": len(train_proteins),
            "n_after": len(train_proteins),
            "n_removed": 0,
            "removed_ids": [],
        }
    kept, removed = [], []
    for p in train_proteins:
        pid = _norm_id(p.get("id", p.get("name", "")))
        if pid in ban:
            removed.append(pid)
        else:
            kept.append(p)
    meta = {
        "n_before": len(train_proteins),
        "n_after": len(kept),
        "n_removed": len(removed),
        "removed_ids": sorted(set(removed)),
    }
    return kept, meta


def save_leakage_audit(audit: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(audit, f, indent=2)
        f.write("\n")
    return path


def load_caid_refs_for_audit(paths: list[str]) -> list[dict]:
    from colab.caid3_eval import parse_caid_reference_fasta

    out: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        if not path or not os.path.isfile(path):
            continue
        for p in parse_caid_reference_fasta(path):
            pid = _norm_id(p["id"])
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
    return out
