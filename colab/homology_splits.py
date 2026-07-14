"""
Homology-aware CV splits for CAID-credible evaluation.

Clusters proteins by sequence similarity (default 40% identity) so homologs
never appear in both train and validation folds — closer to CAID/ESMDisPred
protocol than protein-ID-only GroupKFold.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Optional

import numpy as np


def sequence_identity(seq_a: str, seq_b: str) -> float:
    """Global alignment-free identity ratio (SequenceMatcher)."""
    if not seq_a or not seq_b:
        return 0.0
    return SequenceMatcher(None, seq_a, seq_b).ratio()


def _length_bin(length: int, width: int = 50) -> int:
    return length // width


def cluster_proteins_by_homology(
    proteins: list,
    min_identity: float = 0.40,
    length_bin_width: int = 50,
) -> tuple[np.ndarray, dict]:
    """
    Greedy single-linkage clustering within length bins.

    Returns (cluster_ids, metadata) where cluster_ids[i] is the cluster for proteins[i].
    """
    n = len(proteins)
    cluster_ids = np.arange(n, dtype=np.int64)
    parent = {i: i for i in range(n)}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    buckets: dict[int, list[int]] = {}
    for i, p in enumerate(proteins):
        b = _length_bin(p["length"], length_bin_width)
        buckets.setdefault(b, []).append(i)

    n_merges = 0
    for indices in buckets.values():
        for ii in range(len(indices)):
            i = indices[ii]
            si = proteins[i]["sequence"]
            for jj in range(ii + 1, len(indices)):
                j = indices[jj]
                if find(i) == find(j):
                    continue
                sj = proteins[j]["sequence"]
                if sequence_identity(si, sj) >= min_identity:
                    union(i, j)
                    n_merges += 1

    roots = [find(i) for i in range(n)]
    unique_roots = {r: idx for idx, r in enumerate(sorted(set(roots)))}
    cluster_ids = np.array([unique_roots[r] for r in roots], dtype=np.int64)

    meta = {
        "n_proteins": n,
        "n_clusters": int(len(unique_roots)),
        "n_merges": n_merges,
        "min_identity": min_identity,
        "length_bin_width": length_bin_width,
        "method": "greedy_length_bucketed",
    }
    return cluster_ids, meta


def get_homology_cv_splits(
    proteins: list,
    n_folds: int,
    min_identity: float = 0.40,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    GroupKFold on homology cluster IDs (not individual protein IDs).
    """
    from sklearn.model_selection import GroupKFold

    cluster_ids, meta = cluster_proteins_by_homology(proteins, min_identity=min_identity)
    groups = cluster_ids
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(len(proteins)), groups=groups))
    meta["split_method"] = "homology"
    return splits, meta
