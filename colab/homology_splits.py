"""
Homology-aware CV splits for CAID-credible evaluation.

Clusters proteins by sequence similarity (default 40%) so homologues never
appear in both train and validation folds — closer to the CAID/ESMDisPred
protocol than protein-ID-only GroupKFold.

Similarity metric
-----------------
``sequence_identity`` is the Ratcliff/Obershelp matching-block ratio
(``difflib.SequenceMatcher``), i.e. ``2 * matched_characters / (len_a + len_b)``.
It is an alignment-free *approximation* to sequence identity, not a
Smith-Waterman/BLAST identity, and it is not gap-aware. It is adequate for
partitioning but should be described as such in methods text; a dedicated tool
(MMseqs2, CD-HIT) is the right choice if exact identity thresholds matter.

Two correctness notes, both of which previously made clustering a silent no-op:

* ``SequenceMatcher`` enables the ``autojunk`` heuristic by default, which for
  inputs of length >= 200 treats any element occurring in more than 1% of
  positions as junk. Every amino acid clears 1%, so for a protein longer than
  199 residues *all* characters were junked and two 95%-identical sequences
  scored ~0.01 instead of ~0.95. Nothing ever reached a 0.40 threshold, so every
  protein became its own cluster and "homology" splits silently degenerated into
  "protein" splits. ``autojunk=False`` is mandatory here.
* Candidate pairs were previously bucketed into fixed 50-residue length bins and
  compared only within a bin, so homologues whose lengths straddled a bin edge
  (e.g. 249 vs 251) were never compared. Candidates are now taken from a length
  *window* derived from the metric's own algebra, which has no boundary effect.
"""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import Optional

import numpy as np

# Candidate prefilter: skip the O(L^2) matcher for pairs that share too few
# k-mers to be homologous. This is a *speed* filter — a threshold that is too low
# only costs time, never correctness — so it is deliberately set well inside the
# recall margin.
#
# Calibrated on synthetic pairs spanning 250-1500 residues:
#   k=3  homologue min 0.128 vs unrelated max 0.170  -> unusable; 3-mer sets
#        saturate the 8000-mer space on long sequences, so everything looks similar
#   k=4  homologue min 0.041 vs unrelated max 0.009  -> ~4.3x separation
#   k=5  homologue min 0.004                          -> margin too thin/noisy
# k=4 at 0.01 keeps a ~4x margin below the homologue floor.
_KMER_K = 4
_KMER_PREFILTER_CONTAINMENT = 0.01
# Below this length the prefilter is pure overhead.
_KMER_MIN_LEN = 60


def sequence_identity(seq_a: str, seq_b: str) -> float:
    """Alignment-free identity approximation in [0, 1].

    ``autojunk`` is disabled — see the module docstring for why leaving it on
    silently zeroes the score for any sequence longer than 199 residues.
    """
    if not seq_a or not seq_b:
        return 0.0
    return float(SequenceMatcher(None, seq_a, seq_b, autojunk=False).ratio())


def _max_length_ratio(min_identity: float) -> float:
    """Longest partner length, as a multiple of the shorter, that can still reach
    ``min_identity``.

    Matched characters M satisfy M <= min(La, Lb), so
    ``ratio = 2M / (La + Lb) <= 2 * min(La, Lb) / (La + Lb)``. Requiring that
    upper bound to reach ``t`` with ``La <= Lb`` gives ``Lb <= La * (2 - t) / t``.
    This is exact: pairs outside the window cannot clear the threshold, so
    skipping them discards nothing.
    """
    t = min(max(float(min_identity), 1e-6), 1.0)
    return (2.0 - t) / t


def _kmer_set(seq: str, k: int = _KMER_K) -> frozenset:
    if len(seq) < k:
        return frozenset()
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1))


def _kmer_containment(ka: frozenset, kb: frozenset) -> float:
    if not ka or not kb:
        return 1.0  # too short to filter on; defer to the real comparison
    return len(ka & kb) / min(len(ka), len(kb))


_WORKER_SEQS: list = []
_WORKER_MIN_IDENTITY: float = 0.40


def _init_worker(seqs: list, min_identity: float) -> None:
    """Seed worker globals once, instead of pickling every sequence per chunk."""
    global _WORKER_SEQS, _WORKER_MIN_IDENTITY
    _WORKER_SEQS = seqs
    _WORKER_MIN_IDENTITY = min_identity


def _identity_chunk(pairs: list) -> list:
    """Worker: verify a chunk of candidate pairs. Module-level so it pickles."""
    seqs = _WORKER_SEQS
    thr = _WORKER_MIN_IDENTITY
    return [sequence_identity(seqs[i], seqs[j]) >= thr for i, j in pairs]


def _verify_pairs(
    seqs: list,
    pairs: list,
    min_identity: float,
    n_jobs: Optional[int] = None,
) -> list:
    """Evaluate ``sequence_identity >= min_identity`` for each candidate pair."""
    if not pairs:
        return []

    if n_jobs is None:
        n_jobs = int(os.environ.get("DISORDERNET_HOMOLOGY_JOBS", "0")) or (os.cpu_count() or 1)
    n_jobs = max(1, min(int(n_jobs), os.cpu_count() or 1))

    # Process startup only pays off on a real workload.
    if n_jobs == 1 or len(pairs) < 512:
        _init_worker(seqs, min_identity)
        return _identity_chunk(pairs)

    # Many small chunks keep workers busy when pair costs vary by ~L^2.
    chunk = max(32, (len(pairs) + (n_jobs * 8) - 1) // (n_jobs * 8))
    chunks = [pairs[s:s + chunk] for s in range(0, len(pairs), chunk)]
    try:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=n_jobs,
            initializer=_init_worker,
            initargs=(seqs, min_identity),
        ) as ex:
            results = list(ex.map(_identity_chunk, chunks))
        out: list = []
        for r in results:
            out.extend(r)
        return out
    except Exception:
        # Restricted environments (no fork, no /dev/shm) must still produce the
        # correct clustering, just slower.
        _init_worker(seqs, min_identity)
        return _identity_chunk(pairs)


def cluster_proteins_by_homology(
    proteins: list,
    min_identity: float = 0.40,
    length_bin_width: Optional[int] = None,  # deprecated; retained for compatibility
    n_jobs: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """
    Greedy single-linkage clustering by sequence identity.

    Returns ``(cluster_ids, metadata)`` where ``cluster_ids[i]`` is the cluster
    of ``proteins[i]`` in the caller's original ordering.

    ``length_bin_width`` is accepted but ignored: fixed-width bins dropped
    homologous pairs that straddled a bin edge. Candidate generation now uses an
    exact length window instead.
    """
    n = len(proteins)
    if n == 0:
        return np.zeros(0, dtype=np.int64), {
            "n_proteins": 0, "n_clusters": 0, "n_merges": 0,
            "min_identity": min_identity, "method": "greedy_single_linkage",
            "degenerate": False,
        }

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    seqs = [p["sequence"] for p in proteins]
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    ratio = _max_length_ratio(min_identity)

    # Candidate generation via an inverted k-mer index. Scanning the full length
    # window pairwise is O(n^2) matcher calls — ~5.5M for DisProt's 3333
    # proteins, which does not finish. The index instead visits only proteins
    # that actually share k-mers, so the expensive comparison runs on a
    # candidate list that is orders of magnitude smaller.
    kmer_sets: list[frozenset] = [
        _kmer_set(s) if len(s) >= _KMER_MIN_LEN else frozenset() for s in seqs
    ]
    postings: dict[str, list[int]] = {}
    for idx, ks in enumerate(kmer_sets):
        for km in ks:
            postings.setdefault(km, []).append(idx)

    # Pass 1 — collect candidate pairs (cheap, index-driven).
    candidate_pairs: list[tuple[int, int]] = []
    for i in range(n):
        li = int(lengths[i])
        if li == 0:
            continue
        ki = kmer_sets[i]
        max_partner_len = li * ratio
        min_partner_len = li / ratio

        if ki:
            shared: dict[int, int] = {}
            for km in ki:
                for j in postings.get(km, ()):
                    if j > i:
                        shared[j] = shared.get(j, 0) + 1
            candidates = shared.items()
        else:
            # Too short to index: fall back to the exact length window.
            candidates = [
                (j, 0) for j in range(i + 1, n)
                if min_partner_len <= lengths[j] <= max_partner_len
            ]

        for j, n_shared in candidates:
            lj = int(lengths[j])
            if lj > max_partner_len or lj < min_partner_len:
                continue
            kj = kmer_sets[j]
            if ki and kj:
                denom = min(len(ki), len(kj))
                if denom and (n_shared / denom) < _KMER_PREFILTER_CONTAINMENT:
                    continue
            candidate_pairs.append((i, j))

    n_candidates = len(candidate_pairs)

    # Pass 2 — verify candidates with the real metric. Ratcliff/Obershelp is
    # O(L^2) pure Python, so on a few thousand multi-hundred-residue proteins
    # this dominates the whole clustering step; it is also embarrassingly
    # parallel. Union-find stays sequential over the verified results, which
    # only costs a few redundant comparisons for pairs that would have been
    # short-circuited by an earlier merge.
    verdicts = _verify_pairs(seqs, candidate_pairs, min_identity, n_jobs=n_jobs)

    n_merges = 0
    n_compared = len(candidate_pairs)
    for (i, j), is_homologous in zip(candidate_pairs, verdicts):
        if is_homologous and union(i, j):
            n_merges += 1

    roots = [find(i) for i in range(n)]
    remap = {r: idx for idx, r in enumerate(sorted(set(roots)))}
    cluster_ids = np.array([remap[r] for r in roots], dtype=np.int64)

    n_clusters = len(remap)
    meta = {
        "n_proteins": n,
        "n_clusters": int(n_clusters),
        "n_merges": int(n_merges),
        "n_pairs_compared": int(n_compared),
        "n_candidate_pairs": int(n_candidates),
        "min_identity": float(min_identity),
        "method": "greedy_single_linkage",
        "metric": "ratcliff_obershelp_autojunk_off",
        # True when clustering collapsed to one-protein-per-cluster, i.e. the
        # homology split is indistinguishable from a plain protein split. This
        # is the failure mode that previously went unnoticed.
        "degenerate": bool(n_clusters == n and n > 1),
    }
    return cluster_ids, meta


_CLUSTER_CACHE: dict[tuple, tuple[np.ndarray, dict]] = {}
_CLUSTER_CACHE_MAX = 8


def _cluster_cache_key(proteins: list, min_identity: float) -> tuple:
    """Identity of a clustering request: the sequences themselves, in order."""
    import hashlib

    h = hashlib.sha256()
    for p in proteins:
        h.update(str(p.get("id", "")).encode())
        h.update(b"\x00")
        h.update(p["sequence"].encode())
        h.update(b"\x01")
    return (h.hexdigest(), round(float(min_identity), 6))


def cluster_proteins_by_homology_cached(
    proteins: list,
    min_identity: float = 0.40,
) -> tuple[np.ndarray, dict]:
    """Memoised ``cluster_proteins_by_homology``.

    Clustering is O(candidate pairs) x O(L^2) and several downstream consumers
    (fold soup, v6/v6-pro OOF, fusion, stacking, stats) each re-derive the same
    splits. Recomputing per consumer added tens of minutes per run for a result
    that is a pure function of the sequences.
    """
    key = _cluster_cache_key(proteins, min_identity)
    hit = _CLUSTER_CACHE.get(key)
    if hit is not None:
        ids, meta = hit
        return ids.copy(), dict(meta, cached=True)

    ids, meta = cluster_proteins_by_homology(proteins, min_identity=min_identity)
    if len(_CLUSTER_CACHE) >= _CLUSTER_CACHE_MAX:
        _CLUSTER_CACHE.pop(next(iter(_CLUSTER_CACHE)))
    _CLUSTER_CACHE[key] = (ids, meta)
    return ids.copy(), dict(meta, cached=False)


def get_homology_cv_splits(
    proteins: list,
    n_folds: int,
    min_identity: float = 0.40,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict]:
    """
    GroupKFold over homology clusters (not individual protein IDs).

    Falls back to per-protein grouping when there are fewer clusters than folds
    rather than raising: a small or highly redundant protein set is a reason to
    report a caveat, not to abort a multi-hour run.
    """
    from sklearn.model_selection import GroupKFold

    cluster_ids, meta = cluster_proteins_by_homology_cached(
        proteins, min_identity=min_identity
    )
    meta["split_method"] = "homology"

    n = len(proteins)
    n_clusters = int(meta["n_clusters"])
    groups = cluster_ids
    if n_clusters < n_folds:
        meta["fallback"] = "per_protein_groups"
        meta["fallback_reason"] = (
            f"{n_clusters} homology cluster(s) < {n_folds} folds; "
            "homologues may share folds"
        )
        groups = np.arange(n, dtype=np.int64)
        if n < n_folds:
            raise ValueError(
                f"Cannot build {n_folds} folds from {n} proteins"
            )

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(np.arange(n), groups=groups))
    return splits, meta
