# Homology / holdout notes

## What the code does

- `colab/homology_splits.py` — greedy single-linkage clustering over pairwise
  sequence identity. Two backends, chosen by `backend="auto"`:
  - **`blastp`** (preferred; used whenever BLAST+ is on `PATH`). Identity is
    BLASTp percent identity scaled by alignment coverage over the shorter
    sequence, `pident * alignment_length / min(qlen, slen)`, with a 0.50
    coverage floor so a short high-identity local hit cannot merge two otherwise
    unrelated proteins.
  - **`python`** (fallback). `difflib.SequenceMatcher` Ratcliff/Obershelp ratio
    with `autojunk=False`. Alignment-free, not gap-aware — an approximation.
- `TrainConfig.split_method`: `"protein"` (one group per protein) or `"homology"`
- Ultra profiles default to `split_method="homology"` with `homology_min_identity≈0.4`
- `meta["backend"]` on every clustering records which backend produced it, and
  `meta["degenerate"]` flags a clustering that collapsed to one protein per
  cluster (i.e. a homology split that is indistinguishable from a protein split).

## History — this was silently a no-op before 2026-08

`difflib.SequenceMatcher` enables an `autojunk` heuristic that, for inputs of
200 elements or more, treats any element occurring in over 1% of positions as
junk. Every amino acid clears 1%, so for any protein longer than 199 residues
*all* characters were junked: two 95%-identical 400-residue sequences scored
**0.008** instead of 0.95. Nothing reached the 0.40 threshold, no clusters ever
merged, and `split_method="homology"` produced exactly the per-protein split it
exists to replace.

Measured on the DisProt release used for the publish run (`ultra` filters):

| | |
|---|---|
| Proteins after filtering | 2663 |
| Proteins ≥200 residues (invisible to the old code) | 2091 (78.5%) |
| Homology clusters found after the fix | 2239 |
| Proteins absorbed into a multi-member family | **424** |
| BLASTp all-vs-all runtime (8 threads) | 16.1 s |

So 424 proteins had a homologue that the previous code scattered across
train/validation folds. Any pre-fix result described as homology-separated
should be treated as protein-split, and re-run.

The same defect existed in `colab/caid_leakage.py`, meaning
`caid_leakage_audit.json` reported "no leakage" while structurally unable to
detect a near-duplicate of a CAID target.

## What this is not

- Not MMseqs2 / `mmseqs easy-cluster`, and not CD-HIT
- Not PDB-date or temporal holdout
- Not the CAID official BLAST training-exclusion lists (unless you add them)
- Single-linkage clustering chains: A~B and B~C puts A, B, C in one cluster even
  when A and C are dissimilar. This is deliberately conservative for holdout
  purposes (it over-merges rather than under-merges).

## Leak-free CAID publish path

When `CAID_LEAK_FREE_TRAIN=1` (publish default), `rockfish/run_disordernet.py`
audits DisProt train proteins against CAID3 reference sequences and **removes**
ID overlaps and ≥40%-identity homologs **before** CV. Artifacts:

- `checkpoints/caid_leakage_audit.json` (see its `protocol` field for the backend)
- `checkpoints/caid3_eval_report.json`
- `checkpoints/caid_challenge_report.json` (CAID3 tracks + optional CAID4 submission)

CAID4 remains blind until organizer labels (~Dec 2026); we still emit `.caid` +
timings when `CAID4_TARGETS` is provided.

## Paper language

Prefer:

> Cross-validation used homology-aware protein grouping: proteins were clustered
> by single-linkage over pairwise BLASTp identity (≥40%, percent identity scaled
> by alignment coverage over the shorter sequence, ≥50% coverage), and folds were
> assigned over clusters so that homologues never spanned the train/validation
> boundary. For CAID3 credibility-floor scoring, training proteins homologous to
> the CAID3 Disorder-PDB reference (≥40% by the same criterion) were excluded
> prior to CV. This is a conservative internal protocol, not a substitute for the
> community CAID BLAST filters.

Avoid claiming "CAID-identical homology separation" unless you run the official
CAID protocol. State the backend actually used — `caid_leakage_audit.json` and
the clustering metadata both record it, so the claim is checkable.

## Stats coupling

`run_full_statistical_validation` / `run_per_fold_paired_comparison` must receive
the same `split_method` and `homology_min_identity` used in CV so fold ΔAUC tests
align with trained folds.

More generally, every post-training consumer must partition proteins the way
training did. Use `colab.cv_splits.resolve_cv_splits`, which prefers the
`val_ids` recorded on each fold result and otherwise honours the run's own
`split_method`. Calling `get_cv_splits(proteins, n_folds)` re-derives with the
default `"protein"` method and silently disagrees with a homology-split run —
that is how fold soup came to score each checkpoint on proteins it had trained
on.
