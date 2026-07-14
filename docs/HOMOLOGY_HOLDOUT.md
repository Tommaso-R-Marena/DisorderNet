# Homology / holdout notes

## What the code does

- `colab/homology_splits.py` — length-binned greedy clustering with `difflib.SequenceMatcher` identity
- `TrainConfig.split_method`: `"protein"` (random protein groups) or `"homology"`
- Ultra profiles default to `split_method="homology"` with `homology_min_identity≈0.4`

## What this is not

- Not MMseqs2 / mmseqs easy-cluster
- Not PDB-date or temporal holdout
- Not CAID3 official training exclusion lists (unless you add them)

## Paper language

Prefer:

> Cross-validation used homology-aware protein grouping based on pairwise sequence identity (≥40% within length bins). This is a conservative internal protocol, not a substitute for community CAID homology filters.

Avoid claiming “CAID-identical homology separation” unless you run the official CAID protocol.

## Stats coupling

`run_full_statistical_validation` / `run_per_fold_paired_comparison` must receive the same `split_method` and `homology_min_identity` used in CV so fold ΔAUC tests align with trained folds.
