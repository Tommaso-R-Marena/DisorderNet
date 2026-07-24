# Homology / holdout notes

## What the code does

- `colab/homology_splits.py` — length-binned greedy clustering with `difflib.SequenceMatcher` identity
- `TrainConfig.split_method`: `"protein"` (random protein groups) or `"homology"`
- Ultra profiles default to `split_method="homology"` with `homology_min_identity≈0.4`

## What this is not

- Not MMseqs2 / mmseqs easy-cluster
- Not PDB-date or temporal holdout
- Not CAID3 official BLAST training exclusion lists (unless you add them)

## Leak-free CAID publish path

When `CAID_LEAK_FREE_TRAIN=1` (publish default), `rockfish/run_disordernet.py` audits
DisProt train proteins against CAID3 reference sequences and **removes** ID overlaps
and ≥40% SequenceMatcher homologs **before** CV. Artifacts:

- `checkpoints/caid_leakage_audit.json`
- `checkpoints/caid3_eval_report.json`
- `checkpoints/caid_challenge_report.json` (CAID3 tracks + optional CAID4 submission)

CAID4 remains blind until organizer labels (~Dec 2026); we still emit `.caid` + timings
when `CAID4_TARGETS` is provided.

## Paper language

Prefer:

> Cross-validation used homology-aware protein grouping based on pairwise sequence identity (≥40% within length bins). For CAID3 credibility-floor scoring, training proteins homologous to the CAID3 Disorder-PDB reference (≥40% identity) were excluded prior to CV. This is a conservative internal protocol, not a substitute for community CAID BLAST filters.

Avoid claiming “CAID-identical homology separation” unless you run the official CAID protocol.

## Stats coupling

`run_full_statistical_validation` / `run_per_fold_paired_comparison` must receive the same `split_method` and `homology_min_identity` used in CV so fold ΔAUC tests align with trained folds.
