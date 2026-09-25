# Pre-registration 11 — train on the label that was measured, not the one that was rounded

Committed **before** the soft-label cache was built and before any number from
this run existed. Methodology `METHODOLOGY.md`. The control is `mt_hard`, an arm
trained in the same window on exactly the same rows with hard labels — see *The
control* below for why this replaced the pre-existing `mt_control`.

## The argument, from this project's own measurement

`CAPACITY.md` measures something about the *reference* that is equally true of
the *training set*: **8.01% of evidenced residues are context-dependent** —
missing in some deposited structures of a protein and observed in others. The
label is not a function of the sequence. A region ordered in one crystal form
and not another is a genuinely intermediate case, and the training data records
it as a hard 0 or a hard 1 depending on which structure was consulted.

Every disorder predictor trained on PDB-derived labels therefore fits ~8% of its
supervision to a coin flip whose outcome is an artefact of deposition history.
That is not a nuisance to be regularised away; it is a quantity MobiDB publishes
per structure, and it can be put into the target instead of into the noise.

**The change.** For a residue covered by `k ≥ 2` deposited structures of which
`m` call it missing, the target becomes `m / k` rather than `round(m / k)`.
Residues with `k < 2`, or with no MobiDB record, keep their existing hard label.
The loss is unchanged — binary cross-entropy already accepts a soft target — so
this is a change to the data and not to the objective, and nothing else about
the run differs from the control.

**Why this project can make this change and others cannot.** It requires the
per-structure missing-residue calls, which is the same measurement that produced
`ε = 0.0801` and `ε_pair`. Nobody who has not measured the annotation error rate
can construct this target.

## What is predicted, and what would falsify it

Soft targets reduce the label noise the model fits. The prediction is therefore
about **generalisation**, not about fitting the benchmark better:

1. The gain should appear on data the model has not seen, and should be larger
   on the temporal holdout than on CAID3.
2. The gain should be **concentrated on context-dependent residues**. If the
   improvement is uniform across residues, the stated mechanism is wrong and
   something else produced it — that is a falsification, not a caveat, and it is
   reported as one.
3. The gain should appear on the **within-protein** axis, since that is what a
   per-residue supervision change acts on. A pooled-AUC gain with no
   within-protein gain would mean the change bought protein-level calibration,
   which it has no mechanism to buy.

## Primary endpoints

Fixed here, and for the first time in this project chosen on an axis the
capacity result says is resolvable. Pooled AUC on CAID3 is **not** a primary
endpoint: at ε = 0.0801 the benchmark can order 7 methods on marginal scores and
18 on a paired comparison (`RESULT_BOUND_TIGHTNESS.md`), and registering an
endpoint the benchmark cannot decide is what this project did five times before
and is not doing again.

- **P1** — `mt_soft` beats `mt_control` on **within-protein AUC over the 186
  temporally held-out chains**, per-chain paired Wilcoxon. Independent data,
  released after both training caches were built, and not bounded by CAID3's
  capacity.
- **P2** — `mt_soft` beats `mt_control` on **CAID3 Disorder-PDB per-target
  within-protein AUC**, paired Wilcoxon over the 233 two-class targets.

Holm across exactly these two. Two-sided, α = 0.05. Bootstrap intervals resample
proteins, 10,000 resamples, Davison–Hinkley.

## Secondary, declared in advance

- The same comparison on Disorder-NOX, Binding, Binding-IDR and Linker
  within-protein, Holm within that family of four.
- Pooled AUC on all five references, reported with the capacity beside it and
  **not** interpreted as a ranking claim.
- The mechanism test: the per-residue AUC gain restricted to context-dependent
  residues against the gain on the rest. This is the falsification in point 2.
- Operating cost at α = 0.10 and 0.05, since a target that expresses uncertainty
  should move the threshold a user needs.

## Floors

From `PREREGISTRATION_6.md`, unchanged. The variant is rejected if any fails.

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9585 |
| Disorder-NOX | 0.8810 |
| Linker | 0.8998 |
| Binding | 0.7824 |

## The control, and what makes it matched

The registered comparison is `mt_soft` against `mt_hard`: two arms submitted in
the same window, on the same rows, with the same backbone, folds, epochs,
structure dimension, task set, references and holdout salt. The only difference
is the target on residues MobiDB covers with two or more structures.

`mt_control` was the originally named control and is retained as a secondary
reference point, but it was trained in a different window. A difference against
it could be the window rather than the labels, and this project has already
learned once (PREREGISTRATION_5/6, chirality) how much a regime-matched control
changes the reading. Paying for a second arm is cheaper than an ambiguous
result.

## Leak control

Unchanged and non-negotiable. The MobiDB join adds no proteins: it only alters
targets for proteins already in the union. Benchmark targets and their
≥40%-identity homologues stay out. The validation holdout is the same salted
hash of the parent sequence, so the two runs are comparable by construction.

**One new risk this creates, and the check that found the claim was wrong.**
The soft target is derived from the same MobiDB per-structure calls used to
measure ε and ε_pair, which are measured on CAID3 *reference* proteins. An
earlier draft of this section asserted that the cache and the benchmark do not
overlap. **They do**: 141 of the 319 CAID3 reference accessions appear in the
soft-label cache, because the cache is keyed by accession over the whole
`pdb_missing` universe and is built before any filtering.

That is not a leak, and the reason is an ordering that was verified rather than
assumed. In `train_multitask.py` the pdb_missing rows are built (line 852) and
merged (858) *before* `drop_caid_targets` runs (894), which then removes CAID3
targets and their ≥40%-identity homologues from the merged union, before
`reserve_validation_holdout` (916). The cache is a lookup keyed by accession; a
soft target for a benchmark accession is never used because the row carrying it
does not survive to training.

So the invariant to state is the post-filter one, and the check to run is on the
union after `drop_caid_targets`, not on the cache. The run reports both: the
cache intersection (141, expected and harmless) and the post-filter intersection
(must be 0). The first draft would have reported a passing check on the wrong
set.

**And then the implementation made the same class of mistake, one level down.**
The check as first written intersected the reference ids with the cache keys
directly. CAID reference FASTAs are keyed by DisProt id (`DP02732`); the cache
is keyed by UniProt accession (`A0A003`). The two sets are disjoint for every
possible input, so the check printed

    soft-label leak check: 0 cache/reference accessions before the filter, 0 surviving it

in the run of 31214999 — and *would have printed the same line* on a run whose
training set was entirely benchmark targets. It is reported here rather than
quietly repaired because the "0 before the filter" is visibly inconsistent with
the 141 recorded two paragraphs above, and that inconsistency is the only reason
it was caught.

The check now maps the reference through DisProt, which carries both ids,
compares sequences as well as accessions (sequence belongs to no namespace, so
it covers what the id join cannot resolve), and reports whether it resolved any
benchmark accession at all — a run whose check resolves nothing now aborts
instead of passing. `tests/test_soft_labels.py::TestTheLeakCheckCanActuallyFail`
pins all four properties, and one of its cases is a surviving benchmark row with
a soft target that the old logic scored as clean.

**What this does and does not change about the run in flight.** Nothing about
the training. The leak *prevention* is `drop_caid_targets`, which is unchanged,
separately regression-tested, and removed 2,039 of 27,412 proteins (530 exact
id/sequence hits) in this very run. What was missing was the *evidence* that it
left no residual, so the evidence is produced separately by
`rockfish/verify_soft_label_leak.py`, which replays the same inputs through the
same filter on CPU and reports both numbers. Its result is recorded below, and
the arms are not scored until it passes. Restarting nine hours of GPU time to
re-run a verification that can be run beside it would have been the wrong trade.

## Stopping rule

Scored once, under the floors above. No weight sweep, no threshold on how much
soft supervision to use, no second cut of the cache. If P1 fails, the negative is
reported: per-residue supervision noise measured from deposition history is not
what limits this model.

## What this cannot show

It cannot show that the benchmark can see the improvement. At the measured
annotation error rate CAID3 orders 7 methods marginally and 18 paired, so a
pooled-AUC difference between `mt_soft` and `mt_control` is exactly the kind of
comparison the capacity result says is undecidable. That is why P1 is on
independent data and P2 is on the calibration-invariant axis, and why a pooled
gain — if one appears — will be reported as a number and not as a result.

## Status

Registered before the cache was built. The shared `sfried3` scratch quota was
exhausted when this was written; 10.3 GB was freed by removing two regenerable
torch caches from the DisorderNet project directory (`torch_home`,
`torch_cache_from_wrong_repo`), which unblocked writes. Nothing else was
removed, and no run output, checkpoint or benchmark artefact was touched.

`rockfish/build_soft_labels.py` fetched the per-structure calls for the 22,839
accessions of the `pdb_missing` training universe: **18,125 proteins,
6,380,212 residues** with two or more covering structures, of which **14.64%
are strictly between 0 and 1** (mean 0.395). That the training universe is more
ambiguous than the benchmark reference (14.6% against 8.0%) is expected — it is
larger and less curated — and it is the quantity this run acts on.

**Launched as a matched pair**, rather than against the pre-existing
`mt_control`, so that the only difference between the two arms is the target:

| arm | job | workdir |
|---|---:|---|
| `mt_soft` (`--soft-labels`) | 31214999 | `multitask_soft` |
| `mt_hard` (control) | 31215000 | `multitask_hard` |

The first submission of this pair (31210553 / 31210554) died 33 s in: the
`--soft-labels` flag was documented and threaded through `train_multitask.py`
but never registered in `argparse`, so both arms rejected it as an unknown
argument. The flag is now registered and the registration is asserted at import,
and `--help` was checked on both the local and the cluster copy before
resubmitting. The soft arm's log confirms the target was actually used —
`17,400 proteins, 6,053,995 residues, 725,999 strictly between 0 and 1` — which
is the line whose absence would otherwise have let a silently-hard run be
reported as the soft arm.

Identical backbone, folds, epochs, structure dimension, task set, references and
holdout salt; submitted in the same window. `mt_control` is retained as a
secondary reference point but the registered comparison is `mt_soft` against
`mt_hard`.
