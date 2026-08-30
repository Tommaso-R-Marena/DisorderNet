# Pre-registration 11 — train on the label that was measured, not the one that was rounded

Committed **before** the soft-label cache is built and before any number from
this run exists. Methodology `METHODOLOGY.md`. Control is `mt_control`, which
shares this run's training union and validation holdout.

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

`mt_control` shares this run's training union, its homology-clustered folds and
its fixed validation holdout. The only difference is the target on residues
MobiDB covers with two or more structures. If a second control is needed because
the training union changes when the MobiDB join is applied, it is trained in the
same window with hard labels on **exactly the same rows**, and that is the
comparison reported.

## Leak control

Unchanged and non-negotiable. The MobiDB join adds no proteins: it only alters
targets for proteins already in the union. Benchmark targets and their
≥40%-identity homologues stay out. The validation holdout is the same salted
hash of the parent sequence, so the two runs are comparable by construction.

**One new risk this creates.** The soft target is derived from the same MobiDB
per-structure calls used to measure ε and ε_pair. Those measurements are made on
CAID3 *reference* proteins; the training union excludes CAID3 targets and their
homologues, so the two do not overlap. This is asserted here so that it is
checked rather than assumed, and the check is a set intersection reported with
the run.

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

`rockfish/build_soft_labels.py` fetches the per-structure calls for the 22,839
accessions of the `pdb_missing` training universe.
