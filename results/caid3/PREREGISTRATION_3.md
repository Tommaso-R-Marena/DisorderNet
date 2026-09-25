# Pre-registration 3 — the protein-level bias term

Written and committed **before** the run starts and before any of its results
exist. Methodology is `METHODOLOGY.md`, unchanged.

## The theory, stated so it can be wrong

A pooled AUC is the probability a random positive residue outranks a random
negative one. Every such pair sits inside one protein or across two, so:

    AUC_pooled = w_within * AUC_within + w_between * AUC_between

an identity, not an approximation, verified against brute-force pair counting.

Measured on CAID3, `w_within` runs from **0.0029** (Disorder-PDB) to **0.0326**
(Linker). **Between 96.7% and 99.7% of what the benchmark scores is a
between-protein judgement.** The within-protein question — which residues of
this chain carry the label — is at most 3% of it.

That reframes our two weakest results:

| benchmark | our within | leader within | our between | leader between |
|---|---:|---:|---:|---:|
| Binding | **0.8683** | 0.8049 | 0.7734 | 0.7755 |
| Binding-IDR | 0.7004 | 0.6958 | **0.4982** | **0.6400** |

On Binding we are decisively better at the biological question and still lose
the pooled score. On Binding-IDR our within-protein ability matches the leader
and the entire 0.14 deficit is between-protein. A per-residue model with a
213-residue receptive field has no mechanism for a 1,000-residue chain's global
properties, so this is a missing degree of freedom rather than a tuning failure.

## What changes, and only this

A learned per-protein bias — mean and max pooling of the trunk, a linear map to
one scalar per task, added to every residue of that protein — on the two binding
tasks. 68 parameters. Pooling is detached; the term initialises to zero.

Everything else is identical to `mt_windowed`, which is therefore the exact
control: same tasks, frozen ESM-2 650M over layers 21–32, `structure_dim=24`,
wide receptive field, 5 folds, 8 epochs, identity 0.40, long proteins windowed.

## Primary endpoints

On **Binding-IDR**, all 52 targets, coverage 1.00 required, paired
protein-clustered bootstrap (10,000 resamples, two-sided), Holm across **these
two alone**:

- **P1 — pooled AUC above the `mt_windowed` control (0.5007).**
  Does it work at all.
- **P2 — `AUC_between` above the control's 0.4982.**
  Does it work *for the stated reason*. This is the mechanism, and it is the
  more informative of the two: P1 could pass by accident of retraining, P2
  cannot.

## The prediction that makes this falsifiable

A per-protein constant cannot change within-protein ranking — that is
arithmetic. The trunk still retrains, so `AUC_within` is not frozen, but the
theory predicts:

> **|Δ AUC_within| < |Δ AUC_between|** on Binding-IDR.

If the gain arrives through `AUC_within` instead, the term did something other
than what it was built to do, and the explanation is wrong even if the number
improves. That outcome will be reported as a failed mechanism regardless of what
happens to the leaderboard position.

## Non-inferiority — all four placements

Unchanged from `PREREGISTRATION_2.md`, and a single breach rejects the variant:

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9545 |
| Disorder-NOX | 0.8878 |
| Linker | 0.9193 |
| Binding | 0.7884 |

## Secondary — exploratory

Binding's pooled AUC and its within/between split; the same decomposition on the
three disorder tasks; the ensemble including this checkpoint. Corrected within
their own family, reported whether or not they improve.

## Stopping rule

One run, one evaluation, reported once. The single exception, as before: an
evaluation executed with a misconfigured architecture is void and repeated with
the identical analysis. That has happened once, and naming the condition in
advance is what separates a correction from a second attempt.

## Disclosure

The decomposition was computed on CAID3, so this architecture was designed after
inspecting the benchmark. The mechanism is general — it follows from how a
pooled AUC counts pairs, not from anything specific to these 52 targets — but
the choice to build it was informed by them, and that is recorded here rather
than argued later.
