# Pre-registration 7 — a motif-scale read-out for the binding tasks

Committed **before** the run. Methodology `METHODOLOGY.md`. Floors are the
five-reference set from `PREREGISTRATION_6.md`.

## What the last two runs established

`mt_private` detached the binding tasks from the shared trunk. The gradient
guarantee held exactly, and the result was the opposite of the intent: against
its regime-matched control it lost **0.0555 on Binding** and **0.0262 on
Binding-IDR** while sparing the three disorder tasks about **0.006**. Binding
has 891 training proteins against disorder's 21,386, and detaching cuts the
representation in both directions at once.

So the isolation is not the lever. What the certified analysis says the binding
tasks actually need is different: on CAID3 Binding-IDR the method leading both
the pooled *and* the within-protein axis is **LIPNet**, a linear-interacting-
peptide predictor. A binding site inside a disordered region is a short linear
motif of five to fifteen residues, and this head's measured dependency span is
**global** — not the 213 residues the receptive-field formula reports, because
GroupNorm pools statistics over the whole length axis.

## What changes

`mt_motif` is `mt_control` with `--private-narrow --private-attached`:

- the binding read-outs take a private two-block stack reading the **projection**
  rather than the trunk output;
- that stack uses narrow dilations (1, 2) and **position-local channel
  normalisation**, without which GroupNorm makes any stack globally dependent
  however narrow its convolutions;
- **the gradient is not cut.** Binding's loss still shapes the shared trunk, so
  the transfer that makes multi-task training worth doing is retained.

Measured: binding dependency span **6 residues**, disorder **200**, in the same
model. Both are asserted in tests, along with the orthogonality of shape and
gradient — attaching must not widen the read-out.

Everything else is identical to `mt_control`: same five references, same fixed
validation holdout, same windowed training, same wide trunk, no chirality
channel. `mt_control` is therefore the control for this run as well as for
`mt_chiral`, and all three share a training union.

## Primary endpoints

On **Binding-IDR**, all 52 targets, coverage 1.00 required, paired
protein-clustered bootstrap (10,000 resamples), Holm across these two:

- **P1** — `mt_motif` beats `mt_control` on pooled Binding-IDR.
- **P2** — `mt_motif` within-protein AUC on Binding-IDR ≥ 0.7620, matching
  LIPNet. Registered as a stretch: we currently sit at 0.6740, 7th.

## Floors (five-reference family)

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9585 |
| Disorder-NOX | 0.8810 |
| Linker | 0.8998 |
| Binding | 0.7824 |

A single breach rejects the variant, as before.

## Falsifiable prediction

A 6-residue read-out cannot represent a protein-level property, so the gain — if
there is one — must appear in the **within-protein** component of Binding-IDR
and leave the between-protein component roughly where it is. If pooled
Binding-IDR rises while within-protein does not, the narrow read-out is not
what produced it and the motif explanation is wrong however good the number.

The reverse risk is registered too: narrowing the binding read-out removes its
access to long-range context, and if binding sites depend on distant sequence
the run should *lose* on Binding. That is the outcome that would falsify the
motif-scale premise directly, and it is the one this run is most likely to
produce.

## Stopping rule

Scored once, on all five references, under the floors above. No further binding
variant without a new registration. If `mt_motif` does not beat `mt_control` on
Binding-IDR, the negative is reported: the deficit there is not a receptive-
field-scale problem, and the remaining explanation is the label definition —
Binding-IDR's training labels arrive at 67.1% prevalence against the
benchmark's 38.6%, which is a different question, not a harder one.
