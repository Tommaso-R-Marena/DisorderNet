# Pre-registration 6 — handedness, and floors that match the training regime

Committed **before** either run starts. Methodology is `METHODOLOGY.md`.

## Two corrections to how variants have been judged

**1. The floors were cross-regime.** `NON_INFERIORITY_FLOORS` came from
`mt_windowed`, filtered against one CAID reference. Every variant since
`mt_publication` has been filtered against five — CAID3 Disorder-PDB and all
four CAID2 references — which removes more training data and is the only reason
those checkpoints can be scored on CAID2 at all. Charging them the
one-reference floor charges them for data they were required to give up.

From here the five-reference family is judged against five-reference floors:
the best five-reference result per task, less the same 0.005 margin.

| benchmark | best 5-ref (run) | floor |
|---|---:|---:|
| Disorder-PDB | 0.9635 (`mt_pbias`) | **0.9585** |
| Disorder-NOX | 0.8860 (`mt_publication`) | **0.8810** |
| Linker | 0.9048 (`mt_private`) | **0.8998** |
| Binding | 0.7874 (`mt_pbias`) | **0.7824** |

The one-reference floors stay in the file as history and still govern
one-reference runs. Nothing already reported is re-scored.

**2. Runs were not comparable to each other.** Each variant changed the
architecture *and* was compared to a checkpoint trained on a different union.
Both runs registered here reserve the same fixed 5% validation set
(`in_validation_holdout`, keyed on the protein's own sequence, homologues
pulled with it), so for the first time two architectures are measured on
identical held-out chains.

## The two runs

Identical in every respect except the channel under test, both dual-filtered,
both with the holdout, both from the `mt_pbias` architecture — protein bias,
disorder-conditioned binding, windowed training, wide dilations, **no private
trunk**, which the last run showed costs the binding tasks far more than it
protects the disorder ones.

- **`mt_chiral`** — `--chiral`. Two extra structural channels: signed backbone
  handedness `sin(tau)` and its availability flag.
- **`mt_control`** — the same, without `--chiral`. Not an old checkpoint reused
  as a control: the holdout changes the training union, so the comparison needs
  a control that lost the same 5%.

## Why handedness, and the gate it already cleared

Every structural channel the model reads is mirror-invariant. Reflect a protein
and its accessibility, contact density and pLDDT are unchanged. The physics is
not: L-amino acids build right-handed alpha-helices, and polyproline II, which
dominates disordered chains, is left-handed.

`PREREGISTRATION_5.md` set the gate before the probe ran — train only if signed
handedness beats its own achiral control, `|sin(tau)|`, on the within-protein
axis by ≥ 0.010 on some task. Measured training-free on CAID3, direction fitted
on 800 MobiDB proteins with every CAID3 target and sequence excluded:

| benchmark | handedness (within) | achiral control | difference |
|---|---:|---:|---:|
| Disorder-PDB | 0.6435 | 0.4180 | **+0.2256** |
| Disorder-NOX | 0.6494 | 0.3966 | **+0.2529** |
| Linker | 0.6105 | 0.5912 | +0.0193 |
| Binding | 0.5169 | 0.5239 | −0.0070 |
| Binding-IDR | 0.3766 | 0.5945 | −0.2180 |

Cleared by twenty-two times the threshold on Disorder-NOX. The achiral control
sits *below chance* on both disorder tasks, so the entire signal is in the
sign; torsion magnitude alone is worse than nothing. The fitted direction is
the one polymer physics predicts — disordered residues less right-handed (mean
0.074) than ordered (0.211).

Binding-IDR at 0.3766 under a sign fitted on *disorder* labels is 0.6234 with
the sign reversed, which is what coupled folding and binding predicts: the
motifs inside disordered regions frequently become helices. Not tested here.
Recorded so that noticing it later cannot be presented as a prediction.

## Primary endpoints

On **Disorder-NOX**, all 204 targets, coverage 1.00 required, paired
protein-clustered bootstrap (10,000 resamples), Holm across these two:

- **P1** — `mt_chiral` beats `mt_control` on pooled Disorder-NOX. Registered
  as the direct A/B the gate motivates, since the probe's advantage is
  concentrated there.
- **P2** — `mt_chiral` within-protein AUC on Disorder-NOX ≥ 0.8564, matching
  Metapredict-v3, which leads that axis while we sit 5th at 0.8346.

## Falsifiable prediction

A CA virtual torsion spans four residues, so handedness must move the
**within-protein** axis. If pooled Disorder-NOX rises while the within-protein
component does not, the channel is working through protein-level calibration —
the protein-bias mechanism — and the chirality explanation is wrong however
good the number looks.

## Stopping rule

Both runs are scored once, on all five references, under the floors above. No
third run is launched on the strength of these numbers without a further
registration. If `mt_chiral` does not beat `mt_control`, the negative is
reported: handedness is real, absent from every existing channel, measurable
training-free at 0.649 within-protein on Disorder-NOX, **and not usable by this
architecture** — which is a finding about the architecture worth having.
