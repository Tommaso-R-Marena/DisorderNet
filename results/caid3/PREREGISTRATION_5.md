# Pre-registration 5 — backbone handedness as a model channel

Committed **before** the chirality probe has produced a number and before any
run using the channel exists. Methodology is `METHODOLOGY.md`, unchanged.

## The claim being tested

Every structural channel DisorderNet currently reads is **mirror-invariant**.
Reflect a protein and its solvent accessibility, its contact density and
AlphaFold's pLDDT are all unchanged. The model cannot distinguish a structure
from its mirror image.

The physics is not mirror-invariant. L-amino acids build **right-handed**
alpha-helices; polyproline II, the conformation that dominates disordered and
denatured chains, is **left-handed**. Handedness is a local geometric quantity
that separates ordered from disordered backbone, and it is one the model has no
channel for.

The far-UV circular dichroism spectrum — the standard experimental assay for
disorder — is itself a chirality measurement, and the accompanying Lean
development proves it cannot supply this distinction: the PPII and
statistical-coil basis spectra are nearly identical, so the PPII/coil split in
a deconvolution is free up to the noise (`Dichroism.equal_basis_split_free`,
`near_degenerate_tolerance`). The handedness has to be computed from geometry,
which is what the CA-trace virtual torsion does.

## The gate: a decision rule fixed before the number exists

The probe (`results/caid3/chirality_probe.py`) is training-free. It computes,
on CAID3 targets with a matching AlphaFold structure, the within-protein AUC of

- the **signed** smoothed handedness, `sin(tau)`, direction fitted on MobiDB
  training proteins with every CAID3 target and sequence removed; and
- its **achiral control**, `|sin(tau)|` — identical information except for the
  sign.

**Train with the channel only if the signed channel beats its achiral control
on the within-protein axis by ≥ 0.010 on at least one of Disorder-PDB,
Disorder-NOX, Binding or Binding-IDR.** Anything smaller is torsion magnitude,
which the model can already infer from contact density and accessibility, and
would not justify a new input.

If the gate fails, the finding is recorded as a negative — *handedness is
computable, is genuinely absent from every existing channel, and carries no
usable within-protein signal for disorder* — and no run follows. That is a
publishable result about a natural idea, and it is written here so it cannot be
quietly dropped in favour of the next hypothesis.

## If the gate opens

`multitask_chiral`: the `mt_private` architecture, unchanged in every other
respect, with `structure_dim` extended by the two handedness channels
(`ca_torsion` scaled to [-1, 1], and `sin(tau)` smoothed on the rsa window).
Absence is recorded rather than imputed, as with rsa: a residue whose window is
incomplete, or a protein with no AlphaFold entry, gets a zeroed channel *and* a
zeroed availability flag. Filling in 0 alone would assert "planar backbone",
which is a conformation, not an absence.

### Primary endpoints

On **Disorder-NOX**, all 204 targets, coverage 1.00 required, paired
protein-clustered bootstrap (10,000 resamples), Holm across these two:

- **P1** — the within-protein deficit closes: `mt_chiral` within-protein AUC on
  Disorder-NOX ≥ 0.8564, matching Metapredict-v3, which currently leads that
  axis while we sit 5th at 0.8346. This is the benchmark where we top the
  pooled table on protein-level calibration and do not lead on the residue-level
  question, so it is the one where the criticism this project levels at others
  currently applies to us.
- **P2** — pooled Disorder-NOX ≥ 0.8928, i.e. `mt_windowed`'s corrected figure.
  Registered as *non-inferiority*, not superiority.

### Floors

All four, unchanged, a single breach rejecting the variant:

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9545 |
| Disorder-NOX | 0.8878 |
| Linker | 0.9193 |
| Binding | 0.7884 |

### Prediction that makes this falsifiable

Handedness is a **local** quantity: a CA virtual torsion spans four consecutive
residues. So it should move the **within-protein** axis and leave the
**between-protein** axis roughly where it is. If the pooled score rises while
the within-protein component does not, the channel is working through
protein-level calibration — the same mechanism the protein-bias run used — and
the chirality explanation is wrong even if the number improves.

Registered explicitly because it is the outcome that would be easiest to
mistake for success.

## What is not claimed

- Not that L/D stereochemistry matters here. Every residue in every CAID target
  is L; there is no stereocentre variation to exploit. The handedness at issue
  is **conformational**, a property of the backbone trace, not of the
  stereocentres. ChiralFold audits the latter and is a different instrument.
- Not that AlphaFold's handedness is the *ensemble's* handedness. A single
  predicted structure for a disordered region is a point mass, and the Lean
  development is unambiguous that this is the wrong object
  (`Ens.deterministic_iff_var_zero`). What is claimed is narrower and testable:
  the handedness AlphaFold assigns carries information about whether a region
  is disordered, in the same way its accessibility already does at rank 3.
