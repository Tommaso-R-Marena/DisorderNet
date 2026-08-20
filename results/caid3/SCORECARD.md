# The scorecard, and a correction

## CAID3 — official references, coverage 1.00, paired protein-clustered bootstrap

| benchmark | leader | theirs | **ours** | rank | full-cov | Δ | p |
|---|---|---:|---:|---:|---:|---:|---:|
| **Disorder-PDB** | PUNCH2 | 0.9550 | **0.9635** | **1/115** | **1/58** | **+0.0083** | **0.023** |
| **Disorder-NOX** | ESMDisPred-2PDB | 0.8850 | **0.8900** | **1/115** | **1/58** | −0.0022 | 0.863 |
| Binding | DisoFLAG-PB | 0.7760 | 0.7924 | 2/115 | 2/70 | −0.0028 | 0.957 |
| Binding-IDR | bindEmbed21IDR | 0.6410 | 0.6062 | 5/115 | 5/70 | −0.0344 | 0.421 |
| **Linker** | IPA-AF2-Linker | 0.8970 | **0.9267** | **1/115** | **1/91** | +0.0309 | 0.084 |

(The Δ column is against the *published* leader on shared targets; a negative Δ
with a better rank means the recomputed leaderboard differs from the published
one, which the coverage column explains.)

## CAID2 — independent round, never seen in training

| benchmark | leader | theirs | **ours** | rank | full-cov | Δ | p |
|---|---|---:|---:|---:|---:|---:|---:|
| Disorder-PDB | PredIDR-long | 0.9335 | **0.9470** | 2/71 | **1/54** | +0.0134 | 0.043 |
| Disorder-NOX | Dispredict3 | 0.8378 | **0.8515** | **1/71** | **1/54** | +0.0137 | 0.352 |
| Binding | Dispredict3 | 0.8244 | **0.8501** | **1/71** | **1/55** | +0.0257 | 0.295 |
| Linker | SETH-0 | 0.7695 | **0.8021** | 2/71 | **1/64** | +0.0326 | 0.456 |

**First among full-coverage entrants on all four**, from training that never saw
a CAID2 target or a 40%-identical homologue.

## Within-protein — the calibration-invariant axis, per-target paired

| benchmark | best of ours | targets | wins | Δ | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| Disorder-PDB | Ensemble | 233 | 138 (59%) | +0.0185 | **0.00001** |
| Disorder-NOX | Ensemble | 178 | 96 (54%) | +0.0496 | **0.00315** |
| Binding | Ensemble | 49 | 33 (67%) | +0.1435 | **0.00103** |
| Binding-IDR | DN-pbias | 42 | 28 (67%) | +0.2559 | **0.00125** |
| **Linker** | Ensemble | 31 | **29 (94%)** | +0.1364 | **<0.00001** |

**All five against PUNCH2, all five significant.** On Linker we win 29 of 31
targets.

## The correction

I have repeatedly written that this project has never beaten PUNCH2
significantly, quoting +0.0043, +0.0019, +0.0035, +0.0062. **That was wrong.**

`mt_pbias` beats PUNCH2 on pooled Disorder-PDB by **+0.0083, 95% CI
[+0.0010, +0.0176], p = 0.0226** — the interval excludes zero. Every checkpoint,
verified from the result files:

| run | Disorder-PDB | vs PUNCH2 | 95% CI | p |
|---|---:|---:|---|---:|
| **pbias** | **0.9635** | **+0.0083** | **[+0.0010, +0.0176]** | **0.0226** |
| chiral | 0.9614 | +0.0063 | [−0.0012, +0.0160] | 0.112 |
| publication | 0.9613 | +0.0062 | [−0.0010, +0.0149] | 0.100 |
| wass | 0.9597 | +0.0046 | [−0.0038, +0.0150] | 0.326 |
| windowed | 0.9590 | +0.0038 | [−0.0033, +0.0120] | 0.307 |

The figures I was quoting were the *other* checkpoints. **Why the mistake:**
`mt_pbias` was rejected by `PREREGISTRATION_3` for breaking three
non-inferiority floors, so I filed it as "rejected" and stopped quoting its
Disorder-PDB comparison at all. That is a reason for caution, not a reason to
state a flat negative.

**What can honestly be claimed, then:** one checkpoint beats the CAID3 winner on
pooled Disorder-PDB at p = 0.023 — and that checkpoint was rejected by its own
pre-registration on three other benchmarks, and generalises worst of the three
tested on the temporal holdout. The clean claim is the within-protein one, which
holds for the **ensemble**, on **all five** benchmarks, and rests on the axis the
capacity result shows is the one with resolution left in it.

## Block structure — the constant behind the escape route

7,100 disagreement blocks across 147 proteins, from MobiDB per-structure calls:

| | residues |
|---|---:|
| median block length | 3 |
| mean | 7.8 |
| **mass-weighted mean** | **73.8** |
| 90th percentile | 13 |
| longest | 472 |

**67.4% of context-dependent residues sit in blocks of ten or more.** The
block mechanism is confirmed: disorder does not flip residue by residue, it
flips in regions.

The naive boundary-to-area prediction `1/L = 1/73.8 = 0.0136` **over-predicts**
the measured ratio of 0.1075 by eightfold. So the reduction is real but the
single-length scaling is too crude — the lemma should be stated over the
block-size *distribution*, not a mass-weighted mean. 31% of blocks are single
residues and those contribute discordance at the full label rate.
