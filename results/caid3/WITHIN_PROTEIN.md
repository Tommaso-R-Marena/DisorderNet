# What CAID3's headline number ranks methods on

**Status: complete for the 115 published entrants. DisorderNet's own position
is being computed (job 29928404) and this file will carry it when it lands.**

CAID scores by pooling every residue of every protein into one AUC. Every
positive–negative pair in that statistic is either *within* one protein or
*between* two, and the pooled AUC is exactly their pair-weighted average:

```
AUC_pooled = w_within · AUC_within + w_between · AUC_between
```

For n proteins of comparable size only about 1/n of pairs fall inside a
protein. On CAID3 that is measured, not assumed:

| benchmark | targets scored | w_within | 1/n |
|---|---:|---:|---:|
| Disorder-PDB | 233 | 0.5091% | 0.4292% |
| Disorder-NOX | 178 | 0.6747% | 0.5618% |
| Binding | 49 | 2.5518% | 2.0408% |
| Binding-IDR | 42 | 2.8913% | 2.3810% |
| Linker | 31 | 2.8826% | 3.2258% |

So between 97.1% and 99.5% of the metric is the **between-protein** question —
*does this chain carry more disorder than that one* — and the remainder is the
**within-protein** question, *which residues of this chain are disordered*.

**That much is arithmetic and proves nothing.** "99% of the pairs are
between-protein" is close to a restatement of "there are many proteins". The
question worth asking is empirical: **does it change who wins?**

## It does, on three of five benchmarks

| benchmark | CAID3 winner | its within-protein rank | best within-protein |
|---|---|---:|---|
| Disorder-PDB | PUNCH2 | **3** / 57 | AlphaFold-pLDDT (0.9392) |
| Disorder-NOX | flDPnn3a | **25** / 57 | Metapredict-v3 (0.8564) |
| Binding | UdonPred-combined | **21** / 69 | AlphaFold3-binding (0.7998) |
| Binding-IDR | LIPNet | **1** / 69 | LIPNet (0.7620) |
| Linker | LINKER-Pred2 | **25** / 90 | APOD (0.8344) |

On Disorder-NOX, Binding and Linker the method the challenge declares best is
21st to 25th at telling which residues of a chain are disordered. On
Disorder-PDB and Binding-IDR the metric behaves: the winner is 3rd and 1st.

The three Linker specialists sweep the pooled table — LINKER-Pred2, LINKER-Pred
and LINKER-Pred-Lite at #1, #2, #3 — and sit at **#25, #26 and #29** on the
within-protein axis. They are extremely good at ranking whole chains by linker
content and unremarkable at locating linkers inside one.

## Rank agreement, with a protein-clustered interval

| benchmark | Spearman(pooled, within) | 95% CI |
|---|---:|---|
| Disorder-PDB | +0.964 | [+0.910, +0.981] |
| Disorder-NOX | +0.849 | [+0.743, +0.933] |
| Binding | +0.916 | [+0.425, +0.948] |
| Binding-IDR | +0.914 | [+0.731, +0.914] |
| Linker | +0.865 | [+0.642, +0.954] |

**This is the honest counterweight and it belongs in the headline, not a
footnote.** The two orderings agree strongly overall. A method that predicts
residues well is usually calibrated across chains too, so the metric's 1:200
weighting does not wreck the ranking. The decomposition does not overturn
CAID3.

What it does is locate the disagreement. It is concentrated in specific
methods, and the movement is large where it happens: −40 places for
UdonPred-combined on Linker, −26 for flDPlr on Disorder-NOX, +32 for
AlphaFold-pLDDT on Linker.

## The training-free predictor nobody ranks first

**AlphaFold-pLDDT** — a confidence score, not a disorder predictor, requiring
no training on disorder at all — is:

- within-protein **#1 of 57** on Disorder-PDB, while ranking **#9** pooled;
- within-protein **#2 of 90** on Linker, while ranking **#34** pooled;
- within-protein **#7** on Disorder-NOX, while ranking **#24** pooled.

Its weakness is entirely protein-level calibration. Given a chain it is the
best residue-level discriminator in the field on Disorder-PDB; asked which of
two chains is more disordered, it is mediocre, and the pooled metric is almost
entirely that second question.

## What the between-protein axis is not

If the between-protein question were trivial the finding would be deflationary
rather than interesting, so it was priced. A training-free protein-level
descriptor — mean Kyte–Doolittle hydropathy, one number per chain, negated,
broadcast to every residue — has within-protein AUC exactly 0.5 by
construction, so whatever it scores is purely between-protein:

| benchmark | hydropathy AUC_between | best entrant's AUC_between |
|---|---:|---:|
| Disorder-PDB | 0.6276 | 0.9410 |
| Disorder-NOX | 0.6183 | 0.8350 |
| Binding-IDR | 0.4348 | 0.7473 |
| Linker | 0.5844 | 0.8784 |

**The between-protein axis is not a trivial composition effect.** Hydropathy
alone gets nowhere near the leaders. Ranking chains by disorder content is a
real ability that the field is genuinely good at — it is simply not the ability
the benchmark is usually described as measuring.

## Method

- Only methods that predicted **every** reference target, at the reference's
  length, with finite scores. Declining hard targets raises a within-protein
  AUC, so coverage is removed as a variable rather than adjusted for. This is
  also the population CAID reports as full-coverage.
- Only targets carrying **both classes** among evaluated residues. A
  single-class target contributes no within-protein pair, so including it would
  only make the two rankings look more alike than they are.
- `AUC_between` is recovered from the identity rather than by enumerating
  cross-protein pairs, which would be quadratic in residues. Exact, not
  approximate.
- The Spearman interval is a **protein-clustered** bootstrap: proteins are
  resampled, and every method is re-scored on the same redrawn proteins.
  Residues within a chain are anything but independent and a residue bootstrap
  would return an interval far too narrow to mean anything. 200 resamples —
  the interval is set by how many proteins there are, not by how many times
  they are redrawn.
- `AUC_within` is pair-weighted, because the identity requires it. Pair
  weighting lets large chains carry the number, so the one-protein-one-vote
  mean is reported beside it and any claim here has to survive both.

## Caveats that a reader should not have to find

- **The ranks in this document are recomputed on the restricted subset and are
  not CAID's published ranks.** Restricting to full-coverage methods and
  two-class targets changes the pooled numbers too; both columns come from the
  same subset, so the comparison between them is fair, but neither is the
  official table.
- Five benchmarks is five tests. No multiplicity correction is applied to the
  Spearman values because none is claimed as a hypothesis test; they are
  descriptive.
- This says nothing about which ability *matters*. A biologist asking "is this
  protein disordered" wants the between-protein axis. A biologist asking "where
  do I truncate this construct" wants the within-protein one. The point is that
  one number is reported as though it answered both.

## Reproduction

```bash
export ANALYSIS_SCRIPT=results/caid3/within_protein_leaderboard.py
export ANALYSIS_ENV="WPL_BENCHMARK=caid3 WPL_OUT=/path/out.json"
sbatch rockfish/slurm/analysis_cpu.sbatch
```

Raw output: `within_protein_caid3.json`. Every number here comes from the
published `.caid` files of the CAID3 entrants and the official references —
nothing in this document depends on DisorderNet at all.
