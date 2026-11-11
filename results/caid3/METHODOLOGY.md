# Evaluation methodology — fixed

One methodology, applied to every benchmark, every run, reported in full every
time. It is written down because the temptation in a leaderboard project is to
pick, after the fact, whichever framing is kindest — and this project has
already produced one retraction and one set of p-values that did not survive
their own family.

## 1. Predict every target. Always.

CAID pools every residue of every protein into one AUC over whatever targets a
method returned. A method that declines the ones it finds hard is therefore
scored on an easier benchmark, in the same column as one that answered
everywhere. Measured, using a fully-covering method as the yardstick:

| benchmark    | method          | declined | median length | difficulty gap |
|--------------|-----------------|---------:|--------------:|---------------:|
| disorder_nox | ESMDisPred-2PDB | 23 / 204 | 1292          | +0.118 |
| linker       | ESMDisPred-2PDB |   6 / 31 | 1214          | +0.125 |
| disorder_pdb | SPOT-Disorder2  | 21 / 319 | 1684          | +0.035 |

Median length across all CAID3 targets is 344. Long proteins are harder and they
are what gets declined.

We predict every target on every benchmark. The evaluator aborts rather than
scoring a subset. Reported coverage is 1.00 or the run is void.

## 2. Report both fields, never one

**Rank among all scored entrants** is CAID's own accounting. It is the number
that would appear on the leaderboard, and it embeds the advantage above.

**Rank among full-coverage entrants** is the equal-footing field: everyone
scored on the same targets we were.

Both, always. Reporting only the first understates us. Reporting only the second
is cherry-picking — a method that declines targets is not disqualified, it is
just scored on an easier set, and excluding it entirely answers a different
question than penalising it. Neither number alone is the truth; the pair is.

## 3. Settle it with a paired test on shared targets

The comparison that actually decides anything: our scores and a competitor's
scores on the targets **both** predicted, resampling **proteins** rather than
residues, since residues within a chain are heavily correlated and a
residue-level interval is far too narrow.

Two-sided, 10,000 resamples for confirmatory runs. p-values use both tail
boundaries and the Davison–Hinkley +1, so a perfect null returns 1.0 rather
than 0, and p is never reported below 1/(B+1).

## 4. Declare the family before looking

Every p-value belongs to a family fixed in advance, and Holm–Bonferroni is
applied within it. Holm rather than Bonferroni: uniformly more powerful, and it
assumes no independence, which these comparisons badly lack — they share our
predictions and share targets.

The primary family is fixed in `PREREGISTRATION.md` and encoded in
`rockfish/eval_caid3_official.py` so it cannot be redefined after the numbers
land. Everything outside it is secondary, corrected within its own family, and
labelled exploratory.

## 5. Beat the training-free baseline, or the training did nothing

AlphaFold-rsa ranks 3rd on Disorder-PDB at 0.9498 with no training at all, and
is not statistically separable from the first-placed method. Every run is
compared against it. A model that tops the table while failing to beat rsa has
demonstrated nothing about learning.

## 6. State what a number is not

- A **rank** is a point estimate and needs no p-value. "Tops the table" and
  "proven better" are different claims and are written differently.
- A **fused** row that consumes another entrant's published predictions is not
  self-contained. Our own rsa implementation scores 0.938 where CAID's scores
  0.950, so a standalone submission would land lower.
- A **cross-validation** number is not a benchmark number. Two runs whose
  validation sets differ — for instance one that trains on long proteins and one
  that discards them — have incomparable CVs even at identical settings.
- Choices **informed by inspecting the benchmark** are recorded as such, even
  when the model itself predates them.

## 7. Verify before believing

Every reported result is checked before it is stated:

- composition of each reference against counts CAID's dataset API serves
  separately;
- md5 of each reference against a fresh download;
- the published leader of each challenge reproduced through our own pipeline —
  if PUNCH2 stops scoring 0.9552, nothing computed afterwards is trustworthy;
- the leak filter covering every benchmark target;
- the headline recomputed from archived submissions by a separate code path;
- ranks recomputed by `verify_ranks.py`, printing the insertion neighbourhood
  and tie count.

## What this yields, for `mt_full`

| benchmark | ours | rank / all | rank / full-coverage | leader (coverage) |
|---|---:|---:|---:|---|
| Disorder-PDB | 0.9603 | **1 / 115** | **1 / 58** | PUNCH2 0.9552 (1.00) |
| Linker | 0.8885 | 2 / 115 | **1 / 91** | IPA-AF2-Linker 0.8985 (0.87) |
| Disorder-NOX | 0.8422 | 13 / 115 | 4 / 58 | ESMDisPred-2PDB 0.8855 (0.89) |
| Binding | 0.7649 | 11 / 115 | 7 / 70 | DisoFLAG-PB 0.7760 (0.98) |
| Binding-IDR | 0.5180 | 24 / 115 | — | bindEmbed21IDR 0.6407 (1.00) |

Numbers are computed from the archived `.caid` submissions, which store four
decimals as CAID does, so every figure here is reproducible from the artefact
rather than only from the run that produced it.

Zero exact ties. Paired on shared targets we are level with the leaders on
Linker (p=0.89) and, fused, on Disorder-NOX (p=0.73); ahead on Disorder-PDB by
+0.0051 at p=0.19, which is not significant and is not claimed to be.
