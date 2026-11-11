# What this project has, and what it is a paper about

Written to check the story hangs together, and to be honest about which claims
carry weight. Every number below is in a file in this directory with the job id
that produced it.

## The claim

**A benchmark's headline number and a method's usefulness are different
things, and on CAID3 they come apart measurably.** DisorderNet is competitive
on the first and separates on the second.

Not "state of the art". The margin over PUNCH2 on Disorder-PDB is +0.0019 at
p = 0.657, and no amount of framing makes that a win.

## Structure

### 1. The metric decomposes, exactly and with a proof

`AUC_pooled = w_within·AUC_within + w_between·AUC_between`, and
`auc_within_strictMono_invariant` identifies the first as the part **no
per-protein strictly monotone recalibration can change**. So the split is a
partition of CAID's statistic into discrimination and calibration, not a
metaphor. Measured on CAID3: 97.1%–99.5% of the pair weight is the
calibration-sensitive side.
→ `WITHIN_PROTEIN.md`, `CERTIFIED.md`

### 2. It changes who wins, and each inversion is certified

On three of five benchmarks the declared winner is 21st–25th at the
residue-level question. `inversion_requires_between_gap` turns each into a
certificate: on Disorder-NOX, Metapredict-v3's 0.0218 within-protein lead is
overturned by a between-protein gap **560× larger than the theorem requires**
(2305× for AlphaFold3-rsa).

The honest counterweight, in the same section: Spearman(pooled, within) is
+0.964 on Disorder-PDB and never below +0.849. **The decomposition does not
overturn CAID3.** It locates where the orderings disagree.
→ `CERTIFIED.md`

### 3. What no calibration can fix

`ceiling_gap_of_crossed_matching` gives a certified count of irreducible error.
DisorderNet carries the fewest on four of five benchmarks — 23% fewer than
PUNCH2 on Disorder-PDB.
→ `CERTIFIED.md`

### 4. The operational question, with a distribution-free guarantee

Conformal risk control: fix a tolerated miss rate, ask how much of a protein a
method must flag. **Validity is free and identical for everyone; the content is
the price**, and it runs from 32% to 99% across 70 methods.

Buying the same guarantee with a per-protein quantile prices discrimination
alone — invariant by the same theorem as `AUC_within`. The gap is what
calibration is worth operationally, and the leaders lean on it about twice as
hard as we do.

**The sharpest sentence in the paper:** on Disorder-PDB, AUC separates
DisorderNet from PUNCH2 by +0.0019 (p = 0.657); the calibration-invariant
operating cost separates them by 5.3 points, and by 10.2 on Disorder-NOX. *The
two are statistically inseparable on the benchmark's own metric and are not
close as instruments.*
→ `OPERATING_COST.md`, `GUARANTEES.md`

### 5. Generalisation, held out by the calendar

1,916 PDB entities released after the training caches were built; 186 survive
homology filtering. **71% of the "new" chains were already represented** —
a temporal cutoff alone is not a leak control.

`mt_windowed` 0.8933 pooled / 0.8502 within. And the unflattering result the
temporal set exists to produce: **`mt_pbias` is our best CAID3 model and the
worst here**, the only one that fails to separate from a training-free
baseline. The protein-level bias buys CAID3 points and generalises worse.
→ `TEMPORAL_HOLDOUT.md`

### 6. The model

Rank 1 on three of five CAID3 benchmarks on **both** axes; rank 1 among
full-coverage entrants on **all four** CAID2 references, from training that
never saw a CAID2 target; lowest thresholded error count on both disorder
benchmarks.
→ `CORRECTED.md`, `CAID2_REPLICATION.md`, `THRESHOLDED.md`

### 7. What the decomposition implies for training

Three losses partitioning the objective the way the metric partitions: BCE for
per-residue means, W₁ for the protein-level distribution
(`transportCost_line_eq_cdfL1`), pairwise ranking for within-protein ordering.
The third optimises the calibration-invariant component, which nothing in the
standard recipe touches.
→ `PREREGISTRATION_8.md`, `PREREGISTRATION_9.md` — **results pending**

## What is not claimed

- Not state of the art. +0.0019 over PUNCH2, p = 0.657.
- The decomposition does not overturn the leaderboard (ρ ≥ +0.849).
- The between-protein axis is a real ability, not a composition artefact:
  hydropathy reaches 0.6276 against the best entrant's 0.9410.
- Temporal and conformal analyses are **exploratory** — built after the CAID3
  numbers existed. Only the CAID3 primary families were pre-registered.
- The operating guarantee is an expectation over proteins, not a per-protein
  promise.
- No NP-hardness claim. The bias optimum is a linear ordering problem
  (`exists_order_ge`) and the reduction is not formalised. → `LEAN_REQUESTS.md`

## Mistakes this project made and caught

Kept because they are the reason to believe the rest.

| what looked true | what was true |
|---|---|
| Ten CAID3 numbers, all invalid | evaluator built the head with default dilations while the checkpoints were trained WIDE; `strict=True` accepted it silently |
| p = 0.0 between identical methods | both bootstrap tails must carry the ties |
| "no channel for protein-level information" | GroupNorm pools over the length axis; the dependency span is global, not 213 |
| holdout homologues removed | `set(hits)` was a set of tuples; the membership test never matched |
| AlphaFold-rsa's disordered coverage collapses to 0.527 | artefact of unmatched chain sets; on identical chains it is 0.994 |
| every Linker method must flag 100% | `alpha` below `1/(n+1)`; arithmetic about *n*, not the field |
| a source-text test passes | `LEADERS[task]` is a 3-tuple; it failed at runtime |

## Verdict

A strong methods paper, and the operational-guarantee section is, as far as we
can tell, new to this field. Not a revolution. The model is competitive rather
than dominant, and the durable contribution is the measurement framework — which
is also what survives if someone trains a better model tomorrow.
