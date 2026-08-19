# Pre-registration 10 — an ensemble selected where selection is legal

Committed **before** any ensemble is built and before any CAID3 number from one
exists. Methodology `METHODOLOGY.md`.

## Why now and not earlier

This project has had a working ensemble result since early on — a two-model
rank fusion placed first on four of five CAID3 benchmarks — and has never
reported it as a result, for one reason: **there was nowhere legal to choose
the members or the weights.** Choosing them on CAID3 is fitting on the test
set. Choosing them on cross-validation does not work, because CV here is
anti-correlated with CAID3 (Spearman −0.60 to +0.37) and is computed on each
run's own folds.

The fixed validation holdout removes that obstacle. Four checkpoints —
`mt_chiral`, `mt_control`, `mt_motif`, `mt_wass` — reserved the **same 2,991
rows** by the same sequence hash, trained on the same 22,382, and were scored on
those rows by the same code. `mt_rank` joins them when it finishes. For the
first time, members and weights can be chosen on data none of them saw and all
of them share.

The three older checkpoints (`mt_windowed`, `mt_pbias`, `mt_publication`)
trained *on* the holdout chains and are therefore **excluded from the pool**,
however well they score. That is the cost of having built the holdout late, and
it is paid rather than argued around.

## The rule, fixed before any number

1. **Pool** — every checkpoint whose `multitask_results.json` records the
   validation holdout with the same salt and fraction. Verified from the
   checkpoint, not asserted.
2. **Combination** — per-residue **rank fusion**, ranked globally over pooled
   residues, which is the form already implemented and tested
   (`caid3_official.rank_fuse`). Per-target ranking silently changes the metric
   and is not used.
3. **Weights** — equal. Not fitted. With four to five members and one holdout,
   fitting weights invites exactly the overfitting the holdout exists to
   prevent, and equal weighting was already the empirically right choice when
   this project last measured it (the AUC-maximising weight sat between 0.45 and
   0.50 on every task where fusion helped).
4. **Membership** — chosen on the holdout, per task, by a rule stated here:
   include a checkpoint in a task's ensemble iff its holdout AUC on that task is
   within **0.02** of the best holdout AUC for that task. A checkpoint far
   behind on the holdout contributes noise; the threshold is fixed now so it
   cannot be tuned later.
5. **Scored once** on CAID3, all five references, coverage 1.00 required.

## Primary endpoint

On **Disorder-PDB**, all 319 targets, paired protein-clustered bootstrap
(10,000 resamples), Holm across these two — the same primary family this
project registered at the outset:

- **P1** — the ensemble beats **AlphaFold-rsa**, the training-free structural
  baseline that ranks 3rd on this benchmark.
- **P2** — the ensemble beats **PUNCH2**, the CAID3 winner. This is the test
  this project has failed four times: +0.0043 (p = 0.202), +0.0019 (p = 0.657),
  +0.0035 (p = 0.343), +0.0063 (p = 0.112). **Registered as the endpoint that
  matters.**

## Secondary

Disorder-NOX, Linker, Binding and Binding-IDR placements, and the operating
cost at risk ≤ 0.10 against the same field. Exploratory, Holm within that
family.

## Falsifiable prediction

Rank fusion averages out *independent* error. The four members differ by one
flag each and share a trunk, a backbone, a training union and a seed, so their
errors are correlated and the gain should be **small** — smaller than the
+0.0139 this project measured for its earlier two-model fusion, which combined
architectures that differed far more.

If the ensemble gains much more than that, the members are less similar than
they look and the explanation is wrong; if it gains nothing, correlated members
were the reason and that is worth stating.

## The risk, stated in advance

Four near-identical models may simply reproduce one of them. The honest outcome
in that case is that **this project cannot beat PUNCH2**, after five
architectures and an ensemble — which, given that AlphaFold-rsa reaches 0.9300
on this benchmark with no training at all, is a statement about the benchmark's
ceiling rather than about the model.

## Stopping rule

Scored once. If P2 fails, it is reported as the fifth failure and the paper
claims competitiveness, not superiority. **No further ensemble variant is built
on the strength of a CAID3 number.**
