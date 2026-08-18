# What a CAID3 rank is made of, with a theorem behind each step

Every quantity below is computed from the published `.caid` files of the CAID3
entrants and the official references, and every claim names a machine-checked
Lean 4 theorem (sorry-free, standard axioms only). Where the Lean gives a worked
example, this code reproduces it exactly — `Example.inversion` returns
within 1 / pooled 2/3 against within 2/3 / pooled 5/6, which is the strongest
available check that the Python computes the statistic the theorems describe.

## 1. The metric splits into a part calibration can move and a part it cannot

`AUC_pooled = w_within·AUC_within + w_between·AUC_between` (`auc_pooled_decomp`).

`auc_within_strictMono_invariant`: **AUC_within is invariant under any
per-protein strictly monotone recalibration** — not merely an additive shift.
Rescore every chain by its own exponential, cube or logistic squash and it does
not move. `auc_pooled_shift_diff`: the entire effect of such a recalibration on
the reported number is the between-protein term.

So the split is not a metaphor. It is a partition of CAID's statistic into the
part no chain-by-chain rescoring can touch and the part that is nothing else.

| benchmark | targets | w_within | share of the metric a recalibration can move |
|---|---:|---:|---:|
| Disorder-PDB | 233 | 0.5091% | **99.49%** |
| Disorder-NOX | 178 | 0.6747% | **99.33%** |
| Binding | 49 | 2.5518% | 97.45% |
| Binding-IDR | 42 | 2.8913% | 97.11% |
| Linker | 31 | 2.8826% | 97.12% |

## 2. Inversions are certified, and they are enormously over-determined

`inversion_requires_between_gap`: pooled order reverses within-protein order
only when `w_within·(within gap) < w_between·(between gap)`, so an inversion
*forces* a between-protein gap of at least `(w_within/w_between)` times the
within-protein gap.

On Disorder-NOX, against the pooled winner:

| method | within-protein lead | between-protein deficit | **required** | over-determined by |
|---|---:|---:|---:|---:|
| Metapredict-v3 | +0.0218 | 0.0833 | 0.000148 | **560×** |
| AIUPred-2-disorder | +0.0093 | 0.0695 | 0.000063 | 1100× |
| AlphaFold3-pLDDT | +0.0077 | 0.0833 | 0.000053 | 1570× |
| AlphaFold3-rsa | +0.0033 | 0.0523 | 0.000023 | **2305×** |

Every certificate holds. This is the quantitative heart of the finding: a
within-protein deficit of 0.0218 needs a between-protein advantage of 0.000148
to be overturned, and the one actually present is 560 times larger than that.
Calibration differences of ordinary size do not merely outweigh discrimination
differences of ordinary size — they swamp them by three orders of magnitude.

## 3. Where the winners stand on the calibration-invariant part

| benchmark | pooled #1 | its within-protein rank | within #1 (its pooled rank) |
|---|---|---:|---|
| Disorder-PDB | **DisorderNet-pbias** | **1** / 70 | DisorderNet-pbias (1) |
| Disorder-NOX | **DisorderNet-windowed** | 5 / 70 | Metapredict-v3 (18) |
| Binding | **DisorderNet-windowed** | **1** / 94 | DisorderNet-windowed (1) |
| Binding-IDR | LIPNet | **1** / 104 | LIPNet (1) |
| Linker | **DisorderNet-windowed** | **1** / 92 | DisorderNet-windowed (1) |

Ours is first on both parts on Disorder-PDB, Binding and Linker. On
Disorder-NOX we top the pooled table and sit **5th** on the part calibration
cannot move, which is the same criticism this document levels at others and is
stated here rather than in a footnote. On Binding-IDR LIPNet is first on both
and it is not close.

For contrast, the published CAID3 winners on the three benchmarks where our
submissions were not in the field: flDPnn3a is 25th within-protein on
Disorder-NOX, UdonPred-combined 21st on Binding, LINKER-Pred2 25th on Linker.

## 4. How much of each method's remaining error is fixable at all

`auc_shift_le_ceiling` bounds what any per-protein recalibration could reach.
On CAID3 the ceiling is 0.9994–0.9998 for every method — uninformative on its
own, because `w_between ≈ 1`. What is informative is
`ceiling_gap_of_crossed_matching`: a *crossed* pair of cross-protein
comparisons — positive of k below negative of l and positive of l below
negative of k — can never be got right in both directions by any bias, and a
matching of them costs a full comparison each.

The count of irreducibly crossed comparisons is therefore a certified measure
of the error no calibration can remove. Fewest, among the top twenty by pooled
AUC:

| benchmark | fewest crossed | count | second |
|---|---|---:|---|
| Disorder-PDB | **DisorderNet-windowed** | 11,423,881 | DisorderNet-pbias 11,743,296 (PUNCH2 14,781,839) |
| Disorder-NOX | **DisorderNet-windowed** | 48,682,904 | AlphaFold3-rsa 53,523,507 |
| Binding | **DisorderNet-windowed** | 2,644,015 | AlphaFold3-binding 3,243,378 |
| Linker | **DisorderNet-windowed** | 758,376 | LINKER-Pred2 797,298 |
| Binding-IDR | LIPNet | 484,208 | bindEmbed21IDR-idrGeneral 558,002 |

DisorderNet carries the smallest certified irreducible error on four of five.
On Disorder-PDB that is 23% fewer crossed comparisons than PUNCH2, the CAID3
winner. On Linker the three LINKER specialists that sweep the pooled table
carry more irreducible error than we do while ranking 27th, 28th and 31st on
the calibration-invariant part.

The crossed-matching bound is computed for the top twenty per benchmark only —
it enumerates `|pos_k|·|neg_l|` per protein pair, which runs to billions — and
blocks over the cap are skipped and counted. Every skipped block could only
*raise* the matching, so the reported bound stays valid and is merely looser.

## 5. What this does not say

- **It does not overturn CAID3.** Spearman(pooled, within) is +0.964 on
  Disorder-PDB and never below +0.849. The orderings agree strongly overall; the
  decomposition locates the disagreement rather than reversing the table.
- **It does not say between-protein ability is worthless.** A biologist asking
  "is this protein disordered" wants exactly that axis. And it is not a trivial
  composition effect: mean hydropathy, a training-free protein-level descriptor
  with within-protein AUC exactly 0.5 by construction, reaches 0.6276
  between-protein on Disorder-PDB against the best entrant's 0.9410.
- **The ranks here are recomputed** on full-coverage methods and two-class
  targets, and are not CAID's published ranks. Both columns come from the same
  subset, so comparing them is fair; neither is the official table.
- **No NP-hardness is claimed.** The exact optimum over per-protein biases is a
  weighted linear ordering problem (`exists_order_ge`, `exists_bias_of_order`),
  but the reduction from an arbitrary instance to a score table is not
  formalised and no hardness statement is made.

## Reproduction

```bash
export ANALYSIS_SCRIPT=results/caid3/certified_leaderboard.py
sbatch rockfish/slurm/analysis_cpu.sbatch
```

Job 30022076, 15 minutes on 8 CPUs. Raw output `certified_caid3.json`.
Numerical agreement with the Lean statements is asserted in
`tests/test_certified_leaderboard.py`, including the worked example.
