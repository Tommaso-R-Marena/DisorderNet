# Does the bound bind? Measured, and the answer costs us a headline number

Job 30342579, `results/caid3/bound_tightness.py`.

## The question the paper was not asking

`card_le_benchCapacity` is a **worst-case** statement. It assumes the annotation
errors may fall wherever they most damage a comparison, which is what makes the
converse an impossibility rather than a power calculation. It is also the
bound's one soft spot: a label flip on a residue where two methods *agree* moves
both scores identically and cancels exactly. Only flips on residues where they
disagree can change the comparison.

So for a comparison of methods A and B the budget that actually threatens it is
not the whole ambiguous set but its intersection with the disagreement set:

    nu_eff(A,B) = |{ambiguous} & {A != B}|        against   nu = |{ambiguous}|

and whether the worst case is reachable is an empirical question:

    enrichment = P(A != B | ambiguous) / P(A != B)

## The measurement

Ambiguity from MobiDB's per-structure missing-residue calls; disagreement from
the submitted CAID3 predictions, each binarised at its own median over the
evaluated set so that every method calls the same number of residues disordered
and the comparison is about *which* residues rather than how many. The two
sources are independent — unlike a confident-learning flag, which is defined by
model disagreement and would make this circular.

82 accessions with a context-dependent record, 86 methods aligned over **39,392
evaluated residues**, of which **7,389 (18.8%) are context-dependent**. 3,653
method pairs.

| | median | mean | p10 | p90 |
|---|---:|---:|---:|---:|
| P(disagree) | 0.354 | 0.370 | 0.256 | 0.532 |
| P(disagree \| ambiguous) | 0.364 | 0.370 | 0.251 | 0.506 |
| **enrichment** | **0.992** | **1.009** | 0.880 | 1.175 |

46.3% of pairs have enrichment above 1; **none has enrichment above 2**.

## What it means, plainly

**Annotation ambiguity is not concentrated where methods disagree.** The errors
fall essentially independently of what any given comparison turns on. That is
the *opposite* of what would make the worst-case bound tight, and it is the
answer this project would least have chosen.

The consequence is a number:

| CAID3 Disorder-PDB, ε = 0.0801 | capacity |
|---|---:|
| marginal scores — what the paper reports | **7** |
| paired, budget ×0.506 (10% of pairs are worse) | 13 |
| **paired, budget ×0.364 (median pair)** | **18** |
| paired, budget ×0.251 (10% of pairs are better) | 25 |

**CAID3 is over capacity by 6.5×, not 17×.**

## What survives, and what does not

**Survives.** The theorem is correct and unchanged: it bounds what a benchmark
can certify about *marginal* scores, and every entrant's published number is a
marginal score. CAID3 remains far over capacity on any reading — 117 entrants
against a paired capacity of 18. The converse is untouched: the pairs the paired
budget cannot separate are still undecidable, just fewer of them.

**Does not survive.** "Capacity 7, over by a factor of seventeen" as the headline
for a *leaderboard comparison*. A leaderboard comparison is paired — the same
targets, the same labels — so the paired budget is the honest one to quote when
the question is which of two methods is better.

**Applies symmetrically.** The same refinement raises the pairwise item protocol
from 51 to 138. The escape route is not diminished; both numbers move together
and the ratio is preserved.

## The principle underneath, which is the paper's own

This is the paper's thesis applied one level up. The escape route says: *score
pairs of items rather than items, because pairing cancels the error common to
both.* This says: *compare pairs of methods paired on items rather than
comparing marginal scores, for the same reason.* Two instances of one move, and
the second was sitting in plain sight while the first was being written.

## The scope condition, which is what a reader should take away

The bound applies to every benchmark whose score is an average over items. How
*tight* it is does not transfer, because it depends on a dataset-specific
quantity:

    enrichment = P(methods disagree | item is ambiguous) / P(methods disagree)

- **≈ 1** — errors fall independently of the comparison; the paired budget is
  smaller by `P(disagree)` and the marginal bound is loose by that factor. CAID3
  is here.
- **≫ 1** — ambiguous items are the items methods disagree about; the budget
  sits where comparisons are decided and the marginal bound binds as stated.

Any benchmark with repeat annotations can measure this in an afternoon. Any
benchmark without them cannot, and should quote the marginal bound as the
conservative one. **Nobody should assume CAID3's multiplier transfers to their
data**, and the paper should not have implied it.
