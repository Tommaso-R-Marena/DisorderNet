# Third response: the ceiling is nearly vacuous — here is what actually binds

The feedback is right on both counts, and both are now theorems.

1. `w_within·AUC_within + w_between` is close to `1` on any benchmark where the cross-protein
   pairs dominate, so as a *bound* it says almost nothing. Its value was always the
   *attainability* question attached to it, not the number.
2. The gap matrix is separable, so the cycle criterion collapses to a sum of per-protein
   scalars. It collapses further than expected: not `O(K log K)` but a comparison of the two
   largest scalars.

And the part that does bind — the crossed-matching bound — is now stated in its maximal form,
with an algorithm that computes it and a proof that the algorithm is optimal.

New files: `RequestProject/AUCGapSeparable.lean`, `RequestProject/ThresholdMatching.lean`,
`RequestProject/AUCCrossedMatching.lean`, `RequestProject/AUCCrossedMatchingExamples.lean`.
Everything below is proved in Lean 4 with Mathlib, with no `sorry` and no axioms beyond
`propext`, `Classical.choice`, `Quot.sound`.

---

## 1. The separable gap matrix: the ceiling test is two numbers

Write `minpos k` for the smallest score of a positive residue of protein `k` and `maxneg k` for
the largest score of a negative residue of protein `k` (`IsMinPos`, `IsMaxNeg`;
`exists_minPos_maxNeg` builds them in one pass whenever every protein has at least one residue of
each class).

**`isGapMatrix_sep`.** The `K × K` gap matrix is `c k l = maxneg l − minpos k`. So it is carried by
`2K` numbers, and the weight of any closed walk telescopes into a sum of the per-protein scalars

```
overlap k = maxneg k − minpos k
```

— by how much protein `k`'s negative range overtops its positive range.

**`ceiling_attainable_iff_pairwise_overlap`.** A per-protein bias reaches the ceiling **iff**
`overlap k + overlap l < 0` for every pair of distinct proteins. No `K × K` matrix, no `O(K³)`
negative-cycle search.

**`pairwise_overlap_iff_two_largest`.** That condition is decided by the two largest overlaps
alone: if `k₁` carries the largest and `k₂` the largest of the rest, the test is exactly
`overlap k₁ + overlap k₂ < 0`. An `O(K)` scan (an `O(K log K)` sort if you want the whole ranking).

**`ceiling_attained_by_centering`.** When the test passes, the optimum is explicit rather than
searched for: `b k = −(minpos k + maxneg k)/2`, i.e. centre every protein on the midpoint of its
min-positive and max-negative score.

**`ceiling_unattainable_of_two_overlapping`.** The practical corollary, and it confirms the
expectation in the feedback: as soon as **two** proteins have `minpos ≤ maxneg` — one within-protein
comparison at the extremes coming out wrong is enough — the ceiling is unreachable. With a
within-protein AUC near `0.70` this is the typical case, so the ceiling is not attained and the
matching bound is what binds.

## 2. The crossed-matching bound, in its maximal form

Two cross-protein comparisons are *crossed* (`CrossedConfig`) when they are distinct, run between
the same two proteins in opposite directions, and their gaps sum to a non-negative number: at most
one of the two can come out right, whatever the bias. A *crossed matching* (`IsCrossedMatching`) is
a set of comparisons paired up by a fixed-point-free involution into such configurations, and

```
M = maxCrossedCard lab grp s
```

is the size of the largest one (`maxCrossedCard`). It is attained by an actual matching
(`exists_max_crossed_matching`) and dominates every particular matching (`card_le_maxCrossedCard`),
so the bound below is the tightest of its family (`max_crossed_bound_le_matching_bound`).

**`ceiling_gap_of_max_crossed_matching`.** For every per-protein bias `b`,

```
U_pooled(s + b) ≤ U_within(s) + |between| − M/2 .
```

**`auc_shift_le_max_crossed`.** In AUC form,

```
AUC_pooled(s + b) ≤ w_within·AUC_within(s) + w_between − M / (2·|all pairs|) .
```

**`target_unreachable_of_max_crossed`.** The decision this supports: if the right-hand side is
below a target AUC, then *no* per-protein bias reaches the target. It is computed from the score
table alone, before any bias is fitted. `auc_shift_le_of_le_maxCrossedCard` states the same with
any certified lower bound `M₀ ≤ M`, which is what an implementation actually reports.

## 3. The algorithm, and why it is optimal

**The graph decomposes.** Crossing can only relate a comparison running `k → l` with one running
`l → k`, and inside such a block it is the pure threshold condition `gap + gap ≥ 0`
(`crossedConfig_iff_threshold`). So the crossed graph is a disjoint union, over unordered protein
pairs, of bipartite graphs of the form `u_i + v_j ≥ 0`.

**Threshold graphs are matched by sorting** (`RequestProject/ThresholdMatching.lean`). With both
gap lists sorted decreasingly, define the greedy test `GreedyFeasible u v m`: pair the top `m`
entries of `u` against the top `m` entries of `v` *in reverse*, `u t` with `v (m−1−t)`, and check
every sum is non-negative. Then

* `greedy_mem_matchSizes` — the test is sufficient (the reversed pairing *is* a matching);
* `greedy_of_mem_matchSizes` — and necessary: any matching of size `m` forces it;
* `le_maxMatch_iff_greedyFeasible` — hence `maxMatch u v` is exactly the largest `m` passing the
  test, and by `greedyFeasible_of_succ` the test is monotone in `m`, so one scan after an
  `O(n log n)` sort returns the block's maximum. No general matching algorithm is needed.

**Back to the AUC bound.**

* `maxCrossedCard_ge_of_block` — soundness: any block matching of size `m`, however obtained,
  certifies `2m ≤ M`, hence a valid upper bound on the reachable pooled AUC. Summing over blocks
  is legitimate for the same reason.
* `block_card_le_two_mul_maxMatch` — optimality: no crossed matching can take more than
  `2 · maxMatch` comparisons out of a block, so greedy is not merely a heuristic here.
* `maxCrossedCard_eq_two_mul_maxMatch` — when all cross-protein comparisons run between the same
  two proteins, the two statements meet and `M` is computed exactly.

So the pipeline is: for each unordered pair of proteins, collect the two gap lists, sort them
decreasingly, scan for the largest `m` with `u_t + v_{m−1−t} ≥ 0`; sum `2m` over pairs; feed the
total into `auc_shift_le_of_le_maxCrossedCard`. The result is a certified upper bound on what
*any* per-protein calibration term can reach, available before the bias is fitted.

## 4. A worked instance where the ceiling is loose and the matching bound is tight

`RequestProject/AUCCrossedMatchingExamples.lean` takes the six-residue, two-protein instance of
`AUCCeilingExamples.lean`: each protein holds positives at `0` and `20` and a negative at `19`.
The four cross-protein comparisons have gaps `19, −1` in one direction and `19, −1` in the other.

| bound | value |
|---|---|
| ceiling `w_within·AUC_within + w_between` | `6/8` |
| one crossed pair (`ceiling_gap_of_crossed`) | `5/8` |
| maximum crossed matching, `M = 4` (`matching_auc_le`) | `1/2` |
| actually reachable (`matching_auc_attained`, zero bias) | `1/2` |

The maximum matching pairs the gap `19` on one side with the gap `−1` on the other, in both
directions — a pairing the "obvious" like-with-like matching misses, and exactly what the reversed
greedy produces. Here the crossed-matching bound is tight and the ceiling is not.

## 5. What is still not claimed

No measurement enters this repository, and nothing here is fitted to any benchmark. The theorems
are about an arbitrary score table with an arbitrary grouping into proteins; whether a particular
per-protein term can reach a particular benchmark number is decided by running the computation of
§3 on that benchmark's score table, which is an empirical step this project does not perform.
