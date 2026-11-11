# Four theorems about grouped AUC (proteins as groups)

All statements below are machine-checked in Lean 4 with Mathlib, in the files

| file | content |
|---|---|
| `RequestProject/AUCCore.lean` | definitions; `AUC_pooled = w_within·AUC_within + w_between·AUC_between` and `w_within = Σ_k P_k N_k / (Σ_k P_k)(Σ_k N_k)` |
| `RequestProject/AUCInvariance.lean` | **Theorem 1** |
| `RequestProject/AUCCeiling.lean` | **Theorem 2**: attainment, ceiling, obstructions, ordering encoding |
| `RequestProject/DifferenceConstraints.lean` | the cycle criterion for strict difference constraints (proved from scratch) |
| `RequestProject/AUCCeilingCriterion.lean` | **Theorem 2**: the `O(K³)` decision procedure |
| `RequestProject/AUCCeilingExamples.lean` | two worked instances |
| `RequestProject/AUCCoverage.lean` | **Theorem 3** |
| `RequestProject/AUCInversion.lean` | **Theorem 4** |

No `sorry`; every result depends only on Lean's standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).

## Definitions used

Items (residues) `i` carry a label `lab i : Bool`, a group `grp i` (the protein) and a score
`s i : ℝ`.  The Mann–Whitney kernel is `kern x y = 1` if `y < x`, `1/2` if `x = y`, `0` otherwise.
For a set `X` of (positive, negative) pairs, `U X s = Σ_{(p,n) ∈ X} kern (s p) (s n)` and
`auc X s = U X s / |X|`.  With `allPairs`, `withinPairs`, `betweenPairs` this gives
`AUC_pooled`, `AUC_within`, `AUC_between`, and `auc_pooled_decomp` + `weight_within_eq` are the
identity you already verified against brute force.

## Theorem 1 — invariance (`auc_within_shift_invariant`, `auc_pooled_shift_diff`)

Adding `b : G → ℝ` to all residues of each protein leaves `AUC_within` **exactly** invariant, and

```
AUC_pooled(s+b) − AUC_pooled(s) = w_between · (AUC_between(s+b) − AUC_between(s)).
```

So the bias term is safe by construction: it cannot touch any within-protein comparison, and its
entire effect on the benchmark number is the between-protein term.  Proved in the stronger form
`auc_within_strictMono_invariant`: the invariance holds for *any* per-protein strictly monotone
recalibration, not only an additive constant.

## Theorem 2 — the ceiling of `max_b AUC_pooled(s+b)`

**Is the supremum attained?**  Yes, always (`exists_max_bias`, `exists_max_bias_auc`).  Not by
compactness — `ℝ^K` is not compact and the objective is a step function — but because `s+b` enters
the statistic only through the finite comparison pattern it induces, so the objective has finite
range.

**Closed form / tight upper bound.**  Since the within part is frozen (Theorem 1),

```
max_b AUC_pooled(s+b) ≤ w_within·AUC_within(s) + w_between        (auc_shift_le_ceiling)
```

and this is the only closed form available in general.  It is attained **exactly** when every
cross-protein comparison can be made correct, i.e. when the system of strict difference
constraints

```
b k − b l > s n − s p      for every positive p of protein k and negative n of protein l, k ≠ l
```

is feasible (`U_shift_eq_ceiling_iff`, `ceiling_attainable_iff`).

**Is it computable in polynomial time?**  For the question that matters — *can the architecture
reach the ceiling?* — yes, in `O(K³)`:

* `exists_gapMatrix` builds the `K × K` gap matrix `c k l = max { s n − s p }` in one pass;
* `feasible_iff_gap` says the residue-level system and the `K × K` system have the same solutions;
* `feasible_iff_noNonnegCycle` (proved from scratch, by vertex elimination) says a strict
  difference system is feasible **iff every closed walk of the weight matrix has strictly negative
  weight**;
* hence `ceiling_attainable_iff_noNonnegCycle`: the ceiling is reachable iff the gap matrix has no
  non-negative cycle — a Bellman–Ford / Floyd–Warshall test, `O(K³)`, *not* a search over the `K!`
  protein orders.

**What binds when the ceiling is out of reach.**  The obstruction is exact and certifiable:

* `cycle_obstruction` — every cyclic chain of cross-protein comparisons must have strictly negative
  total gap; a single violated cycle is a certificate that the ceiling is unreachable;
* `ceiling_unattainable_of_two_cycle` — the shortest certificate: `c k l + c l k ≥ 0` for two
  proteins;
* `crossed_pair_le_one`, `ceiling_gap_of_crossed` — quantitatively, a *crossed pair* (positive of
  `k` below negative of `l` and positive of `l` below negative of `k`, in the summed sense
  `(s n_l − s p_k) + (s n_k − s p_l) ≥ 0`) can never be got right in both directions: it costs a
  full comparison pair;
* `ceiling_gap_of_crossed_matching` — a whole *matching* `A` of crossed comparisons costs `|A|/2`
  pairs:
  `max_b U_pooled ≤ U_within(s) + |between| − |A|/2`.
  Any matching in the crossing graph gives such a bound, so the bound is computable in polynomial
  time and is the practical answer to "does something much lower bind?".

**Complexity of the exact optimum.**  `exists_order_ge` and `exists_bias_of_order` show that in the
well-separated regime the problem *is* a linear ordering problem: maximising
`Σ_{k≠l} w_{k l}·1[b k − b l > 1]` over `b ∈ ℝ^K` has the same optimum as maximising
`Σ_{π l < π k} w_{k l}` over permutations `π`.  So the exact optimum is a combinatorial
optimisation over orderings of the proteins, and no closed form should be expected.

The converse reduction is now formalised, in `AUCHardness.lean`.  For every weight matrix
`W : Fin K → Fin K → ℕ` with `W k k = 0` there is an explicit residue table — `M` background
positives at score `0` and `M` background negatives at score `-N` per protein (`N = 2K+2`), plus,
for every unit of weight `W k l`, one extra positive in protein `k` at a large isolated score `S`
and one extra negative in protein `l` at `S + 1` — for which

> `max_b U_pooled(s + b) = U_within(s) + C + lopOpt W`   (`bias_optimum_eq_lop`)

with `C` the number of cross-protein comparisons won inside the box and
`lopOpt W = max_π Σ_{π l < π k} W k l` the linear ordering optimum.  Both halves are proved: the
bound holds for *every* bias, and it is attained by the bias `b k = 2·rank(k)` of an optimal
order.  The mechanism is that the background comparisons cost `M²/2` as soon as two biases differ
by `N` or more, which is more than the rest of the table is worth, so the optimiser is confined to
a box in which every comparison except the slot comparisons is decided independently of the bias,
and each slot comparison is won exactly when its protein is placed above its partner.
`lop_reduces_to_bias_optimum` packages this as: every weighted linear ordering instance *is* a
per-protein recalibration problem.

Weighted linear ordering (equivalently maximum acyclic subgraph) is NP-hard — Karp's list;
Garey–Johnson GT44 — so computing the reachable optimum for an arbitrary score table is NP-hard
too.  That last step is the classical citation and is *not* machine-checked: what is machine-checked
is the reduction identity, which is the mathematical content.  This is why the polynomial-time
`ceiling_gap_of_crossed_matching` bound above is the right practical instrument: the exact optimum
is not something a benchmark should expect to compute.

**Two worked instances** (`AUCCeilingExamples.lean`):

* the miscalibrated five-residue predictor `A` below has pooled AUC `2/3`, and with `b = (3, 0)`
  reaches pooled AUC `1` — its ceiling is attained;
* six residues in two proteins, positives at `0, 20` and negative at `19` in each: the two
  proteins are crossed, and *no* bias brings the pooled statistic above `5/8`, while the ceiling
  is `6/8`.  The gap is exactly the one pair predicted by `ceiling_gap_of_crossed`.

## Theorem 3 — coverage bias (`auc_decline_diff`, `auc_decline_increase_iff`, `decline_fraction_lower_bound`)

Let a method decline a set `S` of proteins, `R` the retained pairs, `D` the declined ones.  Then
**exactly**

```
AUC(R) − AUC_pooled = (|D| / |X|) · ( AUC(R) − AUC(D) ).
```

Consequences, all proved:

* declining **strictly increases** the reported AUC iff the method is strictly worse on the pairs
  it declined than on the pairs it kept (`auc_restrict_increase_iff`), and leaves it unchanged iff
  the two agree — nothing else about the declination matters;
* the gain is bounded: `≤ (|D|/|X|)·(1 − AUC(D)) ≤ |D|/|X|`.  So a measured coverage gain of `g`
  proves that at least a fraction `g` of all comparison pairs was declined: the measured `+0.118`
  on Disorder-NOX forces at least `11.8 %` of the pooled pair set to have been thrown away.

## Theorem 4 — non-monotonicity (`Example.inversion`, `auc_inversion_iff`)

Explicit instance: five residues, two proteins (positives `0,1` and negative `2` in protein 0;
positive `3`, negative `4` in protein 1), predictors

```
A = (1, 2, 0, 4, 3)      AUC_within = 1,     AUC_pooled = 2/3
B = (1, −1, 0, 3, −2)    AUC_within = 2/3,   AUC_pooled = 5/6
```

so `AUC_within(A) > AUC_within(B)` while `AUC_pooled(A) < AUC_pooled(B)`: the pooled benchmark
inverts the within-protein ordering.

The exact characterisation (`auc_inversion_iff`) is

```
AUC_pooled(A) < AUC_pooled(B)
  ⟺  w_within·(AUC_within(A) − AUC_within(B)) < w_between·(AUC_between(B) − AUC_between(A)),
```

with the quantitative form `inversion_requires_between_gap`: an inversion forces a between-protein
AUC gap larger than `(Σ_k P_k N_k)/((Σ_k P_k)(Σ_k N_k) − Σ_k P_k N_k)` times the within-protein
gap.  In other words the benchmark reverses the biological ordering exactly when the loser's
cross-protein calibration advantage, weighted by the cross-protein pair fraction, outweighs the
winner's within-protein advantage.

## Addendum

The decision version of the recalibration question, the "ceiling minus `d` pairs" version, and the
polynomial size of the reduction are in `RequestProject/AUCHardnessDecision.lean`; see
`HARDNESS_AND_GUARANTEES.md`.
