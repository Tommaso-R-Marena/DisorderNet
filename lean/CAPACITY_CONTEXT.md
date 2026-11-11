# The capacity of a benchmark with noisy labels

*What a community challenge can ever establish, and how many methods that is.*

Files: `RequestProject/BenchmarkCapacity.lean` (general theory),
`RequestProject/BenchmarkCapacityLabels.lean` (Hamming/label-noise instantiation and the
converse), `RequestProject/BenchmarkCapacityCAID.lean` (numerical instances),
building on `RequestProject/LabelNoise.lean` (`ranking_certified`).

Everything below is machine-checked; no `sorry`, no added axioms
(`propext`, `Classical.choice`, `Quot.sound` only).

## 1. The statement

A benchmark reports a score per method. That score is the quantity of interest plus an error
coming from imperfect reference labels. Two ingredients fix the smallest score difference the
benchmark can certify:

* the **annotation error rate** `ε` — a measured score can be off by `ε` in either direction,
  so a *comparison* can be off by `2ε`. This is exactly `LabelNoise.ranking_certified`: a true
  margin of more than twice the annotation noise certifies the measured ranking, and
  `ranking_certificate_sharp` shows the factor two cannot be improved;
* the **effect size** `δ` — the smallest difference the field agrees is worth calling a
  difference.

The **resolution** is `c = max(δ, 2ε)`. With `n` targets the attainable scores lie on the grid
of denominator `n` inside `[0,1]`, and the number of methods that can be pairwise separated by
more than `c` on that grid is exactly

```
k(n, ε, δ)  =  n / (⌊ c·n ⌋ + 1)  +  1 ,        c = max(δ, 2ε)      (integer division)
```

* upper bound: `IDR.BenchCapacity.card_le_benchCapacity`
* attained: `IDR.BenchCapacity.benchCapacity_attained`

So `k` is the capacity, not an estimate. Consequences proved:

* `benchCapacity_le_ceil` — `k ≤ ⌈1/c⌉` regardless of `n`: more targets do not buy resolution
  the labels do not have;
* `benchCapacity_noise_only` — with `δ = 0`, `k ≤ ⌈1/(2ε)⌉`: the capacity is a property of
  label quality alone;
* `benchCapacity_antitone` — more noise, fewer methods;
* `benchCapacity_eq_one_of_resolution_ge_one` — at `ε ≥ 1/2` the capacity is one: nothing is
  ranked.

The underlying combinatorial facts are proved in two forms that are useful on their own:
integer scores (`capacityNat`, `card_le_capacityNat`, `capacityNat_attained`) and real scores
(`capacityReal = max 1 ⌈R/c⌉`, `card_le_capacityReal`, `capacityReal_attained`, via the spread
lemma `spread_lt`).

## 2. The instantiation on disorder benchmarks

With `R` scorable residues, a truth `T`, an annotation `L` and `ν = noise T L` mislabelled
residues:

* `card_le_capacity_labelNoise` — a family of methods with pairwise true gaps above `2ν` has at
  most `R/(2ν+1) + 1` members;
* `benchmark_faithful_of_certified` — inside such a family the leaderboard *is* the truth: each
  pairwise comparison agrees, in both directions;
* `labelNoise_capacity_attained` — an explicit truth, an annotation with exactly `ν` errors and
  a family of exactly that size which is certifiably ranked;
* `card_le_benchCapacity_of_rate` — the same bound in rate form, connecting `ν` to `ε` and to
  the general `benchCapacity`;
* `capacity_theorem` — the three parts assembled.

## 3. The converse: unresolvable, by anyone

The capacity bound alone would only say that *this* argument stops. The converse says the data
do.

What a benchmark observes is the annotation `L`, not the truth. Every set within `ν` of `L` is
a truth consistent with that observation. `unresolvable_pair` constructs, for two methods whose
measured scores are close, *two* such truths — one making the first method strictly better, one
making the second strictly better. Both are compatible with everything the benchmark recorded.
The ordering is therefore not a function of the data, and no analysis of those data, however
sophisticated, can recover it. The only remedy is better labels.

`unresolvable_of_close` states this in the generic regime (the two methods disagree with the
annotation on at least `ν` residues in each direction): a measured gap below `2ν` is
undecidable. `top_group_unresolvable` applies it to a whole top group: if the leading `k`
methods sit within `2ν` of each other, every pair among them is undecidable, so the reported
order of that group is an artefact of label error. `over_capacity_has_close_pair` closes the
loop: enter more methods than the capacity and such a pair necessarily exists.

## 4. Numerical instances

Illustrative parameters at the scale of a community disorder challenge — chosen for
demonstration, not measured:

| targets `n` | error rate `ε` | effect size `δ` | capacity |
|---|---|---|---|
| 200  | 5%  | 1% | 10 |
| 200  | 10% | 1% | 5  |
| 1000 | 5%  | 1% | 10 |

and at residue level, 100000 scorable residues with 5000 mislabelled: 10 methods
(`capacity_residue_level`); halving the mislabelled count to 2500 doubles it to 20.
`four_way_tie_unresolvable` is a four-residue instance of the converse: a single mislabelled
residue makes a comparison undecidable.

## 5. Scope and honesty

* The noise model is a **worst-case budget**, not a probability model: `ν` (or `ε·R`) bounds
  the number of mislabelled residues, and all statements are uniform over annotations meeting
  that budget. This is why the converse is a hard impossibility rather than a power
  calculation: it exhibits explicit alternative truths, no statistical assumption involved.
* Scores here are Hamming-type error counts (and, in the general theory, arbitrary bounded
  scores). A benchmark using a different aggregation needs its own bound relating measured to
  true score; once it has one, the packing arguments of §1 apply verbatim, since they use only
  a range and a resolution.
* `at capacity ⇏ resolved`: the capacity is the maximum size of a *certifiable* family; a
  family of that size need not be certified. Both directions are stated separately and neither
  is silently promoted to the other.
