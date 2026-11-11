# The discordant-pair identity and the quadratic capacity

All statements below are proved in `RequestProject/DiscordantPairs.lean`, with no `sorry` and no
added axioms (`#print axioms` reports only `propext`, `Classical.choice`, `Quot.sound`).  The
whole project builds.

Notation: `T` is the truth, `L` the annotation, `d = |T \ L|` and `u = |L \ T|` the two flip
classes, `ν = |T Δ L| = d + u` the label noise, `n` the number of residues, `ε = ν / n`.

## 1. The identity

`discordantSet T L` is the set of ordered pairs `(p, q)` that `T` and `L` order oppositely.  A
labelling orders a pair only by calling one residue disordered and the other ordered, so a
reversal forces `p ∈ T \ L` and `q ∈ L \ T` — both residues flip, in opposite directions — or
the same with `p` and `q` exchanged.  Hence

    discordant_eq_flip_product :  discordantPairs T L = 2 * |T \ L| * |L \ T|.

An identity, not a bound.  The proof splits the set as the disjoint union of the two products
`(T \ L) ×ˢ (L \ T)` and `(L \ T) ×ˢ (T \ L)`.

## 2. The second-order bound

`noise_eq_flip_add` : `ν = d + u`, and AM–GM on `d · u ≤ ((d+u)/2)²` gives

    discordant_le_noise_sq :       discordantPairs T L ≤ ν² / 2      (ℕ, floor division)
    discordant_le_noise_sq_real :  (discordantPairs T L : ℝ) ≤ ν² / 2

with equality exactly when the flip classes balance:

    discordant_eq_of_balanced (h : d = u) :  2 * discordantPairs T L = ν².

`sharp_instance` exhibits the equality case concretely (`T = {0,1}`, `L = {2,3}` on four
residues: `d = u = 2`, every cross pair realised, `discordance = 8 = 4² / 2`).

## 3. The protocol, and where the noise of the pairwise answer key lives

`comparablePairs T L` are the ordered pairs that *both* labellings order; these are the pairs
the benchmark can score.  `orderKey T L S` is the answer key a labelling `S` induces on them.
Then

    noise_orderKey :  noise (orderKey T L T) (orderKey T L L) = discordantPairs T L,

i.e. on comparable pairs the pair-level label noise is *exactly* the discordant count.  This is
what lets the existing capacity machinery (`RequestProject.BenchmarkCapacity`,
`RequestProject.BenchmarkCapacityLabels`) be applied verbatim at the pair level:

    pairwise_capacity_bound :  a certified family has at most
        pairwiseCapacity T L = max 1 ⌈1 / (2 ν_pair)⌉  members,
        ν_pair = discordantPairs T L / |comparablePairs T L|.

`card_comparablePairs` counts the denominator exactly:

    |comparablePairs T L| = 2 · ( |T ∩ L| · |(T ∪ L)ᶜ| + d · u ).

## 4. The capacity corollary

`BalancedClasses T L` says `|T ∩ L| = |(T ∪ L)ᶜ|`: among the residues the two labellings agree
on, as many are disordered as ordered.  With that and a noise rate below `1/4`,

    nuPair_le_two_eps_sq :        ν_pair ≤ 2 ε²
    pairwise_capacity_quadratic : ν_pair ≤ 2 ε²  ∧  ⌈1 / (4 ε²)⌉ ≤ pairwiseCapacity T L.

The residue protocol pays `⌈1 / (2 ε)⌉`; the pairwise protocol pays `⌈1 / (4 ε²)⌉`.  The
capacity is quadratic in `1/ε` where it used to be linear.  `capacity_at_eps_0651`:
at `ε = 0.0651`, `residueCapacity = 8` and `⌈1/(4 ε²)⌉ = 59`.

Non-vacuity is exhibited (`exampleHypotheses`, `exampleCounts`, `capacity_example`): sixteen
residues, the annotation missing one disordered residue and adding one, giving `ν = 2`,
`2` discordant pairs out of `100` comparable ones, residue capacity `4` against pairwise
capacity `25`.

## 5. Scope, honestly stated

* The `2 ε²` bound (rather than the `ε²` the balanced computation actually gives) is the loose
  form asked for, and the slack absorbs the noise rate: the exact balanced value is
  `ν_pair = d·u / (|T ∩ L|·|(T ∪ L)ᶜ| + d·u)`.
* `BalancedClasses` is genuinely needed for the corollary, not decoration: the denominator
  `|T ∩ L| · |(T ∪ L)ᶜ|` collapses when the classes are lopsided, and with it the gain.
* The capacity statement is a statement about the certification bound — the number of methods a
  benchmark with that pair rate can place in a certified order — exactly as in the residue-level
  results it is compared against.
* The block-length bound of `RequestProject/PairwiseProtocol.lean`
  (`correlated_noise_reduces_pair_discordance`) is left in place.  It is a correct statement
  about a different mechanism (noise constant on blocks exposes only block boundaries), it is
  not used here, and nothing above depends on it.
