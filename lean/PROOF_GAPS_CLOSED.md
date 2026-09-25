# The requested statements, and where each now stands

Every item below is formalised and proved in Lean, with no `sorry` and on the standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).  The whole library builds.

## 1. The imbalance-corrected bound — `RequestProject/DiscordantImbalance.lean`

With `a = |T ∩ L|`, `e = |(T ∪ L)ᶜ|` the two agreement classes and `ε = ν/n` the residue noise
rate, define the imbalance factor `κ = (a+e)²/(4ae)` (`kappa`).

* `nuPair_le_imbalanced` : `ν_pair ≤ κ · ε²/(1−ε)²`, assuming only that both agreement classes are
  nonempty — **no balance hypothesis and no bound on the noise rate.**
* `one_le_kappa`, `kappa_eq_one_iff_balanced` : `κ ≥ 1`, with equality exactly for balanced classes.
* `nuPair_le_two_eps_sq_of_kappa` : the published bound `ν_pair ≤ 2ε²` recovered as a corollary
  whenever `κ ≤ 2(1−ε)²`; `nuPair_le_two_eps_sq_of_balanced'` specialises it to balanced classes at
  `ε ≤ 1/4`.

The proof is the identity `ν_pair = du/(ae+du)`, the AM–GM step `4du ≤ ν²`, and the substitution
`a + e = n(1−ε)` from `card_split`.

## 2. Conformal p-values are superuniform, and the composition — `RequestProject/ConformalPValueBH.lean`

* `card_low_rank_le` : the deterministic half of the rank argument — at most `k` of the `n+1` items
  can have rank `≤ k`.
* `pr_rank_le` : the probabilistic half, from `ExchangeableScores`.
* `conformalP_superuniform` : **split conformal produces a valid p-value**, `P(p ≤ t) ≤ t`,
  assuming nothing but exchangeability.  This is exactly the `Superuniform` hypothesis
  `RequestProject/DependentBH.lean` assumes and never supplies.
* `conformalP_dependent_of_shared_calibration` : the p-values of a conformal screen are dependent
  by construction — one shared calibration set — so no independence or positive-dependence
  condition is available.
* `conformal_selfConsistent_fdr_le_harmonic`, `conformal_benjamini_yekutieli`,
  `conformal_screen_fdr_le` : **the pipeline** — exchangeable scores in, BH at the deflated level
  `α/H_m` on the conformal p-values, false discovery rate at most `α` out, under arbitrary
  dependence.
* `conformalP_superuniform_uniform_instance` : the hypotheses are satisfiable, so nothing is
  vacuous.

## 3. `crc_valid` — already in the library

`RequestProject/ConformalRisk.lean` proves the deterministic core `crc_validity`, and
`RequestProject/ConformalRiskExchangeable.lean` proves the quotable probabilistic statement
`crc_expected_risk_le` : `𝔼[L_test(t̂)] ≤ α` for exchangeable loss tables, distribution-free.  No
citation is being relied on.

## 4. Chain versus separation count — `RequestProject/SeparationVsChain.lean`

* `pairSeparated_card_le_capacity` : what the capacity theorem does bound — the size of a *pairwise
  separated* family, for an arbitrary score function.
* `many_separations_short_chain` : and what it does not — a two-cluster instance certifying `2k²`
  ordered comparisons in which *every* pairwise separated subfamily has at most `2` members.  The
  count of certified comparisons is no guide to how many methods can be ranked.

## 5. Capacity under an estimated ε — `RequestProject/CapacityConfidence.lean`

* `tail_le_of_rate_le` : a tail bound uniform over the null region `q ≤ q₀`, sharper than the
  sub-Gaussian bound in the small-rate regime.
* `capacity_or_rare_event` : the join — either the true rate is at least `q₀` and the capacity bound
  at `q₀` applies verbatim, or the observed count was a rare event.
* `rate_test_2746`, `capacity_at_four_percent`, `capacity_at_95_confidence` : the numbers — with
  `2746` measured structure pairs, a count of at least `160` disagreements has probability at most
  `1/20` under any true rate below `4%`; at that rate a benchmark certifiably orders at most `13`
  methods.  So: *at 95% confidence, at most 13 methods can be ordered.*

Note the direction: a measurement can only *lower*-bound the noise rate, which is the direction
that shrinks capacity.  That is why the statistical half is a one-sided test.

## 6. A counting converse — `RequestProject/UnresolvableCount.lean`

* `card_closePairs_lower` : `k² ≤ C·(|close ordered pairs| + k)` — a pigeonhole on score blocks plus
  Cauchy–Schwarz.
* `card_closePairs_unordered_lower` : the same in comparisons, `k² ≤ C·(2u + k)`.
* `benchmark_closePairs_lower`, `benchmark_unresolvable_count` : at the benchmark, with the
  unresolvability conclusion attached to each counted comparison.
* `unresolvable_count_117` : a benchmark of capacity `8` faced with `117` methods cannot resolve at
  least `798` of the comparisons between them.

## 7. `auc_target_strictMono_invariant` — `RequestProject/AUCTargetMean.lean`

* `aucTarget`, `aucTargetMean` : the per-target AUC and the **unweighted** mean over targets, which
  is the number the protocol's step 3 reports.
* `auc_target_recal_invariant`, `auc_target_strictMono_invariant`, `auc_target_shift_invariant` :
  the unweighted mean is exactly invariant under per-target strictly monotone recalibration.
* `auc_within_eq_pair_weighted_mean` and `weighted_ne_unweighted_instance` : how the reported number
  relates to the published, pair-weighted statistic, and an explicit instance where the two differ
  (`1/2` against `1/3`) — so both invariance theorems are needed.

## 8. The conformal negative — already in the library

`grouped_validity_underdetermines_element_coverage` in `RequestProject/ConformalRisk.lean`: a
group-level guarantee of `1 − α` is compatible with element-level coverage as small as one likes.
