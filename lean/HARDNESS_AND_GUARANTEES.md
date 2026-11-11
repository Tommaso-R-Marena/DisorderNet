# Three additions: hardness of the recalibration optimum, a proved conformal guarantee, and hardness of panel realisability

All statements below are machine-checked in Lean 4 with Mathlib and depend only on Lean's
standard axioms (`propext`, `Classical.choice`, `Quot.sound`).  No `sorry`.

| file | content |
|---|---|
| `RequestProject/AUCHardness.lean` | the reduction identity `bias_optimum_eq_lop` (earlier work) |
| `RequestProject/AUCHardnessDecision.lean` | **1** — the decision version, the ceiling-deficit version, and the size of the reduction |
| `RequestProject/ConformalRisk.lean` | the deterministic core `crc_validity` and the grouped negative companion (earlier work) |
| `RequestProject/ConformalRiskExchangeable.lean` | **2** — conformal risk control in expectation, for exchangeable data |
| `RequestProject/DistanceCutHardness.lean` | **3** — realisability of a mean-distance panel as a max-cut problem |

---

## 1.  The recalibration optimum: decision version, and a polynomial reduction

`AUCHardness.lean` already produced, from an arbitrary weight matrix `W : Fin K → Fin K → ℕ`, an
explicit residue table whose reachable pooled Mann–Whitney optimum over per-protein biases is

```
max_b U_pooled(s + b) = U_within(s) + C + lopOpt W .
```

Three things are added.

**The decision version** (`bias_threshold_iff_lop`).  For every `m : ℕ`,

```
(∃ b, U_pooled(s + b) ≥ U_within(s) + C + m)   ↔   m ≤ lopOpt W .
```

So the question a benchmark actually asks — *can per-protein recalibration push the reported
number past a stated target?* — is precisely the decision version of weighted linear ordering,
which is NP-complete (Karp; Garey–Johnson GT44).

**The "ceiling minus `d` pairs" version** (`ceiling_deficit_eq`,
`ceiling_deficit_threshold_iff_lop`).  The deficit of the constructed table from its ceiling
`U_within + |betweenPairs|` is exactly `|betweenPairs| − C − lopOpt W`, and asking whether the best
bias comes within `d` comparison pairs of the ceiling is again the same linear ordering threshold
question.  This is the form in which the quantity is quoted in practice.

**The reduction is polynomially sized** (`exists_good_multiplicity_card_le`,
`lop_reduces_to_bias_optimum_poly`).  The background multiplicity that pins the optimum inside the
box can be chosen so that the table has at most `26·(K+1)²·(T+1)` residues, where `T` is the total
weight of the instance.  Weighted linear ordering is NP-hard already for 0/1 weights, where
`T ≤ K²`, so the table is `O(K⁴)`: the map from an ordering instance to a score table is a genuine
polynomial-time many-one reduction, not merely an identity between two optima.

As before, the NP-hardness of weighted linear ordering itself is the classical input and is quoted
rather than machine-checked; the reduction — the mathematical content — is machine-checked.

---

## 2.  Conformal risk control in expectation, for exchangeable data

`ConformalRisk.lean` proves the deterministic core: for monotone losses bounded by `B` on a finite
threshold grid, `∑_j L j (t̂_j) ≤ α·(n+1)`, where `t̂_j` is calibrated on the `n` items other than
`j`.  `ConformalRiskExchangeable.lean` turns that into the statement a paper quotes.

The law of the whole loss table is a probability measure `μ` on
`LossTable n m = Fin (n+1) → Fin (m+1) → ℝ`, and **exchangeability** is invariance of `μ` under
relabelling the items (`Exchangeable`).  Then (`crc_expected_risk_le`)

```
𝔼 [ L_test( t̂ ) ]  ≤  α ,
```

where `t̂` is calibrated on the other `n` items — with no assumption on the data-generating process
beyond exchangeability.  The proof is the informal one made precise:

* `calib_relabel` — relabelling permutes the calibrated thresholds, because the leave-one-out sums
  are permuted; hence `realisedLoss_relabel`;
* `measurable_calib`, `measurable_realisedLoss` — the calibrated threshold takes finitely many
  values and each level set is a finite Boolean combination of the half-spaces
  `{L : (∑_{i≠j} L i t) + B ≤ α(n+1)}`, so the realised loss is measurable;
* `integral_realisedLoss_eq` — exchangeability then gives every item the same expected realised
  loss;
* averaging the deterministic inequality over the `n+1` items finishes.

`crc_expected_risk_le_dirac_zero` is a sanity check that the hypotheses are satisfiable.

The **negative companion** is already in `ConformalRisk.lean`:
`grouped_validity_underdetermines_element_coverage` exhibits two grouped panels with *identical*
group-level coverage, both valid at level `α`, one covering a `1 − α` fraction of residues and the
other an arbitrarily small fraction — a group-level guarantee does not give a per-element one, and
`microCov_ge_of_macroCov` says exactly how the size imbalance inflates the miss rate.

---

## 3.  Realisability of a mean-distance panel is a max-cut question

`DistanceCutEnsembles.lean` shows ensembles realise the cut cone truncated at the contour scale.
`DistanceCutHardness.lean` makes the computational statement, for the *two-state* ensembles that
realise those panels — each residue at one of two positions a distance `t` apart, the
compact/extended exchange in its simplest form.

* `maxCut_eq_greatest_twoState_panelValue` — **the reduction identity.**  For a weight matrix `c`
  on the residues and a scale `t ≥ 0`, the greatest value of the weighted mean-distance functional
  `∑_{i,j} c i j · D i j` over all two-state ensembles at scale `t` is exactly `t · maxCut c`, and
  it is attained by a single conformation.  Fitting the best ensemble to a weighted panel target is
  max cut, which is NP-hard.
* `cutRealisable_iff_forall_weight` — **the realisability test.**  A panel `p` is realisable iff
  `∑_{i,j} c i j · p i j ≤ t · maxCut c` for *every* weight matrix `c`.  The forward direction is a
  computation; the converse is separation — the realisable panels form a compact convex set
  (`isCompact_cutRealisableSet`), and every continuous linear functional on panels is a weighted
  panel test (`exists_weights_of_functional`).
* `not_cutRealisable_iff_exists_weight` — dually, a refutation of realisability *is* a weighting
  that beats its own maximum cut, and
  `maxCut_eq_greatest_panelValue_over_realisable` — the support function of the realisable panels
  is the maximum cut, so an oracle for the fitting optimum is an oracle for max cut.
* `cutRealisable_iff_twoStateEnsemble` — the bridge: the panels above are precisely the
  mean-distance panels of ensembles of two-state chain conformations with all bonds at most `t`.

**What is not claimed.**  General ensemble panels are convex combinations of *Euclidean* metrics, a
cone strictly larger than the cut cone, so what is proved here is hardness of the two-state (cut)
realisability problem, not of general ensemble realisability; the obstacle noted in the ensemble
paper — that containment alone does not transfer hardness — is not removed.  As elsewhere, the
NP-hardness of max cut is the classical input and is quoted, not machine-checked; what is
machine-checked is that the realisability question is that problem.

That obstacle is now *certified* rather than merely noted.
`exists_ensemble_panel_not_cutRealisable` exhibits a single Euclidean conformation — three
residues at the corners of a unit equilateral triangle, every bond of length exactly `1` — whose
mean-distance panel `p i j = 1` for `i ≠ j` is **not** cut-realisable at scale `1`: the all-ones
weighting scores `6` on it (`panelValue_ones_triPanel`) while no split of three residues scores
more than `4` (`maxCut_ones_le_four`).  So the inclusion of the cut cone in the ensemble panels is
strict already at three residues, and hardness genuinely cannot be transferred by inclusion; a
reduction landing inside the sub-family where the two cones agree remains the open step.
