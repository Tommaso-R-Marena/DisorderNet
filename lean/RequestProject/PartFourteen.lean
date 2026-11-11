/-
# Part XIV  The evaluation: which number may a disorder model be judged by

Parts I-XIII fix what a model of an intrinsically disordered region has to *be*.  This part
fixes how it is to be *scored*, which is the other half of the design: a training pipeline
returns whatever its score selects, so a score whose minimiser is not the truth guarantees
a wrong model no matter how good the architecture.

The core file is `RequestProject.Scoring`; here the results are joined to the ensemble
language of `RequestProject.EnsembleCore` and bundled.

* `Ens.popVec` -- the population vector of an ensemble, and `isProbVec_popVec`.
* `brier_rejects_deterministic_model` -- under the (strictly proper) quadratic score, a
  single-structure model scores strictly worse than the true ensemble as soon as two
  conformations are populated.  This is the evaluation-side counterpart of the Part I
  error floor: it is not merely that a single structure is wrong, it is that an honest
  score *detects* that it is wrong.
* `IDR.evaluation_design_laws` -- the bundle: a strictly proper ensemble score exists and
  its excess risk is exactly the squared population error; any strictly proper score
  rejects single-structure output; the sample-distance score in everyday use has an exact
  variance decomposition which makes it *reward* collapse, and is not strictly proper; and
  the best-of-`N` score is blind to populations, rewards hedging and is not strictly
  proper.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Scoring
import RequestProject.ScoreFloor

namespace IDR

open Scoring
open scoped Classical

namespace Ens

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- The population vector of an ensemble on a finite conformation library. -/
noncomputable def popVec (E : Ens X) : X → ℝ := fun x => E.prob x

lemma sum_prob (E : Ens X) : ∑ x, E.prob x = 1 := by
  simp only [prob, expect]
  rw [Finset.sum_comm]
  refine (Finset.sum_congr rfl (fun j _ => ?_)).trans E.w_sum
  have key : ∀ a : X, (∑ i : X, @ite ℝ (a = i) (Classical.propDecidable _) 1 0) = (1 : ℝ) := by
    intro a
    convert Finset.sum_ite_eq (Finset.univ : Finset X) a (fun _ => (1 : ℝ)) using 2 <;> simp
  rw [← Finset.mul_sum, key, mul_one]

lemma isProbVec_popVec (E : Ens X) : IsProbVec E.popVec :=
  ⟨fun x => E.prob_nonneg x, E.sum_prob⟩

omit [Fintype X] in
/-- A single-structure model has the population vector of a point prediction. -/
lemma popVec_of_deterministic {M : Ens X} {x₀ : X} (h : M.Same (dirac x₀)) :
    M.popVec = pointVec x₀ := by
  funext y
  have h1 : M.prob y = (dirac x₀ : Ens X).prob y := prob_eq_of_same h y
  have h2 : (dirac x₀ : Ens X).prob y = if y = x₀ then 1 else 0 := by
    simp [prob, dirac, expect, eq_comm]
  simp [popVec, h1, h2, pointVec]

end Ens

open Ens

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- **An honest score detects a collapsed model.**  If the target populates two distinct
conformations and `M` is a single-structure model, then under the quadratic score -- which
is strictly proper -- the true ensemble strictly beats `M`. -/
theorem brier_rejects_deterministic_model {E M : Ens X} {a b : X}
    (ha : 0 < E.prob a) (hb : 0 < E.prob b) (hab : a ≠ b) (hM : M.Deterministic) :
    expScore brier E.popVec E.popVec < expScore brier M.popVec E.popVec := by
  obtain ⟨x₀, hx₀⟩ := hM
  have hMp : M.popVec = pointVec x₀ := popVec_of_deterministic hx₀
  refine brier_strictly_proper M.popVec E.popVec (hMp ▸ isProbVec_pointVec x₀)
    (isProbVec_popVec E) ?_
  intro hEq
  rw [hMp] at hEq
  -- one of `a`, `b` differs from `x₀`, and would then have to be unpopulated
  rcases eq_or_ne a x₀ with rfl | hane
  · have hbne : b ≠ a := Ne.symm hab
    have : E.prob b = 0 := by
      have := congrFun hEq b
      simpa [pointVec, hbne] using this.symm
    exact absurd this (ne_of_gt hb)
  · have : E.prob a = 0 := by
      have := congrFun hEq a
      simpa [pointVec, hane] using this.symm
    exact absurd this (ne_of_gt ha)

/-- **The design laws of the evaluation.**

1. A strictly proper ensemble score exists -- the quadratic score -- and its excess risk is
   exactly the squared error of the predicted populations.
2. Under *any* strictly proper score, a single-structure prediction is strictly worse than
   the true ensemble unless the truth is that single structure.
3. The sample-distance score (draw a structure, measure its deviation from the observed
   one) decomposes exactly as `var(model) + var(truth) + (bias)²`; hence at fixed mean it
   strictly *rewards* a model for being less dispersed than the truth, and it is not
   strictly proper: on an equally populated three-state rotamer the collapsed model scores
   `2/3` against the truth's `4/3`.
4. The best-of-`N` score is blind to the populations, is lowered by hedging, and awards a
   perfect zero to every prediction covering the truth; it too is not strictly proper.
5. A model that cannot resolve the context is driven by an honest score to the
   context-averaged ensemble, the truth in none of the contexts.

Read together with the model-side laws: the model must output an ensemble, and the score
must be a strictly proper score of that ensemble. -/
theorem evaluation_design_laws :
    -- 1
    (StrictlyProper (X := X) brier ∧
      ∀ p q : X → ℝ, IsProbVec q →
        expScore brier p q - expScore brier q q = ∑ x, (p x - q x) ^ 2) ∧
    -- 2
    (∀ S : (X → ℝ) → X → ℝ, StrictlyProper S → ∀ q : X → ℝ, IsProbVec q → ∀ x₀ : X,
      q x₀ < 1 → expScore S q q < expScore S (pointVec x₀) q) ∧
    -- 3
    ((∀ c p q : X → ℝ, IsProbVec p → IsProbVec q →
        expScore (sqDistScore c) p q
          = varC c p + varC c q + (meanC c p - meanC c q) ^ 2) ∧
      (∀ c p p' q : X → ℝ, IsProbVec p → IsProbVec p' → IsProbVec q →
        meanC c p = meanC c p' → varC c p' < varC c p →
        expScore (sqDistScore c) p' q < expScore (sqDistScore c) p q) ∧
      ¬ StrictlyProper (sqDistScore coord3) ∧
      expScore (sqDistScore coord3) unif3 unif3 = 4/3 ∧
      expScore (sqDistScore coord3) (pointVec (1 : Fin 3)) unif3 = 2/3) ∧
    -- 4
    ((∀ d : X → X → ℝ, ∀ p p' : X → ℝ, suppF p = suppF p' →
        bestOfScore d p = bestOfScore d p') ∧
      (∀ d : X → X → ℝ, ∀ p p' : X → ℝ, IsProbVec p → IsProbVec p' →
        suppF p ⊆ suppF p' → ∀ y : X, bestOfScore d p' y ≤ bestOfScore d p y) ∧
      (∀ d : X → X → ℝ, (∀ x y, 0 ≤ d x y) → (∀ y, d y y = 0) →
        ∀ p q : X → ℝ, IsProbVec p → IsProbVec q → suppF q ⊆ suppF p →
          expScore (bestOfScore d) p q = 0) ∧
      ¬ StrictlyProper (bestOfScore coordDist2)) ∧
    -- 5
    (∀ {ι : Type} [Fintype ι] (r : ι → ℝ) (q : ι → X → ℝ), IsProbVec r →
      (∀ i, IsProbVec (q i)) → ∀ p : X → ℝ, IsProbVec p → p ≠ mixVec r q →
        ∑ i, r i * expScore brier (mixVec r q) (q i) < ∑ i, r i * expScore brier p (q i)) := by
  refine ⟨⟨brier_strictly_proper, fun p q hq => brier_excess p q hq⟩,
    fun S hS q hq x₀ hx₀ => strictlyProper_rejects_point_prediction hS q hq x₀ hx₀,
    ⟨fun c p q hp hq => sqDistScore_expected c p q hp hq,
      fun c p p' q hp hp' hq hmean hvar => sqDistScore_rewards_collapse c p p' q hp hp' hq hmean hvar,
      sqDistScore_not_strictlyProper, sqDist_three_state.1, sqDist_three_state.2⟩,
    ⟨fun d _ _ h => bestOfScore_eq_of_supp_eq d h,
      fun d _ _ hp hp' hsub y => bestOfScore_antitone_supp d hp hp' hsub y,
      fun _ hd0 hdrefl _ _ hp hq hsub => bestOfScore_expected_eq_zero hd0 hdrefl hp hq hsub,
      bestOfScore_not_strictlyProper⟩,
    fun r q hr hq p hp hne => brier_optimal_prediction_is_average hr hq p hp hne⟩

/-- **The design laws of the benchmark.**  An honest score still measures two things at once.

1. It splits exactly into the model's squared population error and a floor `gini q`
   determined by the target alone.
2. The floor is what even a perfect model pays; it is non-negative, vanishes exactly on an
   ordered target, and is largest -- `1 - 1/|X|` -- on the maximally disordered one.
3. Consequently raw benchmark numbers rank *regions*, not models: an exactly correct model of
   three equally populated rotamers scores `2/3`, while a model that misplaces a tenth of the
   population of an ordered region scores `1/50`.

So a disorder model must be reported by its excess over the floor -- equivalently, by the
squared error of its populations -- and never by the raw score. -/
theorem benchmarking_design_laws [Nonempty X] :
    (∀ p q : X → ℝ, IsProbVec q →
      expScore brier p q = (∑ x, (p x - q x) ^ 2) + gini q) ∧
    (∀ q : X → ℝ, IsProbVec q →
      expScore brier q q = gini q ∧ 0 ≤ gini q ∧
        (gini q = 0 ↔ ∃ x₀ : X, ∀ x : X, q x = if x = x₀ then 1 else 0) ∧
        gini q ≤ 1 - 1 / (Fintype.card X : ℝ)) ∧
    (expScore brier skew3 (pointVec (0 : Fin 3)) = 1/50 ∧
      expScore brier unif3 unif3 = 2/3 ∧
      expScore brier skew3 (pointVec (0 : Fin 3)) < expScore brier unif3 unif3) :=
  ⟨fun p q hq => brier_decomposition p q hq,
    fun q hq => ⟨brier_floor q hq, gini_nonneg hq, gini_eq_zero_iff hq,
      gini_le_one_sub_inv_card hq⟩,
    benchmark_confounded⟩

end IDR
