/-
# Part XXII  Uncertainty: what a disorder model must promise about its own answer

Part XIV fixed the score.  This part fixes the two things a pipeline reports *about* the
score: the reliability diagram (calibration) and the prediction set (coverage).  Both are
routinely offered as evidence that a model "knows what it does not know"; both are proved
here to be insufficient in a precise and quantitative way, and both are given the form in
which they do carry information.

The two source files are `RequestProject.Calibration` and `RequestProject.PredictionSets`;
here they are joined to the ensemble language of `RequestProject.EnsembleCore` and bundled.

* `ens_context_blind_calibrated_but_wrong` -- in the project's own ensemble language: a
  model that answers with the population-averaged ensemble of the benchmark is *perfectly
  calibrated* and yet has strictly positive error whenever two contexts differ.  A
  reliability diagram cannot see context blindness, which the earlier parts identify as the
  central failure of disorder prediction.
* `ens_prediction_set_card` -- the size of a valid prediction set for an ensemble is
  bounded below by the flatness of that ensemble.
* `IDR.uncertainty_design_laws` -- the bundle.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.PartFourteen
import RequestProject.Calibration
import RequestProject.PredictionSets

namespace IDR

open Scoring
open scoped Classical

variable {X : Type*}

/-! ## Bridges to the ensemble language -/

/-- **Calibration does not detect context blindness.**  Let the benchmark weight the
contexts by `mu` and let the truth in context `i` be the ensemble `E i`.  The model that
ignores its input and answers with the benchmark-averaged populations is perfectly
calibrated, and its error is strictly positive as soon as two weighted contexts have
different ensembles. -/
theorem ens_context_blind_calibrated_but_wrong [Fintype X] {I : Type*} [Fintype I] (mu : I → ℝ)
    (E : I → Ens X) (hmu : ∀ i, 0 ≤ mu i) (hsum : ∑ i, mu i = 1) {i j : I}
    (hi : 0 < mu i) (hj : 0 < mu j) (hne : (E i).popVec ≠ (E j).popVec) :
    Calib.Calibrated mu (fun l => (E l).popVec) (fun _ => (0 : Fin 1))
        (Calib.bavg mu (fun l => (E l).popVec) (fun _ => (0 : Fin 1))) ∧
      0 < Calib.risk mu (fun l => (E l).popVec) (fun _ => (0 : Fin 1))
        (Calib.bavg mu (fun l => (E l).popVec) (fun _ => (0 : Fin 1))) := by
  have hw : ∀ k : Fin 1, Calib.wt mu (fun _ => (0 : Fin 1)) k ≠ 0 := by
    intro k
    have hfib : Calib.fiber (fun _ : I => (0 : Fin 1)) k = Finset.univ :=
      Calib.fiber_unique _ k
    rw [Calib.wt, hfib, hsum]
    exact one_ne_zero
  exact Calib.calibrated_but_wrong_everywhere mu (fun l => (E l).popVec)
    (fun _ => (0 : Fin 1)) hmu hw hi hj hne

/-- **The size of a valid prediction set is a property of the target.**  A set of
conformations that carries at least `1 - alpha` of an ensemble whose largest population is
`pmax` must contain at least `(1 - alpha)/pmax` conformations. -/
theorem ens_prediction_set_card {E : Ens X} {S : Finset X} {alpha pmax : ℝ}
    (hcov : PredSet.Covers E.popVec S alpha) (hmax : ∀ x, E.prob x ≤ pmax) :
    1 - alpha ≤ (S.card : ℝ) * pmax :=
  PredSet.card_ge_of_covers hcov hmax

/-- **The design laws of uncertainty reporting.**

1. *The calibration-resolution decomposition.*  For a model that answers through a finite
   internal code, the population error splits exactly into a calibration term and the
   variance of the truth inside the code classes.
2. *Calibration is not sufficient.*  A calibrated model has paid the calibration term and
   nothing else: its whole remaining error is that resolution term, which is strictly
   positive as soon as two weighted contexts with different targets share a code.  In
   particular a context-blind model can be perfectly calibrated and wrong everywhere.
3. *Recalibration is always an improvement and never enough.*  Answering with the class
   averages attains the resolution floor, and no model using the same code does better.
4. *Validity is free.*  The whole conformation library is a valid prediction set at every
   level, so coverage alone ranks no model; the content of a prediction set is its size.
5. *The size law.*  A set covering at level `alpha` needs at least `(1 - alpha)/pmax`
   conformations, and for `m` equally populated conformations this is sharp: covering is
   *equivalent* to containing at least `(1 - alpha)·m` of them.  Hence a single structure
   is not a valid answer for any target flatter than `1 - alpha`.
6. *Coverage must be conditional.*  A reported coverage is a marginal over the benchmark's
   contexts: there are two contexts on which a context-blind set reports 90% coverage while
   covering one of them with probability zero.

Read with Part XIV: the score must be strictly proper, the reported error must be its
excess over the target's floor, calibration must be reported together with resolution, and
coverage must be reported per context together with set size. -/
theorem uncertainty_design_laws [Fintype X] {I K : Type*} [Fintype I] [Fintype K]
    [DecidableEq K] :
    -- 1
    (∀ (mu : I → ℝ) (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ), (∀ i, 0 ≤ mu i) →
      Calib.risk mu T b A = Calib.calError mu T b A + Calib.resolution mu T b) ∧
    -- 2
    ((∀ (mu : I → ℝ) (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ), (∀ i, 0 ≤ mu i) →
        Calib.Calibrated mu T b A → Calib.risk mu T b A = Calib.resolution mu T b) ∧
      (∀ (mu : I → ℝ) (T : I → X → ℝ) (b : I → K) (i j : I), (∀ i, 0 ≤ mu i) →
        0 < mu i → 0 < mu j → b i = b j → T i ≠ T j → 0 < Calib.resolution mu T b)) ∧
    -- 3
    ((∀ (mu : I → ℝ) (T : I → X → ℝ) (b : I → K),
        Calib.risk mu T b (Calib.bavg mu T b) = Calib.resolution mu T b) ∧
      (∀ (mu : I → ℝ) (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ), (∀ i, 0 ≤ mu i) →
        Calib.risk mu T b (Calib.bavg mu T b) ≤ Calib.risk mu T b A)) ∧
    -- 4
    (∀ alpha : ℝ, 0 ≤ alpha → ∀ q : X → ℝ, IsProbVec q →
      ∃ S : Finset X, PredSet.Covers q S alpha) ∧
    -- 5
    ((∀ (q : X → ℝ) (S : Finset X) (alpha pmax : ℝ), PredSet.Covers q S alpha →
        (∀ x, q x ≤ pmax) → 1 - alpha ≤ (S.card : ℝ) * pmax) ∧
      (∀ (m : ℕ), 0 < m → ∀ (S : Finset (Fin m)) (alpha : ℝ),
        PredSet.Covers (fun _ : Fin m => (m : ℝ)⁻¹) S alpha
          ↔ (1 - alpha) * m ≤ (S.card : ℝ)) ∧
      (∀ (q : X → ℝ) (alpha pmax : ℝ), (∀ x, q x ≤ pmax) → pmax < 1 - alpha →
        ∀ x₀ : X, ¬ PredSet.Covers q {x₀} alpha)) ∧
    -- 6
    (∃ (T : Fin 2 → Fin 2 → ℝ) (mu : Fin 2 → ℝ) (S : Fin 2 → Finset (Fin 2)),
      (∀ i, IsProbVec (T i)) ∧ (∀ i, 0 ≤ mu i) ∧ (∑ i, mu i = 1) ∧
      (9 / 10 : ℝ) ≤ ∑ i, mu i * PredSet.mass (T i) (S i) ∧
      PredSet.mass (T 1) (S 1) = 0) := by
  refine ⟨fun mu T b A hmu => Calib.risk_decomposition mu T b A hmu,
    ⟨fun mu T b A hmu hcal => Calib.risk_eq_resolution_of_calibrated hmu hcal,
      fun mu T b i j hmu hi hj hcode hne =>
        Calib.resolution_pos_of_conflated (i := i) (j := j) hmu hi hj hcode hne⟩,
    ⟨fun mu T b => Calib.recalibrated_risk_eq_resolution mu T b,
      fun mu T b A hmu => Calib.recalibration_improves mu T b A hmu⟩,
    fun _ halpha => PredSet.validity_is_free halpha,
    ⟨fun _ _ _ _ hcov hmax => PredSet.card_ge_of_covers hcov hmax,
      fun m hm S alpha => PredSet.unif_covers_iff hm S alpha,
      fun _ _ _ hmax hlt x₀ => PredSet.no_singleton_covers hmax hlt x₀⟩,
    PredSet.marginal_not_conditional⟩

end IDR
