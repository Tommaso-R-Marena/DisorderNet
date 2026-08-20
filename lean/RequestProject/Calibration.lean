/-
# Part XXII.1  Calibration is not enough: the resolution a model throws away

Part XIV fixed the *number* by which a model of a disordered region may be judged: an
honest (strictly proper) score.  This file addresses the diagnostic that is reported in
practice instead of, or alongside, that number -- **calibration**: the claim that "when the
model says 30% helix, the region is helical 30% of the time".

The setting is the one forced by the earlier parts.  A model does not see the target
directly; it computes a finite internal code `k = b i` of its input `i` (the sequence, the
context, the bin of a confidence readout) and answers with the population vector `A k` on
the conformation library `X`.  The inputs are weighted by `mu`, the benchmark's input
distribution, and the truth at input `i` is the population vector `T i`.

* `Calibrated` -- inside every code class the predicted populations equal the average true
  populations.  This is exactly what a reliability diagram checks.
* `risk_decomposition` -- the exact identity

      risk = calError + resolution,

  the squared population error splits into a calibration term and a *resolution* term,
  the weighted variance of the truth inside the code classes.
* `calError_eq_zero_of_calibrated`, `risk_eq_resolution_of_calibrated` -- a calibrated
  model has paid the calibration term and *nothing else*: its whole remaining error is the
  contextual variation it discarded, which no amount of recalibration can touch.
* `resolution_pos_of_conflated` -- and that term is strictly positive as soon as two
  inputs with different targets share a code.  Calibration is therefore blind to exactly
  the failure the earlier parts identify as the hard one: conflating contexts.
* `recalibrated_risk_eq_resolution`, `recalibration_improves` -- recalibration (replacing
  the answers by the class averages) is always an improvement, and it is the *best* the
  given code can do; the floor it reaches is the resolution term.
* `single_bucket_calibrated`, `single_bucket_risk`, `calibrated_but_wrong_everywhere` --
  the extreme case: a model that ignores its input entirely and always reports the
  population-averaged ensemble is perfectly calibrated while being wrong at every single
  context.  Calibration is necessary, never sufficient.
-/
import Mathlib
import RequestProject.Scoring

namespace IDR

namespace Calib

open Finset
open scoped BigOperators Classical

variable {X I K : Type*}

/-! ## The weighted bias-variance split -/

/-- Scalar bias-variance split about a weighted mean `m`. -/
lemma sq_split_scalar (B : Finset I) (w : I → ℝ) (a : I → ℝ) (p m : ℝ)
    (hm : (∑ i ∈ B, w i) * m = ∑ i ∈ B, w i * a i) :
    ∑ i ∈ B, w i * (p - a i) ^ 2
      = (∑ i ∈ B, w i) * (p - m) ^ 2 + ∑ i ∈ B, w i * (a i - m) ^ 2 := by
  have hcongr : ∀ i ∈ B, w i * (p - a i) ^ 2
      = w i * (p - m) ^ 2 + w i * (a i - m) ^ 2 + 2 * (m - p) * (w i * a i - w i * m) := by
    intro i _; ring
  have h1 : ∑ i ∈ B, w i * (p - m) ^ 2 = (∑ i ∈ B, w i) * (p - m) ^ 2 :=
    (Finset.sum_mul _ _ _).symm
  have hzero : ∑ i ∈ B, 2 * (m - p) * (w i * a i - w i * m) = 0 := by
    rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul, ← hm]
    ring
  rw [Finset.sum_congr rfl hcongr, Finset.sum_add_distrib, Finset.sum_add_distrib, h1, hzero,
    add_zero]

/-! ## The setting: inputs, codes, answers -/

/-- The inputs that the model maps to the code `k`. -/
def fiber [Fintype I] [DecidableEq K] (b : I → K) (k : K) : Finset I :=
  Finset.univ.filter (fun i => b i = k)

/-- The benchmark weight of a code class. -/
noncomputable def wt [Fintype I] [DecidableEq K] (mu : I → ℝ) (b : I → K) (k : K) : ℝ :=
  ∑ i ∈ fiber b k, mu i

/-- The average true population vector inside a code class. -/
noncomputable def bavg [Fintype I] [DecidableEq K] (mu : I → ℝ) (T : I → X → ℝ) (b : I → K)
    (k : K) : X → ℝ :=
  fun x => (∑ i ∈ fiber b k, mu i * T i x) / wt mu b k

/-- The population-space risk: the `mu`-average squared population error.  By
`Scoring.brier_excess` this is exactly the excess risk of the honest (quadratic) score,
see `risk_eq_brier_excess`. -/
noncomputable def risk [Fintype X] [Fintype I] (mu : I → ℝ) (T : I → X → ℝ) (b : I → K)
    (A : K → X → ℝ) : ℝ :=
  ∑ i, mu i * ∑ x, (A (b i) x - T i x) ^ 2

/-- The calibration term: how far each answer is from the average truth of its class. -/
noncomputable def calError [Fintype X] [Fintype I] [Fintype K] [DecidableEq K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ) : ℝ :=
  ∑ k, wt mu b k * ∑ x, (A k x - bavg mu T b k x) ^ 2

/-- The resolution term: the weighted variance of the truth *inside* the code classes,
i.e. exactly the contextual information the code `b` destroys. -/
noncomputable def resolution [Fintype X] [Fintype I] [DecidableEq K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) : ℝ :=
  ∑ i, mu i * ∑ x, (T i x - bavg mu T b (b i) x) ^ 2

/-- A model is *calibrated* when, inside every code class, the predicted populations equal
the average true populations.  This is what a reliability diagram checks. -/
def Calibrated [Fintype I] [DecidableEq K] (mu : I → ℝ) (T : I → X → ℝ) (b : I → K)
    (A : K → X → ℝ) : Prop :=
  ∀ k x, ∑ i ∈ fiber b k, mu i * T i x = wt mu b k * A k x

/-! ## Basic identities -/

lemma sum_fiber [Fintype I] [Fintype K] [DecidableEq K] (b : I → K) (g : I → ℝ) :
    ∑ k, ∑ i ∈ fiber b k, g i = ∑ i, g i :=
  Finset.sum_fiberwise Finset.univ b g

lemma bavg_spec [Fintype I] [DecidableEq K] {mu : I → ℝ} {T : I → X → ℝ} {b : I → K} {k : K}
    (h : wt mu b k ≠ 0) (x : X) :
    wt mu b k * bavg mu T b k x = ∑ i ∈ fiber b k, mu i * T i x := by
  unfold bavg
  field_simp

/-- The risk is the excess risk of the strictly proper quadratic score of Part XIV. -/
lemma risk_eq_brier_excess [Fintype X] [Fintype I] (mu : I → ℝ) (T : I → X → ℝ) (b : I → K)
    (A : K → X → ℝ) (hT : ∀ i, Scoring.IsProbVec (T i)) :
    risk mu T b A
      = ∑ i, mu i * (Scoring.expScore Scoring.brier (A (b i)) (T i)
          - Scoring.expScore Scoring.brier (T i) (T i)) := by
  refine Finset.sum_congr rfl (fun i _ => ?_)
  rw [Scoring.brier_excess (A (b i)) (T i) (hT i)]

/-! ## The decomposition -/

/-- **Calibration-resolution decomposition.**  The population error of a coded model
splits *exactly* into a calibration term and the variance of the truth inside the code
classes. -/
theorem risk_decomposition [Fintype X] [Fintype I] [Fintype K] [DecidableEq K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ) (hmu : ∀ i, 0 ≤ mu i) :
    risk mu T b A = calError mu T b A + resolution mu T b := by
  have hrisk : risk mu T b A
      = ∑ k, ∑ i ∈ fiber b k, mu i * ∑ x, (A (b i) x - T i x) ^ 2 :=
    (sum_fiber b (fun i => mu i * ∑ x, (A (b i) x - T i x) ^ 2)).symm
  have hres : resolution mu T b
      = ∑ k, ∑ i ∈ fiber b k, mu i * ∑ x, (T i x - bavg mu T b (b i) x) ^ 2 :=
    (sum_fiber b (fun i => mu i * ∑ x, (T i x - bavg mu T b (b i) x) ^ 2)).symm
  rw [hrisk, hres, calError, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  -- inside one class, `b i = k`, and the split is the weighted bias-variance identity
  have hmem : ∀ i ∈ fiber b k, b i = k := by
    intro i hi; simpa [fiber] using hi
  have hL : ∑ i ∈ fiber b k, mu i * ∑ x, (A (b i) x - T i x) ^ 2
      = ∑ x, ∑ i ∈ fiber b k, mu i * (A k x - T i x) ^ 2 := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun i hi => ?_)
    rw [hmem i hi, Finset.mul_sum]
  have hR : ∑ i ∈ fiber b k, mu i * ∑ x, (T i x - bavg mu T b (b i) x) ^ 2
      = ∑ x, ∑ i ∈ fiber b k, mu i * (T i x - bavg mu T b k x) ^ 2 := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl (fun i hi => ?_)
    rw [hmem i hi, Finset.mul_sum]
  have hC : wt mu b k * ∑ x, (A k x - bavg mu T b k x) ^ 2
      = ∑ x, wt mu b k * (A k x - bavg mu T b k x) ^ 2 := Finset.mul_sum _ _ _
  rw [hL, hR, hC, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun x _ => ?_)
  by_cases hw : wt mu b k = 0
  · -- a class of zero weight contributes nothing on either side
    have hzero : ∀ i ∈ fiber b k, mu i = 0 := by
      intro i hi
      have h0 : ∑ i ∈ fiber b k, mu i = 0 := hw
      exact (Finset.sum_eq_zero_iff_of_nonneg (fun i _ => hmu i)).1 h0 i hi
    have h1 : ∑ i ∈ fiber b k, mu i * (A k x - T i x) ^ 2 = 0 :=
      Finset.sum_eq_zero (fun i hi => by rw [hzero i hi, zero_mul])
    have h2 : ∑ i ∈ fiber b k, mu i * (T i x - bavg mu T b k x) ^ 2 = 0 :=
      Finset.sum_eq_zero (fun i hi => by rw [hzero i hi, zero_mul])
    rw [h1, h2, hw, zero_mul, add_zero]
  · have hm : (∑ i ∈ fiber b k, mu i) * bavg mu T b k x = ∑ i ∈ fiber b k, mu i * T i x := by
      have := bavg_spec (mu := mu) (T := T) (b := b) (k := k) hw x
      simpa [wt] using this
    have hsplit := sq_split_scalar (I := I) (fiber b k) mu (fun i => T i x) (A k x)
      (bavg mu T b k x) hm
    simpa [wt] using hsplit

/-! ## What calibration does, and does not, buy -/

theorem calError_eq_zero_of_calibrated [Fintype X] [Fintype I] [Fintype K] [DecidableEq K]
    {mu : I → ℝ} {T : I → X → ℝ} {b : I → K} {A : K → X → ℝ} (h : Calibrated mu T b A) :
    calError mu T b A = 0 := by
  refine Finset.sum_eq_zero (fun k _ => ?_)
  by_cases hw : wt mu b k = 0
  · rw [hw, zero_mul]
  · have hA : ∀ x, A k x = bavg mu T b k x := by
      intro x
      have hcal := h k x
      unfold bavg
      field_simp
      linarith [hcal]
    have hz : ∑ x, (A k x - bavg mu T b k x) ^ 2 = 0 :=
      Finset.sum_eq_zero (fun x _ => by rw [hA x, sub_self]; ring)
    rw [hz, mul_zero]

/-- **A calibrated model has only its resolution deficit left.**  Everything a reliability
diagram can detect has been removed; what remains is exactly the contextual variance the
model's internal code destroyed. -/
theorem risk_eq_resolution_of_calibrated [Fintype X] [Fintype I] [Fintype K] [DecidableEq K]
    {mu : I → ℝ} {T : I → X → ℝ} {b : I → K} {A : K → X → ℝ} (hmu : ∀ i, 0 ≤ mu i)
    (h : Calibrated mu T b A) :
    risk mu T b A = resolution mu T b := by
  rw [risk_decomposition mu T b A hmu, calError_eq_zero_of_calibrated h, zero_add]

lemma wt_nonneg [Fintype I] [DecidableEq K] (mu : I → ℝ) (b : I → K) (k : K)
    (hmu : ∀ i, 0 ≤ mu i) : 0 ≤ wt mu b k :=
  Finset.sum_nonneg (fun i _ => hmu i)

lemma resolution_nonneg [Fintype X] [Fintype I] [DecidableEq K] (mu : I → ℝ) (T : I → X → ℝ)
    (b : I → K) (hmu : ∀ i, 0 ≤ mu i) : 0 ≤ resolution mu T b :=
  Finset.sum_nonneg (fun i _ =>
    mul_nonneg (hmu i) (Finset.sum_nonneg (fun _ _ => sq_nonneg _)))

lemma calError_nonneg [Fintype X] [Fintype I] [Fintype K] [DecidableEq K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ) (hmu : ∀ i, 0 ≤ mu i) :
    0 ≤ calError mu T b A :=
  Finset.sum_nonneg (fun k _ =>
    mul_nonneg (wt_nonneg mu b k hmu) (Finset.sum_nonneg (fun _ _ => sq_nonneg _)))

/-- **No model with this code beats the resolution floor.** -/
theorem risk_ge_resolution [Fintype X] [Fintype I] [Fintype K] [DecidableEq K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ) (hmu : ∀ i, 0 ≤ mu i) :
    resolution mu T b ≤ risk mu T b A := by
  rw [risk_decomposition mu T b A hmu]
  linarith [calError_nonneg mu T b A hmu]

/-- Recalibration -- answering with the class averages -- attains the floor. -/
theorem recalibrated_risk_eq_resolution [Fintype X] [Fintype I] [DecidableEq K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) :
    risk mu T b (bavg mu T b) = resolution mu T b := by
  refine Finset.sum_congr rfl (fun i _ => ?_)
  congr 1
  exact Finset.sum_congr rfl (fun x _ => by ring)

/-- **Recalibration never hurts, and it is optimal for the given code.** -/
theorem recalibration_improves [Fintype X] [Fintype I] [Fintype K] [DecidableEq K]
    (mu : I → ℝ) (T : I → X → ℝ) (b : I → K) (A : K → X → ℝ) (hmu : ∀ i, 0 ≤ mu i) :
    risk mu T b (bavg mu T b) ≤ risk mu T b A := by
  rw [recalibrated_risk_eq_resolution]
  exact risk_ge_resolution mu T b A hmu

/-- **The failure calibration cannot see.**  If two inputs with strictly positive weight
share a code but have different targets, the resolution term is strictly positive -- so a
perfectly calibrated model is still strictly wrong. -/
theorem resolution_pos_of_conflated [Fintype X] [Fintype I] [DecidableEq K] {mu : I → ℝ}
    {T : I → X → ℝ} {b : I → K} {i j : I} (hmu : ∀ i, 0 ≤ mu i) (hi : 0 < mu i)
    (hj : 0 < mu j) (hcode : b i = b j) (hne : T i ≠ T j) :
    0 < resolution mu T b := by
  obtain ⟨x, hx⟩ : ∃ x, T i x ≠ T j x := by
    by_contra hcon
    push_neg at hcon
    exact hne (funext hcon)
  have hsplit : T i x ≠ bavg mu T b (b i) x ∨ T j x ≠ bavg mu T b (b j) x := by
    by_contra hcon
    push_neg at hcon
    rw [hcode] at hcon
    exact hx (hcon.1.trans hcon.2.symm)
  have key : ∀ l : I, 0 < mu l → T l x ≠ bavg mu T b (b l) x → 0 < resolution mu T b := by
    intro l hl hne'
    have hterm : 0 < mu l * ∑ x, (T l x - bavg mu T b (b l) x) ^ 2 := by
      refine mul_pos hl ?_
      have hpos : 0 < (T l x - bavg mu T b (b l) x) ^ 2 :=
        pow_pos (abs_pos.mpr (sub_ne_zero.mpr hne')) 2 |>.trans_le (le_of_eq (by rw [sq_abs]))
      refine lt_of_lt_of_le hpos ?_
      exact Finset.single_le_sum (f := fun y => (T l y - bavg mu T b (b l) y) ^ 2)
        (fun y _ => sq_nonneg _) (Finset.mem_univ x)
    refine lt_of_lt_of_le hterm ?_
    exact Finset.single_le_sum
      (f := fun m => mu m * ∑ y, (T m y - bavg mu T b (b m) y) ^ 2)
      (fun m _ => mul_nonneg (hmu m) (Finset.sum_nonneg (fun y _ => sq_nonneg _)))
      (Finset.mem_univ l)
  rcases hsplit with h | h
  · exact key i hi h
  · exact key j hj h

/-! ## The extreme case: a model that ignores its input -/

/-- With a single code class the fibre is everything. -/
lemma fiber_unique [Fintype I] [DecidableEq K] [Subsingleton K] (b : I → K) (k : K) :
    fiber b k = Finset.univ := by
  ext i; simp [fiber, Subsingleton.elim (b i) k]

/-- A model that ignores its input and reports the population-averaged ensemble is
perfectly calibrated. -/
theorem single_bucket_calibrated [Fintype I] [DecidableEq K] [Subsingleton K] (mu : I → ℝ)
    (T : I → X → ℝ) (b : I → K) (hw : ∀ k : K, wt mu b k ≠ 0) :
    Calibrated mu T b (bavg mu T b) := by
  intro k x
  exact (bavg_spec (mu := mu) (T := T) (b := b) (k := k) (hw k) x).symm

/-- ... and it is nevertheless wrong at every context whose target differs from the
average: a perfect reliability diagram is compatible with an error at every single input.
-/
theorem calibrated_but_wrong_everywhere [Fintype X] [Fintype I] [DecidableEq K]
    [Subsingleton K] (mu : I → ℝ) (T : I → X → ℝ) (b : I → K) (hmu : ∀ i, 0 ≤ mu i)
    (hw : ∀ k : K, wt mu b k ≠ 0) {i j : I} (hi : 0 < mu i) (hj : 0 < mu j)
    (hne : T i ≠ T j) :
    Calibrated mu T b (bavg mu T b) ∧ 0 < risk mu T b (bavg mu T b) := by
  refine ⟨single_bucket_calibrated mu T b hw, ?_⟩
  have hcode : b i = b j := Subsingleton.elim _ _
  rw [recalibrated_risk_eq_resolution]
  exact resolution_pos_of_conflated (mu := mu) (T := T) (b := b) hmu hi hj hcode hne

end Calib

end IDR
