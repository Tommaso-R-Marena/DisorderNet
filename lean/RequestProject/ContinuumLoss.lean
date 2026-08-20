/-
# Part XI.3  The error floor survives the continuum limit

`Ens.pointLoss_eq` (Parts I--II) decomposes the squared error of a single-structure
prediction over a *finite* library into variance plus squared bias, and deduces an
irreducible floor on any disordered degree of freedom.  A referee is entitled to ask whether
that floor is an artefact of the finite idealisation.  It is not.

This file proves the same statements for the true object: a probability measure `mu` on an
arbitrary conformation space, and a real-valued structural coordinate `q` -- a torsion angle,
an interatomic distance, a radius of gyration, the projection on any reaction coordinate --
with a finite second moment.

* `coord_pointLoss_eq` -- the exact bias--variance identity
  `∫ (q x - a)² dmu = Var(q) + (a - ⟨q⟩)²` for every predicted value `a`.
* `coord_pointLoss_min` -- the mean is the optimal single value, and
* `coordVar_pos_of_noAtoms` -- **the floor is strictly positive whenever the coordinate is
  continuously distributed** (its law has no atoms), which is exactly what "the region is
  disordered in this coordinate" means for a real chain.
* `single_structure_floor`, `continuum_error_floor` -- consequently every single-structure
  predictor, and indeed every constant prediction of the coordinate, incurs at least
  `Var(q) > 0`, with no assumption of a finite conformational library anywhere.
-/
import Mathlib

namespace IDR

open MeasureTheory

variable {X : Type*} [MeasurableSpace X]

/-- The ensemble average of a structural coordinate. -/
noncomputable def coordMean (mu : Measure X) (q : X → ℝ) : ℝ := ∫ x, q x ∂mu

/-- The ensemble variance of a structural coordinate. -/
noncomputable def coordVar (mu : Measure X) (q : X → ℝ) : ℝ :=
  ∫ x, (q x - coordMean mu q) ^ 2 ∂mu

/-- A coordinate with a finite second moment is integrable under a probability measure. -/
lemma integrable_of_sq {mu : Measure X} [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu) : Integrable q mu := by
  have hb : Integrable (fun x => 1 + q x ^ 2) mu := (integrable_const 1).add hint
  refine hb.mono' hq.aestronglyMeasurable (Filter.Eventually.of_forall fun x => ?_)
  rw [Real.norm_eq_abs]
  nlinarith [abs_nonneg (q x), sq_abs (q x), sq_nonneg (|q x| - 1)]

lemma integrable_centered_sq {mu : Measure X} [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu) (b : ℝ) :
    Integrable (fun x => (q x - b) ^ 2) mu := by
  have h' : (fun x => (q x - b) ^ 2) = fun x => (q x ^ 2 - 2 * b * q x) + b ^ 2 := by
    funext x; ring
  rw [h']
  exact (hint.sub ((integrable_of_sq hq hint).const_mul (2 * b))).add (integrable_const _)

/-- **Bias--variance, in the continuum.**  Predicting the single value `a` for the
coordinate `q` costs exactly the variance of `q` plus the squared bias. -/
theorem coord_pointLoss_eq (mu : Measure X) [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu) (a : ℝ) :
    ∫ x, (q x - a) ^ 2 ∂mu = coordVar mu q + (a - coordMean mu q) ^ 2 := by
  have hq1 : Integrable q mu := integrable_of_sq hq hint
  set m := coordMean mu q with hm
  have e1 : ∀ b : ℝ, ∫ x, (q x - b) ^ 2 ∂mu = (∫ x, q x ^ 2 ∂mu) - 2 * b * m + b ^ 2 := by
    intro b
    have i1 : Integrable (fun x => q x ^ 2 - 2 * b * q x) mu :=
      hint.sub (hq1.const_mul (2 * b))
    have i2 : Integrable (fun _ : X => b ^ 2) mu := integrable_const _
    have hfun : (fun x => (q x - b) ^ 2) = fun x => (q x ^ 2 - 2 * b * q x) + b ^ 2 := by
      funext x; ring
    rw [hfun, integral_add i1 i2, integral_sub hint (hq1.const_mul (2 * b)), integral_const,
      integral_const_mul]
    simp [hm, coordMean]
  rw [e1 a, coordVar, ← hm, e1 m]
  ring

/-- The mean is the optimal single value, and every other choice is strictly worse. -/
theorem coord_pointLoss_min (mu : Measure X) [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu) (a : ℝ) :
    coordVar mu q ≤ ∫ x, (q x - a) ^ 2 ∂mu := by
  rw [coord_pointLoss_eq mu hq hint a]
  nlinarith [sq_nonneg (a - coordMean mu q)]

/-- **The floor is strictly positive for a continuously distributed coordinate.**  If the
law of `q` has no atoms -- no single value of the torsion angle, distance or radius carries
positive probability, which is what disorder means for a real chain -- then the variance is
strictly positive. -/
theorem coordVar_pos_of_noAtoms (mu : Measure X) [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu)
    [NoAtoms (mu.map q)] : 0 < coordVar mu q := by
  set m := coordMean mu q with hm
  have hi : Integrable (fun x => (q x - m) ^ 2) mu := integrable_centered_sq hq hint m
  have hnn : (0 : ℝ) ≤ coordVar mu q :=
    integral_nonneg (f := fun x => (q x - m) ^ 2) fun _ => sq_nonneg _
  rcases hnn.lt_or_eq with h | h
  · exact h
  · exfalso
    have hae : (fun x => (q x - m) ^ 2) =ᵐ[mu] 0 :=
      (integral_eq_zero_iff_of_nonneg (fun _ => sq_nonneg _) hi).1 h.symm
    have hqm : ∀ᵐ x ∂mu, q x = m := by
      filter_upwards [hae] with x hx
      have h0 : (q x - m) ^ 2 = 0 := hx
      have h1 := (pow_eq_zero_iff (n := 2) (by norm_num)).1 h0
      linarith [sub_eq_zero.1 h1]
    have hmeas : MeasurableSet (q ⁻¹' {m}) := hq (measurableSet_singleton m)
    have h0 : mu ((q ⁻¹' {m})ᶜ) = 0 := by rw [ae_iff] at hqm; exact hqm
    have h1 : mu (q ⁻¹' {m}) = 1 := (prob_compl_eq_zero_iff hmeas).1 h0
    have h2 : (mu.map q) {m} = 0 := measure_singleton m
    rw [Measure.map_apply hq (measurableSet_singleton m), h1] at h2
    exact one_ne_zero h2

/-- **The single-structure error floor, in the continuum.**  Whatever structure `x0` the
model returns, the mean squared error of the coordinate it reports is at least the variance
of that coordinate in the true ensemble -- and that variance is strictly positive as soon as
the coordinate is continuously distributed. -/
theorem single_structure_floor (mu : Measure X) [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu)
    [NoAtoms (mu.map q)] (x0 : X) :
    0 < coordVar mu q ∧ coordVar mu q ≤ ∫ x, (q x - q x0) ^ 2 ∂mu :=
  ⟨coordVar_pos_of_noAtoms mu hq hint, coord_pointLoss_min mu hq hint (q x0)⟩

/-- The same statement for an arbitrary reported value, e.g. a regression output that is not
required to come from any actual structure: the floor is a property of the target, not of
the predictor. -/
theorem continuum_error_floor (mu : Measure X) [IsProbabilityMeasure mu] {q : X → ℝ}
    (hq : Measurable q) (hint : Integrable (fun x => q x ^ 2) mu)
    [NoAtoms (mu.map q)] :
    ∃ c > 0, ∀ a : ℝ, c ≤ ∫ x, (q x - a) ^ 2 ∂mu :=
  ⟨coordVar mu q, coordVar_pos_of_noAtoms mu hq hint,
    fun a => coord_pointLoss_min mu hq hint a⟩

end IDR
