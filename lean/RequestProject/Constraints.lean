/-
# Part XXV.4  Rigid constraints are not the stiff limit: the Fixman factor, exactly

Molecular models routinely freeze fast degrees of freedom -- bond lengths, bond angles --
and sample the remaining coordinates with a rigid constraint.  The Boltzmann ensemble of the
*constrained* system is not the stiff limit of the Boltzmann ensemble of the *unconstrained*
one: the two differ by the square root of the determinant of the metric restricted to the
constrained coordinates (the Fixman correction).  This file proves that statement exactly in
the setting where it can be computed in closed form: one soft coordinate `q` and one stiff
coordinate `z` with stiffness `w q / eps`.

* `softMarginal_eq` -- the marginal density of `q` in the unconstrained (stiff-spring)
  ensemble is `exp(-beta V q) * sqrt(2 pi eps / (beta w q))`, for *every* `eps`.
* `soft_ratio` -- the ratio of soft marginals at two values of `q` is the constrained ratio
  `exp(-beta (V q1 - V q2))` multiplied by `sqrt (w q2 / w q1)`, independently of `eps`: the
  correction does not vanish in the stiff limit.
* `soft_eq_rigid_iff` -- the two ensembles agree at a pair of points precisely when the
  stiffness is the same there, and `rigid_ne_soft` exhibits a system where they disagree.

The conclusion for a model: a constrained model must either carry the Fixman factor
explicitly or declare that it samples the constrained ensemble, which is a different
distribution from the physical (unconstrained) one.
-/
import Mathlib

namespace IDR

namespace Constraints

open Real MeasureTheory

/-- The unnormalised marginal of the soft coordinate `q` when the stiff coordinate `z` is
harmonically confined with stiffness `w q / eps`. -/
noncomputable def softMarginal (beta : ℝ) (V w : ℝ → ℝ) (eps q : ℝ) : ℝ :=
  ∫ z : ℝ, Real.exp (-(beta * (V q + w q * z ^ 2 / (2 * eps))))

/-- The unnormalised density of the rigidly constrained (`z = 0`) ensemble. -/
noncomputable def rigidMarginal (beta : ℝ) (V : ℝ → ℝ) (q : ℝ) : ℝ :=
  Real.exp (-(beta * V q))

/-- **The stiff marginal, in closed form.** -/
theorem softMarginal_eq {beta eps : ℝ} (V w : ℝ → ℝ) (hbeta : 0 < beta) (heps : 0 < eps)
    {q : ℝ} (hw : 0 < w q) :
    softMarginal beta V w eps q
      = Real.exp (-(beta * V q)) * Real.sqrt (2 * Real.pi * eps / (beta * w q)) := by
  have hb : 0 < beta * w q / (2 * eps) := by positivity
  have hpoint : ∀ z : ℝ, Real.exp (-(beta * (V q + w q * z ^ 2 / (2 * eps))))
      = Real.exp (-(beta * V q)) * Real.exp (-(beta * w q / (2 * eps)) * z ^ 2) := by
    intro z
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  unfold softMarginal
  rw [integral_congr_ae (Filter.Eventually.of_forall hpoint), integral_const_mul,
    integral_gaussian]
  congr 2
  field_simp

/-- **The Fixman factor.**  The ratio of stiff marginals equals the ratio of the constrained
densities times `sqrt (w q2 / w q1)`, for every stiffness parameter `eps`: the discrepancy
is not a finite-stiffness artefact, it survives the rigid limit. -/
theorem soft_ratio {beta eps : ℝ} (V w : ℝ → ℝ) (hbeta : 0 < beta) (heps : 0 < eps)
    {q1 q2 : ℝ} (hw1 : 0 < w q1) (hw2 : 0 < w q2) :
    softMarginal beta V w eps q1 / softMarginal beta V w eps q2
      = (rigidMarginal beta V q1 / rigidMarginal beta V q2) * Real.sqrt (w q2 / w q1) := by
  have hc : (0:ℝ) < 2 * Real.pi * eps := by positivity
  rw [softMarginal_eq V w hbeta heps hw1, softMarginal_eq V w hbeta heps hw2]
  unfold rigidMarginal
  have hsplit : Real.sqrt (2 * Real.pi * eps / (beta * w q1))
      / Real.sqrt (2 * Real.pi * eps / (beta * w q2)) = Real.sqrt (w q2 / w q1) := by
    rw [← Real.sqrt_div' _ (by positivity)]
    congr 1
    field_simp
  have hrw : Real.exp (-(beta * V q1)) * Real.sqrt (2 * Real.pi * eps / (beta * w q1))
      / (Real.exp (-(beta * V q2)) * Real.sqrt (2 * Real.pi * eps / (beta * w q2)))
      = (Real.exp (-(beta * V q1)) / Real.exp (-(beta * V q2)))
        * (Real.sqrt (2 * Real.pi * eps / (beta * w q1))
          / Real.sqrt (2 * Real.pi * eps / (beta * w q2))) := by
    field_simp
  rw [hrw, hsplit]

/-- The stiff and constrained ensembles assign the same relative weight to two conformations
exactly when the constrained coordinate is equally stiff at both. -/
theorem soft_eq_rigid_iff {beta eps : ℝ} (V w : ℝ → ℝ) (hbeta : 0 < beta) (heps : 0 < eps)
    {q1 q2 : ℝ} (hw1 : 0 < w q1) (hw2 : 0 < w q2) :
    softMarginal beta V w eps q1 / softMarginal beta V w eps q2
        = rigidMarginal beta V q1 / rigidMarginal beta V q2
      ↔ w q1 = w q2 := by
  have hr1 : 0 < rigidMarginal beta V q1 := Real.exp_pos _
  have hr2 : 0 < rigidMarginal beta V q2 := Real.exp_pos _
  rw [soft_ratio V w hbeta heps hw1 hw2]
  constructor
  · intro h
    have hne : rigidMarginal beta V q1 / rigidMarginal beta V q2 ≠ 0 :=
      ne_of_gt (div_pos hr1 hr2)
    have hratio : Real.sqrt (w q2 / w q1) = 1 :=
      (mul_right_eq_self₀.mp h).resolve_right hne
    have hone : w q2 / w q1 = 1 := Real.sqrt_eq_one.mp hratio
    exact ((div_eq_one_iff_eq (ne_of_gt hw1)).mp hone).symm
  · intro h
    rw [h, div_self (ne_of_gt hw2), Real.sqrt_one, mul_one]

/-- **A constrained model is a different model.**  For a system whose stiff coordinate is
twice as stiff at one conformation as at another, the constrained ensemble and the stiff
limit of the physical ensemble disagree on their relative weights. -/
theorem rigid_ne_soft :
    ∃ (beta eps : ℝ) (V w : ℝ → ℝ) (q1 q2 : ℝ), 0 < beta ∧ 0 < eps ∧ 0 < w q1 ∧ 0 < w q2 ∧
      softMarginal beta V w eps q1 / softMarginal beta V w eps q2
        ≠ rigidMarginal beta V q1 / rigidMarginal beta V q2 := by
  refine ⟨1, 1, fun _ => 0, fun q => if q = 0 then 1 else 2, 0, 1, one_pos, one_pos,
    by norm_num, by norm_num, ?_⟩
  intro hcon
  have hiff := (soft_eq_rigid_iff (beta := 1) (eps := 1) (fun _ => 0)
    (fun q => if q = 0 then 1 else 2) one_pos one_pos (q1 := 0) (q2 := 1)
    (by norm_num) (by norm_num)).mp hcon
  norm_num at hiff

end Constraints

end IDR
