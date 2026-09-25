/-
# Part XV.2  Force spectroscopy: what a pulling experiment measures, and what it cannot see

Optical-tweezer and AFM experiments interrogate a disordered region by applying a force `F`
along an end-to-end coordinate `x` and recording the mean extension.  Thermodynamically
this is again an exponential tilt: the applied potential is `-F·x`, so the populations at
force `F` are `pulled q x beta F ∝ q · exp (beta·F·x)`.

* `hasDerivAt_extension` / `stiffness_eq_beta_var` -- the slope of the force--extension
  curve is `beta` times the *variance* of the extension in the ensemble at that force.
  Compliance is fluctuation: the softness of a disordered region is its disorder.
* `rigid_is_inextensible` -- a region with a single extension has an exactly flat
  force--extension curve at every force.  A single-structure model of an IDR therefore
  predicts a vertical (infinitely stiff) pulling curve; entropic elasticity is not a
  property any deterministic predictor can have.
* `extension_strictMono` -- conversely, two populated conformations of different extension
  already force a strictly increasing curve.
* `two_state_bond_extension` -- the exactly solvable case: a two-state bond of length `b`
  has mean extension `b · tanh (beta·F·b)`, with `two_state_stiffness_zero_force` giving
  the zero-force stiffness `beta·b²` and `two_state_extension_lt` the saturation bound.
* `same_extension_law_same_force_curve` and `force_curve_blind_to_structure` -- the whole
  force--extension curve is a functional of the *law of the extension coordinate* alone.
  Two ensembles that differ in every other structural respect can therefore have exactly
  the same pulling curve at every force: a force experiment constrains one marginal, and a
  model fitted to it is underdetermined in precisely the way the earlier parts describe.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.FreeEnergy
import RequestProject.Response
import RequestProject.Crowding

namespace IDR

open Finset
open scoped Classical

namespace Force

variable {n m : ℕ}

/-- The populations under an applied force `F` conjugate to the extension coordinate `x`. -/
noncomputable def pulled (q x : Fin n → ℝ) (beta F : ℝ) : Fin n → ℝ :=
  Response.tilted q x (beta * F)

/-- The mean extension at force `F`: the force--extension curve. -/
noncomputable def extension (q x : Fin n → ℝ) (beta F : ℝ) : ℝ :=
  Response.meanObs q x x (beta * F)

/-- The variance of the extension at force `F`. -/
noncomputable def extVar (q x : Fin n → ℝ) (beta F : ℝ) : ℝ :=
  Response.var q x x (beta * F)

lemma pulled_pos {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) (beta F : ℝ) (j : Fin n) :
    0 < pulled q x beta F j := Crowding.tilted_pos hn hq _ j

lemma pulled_sum_one {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) (beta F : ℝ) :
    ∑ j, pulled q x beta F j = 1 := Response.tilted_sum_one hn hq _

/-- At zero force the pulled ensemble is the unperturbed one. -/
lemma pulled_zero {q x : Fin n → ℝ} (hq1 : ∑ j, q j = 1) (beta : ℝ) :
    pulled q x beta 0 = q := by
  funext j
  have hpart : Response.part q x 0 = 1 := by simp [Response.part, hq1]
  simp [pulled, Response.tilted, mul_zero, hpart]

/-! ## Compliance is fluctuation -/

/-- **The slope of the pulling curve is a variance.**  `d⟨x⟩/dF = beta · Var(x)`. -/
theorem hasDerivAt_extension {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (beta F : ℝ) :
    HasDerivAt (extension q x beta) (beta * extVar q x beta F) F := by
  have hlin := Response.susceptibility_eq_variance (q := q) (A := x) hn hq (beta * F)
  have hmul : HasDerivAt (fun G : ℝ => beta * G) beta F := by
    simpa using (hasDerivAt_id F).const_mul beta
  have hcomp := hlin.comp F hmul
  have hval : Response.var q x x (beta * F) * beta = beta * extVar q x beta F := by
    rw [extVar]; ring
  rw [← hval]
  exact hcomp

/-- The measured stiffness (inverse compliance) reports the conformational fluctuation of
the extension coordinate: an experiment that measures the slope measures the variance. -/
theorem stiffness_eq_beta_var {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j)
    (beta F : ℝ) : deriv (extension q x beta) F = beta * extVar q x beta F :=
  (hasDerivAt_extension hn hq beta F).deriv

/-- Nonnegative compliance: pulling harder never shortens the chain. -/
theorem extension_mono {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) {beta : ℝ}
    (hbeta : 0 ≤ beta) : Monotone (extension q x beta) := by
  have hderiv : ∀ F : ℝ, HasDerivAt (extension q x beta) (beta * extVar q x beta F) F :=
    fun F => hasDerivAt_extension hn hq beta F
  have hdiff : Differentiable ℝ (extension q x beta) := fun F => (hderiv F).differentiableAt
  refine monotone_of_deriv_nonneg hdiff fun F => ?_
  rw [(hderiv F).deriv]
  exact mul_nonneg hbeta (Response.var_nonneg hn hq _)

/-- **A rigid region is inextensible.**  If every conformation has the same extension, the
force--extension curve is exactly flat: no deterministic (single-structure) model of a
disordered region can reproduce entropic elasticity. -/
theorem rigid_is_inextensible {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) {c : ℝ}
    (hconst : ∀ j, x j = c) (beta : ℝ) : ∀ F, extension q x beta F = c := by
  intro F
  simp only [extension, Response.meanObs, hconst, ← Finset.sum_mul,
    Response.tilted_sum_one hn hq (beta * F), one_mul]

/-- Conversely, two populated conformations of different extension make the curve strictly
increasing at every force: elasticity is exactly the signature of conformational
heterogeneity along the pulling coordinate. -/
theorem extension_strictMono {q x : Fin n → ℝ} (hn : 0 < n) (hq : ∀ j, 0 < q j) {beta : ℝ}
    (hbeta : 0 < beta) {j₁ j₂ : Fin n} (hne : x j₁ ≠ x j₂) :
    StrictMono (extension q x beta) := by
  have hderiv : ∀ F : ℝ, HasDerivAt (extension q x beta) (beta * extVar q x beta F) F :=
    fun F => hasDerivAt_extension hn hq beta F
  refine strictMono_of_deriv_pos fun F => ?_
  rw [(hderiv F).deriv]
  exact mul_pos hbeta (Response.response_of_disordered hn hq _ hne)

/-! ## The exactly solvable two-state bond -/

/-- A single bond of length `b` that can point forward or backward with equal prior
weight: the one-dimensional freely jointed link. -/
noncomputable def bondQ : Fin 2 → ℝ := fun _ => 1 / 2

/-- Its extension coordinate. -/
def bondX (b : ℝ) : Fin 2 → ℝ := ![b, -b]

lemma bondQ_pos : ∀ j, 0 < bondQ j := by
  intro j; simp [bondQ]

lemma bondQ_sum : ∑ j, bondQ j = 1 := by
  simp [bondQ]

/-- **The Langevin/`tanh` law.**  The mean extension of the two-state bond is
`b · tanh (beta·F·b)`: an exact, closed-form force--extension curve derived from the
ensemble, not postulated. -/
theorem two_state_bond_extension (b beta F : ℝ) :
    extension bondQ (bondX b) beta F = b * Real.tanh (beta * F * b) := by
  have hc : 0 < Real.cosh (beta * F * b) := Real.cosh_pos _
  have hpart : Response.part bondQ (bondX b) (beta * F)
      = Real.cosh (beta * F * b) := by
    simp only [Response.part, Fin.sum_univ_two, bondQ, bondX, Matrix.cons_val_zero,
      Matrix.cons_val_one, Real.cosh_eq]
    rw [mul_comm (beta * F) b]
    ring_nf
  have hun : Response.unAvg bondQ (bondX b) (bondX b) (beta * F)
      = b * Real.sinh (beta * F * b) := by
    simp only [Response.unAvg, Fin.sum_univ_two, bondQ, bondX, Matrix.cons_val_zero,
      Matrix.cons_val_one, Real.sinh_eq]
    rw [mul_comm (beta * F) b]
    ring_nf
  rw [extension, Response.meanObs_eq, hun, hpart, Real.tanh_eq_sinh_div_cosh]
  field_simp

/-- The zero-force stiffness of the two-state bond is `beta·b²` -- a purely entropic
spring constant, proportional to temperature⁻¹ and to the squared bond length. -/
theorem two_state_stiffness_zero_force (b beta : ℝ) :
    HasDerivAt (extension bondQ (bondX b) beta) (beta * b ^ 2) 0 := by
  have h := hasDerivAt_extension (q := bondQ) (x := bondX b) (by norm_num) bondQ_pos beta 0
  have hpart : Response.part bondQ (![b, -b] : Fin 2 → ℝ) 0 = 1 := by
    simp [Response.part, bondQ]
  have hvar : extVar bondQ (bondX b) beta 0 = b ^ 2 := by
    simp [extVar, Response.var, Response.cov, Response.meanObs, Response.tilted, hpart,
      bondQ, bondX, Fin.sum_univ_two]
    ring
  rwa [hvar] at h

/-- The two-state bond never extends beyond its contour length. -/
theorem two_state_extension_lt {b : ℝ} (hb : 0 < b) (beta F : ℝ) :
    |extension bondQ (bondX b) beta F| < b := by
  rw [two_state_bond_extension, abs_mul, abs_of_pos hb]
  have h := Real.abs_tanh_lt_one (beta * F * b)
  nlinarith

/-! ## What a pulling curve cannot see -/

/-- The law of the extension coordinate: the distribution of `x` induced by the
populations, probed by an arbitrary test function. -/
def extLaw (q x : Fin n → ℝ) (g : ℝ → ℝ) : ℝ := ∑ j, q j * g (x j)

/-- **A force--extension curve is a functional of one marginal.**  Two ensembles -- of
different sizes, on different conformation spaces -- inducing the same law of the extension
coordinate have identical force--extension curves at every force and every temperature. -/
theorem same_extension_law_same_force_curve {q x : Fin n → ℝ} {q' x' : Fin m → ℝ}
    (h : ∀ g : ℝ → ℝ, extLaw q x g = extLaw q' x' g) (beta F : ℝ) :
    extension q x beta F = extension q' x' beta F := by
  have hnum := h (fun t => t * Real.exp (beta * F * t))
  have hden := h (fun t => Real.exp (beta * F * t))
  have hn' : Response.unAvg q x x (beta * F) = Response.unAvg q' x' x' (beta * F) := by
    simp only [Response.unAvg, extLaw] at *
    calc ∑ j, q j * x j * Real.exp (beta * F * x j)
        = ∑ j, q j * (x j * Real.exp (beta * F * x j)) := by
          exact Finset.sum_congr rfl fun j _ => by ring
      _ = ∑ j, q' j * (x' j * Real.exp (beta * F * x' j)) := hnum
      _ = ∑ j, q' j * x' j * Real.exp (beta * F * x' j) := by
          exact Finset.sum_congr rfl fun j _ => by ring
  have hd' : Response.part q x (beta * F) = Response.part q' x' (beta * F) := by
    simpa [Response.part, extLaw] using hden
  rw [extension, extension, Response.meanObs_eq, Response.meanObs_eq, hn', hd']

/-- A two-conformation library whose extensions are `±b` and whose second structural
coordinate is constant. -/
noncomputable def libA (b : ℝ) : Fin 2 → ℝ × ℝ := ![(b, 0), (-b, 0)]

/-- A second library with the *same* extensions and populations but a different second
coordinate: a different structural ensemble. -/
noncomputable def libB (b : ℝ) : Fin 2 → ℝ × ℝ := ![(b, 1), (-b, 1)]

/-- **Force spectroscopy is blind to structure.**  The two libraries above have identical
force--extension curves at every force and every temperature, while differing in a
structural observable by a full unit: fitting a model to a pulling curve leaves every
coordinate orthogonal to the pulling axis unconstrained. -/
theorem force_curve_blind_to_structure (b beta F : ℝ) :
    extension bondQ (fun j => (libA b j).1) beta F
        = extension bondQ (fun j => (libB b j).1) beta F
      ∧ (∑ j, bondQ j * (libA b j).2) + 1 = ∑ j, bondQ j * (libB b j).2 := by
  constructor
  · refine same_extension_law_same_force_curve (fun g => ?_) beta F
    simp [extLaw, Fin.sum_univ_two, libA, libB]
  · simp [bondQ, Fin.sum_univ_two, libA, libB]
    norm_num

end Force

end IDR
