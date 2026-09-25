/-
# Part XXIV.6  Scoring a density, and the exact price of not being equivariant

Part XIV scored discrete population vectors.  A generative model of a disordered region
outputs a *density* on `R^(3N)`, so the score must be an integral, not a sum.  This file
redoes the two things that matter in the continuous setting.

* `l2_excess_decomposition`, `l2risk_eq_zero_iff` -- the continuous quadratic (Brier) score
  `int f^2 - 2 int f p` exceeds its minimum `- int p^2` by exactly `int (f - p)^2`, and that
  excess vanishes precisely when the model density equals the truth almost everywhere.  A
  strictly proper score for densities exists.
* `l2risk_rigid_invariant` -- if the truth is invariant under a rigid motion, so is the risk
  of the moved model: the score cannot distinguish a molecule from its rotated copy.
* `symmetrization_identity` -- **the exact price of non-equivariance**:

    `risk f = risk (symmetrised f) + (1/4) * int (f - f o g)^2`.

  Symmetrising a model over a symmetry of the target never hurts, and the improvement is
  exactly a quarter of the model's own non-equivariance.  Hence `symmetrization_improves`
  and `equal_risk_iff_equivariant`: a model that is not equivariant is *strictly* beaten by
  its own symmetrisation.  This is the formal reason an architecture should be equivariant
  by construction rather than by data augmentation.
-/
import Mathlib
import RequestProject.GibbsField

namespace IDR

namespace ContScore

open MeasureTheory Real MM Gibbs

variable {N : ℕ}

/-- The excess risk of the continuous quadratic score: the squared `L^2` distance between
the model density and the true density. -/
noncomputable def l2risk (f p : Conf N → ℝ) : ℝ := ∫ x, (f x - p x) ^ 2

/-- **The continuous quadratic score is strictly proper.**  Its expected value exceeds the
value at the truth by exactly the squared `L^2` error. -/
theorem l2_excess_decomposition {f p : Conf N → ℝ}
    (hff : Integrable (fun x => f x ^ 2)) (hpp : Integrable (fun x => p x ^ 2))
    (hfp : Integrable (fun x => f x * p x)) :
    (∫ x, f x ^ 2) - 2 * ∫ x, f x * p x = l2risk f p - ∫ x, p x ^ 2 := by
  unfold l2risk
  have hexp : ∀ x : Conf N, (f x - p x) ^ 2 = f x ^ 2 - 2 * (f x * p x) + p x ^ 2 := by
    intro x; ring
  rw [integral_congr_ae (Filter.Eventually.of_forall hexp)]
  rw [integral_add (by exact hff.sub (hfp.const_mul 2)) hpp,
    integral_sub hff (hfp.const_mul 2), integral_const_mul]
  ring

/-- The excess vanishes exactly when the model is right almost everywhere. -/
theorem l2risk_eq_zero_iff {f p : Conf N → ℝ}
    (h : Integrable (fun x => (f x - p x) ^ 2)) :
    l2risk f p = 0 ↔ f =ᵐ[volume] p := by
  unfold l2risk
  rw [integral_eq_zero_iff_of_nonneg (fun x => sq_nonneg _) h]
  constructor
  · intro hae
    filter_upwards [hae] with x hx
    have : (f x - p x) ^ 2 = 0 := hx
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
    linarith
  · intro hae
    filter_upwards [hae] with x hx
    simp [hx]

/-! ## Invariance of the score -/

/-- If the truth is invariant, moving the model rigidly does not change its risk. -/
theorem l2risk_rigid_invariant {f p : Conf N → ℝ} (g : RigidMotion)
    (hp : ∀ x, p (g.act x) = p x) :
    l2risk (fun x => f (g.act x)) p = l2risk f p := by
  unfold l2risk
  have hstep : ∀ x : Conf N, (f (g.act x) - p x) ^ 2 = (fun y => (f y - p y) ^ 2) (g.act x) := by
    intro x
    simp only
    rw [hp x]
  rw [integral_congr_ae (Filter.Eventually.of_forall hstep)]
  exact (act_measurePreserving g).integral_comp (act_measurableEmbedding g)
    (fun y => (f y - p y) ^ 2)

/-- Integrability of the risk integrand transfers along a rigid motion. -/
lemma integrable_comp_act {f p : Conf N → ℝ} (g : RigidMotion)
    (hp : ∀ x, p (g.act x) = p x)
    (h : Integrable (fun x => (f x - p x) ^ 2)) :
    Integrable (fun x => (f (g.act x) - p x) ^ 2) := by
  have hstep : (fun x => (f (g.act x) - p x) ^ 2)
      = (fun y => (f y - p y) ^ 2) ∘ g.act := by
    funext x
    simp only [Function.comp_apply]
    rw [hp x]
  rw [hstep]
  exact (MeasurePreserving.integrable_comp_emb (ε := ℝ) (act_measurePreserving g)
    (act_measurableEmbedding g)).mpr h

/-! ## The exact price of non-equivariance -/

/-- The symmetrisation of a model over a symmetry `g` of the target. -/
noncomputable def symmetrise (f : Conf N → ℝ) (g : RigidMotion) : Conf N → ℝ :=
  fun x => (f x + f (g.act x)) / 2

/-- **The exact price of non-equivariance.**  For a symmetry `g` of the truth,

  `risk f = risk (symmetrise f g) + (1/4) * int (f - f o g)^2`.

Symmetrising is free of charge and the gain is exactly a quarter of the model's own
non-equivariance. -/
theorem symmetrization_identity {f p : Conf N → ℝ} (g : RigidMotion)
    (hp : ∀ x, p (g.act x) = p x)
    (hA : Integrable (fun x => (f x - p x) ^ 2))
    (hC : Integrable (fun x => (symmetrise f g x - p x) ^ 2)) :
    l2risk f p
      = l2risk (symmetrise f g) p + (1/4) * ∫ x, (f x - f (g.act x)) ^ 2 := by
  have hB : Integrable (fun x => (f (g.act x) - p x) ^ 2) := integrable_comp_act g hp hA
  have hBrisk : l2risk (fun x => f (g.act x)) p = l2risk f p :=
    l2risk_rigid_invariant g hp
  have hpoint : ∀ x : Conf N, (f x - f (g.act x)) ^ 2
      = 2 * (f x - p x) ^ 2 + 2 * (f (g.act x) - p x) ^ 2
        - 4 * (symmetrise f g x - p x) ^ 2 := by
    intro x
    unfold symmetrise
    ring
  have hD : ∫ x, (f x - f (g.act x)) ^ 2
      = 2 * (∫ x, (f x - p x) ^ 2) + 2 * (∫ x, (f (g.act x) - p x) ^ 2)
        - 4 * ∫ x, (symmetrise f g x - p x) ^ 2 := by
    rw [integral_congr_ae (Filter.Eventually.of_forall hpoint)]
    rw [integral_sub
        (f := fun x => 2 * (f x - p x) ^ 2 + 2 * (f (g.act x) - p x) ^ 2)
        (g := fun x => 4 * (symmetrise f g x - p x) ^ 2)
        ((hA.const_mul 2).add (hB.const_mul 2)) (hC.const_mul 4),
      integral_add (f := fun x => 2 * (f x - p x) ^ 2)
        (g := fun x => 2 * (f (g.act x) - p x) ^ 2)
        (hA.const_mul 2) (hB.const_mul 2),
      integral_const_mul, integral_const_mul, integral_const_mul]
  have hBrisk' : (∫ x, (f (g.act x) - p x) ^ 2) = ∫ x, (f x - p x) ^ 2 := hBrisk
  unfold l2risk
  rw [hD, hBrisk']
  ring

/-- Symmetrising a model over a symmetry of the target never increases its risk. -/
theorem symmetrization_improves {f p : Conf N → ℝ} (g : RigidMotion)
    (hp : ∀ x, p (g.act x) = p x)
    (hA : Integrable (fun x => (f x - p x) ^ 2))
    (hC : Integrable (fun x => (symmetrise f g x - p x) ^ 2)) :
    l2risk (symmetrise f g) p ≤ l2risk f p := by
  have hid := symmetrization_identity g hp hA hC
  have hnn : 0 ≤ ∫ x, (f x - f (g.act x)) ^ 2 :=
    integral_nonneg (fun x => sq_nonneg _)
  linarith

/-- **A model that is not equivariant is strictly beaten by its own symmetrisation.**  The
risks agree exactly when the model already respects the symmetry almost everywhere. -/
theorem equal_risk_iff_equivariant {f p : Conf N → ℝ} (g : RigidMotion)
    (hp : ∀ x, p (g.act x) = p x)
    (hA : Integrable (fun x => (f x - p x) ^ 2))
    (hC : Integrable (fun x => (symmetrise f g x - p x) ^ 2))
    (hD : Integrable (fun x => (f x - f (g.act x)) ^ 2)) :
    l2risk (symmetrise f g) p = l2risk f p ↔ (fun x => f x) =ᵐ[volume] fun x => f (g.act x) := by
  have hid := symmetrization_identity g hp hA hC
  constructor
  · intro heq
    have hzero : ∫ x, (f x - f (g.act x)) ^ 2 = 0 := by linarith
    have := (integral_eq_zero_iff_of_nonneg (fun x => sq_nonneg _) hD).mp hzero
    filter_upwards [this] with x hx
    have hx2 : (f x - f (g.act x)) ^ 2 = 0 := hx
    have := pow_eq_zero_iff (n := 2) (by norm_num) |>.mp hx2
    linarith
  · intro hae
    have hzero : ∫ x, (f x - f (g.act x)) ^ 2 = 0 := by
      refine integral_eq_zero_of_ae ?_
      filter_upwards [hae] with x hx
      simp [hx]
    linarith

end ContScore

end IDR
