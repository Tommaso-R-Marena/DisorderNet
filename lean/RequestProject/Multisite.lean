/-
# Part XVII.2  Multisite modification: single-site data do not add up

Disordered regions are the principal substrates of multisite post-translational modification,
and the standard experiment measures one modification at a time.  This file asks when that is
enough.

A modification that couples to the ensemble through an observable is an exponential tilt, so
two modifications are the doubly tilted family of `RequestProject.Linkage`.  Write
`logPart q A B lam mu = log Z(lam, mu)` and

`coupling q A B lam mu = log Z(lam,mu) + log Z(0,0) − log Z(lam,0) − log Z(0,mu)`,

the interaction free energy of the two modifications (in units of `−kT`): it is exactly the
deviation of the joint effect from the sum of the single effects.

* `coupling_eq_zero_of_constant` -- if one modification does not discriminate between
  conformations, effects are additive: the single-site experiments do predict the double.
* `coupling_pos_of_correlated` -- and in general they do not.  For an explicit two-conformation
  region in which both modifications favour the same conformation, the interaction free energy
  is strictly positive, with the exact value `log (2(e²+1)/(e+1)²)`.
* `multisite_effects_not_additive` -- so a model calibrated on singly modified samples is not
  entitled to predict the multiply modified ensemble; the coupling is a separate parameter,
  and by the linkage theorem of Part XVI it is the same parameter that governs the reciprocal
  effect.
-/
import Mathlib
import RequestProject.Linkage

namespace IDR

open Finset
open scoped Classical

namespace Multisite

variable {n : ℕ}

/-- The log partition function of the doubly modified region. -/
noncomputable def logPart (q A B : Fin n → ℝ) (lam mu : ℝ) : ℝ :=
  Real.log (Linkage.part2 q A B lam mu)

/-- The interaction free energy of the two modifications: the failure of additivity. -/
noncomputable def coupling (q A B : Fin n → ℝ) (lam mu : ℝ) : ℝ :=
  logPart q A B lam mu + logPart q A B 0 0 - logPart q A B lam 0 - logPart q A B 0 mu

/-- If the second modification does not discriminate between conformations, the two effects
are exactly additive and single-site data suffice. -/
theorem coupling_eq_zero_of_constant (hn : 0 < n) {q A B : Fin n → ℝ} (hq : ∀ j, 0 < q j)
    {c : ℝ} (hconst : ∀ j, B j = c) (lam mu : ℝ) : coupling q A B lam mu = 0 := by
  have hfac : ∀ l m : ℝ, Linkage.part2 q A B l m
      = Real.exp (m * c) * Linkage.part2 q A B l 0 := by
    intro l m
    simp only [Linkage.part2, hconst, Finset.mul_sum]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [← mul_assoc, mul_comm (Real.exp (m * c)) (q j), mul_assoc, ← Real.exp_add]
    ring_nf
  have hpos : ∀ l : ℝ, 0 < Linkage.part2 q A B l 0 := fun l =>
    Linkage.part2_pos (A := A) (B := B) hn hq l 0
  have e1 : logPart q A B lam mu = mu * c + logPart q A B lam 0 := by
    rw [logPart, hfac lam mu, Real.log_mul (Real.exp_ne_zero _) (hpos lam).ne',
      Real.log_exp, logPart]
  have e2 : logPart q A B 0 mu = mu * c + logPart q A B 0 0 := by
    rw [logPart, hfac 0 mu, Real.log_mul (Real.exp_ne_zero _) (hpos 0).ne',
      Real.log_exp, logPart]
  simp only [coupling, e1, e2]
  ring

/-! ## The witness: two conformations, two modifications favouring the same one -/

/-- Two equally populated conformations. -/
noncomputable def twoQ : Fin 2 → ℝ := fun _ => 1 / 2

/-- Both modifications couple to the same conformation. -/
def indic : Fin 2 → ℝ := ![1, 0]

lemma twoQ_pos : ∀ j, 0 < twoQ j := by intro j; simp [twoQ]

lemma part2_witness (lam mu : ℝ) :
    Linkage.part2 twoQ indic indic lam mu = (Real.exp (lam + mu) + 1) / 2 := by
  simp only [Linkage.part2, Fin.sum_univ_two, twoQ, indic, Matrix.cons_val_zero,
    Matrix.cons_val_one]
  norm_num
  ring

lemma exp_one_gt_one : (1:ℝ) < Real.exp 1 := by
  nlinarith [Real.add_one_le_exp (1:ℝ), Real.exp_pos (1:ℝ)]

/-- **Modifications do not add.**  For this region the interaction free energy of the two
modifications is strictly positive: the doubly modified ensemble is more stabilised than the
sum of the two single-site effects predicts. -/
theorem coupling_pos_of_correlated : 0 < coupling twoQ indic indic 1 1 := by
  have he : (1:ℝ) < Real.exp 1 := exp_one_gt_one
  have hZ11 : Linkage.part2 twoQ indic indic 1 1 = (Real.exp 2 + 1) / 2 := by
    rw [part2_witness]; norm_num
  have hZ00 : Linkage.part2 twoQ indic indic 0 0 = 1 := by
    rw [part2_witness]; norm_num
  have hZ10 : Linkage.part2 twoQ indic indic 1 0 = (Real.exp 1 + 1) / 2 := by
    rw [part2_witness]; norm_num
  have hZ01 : Linkage.part2 twoQ indic indic 0 1 = (Real.exp 1 + 1) / 2 := by
    rw [part2_witness]; norm_num
  have hexp2 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by
    rw [← Real.exp_add]; norm_num
  have h11 : (0:ℝ) < (Real.exp 2 + 1) / 2 := by positivity
  have h10 : (0:ℝ) < (Real.exp 1 + 1) / 2 := by positivity
  simp only [coupling, logPart, hZ11, hZ00, hZ10, hZ01, Real.log_one]
  have hkey : 2 * Real.log ((Real.exp 1 + 1) / 2) < Real.log ((Real.exp 2 + 1) / 2) := by
    have hsq : Real.log ((Real.exp 1 + 1) / 2) * 2
        = Real.log (((Real.exp 1 + 1) / 2) ^ 2) := by
      rw [Real.log_pow]; push_cast; ring
    have hlt : ((Real.exp 1 + 1) / 2) ^ 2 < (Real.exp 2 + 1) / 2 := by
      rw [hexp2]
      nlinarith [he]
    have := Real.log_lt_log (by positivity) hlt
    rw [← hsq] at this
    linarith
  linarith

/-- The design statement: single-site experiments do not determine the multiply modified
ensemble.  There is a region and a pair of modifications whose joint free-energy effect
differs from the sum of the single effects. -/
theorem multisite_effects_not_additive :
    ∃ (q A B : Fin 2 → ℝ) (lam mu : ℝ),
      (∀ j, 0 < q j) ∧
      logPart q A B lam mu + logPart q A B 0 0
        ≠ logPart q A B lam 0 + logPart q A B 0 mu :=
  ⟨twoQ, indic, indic, 1, 1, twoQ_pos, by
    have h := coupling_pos_of_correlated
    simp only [coupling] at h
    intro hEq
    linarith⟩

end Multisite

end IDR
