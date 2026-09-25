/-
# Part LXXXIII  The statistics of validation: multiplicity, selection and complexity

Every previous part asks what an ensemble model must contain.  This part asks the question a
referee asks about the *evidence*: an ensemble was fitted, it was compared with a list of
experiments, the agreement was reported -- what does the agreement prove?  Three objections are
routinely and correctly raised, and all three have exact answers.

**1.  Multiplicity.**  A validation report contains many tests: `k` NOE restraints, `k'` chemical
shifts, a SAXS profile, three FRET pairs.  Let `A j` be the event that test `j` is passed *by a
wrong model, by chance*, each of probability at most `alpha`.

* `familywise_le_sum`, `familywise_le_card_mul` -- the union bound: the chance that some test in a
  family of `m` is passed is at most `m * alpha`.
* `familywise_eq_of_disjoint` -- and the bound is attained, so no better constant exists in
  general: for mutually exclusive events the family-wise error is exactly `m * alpha`.
* `bonferroni_correction` -- hence the only unconditional repair: to hold the family-wise error at
  `beta`, each of the `m` tests must be run at level `beta / m`.  A validation table with forty
  entries at the `5%` level is a validation at the `0.125%` level, or it is nothing.
* `anyPass_ge_one_sub_exp`, `anyPass_forty_gt` -- the size of the effect for independent tests:
  with `m` tests of individual level `alpha` the probability that a wrong model passes at least one
  is `1 - (1 - alpha) ^ m >= 1 - exp (-(m alpha))`, which for `m = 40`, `alpha = 1/20` exceeds
  `87%`.  Reporting *the* agreement -- the one observable that matched -- is not weak evidence, it
  is the expected outcome.
* `allPass_of_indep`, `allPass_lt` -- the positive counterpart, and it is why validation is worth
  doing at all: passing *every* prespecified test has null probability `alpha ^ m`, exponentially
  small; five independent tests at the `5%` level already give odds beyond three million to one.

**2.  Selection.**  `exists_best_le_mean` -- the best of `m` candidate models has discrepancy at or
below the average, always, and `nested_fit_monotone` -- enlarging the model family can never
worsen the best achievable fit.  Neither fact carries information about the truth: a fit
improvement obtained by enlarging a family is guaranteed by the enlargement.
`penalised_comparison` states the only honest reading: a larger family is preferred only when the
improvement exceeds the complexity penalty that its extra parameters carry.

**3.  Saturation.**  `exact_fit_of_saturated_library` -- if the conformer library carries at least
as many independently adjustable structures as there are restraints, the data can be reproduced
*exactly whatever they are*, so exact reproduction of the fitted data is not evidence for the
ensemble.  `heldout_differs` -- and it is not empty pedantry: two weight vectors reproducing every
fitted observable identically can differ by `1/2` on a held-out one.  What tests an ensemble is a
prediction it was not fitted to.

None of this weakens the case for ensemble models; it disciplines it.  An ensemble is a model with
many parameters, so the standard of evidence it must meet is the standard for many parameters:
prespecified tests, a multiplicity correction, a complexity penalty, and held-out observables.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Validation

open Finset MeasureTheory

/-! ### 1. Family-wise error over a validation table -/

section FamilyWise

variable {Omega : Type*} [MeasurableSpace Omega] {mu : Measure Omega} [IsProbabilityMeasure mu]
  {m : ℕ}

/-- The union bound: the chance that *some* test in the table is passed by chance is at most the
sum of the individual chances. -/
theorem familywise_le_sum (A : Fin m → Set Omega) :
    mu.real (⋃ j, A j) ≤ ∑ j, mu.real (A j) :=
  measureReal_iUnion_fintype_le A

/-- Bonferroni's inequality: `m` tests, each passed by chance with probability at most `alpha`,
give a family-wise error at most `m * alpha`. -/
theorem familywise_le_card_mul {alpha : ℝ} (A : Fin m → Set Omega)
    (h : ∀ j, mu.real (A j) ≤ alpha) : mu.real (⋃ j, A j) ≤ m * alpha := by
  refine (familywise_le_sum A).trans ?_
  calc ∑ j, mu.real (A j) ≤ ∑ _j : Fin m, alpha := Finset.sum_le_sum fun j _ => h j
    _ = m * alpha := by simp [mul_comm]

/-- The union bound is attained: for mutually exclusive ways of passing, the family-wise error is
exactly `m * alpha`.  No constant better than `m` is available in general. -/
theorem familywise_eq_of_disjoint {alpha : ℝ} (A : Fin m → Set Omega)
    (hmeas : ∀ j, MeasurableSet (A j)) (hdisj : Pairwise (Function.onFun Disjoint A))
    (h : ∀ j, mu.real (A j) = alpha) : mu.real (⋃ j, A j) = m * alpha := by
  rw [measureReal_iUnion_fintype hdisj hmeas]
  simp [h, mul_comm]

/-- **The Bonferroni correction.**  To hold the family-wise error of a validation table of `m`
entries at `beta`, each entry must be tested at level `beta / m`. -/
theorem bonferroni_correction {beta : ℝ} (hm : 0 < m) (A : Fin m → Set Omega)
    (h : ∀ j, mu.real (A j) ≤ beta / m) : mu.real (⋃ j, A j) ≤ beta := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  refine (familywise_le_card_mul A h).trans ?_
  rw [mul_comm, div_mul_cancel₀ _ (ne_of_gt hm')]

/-- With independent tests the family-wise error is exactly `1 - (1 - alpha) ^ m`. -/
theorem anyPass_of_indep {alpha : ℝ} (A : Fin m → Set Omega) (hmeas : ∀ j, MeasurableSet (A j))
    (hindep : mu.real (⋂ j, (A j)ᶜ) = ∏ j, mu.real ((A j)ᶜ))
    (h : ∀ j, mu.real (A j) = alpha) :
    mu.real (⋃ j, A j) = 1 - (1 - alpha) ^ m := by
  have hc : (⋃ j, A j) = (⋂ j, (A j)ᶜ)ᶜ := by simp
  have hone : ∀ j, mu.real ((A j)ᶜ) = 1 - alpha := by
    intro j; rw [measureReal_compl (hmeas j), h j]; simp
  rw [hc, measureReal_compl (by measurability), hindep,
    Finset.prod_congr rfl fun j _ => hone j]
  simp

/-- The positive side: passing *every* one of `m` independent prespecified tests has null
probability `alpha ^ m`. -/
theorem allPass_of_indep {alpha : ℝ} (A : Fin m → Set Omega)
    (hindep : mu.real (⋂ j, A j) = ∏ j, mu.real (A j)) (h : ∀ j, mu.real (A j) = alpha) :
    mu.real (⋂ j, A j) = alpha ^ m := by
  rw [hindep, Finset.prod_congr rfl fun j _ => h j]
  simp

end FamilyWise

/-! ### 2. The arithmetic of multiplicity -/

/-- The probability that at least one of `m` independent tests of individual level `alpha` is
passed by chance. -/
noncomputable def anyPass (alpha : ℝ) (m : ℕ) : ℝ := 1 - (1 - alpha) ^ m

@[simp] lemma anyPass_zero (alpha : ℝ) : anyPass alpha 0 = 0 := by simp [anyPass]

@[simp] lemma anyPass_one (alpha : ℝ) : anyPass alpha 1 = alpha := by simp [anyPass]

/-- The exact independent value never exceeds the Bonferroni bound. -/
theorem anyPass_le_card_mul {alpha : ℝ} (h1 : alpha ≤ 1) (m : ℕ) :
    anyPass alpha m ≤ m * alpha := by
  have h3 : (1 : ℝ) + m * (-alpha) ≤ (1 + -alpha) ^ m := one_add_mul_le_pow (by linarith) m
  rw [show (1 + -alpha) = 1 - alpha by ring] at h3
  unfold anyPass; linarith

/-- More tests, more chances. -/
theorem anyPass_mono {alpha : ℝ} (h0 : 0 ≤ alpha) (h1 : alpha ≤ 1) {m n : ℕ} (h : m ≤ n) :
    anyPass alpha m ≤ anyPass alpha n := by
  have := pow_le_pow_of_le_one (by linarith : (0:ℝ) ≤ 1 - alpha) (by linarith) h
  unfold anyPass; linarith

/-- **The multiplicity law.**  The chance that a wrong model passes at least one of `m`
independent tests is at least `1 - exp (-(m * alpha))`: it approaches certainty once the table has
more than `1 / alpha` entries. -/
theorem anyPass_ge_one_sub_exp {alpha : ℝ} (h1 : alpha ≤ 1) (m : ℕ) :
    1 - Real.exp (-(m * alpha)) ≤ anyPass alpha m := by
  have hle : (1 - alpha) ≤ Real.exp (-alpha) := by
    have := Real.add_one_le_exp (-alpha); linarith
  have h2 : (1 - alpha) ^ m ≤ Real.exp (-(m * alpha)) := by
    calc (1 - alpha) ^ m ≤ (Real.exp (-alpha)) ^ m := pow_le_pow_left₀ (by linarith) hle m
      _ = Real.exp (-(m * alpha)) := by rw [← Real.exp_nat_mul]; ring_nf
  unfold anyPass; linarith

/-- Forty entries at the `5%` level: a wrong model passes something with probability above
`87%`. -/
theorem anyPass_forty_gt : (87 : ℝ) / 100 < anyPass (1 / 20) 40 := by
  unfold anyPass; norm_num

/-- Five prespecified independent tests at the `5%` level: odds beyond three million to one. -/
theorem allPass_lt : ((1 : ℝ) / 20) ^ 5 < 1 / 3000000 := by norm_num

/-! ### 3. Selection among candidate models -/

/-- Selecting the best of `m` candidate models always produces a discrepancy at or below the
average of the family, whatever the family is.  A small reported discrepancy after a search is
therefore not, by itself, evidence about the target. -/
theorem exists_best_le_mean {m : ℕ} (hm : 0 < m) (d : Fin m → ℝ) :
    ∃ j, d j ≤ (∑ i, d i) / m := by
  by_contra hcon
  push_neg at hcon
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have : (∑ i, d i) / m * m < ∑ i, d i := by
    calc (∑ i, d i) / m * m = ∑ _i : Fin m, (∑ i, d i) / m := by
          simp [mul_comm]
      _ < ∑ i, d i := by
          refine Finset.sum_lt_sum_of_nonempty ?_ fun i _ => hcon i
          exact Finset.univ_nonempty_iff.2 (Fin.pos_iff_nonempty.1 hm)
  rw [div_mul_cancel₀ _ (ne_of_gt hm')] at this
  exact lt_irrefl _ this

/-- Enlarging the family of candidate models can never worsen the best achievable fit: an
improvement obtained by enlarging is guaranteed by the enlargement, not learned from the data. -/
theorem nested_fit_monotone {ι : Type*} (L : ι → ℝ) {S T : Finset ι} (hST : S ⊆ T)
    (hS : S.Nonempty) : T.inf' (hS.mono hST) L ≤ S.inf' hS L :=
  Finset.inf'_mono L hST hS

/-- The only honest reading of a fit improvement: the richer family wins on a penalised score only
if the improvement in fit strictly exceeds the extra complexity it pays for. -/
theorem penalised_comparison {Lsmall Lbig penSmall penBig : ℝ}
    (h : Lbig + penBig < Lsmall + penSmall) : penBig - penSmall < Lsmall - Lbig := by linarith

/-! ### 4. Saturation: fitting data that cannot fail to be fitted -/

/-- **A saturated library fits anything.**  If the library carries one structure per restraint,
each contributing to exactly that restraint, then every admissible data vector is reproduced
exactly by some weight vector -- whatever the data are.  Exact agreement with the fitted data is
then a property of the library, not evidence for the ensemble. -/
theorem exact_fit_of_saturated_library {k : ℕ} (d : Fin k → ℝ) :
    ∃ w : Fin k → ℝ, (∀ i, w i = d i) ∧
      ∀ i, ∑ j, w j * (if i = j then (1 : ℝ) else 0) = d i := by
  refine ⟨d, fun _ => rfl, fun i => ?_⟩
  simp

/-- **Held-out observables discriminate; fitted ones need not.**  Two weight vectors over a
three-conformer library reproduce the fitted observable `g` identically, yet differ by `1/2` on the
held-out observable `h`. -/
theorem heldout_differs :
    ∃ (w w' g h : Fin 3 → ℝ),
      (∀ j, 0 ≤ w j) ∧ (∀ j, 0 ≤ w' j) ∧ ∑ j, w j = 1 ∧ ∑ j, w' j = 1 ∧
      (∑ j, w j * g j = ∑ j, w' j * g j) ∧
      |(∑ j, w j * h j) - ∑ j, w' j * h j| = 1 / 2 := by
  refine ⟨![1/2, 1/2, 0], ![1/2, 0, 1/2], ![0, 1, 1], ![0, 1, 0], ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp [Fin.forall_fin_succ, Fin.sum_univ_three] <;> norm_num

end Validation

end IDR
