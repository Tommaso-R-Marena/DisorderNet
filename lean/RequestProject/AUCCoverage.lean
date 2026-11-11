/-
# Theorem 3 — coverage bias: what declining a subset of proteins does to the pooled AUC

A method that declines to predict on a set `S` of proteins is not scored on the same pair set as
one that predicts everywhere: the benchmark silently restricts to the pairs both of whose members
lie in retained proteins.  This file gives the exact arithmetic of that restriction.

Write `X` for the pooled pair set, `R ⊆ X` for the retained pairs and `D = X \ R` for the
discarded ones.

* `auc_restrict_diff` — **the exact change**
  `AUC(R) − AUC(X) = (|D| / |X|) · (AUC(R) − AUC(D))`:
  restricting moves the reported number by the *discarded pair fraction* times the AUC gap
  between the pairs kept and the pairs thrown away.
* `auc_restrict_increase_iff` — hence declining **strictly increases** the reported AUC exactly
  when the method's AUC on the discarded pairs is strictly below its AUC on the retained ones,
  and leaves it unchanged exactly when the two agree.  No assumption about *why* the proteins
  were declined is needed; only this comparison matters.
* `auc_restrict_gain_le`, `discarded_fraction_lower_bound` — **the bound**.  The gain is at most
  `(|D|/|X|)·(1 − AUC(D)) ≤ |D|/|X|`.  So an observed coverage gain of `g` is impossible unless
  the method declined at least a fraction `g` of all comparison pairs: a measured gain is a
  lower bound on how much of the benchmark the method refused.
* `keptPairs`, `declinedPairs`, `auc_decline_diff`, `auc_decline_increase_iff`,
  `decline_fraction_lower_bound` — the same statements for the concrete case where the discarded
  pairs are exactly those touching a declined set `S` of proteins.
-/
import RequestProject.AUCCore

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Restrict

variable {I G : Type*} [Fintype I] [DecidableEq I] [DecidableEq G]

omit [Fintype I] in
lemma U_sdiff_add (X R : Finset (I × I)) (hR : R ⊆ X) (s : I → ℝ) :
    U R s + U (X \ R) s = U X s := by
  unfold U
  rw [add_comm]
  exact Finset.sum_sdiff hR

omit [Fintype I] in
lemma card_sdiff_add (X R : Finset (I × I)) (hR : R ⊆ X) :
    R.card + (X \ R).card = X.card := by
  rw [add_comm]
  exact Finset.card_sdiff_add_card_eq_card hR

omit [Fintype I] in
/-- **Theorem 3 (exact change).**  Restricting the pooled AUC from `X` to a subset `R` of the
pairs changes it by the discarded fraction times the AUC gap between kept and discarded pairs. -/
theorem auc_restrict_diff (X R : Finset (I × I)) (hR : R ⊆ X) (s : I → ℝ)
    (hRne : R.card ≠ 0) (hDne : (X \ R).card ≠ 0) :
    auc R s - auc X s = (((X \ R).card : ℝ) / X.card) * (auc R s - auc (X \ R) s) := by
  have hcard : (R.card : ℝ) + ((X \ R).card : ℝ) = (X.card : ℝ) := by
    exact_mod_cast congrArg (fun n : ℕ => (n : ℝ)) (card_sdiff_add X R hR)
  have hRc : (R.card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hRne
  have hDc : ((X \ R).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hDne
  have hXc : (X.card : ℝ) ≠ 0 := by
    rw [← hcard]
    positivity
  have hU : U R s + U (X \ R) s = U X s := U_sdiff_add X R hR s
  unfold auc
  rw [← hU, ← hcard]
  field_simp
  ring

omit [Fintype I] in
/-- **Theorem 3 (direction).**  Declining strictly increases the reported pooled AUC exactly when
the AUC on the discarded pairs is strictly smaller than the AUC on the retained pairs. -/
theorem auc_restrict_increase_iff (X R : Finset (I × I)) (hR : R ⊆ X) (s : I → ℝ)
    (hRne : R.card ≠ 0) (hDne : (X \ R).card ≠ 0) :
    auc X s < auc R s ↔ auc (X \ R) s < auc R s := by
  have hDc : (0 : ℝ) < ((X \ R).card : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero hDne
  have hXc : (0 : ℝ) < (X.card : ℝ) := by
    have : 0 < X.card := lt_of_lt_of_le (Nat.pos_of_ne_zero hDne) (Finset.card_le_card
      (Finset.sdiff_subset))
    exact_mod_cast this
  have h := auc_restrict_diff X R hR s hRne hDne
  constructor
  · intro hlt
    have h0 : 0 < auc R s - auc X s := by linarith
    rw [h] at h0
    have hpos : (0 : ℝ) < ((X \ R).card : ℝ) / X.card := by positivity
    nlinarith [h0, hpos]
  · intro hlt
    have h0 : 0 < auc R s - auc (X \ R) s := by linarith
    have : 0 < (((X \ R).card : ℝ) / X.card) * (auc R s - auc (X \ R) s) := by positivity
    rw [← h] at this
    linarith

omit [Fintype I] in
/-- **Theorem 3 (bound).**  The coverage gain is at most the discarded pair fraction times the
headroom on the discarded pairs, hence at most the discarded pair fraction itself. -/
theorem auc_restrict_gain_le (X R : Finset (I × I)) (hR : R ⊆ X) (s : I → ℝ)
    (hRne : R.card ≠ 0) (hDne : (X \ R).card ≠ 0) :
    auc R s - auc X s ≤ (((X \ R).card : ℝ) / X.card) * (1 - auc (X \ R) s)
      ∧ auc R s - auc X s ≤ ((X \ R).card : ℝ) / X.card := by
  have hXc : (0 : ℝ) < (X.card : ℝ) := by
    have : 0 < X.card := lt_of_lt_of_le (Nat.pos_of_ne_zero hDne) (Finset.card_le_card
      (Finset.sdiff_subset))
    exact_mod_cast this
  have hfrac : (0 : ℝ) ≤ ((X \ R).card : ℝ) / X.card := by positivity
  have hfrac1 : ((X \ R).card : ℝ) / X.card ≤ 1 := by
    rw [div_le_one hXc]
    exact_mod_cast Finset.card_le_card (Finset.sdiff_subset)
  have h := auc_restrict_diff X R hR s hRne hDne
  have hR1 : auc R s ≤ 1 := auc_le_one _ _
  have hD0 : 0 ≤ auc (X \ R) s := auc_nonneg _ _
  constructor
  · rw [h]
    exact mul_le_mul_of_nonneg_left (by linarith) hfrac
  · rw [h]
    have : auc R s - auc (X \ R) s ≤ 1 := by linarith
    calc (((X \ R).card : ℝ) / X.card) * (auc R s - auc (X \ R) s)
        ≤ (((X \ R).card : ℝ) / X.card) * 1 := mul_le_mul_of_nonneg_left this hfrac
      _ = ((X \ R).card : ℝ) / X.card := by ring

omit [Fintype I] in
/-- A measured coverage gain is a lower bound on the fraction of comparison pairs the method
declined: to gain `g` you must throw away at least a fraction `g` of the pairs. -/
theorem discarded_fraction_lower_bound (X R : Finset (I × I)) (hR : R ⊆ X) (s : I → ℝ)
    (hRne : R.card ≠ 0) (hDne : (X \ R).card ≠ 0) (g : ℝ)
    (hg : g ≤ auc R s - auc X s) :
    g ≤ ((X \ R).card : ℝ) / X.card :=
  le_trans hg (auc_restrict_gain_le X R hR s hRne hDne).2

end Restrict

/-! ## Declining a set of proteins -/

section Decline

variable {I G : Type*} [Fintype I] [DecidableEq I] [DecidableEq G]

/-- The pairs a method that declines the proteins in `S` is still scored on. -/
def keptPairs (lab : I → Bool) (grp : I → G) (S : Finset G) : Finset (I × I) :=
  (allPairs lab).filter (fun q => grp q.1 ∉ S ∧ grp q.2 ∉ S)

/-- The pairs it is no longer scored on: those touching a declined protein. -/
def declinedPairs (lab : I → Bool) (grp : I → G) (S : Finset G) : Finset (I × I) :=
  (allPairs lab).filter (fun q => ¬ (grp q.1 ∉ S ∧ grp q.2 ∉ S))

omit [DecidableEq I] in
lemma keptPairs_subset (lab : I → Bool) (grp : I → G) (S : Finset G) :
    keptPairs lab grp S ⊆ allPairs lab := Finset.filter_subset _ _

lemma sdiff_keptPairs (lab : I → Bool) (grp : I → G) (S : Finset G) :
    allPairs lab \ keptPairs lab grp S = declinedPairs lab grp S := by
  unfold keptPairs declinedPairs
  ext q
  simp only [Finset.mem_sdiff, Finset.mem_filter, not_and]
  tauto

/-- **Theorem 3 for protein declination.**  The exact change in pooled AUC caused by declining the
proteins in `S`. -/
theorem auc_decline_diff (lab : I → Bool) (grp : I → G) (S : Finset G) (s : I → ℝ)
    (hK : (keptPairs lab grp S).card ≠ 0) (hD : (declinedPairs lab grp S).card ≠ 0) :
    auc (keptPairs lab grp S) s - auc (allPairs lab) s
      = (((declinedPairs lab grp S).card : ℝ) / (allPairs lab).card)
          * (auc (keptPairs lab grp S) s - auc (declinedPairs lab grp S) s) := by
  have h := auc_restrict_diff (allPairs lab) (keptPairs lab grp S)
    (keptPairs_subset lab grp S) s hK (by rwa [sdiff_keptPairs])
  rwa [sdiff_keptPairs] at h

/-- Declining strictly inflates the benchmark number exactly when the method is worse on the pairs
it declined than on the pairs it kept. -/
theorem auc_decline_increase_iff (lab : I → Bool) (grp : I → G) (S : Finset G) (s : I → ℝ)
    (hK : (keptPairs lab grp S).card ≠ 0) (hD : (declinedPairs lab grp S).card ≠ 0) :
    auc (allPairs lab) s < auc (keptPairs lab grp S) s
      ↔ auc (declinedPairs lab grp S) s < auc (keptPairs lab grp S) s := by
  have h := auc_restrict_increase_iff (allPairs lab) (keptPairs lab grp S)
    (keptPairs_subset lab grp S) s hK (by rwa [sdiff_keptPairs])
  rwa [sdiff_keptPairs] at h

/-- A coverage gain of `g` forces at least a fraction `g` of all pairs to have been declined. -/
theorem decline_fraction_lower_bound (lab : I → Bool) (grp : I → G) (S : Finset G) (s : I → ℝ)
    (hK : (keptPairs lab grp S).card ≠ 0) (hD : (declinedPairs lab grp S).card ≠ 0) (g : ℝ)
    (hg : g ≤ auc (keptPairs lab grp S) s - auc (allPairs lab) s) :
    g ≤ ((declinedPairs lab grp S).card : ℝ) / (allPairs lab).card := by
  have h := discarded_fraction_lower_bound (allPairs lab) (keptPairs lab grp S)
    (keptPairs_subset lab grp S) s hK (by rwa [sdiff_keptPairs]) g hg
  rwa [sdiff_keptPairs] at h

end Decline

end IDR.GroupedAUC
