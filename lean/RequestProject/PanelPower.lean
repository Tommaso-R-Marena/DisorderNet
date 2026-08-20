/-
# Part LXXXIX.4  A panel of systems: multiplicity, and the power of the whole study

One system, one threshold, one confirmation is a design rule.  What decides whether a
capacity law is worth anyone's compute is whether it fires selectively across a *panel* of
systems with different measured state counts — and a panel raises the question every
methodologically careful referee asks next: with several tests, what is the family-wise error?

This file supplies the two statements the panel needs.

* `one_sub_sum_le_prod` — Weierstrass' product inequality, the elementary fact behind the
  Bonferroni split.
* `panel_power` — if each system of an `n`-system panel is run at level `alpha/n`, i.e. each
  run's probability of failing to refute its under-capacity baseline is at most `alpha/n`,
  then the probability that *every* system in the panel refutes is at least `1 - alpha`.  The
  per-system sample size that achieves this is `IDR.Power.samplesFor (alpha/n) τᵢ`, computed
  from each system's independently measured tail.
* `panel_samples_suffice` — the same statement with the sample sizes filled in: freeze each
  record at level `alpha/n`, run `samplesFor (alpha/n) τᵢ` observations on system `i`, and the
  whole panel refutes with probability at least `1 - alpha`.

Independence across systems is an assumption, stated explicitly as the hypothesis that the
joint probability of failing on every system is the product of the per-system probabilities;
separate preparations of different proteins is the case it is meant for.
-/
import Mathlib
import RequestProject.DetectionPower

set_option autoImplicit false

namespace IDR
namespace Panel

open Finset

/-- **Weierstrass' product inequality.** -/
lemma one_sub_sum_le_prod' {ι : Type*} (s : Finset ι) (x : ι → ℝ)
    (h0 : ∀ i ∈ s, 0 ≤ x i) (h1 : ∀ i ∈ s, x i ≤ 1) :
    1 - ∑ i ∈ s, x i ≤ ∏ i ∈ s, (1 - x i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp
  | insert a s ha ih =>
      have h0' : ∀ i ∈ s, 0 ≤ x i := fun i hi => h0 i (Finset.mem_insert_of_mem hi)
      have h1' : ∀ i ∈ s, x i ≤ 1 := fun i hi => h1 i (Finset.mem_insert_of_mem hi)
      have hIH := ih h0' h1'
      have hsum0 : 0 ≤ ∑ i ∈ s, x i := Finset.sum_nonneg h0'
      have hxa0 : 0 ≤ x a := h0 a (Finset.mem_insert_self a s)
      have hxa1 : x a ≤ 1 := h1 a (Finset.mem_insert_self a s)
      rw [Finset.prod_insert ha, Finset.sum_insert ha]
      nlinarith [hIH, hsum0, hxa0, hxa1]

/-- **Weierstrass' product inequality**, over a finite index type. -/
lemma one_sub_sum_le_prod {n : ℕ} (x : Fin n → ℝ) (h0 : ∀ i, 0 ≤ x i) (h1 : ∀ i, x i ≤ 1) :
    1 - ∑ i, x i ≤ ∏ i, (1 - x i) :=
  one_sub_sum_le_prod' Finset.univ x (fun i _ => h0 i) (fun i _ => h1 i)

/-- **The power of the panel, with multiplicity controlled.**  Run each of the `n` systems at
level `alpha/n` — that is, take enough observations on system `i` that its probability
`miss i` of failing to refute the under-capacity baseline is at most `alpha/n`.  Then the
probability that every system in the panel refutes is at least `1 - alpha`.  The Bonferroni
split is paid for by the sample sizes, which are known in advance from the measured tails. -/
theorem panel_power {n : ℕ} (hn : 0 < n) (miss : Fin n → ℝ) {alpha : ℝ}
    (halpha1 : alpha ≤ 1) (h0 : ∀ i, 0 ≤ miss i) (h : ∀ i, miss i ≤ alpha / (n : ℝ)) :
    1 - alpha ≤ ∏ i, (1 - miss i) := by
  have hnR : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have h1 : ∀ i, miss i ≤ 1 := by
    intro i
    refine le_trans (h i) ?_
    rw [div_le_one hnR]
    have : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
    linarith
  have hsum : ∑ i, miss i ≤ alpha := by
    calc ∑ i, miss i ≤ ∑ _i : Fin n, alpha / (n : ℝ) := Finset.sum_le_sum fun i _ => h i
      _ = alpha := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          field_simp
  have := one_sub_sum_le_prod miss h0 h1
  linarith

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- **The panel design, with the sample sizes filled in.**  Suppose system `i` of the panel
has a measured tail `tau i > 0` outside the baseline's component count, and is observed
`samplesFor (alpha/n) (tau i)` times, so that its miss probability obeys the exact law of
`IDR.Power.missProb_le_of_mass_ge`.  Then the whole panel refutes with probability at least
`1 - alpha`, family-wise. -/
theorem panel_samples_suffice {n : ℕ} (hn : 0 < n) {alpha : ℝ} (ha : 0 < alpha)
    (halpha1 : alpha ≤ 1) (tau : Fin n → ℝ) (htau : ∀ i, 0 < tau i) (htau1 : ∀ i, tau i ≤ 1)
    (runs : Fin n → ℕ) (hruns : ∀ i, Power.samplesFor (alpha / (n : ℝ)) (tau i) ≤ runs i)
    (miss : Fin n → ℝ) (hmiss0 : ∀ i, 0 ≤ miss i)
    (hmiss : ∀ i, miss i ≤ (1 - tau i) ^ (runs i)) :
    1 - alpha ≤ ∏ i, (1 - miss i) := by
  have hnR : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  refine panel_power hn miss halpha1 hmiss0 fun i => ?_
  refine le_trans (hmiss i) ?_
  exact Power.samplesFor_spec (div_pos ha hnR) (htau i) (htau1 i) (hruns i)

end Panel
end IDR
