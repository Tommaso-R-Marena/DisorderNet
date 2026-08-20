/-
# Part XIV.2  The irreducible floor: why raw scores are not comparable across regions

A strictly proper score (Part XIV.1) ranks models correctly *against a fixed target*.  It does
not, however, measure difficulty: the same score answers a different question on an ordered
domain and on a disordered one.  This file quantifies that exactly.

* `brier_decomposition` : `expScore brier p q = ∑ x (p x - q x)² + gini q`, where
  `gini q = 1 - ∑ x q x ^ 2` depends on the *target alone*.  So the score splits into the
  model's squared population error and an irreducible floor.
* `brier_floor` : the perfect model still scores `gini q`.
* `gini_eq_zero_iff` : the floor vanishes exactly on an ordered (single-conformation) target,
  and `gini_le_one_sub_inv_card` / `gini_unif` : it is largest, `1 - 1/|X|`, for the maximally
  disordered one.  Numbers: `0` for an ordered region, `2/3` for three equally populated
  rotamers.
* `benchmark_confounded` : an *exactly correct* model of a disordered region scores worse than
  a *wrong* model of an ordered one.  Raw benchmark numbers therefore rank regions, not
  models; only the excess over the floor -- which by `brier_decomposition` is exactly the
  squared population error -- is a measure of model quality.
-/
import Mathlib
import RequestProject.Scoring

namespace IDR

namespace Scoring

open Finset
open scoped BigOperators Classical

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- The intrinsic uncertainty of a target: `1 - ∑ q x ^ 2`, the probability that two
independent draws from the ensemble give different conformations. -/
def gini (q : X → ℝ) : ℝ := 1 - ∑ x, q x ^ 2

/-- **Score = model error + irreducible floor.**  The quadratic score of a prediction splits
into the squared error of the predicted populations, which is the model's responsibility, and
a term depending only on how disordered the target is. -/
theorem brier_decomposition (p q : X → ℝ) (hq : IsProbVec q) :
    expScore brier p q = (∑ x, (p x - q x) ^ 2) + gini q := by
  have hfloor : expScore brier q q = gini q := by
    rw [expScore_brier q q hq, gini]
    have h : ∀ x : X, q x * q x = q x ^ 2 := fun x => (sq (q x)).symm
    simp only [h]
    ring
  have := brier_excess p q hq
  rw [hfloor] at this
  linarith

/-- Even an exactly correct model pays the floor. -/
theorem brier_floor (q : X → ℝ) (hq : IsProbVec q) : expScore brier q q = gini q := by
  rw [brier_decomposition q q hq]
  simp

omit [DecidableEq X] in
lemma gini_nonneg {q : X → ℝ} (hq : IsProbVec q) : 0 ≤ gini q := by
  have hle : ∑ x, q x ^ 2 ≤ ∑ x, q x := by
    refine Finset.sum_le_sum fun x _ => ?_
    have h1 : q x ≤ 1 := by
      by_contra hcon
      push_neg at hcon
      have : ∑ y ∈ Finset.univ.erase x, q y + q x ≤ 1 := by
        rw [Finset.sum_erase_add _ _ (Finset.mem_univ x), hq.2]
      have hpos : 0 ≤ ∑ y ∈ Finset.univ.erase x, q y :=
        Finset.sum_nonneg fun y _ => hq.1 y
      linarith
    nlinarith [hq.1 x]
  rw [gini, hq.2] at *
  linarith

/-- The floor vanishes exactly on an ordered target: one conformation carrying everything. -/
theorem gini_eq_zero_iff {q : X → ℝ} (hq : IsProbVec q) :
    gini q = 0 ↔ ∃ x₀ : X, ∀ x : X, q x = if x = x₀ then 1 else 0 := by
  constructor
  · intro h
    have hsum : ∑ x, q x ^ 2 = 1 := by rw [gini] at h; linarith
    -- some conformation carries positive weight
    have hex : ∃ x₀ : X, q x₀ ≠ 0 := by
      by_contra hcon
      push_neg at hcon
      have : (1 : ℝ) = 0 := by
        rw [← hq.2]; exact Finset.sum_eq_zero fun x _ => hcon x
      norm_num at this
    obtain ⟨x₀, hx₀⟩ := hex
    have hx₀pos : 0 < q x₀ := lt_of_le_of_ne (hq.1 x₀) (Ne.symm hx₀)
    -- every weight is at most one, so `q x ^ 2 ≤ q x` with equality only at `0` and `1`
    have hle : ∀ x : X, q x ≤ 1 := by
      intro x
      have hrest : 0 ≤ ∑ y ∈ Finset.univ.erase x, q y :=
        Finset.sum_nonneg fun y _ => hq.1 y
      have : ∑ y ∈ Finset.univ.erase x, q y + q x = 1 := by
        rw [Finset.sum_erase_add _ _ (Finset.mem_univ x), hq.2]
      linarith
    have hterm : ∀ x : X, q x ^ 2 ≤ q x := fun x => by nlinarith [hq.1 x, hle x]
    have heq : ∀ x : X, q x ^ 2 = q x := by
      intro x
      by_contra hne
      have hlt : q x ^ 2 < q x := lt_of_le_of_ne (hterm x) hne
      have : ∑ y, q y ^ 2 < ∑ y, q y :=
        Finset.sum_lt_sum (fun y _ => hterm y) ⟨x, Finset.mem_univ x, hlt⟩
      rw [hsum, hq.2] at this
      exact absurd this (lt_irrefl 1)
    have hone : q x₀ = 1 := by
      have := heq x₀
      have : q x₀ * (q x₀ - 1) = 0 := by nlinarith
      rcases mul_eq_zero.1 this with h' | h'
      · exact absurd h' hx₀
      · linarith
    refine ⟨x₀, fun x => ?_⟩
    by_cases hx : x = x₀
    · simp [hx, hone]
    · have hrest : q x + ∑ y ∈ (Finset.univ.erase x₀).erase x, q y
          = ∑ y ∈ Finset.univ.erase x₀, q y := by
        refine Finset.add_sum_erase _ _ ?_
        exact Finset.mem_erase.2 ⟨hx, Finset.mem_univ x⟩
      have htot : ∑ y ∈ Finset.univ.erase x₀, q y + q x₀ = 1 := by
        rw [Finset.sum_erase_add _ _ (Finset.mem_univ x₀), hq.2]
      have hzero : ∑ y ∈ Finset.univ.erase x₀, q y = 0 := by rw [hone] at htot; linarith
      have hnn : 0 ≤ ∑ y ∈ (Finset.univ.erase x₀).erase x, q y :=
        Finset.sum_nonneg fun y _ => hq.1 y
      have : q x = 0 := by
        have := hq.1 x
        linarith [hrest, hzero]
      simp [hx, this]
  · rintro ⟨x₀, hx₀⟩
    have h : ∀ x : X, q x ^ 2 = q x := by
      intro x
      rw [hx₀ x]
      by_cases hx : x = x₀ <;> simp [hx]
    rw [gini]
    simp only [h, hq.2]
    ring

omit [DecidableEq X] in
/-- The floor is largest for the maximally disordered target. -/
theorem gini_le_one_sub_inv_card {q : X → ℝ} (hq : IsProbVec q) [Nonempty X] :
    gini q ≤ 1 - 1 / (Fintype.card X : ℝ) := by
  have hcard : (0 : ℝ) < (Fintype.card X : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hcs : (∑ x, q x) ^ 2 ≤ (Fintype.card X : ℝ) * ∑ x, q x ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset X)) (f := q)
  rw [hq.2] at hcs
  have hq2 : 1 / (Fintype.card X : ℝ) ≤ ∑ x, q x ^ 2 := by
    rw [div_le_iff₀ hcard]
    nlinarith
  rw [gini]
  linarith

lemma gini_pointVec (x₀ : X) : gini (pointVec x₀) = 0 :=
  (gini_eq_zero_iff (isProbVec_pointVec x₀)).2 ⟨x₀, fun _ => rfl⟩

/-! ## What the floor does to a benchmark -/

section Benchmark

lemma gini_unif3 : gini unif3 = 2/3 := by
  norm_num [gini, unif3, Fin.sum_univ_three]

/-- A *wrong* model of an ordered region: it misplaces a tenth of the population. -/
noncomputable def skew3 : Fin 3 → ℝ := fun i => if i = 0 then 9/10 else if i = 1 then 1/10 else 0

lemma isProbVec_skew3 : IsProbVec skew3 := by
  refine ⟨fun x => ?_, ?_⟩
  · fin_cases x <;> norm_num [skew3]
  · rw [Fin.sum_univ_three]
    norm_num [skew3, Fin.ext_iff]

/-- **A benchmark number ranks regions, not models.**  A model that is *exactly right* about
three equally populated rotamers scores `2/3`; a model that is *wrong* about an ordered
region -- it misplaces a tenth of the population -- scores `1/50`.  Comparing raw scores
across regions is therefore meaningless: only the excess over the floor, which by
`brier_decomposition` is exactly the squared population error, measures model quality. -/
theorem benchmark_confounded :
    expScore brier skew3 (pointVec (0 : Fin 3)) = 1/50 ∧
    expScore brier unif3 unif3 = 2/3 ∧
    expScore brier skew3 (pointVec (0 : Fin 3)) < expScore brier unif3 unif3 := by
  have h1 : expScore brier skew3 (pointVec (0 : Fin 3)) = 1/50 := by
    rw [brier_decomposition skew3 (pointVec (0 : Fin 3)) (isProbVec_pointVec _),
      gini_pointVec, Fin.sum_univ_three]
    norm_num [skew3, pointVec, Fin.ext_iff]
  have h2 : expScore brier unif3 unif3 = 2/3 := by
    rw [brier_floor unif3 isProbVec_unif3, gini_unif3]
  refine ⟨h1, h2, ?_⟩
  rw [h1, h2]
  norm_num

end Benchmark

end Scoring

end IDR
