/-
# Part LXXXVII  How much of the ensemble has the sampling seen?

Every ensemble model of a disordered region is built from a finite sample: a simulation trajectory,
a generated pool, a set of deposited conformers.  The population the sample never visited is
invisible to every diagnostic computed from the sample, and it is exactly the quantity a modeller
needs to bound.  This file proves what can be said about it.

The sample space is `Fin N → Fin m`: `N` independent draws from an ensemble with `m` conformational
states of populations `w`.  `sw w s = ∏ i, w (s i)` is the probability of a sample, `occ s x` the
number of times state `x` was drawn, and `unseenMass w s` the population of the states the sample
never visited.

* `sum_sw_eq_one` — the sample weights are a probability distribution, so the expectations below
  are honest expectations.
* `prob_unseen` — the probability that state `x` is never drawn is exactly `(1 - w x)^N`.
* `expected_unseenMass` — hence **the expected unseen population is exactly `Σ_x w x (1 - w x)^N`**,
  the *missing mass* `missingMass w N`.
* `missingMass_pos` — it is strictly positive whenever at least two states carry population, for
  every sample size: no finite sample ever certifies that it has seen the whole of a broad ensemble.
* `missingMass_uniform`, `missingMass_uniform_ge_one_sub`, `sample_size_needed` — on a uniform
  ensemble over `m` states the missing mass is `(1 - 1/m)^N ≥ 1 - N/m`, so **leaving at most `eps`
  of the population unseen requires at least `(1 - eps)·m` draws**: the sampling cost is linear in
  the number of populated states, which elsewhere in this development is exponential in the chain
  length.
* `expected_singletonCount`, `good_turing` — the positive half.  The expected number of states seen
  *exactly once* in a sample of size `N + 1` is `(N + 1)` times the missing mass at size `N`.  So
  the fraction of the sample made of states seen exactly once is an unbiased estimate of the
  population the sample is missing — a quantity computable from the sample itself, with no
  knowledge of `m` or of `w`.  That is the number an ensemble model of a disordered region should
  report alongside its fit.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset

namespace Coverage

variable {m N : ℕ}

/-- Probability of drawing the sample `s` from the ensemble with populations `w`. -/
def sw (w : Fin m → ℝ) (s : Fin N → Fin m) : ℝ := ∏ i, w (s i)

/-- Number of times the sample `s` visited state `x`. -/
def occ (s : Fin N → Fin m) (x : Fin m) : ℕ := (univ.filter fun i => s i = x).card

/-- The population of the states the sample never visited. -/
def unseenMass (w : Fin m → ℝ) (s : Fin N → Fin m) : ℝ :=
  ∑ x ∈ univ.filter fun x => occ s x = 0, w x

/-- The number of states the sample visited exactly once. -/
def singletonCount (s : Fin N → Fin m) : ℕ := (univ.filter fun x => occ s x = 1).card

/-- The expected unseen population after `N` draws. -/
def missingMass (w : Fin m → ℝ) (N : ℕ) : ℝ := ∑ x, w x * (1 - w x) ^ N

/-! ## The sample distribution -/

theorem sum_sw_eq_one (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (N : ℕ) :
    ∑ s : Fin N → Fin m, sw w s = 1 := by
  have h := Finset.prod_univ_sum (t := fun _ : Fin N => (univ : Finset (Fin m)))
    (f := fun (_ : Fin N) (j : Fin m) => w j)
  simp only [Fintype.piFinset_univ] at h
  simp only [sw]
  rw [← h, hsum, Finset.prod_const_one]

/-- The samples that never draw `x`. -/
theorem avoid_eq_piFinset (x : Fin m) :
    (univ.filter fun s : Fin N → Fin m => occ s x = 0)
      = Fintype.piFinset fun _ : Fin N => univ.erase x := by
  ext s
  have hocc : occ s x = 0 ↔ ∀ i, s i ≠ x := by
    rw [occ, Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    exact ⟨fun h i => h (mem_univ i), fun h i _ => h i⟩
  rw [mem_filter, Fintype.mem_piFinset]
  simp only [mem_univ, true_and, hocc, mem_erase, and_true]

/-- **The probability that a state is never drawn.** -/
theorem prob_unseen (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (x : Fin m) (N : ℕ) :
    ∑ s ∈ univ.filter fun s : Fin N → Fin m => occ s x = 0, sw w s = (1 - w x) ^ N := by
  rw [avoid_eq_piFinset x]
  have h := Finset.prod_univ_sum (t := fun _ : Fin N => (univ.erase x : Finset (Fin m)))
    (f := fun (_ : Fin N) (j : Fin m) => w j)
  have hx : ∑ j ∈ univ.erase x, w j = 1 - w x := by
    rw [Finset.sum_erase_eq_sub (mem_univ x), hsum]
  simp only [sw]
  rw [← h, hx, Finset.prod_const, Finset.card_univ, Fintype.card_fin]

/-! ## The missing mass -/

/-- **The expected unseen population is the missing mass.** -/
theorem expected_unseenMass (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (N : ℕ) :
    ∑ s : Fin N → Fin m, sw w s * unseenMass w s = missingMass w N := by
  have key : ∀ s : Fin N → Fin m, sw w s * unseenMass w s
      = ∑ x, (if occ s x = 0 then sw w s * w x else 0) := by
    intro s
    rw [unseenMass, Finset.mul_sum, Finset.sum_filter]
  rw [Finset.sum_congr rfl fun s _ => key s, Finset.sum_comm, missingMass]
  refine Finset.sum_congr rfl fun x _ => ?_
  have hx : ∑ s : Fin N → Fin m, (if occ s x = 0 then sw w s * w x else 0)
      = (∑ s ∈ univ.filter fun s : Fin N → Fin m => occ s x = 0, sw w s) * w x := by
    rw [Finset.sum_filter, Finset.sum_mul]
    exact Finset.sum_congr rfl fun s _ => by by_cases h : occ s x = 0 <;> simp [h]
  rw [hx, prob_unseen w hsum x N, mul_comm]

theorem missingMass_nonneg (w : Fin m → ℝ) (hw : ∀ j, 0 ≤ w j) (hle : ∀ j, w j ≤ 1) (N : ℕ) :
    0 ≤ missingMass w N :=
  Finset.sum_nonneg fun x _ => mul_nonneg (hw x) (pow_nonneg (by linarith [hle x]) N)

/-- **No finite sample sees a broad ensemble.**  If some state carries population strictly between
`0` and `1`, the expected unseen population is strictly positive at every sample size. -/
theorem missingMass_pos (w : Fin m → ℝ) (hw : ∀ j, 0 ≤ w j) (hle : ∀ j, w j ≤ 1)
    {x : Fin m} (hx0 : 0 < w x) (hx1 : w x < 1) (N : ℕ) : 0 < missingMass w N := by
  refine Finset.sum_pos' (fun j _ => mul_nonneg (hw j) (pow_nonneg (by linarith [hle j]) N))
    ⟨x, mem_univ x, ?_⟩
  exact mul_pos hx0 (pow_pos (by linarith) N)

/-- The uniform ensemble over `m` states. -/
theorem missingMass_uniform (m : ℕ) (hm : 0 < m) (N : ℕ) :
    missingMass (fun _ : Fin m => (1 : ℝ) / m) N = (1 - 1 / m) ^ N := by
  have hm' : (m : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hm.ne'
  simp only [missingMass, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- Bernoulli's inequality applied to the uniform missing mass. -/
theorem missingMass_uniform_ge_one_sub (m : ℕ) (hm : 0 < m) (N : ℕ) :
    1 - N / m ≤ missingMass (fun _ : Fin m => (1 : ℝ) / m) N := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  rw [missingMass_uniform m hm N]
  have hb : (1 : ℝ) + N * (-(1 / m)) ≤ (1 + -(1 / m)) ^ N :=
    one_add_mul_le_pow (by
      have : (0 : ℝ) < 1 / m := by positivity
      linarith [div_le_one_of_le₀ (by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hm.ne'
        : (1 : ℝ) ≤ m) hm'.le]) N
  have h1 : (1 : ℝ) + -(1 / m) = 1 - 1 / m := by ring
  have h2 : (1 : ℝ) + N * (-(1 / m)) = 1 - N / m := by
    field_simp
    ring
  rw [h1, h2] at hb
  exact hb

/-- **The sampling cost of coverage.**  To leave at most `eps` of a uniform `m`-state ensemble
unseen, at least `(1 - eps)·m` draws are needed. -/
theorem sample_size_needed (m : ℕ) (hm : 0 < m) (N : ℕ) {eps : ℝ}
    (h : missingMass (fun _ : Fin m => (1 : ℝ) / m) N ≤ eps) : (1 - eps) * m ≤ N := by
  have hm' : (0 : ℝ) < m := by exact_mod_cast hm
  have hb := missingMass_uniform_ge_one_sub m hm N
  have : 1 - (N : ℝ) / m ≤ eps := le_trans hb h
  have h2 : (1 - eps) ≤ (N : ℝ) / m := by linarith
  calc (1 - eps) * m ≤ ((N : ℝ) / m) * m := by nlinarith
    _ = N := by field_simp

/-! ## Good–Turing: the missing mass is estimable from the sample -/

/-- The samples in which `x` occurs exactly at position `i₀`. -/
theorem once_at_eq_piFinset (x : Fin m) (i₀ : Fin N) :
    (univ.filter fun s : Fin N → Fin m => s i₀ = x ∧ ∀ i, i ≠ i₀ → s i ≠ x)
      = Fintype.piFinset fun i => if i = i₀ then {x} else univ.erase x := by
  ext s
  rw [mem_filter, Fintype.mem_piFinset]
  constructor
  · rintro ⟨-, h0, h1⟩ i
    by_cases hi : i = i₀
    · subst hi; simp [h0]
    · simp only [hi, if_false, mem_erase]
      exact ⟨h1 i hi, mem_univ _⟩
  · intro h
    refine ⟨mem_univ _, ?_, ?_⟩
    · have := h i₀; simpa using this
    · intro i hi
      have := h i
      simp only [hi, if_false, mem_erase] at this
      exact this.1

theorem sum_sw_once_at (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (x : Fin m) (i₀ : Fin N) :
    ∑ s ∈ univ.filter fun s : Fin N → Fin m => s i₀ = x ∧ ∀ i, i ≠ i₀ → s i ≠ x, sw w s
      = w x * (1 - w x) ^ (N - 1) := by
  rw [once_at_eq_piFinset x i₀]
  have h := Finset.prod_univ_sum
    (t := fun i : Fin N => if i = i₀ then ({x} : Finset (Fin m)) else univ.erase x)
    (f := fun (_ : Fin N) (j : Fin m) => w j)
  simp only [sw]
  rw [← h]
  have hx : ∑ j ∈ univ.erase x, w j = 1 - w x := by
    rw [Finset.sum_erase_eq_sub (mem_univ x), hsum]
  have hterm : ∀ i : Fin N,
      (∑ j ∈ (if i = i₀ then ({x} : Finset (Fin m)) else univ.erase x), w j)
        = if i = i₀ then w x else 1 - w x := by
    intro i
    by_cases hi : i = i₀ <;> simp [hi, hx]
  rw [Finset.prod_congr rfl fun i _ => hterm i,
    ← Finset.prod_erase_mul (univ : Finset (Fin N)) _ (mem_univ i₀)]
  have h1 : ∀ i ∈ univ.erase i₀, (if i = i₀ then w x else 1 - w x) = 1 - w x := by
    intro i hi
    simp [(mem_erase.mp hi).1]
  rw [Finset.prod_congr rfl h1, Finset.prod_const, Finset.card_erase_of_mem (mem_univ i₀)]
  simp [Finset.card_univ, mul_comm]

/-- The samples in which `x` occurs exactly once, decomposed by the position of the occurrence. -/
theorem prob_once (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (x : Fin m) (N : ℕ) :
    ∑ s ∈ univ.filter fun s : Fin N → Fin m => occ s x = 1, sw w s
      = N * (w x * (1 - w x) ^ (N - 1)) := by
  classical
  have hbi : (univ.filter fun s : Fin N → Fin m => occ s x = 1)
      = univ.biUnion fun i₀ : Fin N =>
        univ.filter fun s : Fin N → Fin m => s i₀ = x ∧ ∀ i, i ≠ i₀ → s i ≠ x := by
    ext s
    simp only [mem_filter, mem_univ, true_and, mem_biUnion]
    constructor
    · intro h
      obtain ⟨i₀, hi₀⟩ := Finset.card_eq_one.mp h
      refine ⟨i₀, ?_, ?_⟩
      · have : i₀ ∈ univ.filter fun i => s i = x := by rw [hi₀]; exact mem_singleton_self i₀
        simpa using this
      · intro i hi hsi
        have : i ∈ univ.filter fun i => s i = x := by simpa using hsi
        rw [hi₀, mem_singleton] at this
        exact hi this
    · rintro ⟨i₀, h0, h1⟩
      have : (univ.filter fun i => s i = x) = {i₀} := by
        ext i
        simp only [mem_filter, mem_univ, true_and, mem_singleton]
        constructor
        · intro hsi
          by_contra hne
          exact h1 i hne hsi
        · rintro rfl; exact h0
      rw [occ, this, Finset.card_singleton]
  rw [hbi]
  rw [Finset.sum_biUnion]
  · rw [Finset.sum_congr rfl fun i₀ _ => sum_sw_once_at w hsum x i₀]
    simp [Finset.sum_const]
  · intro i₀ _ i₁ _ hne
    simp only [Finset.disjoint_left, mem_filter, mem_univ, true_and]
    rintro s ⟨h0, h1⟩ ⟨h0', h1'⟩
    exact h1' i₀ hne h0

/-- **The expected number of states seen exactly once.** -/
theorem expected_singletonCount (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (N : ℕ) :
    ∑ s : Fin N → Fin m, sw w s * (singletonCount s : ℝ)
      = ∑ x, N * (w x * (1 - w x) ^ (N - 1)) := by
  have key : ∀ s : Fin N → Fin m, sw w s * (singletonCount s : ℝ)
      = ∑ x, (if occ s x = 1 then sw w s else 0) := by
    intro s
    rw [← Finset.sum_filter, Finset.sum_const, singletonCount, nsmul_eq_mul, mul_comm]
  rw [Finset.sum_congr rfl fun s _ => key s, Finset.sum_comm]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [← prob_once w hsum x N, Finset.sum_filter]

/-- **The Good–Turing identity.**  The expected number of states seen exactly once in a sample of
size `N + 1` is `(N + 1)` times the expected population left unseen by a sample of size `N`.  The
singleton fraction of a finite conformational sample is therefore an unbiased estimate of the
population that sample is missing — computable without knowing how many states there are, or what
their populations are. -/
theorem good_turing (w : Fin m → ℝ) (hsum : ∑ j, w j = 1) (N : ℕ) :
    ∑ s : Fin (N + 1) → Fin m, sw w s * (singletonCount s : ℝ)
      = (N + 1) * missingMass w N := by
  rw [expected_singletonCount w hsum (N + 1), missingMass, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  simp only [Nat.add_sub_cancel]
  push_cast
  ring

end Coverage

end IDR
