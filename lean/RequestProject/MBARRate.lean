/-
# Part CX  A convergence rate for the MBAR iteration

Part CV (`RequestProject.MBAR`) proves that the WHAM/MBAR self-consistency equations are scale
invariant, consistent, uniquely solvable up to that scale, and that the iteration is stable in the
ratio bracket, and it records one thing it does not prove: *a rate* of convergence.  This file
proves the rate.

The mechanism is the one that makes positive maps contract: the self-consistency map is a
composition of two averaging steps with an inversion between them, and an average pulls the
largest and smallest coordinates towards each other by exactly the mass sitting on the extreme
coordinate.  Write `y` for the solution and

    t i = z i / y i

for the current guess in units of the solution — the object the scale freedom leaves meaningful.
The **overlap constant** is

    eps  ≤  mixWeight N W y w j = (N j · W j w / y j) / denom N W y w,

the smallest share any window `j` has of the mixture at any sample `w`; `eps > 0` is exactly the
positive-overlap hypothesis under which Part CV proves uniqueness.  Define the *spread*
`spread y z = (max_i t i)/(min_i t i) ≥ 1`, whose logarithm is the Hilbert projective distance
from `z` to the solution and which is `1` exactly when `z` is the solution.

* `spread_contraction` — **one iteration contracts the spread geometrically**:

      spread y (T z) − 1  ≤  (1 − 2·eps) · (spread y z − 1).

* `spread_iterate` — hence after `n` iterations `spread y (T^[n] z) − 1 ≤ (1−2eps)^n · (spread y z − 1)`.
* `freeEnergy_error_le` — and the free energy *differences*, the quantities the estimator actually
  reports, are within `(1−2eps)^n · (spread y z − 1)` of the truth after `n` iterations, in nats.
* `freeEnergy_error_tendsto_zero` — so the reported profile converges to the true one, at a rate
  that is explicit in the data: the number of iterations needed for accuracy `delta` is
  `log((spread−1)/delta) / log(1/(1−2eps))`, linear in the log of the required accuracy and inverse
  in the overlap.

The constant is the honest one: `eps ≤ 1/|I|` always, so the guaranteed factor is never better than
`1 − 2/|I|`, and it degrades to `1` as the overlap degrades — which is the true behaviour of the
iteration, not an artefact.  What the theorem buys is that the rate is *geometric with a computable
ratio*: the overlap constant is a statistic of the sampled data, so a run can be certified.
-/
import Mathlib
import RequestProject.MBAR

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.MBAR

open Finset

/-! ## Weighted averages and the extreme-coordinate estimate -/

section Avg

variable {α : Type*} [Fintype α]

lemma sum_weighted_le_hi {p g : α → ℝ} {hi : ℝ} (hp : ∀ a, 0 ≤ p a) (hs : ∑ a, p a = 1)
    (hg : ∀ a, g a ≤ hi) : ∑ a, p a * g a ≤ hi := by
  calc ∑ a, p a * g a ≤ ∑ a, p a * hi :=
        Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hg a) (hp a)
    _ = hi := by rw [← Finset.sum_mul, hs, one_mul]

lemma sum_weighted_lo_le {p g : α → ℝ} {lo : ℝ} (hp : ∀ a, 0 ≤ p a) (hs : ∑ a, p a = 1)
    (hg : ∀ a, lo ≤ g a) : lo ≤ ∑ a, p a * g a := by
  calc lo = ∑ a, p a * lo := by rw [← Finset.sum_mul, hs, one_mul]
    _ ≤ ∑ a, p a * g a := Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hg a) (hp a)

/-- **The extreme-coordinate estimate, from below.**  A weighted average of values in `[lo, hi]`
which puts mass at least `eps` on a coordinate where the value is `hi` is at least
`lo + eps·(hi − lo)`. -/
lemma sum_weighted_ge_of_point {p g : α → ℝ} {lo hi eps : ℝ} (hp : ∀ a, 0 ≤ p a)
    (hs : ∑ a, p a = 1) (hlo : ∀ a, lo ≤ g a) (a₀ : α) (h0 : hi ≤ g a₀) (heps : eps ≤ p a₀)
    (hle : lo ≤ hi) : lo + eps * (hi - lo) ≤ ∑ a, p a * g a := by
  have key : eps * (hi - lo) ≤ ∑ a, p a * (g a - lo) := by
    have h1 : p a₀ * (g a₀ - lo) ≤ ∑ a, p a * (g a - lo) :=
      Finset.single_le_sum (f := fun a => p a * (g a - lo))
        (fun a _ => mul_nonneg (hp a) (by linarith [hlo a])) (Finset.mem_univ a₀)
    have h2 : eps * (hi - lo) ≤ p a₀ * (hi - lo) :=
      mul_le_mul_of_nonneg_right heps (by linarith)
    have h3 : p a₀ * (hi - lo) ≤ p a₀ * (g a₀ - lo) :=
      mul_le_mul_of_nonneg_left (by linarith) (hp a₀)
    linarith
  have hsplit : ∑ a, p a * (g a - lo) = (∑ a, p a * g a) - lo := by
    have : ∑ a, p a * (g a - lo) = (∑ a, p a * g a) - (∑ a, p a) * lo := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [this, hs, one_mul]
  linarith [key, hsplit.symm.le, hsplit.le]

/-- **The extreme-coordinate estimate, from above.** -/
lemma sum_weighted_le_of_point {p g : α → ℝ} {lo hi eps : ℝ} (hp : ∀ a, 0 ≤ p a)
    (hs : ∑ a, p a = 1) (hhi : ∀ a, g a ≤ hi) (a₀ : α) (h0 : g a₀ ≤ lo) (heps : eps ≤ p a₀)
    (hle : lo ≤ hi) : ∑ a, p a * g a ≤ hi - eps * (hi - lo) := by
  have key : eps * (hi - lo) ≤ ∑ a, p a * (hi - g a) := by
    have h1 : p a₀ * (hi - g a₀) ≤ ∑ a, p a * (hi - g a) :=
      Finset.single_le_sum (f := fun a => p a * (hi - g a))
        (fun a _ => mul_nonneg (hp a) (by linarith [hhi a])) (Finset.mem_univ a₀)
    have h2 : eps * (hi - lo) ≤ p a₀ * (hi - lo) :=
      mul_le_mul_of_nonneg_right heps (by linarith)
    have h3 : p a₀ * (hi - lo) ≤ p a₀ * (hi - g a₀) :=
      mul_le_mul_of_nonneg_left (by linarith) (hp a₀)
    linarith
  have hsplit : ∑ a, p a * (hi - g a) = hi - (∑ a, p a * g a) := by
    have : ∑ a, p a * (hi - g a) = (∑ a, p a) * hi - (∑ a, p a * g a) := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun a _ => by ring
    rw [this, hs, one_mul]
  linarith [key, hsplit.le, hsplit.symm.le]

end Avg

variable {I Ω : Type*} [Fintype I] [Fintype Ω]

/-! ## The two families of weights -/

/-- The share of window `j` in the mixture at sample `w`, evaluated at `y`. -/
noncomputable def mixWeight (N : I → ℝ) (W : I → Ω → ℝ) (y : I → ℝ) (w : Ω) (j : I) : ℝ :=
  (N j * W j w / y j) / denom N W y w

/-- The share of sample `w` in the update of window `i`, evaluated at `y`. -/
noncomputable def sampWeight (N : I → ℝ) (W : I → Ω → ℝ) (mu : Ω → ℝ) (y : I → ℝ) (i : I)
    (w : Ω) : ℝ := (mu w * (W i w / denom N W y w)) / y i

omit [Fintype Ω] in
lemma mixWeight_nonneg [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {y : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hy : ∀ i, 0 < y i) (w : Ω) (j : I) :
    0 ≤ mixWeight N W y w j :=
  div_nonneg (div_nonneg (mul_pos (hN j) (hW j w)).le (hy j).le) (denom_pos hN hW hy w).le

omit [Fintype Ω] in
lemma sum_mixWeight [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {y : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hy : ∀ i, 0 < y i) (w : Ω) :
    ∑ j, mixWeight N W y w j = 1 := by
  unfold mixWeight
  rw [← Finset.sum_div]
  exact div_self (denom_pos hN hW hy w).ne'

omit [Fintype Ω] in
lemma sampWeight_nonneg [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {y : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 ≤ mu w) (hy : ∀ i, 0 < y i)
    (i : I) (w : Ω) : 0 ≤ sampWeight N W mu y i w :=
  div_nonneg (mul_nonneg (hmu w) (div_nonneg (hW i w).le (denom_pos hN hW hy w).le)) (hy i).le

lemma sum_sampWeight [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {y : I → ℝ}
    (hy : ∀ i, 0 < y i) (hfix : mbarMap N W mu y = y) (i : I) :
    ∑ w, sampWeight N W mu y i w = 1 := by
  unfold sampWeight
  rw [← Finset.sum_div]
  have : ∑ w, mu w * (W i w / denom N W y w) = y i := congrFun hfix i
  rw [this, div_self (hy i).ne']

/-! ## The two averaging steps -/

omit [Fintype Ω] in
/-- The mixture denominator at `z`, written as the denominator at `y` times an average of the
inverse ratios. -/
lemma denom_eq_mix [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {y z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hy : ∀ i, 0 < y i) (hz : ∀ i, 0 < z i) (w : Ω) :
    denom N W z w = denom N W y w * ∑ j, mixWeight N W y w j * (y j / z j) := by
  have hd : denom N W y w ≠ 0 := (denom_pos hN hW hy w).ne'
  unfold denom mixWeight
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun j _ => ?_
  have hyj : y j ≠ 0 := (hy j).ne'
  have hzj : z j ≠ 0 := (hz j).ne'
  field_simp
  rw [denom]

/-- The update at `z`, written as `y` times an average of the denominator ratios. -/
lemma mbarMap_eq_samp [Nonempty I] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {y z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hy : ∀ i, 0 < y i) (hz : ∀ i, 0 < z i) (i : I) :
    mbarMap N W mu z i
      = y i * ∑ w, sampWeight N W mu y i w * (denom N W y w / denom N W z w) := by
  unfold mbarMap sampWeight
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun w _ => ?_
  have hdy : denom N W y w ≠ 0 := (denom_pos hN hW hy w).ne'
  have hdz : denom N W z w ≠ 0 := (denom_pos hN hW hz w).ne'
  have hyi : y i ≠ 0 := (hy i).ne'
  field_simp

lemma mbarMap_pos [Nonempty I] [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w) (hz : ∀ i, 0 < z i)
    (i : I) : 0 < mbarMap N W mu z i :=
  Finset.sum_pos (fun w _ => mul_pos (hmu w) (div_pos (hW i w) (denom_pos hN hW hz w)))
    Finset.univ_nonempty

/-! ## The spread and its contraction -/

/-- The largest ratio `z i / y i`. -/
noncomputable def ratMax [Nonempty I] (y z : I → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun i => z i / y i

/-- The smallest ratio `z i / y i`. -/
noncomputable def ratMin [Nonempty I] (y z : I → ℝ) : ℝ :=
  Finset.univ.inf' Finset.univ_nonempty fun i => z i / y i

/-- The spread of `z` around the solution `y`: the ratio of the largest to the smallest
coordinatewise ratio.  Its logarithm is the Hilbert projective distance; it equals `1` exactly
when `z` is a multiple of `y`. -/
noncomputable def spread [Nonempty I] (y z : I → ℝ) : ℝ := ratMax y z / ratMin y z

variable [Nonempty I]

lemma ratMin_le (y z : I → ℝ) (i : I) : ratMin y z ≤ z i / y i := by
  unfold ratMin; exact Finset.inf'_le (fun k => z k / y k) (Finset.mem_univ i)

lemma le_ratMax (y z : I → ℝ) (i : I) : z i / y i ≤ ratMax y z := by
  unfold ratMax; exact Finset.le_sup' (fun k => z k / y k) (Finset.mem_univ i)

lemma exists_ratMin (y z : I → ℝ) : ∃ i, z i / y i = ratMin y z := by
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_inf' (Finset.univ_nonempty (α := I)) fun i => z i / y i
  exact ⟨i, hi.symm⟩

lemma exists_ratMax (y z : I → ℝ) : ∃ i, z i / y i = ratMax y z := by
  obtain ⟨i, -, hi⟩ := Finset.exists_mem_eq_sup' (Finset.univ_nonempty (α := I)) fun i => z i / y i
  exact ⟨i, hi.symm⟩

lemma ratMin_pos {y z : I → ℝ} (hy : ∀ i, 0 < y i) (hz : ∀ i, 0 < z i) : 0 < ratMin y z := by
  obtain ⟨i, hi⟩ := exists_ratMin y z
  rw [← hi]
  exact div_pos (hz i) (hy i)

lemma ratMin_le_ratMax (y z : I → ℝ) : ratMin y z ≤ ratMax y z := by
  obtain ⟨i, hi⟩ := exists_ratMin y z
  exact hi ▸ le_ratMax y z i

lemma one_le_spread {y z : I → ℝ} (hy : ∀ i, 0 < y i) (hz : ∀ i, 0 < z i) : 1 ≤ spread y z :=
  (one_le_div (ratMin_pos hy hz)).2 (ratMin_le_ratMax y z)

/-- **One iteration contracts the spread.**  With overlap constant `eps ∈ (0, 1/2]`, one step of
the MBAR iteration multiplies the excess spread `spread − 1` by at most `1 − 2·eps`. -/
theorem spread_contraction [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {y z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w)
    (hy : IsSolution N W mu y) (hz : ∀ i, 0 < z i)
    {eps : ℝ} (heps0 : 0 < eps) (heps1 : eps ≤ 1 / 2)
    (hov : ∀ w j, eps ≤ mixWeight N W y w j) :
    spread y (mbarMap N W mu z) - 1 ≤ (1 - 2 * eps) * (spread y z - 1) := by
  obtain ⟨hyp, hyfix⟩ := hy
  set m := ratMin y z with hm
  set M := ratMax y z with hM
  have hmpos : 0 < m := ratMin_pos hyp hz
  have hmM : m ≤ M := ratMin_le_ratMax y z
  have hMpos : 0 < M := lt_of_lt_of_le hmpos hmM
  obtain ⟨imin, himin⟩ := exists_ratMin y z
  obtain ⟨imax, himax⟩ := exists_ratMax y z
  -- the inverse ratios
  have hg_lo : ∀ j, 1 / M ≤ y j / z j := by
    intro j
    rw [div_le_div_iff₀ hMpos (hz j)]
    have := le_ratMax y z j
    rw [div_le_iff₀ (hyp j)] at this
    linarith
  have hg_hi : ∀ j, y j / z j ≤ 1 / m := by
    intro j
    rw [div_le_div_iff₀ (hz j) hmpos]
    have := ratMin_le y z j
    rw [le_div_iff₀ (hyp j)] at this
    linarith
  have hg_min : 1 / m ≤ y imin / z imin := by
    have hzi : z imin = m * y imin := (div_eq_iff (hyp imin).ne').1 himin
    have hval : y imin / (m * y imin) = 1 / m := by
      rw [mul_comm, ← div_div, div_self (hyp imin).ne']
    rw [hzi, hval]
  have hg_max : y imax / z imax ≤ 1 / M := by
    have hzi : z imax = M * y imax := (div_eq_iff (hyp imax).ne').1 himax
    have hval : y imax / (M * y imax) = 1 / M := by
      rw [mul_comm, ← div_div, div_self (hyp imax).ne']
    rw [hzi, hval]
  set a : ℝ := 1 / M + eps * (1 / m - 1 / M) with ha
  set b : ℝ := 1 / m - eps * (1 / m - 1 / M) with hb
  have hinvle : 1 / M ≤ 1 / m := by
    apply one_div_le_one_div_of_le hmpos hmM
  have hapos : 0 < a := by
    have : 0 < 1 / M := by positivity
    nlinarith [hinvle, heps0.le]
  have hab : a ≤ b := by
    have : eps * (1 / m - 1 / M) ≤ (1 / 2) * (1 / m - 1 / M) :=
      mul_le_mul_of_nonneg_right heps1 (by linarith)
    nlinarith [hinvle]
  have hbpos : 0 < b := lt_of_lt_of_le hapos hab
  -- step 1: the denominators
  have hden : ∀ w, denom N W y w * a ≤ denom N W z w ∧ denom N W z w ≤ denom N W y w * b := by
    intro w
    have hrepr := denom_eq_mix (N := N) (W := W) (y := y) (z := z) hN hW hyp hz w
    have hp := mixWeight_nonneg (N := N) (W := W) (y := y) hN hW hyp w
    have hs := sum_mixWeight (N := N) (W := W) (y := y) hN hW hyp w
    have hlow : a ≤ ∑ j, mixWeight N W y w j * (y j / z j) := by
      have := sum_weighted_ge_of_point (p := mixWeight N W y w) (g := fun j => y j / z j)
        (lo := 1 / M) (hi := 1 / m) (eps := eps) hp hs hg_lo imin hg_min (hov w imin) hinvle
      linarith [this]
    have hhigh : ∑ j, mixWeight N W y w j * (y j / z j) ≤ b := by
      have := sum_weighted_le_of_point (p := mixWeight N W y w) (g := fun j => y j / z j)
        (lo := 1 / M) (hi := 1 / m) (eps := eps) hp hs hg_hi imax hg_max (hov w imax) hinvle
      linarith [this]
    constructor
    · rw [hrepr]
      exact mul_le_mul_of_nonneg_left hlow (denom_pos hN hW hyp w).le
    · rw [hrepr]
      exact mul_le_mul_of_nonneg_left hhigh (denom_pos hN hW hyp w).le
  -- step 2: the update
  have hupd : ∀ i, 1 / b ≤ mbarMap N W mu z i / y i ∧ mbarMap N W mu z i / y i ≤ 1 / a := by
    intro i
    have hrepr := mbarMap_eq_samp (N := N) (W := W) (mu := mu) (y := y) (z := z) hN hW hyp hz i
    have hp := sampWeight_nonneg (N := N) (W := W) (mu := mu) (y := y) hN hW
      (fun w => (hmu w).le) hyp i
    have hs := sum_sampWeight (N := N) (W := W) (mu := mu) (y := y) hyp hyfix i
    have hratio : ∀ w, 1 / b ≤ denom N W y w / denom N W z w
        ∧ denom N W y w / denom N W z w ≤ 1 / a := by
      intro w
      obtain ⟨h1, h2⟩ := hden w
      have hdy := denom_pos hN hW hyp w
      have hdz := denom_pos hN hW hz w
      constructor
      · rw [div_le_div_iff₀ hbpos hdz]
        nlinarith
      · rw [div_le_div_iff₀ hdz hapos]
        nlinarith
    have hlo := sum_weighted_lo_le (p := sampWeight N W mu y i)
      (g := fun w => denom N W y w / denom N W z w) hp hs (fun w => (hratio w).1)
    have hhi := sum_weighted_le_hi (p := sampWeight N W mu y i)
      (g := fun w => denom N W y w / denom N W z w) hp hs (fun w => (hratio w).2)
    rw [hrepr]
    constructor
    · rw [le_div_iff₀ (hyp i)]
      nlinarith [hyp i]
    · rw [div_le_iff₀ (hyp i)]
      nlinarith [hyp i]
  -- assemble
  set Tz := mbarMap N W mu z with hTz
  have hTzpos : ∀ i, 0 < Tz i := fun i => mbarMap_pos hN hW hmu hz i
  have hM' : ratMax y Tz ≤ 1 / a := by
    obtain ⟨i, hi⟩ := exists_ratMax y Tz
    rw [← hi]
    exact (hupd i).2
  have hm' : 1 / b ≤ ratMin y Tz := by
    obtain ⟨i, hi⟩ := exists_ratMin y Tz
    rw [← hi]
    exact (hupd i).1
  have hminpos : 0 < ratMin y Tz := ratMin_pos hyp hTzpos
  have hsp : spread y Tz ≤ b / a := by
    have h1 : ratMax y Tz ≤ 1 / a := hM'
    have h2 : (1 : ℝ) / b ≤ ratMin y Tz := hm'
    have hA : ratMax y Tz * a ≤ 1 := by
      have h := mul_le_mul_of_nonneg_right h1 hapos.le
      rwa [one_div, inv_mul_cancel₀ hapos.ne'] at h
    have hB : (1 : ℝ) ≤ b * ratMin y Tz := by
      have h := mul_le_mul_of_nonneg_left h2 hbpos.le
      rwa [mul_one_div, div_self hbpos.ne'] at h
    rw [spread, div_le_div_iff₀ hminpos hapos]
    linarith
  -- the Möbius estimate
  have hspz : spread y z = M / m := rfl
  have hkey : b / a - 1 ≤ (1 - 2 * eps) * (M / m - 1) := by
    have hu : (0 : ℝ) ≤ 1 / m - 1 / M := by linarith
    have hba : b - a = (1 - 2 * eps) * (1 / m - 1 / M) := by rw [ha, hb]; ring
    have hMu : M * (1 / m - 1 / M) = M / m - 1 := by
      field_simp
    have hMa : 1 ≤ M * a := by
      have hexp : M * a = 1 + eps * (M * (1 / m - 1 / M)) := by
        rw [ha]; field_simp
      have : 0 ≤ eps * (M * (1 / m - 1 / M)) :=
        mul_nonneg heps0.le (mul_nonneg hMpos.le hu)
      linarith [hexp]
    have h2e : (0 : ℝ) ≤ 1 - 2 * eps := by linarith
    have hdiv : b / a - 1 = (b - a) / a := by field_simp
    rw [hdiv, div_le_iff₀ hapos, hba, ← hMu]
    have hX : 0 ≤ (1 - 2 * eps) * (1 / m - 1 / M) := mul_nonneg h2e hu
    nlinarith [hX, hMa]
  rw [hspz]
  linarith [hsp, hkey]

/-! ## Iterating -/

/-- The MBAR iteration, `n` steps from the guess `z`. -/
noncomputable def iter (N : I → ℝ) (W : I → Ω → ℝ) (mu : Ω → ℝ) (z : I → ℝ) : ℕ → (I → ℝ)
  | 0 => z
  | n + 1 => mbarMap N W mu (iter N W mu z n)

lemma iter_pos [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w) (hz : ∀ i, 0 < z i) :
    ∀ n i, 0 < iter N W mu z n i := by
  intro n
  induction n with
  | zero => exact hz
  | succ k ih => exact fun i => mbarMap_pos hN hW hmu ih i

/-- **Geometric convergence of the MBAR iteration.**  After `n` iterations the excess spread has
been multiplied by at most `(1 − 2·eps)^n`. -/
theorem spread_iterate [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {y z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w)
    (hy : IsSolution N W mu y) (hz : ∀ i, 0 < z i)
    {eps : ℝ} (heps0 : 0 < eps) (heps1 : eps ≤ 1 / 2)
    (hov : ∀ w j, eps ≤ mixWeight N W y w j) (n : ℕ) :
    spread y (iter N W mu z n) - 1 ≤ (1 - 2 * eps) ^ n * (spread y z - 1) := by
  induction n with
  | zero => simp [iter]
  | succ k ih =>
      have hk := iter_pos (N := N) (W := W) (mu := mu) (z := z) hN hW hmu hz k
      have hstep := spread_contraction (N := N) (W := W) (mu := mu) (y := y)
        (z := iter N W mu z k) hN hW hmu hy hk heps0 heps1 hov
      have hfac : (0 : ℝ) ≤ 1 - 2 * eps := by linarith
      calc spread y (iter N W mu z (k + 1)) - 1
          = spread y (mbarMap N W mu (iter N W mu z k)) - 1 := by rw [iter]
        _ ≤ (1 - 2 * eps) * (spread y (iter N W mu z k) - 1) := hstep
        _ ≤ (1 - 2 * eps) * ((1 - 2 * eps) ^ k * (spread y z - 1)) :=
            mul_le_mul_of_nonneg_left ih hfac
        _ = (1 - 2 * eps) ^ (k + 1) * (spread y z - 1) := by ring

/-! ## What the estimator reports: free energy differences -/

/-- The error of a free energy difference is at most the log of the spread. -/
lemma freeEnergy_diff_error_le_log_spread {y z : I → ℝ} (hy : ∀ i, 0 < y i) (hz : ∀ i, 0 < z i)
    (i j : I) :
    |(freeEnergy z i - freeEnergy z j) - (freeEnergy y i - freeEnergy y j)|
      ≤ Real.log (spread y z) := by
  have hmpos : 0 < ratMin y z := ratMin_pos hy hz
  have hMpos : 0 < ratMax y z := lt_of_lt_of_le hmpos (ratMin_le_ratMax y z)
  have key : (freeEnergy z i - freeEnergy z j) - (freeEnergy y i - freeEnergy y j)
      = Real.log (z j / y j) - Real.log (z i / y i) := by
    unfold freeEnergy
    rw [Real.log_div (hz j).ne' (hy j).ne', Real.log_div (hz i).ne' (hy i).ne']
    ring
  rw [key, abs_le]
  have hlogsp : Real.log (spread y z) = Real.log (ratMax y z) - Real.log (ratMin y z) := by
    rw [spread, Real.log_div hMpos.ne' hmpos.ne']
  have h1 : ∀ k : I, Real.log (ratMin y z) ≤ Real.log (z k / y k) := fun k =>
    Real.log_le_log hmpos (ratMin_le y z k)
  have h2 : ∀ k : I, Real.log (z k / y k) ≤ Real.log (ratMax y z) := fun k =>
    Real.log_le_log (div_pos (hz k) (hy k)) (le_ratMax y z k)
  constructor
  · linarith [h1 j, h2 i, hlogsp]
  · linarith [h2 j, h1 i, hlogsp]

/-- **The reported free energy differences converge geometrically.**  After `n` iterations every
free energy difference is within `(1 − 2·eps)^n · (spread − 1)` nats of the truth. -/
theorem freeEnergy_error_le [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ} {y z : I → ℝ}
    (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w)
    (hy : IsSolution N W mu y) (hz : ∀ i, 0 < z i)
    {eps : ℝ} (heps0 : 0 < eps) (heps1 : eps ≤ 1 / 2)
    (hov : ∀ w j, eps ≤ mixWeight N W y w j) (n : ℕ) (i j : I) :
    |(freeEnergy (iter N W mu z n) i - freeEnergy (iter N W mu z n) j)
        - (freeEnergy y i - freeEnergy y j)|
      ≤ (1 - 2 * eps) ^ n * (spread y z - 1) := by
  have hk := iter_pos (N := N) (W := W) (mu := mu) (z := z) hN hW hmu hz n
  have h1 := freeEnergy_diff_error_le_log_spread (y := y) (z := iter N W mu z n) hy.1 hk i j
  have h2 := spread_iterate (N := N) (W := W) (mu := mu) (y := y) (z := z) hN hW hmu hy hz
    heps0 heps1 hov n
  have h3 : Real.log (spread y (iter N W mu z n)) ≤ spread y (iter N W mu z n) - 1 := by
    have := Real.log_le_sub_one_of_pos (x := spread y (iter N W mu z n))
      (lt_of_lt_of_le zero_lt_one (one_le_spread hy.1 hk))
    linarith
  linarith

/-- **The iteration converges**, at the geometric rate given by the overlap constant. -/
theorem freeEnergy_error_tendsto_zero [Nonempty Ω] {N : I → ℝ} {W : I → Ω → ℝ} {mu : Ω → ℝ}
    {y z : I → ℝ} (hN : ∀ i, 0 < N i) (hW : ∀ i w, 0 < W i w) (hmu : ∀ w, 0 < mu w)
    (hy : IsSolution N W mu y) (hz : ∀ i, 0 < z i)
    {eps : ℝ} (heps0 : 0 < eps) (heps1 : eps ≤ 1 / 2)
    (hov : ∀ w j, eps ≤ mixWeight N W y w j) (i j : I) :
    Filter.Tendsto
      (fun n => (freeEnergy (iter N W mu z n) i - freeEnergy (iter N W mu z n) j))
      Filter.atTop (nhds (freeEnergy y i - freeEnergy y j)) := by
  rw [← tendsto_sub_nhds_zero_iff]
  refine squeeze_zero_norm (a := fun n => (1 - 2 * eps) ^ n * (spread y z - 1))
    (fun n => ?_) ?_
  · rw [Real.norm_eq_abs]
    exact freeEnergy_error_le hN hW hmu hy hz heps0 heps1 hov n i j
  · have hgeom : Filter.Tendsto (fun n : ℕ => (1 - 2 * eps) ^ n) Filter.atTop (nhds 0) :=
      tendsto_pow_atTop_nhds_zero_of_lt_one (by linarith) (by linarith)
    simpa using hgeom.mul_const (spread y z - 1)

end IDR.MBAR
