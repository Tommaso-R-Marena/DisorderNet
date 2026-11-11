/-
# Part XXIX.3  Error bars: frames of a trajectory are not independent samples

The third systematic error in a reported ensemble is statistical, and it is the one most often
reported wrongly: populations and averages extracted from a molecular dynamics trajectory are
quoted with `σ/√N` error bars computed from `N` *frames*, while the frames are correlated over
the conformational relaxation time of the region.  This file computes the true variance of a
trajectory average for the standard model of a correlated stationary series -- exponentially
decaying autocorrelation `γ(k) = σ² ρ^k`, `ρ = e^{-Δt/τ}`, which is exactly the autocorrelation
of the two-state exchange of `RequestProject.Relaxation` and of a Rouse mode
(`RequestProject.Rouse`) sampled at interval `Δt`.

* `corrSum_eq` -- the exact double sum: `Σ_{i,j<N} ρ^{|i-j|} = N(1+ρ)/(1-ρ) - 2ρ(1-ρ^N)/(1-ρ)²`.
* `varMean_eq_iid` -- at `ρ = 0` this is the textbook `σ²/N`.
* `varMean_frozen` -- at `ρ = 1` it is `σ²`, *independently of `N`*: a trajectory shorter than
  the relaxation time contains exactly one sample however often it is written to disk.
* `varMean_ge_iid` -- in between, the true variance is never smaller than the independent-sample
  formula: naive error bars are always optimistic.
* `varMean_ge_inflated` -- and by how much: the variance is at least
  `(σ²/N)·((1+ρ)/(1-ρ) - 2ρ/(N(1-ρ)²))`, i.e. the statistical inefficiency `(1+ρ)/(1-ρ) ≈ 2τ/Δt`
  multiplies the naive variance.
* `frames_needed` -- inverting it, reaching a target error `eps` on an ensemble average needs
  at least `(σ²/eps²)·((1+ρ)/(1-ρ)) - 2ρ/(1-ρ)²` frames: the cost of an error bar scales with
  the relaxation time, which for a disordered region under a barrier is itself exponential
  (`RequestProject.Relaxation`, `RequestProject.Mixing`).

Design consequence: a model of a disordered region that is fit to, or validated against,
simulation-derived populations must propagate the *effective* sample size `N(1-ρ)/(1+ρ)`, not
the frame count.  Every capacity and sample-complexity bound in Parts V--VII is stated in
independent samples, and this is the conversion factor.
-/
import Mathlib

set_option autoImplicit false

namespace CorrSample

open Finset

/-- `Σ_{i,j<N} ρ^{|i-j|}`, the sum of the autocorrelation over all pairs of frames. -/
noncomputable def corrSum (rho : ℝ) (N : ℕ) : ℝ :=
  ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N, rho ^ (max i j - min i j)

/-- Variance of the average of `N` frames with variance `sigma2` and autocorrelation
`ρ^{|i-j|}`. -/
noncomputable def varMean (sigma2 rho : ℝ) (N : ℕ) : ℝ := sigma2 / (N : ℝ) ^ 2 * corrSum rho N

lemma corrSum_zero (rho : ℝ) : corrSum rho 0 = 0 := by simp [corrSum]

/-- One more frame adds its correlation with all the previous ones, twice. -/
lemma corrSum_succ (rho : ℝ) (N : ℕ) :
    corrSum rho (N + 1)
      = corrSum rho N + 2 * (∑ k ∈ Finset.range N, rho ^ (k + 1)) + 1 := by
  have hlast : ∀ i ∈ Finset.range N, rho ^ (max i N - min i N) = rho ^ (N - i) := by
    intro i hi
    have hiN : i ≤ N := (Finset.mem_range.mp hi).le
    rw [max_eq_right hiN, min_eq_left hiN]
  have hrow : ∀ i ∈ Finset.range N,
      ∑ j ∈ Finset.range (N + 1), rho ^ (max i j - min i j)
        = (∑ j ∈ Finset.range N, rho ^ (max i j - min i j)) + rho ^ (N - i) := by
    intro i hi
    rw [Finset.sum_range_succ, hlast i hi]
  have hcol : ∑ j ∈ Finset.range N, rho ^ (max N j - min N j)
      = ∑ j ∈ Finset.range N, rho ^ (N - j) := by
    refine Finset.sum_congr rfl fun j hj => ?_
    have hjN : j ≤ N := (Finset.mem_range.mp hj).le
    rw [max_eq_left hjN, min_eq_right hjN]
  have hreflect : ∑ i ∈ Finset.range N, rho ^ (N - i) = ∑ k ∈ Finset.range N, rho ^ (k + 1) := by
    rw [← Finset.sum_range_reflect]
    refine Finset.sum_congr rfl fun k hk => ?_
    have hk' : k < N := Finset.mem_range.mp hk
    congr 1
    omega
  rw [corrSum, Finset.sum_range_succ, Finset.sum_congr rfl hrow, Finset.sum_add_distrib,
    Finset.sum_range_succ, hcol, hreflect]
  simp [corrSum]
  ring

/-- **The exact variance of a correlated average.** -/
theorem corrSum_eq {rho : ℝ} (h : rho ≠ 1) (N : ℕ) :
    corrSum rho N = N * (1 + rho) / (1 - rho) - 2 * rho * (1 - rho ^ N) / (1 - rho) ^ 2 := by
  have hne : (1 : ℝ) - rho ≠ 0 := sub_ne_zero.mpr (Ne.symm h)
  induction N with
  | zero => simp [corrSum_zero]
  | succ N ih =>
    have hgeom : ∑ k ∈ Finset.range N, rho ^ (k + 1) = rho * (1 - rho ^ N) / (1 - rho) := by
      have : ∑ k ∈ Finset.range N, rho ^ (k + 1) = rho * ∑ k ∈ Finset.range N, rho ^ k := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun k _ => by ring
      have h' : rho - 1 ≠ 0 := sub_ne_zero.mpr h
      rw [this, geom_sum_eq h N]
      field_simp
      ring
    rw [corrSum_succ, ih, hgeom]
    push_cast
    field_simp
    ring

/-- At zero autocorrelation the textbook formula is recovered. -/
theorem varMean_eq_iid (sigma2 : ℝ) {N : ℕ} (hN : 0 < N) :
    varMean sigma2 0 N = sigma2 / N := by
  have hNR : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have h : corrSum 0 N = N := by
    rw [corrSum_eq (by norm_num) N]
    simp
  rw [varMean, h]
  field_simp

/-- **A frozen trajectory has one frame.**  With perfect correlation the variance of the average
is the variance of a single frame, however many frames are written. -/
theorem varMean_frozen (sigma2 : ℝ) {N : ℕ} (hN : 0 < N) : varMean sigma2 1 N = sigma2 := by
  have hNR : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have h : corrSum 1 N = (N : ℝ) ^ 2 := by
    simp [corrSum, Finset.sum_const, Finset.card_range]
    ring
  rw [varMean, h]
  field_simp

lemma corrSum_ge_card {rho : ℝ} (hrho : 0 ≤ rho) (N : ℕ) : (N : ℝ) ≤ corrSum rho N := by
  have hdiag : ∀ i ∈ Finset.range N, (1 : ℝ) ≤ ∑ j ∈ Finset.range N, rho ^ (max i j - min i j) := by
    intro i hi
    have hnn : ∀ j ∈ Finset.range N, 0 ≤ rho ^ (max i j - min i j) :=
      fun j _ => pow_nonneg hrho _
    have hii : (1 : ℝ) = rho ^ (max i i - min i i) := by simp
    rw [hii]
    exact Finset.single_le_sum hnn hi
  calc (N : ℝ) = ∑ _i ∈ Finset.range N, (1 : ℝ) := by simp
    _ ≤ corrSum rho N := Finset.sum_le_sum hdiag

/-- **Naive error bars are always optimistic.** -/
theorem varMean_ge_iid {sigma2 rho : ℝ} (hs : 0 ≤ sigma2) (hrho : 0 ≤ rho) {N : ℕ} (hN : 0 < N) :
    sigma2 / N ≤ varMean sigma2 rho N := by
  have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  have h := corrSum_ge_card hrho N
  rw [varMean, div_le_iff₀ hNR]
  have hfac : 0 ≤ sigma2 / (N : ℝ) ^ 2 := div_nonneg hs (by positivity)
  calc sigma2 = sigma2 / (N : ℝ) ^ 2 * (N : ℝ) * N := by field_simp
    _ ≤ sigma2 / (N : ℝ) ^ 2 * corrSum rho N * N := by
        have := mul_le_mul_of_nonneg_left h hfac
        exact mul_le_mul_of_nonneg_right this hNR.le

/-- **The statistical inefficiency.**  The true variance is at least the naive one inflated by
`(1+ρ)/(1-ρ)`, up to a `1/N` correction: for `ρ = e^{-Δt/τ}` this factor is `≈ 2τ/Δt`. -/
theorem varMean_ge_inflated {sigma2 rho : ℝ} (hs : 0 ≤ sigma2) (hrho : 0 ≤ rho) (h1 : rho < 1)
    {N : ℕ} (hN : 0 < N) :
    sigma2 / N * ((1 + rho) / (1 - rho) - 2 * rho / ((N : ℝ) * (1 - rho) ^ 2))
      ≤ varMean sigma2 rho N := by
  have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  have hne : (0 : ℝ) < 1 - rho := by linarith
  have hpow : 0 ≤ rho ^ N := pow_nonneg hrho N
  have hexact := corrSum_eq (ne_of_lt h1) N
  have hlb : (N : ℝ) * (1 + rho) / (1 - rho) - 2 * rho / (1 - rho) ^ 2 ≤ corrSum rho N := by
    rw [hexact]
    have hnum : 2 * rho * (1 - rho ^ N) ≤ 2 * rho := by nlinarith [mul_nonneg hrho hpow]
    have : 2 * rho * (1 - rho ^ N) / (1 - rho) ^ 2 ≤ 2 * rho / (1 - rho) ^ 2 := by
      gcongr
    linarith
  have hfac : 0 ≤ sigma2 / (N : ℝ) ^ 2 := div_nonneg hs (by positivity)
  have hmul := mul_le_mul_of_nonneg_left hlb hfac
  rw [varMean]
  refine le_trans (le_of_eq ?_) hmul
  field_simp

/-- **How many frames an error bar costs.**  Reaching variance `eps` on an ensemble average
requires at least `(σ²/eps)·((1+ρ)/(1-ρ) - 2ρ/(1-ρ)²)` frames: the relaxation time of the
region, not the frame count, sets the price of an error bar. -/
theorem frames_needed {sigma2 rho eps : ℝ} (hs : 0 ≤ sigma2) (hrho : 0 ≤ rho) (h1 : rho < 1)
    (heps : 0 < eps) {N : ℕ} (hN : 0 < N) (hgoal : varMean sigma2 rho N ≤ eps) :
    sigma2 / eps * ((1 + rho) / (1 - rho) - 2 * rho / (1 - rho) ^ 2) ≤ N := by
  have hNR : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  have hne : (0 : ℝ) < 1 - rho := by linarith
  have hstep := varMean_ge_inflated hs hrho h1 hN
  have hchain : sigma2 / N * ((1 + rho) / (1 - rho) - 2 * rho / ((N : ℝ) * (1 - rho) ^ 2))
      ≤ eps := le_trans hstep hgoal
  have hexp : sigma2 / N * ((1 + rho) / (1 - rho) - 2 * rho / ((N : ℝ) * (1 - rho) ^ 2))
      = (sigma2 * ((1 + rho) / (1 - rho)) - sigma2 * (2 * rho / (1 - rho) ^ 2) / N) / N := by
    field_simp
  rw [hexp, div_le_iff₀ hNR] at hchain
  have hpos2 : 0 ≤ sigma2 * (2 * rho / (1 - rho) ^ 2) / N :=
    div_nonneg (mul_nonneg hs (div_nonneg (by linarith) (by positivity))) hNR.le
  have hlin : sigma2 * ((1 + rho) / (1 - rho)) ≤ eps * N + sigma2 * (2 * rho / (1 - rho) ^ 2) / N :=
    by linarith
  have hbound : sigma2 * (2 * rho / (1 - rho) ^ 2) / N ≤ sigma2 * (2 * rho / (1 - rho) ^ 2) := by
    rcases eq_or_lt_of_le (Nat.one_le_cast.mpr hN : (1 : ℝ) ≤ N) with h | h
    · rw [← h]; simp
    · exact div_le_self (mul_nonneg hs (div_nonneg (by linarith) (by positivity))) h.le
  have : sigma2 * ((1 + rho) / (1 - rho)) - sigma2 * (2 * rho / (1 - rho) ^ 2) ≤ eps * N := by
    linarith
  have hdiv : sigma2 / eps * ((1 + rho) / (1 - rho)) - sigma2 / eps * (2 * rho / (1 - rho) ^ 2)
      ≤ N := by
    rw [div_mul_eq_mul_div, div_mul_eq_mul_div, div_sub_div_same, div_le_iff₀ heps]
    linarith [this]
  rw [mul_sub]
  exact hdiv

end CorrSample
