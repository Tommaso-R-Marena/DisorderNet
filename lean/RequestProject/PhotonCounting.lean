/-
# Part XXXIII  Single-molecule histograms: how much of the width is the molecule?

Single-molecule FRET is the experiment most often cited as *direct* evidence that a disordered
region is heterogeneous: one records a burst of photons from each molecule, computes the
fraction detected in the acceptor channel, and reports a histogram.  A broad histogram is read
as a broad conformational ensemble.  This file shows exactly how much of that width is the
molecule and how much is the detector, and it does so from first principles: the photons of a
burst are counted, and a burst of `N` photons from a conformation of transfer efficiency `e`
yields `a` acceptor photons with the binomial weight `binom N e a`, whose moments are obtained
here from Mathlib's Bernstein-polynomial identities (`binom_sum`, `binom_mean`, `binom_var`).

Write `M = Σ_k w_k e_k` for the ensemble-mean efficiency (`meanEff`), `varConf` for the
conformational variance of the efficiency, and `shotNoise w eff N = (Σ_k w_k e_k(1−e_k))/N`.

* `measured_unbiased` -- the histogram is centred correctly: the mean of the measured
  efficiencies is exactly the ensemble mean efficiency, for every burst size.
* `shotnoise_decomposition` -- **the exact law of total variance for the experiment**:
  `Var(measured) = varConf + shotNoise`.  The width of the histogram is the conformational
  width *plus* a detector term that no amount of data reduction removes from a single burst.
* `homogeneous_histogram_has_width` -- a completely homogeneous ensemble, a single conformation
  with `0 < e < 1`, produces a histogram of strictly positive width `e(1−e)/N`.  Width is
  therefore not evidence of heterogeneity.
* `width_not_evidence_of_heterogeneity` -- quantitatively: a single conformation at `e = 1/2`
  observed with `100` photons per burst, and a genuinely two-state ensemble at `e = 0.46, 0.54`
  observed with `276` photons per burst, produce histograms of *exactly* the same variance
  `1/400`, though one has zero conformational variance and the other `1/625`.  Only the photon
  budget distinguishes them.
* `shotNoise_le_quarter`, `photons_needed` -- what it takes to see the molecule: the shot-noise
  term never exceeds `1/(4N)`, so `N ≥ 1/(4·varConf)` photons per burst suffice to bring the
  detector term below the conformational term — the photon budget required to resolve a
  sub-population grows as the inverse square of the efficiency splitting it produces.
* `heterogeneity_detected` -- and a positive result: a histogram wider than `1/(4N)` *is*
  evidence, since the excess is a lower bound on the conformational variance.

Design consequence: a model of a disordered region must be compared with single-molecule data
through the forward model — predicted efficiencies, convolved with the photon statistics of the
actual bursts — and a reported histogram width is an upper bound on the ensemble's width, not a
measurement of it.
-/
import Mathlib

set_option autoImplicit false

namespace Photon

open Finset

/-- The probability of `a` acceptor photons in a burst of `N` photons from a conformation of
transfer efficiency `e`. -/
noncomputable def binom (N : ℕ) (e : ℝ) (a : ℕ) : ℝ :=
  (N.choose a : ℝ) * e ^ a * (1 - e) ^ (N - a)

lemma binom_eq_eval (N : ℕ) (e : ℝ) (a : ℕ) :
    binom N e a = (bernsteinPolynomial ℝ N a).eval e := by
  simp [binom, bernsteinPolynomial]

/-- The photon counts are a probability distribution. -/
lemma binom_sum (N : ℕ) (e : ℝ) : ∑ a ∈ Finset.range (N + 1), binom N e a = 1 := by
  simpa [binom_eq_eval, Polynomial.eval_finset_sum] using
    congrArg (Polynomial.eval e) (bernsteinPolynomial.sum ℝ N)

/-- The mean number of acceptor photons is `N e`. -/
lemma binom_mean (N : ℕ) (e : ℝ) :
    ∑ a ∈ Finset.range (N + 1), (a : ℝ) * binom N e a = N * e := by
  simpa [binom_eq_eval, Polynomial.eval_finset_sum, nsmul_eq_mul] using
    congrArg (Polynomial.eval e) (bernsteinPolynomial.sum_smul ℝ N)

/-- The variance of the number of acceptor photons is `N e (1 − e)`. -/
lemma binom_var (N : ℕ) (e : ℝ) :
    ∑ a ∈ Finset.range (N + 1), ((N : ℝ) * e - a) ^ 2 * binom N e a = N * e * (1 - e) := by
  simpa [binom_eq_eval, Polynomial.eval_finset_sum, nsmul_eq_mul] using
    congrArg (Polynomial.eval e) (bernsteinPolynomial.variance ℝ N)

/-- A burst reports the transfer efficiency without bias. -/
lemma burst_mean {N : ℕ} (hN : 0 < N) (e : ℝ) :
    ∑ a ∈ Finset.range (N + 1), ((a : ℝ) / N) * binom N e a = e := by
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hrw : ∑ a ∈ Finset.range (N + 1), ((a : ℝ) / N) * binom N e a
      = (1 / (N : ℝ)) * ∑ a ∈ Finset.range (N + 1), (a : ℝ) * binom N e a := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun a _ => by field_simp)
  rw [hrw, binom_mean]
  field_simp

/-- The mean squared deviation of a single burst from an arbitrary reference `M`. -/
lemma burst_second_moment {N : ℕ} (hN : 0 < N) (e M : ℝ) :
    ∑ a ∈ Finset.range (N + 1), ((a : ℝ) / N - M) ^ 2 * binom N e a
      = e * (1 - e) / N + (e - M) ^ 2 := by
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have key : ∀ a ∈ Finset.range (N + 1),
      ((a : ℝ) / N - M) ^ 2 * binom N e a
        = (1 / (N : ℝ) ^ 2) * (((N : ℝ) * e - a) ^ 2 * binom N e a)
          + (2 * (e - M) / (N : ℝ)) * ((a : ℝ) * binom N e a)
          + (-(2 * (e - M) * e) + (e - M) ^ 2) * binom N e a := by
    intro a _
    field_simp
    ring
  rw [Finset.sum_congr rfl key]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [binom_var, binom_mean, binom_sum]
  field_simp
  ring

/-! ### The ensemble seen through the photon budget -/

variable {m : ℕ}

/-- The ensemble-mean transfer efficiency. -/
noncomputable def meanEff (w eff : Fin m → ℝ) : ℝ := ∑ k, w k * eff k

/-- The conformational variance of the transfer efficiency: the width of the *ensemble*. -/
noncomputable def varConf (w eff : Fin m → ℝ) : ℝ := ∑ k, w k * (eff k - meanEff w eff) ^ 2

/-- The shot-noise term: the width contributed by the finite photon budget. -/
noncomputable def shotNoise (w eff : Fin m → ℝ) (N : ℕ) : ℝ :=
  (∑ k, w k * (eff k * (1 - eff k))) / N

/-- The mean of the measured burst efficiencies. -/
noncomputable def measuredMean (w eff : Fin m → ℝ) (N : ℕ) : ℝ :=
  ∑ k, w k * ∑ a ∈ Finset.range (N + 1), ((a : ℝ) / N) * binom N (eff k) a

/-- The variance of the measured burst efficiencies: the width of the *histogram*. -/
noncomputable def measuredVar (w eff : Fin m → ℝ) (N : ℕ) : ℝ :=
  ∑ k, w k * ∑ a ∈ Finset.range (N + 1),
    ((a : ℝ) / N - meanEff w eff) ^ 2 * binom N (eff k) a

/-- **The histogram is centred correctly.** -/
theorem measured_unbiased {N : ℕ} (hN : 0 < N) (w eff : Fin m → ℝ) :
    measuredMean w eff N = meanEff w eff := by
  simp only [measuredMean, meanEff]
  exact Finset.sum_congr rfl (fun k _ => by rw [burst_mean hN])

/-- **The exact law of total variance for a single-molecule experiment.**  The width of the
histogram is the conformational width plus a shot-noise term. -/
theorem shotnoise_decomposition {N : ℕ} (hN : 0 < N) (w eff : Fin m → ℝ) :
    measuredVar w eff N = varConf w eff + shotNoise w eff N := by
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  simp only [measuredVar, varConf, shotNoise]
  rw [Finset.sum_div, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [burst_second_moment hN]
  field_simp
  ring

/-- **A homogeneous ensemble still produces a broad histogram.**  A single conformation with
`0 < e < 1` gives a histogram of strictly positive variance `e(1−e)/N`, and its conformational
variance is zero. -/
theorem homogeneous_histogram_has_width {N : ℕ} (hN : 0 < N) {e : ℝ} (h0 : 0 < e) (h1 : e < 1) :
    varConf ![1] ![e] = 0 ∧ measuredVar ![1] ![e] N = e * (1 - e) / N ∧
      0 < measuredVar ![1] ![e] N := by
  have hmean : meanEff ![1] ![e] = e := by simp [meanEff]
  have hvar : varConf ![1] ![e] = 0 := by simp [varConf, hmean]
  have hshot : shotNoise ![1] ![e] N = e * (1 - e) / N := by simp [shotNoise]
  have hmeas : measuredVar ![1] ![e] N = e * (1 - e) / N := by
    rw [shotnoise_decomposition hN, hvar, hshot, zero_add]
  refine ⟨hvar, hmeas, ?_⟩
  rw [hmeas]
  have : (0 : ℝ) < N := by exact_mod_cast hN
  have h1' : 0 < 1 - e := by linarith
  positivity

/-- **Histogram width is not evidence of heterogeneity.**  A single conformation at `e = 1/2`
seen with `100` photons per burst and a two-state ensemble at `e = 0.46, 0.54` seen with `276`
photons per burst give histograms of exactly the same variance `1/400`, while their
conformational variances are `0` and `1/625`. -/
theorem width_not_evidence_of_heterogeneity :
    varConf ![1] ![1/2] = 0 ∧
    varConf ![1/2, 1/2] ![23/50, 27/50] = 1/625 ∧
    measuredVar ![1] ![(1/2 : ℝ)] 100 = 1/400 ∧
    measuredVar ![1/2, 1/2] ![(23/50 : ℝ), 27/50] 276 = 1/400 := by
  have hmean1 : meanEff ![1] ![(1/2 : ℝ)] = 1/2 := by norm_num [meanEff]
  have hmean2 : meanEff ![1/2, 1/2] ![(23/50 : ℝ), 27/50] = 1/2 := by
    norm_num [meanEff, Fin.sum_univ_two]
  have hv1 : varConf ![1] ![(1/2 : ℝ)] = 0 := by norm_num [varConf, hmean1]
  have hv2 : varConf ![1/2, 1/2] ![(23/50 : ℝ), 27/50] = 1/625 := by
    norm_num [varConf, hmean2, Fin.sum_univ_two]
  refine ⟨hv1, hv2, ?_, ?_⟩
  · rw [shotnoise_decomposition (by norm_num), hv1]
    norm_num [shotNoise]
  · rw [shotnoise_decomposition (by norm_num), hv2]
    norm_num [shotNoise, Fin.sum_univ_two]

/-- The shot-noise term never exceeds `1/(4N)`. -/
theorem shotNoise_le_quarter {N : ℕ} (hN : 0 < N) {w eff : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k)
    (hsum : ∑ k, w k = 1) : shotNoise w eff N ≤ 1 / (4 * N) := by
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have hterm : ∑ k, w k * (eff k * (1 - eff k)) ≤ ∑ k, w k * (1 / 4) := by
    refine Finset.sum_le_sum (fun k _ => ?_)
    have : eff k * (1 - eff k) ≤ 1 / 4 := by nlinarith [sq_nonneg (eff k - 1 / 2)]
    exact mul_le_mul_of_nonneg_left this (hw k)
  rw [← Finset.sum_mul, hsum, one_mul] at hterm
  rw [shotNoise, div_le_div_iff₀ hNpos (by positivity)]
  nlinarith [hterm, hNpos]

/-- **The photon budget needed to see the molecule.**  With `N ≥ 1/(4·varConf)` photons per
burst the detector contributes less than the ensemble does. -/
theorem photons_needed {N : ℕ} (hN : 0 < N) {w eff : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k)
    (hsum : ∑ k, w k = 1) (hv : 0 < varConf w eff) (hbudget : 1 / (4 * varConf w eff) ≤ N) :
    shotNoise w eff N ≤ varConf w eff := by
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have h1 : shotNoise w eff N ≤ 1 / (4 * N) := shotNoise_le_quarter hN hw hsum
  have h2 : 1 / (4 * (N : ℝ)) ≤ varConf w eff := by
    rw [div_le_iff₀ (by positivity)]
    rw [div_le_iff₀ (by positivity)] at hbudget
    linarith
  linarith

/-- **A positive result: excess width is evidence.**  Whatever the photon budget, the amount by
which the histogram exceeds `1/(4N)` is a lower bound on the conformational variance. -/
theorem heterogeneity_detected {N : ℕ} (hN : 0 < N) {w eff : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k)
    (hsum : ∑ k, w k = 1) :
    measuredVar w eff N - 1 / (4 * N) ≤ varConf w eff := by
  have h1 : shotNoise w eff N ≤ 1 / (4 * N) := shotNoise_le_quarter hN hw hsum
  rw [shotnoise_decomposition hN]
  linarith


/-! ### Dynamic averaging inside a burst

A burst is not instantaneous.  If the chain interconverts on a timescale shorter than the burst
duration, the photons of one burst are emitted by *several* independently sampled
conformations, and the burst reports their average.  Splitting a burst of `2n` photons into two
independently sampled halves of `n` photons each, the histogram width becomes `varConf/2` plus
*the same* shot-noise term (`dynamic_averaging`): time averaging inside the burst destroys the
conformational width while leaving the detector width untouched.  A narrow histogram is
therefore no more evidence of homogeneity than a broad one is of heterogeneity — it may simply
mean that the exchange is fast on the burst timescale. -/

/-- The variance of an average of two independent draws, with an arbitrary reference point. -/
theorem pair_average_var {iota : Type*} [Fintype iota] (p v : iota → ℝ) (hp : ∑ i, p i = 1)
    (M : ℝ) :
    ∑ i, ∑ j, p i * p j * (((v i + v j) / 2) - M) ^ 2
      = (1 / 2) * (∑ i, p i * (v i - M) ^ 2) + (1 / 2) * (∑ i, p i * (v i - M)) ^ 2 := by
  have key : ∀ i j : iota, p i * p j * (((v i + v j) / 2) - M) ^ 2
      = (1 / 4) * (p i * (v i - M) ^ 2 * p j)
        + (1 / 2) * ((p i * (v i - M)) * (p j * (v j - M)))
        + (1 / 4) * (p i * (p j * (v j - M) ^ 2)) := by
    intro i j; ring
  have expand : ∀ i : iota, ∑ j, p i * p j * (((v i + v j) / 2) - M) ^ 2
      = (1 / 4) * (p i * (v i - M) ^ 2) * (∑ j, p j)
        + (1 / 2) * (p i * (v i - M)) * (∑ j, p j * (v j - M))
        + (1 / 4) * p i * (∑ j, p j * (v j - M) ^ 2) := by
    intro i
    simp only [key, Finset.sum_add_distrib, ← Finset.mul_sum]
    ring
  simp only [expand, hp, mul_one]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib]
  simp only [← Finset.sum_mul, ← Finset.mul_sum]
  rw [hp]
  ring

/-- The joint weight of (conformation, acceptor count) for a sub-burst of `n` photons. -/
noncomputable def subWeight (w eff : Fin m → ℝ) (n : ℕ) : Fin m × Fin (n + 1) → ℝ :=
  fun i => w i.1 * binom n (eff i.1) i.2

/-- The efficiency reported by a sub-burst of `n` photons. -/
noncomputable def subEff (m n : ℕ) : Fin m × Fin (n + 1) → ℝ := fun i => (i.2 : ℝ) / n

/-- The width of the histogram when each burst of `2n` photons samples two independent
conformations. -/
noncomputable def splitVar (w eff : Fin m → ℝ) (n : ℕ) : ℝ :=
  ∑ i, ∑ j, subWeight w eff n i * subWeight w eff n j *
    ((subEff m n i + subEff m n j) / 2 - meanEff w eff) ^ 2

lemma sum_subWeight {n : ℕ} {w eff : Fin m → ℝ} (hw : ∑ k, w k = 1) :
    ∑ i : Fin m × Fin (n + 1), subWeight w eff n i = 1 := by
  rw [Fintype.sum_prod_type]
  have hk : ∀ k : Fin m, ∑ a : Fin (n + 1), subWeight w eff n (k, a) = w k := by
    intro k
    simp only [subWeight]
    rw [← Finset.mul_sum]
    rw [show (∑ a : Fin (n + 1), binom n (eff k) (a : ℕ))
        = ∑ a ∈ Finset.range (n + 1), binom n (eff k) a from
      Fin.sum_univ_eq_sum_range (fun a => binom n (eff k) a) (n + 1)]
    rw [binom_sum, mul_one]
  simp only [hk, hw]

lemma sum_subWeight_dev {n : ℕ} (hn : 0 < n) {w eff : Fin m → ℝ} (hw : ∑ k, w k = 1) :
    ∑ i : Fin m × Fin (n + 1), subWeight w eff n i * (subEff m n i - meanEff w eff) = 0 := by
  rw [Fintype.sum_prod_type]
  have hk : ∀ k : Fin m,
      ∑ a : Fin (n + 1), subWeight w eff n (k, a) * (subEff m n (k, a) - meanEff w eff)
        = w k * (eff k - meanEff w eff) := by
    intro k
    have hexp : ∀ a : Fin (n + 1),
        subWeight w eff n (k, a) * (subEff m n (k, a) - meanEff w eff)
          = w k * (((a : ℕ) : ℝ) / n * binom n (eff k) a)
            - w k * meanEff w eff * binom n (eff k) a := by
      intro a; simp only [subWeight, subEff]; ring
    rw [Finset.sum_congr rfl (fun a _ => hexp a)]
    rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    rw [show (∑ a : Fin (n + 1), ((a : ℕ) : ℝ) / n * binom n (eff k) a)
        = ∑ a ∈ Finset.range (n + 1), (a : ℝ) / n * binom n (eff k) a from
      Fin.sum_univ_eq_sum_range (fun a => (a : ℝ) / n * binom n (eff k) a) (n + 1)]
    rw [show (∑ a : Fin (n + 1), binom n (eff k) (a : ℕ))
        = ∑ a ∈ Finset.range (n + 1), binom n (eff k) a from
      Fin.sum_univ_eq_sum_range (fun a => binom n (eff k) a) (n + 1)]
    rw [burst_mean hn, binom_sum, mul_one]
    ring
  simp only [hk]
  have hsplit : ∑ k, w k * (eff k - meanEff w eff)
      = (∑ k, w k * eff k) - (∑ k, w k) * meanEff w eff := by
    rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun k _ => by ring)
  rw [hsplit, hw, one_mul]
  simp [meanEff]

lemma sum_subWeight_dev_sq {n : ℕ} {w eff : Fin m → ℝ} :
    ∑ i : Fin m × Fin (n + 1), subWeight w eff n i * (subEff m n i - meanEff w eff) ^ 2
      = measuredVar w eff n := by
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  simp only [subWeight, subEff]
  rw [Finset.mul_sum]
  rw [show (∑ a : Fin (n + 1),
        w k * binom n (eff k) (a : ℕ) * (((a : ℕ) : ℝ) / n - meanEff w eff) ^ 2)
      = ∑ a ∈ Finset.range (n + 1),
        w k * binom n (eff k) a * ((a : ℝ) / n - meanEff w eff) ^ 2 from
    Fin.sum_univ_eq_sum_range
      (fun a => w k * binom n (eff k) a * ((a : ℝ) / n - meanEff w eff) ^ 2) (n + 1)]
  exact Finset.sum_congr rfl (fun a _ => by ring)

/-- **Dynamic averaging inside a burst halves the conformational width and leaves the shot
noise unchanged.**  If each burst of `2n` photons samples two independent conformations, the
histogram variance is `varConf/2 + shotNoise(2n)` — the same detector term as a static burst of
`2n` photons, but only half the conformational term. -/
theorem dynamic_averaging {n : ℕ} (hn : 0 < n) {w eff : Fin m → ℝ} (hw : ∑ k, w k = 1) :
    splitVar w eff n = varConf w eff / 2 + shotNoise w eff (2 * n) := by
  have hNne : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [splitVar, pair_average_var (subWeight w eff n) (subEff m n) (sum_subWeight hw)
    (meanEff w eff), sum_subWeight_dev hn hw, sum_subWeight_dev_sq,
    shotnoise_decomposition hn]
  have hcast : ((2 * n : ℕ) : ℝ) = 2 * (n : ℝ) := by push_cast; ring
  simp only [shotNoise, hcast]
  field_simp
  ring

end Photon
