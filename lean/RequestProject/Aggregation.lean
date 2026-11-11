/-
# Part XXXVII  Aggregation kinetics: the lag time is a logarithm

Disordered regions are the parts of a proteome that aggregate, and the standard experiment is a
thioflavin curve: fibril mass against time, showing a flat lag phase, a steep growth phase and a
plateau.  The lag time is then quoted, and read as the time taken to nucleate.  This file asks
what a lag time is a statement about, in the standard nucleation–elongation model in its
exactly solvable early-time form: fibril mass obeys `M'' = κ² M` with `M(0) = 0` and `M'(0) = v`,
where `v` is set by the primary nucleation rate and `κ` by elongation and secondary nucleation.
The solution is `mass v κ t = (v/κ)·sinh(κt)` and everything below is exact.

* `mass_hasDerivAt`, `mass_second_deriv`, `mass_zero` -- the model is the stated initial value
  problem, and `v`, `κ` mean what they are said to mean.
* `mass_pos` -- there is no lag phase.  The mass is strictly positive at every positive time; the
  flat portion of a thioflavin trace is the interval on which the mass is below the detection
  threshold, not an interval on which nothing happens.
* `mass_invariant` -- the model has an exact constant of the motion, `(M')² − κ²M² = v²`, so a
  single time point with its slope determines `v` once `κ` is known.  This is the identifiable
  content of the curve.
* `mass_lagTime` -- the time at which the mass first reaches a threshold `Mc` is
  `lagTime = arsinh(κMc/v)/κ`: a *logarithm* of the nucleation rate.
* `lag_shift_le`, `lag_shift_ge` -- hence the two insensitivity laws.  Multiplying the detection
  threshold by `r ≥ 1` moves the apparent lag time by at most `log r / κ`, and multiplying the
  nucleation rate by `r ≥ 1` shortens it by at most `log r / κ`; and for a threshold at or above
  `v/κ` the shift is at least `(log r − log(3/2))/κ`, so the dependence is genuinely logarithmic
  and not weaker.  A model of the soluble ensemble that predicts the nucleation rate to within a
  factor of a hundred predicts the lag time to within `4.6/κ` — and conversely, a lag time
  measured to 10 % constrains the nucleation rate only to within a large multiplicative factor.
* `lag_underdetermined`, `two_models_one_lag` -- indeed the lag time alone determines nothing: for
  *every* growth rate `κ` there is a nucleation rate `v` reproducing a given lag time exactly, and
  two such fits with different `κ` have genuinely different `v`.  Lag time and growth rate must be
  reported together, and a model must be compared with the whole curve.

Design consequence: an ensemble model of a disordered region is not falsified or confirmed by a
lag time.  The quantity it can be held to is the pair `(v, κ)` — equivalently the curve — and the
`v` it predicts enters the observable only through its logarithm.
-/
import Mathlib

set_option autoImplicit false

namespace Agg

/-! ### The exactly solvable early-time model -/

/-- Fibril mass in the nucleation–elongation model: the solution of `M'' = κ²M`, `M(0) = 0`,
`M'(0) = v`, where `v` is the primary nucleation flux and `κ` the growth rate. -/
noncomputable def mass (v κ t : ℝ) : ℝ := v / κ * Real.sinh (κ * t)

/-- The time at which the mass reaches a detection threshold `Mc`. -/
noncomputable def lagTime (v κ Mc : ℝ) : ℝ := Real.arsinh (κ * Mc / v) / κ

@[simp] lemma mass_zero (v κ : ℝ) : mass v κ 0 = 0 := by simp [mass]

lemma mass_hasDerivAt (v κ : ℝ) (hκ : κ ≠ 0) (t : ℝ) :
    HasDerivAt (mass v κ) (v * Real.cosh (κ * t)) t := by
  have h1 : HasDerivAt (fun t : ℝ => κ * t) κ t := by
    simpa using (hasDerivAt_id t).const_mul κ
  have h2 : HasDerivAt (fun t : ℝ => Real.sinh (κ * t)) (Real.cosh (κ * t) * κ) t :=
    (Real.hasDerivAt_sinh (κ * t)).comp t h1
  have h3 := h2.const_mul (v / κ)
  convert h3 using 1
  field_simp

/-- The mass obeys `M'' = κ²M`: the second derivative is `κ²` times the mass. -/
lemma mass_second_deriv (v κ : ℝ) (hκ : κ ≠ 0) (t : ℝ) :
    HasDerivAt (fun s => v * Real.cosh (κ * s)) (κ ^ 2 * mass v κ t) t := by
  have h1 : HasDerivAt (fun t : ℝ => κ * t) κ t := by
    simpa using (hasDerivAt_id t).const_mul κ
  have h2 : HasDerivAt (fun t : ℝ => Real.cosh (κ * t)) (Real.sinh (κ * t) * κ) t :=
    (Real.hasDerivAt_cosh (κ * t)).comp t h1
  have h3 := h2.const_mul v
  convert h3 using 1
  simp only [mass]
  field_simp

/-- **There is no lag phase.**  For every positive time the fibril mass is strictly positive. -/
lemma mass_pos {v κ t : ℝ} (hv : 0 < v) (hκ : 0 < κ) (ht : 0 < t) : 0 < mass v κ t := by
  have h1 : 0 < Real.sinh (κ * t) := Real.sinh_pos_iff.mpr (by positivity)
  exact mul_pos (div_pos hv hκ) h1

/-- **The constant of the motion.**  `(M')² − κ²M² = v²` at every time, so a single measurement
of the mass and its slope determines the nucleation flux once the growth rate is known. -/
lemma mass_invariant (v κ : ℝ) (hκ : κ ≠ 0) (t : ℝ) :
    (v * Real.cosh (κ * t)) ^ 2 - κ ^ 2 * (mass v κ t) ^ 2 = v ^ 2 := by
  have hc := Real.cosh_sq (κ * t)
  simp only [mass]
  field_simp
  nlinarith [hc]

/-- The threshold is reached exactly at `lagTime`. -/
lemma mass_lagTime {v κ Mc : ℝ} (hv : v ≠ 0) (hκ : κ ≠ 0) :
    mass v κ (lagTime v κ Mc) = Mc := by
  simp only [mass, lagTime]
  rw [mul_div_cancel₀ _ hκ, Real.sinh_arsinh]
  field_simp

/-! ### The logarithmic law -/

/-- The key inequality: `arsinh` grows at most logarithmically, `arsinh (r x) ≤ log r + arsinh x`
for `r ≥ 1` and `x ≥ 0`. -/
lemma arsinh_mul_le {x r : ℝ} (hx : 0 ≤ x) (hr : 1 ≤ r) :
    Real.arsinh (r * x) ≤ Real.log r + Real.arsinh x := by
  have hr0 : (0:ℝ) < r := lt_of_lt_of_le zero_lt_one hr
  have hB : 0 < x + Real.sqrt (1 + x ^ 2) := by
    have : 0 < Real.sqrt (1 + x ^ 2) := Real.sqrt_pos.mpr (by positivity)
    linarith
  have hA : 0 < r * x + Real.sqrt (1 + (r * x) ^ 2) := by
    have : 0 < Real.sqrt (1 + (r * x) ^ 2) := Real.sqrt_pos.mpr (by positivity)
    nlinarith
  have hsq : Real.sqrt (1 + (r * x) ^ 2) ≤ r * Real.sqrt (1 + x ^ 2) := by
    have h1 : r * Real.sqrt (1 + x ^ 2) = Real.sqrt (r ^ 2 * (1 + x ^ 2)) := by
      rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hr0.le]
    rw [h1]
    apply Real.sqrt_le_sqrt
    nlinarith
  have hkey : r * x + Real.sqrt (1 + (r * x) ^ 2) ≤ r * (x + Real.sqrt (1 + x ^ 2)) := by
    nlinarith
  have hlog := Real.log_le_log hA hkey
  rw [Real.log_mul (ne_of_gt hr0) (ne_of_gt hB)] at hlog
  simpa [Real.arsinh] using hlog

/-- **The apparent lag time is logarithmic in the threshold and in the nucleation rate.**
Raising the detection threshold by a factor `r ≥ 1` delays the apparent lag by at most
`log r / κ`. -/
theorem lag_shift_le {v κ Mc r : ℝ} (hv : 0 < v) (hκ : 0 < κ) (hMc : 0 ≤ Mc) (hr : 1 ≤ r) :
    lagTime v κ (r * Mc) - lagTime v κ Mc ≤ Real.log r / κ := by
  have hx : 0 ≤ κ * Mc / v := by positivity
  have h := arsinh_mul_le hx hr
  have harg : κ * (r * Mc) / v = r * (κ * Mc / v) := by ring
  simp only [lagTime, harg]
  rw [div_sub_div_same, div_le_div_iff_of_pos_right hκ]
  linarith

/-- The same law in the other variable: multiplying the nucleation rate by `r ≥ 1` shortens the
lag time by at most `log r / κ`. -/
theorem lag_rate_shift_le {v κ Mc r : ℝ} (hv : 0 < v) (hκ : 0 < κ) (hMc : 0 ≤ Mc) (hr : 1 ≤ r) :
    lagTime v κ Mc - lagTime (r * v) κ Mc ≤ Real.log r / κ := by
  have hx : 0 ≤ κ * Mc / (r * v) := by
    have : (0:ℝ) < r := lt_of_lt_of_le zero_lt_one hr
    positivity
  have h := arsinh_mul_le hx hr
  have hr0 : (0:ℝ) < r := lt_of_lt_of_le zero_lt_one hr
  have harg : r * (κ * Mc / (r * v)) = κ * Mc / v := by field_simp
  rw [harg] at h
  simp only [lagTime]
  rw [div_sub_div_same, div_le_div_iff_of_pos_right hκ]
  linarith

/-- And the dependence really is logarithmic: for a threshold at or above `v/κ`, raising it by a
factor `r` delays the apparent lag by at least `(log r − log(3/2))/κ`. -/
theorem lag_shift_ge {v κ Mc r : ℝ} (hv : 0 < v) (hκ : 0 < κ) (hMc : v / κ ≤ Mc) (hr : 1 ≤ r) :
    (Real.log r - Real.log (3/2)) / κ ≤ lagTime v κ (r * Mc) - lagTime v κ Mc := by
  have hr0 : (0:ℝ) < r := lt_of_lt_of_le zero_lt_one hr
  set x : ℝ := κ * Mc / v with hxdef
  have hx1 : 1 ≤ x := by
    rw [hxdef, le_div_iff₀ hv]
    calc (1:ℝ) * v = v := one_mul v
      _ = κ * (v / κ) := by field_simp
      _ ≤ κ * Mc := by exact mul_le_mul_of_nonneg_left hMc hκ.le
  have hx0 : 0 < x := lt_of_lt_of_le zero_lt_one hx1
  -- lower bound on `arsinh (r x)`
  have hlow : Real.log (2 * (r * x)) ≤ Real.arsinh (r * x) := by
    have hpos : 0 < 2 * (r * x) := by positivity
    have hle : 2 * (r * x) ≤ r * x + Real.sqrt (1 + (r * x) ^ 2) := by
      have : r * x ≤ Real.sqrt (1 + (r * x) ^ 2) := by
        have h1 : Real.sqrt ((r * x) ^ 2) ≤ Real.sqrt (1 + (r * x) ^ 2) :=
          Real.sqrt_le_sqrt (by nlinarith)
        rwa [Real.sqrt_sq (by positivity)] at h1
      linarith
    simpa [Real.arsinh] using Real.log_le_log hpos hle
  -- upper bound on `arsinh x`
  have hup : Real.arsinh x ≤ Real.log (3 * x) := by
    have hle : x + Real.sqrt (1 + x ^ 2) ≤ 3 * x := by
      have h1 : Real.sqrt (1 + x ^ 2) ≤ 1 + x := by
        have h2 : Real.sqrt (1 + x ^ 2) ≤ Real.sqrt ((1 + x) ^ 2) :=
          Real.sqrt_le_sqrt (by nlinarith)
        rwa [Real.sqrt_sq (by linarith)] at h2
      linarith
    have hpos : 0 < x + Real.sqrt (1 + x ^ 2) := by
      have : 0 < Real.sqrt (1 + x ^ 2) := Real.sqrt_pos.mpr (by positivity)
      linarith
    simpa [Real.arsinh] using Real.log_le_log hpos hle
  have hdiff : Real.log r - Real.log (3/2) ≤ Real.arsinh (r * x) - Real.arsinh x := by
    have h1 : Real.log (2 * (r * x)) = Real.log 2 + Real.log r + Real.log x := by
      rw [Real.log_mul (by norm_num) (by positivity), Real.log_mul (ne_of_gt hr0) (ne_of_gt hx0)]
      ring
    have h2 : Real.log (3 * x) = Real.log 3 + Real.log x := by
      rw [Real.log_mul (by norm_num) (ne_of_gt hx0)]
    have h3 : Real.log (3/2) = Real.log 3 - Real.log 2 := by
      rw [Real.log_div (by norm_num) (by norm_num)]
    rw [h1] at hlow
    rw [h2] at hup
    rw [h3]
    linarith
  have harg : κ * (r * Mc) / v = r * x := by rw [hxdef]; ring
  simp only [lagTime, harg, ← hxdef]
  rw [div_sub_div_same, div_le_div_iff_of_pos_right hκ]
  linarith

/-! ### The lag time alone determines nothing -/

/-- **Every growth rate fits.**  For any growth rate `κ`, any threshold and any observed lag time
there is a nucleation flux reproducing that lag time exactly. -/
theorem lag_underdetermined {κ Mc T : ℝ} (hκ : 0 < κ) (hMc : 0 < Mc) (hT : 0 < T) :
    ∃ v : ℝ, 0 < v ∧ v = κ * Mc / Real.sinh (κ * T) ∧ lagTime v κ Mc = T := by
  have hs : 0 < Real.sinh (κ * T) := Real.sinh_pos_iff.mpr (by positivity)
  refine ⟨κ * Mc / Real.sinh (κ * T), by positivity, rfl, ?_⟩
  have harg : κ * Mc / (κ * Mc / Real.sinh (κ * T)) = Real.sinh (κ * T) := by
    field_simp
  simp only [lagTime, harg, Real.arsinh_sinh]
  field_simp

/-- **Two fits, one lag time.**  With unit threshold and unit lag time, the growth rates `κ = 1`
and `κ = 2` both fit, with strictly different nucleation fluxes. -/
theorem two_models_one_lag :
    lagTime (1 / Real.sinh 1) 1 1 = 1 ∧ lagTime (2 / Real.sinh 2) 2 1 = 1 ∧
      2 / Real.sinh 2 < 1 / Real.sinh 1 := by
  have hs1 : 0 < Real.sinh 1 := Real.sinh_pos_iff.mpr one_pos
  have hs2 : Real.sinh 2 = 2 * Real.sinh 1 * Real.cosh 1 := by
    have := Real.sinh_two_mul 1
    norm_num at this
    linarith
  have hc1 : 1 < Real.cosh 1 := Real.one_lt_cosh.mpr one_ne_zero
  have hs2pos : 0 < Real.sinh 2 := by rw [hs2]; positivity
  refine ⟨?_, ?_, ?_⟩
  · have harg : (1:ℝ) * 1 / (1 / Real.sinh 1) = Real.sinh 1 := by field_simp
    simp only [lagTime, harg, Real.arsinh_sinh]
    norm_num
  · have harg : (2:ℝ) * 1 / (2 / Real.sinh 2) = Real.sinh 2 := by field_simp
    have h2 : Real.sinh 2 = Real.sinh (2 * 1) := by norm_num
    simp only [lagTime, harg]
    rw [h2, Real.arsinh_sinh]
    norm_num
  · rw [div_lt_div_iff₀ hs2pos hs1, hs2]
    nlinarith

end Agg
