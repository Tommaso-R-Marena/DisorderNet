/-
# Part XLVI.1  Enthalpy--entropy compensation is a property of the fit

Coupled folding and binding of a disordered region is characterised thermodynamically by a van
't Hoff or calorimetric decomposition of the affinity into an enthalpy and an entropy, and the
literature on such decompositions is dominated by one observation: across variants, ligands and
conditions, the fitted `ΔH` and `ΔS` are strongly correlated, so that large enthalpic gains are
almost cancelled by entropic losses.  This is routinely read as a mechanism.  This file shows
that, for the standard fit, the correlation is an identity of least squares with a slope fixed
by the experimental design, and says exactly what a genuine mechanism would have to look like
instead.

The setting is the van 't Hoff fit: at inverse temperatures `x i = 1/T i` the measured
log-constants are `y i`, and the ordinary least-squares line `y = intercept + slope · x` gives
`ΔH = −R·slope` and `ΔS = R·intercept`.

* `intercept_eq_mean_sub` -- the identity everything follows from: the fitted intercept is
  `ȳ − x̄·slope`.  The entropy is an extrapolation of the data to infinite temperature.
* `compensation_of_equal_mean` -- **the compensation identity.**  Two data sets with the same
  mean log-constant have fitted intercepts differing by exactly `−x̄` times the difference of
  their slopes: the fitted `(ΔH, ΔS)` pairs lie on a straight line whose slope is the *harmonic
  mean of the experimental temperatures* (`compensation_temperature`,
  `compensation_temperature_eq_harmonic_mean`), no matter what the molecules are.
* `deviation_from_compensation` -- and the general case: the deviation from that line is exactly
  the difference in mean log-constant, i.e. in mean affinity over the temperature window.  This
  is the only part of a measured compensation plot that carries thermodynamic information.
* `compensation_amplifies_error` -- the same identity read as error propagation: an error in the
  fitted enthalpy appears in the fitted entropy multiplied by `x̄`, so `ΔS` is never better
  determined than `ΔH` divided by the harmonic mean temperature.
* `compensation_example` -- a two-point instance in which both differences are nonzero, so the
  identity is not vacuous.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset

namespace Compensation

variable {n : ℕ}

/-- Sample mean. -/
noncomputable def mean (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n

/-- Total variation of the regressor. -/
noncomputable def sxx (x : Fin n → ℝ) : ℝ := ∑ i, (x i - mean x) ^ 2

/-- Covariation of regressor and response. -/
noncomputable def sxy (x y : Fin n → ℝ) : ℝ := ∑ i, (x i - mean x) * (y i - mean y)

/-- Ordinary least-squares slope of `y` on `x`. -/
noncomputable def slope (x y : Fin n → ℝ) : ℝ := sxy x y / sxx x

/-- Ordinary least-squares intercept of `y` on `x`. -/
noncomputable def intercept (x y : Fin n → ℝ) : ℝ := mean y - slope x y * mean x

/-- **The fitted intercept is an extrapolation.**  It is the mean response minus the slope times
the mean regressor: in the van 't Hoff fit, the entropy is the log-constant extrapolated from the
experimental window to infinite temperature. -/
theorem intercept_eq_mean_sub (x y : Fin n → ℝ) :
    intercept x y = mean y - slope x y * mean x := rfl

/-! ## The compensation identity -/

/-- **Compensation.**  Two data sets with the same mean log-constant have fitted intercepts
differing by exactly `−x̄` times the difference of their fitted slopes.  In thermodynamic
variables: the fitted entropies differ by `x̄` times the difference of the fitted enthalpies, so
the `(ΔH, ΔS)` points of any such family lie exactly on a line -- whatever the molecules are and
whether or not anything is compensating. -/
theorem compensation_of_equal_mean (x y y' : Fin n → ℝ) (hmean : mean y = mean y') :
    intercept x y - intercept x y' = -(mean x) * (slope x y - slope x y') := by
  unfold intercept
  rw [hmean]
  ring

/-- **The general case.**  Without the equal-mean assumption, the deviation from the compensation
line is exactly the difference in mean log-constant over the temperature window: that difference,
and nothing else in the plot, is thermodynamic information about the molecules. -/
theorem deviation_from_compensation (x y y' : Fin n → ℝ) :
    (intercept x y - intercept x y') + mean x * (slope x y - slope x y')
      = mean y - mean y' := by
  unfold intercept
  ring

/-- The compensation temperature of an experimental design: the reciprocal of the mean inverse
temperature. -/
noncomputable def compensationTemperature (x : Fin n → ℝ) : ℝ := (mean x)⁻¹

/-- **The compensation temperature is the harmonic mean of the experimental temperatures.**  With
`x i = 1/T i`, the slope of the compensation line is `n / Σ_i (1/T i)`, a property of the
temperatures at which the experiment was done and of nothing else. -/
theorem compensation_temperature_eq_harmonic_mean (T : Fin n → ℝ) :
    compensationTemperature (fun i => (T i)⁻¹) = (n : ℝ) / ∑ i, (T i)⁻¹ := by
  unfold compensationTemperature mean
  rw [inv_div]

/-- **Compensation in thermodynamic variables.**  Writing `dH = −slope` and `dS = intercept` (in
units of the gas constant), two data sets with equal mean log-constant satisfy
`dH − dH' = T_hm · (dS − dS')` with `T_hm` the harmonic mean temperature. -/
theorem compensation_thermo (x y y' : Fin n → ℝ) (hmean : mean y = mean y')
    (hx : mean x ≠ 0) :
    (-(slope x y) - -(slope x y'))
      = compensationTemperature x * (intercept x y - intercept x y') := by
  rw [compensation_of_equal_mean x y y' hmean, compensationTemperature]
  field_simp
  ring

/-- **Compensation as error propagation.**  The same identity says that an error in the fitted
slope reappears in the fitted intercept multiplied by `|x̄|`: the entropy is determined no better
than the enthalpy divided by the harmonic mean temperature. -/
theorem compensation_amplifies_error (x y y' : Fin n → ℝ) (hmean : mean y = mean y') :
    |intercept x y - intercept x y'| = |mean x| * |slope x y - slope x y'| := by
  rw [compensation_of_equal_mean x y y' hmean, abs_mul, abs_neg]

/-! ## The identity is not vacuous -/

/-- Two two-point data sets at inverse temperatures `1` and `2`, with the same mean response and
different slopes: the intercepts differ by `−x̄` times the difference of slopes, with both
differences nonzero. -/
theorem compensation_example :
    mean ![(0 : ℝ), 0] = mean ![(1 : ℝ), -1] ∧
    slope ![(1 : ℝ), 2] ![0, 0] = 0 ∧
    slope ![(1 : ℝ), 2] ![1, -1] = -2 ∧
    intercept ![(1 : ℝ), 2] ![0, 0] = 0 ∧
    intercept ![(1 : ℝ), 2] ![1, -1] = 3 ∧
    intercept ![(1 : ℝ), 2] ![0, 0] - intercept ![(1 : ℝ), 2] ![1, -1]
      = -(mean ![(1 : ℝ), 2]) *
          (slope ![(1 : ℝ), 2] ![0, 0] - slope ![(1 : ℝ), 2] ![1, -1]) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [mean, slope, intercept, sxx, sxy, Fin.sum_univ_two]

end Compensation

end IDR
