/-
# Part XLVI  Enthalpy--entropy compensation is a property of the fit

`RequestProject.Compensation` analyses the van 't Hoff fit that turns temperature-dependent
binding constants of a disordered region into an enthalpy and an entropy, and shows that the
celebrated correlation between the two is, for the standard least-squares fit, an identity whose
slope is fixed by the temperatures at which the experiment was done.

`IDR.compensation_laws` bundles five statements:

1. the fitted intercept is the mean response minus the slope times the mean regressor -- the
   entropy is an extrapolation of the data out of the experimental window;
2. **compensation**: two data sets with the same mean log-constant have fitted intercepts
   differing by exactly `−x̄` times the difference of their slopes, so the `(ΔH, ΔS)` points lie
   on an exact straight line whatever the molecules are;
3. the slope of that line is the harmonic mean of the experimental temperatures, a property of
   the design and of nothing else;
4. the deviation from the line is exactly the difference in mean log-constant -- the only part
   of a compensation plot that carries information about the molecules;
5. read as error propagation, the same identity says the fitted entropy inherits the error of
   the fitted enthalpy scaled by `|x̄|`; and an explicit two-point instance shows all of this is
   non-vacuous.
-/
import Mathlib
import RequestProject.Compensation

set_option autoImplicit false

namespace IDR

open IDR.Compensation

/-- **The compensation laws for van 't Hoff fits.**

1. *The intercept is an extrapolation*: `intercept = ȳ − slope · x̄`.
2. *Compensation is an identity*: equal mean log-constant forces
   `Δintercept = −x̄ · Δslope`, i.e. an exact line in the `(ΔH, ΔS)` plane.
3. *Its slope is the harmonic mean temperature* `n / Σ_i T_i⁻¹`.
4. *The residual is the information*: in general
   `Δintercept + x̄ · Δslope = Δ(mean log-constant)`.
5. *Error propagation*: `|Δintercept| = |x̄| · |Δslope|`; and the two-point example
   `x = (1,2)`, `y = (0,0)` versus `y' = (1,−1)` realises the identity with both differences
   nonzero. -/
theorem compensation_laws :
    (∀ (n : ℕ) (x y : Fin n → ℝ), intercept x y = mean y - slope x y * mean x) ∧
    (∀ (n : ℕ) (x y y' : Fin n → ℝ), mean y = mean y' →
        intercept x y - intercept x y' = -(mean x) * (slope x y - slope x y')) ∧
    (∀ (n : ℕ) (T : Fin n → ℝ),
        compensationTemperature (fun i => (T i)⁻¹) = (n : ℝ) / ∑ i, (T i)⁻¹) ∧
    (∀ (n : ℕ) (x y y' : Fin n → ℝ),
        (intercept x y - intercept x y') + mean x * (slope x y - slope x y')
          = mean y - mean y') ∧
    ((∀ (n : ℕ) (x y y' : Fin n → ℝ), mean y = mean y' →
        |intercept x y - intercept x y'| = |mean x| * |slope x y - slope x y'|) ∧
      mean ![(0 : ℝ), 0] = mean ![(1 : ℝ), -1] ∧
      slope ![(1 : ℝ), 2] ![0, 0] = 0 ∧
      slope ![(1 : ℝ), 2] ![1, -1] = -2 ∧
      intercept ![(1 : ℝ), 2] ![0, 0] = 0 ∧
      intercept ![(1 : ℝ), 2] ![1, -1] = 3 ∧
      intercept ![(1 : ℝ), 2] ![0, 0] - intercept ![(1 : ℝ), 2] ![1, -1]
        = -(mean ![(1 : ℝ), 2]) *
            (slope ![(1 : ℝ), 2] ![0, 0] - slope ![(1 : ℝ), 2] ![1, -1])) := by
  exact ⟨fun n x y => intercept_eq_mean_sub x y,
    fun n x y y' h => compensation_of_equal_mean x y y' h,
    fun n T => compensation_temperature_eq_harmonic_mean T,
    fun n x y y' => deviation_from_compensation x y y',
    ⟨fun n x y y' h => compensation_amplifies_error x y y' h, compensation_example⟩⟩

end IDR
