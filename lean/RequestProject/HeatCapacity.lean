/-
# Part LV.1  Heat capacity: a fitted van 't Hoff enthalpy is an enthalpy at one temperature

Part XLVI proves that the correlation between fitted enthalpies and entropies is an identity of
the least-squares fit, and states plainly that it treats the *linear* van 't Hoff fit only:
nonlinear fits with a heat-capacity term "have their own error structure and are not covered".
For a disordered region that omission is not a detail.  Coupled folding and binding buries
apolar surface, and burial of apolar surface is what makes `ΔCp` large and negative; a
disorder-to-order transition is the standard example of a strongly curved van 't Hoff plot.
This file covers the missing case exactly.

The model is the standard constant-heat-capacity one, with a reference temperature `T₀`:

  `ΔH(T) = ΔH₀ + ΔCp (T - T₀)`,  `ΔS(T) = ΔS₀ + ΔCp log(T/T₀)`,
  `log K(T) = ΔS(T)/R - ΔH(T)/(R T)`.

* `vantHoff` -- the two-point van 't Hoff enthalpy read off a window `[T₁, T₂]`: minus `R` times
  the secant slope of `log K` against `1/T`.
* `vantHoff_eq_enthalpy_at` -- **the exact result.**  That fitted enthalpy is the *true*
  enthalpy, evaluated at one specific interior temperature:

  `ΔH_vH = ΔH(M)`,  `M = T₁T₂ log(T₁/T₂) / (T₁ - T₂)`.

  No approximation and no small-`ΔCp` expansion: an identity.
* `logMeanRecip_mem_Ioo` -- and `M` lies strictly between `T₁` and `T₂`, so the fitted enthalpy
  belongs to a temperature *inside* the window, in general neither endpoint.
* `vantHoff_ne_endpoints` -- consequently, when `ΔCp ≠ 0`, the fitted number is not the enthalpy
  at either temperature of the window.  Quoting "the van 't Hoff enthalpy" without the
  temperature it belongs to is not a rounding error.
* `secondDiff_eq` -- **curvature is exactly `ΔCp`.**  Sampling `log K` at three temperatures
  equally spaced in `1/T`, the second difference is `(ΔCp/R) log((T₁+T₂)²/(4T₁T₂))`: it
  vanishes identically when `ΔCp = 0` (`secondDiff_eq_zero_of_dCp_zero`) and is nonzero, with
  the sign of `ΔCp`, whenever `ΔCp ≠ 0` and the window is nondegenerate
  (`secondDiff_ne_zero`, by the strict arithmetic-geometric mean inequality).

So `ΔCp` is identifiable from `log K` alone -- and completely invisible to the linear fit whose
algebra Part XLVI analysed.  A model of a disordered region that reports a binding enthalpy owes
the temperature at which it holds and the curvature it was fitted against.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace HeatCapacity

open Real

/-- The enthalpy at temperature `T` for a constant heat capacity `dCp`. -/
noncomputable def enthalpy (dH0 dCp T0 T : ℝ) : ℝ := dH0 + dCp * (T - T0)

/-- The entropy at temperature `T` for a constant heat capacity `dCp`. -/
noncomputable def entropy (dS0 dCp T0 T : ℝ) : ℝ := dS0 + dCp * Real.log (T / T0)

/-- The equilibrium constant, in logarithmic form. -/
noncomputable def lnK (R dH0 dS0 dCp T0 T : ℝ) : ℝ :=
  entropy dS0 dCp T0 T / R - enthalpy dH0 dCp T0 T / (R * T)

/-- The two-point van 't Hoff enthalpy read off the window `[T₁, T₂]`. -/
noncomputable def vantHoff (R T1 T2 y1 y2 : ℝ) : ℝ := -R * (y2 - y1) / (1 / T2 - 1 / T1)

/-- The interior temperature at which the van 't Hoff enthalpy is the true enthalpy. -/
noncomputable def logMeanRecip (T1 T2 : ℝ) : ℝ := T1 * T2 * Real.log (T1 / T2) / (T1 - T2)

/-- **The van 't Hoff enthalpy of a window is the true enthalpy at one interior temperature.** -/
theorem vantHoff_eq_enthalpy_at {R dH0 dS0 dCp T0 T1 T2 : ℝ} (hR : R ≠ 0) (hT1 : 0 < T1)
    (hT2 : 0 < T2) (hT0 : 0 < T0) (hne : T1 ≠ T2) :
    vantHoff R T1 T2 (lnK R dH0 dS0 dCp T0 T1) (lnK R dH0 dS0 dCp T0 T2)
      = enthalpy dH0 dCp T0 (logMeanRecip T1 T2) := by
  have hd : T1 - T2 ≠ 0 := sub_ne_zero.mpr hne
  have hlog1 : Real.log (T1 / T0) = Real.log T1 - Real.log T0 :=
    Real.log_div (ne_of_gt hT1) (ne_of_gt hT0)
  have hlog2 : Real.log (T2 / T0) = Real.log T2 - Real.log T0 :=
    Real.log_div (ne_of_gt hT2) (ne_of_gt hT0)
  have hlog12 : Real.log (T1 / T2) = Real.log T1 - Real.log T2 :=
    Real.log_div (ne_of_gt hT1) (ne_of_gt hT2)
  simp only [vantHoff, lnK, enthalpy, entropy, logMeanRecip, hlog1, hlog2, hlog12]
  have h1 : T1 ≠ 0 := ne_of_gt hT1
  have h2 : T2 ≠ 0 := ne_of_gt hT2
  field_simp
  ring

/-- **The logarithmic-mean temperature lies strictly inside the window.** -/
theorem logMeanRecip_mem_Ioo {T1 T2 : ℝ} (hT1 : 0 < T1) (hT2 : 0 < T2) (hlt : T1 < T2) :
    T1 < logMeanRecip T1 T2 ∧ logMeanRecip T1 T2 < T2 := by
  have hd : T1 - T2 < 0 := by linarith
  have hratio : 0 < T1 / T2 := by positivity
  have hne1 : T1 / T2 ≠ 1 := by
    intro h
    rw [div_eq_one_iff_eq (ne_of_gt hT2)] at h
    linarith
  have hne2 : T2 / T1 ≠ 1 := by
    intro h
    rw [div_eq_one_iff_eq (ne_of_gt hT1)] at h
    linarith
  have hA : Real.log (T1 / T2) < T1 / T2 - 1 := Real.log_lt_sub_one_of_pos hratio hne1
  have hB : Real.log (T2 / T1) < T2 / T1 - 1 :=
    Real.log_lt_sub_one_of_pos (by positivity) hne2
  have hBB : Real.log (T2 / T1) = -Real.log (T1 / T2) := by
    rw [← Real.log_inv]
    congr 1
    field_simp
  have hA' : T1 / T2 - 1 = (T1 - T2) / T2 := by field_simp
  have hB' : T2 / T1 - 1 = (T2 - T1) / T1 := by field_simp
  rw [hA'] at hA
  rw [hBB, hB'] at hB
  have h1 : T2 * Real.log (T1 / T2) < T1 - T2 := by
    have := (lt_div_iff₀ hT2).mp hA
    linarith [this]
  have h2 : T1 - T2 < T1 * Real.log (T1 / T2) := by
    have := (lt_div_iff₀ hT1).mp hB
    nlinarith [this]
  constructor
  · rw [logMeanRecip, lt_div_iff_of_neg hd]
    nlinarith
  · rw [logMeanRecip, div_lt_iff_of_neg hd]
    nlinarith

/-- **The fitted enthalpy is not the enthalpy at either end of the window.** -/
theorem vantHoff_ne_endpoints {R dH0 dS0 dCp T0 T1 T2 : ℝ} (hR : R ≠ 0) (hT1 : 0 < T1)
    (hT2 : 0 < T2) (hT0 : 0 < T0) (hlt : T1 < T2) (hCp : dCp ≠ 0) :
    vantHoff R T1 T2 (lnK R dH0 dS0 dCp T0 T1) (lnK R dH0 dS0 dCp T0 T2)
        ≠ enthalpy dH0 dCp T0 T1 ∧
      vantHoff R T1 T2 (lnK R dH0 dS0 dCp T0 T1) (lnK R dH0 dS0 dCp T0 T2)
        ≠ enthalpy dH0 dCp T0 T2 := by
  have hne : T1 ≠ T2 := ne_of_lt hlt
  obtain ⟨hM1, hM2⟩ := logMeanRecip_mem_Ioo hT1 hT2 hlt
  rw [vantHoff_eq_enthalpy_at hR hT1 hT2 hT0 hne]
  constructor
  · intro h
    have : dCp * (logMeanRecip T1 T2 - T1) = 0 := by
      simp only [enthalpy] at h; linarith
    rcases mul_eq_zero.mp this with h1 | h2
    · exact hCp h1
    · linarith
  · intro h
    have : dCp * (logMeanRecip T1 T2 - T2) = 0 := by
      simp only [enthalpy] at h; linarith
    rcases mul_eq_zero.mp this with h1 | h2
    · exact hCp h1
    · linarith

/-- The temperature whose reciprocal is the midpoint of the reciprocals of `T1` and `T2`. -/
noncomputable def harmonicMid (T1 T2 : ℝ) : ℝ := 2 * T1 * T2 / (T1 + T2)

/-- The second difference of `log K` over three temperatures equally spaced in `1/T`. -/
noncomputable def secondDiff (R dH0 dS0 dCp T0 T1 T2 : ℝ) : ℝ :=
  lnK R dH0 dS0 dCp T0 T1 + lnK R dH0 dS0 dCp T0 T2
    - 2 * lnK R dH0 dS0 dCp T0 (harmonicMid T1 T2)

/-- **The curvature of a van 't Hoff plot is exactly the heat capacity.** -/
theorem secondDiff_eq {R dH0 dS0 dCp T0 T1 T2 : ℝ} (hR : R ≠ 0) (hT1 : 0 < T1) (hT2 : 0 < T2)
    (hT0 : 0 < T0) :
    secondDiff R dH0 dS0 dCp T0 T1 T2 = dCp / R * Real.log ((T1 + T2) ^ 2 / (4 * T1 * T2)) := by
  have hsum : 0 < T1 + T2 := by linarith
  have hmid : harmonicMid T1 T2 = 2 * T1 * T2 / (T1 + T2) := rfl
  have hmidpos : 0 < harmonicMid T1 T2 := by rw [hmid]; positivity
  have l1 : Real.log (T1 / T0) = Real.log T1 - Real.log T0 :=
    Real.log_div (ne_of_gt hT1) (ne_of_gt hT0)
  have l2 : Real.log (T2 / T0) = Real.log T2 - Real.log T0 :=
    Real.log_div (ne_of_gt hT2) (ne_of_gt hT0)
  have l3 : Real.log (harmonicMid T1 T2 / T0)
      = Real.log 2 + Real.log T1 + Real.log T2 - Real.log (T1 + T2) - Real.log T0 := by
    rw [Real.log_div (ne_of_gt hmidpos) (ne_of_gt hT0), hmid,
      Real.log_div (by positivity) (ne_of_gt hsum), Real.log_mul (by positivity) (ne_of_gt hT2),
      Real.log_mul (by norm_num) (ne_of_gt hT1)]
  have l4 : Real.log ((T1 + T2) ^ 2 / (4 * T1 * T2))
      = 2 * Real.log (T1 + T2) - 2 * Real.log 2 - Real.log T1 - Real.log T2 := by
    rw [Real.log_div (by positivity) (by positivity), Real.log_pow,
      Real.log_mul (by positivity) (ne_of_gt hT2), Real.log_mul (by norm_num) (ne_of_gt hT1),
      show (4:ℝ) = 2 ^ 2 by norm_num, Real.log_pow]
    push_cast
    ring
  simp only [secondDiff, lnK, entropy, enthalpy, l1, l2, l3, l4]
  rw [hmid]
  field_simp
  ring

/-- **With no heat capacity the van 't Hoff plot is exactly straight.** -/
theorem secondDiff_eq_zero_of_dCp_zero {R dH0 dS0 T0 T1 T2 : ℝ} (hR : R ≠ 0) (hT1 : 0 < T1)
    (hT2 : 0 < T2) (hT0 : 0 < T0) : secondDiff R dH0 dS0 0 T0 T1 T2 = 0 := by
  rw [secondDiff_eq hR hT1 hT2 hT0]
  simp

/-- **With a heat capacity it is exactly not straight**, so `ΔCp` is identifiable from the
equilibrium constant alone -- and invisible to a linear fit. -/
theorem secondDiff_ne_zero {R dH0 dS0 dCp T0 T1 T2 : ℝ} (hR : R ≠ 0) (hT1 : 0 < T1)
    (hT2 : 0 < T2) (hT0 : 0 < T0) (hne : T1 ≠ T2) (hCp : dCp ≠ 0) :
    secondDiff R dH0 dS0 dCp T0 T1 T2 ≠ 0 := by
  rw [secondDiff_eq hR hT1 hT2 hT0]
  have hgt : 1 < (T1 + T2) ^ 2 / (4 * T1 * T2) := by
    rw [lt_div_iff₀ (by positivity)]
    have : 0 < (T1 - T2) ^ 2 := by
      have : T1 - T2 ≠ 0 := sub_ne_zero.mpr hne
      positivity
    nlinarith
  have hlog : 0 < Real.log ((T1 + T2) ^ 2 / (4 * T1 * T2)) := Real.log_pos hgt
  have hdiv : dCp / R ≠ 0 := div_ne_zero hCp hR
  exact mul_ne_zero hdiv (ne_of_gt hlog)

end HeatCapacity

end IDR
