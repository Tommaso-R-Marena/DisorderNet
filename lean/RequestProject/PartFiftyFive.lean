/-
# Part LV  Heat capacity: the curved van 't Hoff plot of a disorder-to-order transition

Part XLVI analysed the linear van 't Hoff fit and said explicitly that the heat-capacity term
was outside it.  `RequestProject.HeatCapacity` puts it inside, exactly: with a constant `ΔCp`
the equilibrium constant is computed in closed form and every statement below is an identity,
not an expansion.

`IDR.heat_capacity_laws` bundles four statements:

1. *The fitted enthalpy is a true enthalpy -- at one temperature.*  The two-point van 't Hoff
   enthalpy of the window `[T₁, T₂]` equals `ΔH(M)` with `M = T₁T₂ log(T₁/T₂)/(T₁-T₂)`.
2. *That temperature is interior*: `T₁ < M < T₂`.
3. *So it is not the enthalpy at either endpoint* once `ΔCp ≠ 0`.
4. *Curvature is the heat capacity*: the second difference of `log K` at three temperatures
   equally spaced in `1/T` is `(ΔCp/R) log((T₁+T₂)²/(4T₁T₂))` -- identically zero when
   `ΔCp = 0`, and nonzero with the sign of `ΔCp` otherwise.

For a disordered region this is the ordinary case, not a refinement: burial of apolar surface on
binding is exactly what makes `|ΔCp|` large, so the van 't Hoff plot of a coupled folding and
binding equilibrium is curved, and a single quoted `ΔH` is a statement about one temperature
inside the window that was measured.  With Part XLVI, the pair says what a reported
thermodynamic decomposition does and does not determine: the compensation is in the fit, the
curvature is in the physics, and neither is what the number alone conveys.
-/
import Mathlib
import RequestProject.HeatCapacity

set_option autoImplicit false

namespace IDR

open IDR.HeatCapacity

/-- **The heat-capacity laws.**

1. the van 't Hoff enthalpy of a window is the true enthalpy at the interior temperature `M`;
2. `M` is strictly inside the window;
3. hence it is not the enthalpy at either endpoint when `ΔCp ≠ 0`;
4. the curvature of the plot is exactly the heat capacity. -/
theorem heat_capacity_laws :
    (∀ R dH0 dS0 dCp T0 T1 T2 : ℝ, R ≠ 0 → 0 < T1 → 0 < T2 → 0 < T0 → T1 ≠ T2 →
        vantHoff R T1 T2 (lnK R dH0 dS0 dCp T0 T1) (lnK R dH0 dS0 dCp T0 T2)
          = enthalpy dH0 dCp T0 (logMeanRecip T1 T2)) ∧
    (∀ T1 T2 : ℝ, 0 < T1 → 0 < T2 → T1 < T2 →
        T1 < logMeanRecip T1 T2 ∧ logMeanRecip T1 T2 < T2) ∧
    (∀ R dH0 dS0 dCp T0 T1 T2 : ℝ, R ≠ 0 → 0 < T1 → 0 < T2 → 0 < T0 → T1 < T2 → dCp ≠ 0 →
        vantHoff R T1 T2 (lnK R dH0 dS0 dCp T0 T1) (lnK R dH0 dS0 dCp T0 T2)
            ≠ enthalpy dH0 dCp T0 T1 ∧
          vantHoff R T1 T2 (lnK R dH0 dS0 dCp T0 T1) (lnK R dH0 dS0 dCp T0 T2)
            ≠ enthalpy dH0 dCp T0 T2) ∧
    (∀ R dH0 dS0 dCp T0 T1 T2 : ℝ, R ≠ 0 → 0 < T1 → 0 < T2 → 0 < T0 →
        secondDiff R dH0 dS0 dCp T0 T1 T2 = dCp / R * Real.log ((T1 + T2) ^ 2 / (4 * T1 * T2)) ∧
          secondDiff R dH0 dS0 0 T0 T1 T2 = 0 ∧
          (T1 ≠ T2 → dCp ≠ 0 → secondDiff R dH0 dS0 dCp T0 T1 T2 ≠ 0)) :=
  ⟨fun _ _ _ _ _ _ _ hR hT1 hT2 hT0 hne => vantHoff_eq_enthalpy_at hR hT1 hT2 hT0 hne,
    fun _ _ hT1 hT2 hlt => logMeanRecip_mem_Ioo hT1 hT2 hlt,
    fun _ _ _ _ _ _ _ hR hT1 hT2 hT0 hlt hCp => vantHoff_ne_endpoints hR hT1 hT2 hT0 hlt hCp,
    fun _ _ _ _ _ _ _ hR hT1 hT2 hT0 =>
      ⟨secondDiff_eq hR hT1 hT2 hT0, secondDiff_eq_zero_of_dCp_zero hR hT1 hT2 hT0,
        fun hne hCp => secondDiff_ne_zero hR hT1 hT2 hT0 hne hCp⟩⟩

end IDR
