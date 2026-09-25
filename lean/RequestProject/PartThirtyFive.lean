/-
# Part XXXV  Titration curves: the cooperativity is fitted, not measured

`RequestProject.Denaturant` treats the last derived quantity routinely quoted for a disordered
region: the `m`-value of a chemical denaturation curve, read as the cooperativity of a
transition that, for a region with no folded state, is not a transition at all.

The two-state linear-extrapolation curve is the logistic `frac dG m RT x = sigmoid((m x − ΔG)/RT)`.
It crosses `1/2` at `x = ΔG/m` with slope exactly `m/(4RT)`, so the fitted `m` *is* four `RT`
times the measured midpoint slope; every signal crossing the midpoint with a positive slope is
matched there by a two-state model; and the match is not only first order — the logistic curve
never departs from its own midpoint tangent by more than `|u|³/48` in the reduced variable
`u = m(x − x½)/RT`.  A strictly linear, non-cooperative expansion is therefore reproduced by a
two-state fit to cubic accuracy, and the only qualitative difference between the two is
saturation, which lives at the ends of the titration where the baselines are fitted.

`IDR.titration_laws` bundles the five statements.
-/
import Mathlib
import RequestProject.Denaturant

set_option autoImplicit false

namespace IDR

/-- **The design laws of a denaturation curve.**

1. *The midpoint*: the two-state curve passes through `1/2` at `x = ΔG/m`.
2. *The `m`-value is the slope*: its slope there is exactly `m/(4RT)`, so a measured midpoint
   slope `s` determines and is determined by `m = 4RT·s`.
3. *Any transition is matched*: for every midpoint and every positive slope there is a two-state
   model agreeing there in value and slope.
4. *To cubic order*: the two-state curve stays within `(4s|x − x₀|)³/48` of the straight line of
   slope `s` through the midpoint, so a gradual expansion is fitted by a two-state model to cubic
   accuracy.
5. *Only saturation distinguishes them*: the two-state curve is confined to `(0,1)`. -/
theorem titration_laws :
    -- 1  the midpoint
    (∀ dG m RT : ℝ, m ≠ 0 → Denat.frac dG m RT (dG / m) = 1 / 2) ∧
    -- 2  the m-value is four RT times the midpoint slope
    (∀ dG m RT : ℝ, m ≠ 0 → RT ≠ 0 →
        HasDerivAt (Denat.frac dG m RT) (m / (4 * RT)) (dG / m)) ∧
    (∀ dG m RT s : ℝ, m ≠ 0 → RT ≠ 0 →
        HasDerivAt (Denat.frac dG m RT) s (dG / m) → m = 4 * RT * s) ∧
    -- 3  every transition with a positive midpoint slope is matched by a two-state fit
    (∀ RT s x0 : ℝ, 0 < RT → 0 < s →
        ∃ dG m : ℝ, m = 4 * RT * s ∧ dG = m * x0 ∧ Denat.frac dG m RT x0 = 1 / 2 ∧
          HasDerivAt (Denat.frac dG m RT) s x0) ∧
    -- 4  and matched to cubic order away from the midpoint
    (∀ RT s x0 x : ℝ, 0 < RT → 0 < s →
        |Denat.frac ((4 * RT * s) * x0) (4 * RT * s) RT x - (1 / 2 + s * (x - x0))|
          ≤ |4 * s * (x - x0)| ^ 3 / 48) ∧
    -- 5  the two-state curve is confined to `(0,1)`
    (∀ dG m RT x : ℝ, Denat.frac dG m RT x ∈ Set.Ioo (0 : ℝ) 1) := by
  refine ⟨fun dG m RT hm => Denat.frac_midpoint hm,
    fun dG m RT hm hRT => Denat.frac_deriv_midpoint hm hRT,
    fun dG m RT s hm hRT h => Denat.m_eq_four_RT_slope hm hRT h,
    fun RT s x0 hRT hs => Denat.fit_matches_any_curve hRT hs,
    fun RT s x0 x hRT hs => Denat.frac_close_to_linear hRT hs,
    fun dG m RT x => Denat.frac_mem_Ioo dG m RT x⟩

end IDR
