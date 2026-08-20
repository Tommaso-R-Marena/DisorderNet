/-
# Part XXXVIII  Density maps: occupancy and disorder are the same parameter

`RequestProject.Density` treats the operational definition of a disordered region — the part of
the chain with no interpretable density in a crystallographic or cryo-EM map — in the standard
harmonic treatment: occupancy `q`, isotropic displacement parameter `B`, structure-factor
contribution `q·exp(−Bs²/4)` and Gaussian peak height `q·(4π/B)^{3/2}`.

A single resolution shell determines nothing, since any occupancy can be matched by a suitable
`B`; two shells determine both in principle; but two models agreeing at a shell `s₀` differ at a
higher shell by at most `F₀·|B − B'|·(s² − s₀²)/4`, so separating occupancy from disorder needs a
span of resolution that a disordered region does not supply.  In real space the degeneracy is
exact: multiplying the occupancy by `c³` and the displacement parameter by `c²` leaves the peak
height unchanged, so a missing side chain and a fully occupied but mobile one are the same map.
Raising `B` from 25 Å² to 100 Å² divides the peak by eight; a peak at contour level `τ` forces
`B³τ² ≤ (4π)³q²`, and any `B` beyond that bound is invisible.  Absence of density is a bound on
`q²/B³`, not the absence of a residue.

`IDR.density_laws` bundles the seven statements.
-/
import Mathlib
import RequestProject.Density

set_option autoImplicit false

namespace IDR

/-- **The design laws of a density map.**

1. *One shell determines nothing*: any occupancy is matched at a single resolution shell by a
   suitable displacement parameter.
2. *Two shells determine both*: occupancy and displacement parameter are identifiable from two
   distinct shells.
3. *But only with a span of resolution*: models agreeing at `s₀` differ at `s ≥ s₀` by at most
   `F₀·|B − B'|·(s² − s₀²)/4`.
4. *The occupancy–`B` degeneracy is exact*: `peak (c³q) (c²B) = peak q B`.
5. *Disorder costs peak height*: the peak is decreasing in `B`, and `B = 100` gives one eighth of
   the peak at `B = 25`.
6. *Visibility is a bound on `q²/B³`*: a peak at contour `τ` forces `B³τ² ≤ (4π)³q²`.
7. *And its converse*: any `B` with `(4π)³q² < B³τ²` produces no peak at that contour. -/
theorem density_laws :
    -- 1  one resolution shell determines nothing
    (∀ q q' s₀ : ℝ, 0 < q → 0 < q' → s₀ ≠ 0 → ∀ B : ℝ,
        ∃ B' : ℝ, B' = B + 4 * Real.log (q' / q) / s₀ ^ 2 ∧
          Dens.formFactor q' B' s₀ = Dens.formFactor q B s₀) ∧
    -- 2  two shells determine both
    (∀ q q' B B' s₁ s₂ : ℝ, 0 < q → 0 < q' → s₁ ^ 2 ≠ s₂ ^ 2 →
        Dens.formFactor q B s₁ = Dens.formFactor q' B' s₁ →
        Dens.formFactor q B s₂ = Dens.formFactor q' B' s₂ → q = q' ∧ B = B') ∧
    -- 3  separating them needs a span of resolution
    (∀ q q' B B' s₀ s : ℝ, 0 ≤ B → 0 ≤ B' → s₀ ^ 2 ≤ s ^ 2 → 0 < q →
        Dens.formFactor q B s₀ = Dens.formFactor q' B' s₀ →
        |Dens.formFactor q B s - Dens.formFactor q' B' s|
          ≤ Dens.formFactor q B s₀ * (|B - B'| * (s ^ 2 - s₀ ^ 2) / 4)) ∧
    -- 4  the occupancy–B degeneracy, exactly
    (∀ q B c : ℝ, 0 < B → 0 < c → Dens.peak (c ^ 3 * q) (c ^ 2 * B) = Dens.peak q B) ∧
    -- 5  disorder costs peak height
    (∀ q B B' : ℝ, 0 ≤ q → 0 < B → B ≤ B' → Dens.peak q B' ≤ Dens.peak q B) ∧
    (∀ q : ℝ, Dens.peak q 100 = Dens.peak q 25 / 8) ∧
    -- 6  what visible density bounds
    (∀ q B tau : ℝ, 0 < q → 0 < B → 0 < tau → tau ≤ Dens.peak q B →
        B ^ 3 * tau ^ 2 ≤ (4 * Real.pi) ^ 3 * q ^ 2) ∧
    -- 7  and the converse
    (∀ q B tau : ℝ, 0 < q → 0 < B → 0 < tau →
        (4 * Real.pi) ^ 3 * q ^ 2 < B ^ 3 * tau ^ 2 → Dens.peak q B < tau) := by
  refine ⟨fun q q' s₀ hq hq' hs B => Dens.single_shell_degenerate hq hq' hs B,
    fun q q' B B' s₁ s₂ hq hq' hs h1 h2 => Dens.two_shells_identify hq hq' hs h1 h2,
    fun q q' B B' s₀ s hB hB' hs hq hagree => Dens.formFactor_close hB hB' hs hagree hq,
    fun q B c hB hc => Dens.peak_scale_invariant q hB hc,
    fun q B B' hq hB hBB => Dens.peak_antitone hq hB hBB,
    Dens.peak_hundred,
    fun q B tau hq hB htau hvis => Dens.visibility_bound hq hB htau hvis,
    fun q B tau hq hB htau hbig => Dens.invisible_of_large_B hq hB htau hbig⟩

end IDR
