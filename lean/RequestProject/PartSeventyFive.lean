/-
# Part LXXV  The exact critical point of a condensate of disordered chains

Part VII.1 shows that demixing is exactly non-convexity of the free-energy density, and settles
the symmetric case.  `RequestProject.FloryHuggins` carries the chain length `N` through, which is
what makes biological condensates form at low concentration.

`IDR.flory_huggins_laws` bundles five statements about
`f(phi) = (phi/N) log phi + (1-phi) log(1-phi) + chi phi (1-phi)`.

1. *The curvature.*  `f'' = 1/(N phi) + 1/(1-phi) - 2 chi`, and the excess over
   `2 chiC N = (1 + sqrt N)^2/N` is the perfect square
   `((1 + sqrt N) phi - 1)^2 / (N phi (1-phi))`.
2. *The critical point is exact.*  Hence `f''` is minimised at `phiC N = 1/(1 + sqrt N)`, where
   it equals `2(chiC N - chi)`.
3. *Below `chiC N`, stability.*  `f` is convex on `[0,1]`, so no composition demixes.
4. *Above `chiC N`, demixing.*  `f` is strictly concave near `phiC N`, and two explicit
   coexisting compositions in equal proportion beat the homogeneous state.  So `chiC N` is the
   critical coupling exactly, not merely a bound.
5. *The chain-length law.*  `chiC` is strictly decreasing in `N`, stays above `1/2`, tends to
   `1/2`, and the critical composition `phiC N` tends to `0`.  For `N = 1` the density and the
   threshold reduce to those of Part VII.1.

The statement a model of a disordered region must therefore carry is not "`chi`" but
"`chi` at length `N`": the same interaction strength is subcritical for a short region and
supercritical for a long one, and the critical concentration of a long region is small precisely
because it scales as `1/sqrt N`.
-/
import Mathlib
import RequestProject.FloryHuggins

set_option autoImplicit false

namespace IDR

open Set IDR.Phase IDR.FH

/-- **The Flory--Huggins laws with chain length.**

1. the curvature identity with its perfect square;
2. the critical curvature at `phiC N`;
3. convexity, hence stability, below `chiC N`;
4. demixing above `chiC N`;
5. `chiC` strictly decreasing in `N`, bounded below by `1/2`, with `chiC 1 = 2` and
   `fh 1 = floryFE`. -/
theorem flory_huggins_laws :
    (∀ (N chi : ℝ), 0 < N → ∀ phi ∈ Ioo (0 : ℝ) 1,
        HasDerivAt (fh' N chi) (curvature N chi phi) phi ∧
        1 / (N * phi) + 1 / (1 - phi) - 2 * chiC N
          = ((1 + Real.sqrt N) * phi - 1) ^ 2 / (N * phi * (1 - phi))) ∧
    (∀ N : ℝ, 0 < N → phiC N ∈ Ioo (0 : ℝ) 1 ∧
        1 / (N * phiC N) + 1 / (1 - phiC N) = 2 * chiC N ∧
        ∀ phi ∈ Ioo (0 : ℝ) 1, 2 * chiC N ≤ 1 / (N * phi) + 1 / (1 - phi)) ∧
    (∀ (N chi : ℝ), 0 < N → chi ≤ chiC N →
        ConvexOn ℝ (Icc (0 : ℝ) 1) (fh N chi) ∧
        ∀ c, ¬ PhaseSeparates (Icc (0 : ℝ) 1) (fh N chi) c) ∧
    (∀ (N chi : ℝ), 0 < N → chiC N < chi → ∃ c, PhaseSeparates (Icc (0 : ℝ) 1) (fh N chi) c) ∧
    ((∀ N M : ℝ, 0 < N → N < M → chiC M < chiC N) ∧
      (∀ N : ℝ, 0 < N → 1 / 2 < chiC N) ∧
      Filter.Tendsto chiC Filter.atTop (nhds (1 / 2)) ∧
      Filter.Tendsto phiC Filter.atTop (nhds 0) ∧
      chiC 1 = 2 ∧ ∀ chi : ℝ, fh 1 chi = floryFE chi) := by
  refine ⟨fun N chi hN phi hphi =>
      ⟨fh_hasDerivAt2 hN.ne' chi hphi, curvature_identity hN hphi.1 hphi.2⟩,
    fun N hN => ⟨phiC_mem hN, curvature_at_phiC hN,
      fun phi hphi => curvature_min hN hphi.1 hphi.2⟩,
    fun N chi hN hchi => ⟨fh_convexOn hN hchi, fun c => no_demixing_below_chiC hN hchi c⟩,
    fun N chi hN hchi => fh_demixes_above_chiC hN hchi,
    fun N M hN hNM => chiC_strictAnti hN hNM,
    fun N hN => chiC_gt_half hN,
    chiC_tendsto_half, phiC_tendsto_zero, chiC_one, fh_one⟩

end IDR
