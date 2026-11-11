/-
# Part XLIII  The material state: what the rheology of a condensate excludes

`RequestProject.Rheology` treats the three measurements that decide what kind of material a
condensate of disordered regions is -- the relaxation modulus, the probe mean-square
displacement, and their dependence on the age of the droplet -- and proves what each of them
rules out about a model.

`IDR.material_state_laws` bundles six statements:

1. a finite Maxwell spectrum decays exponentially at the rate of its slowest mode;
2. hence it eventually falls strictly below every power law: finitely many relaxation modes
   cannot reproduce power-law rheology at long times;
3. a finite spectrum always has a terminal viscosity, and it is exactly `Σ_k g_k tau_k`;
4. a power law with exponent at most one has none, so "the condensate has a viscosity" is a
   qualitatively different claim, not an approximation;
5. uncorrelated stationary steps give an exactly linear mean-square displacement, so a measured
   sublinear one forces net anticorrelation between steps and, in particular, memory;
6. and a modulus that depends on the waiting time forces any single stationary predicted curve
   to be wrong by at least half the ageing difference on one of the two waiting times.
-/
import Mathlib
import RequestProject.Rheology

set_option autoImplicit false

namespace IDR

open MeasureTheory IDR.Rheology

/-- **The material-state laws for a condensate of disordered regions.**

1. *Exponential decay.*  A finite Maxwell spectrum with all relaxation times at most `T`
   satisfies `G(t) ≤ G(0)·e^{-t/T}` for `t ≥ 0`.
2. *No power law.*  For every `C > 0` and exponent `alpha` there is a time beyond which the
   model lies strictly below `C·t^{-alpha}`.
3. *Terminal viscosity exists* for a finite spectrum, and equals `Σ_k g_k tau_k`.
4. *It does not exist for a power law* with `alpha ≤ 1`.
5. *Linear diffusion from memoryless steps.*  Uncorrelated stationary steps give
   `MSD(m) = m·σ²` exactly; a strictly smaller measured value forces the off-diagonal
   covariances to sum to a negative number and some pair of steps to be correlated -- in
   particular for a power-law `MSD(m) = A·m^alpha` with `alpha < 1`, at every `m ≥ 2`.
6. *Ageing.*  A waiting-time-independent prediction is off by at least half the difference
   between the two measured curves on one of them. -/
theorem material_state_laws :
    (∀ (n : ℕ) (g tau : Fin n → ℝ) (T : ℝ), (∀ k, 0 ≤ g k) → (∀ k, 0 < tau k) →
        (∀ k, tau k ≤ T) → ∀ t : ℝ, 0 ≤ t →
        maxwell g tau t ≤ (∑ k, g k) * Real.exp (-(t / T))) ∧
    (∀ (n : ℕ) (g tau : Fin n → ℝ) (T : ℝ), (∀ k, 0 ≤ g k) → (∀ k, 0 < tau k) →
        (∀ k, tau k ≤ T) → 0 < T → ∀ C alpha : ℝ, 0 < C →
        ∃ t₀ : ℝ, 0 < t₀ ∧ ∀ t ≥ t₀, maxwell g tau t < C * t ^ (-alpha)) ∧
    (∀ (n : ℕ) (g tau : Fin n → ℝ), (∀ k, 0 < tau k) →
        IntegrableOn (maxwell g tau) (Set.Ioi 0) ∧
        ∫ t in Set.Ioi (0 : ℝ), maxwell g tau t = ∑ k, g k * tau k) ∧
    (∀ C alpha : ℝ, C ≠ 0 → alpha ≤ 1 →
        ¬ IntegrableOn (fun t : ℝ => C * t ^ (-alpha)) (Set.Ioi 1)) ∧
    (∀ (C : ℕ → ℕ → ℝ) (s2 : ℝ), (∀ i, C i i = s2) →
        ((∀ i j, i ≠ j → C i j = 0) → ∀ m : ℕ, msd C m = m * s2) ∧
        (∀ m : ℕ, msd C m < m * s2 →
          (∑ i ∈ Finset.range m, ∑ j ∈ Finset.range m, (if i = j then 0 else C i j)) < 0 ∧
          ∃ i j, i ≠ j ∧ C i j ≠ 0)) ∧
    (∀ (C : ℕ → ℕ → ℝ) (A alpha : ℝ), 0 < A → alpha < 1 → (∀ i, C i i = A) →
        (∀ m : ℕ, msd C m = A * (m : ℝ) ^ alpha) → ∀ m : ℕ, 2 ≤ m →
        ∃ i j, i ≠ j ∧ C i j ≠ 0) ∧
    (∀ (G : ℝ → ℝ → ℝ) (M : ℝ → ℝ) (tw₁ tw₂ t : ℝ),
        |G tw₁ t - G tw₂ t| / 2 ≤ max |M t - G tw₁ t| (|M t - G tw₂ t|)) := by
  refine ⟨fun n g tau T hg hpos hT t ht => maxwell_le_exp hg hpos hT ht,
    fun n g tau T hg hpos hT hTpos C alpha hC => maxwell_lt_power_law hg hpos hT hTpos hC,
    fun n g tau hpos => ⟨maxwell_integrableOn hpos, maxwell_viscosity hpos⟩,
    fun C alpha hC halpha => power_law_not_integrableOn hC halpha,
    fun C s2 hdiag => ⟨fun hoff m => msd_of_uncorrelated hdiag hoff m,
      fun m hm => ⟨subdiffusion_forces_anticorrelation hdiag m hm,
        subdiffusion_forces_memory hdiag m hm⟩⟩,
    fun C A alpha hA halpha hdiag hmsd m hm =>
      power_law_msd_forces_memory hA halpha hdiag hmsd hm,
    fun G M tw₁ tw₂ t => aging_forces_error G M tw₁ tw₂ t⟩

end IDR
