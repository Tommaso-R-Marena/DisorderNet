/-
# Part XXXII  Dynamics from NMR: what a relaxation experiment reports

Parts I--XXXI fix what a model of a disordered region must *be*, what it may be *scored* by,
and how the structural restraints that constrain it are to be read.  The dynamics are
constrained by two families of NMR experiments, and this part prices both.

`RequestProject.SpinRelaxation` (`Relax`): the spectral density `J(ω) = Σ w_k·2τ_k/(1+(ωτ_k)²)`
is the only way motion enters `R₁`, `R₂` and the heteronuclear NOE.  `J(0)` is twice the *mean*
correlation time and so is dominated by any slow minority state; each Lorentzian is two-to-one
in `τ` (the exact `τ_c` ambiguity); two four-component motional models with strictly positive
populations agree at three frequencies while differing in `J(0)`; and the Lipari--Szabo
"model-free" form is exactly the two-component case, so `S²` is a fitted population.

`RequestProject.ChemicalExchange` (`Exchange`): under the standard fast-exchange
(Luz--Meiboom) forward model, a CPMG dispersion profile has total amplitude `Φ_ex/k_ex` with
`Φ_ex = p_A p_B Δω²`.  A measured amplitude *does* bound the minority population from below
once `Δω` is capped by the spectral range — but it never determines it, since every population
in `(0,1/2]` reproduces the profile with a suitable `Δω`; and at large `k_ex` a state of any
population and any shift contributes less than any given detection threshold.

`IDR.nmr_dynamics_laws` bundles the eight statements.
-/
import Mathlib
import RequestProject.SpinRelaxation
import RequestProject.ChemicalExchange

set_option autoImplicit false

namespace IDR

/-- **The design laws of NMR dynamics data for a disordered region.**

1. *What `J(0)` is*: twice the mean correlation time — so, by 2, a slow minority state
   dominates it.
2. *A minority slow state dominates*: `1%` of a state `1000×` slower than the bulk already
   raises `J(0)` above ten times the bulk value.
3. *The `τ_c` ambiguity is exact*: at one frequency the Lorentzian identifies the correlation
   time only up to `τ ↦ 1/(ω²τ)`.
4. *One field does not determine the motion*: two strictly positive four-component models agree
   at three frequencies and differ in `J(0)` by `77/240`.
5. *"Model-free" is a two-component fit*: the Lipari--Szabo form is exactly the two-Lorentzian
   case, with `S²` and `1 − S²` as populations.
6. *Exchange broadening is evidence*: with `|Δω| ≤ Δω_max`, a measured `Φ_ex` forces
   `p_B ≥ Φ_ex/Δω_max²`.
7. *But not a measurement of the population*: every `p_B ∈ (0,1/2]` reproduces a given
   dispersion profile at every cycle time, with a suitable `Δω`.
8. *And a state can be invisible*: for any population, any shift and any threshold there is an
   exchange rate at which the profile is flat to within the threshold. -/
theorem nmr_dynamics_laws :
    -- 1  the spectral density at zero frequency is twice the mean correlation time
    (∀ (m : ℕ) (w tau : Fin m → ℝ), Relax.specDens w tau 0 = 2 * ∑ k, w k * tau k) ∧
    -- 2  a minority slow state dominates it
    (∀ tau0 : ℝ, 0 < tau0 →
      10 * (2 * tau0) < Relax.specDens ![99/100, 1/100] ![tau0, 1000 * tau0] 0) ∧
    -- 3  the correlation time is identified only up to the `τ_c` reflection
    (∀ t1 t2 om : ℝ, 0 < t1 → 0 < t2 →
      (Relax.lorentz t1 om = Relax.lorentz t2 om ↔ t1 = t2 ∨ om ^ 2 * t1 * t2 = 1)) ∧
    -- 4  a one-field data set does not determine the distribution of correlation times
    ((∀ k, 0 < Relax.pW k) ∧ (∀ k, 0 < Relax.qW k) ∧
      ∑ k, Relax.pW k = 1 ∧ ∑ k, Relax.qW k = 1 ∧
      Relax.specDens Relax.pW Relax.tauW (1/2) = Relax.specDens Relax.qW Relax.tauW (1/2) ∧
      Relax.specDens Relax.pW Relax.tauW 1 = Relax.specDens Relax.qW Relax.tauW 1 ∧
      Relax.specDens Relax.pW Relax.tauW 2 = Relax.specDens Relax.qW Relax.tauW 2 ∧
      Relax.specDens Relax.qW Relax.tauW 0 - Relax.specDens Relax.pW Relax.tauW 0 = 77/240) ∧
    -- 5  the model-free form is the two-component case
    (∀ S2 taum taue om : ℝ, Relax.modelFree S2 taum taue om
      = Relax.specDens ![S2, 1 - S2] ![taum, taum * taue / (taum + taue)] om) ∧
    -- 6  exchange broadening bounds the minority population from below
    (∀ pB dw dwmax : ℝ, 0 ≤ pB → |dw| ≤ dwmax → 0 < dwmax →
      Exchange.phiEx pB dw / dwmax ^ 2 ≤ pB) ∧
    -- 7  but every population reproduces the profile
    (∀ R20 pB0 dw0 kex : ℝ, 0 < pB0 → pB0 < 1 → 0 < dw0 → ∀ pB : ℝ, 0 < pB → pB ≤ 1/2 →
      ∃ dw : ℝ, 0 < dw ∧ ∀ tcp,
        Exchange.R2eff R20 pB dw kex tcp = Exchange.R2eff R20 pB0 dw0 kex tcp) ∧
    -- 8  and a state of any population can be invisible
    (∀ pB dw eps : ℝ, 0 ≤ pB → pB ≤ 1 → 0 < eps →
      ∃ kex : ℝ, 0 < kex ∧ ∀ R20 tcp : ℝ, 0 < tcp →
        |Exchange.R2eff R20 pB dw kex tcp - R20| < eps) := by
  refine ⟨fun m w tau => Relax.specDens_zero w tau,
    fun tau0 h => Relax.minority_slow_state_dominates h,
    fun t1 t2 om h1 h2 => Relax.lorentz_eq_iff h1 h2,
    Relax.relaxation_underdetermined,
    fun S2 taum taue om => Relax.modelFree_eq_specDens S2 taum taue om,
    fun pB dw dwmax h0 hdw hmax => Exchange.population_lower_bound h0 hdw hmax,
    fun R20 pB0 dw0 kex h00 h01 hdw0 pB hp0 hp1 =>
      Exchange.invisible_state_population_unidentifiable h00 h01 hdw0 hp0 hp1,
    fun pB dw eps h0 h1 heps => Exchange.arbitrarily_populated_invisible_state h0 h1 heps⟩

end IDR
