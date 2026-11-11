/-
# Parts XVIII--XIX capstone: the region in a living cell is not at equilibrium

Every equilibrium statement in the earlier parts -- Boltzmann populations, the variational
free energy, exponential tilting, linkage -- presupposes detailed balance.  A cell does not
supply it.  These two parts prove, in the smallest system that can show it, what the drive
costs a model.

`nonequilibrium_design_laws` bundles three clauses.

1. **A driven region still has a steady state, and time averages still report it.**  The
   three-state cycle `Driven.cycleP a b` is a legitimate kinetics (`cycleP_stochastic`) with
   the uniform stationary distribution (`cycle_stationary_unif`), so the whole observation
   theory of Part XV applies: a stationary time average equals the ensemble average
   (`Trajectory.timeAvg_stationary`).  Nothing about the *measurement* theory breaks.
2. **But it is the reversible equilibrium of no energy function.**  The steady state carries
   a current `(a − b)/3` around the cycle (`cycle_current`), and by Kolmogorov's criterion no
   strictly positive distribution whatsoever satisfies detailed balance with the driven
   kernel (`no_detailed_balance_of_driven`).  So a landscape model -- Boltzmann weights, free
   energy, tilting -- has no target to fit.
3. **The drive has a price, and the populations do not reveal it.**  The entropy production
   rate is nonnegative for every strictly positive kinetics (`epRate_nonneg`) and vanishes
   *exactly* at detailed balance (`epRate_eq_zero_iff_detailedBalance`), so a landscape model
   is committed to predicting zero dissipation.  For the driven cycle it equals
   `(a − b) log (a/b) > 0` (`cycle_epRate`, `cycle_epRate_pos`).  And two kinetics with the
   *same* stationary populations can have zero and positive dissipation respectively
   (`dissipation_not_determined_by_populations`): the energy budget is a separate parameter,
   not a functional of the reported ensemble.

The design consequence: for a disordered region in a living cell the object to be modelled is
a kinetics together with its dissipation, not a landscape.
-/
import Mathlib
import RequestProject.Driven
import RequestProject.EntropyProduction

namespace IDR

/-- **The non-equilibrium design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in Parts XVIII--XIX. -/
theorem nonequilibrium_design_laws :
    -- (1) a driven region has a steady state, and stationary time averages report it
    (∀ a b : ℝ, 0 ≤ a → 0 ≤ b → a + b ≤ 1 →
        Kinetics.IsStochastic (Driven.cycleP a b) ∧
        Kinetics.Stationary (Driven.cycleP a b) Driven.unif3 ∧
        ∀ f : Fin 3 → ℝ, ∀ T : ℕ, 0 < T →
          Trajectory.timeAvg (Driven.cycleP a b) Driven.unif3 f T
            = ∑ j, Driven.unif3 j * f j) ∧
    -- (2) but it is the reversible equilibrium of no energy function
    (∀ a b : ℝ, a ≠ b →
        (Driven.unif3 0 * Driven.cycleP a b 0 1
          - Driven.unif3 1 * Driven.cycleP a b 1 0 = (a - b) / 3) ∧
        ¬ ∃ pi : Fin 3 → ℝ, (∀ i, 0 < pi i) ∧
            Kinetics.DetailedBalance (Driven.cycleP a b) pi) ∧
    -- (3) the drive costs dissipation, and the populations do not determine it
    ((∀ (m : ℕ) (P : Fin m → Fin m → ℝ) (pi : Fin m → ℝ),
        (∀ i j, 0 < P i j) → (∀ i, 0 < pi i) → 0 ≤ EntropyProduction.epRate P pi) ∧
      (∀ (m : ℕ) (P : Fin m → Fin m → ℝ) (pi : Fin m → ℝ),
        (∀ i j, 0 < P i j) → (∀ i, 0 < pi i) →
          (EntropyProduction.epRate P pi = 0 ↔ Kinetics.DetailedBalance P pi)) ∧
      (∀ a b : ℝ, 0 < a → 0 < b →
        EntropyProduction.epRate (Driven.cycleP a b) Driven.unif3
          = (a - b) * (Real.log a - Real.log b)) ∧
      (∃ P Q : Fin 3 → Fin 3 → ℝ,
        Kinetics.IsStochastic P ∧ Kinetics.IsStochastic Q ∧
        Kinetics.Stationary P Driven.unif3 ∧ Kinetics.Stationary Q Driven.unif3 ∧
        EntropyProduction.epRate P Driven.unif3 = 0 ∧
        0 < EntropyProduction.epRate Q Driven.unif3)) := by
  refine ⟨fun a b ha hb hab =>
      ⟨Driven.cycleP_stochastic ha hb hab, Driven.cycle_stationary_unif a b,
        fun f T hT => Trajectory.timeAvg_stationary (Driven.cycle_stationary_unif a b) f hT⟩,
    fun a b hne => ⟨(Driven.cycle_current a b).1, Driven.no_detailed_balance_of_driven hne⟩,
    ?_⟩
  exact ⟨fun _ _ _ hP hpi => EntropyProduction.epRate_nonneg hP hpi,
    fun _ _ _ hP hpi => EntropyProduction.epRate_eq_zero_iff_detailedBalance hP hpi,
    fun _ _ ha hb => EntropyProduction.cycle_epRate ha hb,
    EntropyProduction.dissipation_not_determined_by_populations⟩

end IDR
