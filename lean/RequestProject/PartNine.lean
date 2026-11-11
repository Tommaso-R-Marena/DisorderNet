/-
# Part IX capstone: the physical design laws

Parts I--VIII are information-theoretic: they say what a model of an intrinsically disordered
region must *be* (a context-conditional distribution), how large it must be, how accurately
it can be fitted, and what the data cost.  None of them says anything about polypeptides in
water.  Part IX supplies the physics, in the form the earlier parts can use: exact chain
statistics, exact forward models of the three experiments that constrain disordered
ensembles, an explicit screened-electrostatic Hamiltonian, the temperature axis, and the
excluded-volume balance that fixes the size exponent.

`physical_realism_design_laws` bundles seven clauses.

1. **Chain statistics are not the random walk.**  The exact discrete worm-like-chain formula
   for `⟨R²⟩` at every `N` (`Polymer.msd_eq`), with the ideal chain and the rigid rod as its
   two limits, and the exact ideal-chain radius of gyration `b²(N²-1)/(6N)`.
2. **Small-angle scattering is a pair statistic and, in the Guinier regime, one number.**
   `Rg² = (1/2N²)Σ|r_i-r_j|²` exactly (`Polymer.gyration_eq_pair_sum`), the scattering curve
   is invariant under relabelling the chain (`Observables.debye_perm_invariant`), and it
   equals `1 - q²Rg²/3` up to `(5/96)(qD)³` (`Observables.guinier`).
3. **`r^{-6}` restraints are minority reports.**  The apparent PRE/NOE distance never exceeds
   the mean distance (`Observables.preApparent_le_mean`) and is squeezed below
   `w^{-1/6}·d` by any conformer of weight `w` at distance `d`
   (`Observables.preApparent_le_of_weight`).  Restraints must be applied by forward-modelling
   the observable, never by fitting to "experimental distances".
4. **Charge patterning is physical, and salt-tunable.**  Two sequences of identical
   composition have different charge decoration and different Debye--Hückel energies
   (`Electro.scd_blocky_lt_alternating`, `Electro.screenedEnergy_perm_not_invariant`), while
   at high salt every electrostatic energy is bounded by `(Σ|q|)²exp(-kappa b)/b`
   (`Electro.screenedEnergy_abs_le`).  A model must take the sequence *pattern* and the ionic
   strength as inputs.
5. **The ensemble is a function of temperature.**  The heat capacity is the energy
   fluctuation and is nonnegative (`Thermo.heatCapacity_eq_variance`,
   `Thermo.heatCapacity_nonneg`), it vanishes exactly for a rigid model
   (`Thermo.rigid_zero_heatCapacity`), and a positive `ΔCp` forces transition temperatures on
   both sides of the stability maximum -- cold denaturation
   (`Thermo.denaturation_two_temperatures`), never three
   (`Thermo.no_three_transition_temperatures`).
6. **Chain dynamics fixes the sampling cost.**  The Rouse modes diagonalise the free-end
   chain Laplacian exactly (`Rouse.rouseMode_eigen`, `Rouse.rouseMode_boundary_left`,
   `Rouse.rouseMode_boundary_right`), the slowest eigenvalue is at most `π²/N²`
   (`Rouse.rouse_slowest_eigenvalue_le`), and hence a run that reports populations to
   accuracy `eps` from a start `d` away needs at least `(1 - eps/d)N²/π²` steps
   (`Rouse.rouse_run_length`): connectivity alone makes equilibration cost grow
   quadratically with the length of the region, before any barrier.
7. **Excluded volume fixes the exponent.**  The Flory free energy has a unique strict
   minimiser `R* = (3v/2a)^{1/5} N^{3/5}` (`Flory.floryFreeEnergy_min`), so the swollen-coil
   exponent is `3/5` and the chain outgrows every ideal chain (`Flory.flory_swelling`).

Together with `IDR.model_must_be`, `IDR.model_cannot_be`, `IDR.quantitative_design_laws`,
`IDR.physical_design_laws`, `IDR.statistical_design_laws`, `IDR.precision_design_laws`,
`IDR.collective_design_laws` and `IDR.temporal_design_laws`, the specification is now a
physical one: the object to be predicted is a temperature- and salt-dependent, sequence-
pattern-dependent conformational distribution, and the data that constrain it are nonlinear
functionals of that distribution with the exact forms proved here.
-/
import Mathlib
import RequestProject.Polymer
import RequestProject.Observables
import RequestProject.Electrostatics
import RequestProject.Thermo
import RequestProject.Flory
import RequestProject.Rouse
import RequestProject.PartEight

namespace IDR

/-- **The physical design laws for a model of an intrinsically disordered region.**  Each
clause is an instance of a theorem proved in Part IX. -/
theorem physical_realism_design_laws :
    -- (1) exact chain statistics: worm-like chain, its two limits, and the ideal `Rg`
    ((∀ (a b : ℝ), a ≠ 1 → ∀ N : ℕ,
        Polymer.msd a b N
          = b ^ 2 * (N * (1 + a) / (1 - a) - 2 * a * (1 - a ^ N) / (1 - a) ^ 2)) ∧
      (∀ (b : ℝ) (N : ℕ), Polymer.msd 0 b N = N * b ^ 2 ∧
        Polymer.msd 1 b N = (N : ℝ) ^ 2 * b ^ 2) ∧
      (∀ (b : ℝ) (N : ℕ), 0 < N →
        (1 / (2 * (N : ℝ) ^ 2)) * ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N,
            (Nat.dist i j : ℝ) * b ^ 2
          = b ^ 2 * ((N : ℝ) ^ 2 - 1) / (6 * N))) ∧
    -- (2) scattering: a pair statistic, blind to relabelling, one number in the Guinier regime
    ((∀ (N : ℕ), 0 < N → ∀ r : Fin N → EuclideanSpace ℝ (Fin 3),
        Polymer.gyrationSq r
          = (1 / (2 * (N : ℝ) ^ 2)) * ∑ i, ∑ j, ‖r i - r j‖ ^ 2) ∧
      (∀ (N : ℕ) (q : ℝ) (r : Fin N → EuclideanSpace ℝ (Fin 3)) (s : Equiv.Perm (Fin N)),
        Observables.debye q (r ∘ s) = Observables.debye q r) ∧
      (∀ (N : ℕ), 0 < N → ∀ (r : Fin N → EuclideanSpace ℝ (Fin 3)) (q D : ℝ), 0 ≤ q →
        (∀ i j, ‖r i - r j‖ ≤ D) → q * D ≤ 1 →
        |Observables.debye q r - (1 - q ^ 2 * Polymer.gyrationSq r / 3)|
          ≤ 5 / 96 * (q * D) ^ 3)) ∧
    -- (3) `r^{-6}` averaging: the reading is a minority report, never the mean
    ((∀ (m : ℕ) (w d : Fin m → ℝ), (∀ i, 0 ≤ w i) → (∑ i, w i) = 1 → (∀ i, 0 < d i) →
        Observables.preApparent w d ≤ ∑ i, w i * d i) ∧
      (∀ (m : ℕ) (w d : Fin m → ℝ), (∀ i, 0 ≤ w i) → (∀ i, 0 < d i) → ∀ k : Fin m, 0 < w k →
        Observables.preApparent w d ≤ (w k) ^ (-(1 : ℝ) / 6) * d k)) ∧
    -- (4) charge patterning is physical, and washed out by salt
    (Electro.scd Electro.blocky < Electro.scd Electro.alternating ∧
      (∀ b : ℝ, 0 < b →
        Electro.screenedEnergy b 0 Electro.blocky ≠ Electro.screenedEnergy b 0
          Electro.alternating) ∧
      (∀ (N : ℕ) (b kappa : ℝ), 0 < b → 0 ≤ kappa → ∀ q : Fin N → ℝ,
        |Electro.screenedEnergy b kappa q|
          ≤ (∑ i, |q i|) ^ 2 * (Real.exp (-(kappa * b)) / b))) ∧
    -- (5) temperature: the heat capacity is the fluctuation, and cold denaturation is forced
    ((∀ (n : ℕ), 0 < n → ∀ (U : Fin n → ℝ) (beta : ℝ),
        Thermo.heatCapacity U beta = -beta ^ 2 * deriv (Thermo.meanE U) beta ∧
          0 ≤ Thermo.heatCapacity U beta) ∧
      (∀ (n : ℕ), 0 < n → ∀ (U : Fin n → ℝ) (c : ℝ), (∀ j, U j = c) → ∀ beta : ℝ,
        Thermo.heatCapacity U beta = 0) ∧
      (∀ dH0 dS0 dCp T0 : ℝ, 0 < T0 → ∀ T1 Tm T2 : ℝ, 0 < T1 → T1 < Tm → Tm < T2 →
        Thermo.gibbsHelmholtz dH0 dS0 dCp T0 T1 < 0 →
        0 < Thermo.gibbsHelmholtz dH0 dS0 dCp T0 Tm →
        Thermo.gibbsHelmholtz dH0 dS0 dCp T0 T2 < 0 →
        (∃ Tc ∈ Set.Ioo T1 Tm, Thermo.gibbsHelmholtz dH0 dS0 dCp T0 Tc = 0) ∧
          (∃ Th ∈ Set.Ioo Tm T2, Thermo.gibbsHelmholtz dH0 dS0 dCp T0 Th = 0)) ∧
      (∀ dH0 dS0 dCp T0 : ℝ, 0 < T0 → 0 < dCp → ∀ T1 T2 T3 : ℝ, 0 < T1 → T1 < T2 → T2 < T3 →
        Thermo.gibbsHelmholtz dH0 dS0 dCp T0 T1 = 0 →
        Thermo.gibbsHelmholtz dH0 dS0 dCp T0 T2 = 0 →
        Thermo.gibbsHelmholtz dH0 dS0 dCp T0 T3 = 0 → False)) ∧
    -- (6) chain dynamics: exact Rouse spectrum and the quadratic cost of equilibration
    ((∀ (N p : ℕ), 0 < N → ∀ s : ℝ,
        2 * Rouse.rouseMode N p s - Rouse.rouseMode N p (s - 1) - Rouse.rouseMode N p (s + 1)
          = Rouse.rouseEigen N p * Rouse.rouseMode N p s) ∧
      (∀ (N p : ℕ), 0 < N →
        Rouse.rouseMode N p (-1) = Rouse.rouseMode N p 0 ∧
          Rouse.rouseMode N p (N : ℝ) = Rouse.rouseMode N p ((N : ℝ) - 1)) ∧
      (∀ N : ℕ, 0 < N → ∀ d eps : ℝ, 0 < d → 0 ≤ 1 - Rouse.rouseEigen N 1 → ∀ t : ℕ,
        (1 - Rouse.rouseEigen N 1) ^ t * d ≤ eps →
        (1 - eps / d) * (N : ℝ) ^ 2 / Real.pi ^ 2 ≤ t)) ∧
    -- (7) excluded volume: a unique optimal size, with the exponent 3/5
    ((∀ a v N : ℝ, 0 < a → 0 < v → 0 < N → ∀ R : ℝ, 0 < R → R ≠ Flory.floryRadius a v N →
        Flory.floryFreeEnergy a v N (Flory.floryRadius a v N) < Flory.floryFreeEnergy a v N R) ∧
      (∀ a v N : ℝ,
        Flory.floryRadius a v N
          = (3 * v / (2 * a)) ^ ((1 : ℝ) / 5) * N ^ ((3 : ℝ) / 5))) := by
  refine ⟨⟨fun a b ha N => Polymer.msd_eq a b ha N,
      fun b N => ⟨Polymer.msd_ideal b N, Polymer.msd_rod b N⟩,
      fun b N hN => Polymer.ideal_gyration b hN⟩,
    ⟨fun N hN r => Polymer.gyration_eq_pair_sum hN r,
      fun N q r s => Observables.debye_perm_invariant q r s,
      fun N hN r q D hq hr hqD => Observables.guinier hN r hq hr hqD⟩,
    ⟨fun m w d hw hsum hd => Observables.preApparent_le_mean hw hsum hd,
      fun m w d hw hd k hk => Observables.preApparent_le_of_weight hw hd k hk⟩,
    ⟨Electro.scd_blocky_lt_alternating,
      fun b hb => Electro.screenedEnergy_perm_not_invariant hb,
      fun N b kappa hb hkappa q => Electro.screenedEnergy_abs_le hb hkappa q⟩,
    ⟨fun n hn U beta => ⟨Thermo.heatCapacity_eq_variance hn U beta,
        Thermo.heatCapacity_nonneg hn U beta⟩,
      fun n hn U c hconst beta => Thermo.rigid_zero_heatCapacity hn hconst beta,
      fun dH0 dS0 dCp T0 hT0 T1 Tm T2 h1 h1m hm2 hlow hmid hhigh =>
        Thermo.denaturation_two_temperatures hT0 h1 h1m hm2 hlow hmid hhigh,
      fun dH0 dS0 dCp T0 hT0 hCp T1 T2 T3 h1 h12 h23 z1 z2 z3 =>
        Thermo.no_three_transition_temperatures hT0 hCp h1 h12 h23 z1 z2 z3⟩,
    ⟨fun N p hN s => Rouse.rouseMode_eigen N p hN s,
      fun N p hN => ⟨Rouse.rouseMode_boundary_left N p, Rouse.rouseMode_boundary_right N p hN⟩,
      fun N hN d eps hd hlam t hrun => Rouse.rouse_run_length hN hd hlam t hrun⟩,
    ⟨fun a v N ha hv hN R hR hne => Flory.floryFreeEnergy_min ha hv hN hR hne,
      fun a v N => Flory.floryRadius_eq a v N⟩⟩

end IDR
