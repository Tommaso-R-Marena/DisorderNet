/-
# Part XV capstone: the design laws of the *situated* region

Parts I–XIV fix what a model of an intrinsically disordered region must be (a distribution),
what it must predict (nonlinear ensemble functionals), what data can pin it down, and by
what score it may be judged.  All of that still describes a region in a tube, at
equilibrium, watched for as long as one likes.  Part XV removes the last three
idealisations, and each removal changes the specification of the model.

`situated_design_laws` bundles three clauses.

1. **The functional ensemble is not the measured ensemble.**  A crowded background is an
   exponential tilt by the excluded volume (`Crowding.crowded`), so every structural average
   responds with `d⟨f⟩/dPi = -beta·Cov(f, v)` (`Crowding.hasDerivAt_crowdedMean`).  If larger
   conformations exclude more volume the mean size is *strictly* decreasing in the crowder
   pressure (`Crowding.crowding_strictly_compacts`), so the dilute ensemble a model is
   trained on is never the in-cell ensemble it is asked about
   (`Crowding.in_cell_ne_in_vitro`) -- while a region with a single excluded volume is
   untouched (`Crowding.rigid_region_ignores_crowding`), which is why the correction is
   specific to disorder.  And the correction is a new parameter: an explicit three-state
   region shows that no refitted temperature reproduces the crowded populations
   (`Crowding.crowding_is_not_a_temperature`).
2. **Mechanical data measure a fluctuation of one marginal.**  The slope of a
   force--extension curve is `beta` times the variance of the extension coordinate
   (`Force.stiffness_eq_beta_var`), so a single-structure model is exactly inextensible
   (`Force.rigid_is_inextensible`) while any two populated conformations of different
   extension force a strictly rising curve (`Force.extension_strictMono`); the two-state
   bond gives the closed form `b·tanh(beta·F·b)` (`Force.two_state_bond_extension`) with
   entropic stiffness `beta·b²` (`Force.two_state_stiffness_zero_force`).  The whole curve
   depends only on the law of the pulling coordinate
   (`Force.same_extension_law_same_force_curve`), so structurally different ensembles can
   share it exactly (`Force.force_curve_blind_to_structure`).
3. **A single trajectory identifies only the basin it visits.**  Time averaging is unbiased
   from equilibrium (`Trajectory.timeAvg_stationary`), but two kinetic models agreeing on a
   closed set of conformations give identical statistics for every observable and every
   window length (`Trajectory.timeAvg_eq_of_agree`), with an explicit pair whose equilibrium
   ensembles differ (`Trajectory.single_molecule_cannot_identify_the_ensemble`).  Once
   exchange is broken the equilibrium ensemble is not even a function of the kinetics
   (`Trajectory.reducible_stationary_not_unique`), so a model must carry the basin
   decomposition, or declare the between-basin weights unidentified.

Read with the earlier capstones, the specification is now: a *conditional, situated*
distribution -- conditioned on sequence, partner, temperature, salt, crowder pressure and
preparation -- forward-modelled through the exact nonlinear functionals of each experiment,
fitted with an honest (strictly proper) ensemble score, and reported with the identifiability
limits proved above.
-/
import Mathlib
import RequestProject.Crowding
import RequestProject.Force
import RequestProject.Trajectory

namespace IDR

/-- **The design laws of the situated region.**  Each clause is an instance of a theorem
proved in Part XV. -/
theorem situated_design_laws :
    -- (1) the cell: crowding tilts, compacts, spares rigid regions, and is not a temperature
    ((∀ (n : ℕ) (q v f : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ beta Pi : ℝ,
        HasDerivAt (Crowding.crowdedMean q v f beta)
          (-(beta * Response.cov q (fun j => -v j) f v (beta * Pi))) Pi) ∧
      (∀ (n : ℕ) (q v f : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → (∑ j, q j) = 1 →
        ∀ beta : ℝ, 0 < beta → Crowding.Comonotone f v →
        ∀ i₀ j₀ : Fin n, 0 < (f i₀ - f j₀) * (v i₀ - v j₀) →
        ∀ Pi : ℝ, 0 < Pi →
          Crowding.crowdedMean q v f beta Pi < ∑ j, q j * f j ∧
            Crowding.crowded q v beta Pi ≠ q) ∧
      (∀ (n : ℕ) (q v : Fin n → ℝ), (∑ j, q j) = 1 → ∀ c : ℝ, (∀ j, v j = c) →
        ∀ beta Pi : ℝ, Crowding.crowded q v beta Pi = q) ∧
      (∀ beta Pi : ℝ, 0 < beta → 0 < Pi → ∀ beta' : ℝ,
        Crowding.crowded (FreeEnergy.boltz 1 Crowding.ladderU) Crowding.ladderV beta Pi
          ≠ FreeEnergy.boltz beta' Crowding.ladderU)) ∧
    -- (2) the pulling curve: a fluctuation of one marginal, blind to everything else
    ((∀ (n : ℕ) (q x : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ beta F : ℝ,
        deriv (Force.extension q x beta) F = beta * Force.extVar q x beta F) ∧
      (∀ (n : ℕ) (q x : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ c : ℝ, (∀ j, x j = c) →
        ∀ beta F : ℝ, Force.extension q x beta F = c) ∧
      (∀ (n : ℕ) (q x : Fin n → ℝ), 0 < n → (∀ j, 0 < q j) → ∀ beta : ℝ, 0 < beta →
        ∀ j₁ j₂ : Fin n, x j₁ ≠ x j₂ → StrictMono (Force.extension q x beta)) ∧
      (∀ b beta F : ℝ,
        Force.extension Force.bondQ (Force.bondX b) beta F = b * Real.tanh (beta * F * b)) ∧
      (∀ b beta : ℝ,
        HasDerivAt (Force.extension Force.bondQ (Force.bondX b) beta) (beta * b ^ 2) 0) ∧
      (∀ (n m : ℕ) (q x : Fin n → ℝ) (q' x' : Fin m → ℝ),
        (∀ g : ℝ → ℝ, Force.extLaw q x g = Force.extLaw q' x' g) →
        ∀ beta F : ℝ, Force.extension q x beta F = Force.extension q' x' beta F)) ∧
    -- (3) the trajectory: unbiased at equilibrium, blind outside its basin
    ((∀ (n : ℕ) (P : Fin n → Fin n → ℝ) (pi : Fin n → ℝ), Kinetics.Stationary P pi →
        ∀ (f : Fin n → ℝ) (T : ℕ), 0 < T →
          Trajectory.timeAvg P pi f T = ∑ j, pi j * f j) ∧
      (∀ (n : ℕ) (P P' : Fin n → Fin n → ℝ) (S : Finset (Fin n)) (w : Fin n → ℝ),
        Trajectory.Closed P S → (∀ i ∈ S, ∀ j, P i j = P' i j) → Trajectory.SupportedOn w S →
        ∀ (f : Fin n → ℝ) (T : ℕ),
          Trajectory.timeAvg P w f T = Trajectory.timeAvg P' w f T) ∧
      ((∀ w : Fin 3 → ℝ, Trajectory.SupportedOn w Trajectory.basin →
          ∀ (f : Fin 3 → ℝ) (T : ℕ),
            Trajectory.timeAvg Trajectory.trapP w f T
              = Trajectory.timeAvg Trajectory.decayP w f T) ∧
        Kinetics.Stationary Trajectory.trapP Trajectory.deltaTwo ∧
          ¬ Kinetics.Stationary Trajectory.decayP Trajectory.deltaTwo) ∧
      (Kinetics.Stationary Trajectory.trapP Trajectory.deltaTwo ∧
        Kinetics.Stationary Trajectory.trapP Trajectory.basinEq ∧
          Trajectory.deltaTwo ≠ Trajectory.basinEq)) := by
  refine ⟨⟨fun n q v f hn hq beta Pi => Crowding.hasDerivAt_crowdedMean hn hq beta Pi,
      fun n q v f hn hq hq1 beta hbeta hc i₀ j₀ hsep Pi hPi =>
        Crowding.in_cell_ne_in_vitro hn hq hq1 hbeta hc hsep hPi,
      fun n q v hq1 c hconst beta Pi => Crowding.rigid_region_ignores_crowding hq1 hconst beta Pi,
      fun beta Pi hbeta hPi beta' => Crowding.crowding_is_not_a_temperature hbeta hPi beta'⟩,
    ⟨fun n q x hn hq beta F => Force.stiffness_eq_beta_var hn hq beta F,
      fun n q x hn hq c hconst beta F => Force.rigid_is_inextensible hn hq hconst beta F,
      fun n q x hn hq beta hbeta j₁ j₂ hne => Force.extension_strictMono hn hq hbeta hne,
      fun b beta F => Force.two_state_bond_extension b beta F,
      fun b beta => Force.two_state_stiffness_zero_force b beta,
      fun n m q x q' x' h beta F => Force.same_extension_law_same_force_curve h beta F⟩,
    ⟨fun n P pi h f T hT => Trajectory.timeAvg_stationary h f hT,
      fun n P P' S w hC hagree hw f T => Trajectory.timeAvg_eq_of_agree hC hagree hw f T,
      Trajectory.single_molecule_cannot_identify_the_ensemble,
      Trajectory.reducible_stationary_not_unique⟩⟩

end IDR
