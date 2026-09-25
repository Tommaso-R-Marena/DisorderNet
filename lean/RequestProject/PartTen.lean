/-
# Part X capstone: the solution-state design laws

Part IX supplied the physics of the chain: its statistics, its electrostatics, its
temperature dependence, its dynamics and its excluded volume.  Part X supplies the physics
of the *measured solution state* -- the three further classes of observable on which
published disordered-region ensembles actually stand, and the one piece of structural
physics (transient, cooperative secondary structure) that distinguishes a disordered region
from a random coil.

`solution_state_design_laws` bundles three clauses.

1. **Orientational observables are quadratic, hence genuinely ensemble observables.**  The
   generalised order parameter of a bond vector satisfies `0 ≤ S² ≤ 1`
   (`NMR.orderParam_nonneg`, `NMR.orderParam_le_one`) with `S² = 1` exactly for a rigid
   direction (`NMR.orderParam_eq_one_of_aligned`); a measured `S² < 1` *proves* that two
   populated conformers have non-parallel bond vectors
   (`NMR.orientational_disorder_of_orderParam_lt_one`).  And residual dipolar couplings
   average with sign: an explicit ensemble has `D = 0` while no member does
   (`NMR.rdc_cancellation`).  A model must forward-model `S²` and `D` from the distribution;
   it may not fit a structure to them.
2. **Diffusion measures a harmonic mean and is blind to sequence.**  The Kirkwood radius is
   invariant under relabelling the chain (`Hydro.kirkwoodSum_perm`), the Stokes--Einstein
   coefficient of the apparent radius is *exactly* the weight average of the conformers'
   coefficients (`Hydro.stokesEinstein_ensemble`), and therefore the radius that is reported
   never exceeds -- and generically falls strictly below -- the mean radius of the ensemble
   (`Hydro.appRadius_le_mean`, `Hydro.appRadius_lt_mean_two`).  Size restraints from
   diffusion data bias a model compact by exactly the amount of heterogeneity being modelled.
3. **Secondary structure is fractional and cooperative.**  The configuration sum of the
   helix--coil chain obeys the transfer-matrix identity (`HelixCoil.Zhead_succ`), the helical
   population of a residue is strictly inside `(0,1)` at every finite propensity and
   temperature (`HelixCoil.helixFraction_pos`, `HelixCoil.helixFraction_lt_one`), and with a
   positive coupling neighbouring residues are strictly positively correlated
   (`HelixCoil.helix_cooperativity`).  So the output of the model must be residue-level
   *populations with couplings*, not a structure and not independent per-residue propensities.

With `IDR.physical_realism_design_laws` (Part IX) this completes the physical specification:
the object to be predicted is a temperature-, salt- and sequence-dependent conformational
distribution, whose predictions for every standard experiment -- SAXS, PRE/NOE, FRET, NMR
relaxation and RDCs, translational diffusion, calorimetry -- are the exact nonlinear
functionals proved in Parts IX and X.
-/
import Mathlib
import RequestProject.NMR
import RequestProject.Hydrodynamics
import RequestProject.HelixCoil

namespace IDR

/-- **The solution-state design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in Part X. -/
theorem solution_state_design_laws :
    -- (1) orientational observables: bounded, saturating only for rigidity, sign-averaging
    ((∀ (m : ℕ) (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ), (∀ k, 0 ≤ w k) → (∑ k, w k) = 1 →
        (∀ k, NMR.IsBondVector (u k)) →
        0 ≤ NMR.orderParam w u ∧ NMR.orderParam w u ≤ 1) ∧
      (∀ (m : ℕ) (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ), (∑ k, w k) = 1 →
        (∀ k l, NMR.dot (u k) (u l) ^ 2 = 1) → NMR.orderParam w u = 1) ∧
      (∀ (m : ℕ) (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ), (∀ k, 0 ≤ w k) → (∑ k, w k) = 1 →
        (∀ k, NMR.IsBondVector (u k)) → NMR.orderParam w u < 1 →
        ∃ k l, 0 < w k ∧ 0 < w l ∧ NMR.dot (u k) (u l) ^ 2 < 1) ∧
      (∀ Dmax : ℝ, NMR.rdc Dmax NMR.cancelWeights NMR.axis (NMR.axis 0) = 0 ∧
        Dmax * ((3 * NMR.dot (NMR.axis 0) (NMR.axis 0) ^ 2 - 1) / 2) = Dmax ∧
        Dmax * ((3 * NMR.dot (NMR.axis 1) (NMR.axis 0) ^ 2 - 1) / 2) = -(Dmax / 2))) ∧
    -- (2) hydrodynamics: sequence-blind, and a harmonic mean of the conformer radii
    ((∀ (N : ℕ) (r : Fin N → EuclideanSpace ℝ (Fin 3)) (sig : Equiv.Perm (Fin N)),
        Hydro.kirkwoodSum (r ∘ sig) = Hydro.kirkwoodSum r) ∧
      (∀ (m : ℕ) (kT eta : ℝ), 0 < eta → ∀ w R : Fin m → ℝ,
        Hydro.stokesEinstein kT eta (Hydro.appRadius w R)
          = ∑ k, w k * Hydro.stokesEinstein kT eta (R k)) ∧
      (∀ (m : ℕ) (w R : Fin m → ℝ), (∀ k, 0 ≤ w k) → (∑ k, w k) = 1 → (∀ k, 0 < R k) →
        Hydro.appRadius w R ≤ ∑ k, w k * R k) ∧
      (∀ R1 R2 : ℝ, 0 < R1 → 0 < R2 → R1 ≠ R2 →
        Hydro.appRadius ![1 / 2, 1 / 2] ![R1, R2]
          < ∑ k, (![1 / 2, 1 / 2] : Fin 2 → ℝ) k * ![R1, R2] k)) ∧
    -- (3) secondary structure: solvable, fractional, and cooperative
    ((∀ (J h : ℝ) (n : ℕ) (b : Bool),
        HelixCoil.Zhead J h (n + 1) b
          = ∑ c : Bool, HelixCoil.transfer J h b c * HelixCoil.Zhead J h n c) ∧
      (∀ (J h : ℝ) (n : ℕ),
        0 < HelixCoil.helixFraction J h n ∧ HelixCoil.helixFraction J h n < 1) ∧
      (∀ J : ℝ, 0 < J →
        0 < HelixCoil.pairSpin J - HelixCoil.meanSpin J * HelixCoil.meanSpin J)) := by
  refine ⟨⟨fun m w u hw hsum hu =>
        ⟨NMR.orderParam_nonneg hsum hu, NMR.orderParam_le_one hw hsum hu⟩,
      fun m w u hsum halign => NMR.orderParam_eq_one_of_aligned hsum halign,
      fun m w u hw hsum hu hlt =>
        NMR.orientational_disorder_of_orderParam_lt_one hw hsum hu hlt,
      fun Dmax => NMR.rdc_cancellation Dmax⟩,
    ⟨fun N r sig => Hydro.kirkwoodSum_perm r sig,
      fun m kT eta heta w R => Hydro.stokesEinstein_ensemble heta w R,
      fun m w R hw hsum hR => Hydro.appRadius_le_mean hw hsum hR,
      fun R1 R2 h1 h2 hne => Hydro.appRadius_lt_mean_two h1 h2 hne⟩,
    ⟨fun J h n b => HelixCoil.Zhead_succ J h n b,
      fun J h n => ⟨HelixCoil.helixFraction_pos J h n, HelixCoil.helixFraction_lt_one J h n⟩,
      fun J hJ => HelixCoil.helix_cooperativity hJ⟩⟩

end IDR
