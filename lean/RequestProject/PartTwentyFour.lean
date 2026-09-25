/-
# Part XXIV  The molecular model: continuous space, a real force field, real solvent,
# and the Boltzmann-Gibbs ensemble

Parts I-XXIII fix what a model of a disordered region must *report* and what it must not
claim, using the smallest carrier of each phenomenon (finite state spaces, lattices,
two-state populations).  This part removes those carriers and re-proves the load-bearing
statements in the setting in which a molecular model is actually built: atoms at real
coordinates in `R^3`, a class-I molecular-mechanics Hamiltonian with Lennard-Jones 12-6,
Coulomb-with-dielectric and harmonic bonded terms, implicit solvent (Generalized Born plus
solvent-accessible surface area), and the Boltzmann-Gibbs probability measure on
`R^(3N)` with respect to Lebesgue measure.

The developments are

* `RequestProject.Context`        -- the biophysical context: temperature, ionic strength,
                                     partner concentration, and the Debye length;
* `RequestProject.Potentials`     -- the pair potentials and the pair stability theorem;
* `RequestProject.Hamiltonian`    -- the Hamiltonian on `(R^3)^N`, its E(3) invariance, its
                                     lower bound, and its hard core;
* `RequestProject.Solvation`      -- Generalized Born and SASA, and the solvated
                                     Hamiltonian;
* `RequestProject.GibbsField`     -- the Gibbs measure, its Radon-Nikodym derivative, the
                                     partition-function-free ratio and score identities;
* `RequestProject.ContinuousScore`-- the continuous quadratic score and the exact price of
                                     non-equivariance;
* `RequestProject.Generative`     -- the push-forward operator and the existence of a
                                     generator.

Here the clauses are bundled into one statement, `IDR.molecular_design_laws`.

What is *not* claimed.  The Hamiltonian is a class-I additive force field with implicit
solvent; polarisability, explicit-solvent many-body potentials of mean force, protonation
equilibria (pH) and rigid-constraint (Fixman) corrections are outside it.  Each of those is
an additional model, not a missing lemma of this one.  What the results below do establish
is that the standard molecular model is well posed: the energy is bounded below on
non-degenerate configurations, so the Boltzmann weight is integrable on a container; the
resulting measure has a genuine density; that density's log-gradient is `-beta` times the
force; and everything is exactly invariant under rigid motions.
-/
import Mathlib
import RequestProject.Context
import RequestProject.Solvation
import RequestProject.ContinuousScore
import RequestProject.Generative

namespace IDR

open MeasureTheory MM Gibbs Solvation ContScore Generative Potentials

/-- **The design laws of a molecular model of a disordered region.**

1. *The context is a coordinate of the model, not of the experiment.*  Each of temperature,
   ionic strength and partner concentration changes a predicted observable on its own, and
   no constant predictor reproduces the temperature dependence.
2. *Stability.*  The solvated Hamiltonian -- bonded harmonic terms, Coulomb with a
   dielectric, Lennard-Jones 12-6, Generalized Born and a surface-area term -- is bounded
   below on configurations without exactly coincident atoms.  This is what makes the
   Boltzmann weight integrable and the ensemble exist.
3. *Excluded volume is a theorem, not a constraint.*  For any pair of atoms and any bound
   `M` there is a separation below which the energy exceeds `M`: the continuous force field
   reproduces by itself the hard core that lattice models had to postulate.  Moreover the
   set of configurations with two exactly coincident atoms is Lebesgue-null, so the
   singularity is invisible to the measure.
4. *Solvent is part of the energy.*  A charge is strictly stabilised by a more polarisable
   solvent (the Born term is strictly negative), an engulfed atom has zero accessible
   surface, and burying surface lowers the nonpolar term.
5. *Exact E(3) invariance.*  Every rigid motion -- rotation, reflection, translation --
   leaves the solvated Hamiltonian, and hence the Gibbs measure, exactly unchanged.
6. *The answer is a density.*  The Gibbs measure is absolutely continuous with respect to
   Lebesgue measure with Radon-Nikodym derivative the Boltzmann density, and a point
   prediction (a Dirac measure) is not: it has no density at all.
7. *Partition-function-free identities.*  Ratios of densities are `exp(-beta dU)`, and the
   gradient of the log density is exactly `-beta` times the gradient of the energy.  Both
   are what make score-based training and Metropolis sampling of the ensemble well posed.
8. *The exact price of non-equivariance.*  Under the continuous quadratic score, the risk of
   a model equals the risk of its symmetrisation plus a quarter of its own non-equivariance;
   equality holds precisely when the model is already equivariant almost everywhere.
9. *Representability.*  Every Borel probability measure on conformation space -- in
   particular the Gibbs ensemble -- is the push-forward of a one-dimensional uniform latent
   under a measurable generator. -/
theorem molecular_design_laws {N : ℕ} (hN : 0 < N) (F : ForceField N)
    (epsIn epsOut : ℝ) (q R gamma rad : Fin N → ℝ) (probe : ℝ)
    (hR : ∀ i, 0 < R i) (hg : ∀ i, 0 ≤ gamma i) :
    -- 1  the context is load-bearing
    ((∃ C C' : Context.BiophysicalContext,
        C.ionicStrength = C'.ionicStrength ∧ C.ligand = C'.ligand ∧
          Context.boltzmannPop C 1e-20 ≠ Context.boltzmannPop C' 1e-20) ∧
      (∃ C C' : Context.BiophysicalContext,
        C.temperature = C'.temperature ∧ C.ligand = C'.ligand ∧
          Context.screened C 1 1 1e-9 ≠ Context.screened C' 1 1 1e-9) ∧
      (∃ C C' : Context.BiophysicalContext,
        C.temperature = C'.temperature ∧ C.ionicStrength = C'.ionicStrength ∧
          Context.boundFraction C 1e-6 ≠ Context.boundFraction C' 1e-6) ∧
      (∀ f : ℝ, ∃ C : Context.BiophysicalContext, Context.boltzmannPop C 1e-20 ≠ f)) ∧
    -- 2  stability of the solvated Hamiltonian
    (∃ B : ℝ, ∀ x : Conf N, (∀ i j : Fin N, i ≠ j → x i ≠ x j) →
      -B ≤ Hsolv F epsIn epsOut q R gamma rad probe x) ∧
    -- 3  excluded volume, and the null set of exact overlaps
    ((∀ p0 ∈ pairs N, ∀ M : ℝ, ∃ d : ℝ, 0 < d ∧ ∀ x : Conf N,
        (∀ i j : Fin N, i ≠ j → x i ≠ x j) → dist (x p0.1) (x p0.2) < d → M < F.H x) ∧
      volume {x : Conf N | ∃ i j : Fin N, i ≠ j ∧ x i = x j} = 0) ∧
    -- 4  the solvent terms
    ((∀ Q Rad : ℝ, 0 < epsIn → epsIn < epsOut → Q ≠ 0 → 0 < Rad →
        gbPair epsIn epsOut Q Q Rad Rad 0 < 0) ∧
      (∀ (x : Conf N) (i j : Fin N), j ≠ i →
        dist (x i) (x j) + (rad i + probe) < rad j + probe → sasa rad probe x i = 0) ∧
      (∀ x y : Conf N, (∀ i, sasa rad probe y i ≠ ⊤) →
        (∀ i, accessibleSet rad probe x i ⊆ accessibleSet rad probe y i) →
        nonpolarEnergy gamma rad probe x ≤ nonpolarEnergy gamma rad probe y)) ∧
    -- 5  exact E(3) invariance of the energy
    (∀ (g : RigidMotion) (x : Conf N),
      Hsolv F epsIn epsOut q R gamma rad probe (g.act x)
        = Hsolv F epsIn epsOut q R gamma rad probe x) ∧
    -- 6  the answer is a density, and a point prediction is not
    ((∀ G : GibbsData N, G.gibbsMeasure ≪ volume ∧
        G.gibbsMeasure.rnDeriv volume =ᵐ[volume] fun x => ENNReal.ofReal (G.density x)) ∧
      (∀ x : Conf N, ¬ (Measure.dirac x ≪ (volume : Measure (Conf N))))) ∧
    -- 7  partition-function-free ratio and score
    (∀ G : GibbsData N,
      (∀ x1 ∈ G.D, ∀ x2 ∈ G.D,
        G.density x1 / G.density x2 = Real.exp (-(G.beta * (G.U x1 - G.U x2)))) ∧
      (∀ x : Conf N, G.D ∈ nhds x → ∀ dU : Conf N →L[ℝ] ℝ, HasFDerivAt G.U dU x →
        fderiv ℝ (fun y => Real.log (G.density y)) x = -G.beta • dU)) ∧
    -- 8  SE(3) invariance of the ensemble, and the price of non-equivariance
    ((∀ (G : GibbsData N) (g : RigidMotion), (∀ x, G.U (g.act x) = G.U x) →
        (∀ x, (g.act x ∈ G.D ↔ x ∈ G.D)) →
        Measure.map g.act G.gibbsMeasure = G.gibbsMeasure) ∧
      (∀ (f p : Conf N → ℝ) (g : RigidMotion), (∀ x, p (g.act x) = p x) →
        Integrable (fun x => (f x - p x) ^ 2) →
        Integrable (fun x => (symmetrise f g x - p x) ^ 2) →
        l2risk f p = l2risk (symmetrise f g) p + (1/4) * ∫ x, (f x - f (g.act x)) ^ 2) ∧
      (∀ (f p : Conf N → ℝ) (g : RigidMotion), (∀ x, p (g.act x) = p x) →
        Integrable (fun x => (f x - p x) ^ 2) →
        Integrable (fun x => (symmetrise f g x - p x) ^ 2) →
        Integrable (fun x => (f x - f (g.act x)) ^ 2) →
        (l2risk (symmetrise f g) p = l2risk f p ↔
          (fun x => f x) =ᵐ[volume] fun x => f (g.act x)))) ∧
    -- 9  representability of the ensemble as a push-forward
    (∀ mu : Measure (Conf N), IsProbabilityMeasure mu →
      ∃ T : ℝ → Conf N, Measurable T ∧ IsGenerator T unif mu) := by
  refine ⟨⟨Context.context_coordinates_load_bearing.1,
      Context.context_coordinates_load_bearing.2.1,
      Context.context_coordinates_load_bearing.2.2,
      fun f => Context.no_context_free_model f⟩,
    Hsolv_bddBelow F hR hg,
    ⟨fun p0 hp0 M => F.H_repulsive_core hp0 M, coincident_null N⟩,
    ⟨fun Q Rad hin hout hQ hRad => born_self_neg hin hout hQ hRad,
      fun x i j hij hin => sasa_eq_zero_of_engulfed hij hin,
      fun x y hfin h => nonpolar_le_of_occlusion hg hfin h⟩,
    fun g x => Hsolv_rigid_invariant F g x,
    ⟨fun G => ⟨G.gibbs_absolutelyContinuous, G.gibbs_rnDeriv⟩,
      fun x => dirac_not_absolutelyContinuous hN x⟩,
    fun G => ⟨fun x1 h1 x2 h2 => G.density_ratio h1 h2,
      fun x hx dU hU => G.score_eq_neg_beta_grad hx hU⟩,
    ⟨fun G g hU hD => gibbsMeasure_rigid_invariant G g hU hD,
      fun f p g hp hA hC => symmetrization_identity g hp hA hC,
      fun f p g hp hA hC hD => equal_risk_iff_equivariant g hp hA hC hD⟩,
    fun mu hmu => ?_⟩
  haveI := hmu
  exact exists_generator hN mu

end IDR
