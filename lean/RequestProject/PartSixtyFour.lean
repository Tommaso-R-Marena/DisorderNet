/-
# Part LXIV  Structure-based coarse-graining: the inverse problem

Part LX treats the forward direction -- integrating out the solvent leaves a potential of mean
force that is exact, not pairwise, and temperature dependent.  This part treats the inverse
direction, which is how coarse-grained models of disordered regions are actually built: fit a
potential so that the model reproduces a measured structural statistic (a radial distribution
function, a set of contact frequencies, a distance histogram), then use that potential
elsewhere.  Iterative Boltzmann inversion, force matching and relative-entropy coarse-graining
are all instances.  `RequestProject.InversePotential` answers three questions about it exactly.

`IDR.inverse_potential_laws` bundles four statements:

1. *The fit is well posed.*  Henderson's uniqueness theorem, in the discrete setting: two
   parameter vectors whose Boltzmann distributions have the same mean features have the *same*
   Boltzmann distribution.  The proof is the symmetrised relative entropy -- `KL(p‖p') +
   KL(p'‖p)` is exactly `b` times the inner product of the parameter difference with the
   feature-mean difference -- and Gibbs' inequality.  A fitted coarse-grained potential is not
   an arbitrary choice among many that fit.
2. *But the model it determines is blind to higher-order structure.*  On three spins, the parity
   ensemble (uniform on the four configurations with `s₁s₂s₃ = +1`) has all three pair
   correlations zero, exactly like the uniform ensemble, and triple correlation `1`.  By (1),
   every pair-potential model matching those pair correlations *is* the uniform ensemble, whose
   triple correlation is `0`.  Fitting the pair structure exactly gets the three-body structure
   maximally wrong, and no pair potential repairs it.
3. *The hypothesis of (2) is satisfiable* -- the zero potential matches the pair structure -- so
   the statement is not vacuous.
4. *And the fitted potential is a free energy.*  In the smallest two-state model, the potential
   reproducing the feature value `1/3` at inverse temperature `1` is `log 2`, and the one
   reproducing the same value at inverse temperature `2` is `(log 2)/2`; the first, transferred
   to the second temperature, does not reproduce the target.  A structure-based potential is a
   statement about a state point.

Together with Part LX these bracket coarse-graining from both sides: the forward map leaves an
object that is not a pairwise transferable potential, and the inverse map returns a unique but
equally non-transferable one, blind by construction to everything beyond the statistics it was
fitted to.  For a disordered region -- where the interesting behaviour (collapse, cooperativity,
condensation) is exactly the many-body part, and where the state point changes with salt,
temperature and crowding -- both halves of that bracket bite.
-/
import Mathlib
import RequestProject.InversePotential

set_option autoImplicit false

namespace IDR

open IDR.Inverse

/-- **The inverse coarse-graining laws.**

1. matching the structural statistics determines the model (Henderson uniqueness);
2. yet a pair-potential model matching the pair structure of the parity ensemble has the wrong
   three-body structure -- `0` against `1`;
3. that hypothesis is satisfiable, by the zero potential;
4. and the potential fitted to a target statistic at one temperature is not the one fitted to
   the same target at another. -/
theorem inverse_potential_laws :
    (∀ (N m : ℕ) (_ : NeZero N) (n : Fin m → Fin N → ℝ) (theta theta' : Fin m → ℝ) (b : ℝ),
        (∀ a, meanFeature n (gibbs n theta b) a = meanFeature n (gibbs n theta' b) a) →
          gibbs n theta b = gibbs n theta' b) ∧
    (∀ (b : ℝ) (theta : Fin 3 → ℝ),
        (∀ a, meanFeature pairFeat (gibbs pairFeat theta b) a
          = meanFeature pairFeat parityEns a) →
          (∑ x, gibbs pairFeat theta b x * tripleFeat x) = 0 ∧
            (∑ x, parityEns x * tripleFeat x) = 1) ∧
    (∀ (b : ℝ) (a : Fin 3), meanFeature pairFeat (gibbs pairFeat (fun _ => 0) b) a
        = meanFeature pairFeat parityEns a) ∧
    (meanFeature twoFeat (gibbs twoFeat (fun _ => Real.log 2) 1) 0 = 1/3 ∧
      meanFeature twoFeat (gibbs twoFeat (fun _ => Real.log 2 / 2) 2) 0 = 1/3 ∧
      meanFeature twoFeat (gibbs twoFeat (fun _ => Real.log 2) 2) 0 ≠ 1/3) := by
  refine ⟨fun N m _ n theta theta' b h => gibbs_unique_of_meanFeature_eq n theta theta' b h,
    fun b theta h => pair_potentials_blind_to_three_body b theta h,
    fun b a => zero_matches_pair_structure b a,
    inverse_potential_temperature_dependent⟩

end IDR
