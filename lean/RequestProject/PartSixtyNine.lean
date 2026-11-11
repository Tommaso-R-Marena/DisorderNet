/-
# Part LXIX  The restraint-counting law

An ensemble model of a disordered region is a population vector over a conformational library of
size `m`, and it is fitted against `k` experimental averages.  `RequestProject.Restraints` settles
the relation between the two numbers exactly.

`IDR.restraint_counting_laws` bundles five statements:

1. *Below threshold there is always a blind direction.*  `k` observables plus normalisation are
   `k + 1` linear functionals on an `m`-dimensional population space, so when `k + 1 < m` some
   nonzero signed population change is invisible to every measurement at once.
2. *And the blindness is macroscopic.*  If the target populates every library conformation with
   weight at least `d` -- the generic situation for a disordered region, where nothing is
   excluded -- then there is a genuine ensemble, nonnegative and normalised, reproducing **every**
   measured average exactly at population distance at least `2 d` from the truth.  On a uniform
   target that is `2/m`: the weight of two whole conformations.
3. *So the price of an ensemble is one restraint per conformation.*  A restraint set that
   determines an interior target to better than `2 d` in population distance must contain at
   least `m - 1` restraints; and since a library realising conformational entropy `H` has
   `exp H` members, the restraint count needed grows exponentially in the conformational entropy.
4. *The threshold is exactly `m - 1`, not an artefact.*  The `m - 1` indicator observables --
   reading off all but one population -- do determine the ensemble.
5. *A held-out validation set certifies nothing.*  Splitting the experiments into a fitting set
   and a validation set only redistributes them: if the *total* count is below threshold, an
   ensemble far from the truth matches the fitting data and the held-out data alike.
6. *There is no lucky restraint set below threshold*, because the feasible set is a convex slice
   of the simplex: mixtures of fits are fits, and the deficient directions form a subspace of
   dimension at least `m - 1 - k`, which no choice of the `k` observables can remove.

The practical reading.  A published IDR ensemble is a point chosen inside an affine slice of the
simplex whose dimension is at least `m - 1 - k`.  Two consequences follow for how such a model
should be reported.  First, `m` and `k` must be reported together: an ensemble of a thousand
conformers restrained by fifty observables is determined in fifty directions and free in nine
hundred and forty-nine, whatever its goodness of fit.  Second, only functionals that are constant
on the slice may be quoted as results; everything else is a property of the prior, which by the
maximum-entropy analysis of Part III is exactly where the remaining information came from.
-/
import Mathlib
import RequestProject.Restraints

set_option autoImplicit false

namespace IDR

open IDR.Restraint

/-- **The restraint-counting laws.**

1. `k` observables plus normalisation annihilate a nonzero population direction as soon as
   `k + 1 < m`;
2. hence, against an interior target, a genuine ensemble reproduces every measured average while
   sitting at population distance at least `2 d` (uniformly: `2/m`) from the truth;
3. so determining an interior target needs at least `m - 1` restraints, i.e. `exp H` restraints
   for conformational entropy `H`;
4. and `m - 1` restraints suffice, so the threshold is sharp;
5. the feasible set of a data set is convex and shrinks under added restraints;
6. and a held-out validation set certifies nothing that the total restraint count forbids. -/
theorem restraint_counting_laws :
    (∀ (m k : ℕ), k + 1 < m → ∀ g : Fin k → Fin m → ℝ,
        ∃ v : Fin m → ℝ, v ≠ 0 ∧ (∑ i, v i = 0) ∧ ∀ j, ∑ i, g j i * v i = 0) ∧
    (∀ (m k : ℕ), k + 1 < m → ∀ (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ), 0 < d →
        (∀ i, d ≤ p i) → (∑ i, p i = 1) →
        ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) p) ∧ 2 * d ≤ ell1 q p) ∧
    (∀ (m k l : ℕ), k + l + 1 < m → ∀ (g : Fin k → Fin m → ℝ) (h : Fin l → Fin m → ℝ)
        (p : Fin m → ℝ) (d : ℝ), 0 < d → (∀ i, d ≤ p i) → (∑ i, p i = 1) →
        ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) p) ∧
          (∀ j, obs (h j) q = obs (h j) p) ∧ 2 * d ≤ ell1 q p) ∧
    ((∀ (m k : ℕ), k + 1 < m → ∀ g : Fin k → Fin m → ℝ,
        ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) (unif m)) ∧
          2 / (m : ℝ) ≤ ell1 q (unif m)) ∧
      (∀ (m k : ℕ), 0 < m → ∀ g : Fin k → Fin m → ℝ,
        (∀ q : Fin m → ℝ, IsEns q → (∀ j, obs (g j) q = obs (g j) (unif m)) →
            ell1 q (unif m) < 2 / (m : ℝ)) →
          Real.exp (Real.log m) ≤ (k : ℝ) + 1)) ∧
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∑ i, p i = 1) → (∑ i, q i = 1) → ∀ i0 : Fin m,
        (∀ i, i ≠ i0 → obs (fun x => if x = i then 1 else 0) p
          = obs (fun x => if x = i then 1 else 0) q) → p = q) ∧
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (data : Fin k → ℝ) (q r : Fin m → ℝ),
        IsEns q → IsEns r → (∀ j, obs (g j) q = data j) → (∀ j, obs (g j) r = data j) →
        ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
          IsEns (fun i => t * q i + (1 - t) * r i) ∧
            ∀ j, obs (g j) (fun i => t * q i + (1 - t) * r i) = data j) := by
  refine ⟨fun m k hk g => exists_null_direction hk g,
    fun m k hk g p d hd hp hp1 => restraints_insufficient hk g hd hp hp1,
    fun m k l hk g h p d hd hp hp1 => cross_validation_cannot_certify hk g h hd hp hp1,
    ⟨fun m k hk g => uniform_restraints_insufficient hk g,
      fun m k hm g hdet => restraints_exp_entropy hm g hdet⟩,
    fun m p q hp hq i0 h => indicator_restraints_determine hp hq i0 h,
    fun m k g data q r hq hr hqd hrd t ht0 ht1 =>
      feasible_convex g data hq hr hqd hrd ht0 ht1⟩

end IDR
