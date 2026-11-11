/-
# Part LI  Electronic polarisability, priced exactly

Part XXV named three idealisations that its own results did not remove, and put electronic
polarisability first: the force field of Part XXIV assigns every atom a fixed charge, so its
energy is a sum of one- and two-body terms.  A disordered region is exactly where that hurts
most -- a chain of exposed charges and amides, reorganising continuously, with no folded core
to make the induced response a constant background.

`RequestProject.Polarisability` removes the idealisation rather than apologising for it.  The
induced dipoles of a point-polarisable model are not given by a formula but by a
self-consistent linear system; the file proves the system is *well posed* under the standard
damping condition (unique solution, existing for every external field), solves the symmetric
cluster in closed form, and computes the exact residue that a pairwise fit must leave behind.

`IDR.polarisability_laws` bundles four statements:

1. *Well-posedness*: for every damped polarisable model and every external field there is
   exactly one set of induced dipoles.
2. *The closed form*: in a symmetric `m`-site cluster in a uniform field the dipole is
   `a/(1-(m-1)at)` and the energy is `-(m/2)` times it.
3. *The exact three-body term*: `U₃ - 3U₂ + 3U₁ = -3a³t²/((1-2at)(1-at))`.
4. *The obstruction*: in the damped regime with nonzero coupling that residue is strictly
   negative, so no one-body-plus-two-body energy model whatsoever reproduces the cluster
   energies.

The reading for a model of a disordered region: polarisation is a genuine many-body term, it
is cooperative, it is second order in the dipole-dipole coupling -- and it is representable,
at the price of solving a self-consistent field at every configuration.  A fixed-charge model
is not a polarisable model with badly chosen parameters; it is a different model, and the
difference is the number above.
-/
import Mathlib
import RequestProject.Polarisability

set_option autoImplicit false

namespace IDR

open IDR.Polarisability

/-- **Electronic polarisability: well posed, solved, and not pairwise.**

1. *Well-posedness* of the self-consistent field under damping;
2. the *closed form* for the symmetric cluster;
3. the *exact three-body residue*;
4. the *representability obstruction* it creates for any pairwise energy model. -/
theorem polarisability_laws :
    (∀ (n : ℕ) (a : Fin n → ℝ) (T : Fin n → Fin n → ℝ) (c : ℝ), Damped a T c →
        ∀ E : Fin n → ℝ, ∃! mu : Fin n → ℝ, SelfConsistent a T E mu) ∧
    (∀ (m : ℕ) (a t : ℝ), 1 - ((m : ℝ) - 1) * a * t ≠ 0 →
        SelfConsistent (uniformPol m a) (uniformCoupling m t) (uniformField m)
            (fun _ => clusterDipole m a t) ∧
          energy (uniformField m) (fun _ => clusterDipole m a t) = clusterEnergy m a t) ∧
    (∀ a t : ℝ, 1 - 2 * a * t ≠ 0 → 1 - a * t ≠ 0 →
        threeBody a t = -3 * a ^ 3 * t ^ 2 / ((1 - 2 * a * t) * (1 - a * t))) ∧
    (∀ a t : ℝ, 0 < a → 0 < t → a * t < 1 / 2 →
        threeBody a t < 0 ∧ ¬ IsPairEnergy (clusterEnergyOn a t)) := by
  refine ⟨fun n a T c hd E => ?_, fun m a t h => ⟨clusterDipole_selfConsistent m a t h,
      energy_uniform_eq m a t⟩, fun a t h2 h1 => threeBody_eq a t h2 h1,
    fun a t ha ht hd => ⟨threeBody_neg ha ht hd, polarisable_not_pairwise_additive ha ht hd⟩⟩
  obtain ⟨mu, hmu⟩ := selfConsistent_exists hd E
  exact ⟨mu, hmu, fun nu hnu => selfConsistent_unique hd hnu hmu⟩

end IDR
