/-
# Part LXXXVI  What a cryo-EM map of a disordered region establishes

Capstone for `RequestProject.CryoEM`.

A single-particle reconstruction averages over the particles it was built from.  The map of an
atom is therefore its *occupancy distribution over voxels*, and the modelling question is what such
an object determines.  The answer proved here is two-sided: it determines every single-atom
average exactly, and it determines the number of conformations from below through its peak height;
it determines no joint information whatever, and it cannot be repaired by classification alone.
-/
import RequestProject.CryoEM

set_option autoImplicit false

namespace IDR

open Cryo Finset

/-- **The cryo-EM occupancy laws.**  For an ensemble `E` of conformations on a finite voxel grid
`X`, writing `dens E a x` for the population of conformations placing atom `a` in voxel `x`:

1. *the map is an occupancy*: `dens E a ·` is nonnegative, sums to one over the grid, and never
   exceeds the value `1` that a single rigid structure would put at its own voxel;
2. *peak height counts conformations*: if no voxel of the map rises above `p > 0`, the atom
   occupies at least `1/p` distinct voxels — a weak map is a conformation count, not a badly
   resolved structure;
3. *the contour level*: at most `1/t` voxels per atom survive display at level `t`;
4. *every single-atom average is a read-out of the map*, so mean position, positional variance and
   any other one-atom statistic are determined exactly;
5. *and two ensembles with the same map agree on all of them*;
6. *classification does not add information by itself*: for **any** partition of the particles into
   classes, the class maps sum back to the total map, so reproducing the consensus map is no
   evidence that a classification is correct;
7. *a `K`-class reconstruction stays a `K`-point model*: if the particles realise `n > K` distinct
   positions of an atom, every assignment of one structure per class leaves a strictly positive
   mean squared error, however the classes are chosen. -/
theorem cryoem_occupancy_laws {n : ℕ} {A X : Type*} [Fintype X] [DecidableEq X]
    (E : Ens n A X) (a : A) :
    (∀ x, 0 ≤ dens E a x) ∧ (∑ x, dens E a x = 1) ∧ (∀ x, dens E a x ≤ 1) ∧
    (∀ p : ℝ, 0 < p → (∀ x, dens E a x ≤ p) → 1 / p ≤ (support E a).card) ∧
    (∀ t : ℝ, 0 < t → ((univ.filter fun x => t ≤ dens E a x).card : ℝ) ≤ 1 / t) ∧
    (∀ f : X → ℝ, ∑ j, E.w j * f (E.pos j a) = ∑ x, f x * dens E a x) ∧
    (∀ {n' : ℕ} (E' : Ens n' A X), (∀ x, dens E a x = dens E' a x) →
      ∀ f : X → ℝ, ∑ j, E.w j * f (E.pos j a) = ∑ j, E'.w j * f (E'.pos j a)) ∧
    (∀ (K : ℕ) (kap : Fin n → Fin K) (x : X), ∑ k, classDens E kap k a x = dens E a x) ∧
    (∀ (c : X → ℝ), (∀ j, 0 < E.w j) → Function.Injective (fun j => c (E.pos j a)) →
      ∀ (K : ℕ), K < n → ∀ (kap : Fin n → Fin K) (mu : Fin K → ℝ), 0 < resid E a c kap mu) := by
  refine ⟨fun x => dens_nonneg E a x, sum_dens_eq_one E a, fun x => dens_le_one E a x,
    fun p hp0 hp => card_support_ge_inv_peak E a hp0 hp,
    fun t ht => card_above_le_inv_threshold E a ht,
    fun f => expect_from_map E a f,
    fun E' h f => expect_eq_of_dens_eq E E' a f h,
    fun K kap x => sum_classDens E kap a x,
    fun c hpos hinj K hK kap mu => resid_pos_of_injective E a c hpos hinj hK kap mu⟩

/-- **What a map cannot establish, and the disappearance of disordered density.**

1. *Correlations are destroyed by the average*: two explicit two-residue ensembles have identical
   maps for both residues, while in one the residues are always in the same voxel and in the other
   never.  No reconstruction, at any resolution, distinguishes them.
2. *Invisibility is a theorem*: an atom spread uniformly over `m` voxels with `1/m < t` has no
   voxel at all above the contour level `t`, and the sub-threshold voxels carry the entire
   population.  "No density" is a statement about populations, not about absence. -/
theorem cryoem_blindness_laws :
    ((∀ a x : Fin 2, dens contactEns a x = dens antiEns a x) ∧
      contactFreq contactEns = 1 ∧ contactFreq antiEns = 0) ∧
    (∀ (m : ℕ) (hm : 0 < m) (t : ℝ), 1 / m < t →
      (∀ x, dens (uniformSpread m hm) () x < t) ∧
      ∑ x ∈ univ.filter fun x => dens (uniformSpread m hm) () x < t,
        dens (uniformSpread m hm) () x = 1) := by
  refine ⟨map_blind_to_correlation, fun m hm t ht => ?_⟩
  exact ⟨uniform_spread_invisible m hm ht, invisible_mass m hm ht⟩

end IDR
