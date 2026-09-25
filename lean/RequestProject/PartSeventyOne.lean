/-
# Part LXXI  The precision floor of an ensemble measurement

Parts LXIX and LXX assumed the restraints are matched exactly.  Real restraints are matched to a
tolerance: an error bar, a chi-squared target, the statistical uncertainty of a finite run.
`RequestProject.Tolerance` computes what that tolerance costs, and the answer does not improve
with the length of the restraint list.

`IDR.precision_laws` bundles four statements, for a library of size `m`, an interior target `p`
populating every conformation with weight at least `d`, and `k` observables bounded by `G`:

1. *What a transfer of population does.*  Moving weight `c` from conformation `b` to conformation
   `a` changes the `j`-th measured average by exactly `c (g j a - g j b)` and moves the ensemble
   by at least `2 c` in population distance.  Only the *contrast* of an observable between the two
   conformations is visible to the data.
2. *Structural degeneracy.*  If two conformations give equal values of every measured observable,
   their relative population is not determined at all: an ensemble `2 d` away in population
   distance matches every measurement exactly.  This is the conformational counterpart of the
   sequence degeneracy of Part LXVII, and it holds for every `k`.
3. *The precision floor.*  Whatever the number of restraints, an ensemble matched to tolerance
   `eps` is pinned down only to population distance `min (2 d) (eps / G)`.  The bound contains no
   `k`: more experiments cannot push it down, only smaller `eps` can.
4. *The matching ceiling.*  If the populations themselves are known to tolerance `eps`, two
   consistent ensembles differ by at most `2 (m - 1) eps`.  So the resolution is linear in the
   tolerance from both sides.

The three parts LXIX-LXXI together give the full accounting for a reported ensemble of a
disordered region.  The restraint count fixes *how many* independent numbers can be verified
(at most `k + 1`, Part LXX); membership in the measured span fixes *which* ones (Part LXX); and
the tolerance fixes *to what precision* they are verified, at a floor of order `eps / G` that no
further experiment removes (this part).  A reported ensemble is therefore a measurement with a
stated resolution, and everything quoted below that resolution -- or outside the measured span,
or beyond the restraint count -- is a property of the reference ensemble rather than of the
protein.
-/
import Mathlib
import RequestProject.Tolerance

set_option autoImplicit false

namespace IDR

open IDR.Restraint IDR.Tolerance

/-- **The precision laws of an ensemble measurement.**

1. transferring population `c` from `b` to `a` shifts the `j`-th average by `c (g j a - g j b)`
   and the ensemble by at least `2 c`;
2. two conformations with equal values of every observable have completely undetermined relative
   population;
3. at tolerance `eps` and observable bound `G`, the ensemble is determined only to population
   distance `min (2 d) (eps / G)`, for every restraint count `k`;
4. and if the populations are known to tolerance `eps`, two consistent ensembles differ by at
   most `2 (m - 1) eps`. -/
theorem precision_laws :
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d c : ℝ), 0 < d → 0 < c → c ≤ d →
        (∀ i, d ≤ p i) → (∑ i, p i = 1) → ∀ a b : Fin m, a ≠ b →
        ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q - obs (g j) p = c * (g j a - g j b)) ∧
          2 * c ≤ ell1 q p) ∧
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (p : Fin m → ℝ) (d : ℝ), 0 < d → (∀ i, d ≤ p i) →
        (∑ i, p i = 1) → ∀ a b : Fin m, a ≠ b → (∀ j, g j a = g j b) →
        ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, obs (g j) q = obs (g j) p) ∧ 2 * d ≤ ell1 q p) ∧
    (∀ (m k : ℕ) (g : Fin k → Fin m → ℝ) (G eps : ℝ), 0 < G → 0 < eps → (∀ j i, |g j i| ≤ G) →
        ∀ (p : Fin m → ℝ) (d : ℝ), 0 < d → (∀ i, d ≤ p i) → (∑ i, p i = 1) →
        ∀ a b : Fin m, a ≠ b →
        ∃ q : Fin m → ℝ, IsEns q ∧ (∀ j, |obs (g j) q - obs (g j) p| ≤ eps) ∧
          min (2 * d) (eps / G) ≤ ell1 q p) ∧
    (∀ (m : ℕ) (p q : Fin m → ℝ) (eps : ℝ), (∑ i, p i = 1) → (∑ i, q i = 1) → ∀ i0 : Fin m,
        (∀ i, i ≠ i0 → |q i - p i| ≤ eps) → ell1 q p ≤ 2 * (m - 1 : ℕ) * eps) := by
  refine ⟨fun m k g p d c hd hc hcd hp hp1 a b hab =>
      exists_pair_perturbation g hd hc hcd hp hp1 a b hab,
    fun m k g p d hd hp hp1 a b hab hdeg => pair_degeneracy g hd hp hp1 hab hdeg,
    fun m k g G eps hG heps hgb p d hd hp hp1 a b hab =>
      precision_floor g hG heps hgb hd hp hp1 hab,
    fun m p q eps hp1 hq1 i0 h => tolerance_ceiling hp1 hq1 i0 h⟩

end IDR
