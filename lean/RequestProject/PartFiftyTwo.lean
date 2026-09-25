/-
# Part LII  Nuclear quantum effects, and the isotope effect a classical model cannot have

`RequestProject.NuclearQuantum` removes the last of the three idealisations Part XXV named and
did not price: the nuclei of the chain are treated as classical particles.  For one harmonic
mode the whole comparison is exact, and it is not small.

`IDR.nuclear_quantum_laws` bundles five statements:

1. *The sandwich* `F_cl < F_q < w/2`: the classical free energy of a mode is strictly below the
   quantum one at every temperature and every frequency, and the quantum one is strictly below
   the zero-point energy.
2. *Equipartition fails*: the mean energy of the mode exceeds both `kT` and `w/2`.
3. *The classical limit is exact*: as the mode softens, the ratio of partition functions tends
   to `1`, so the soft collective motions of a disordered chain are safely classical.
4. *No classical isotope effect*: a mass substitution scales every frequency by one factor `s`,
   which cancels exactly from every classical free energy difference.  A classical force field
   predicts zero equilibrium isotope effect -- identically, not approximately.
5. *A quantum isotope effect*: for the same substitution the quantum free energy differences
   are not equal at all temperatures, because their low-temperature limits are the zero-point
   differences `(wA - wB)/2` and `s(wA - wB)/2`.

Statement 4 with statement 5 is the reviewer-proof form of the point: an H/D equilibrium
measurement on a disordered region is a measurement of something a classical model assigns the
value zero.  Together with Parts XXIV, XXV and LI, the molecular model now carries an explicit
price for pairwise additivity, fixed charges, finite sampling, rigid constraints, electronic
polarisability and classical nuclei.
-/
import Mathlib
import RequestProject.NuclearQuantum

set_option autoImplicit false

namespace IDR

open IDR.NuclearQuantum
open Filter Topology

/-- **Nuclear quantum effects, exactly.**

1. `F_cl < F_q < w/2`;
2. equipartition fails: `max (kT) (w/2) < U_q`;
3. the classical limit of the partition function is exact;
4. a classical model has *no* equilibrium isotope effect;
5. the quantum model has one. -/
theorem nuclear_quantum_laws :
    (∀ b w : ℝ, 0 < b → 0 < w → clFree b w < qFree b w ∧ qFree b w < w / 2) ∧
    (∀ b w : ℝ, 0 < b → 0 < w → 1 / b < qEnergy b w ∧ w / 2 < qEnergy b w) ∧
    (∀ b : ℝ, 0 < b →
        Tendsto (fun w : ℝ => qPartition b w / clPartition b w) (𝓝[>] (0 : ℝ)) (𝓝 1)) ∧
    (∀ b s wA wB : ℝ, 0 < b → 0 < s → 0 < wA → 0 < wB →
        clFree b (s * wA) - clFree b (s * wB) = clFree b wA - clFree b wB) ∧
    (∀ s wA wB : ℝ, 0 < s → s < 1 → 0 < wB → wB < wA →
        ∃ b : ℝ, 0 < b ∧
          qFree b wA - qFree b wB ≠ qFree b (s * wA) - qFree b (s * wB)) :=
  ⟨fun _ _ hb hw => ⟨clFree_lt_qFree hb hw, qFree_lt_zpe hb hw⟩,
    fun _ _ hb hw => ⟨qEnergy_gt_kT hb hw, qEnergy_gt_zpe hb hw⟩,
    fun _ hb => tendsto_partition_ratio_one hb,
    fun _ _ _ _ hb hs hA hB => classical_isotope_independent hb hs hA hB,
    fun _ _ _ hs0 hs1 hB hAB => quantum_isotope_effect hs0 hs1 hB hAB⟩

end IDR
