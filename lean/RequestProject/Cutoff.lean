/-
# Part XXIX.2  A truncated interaction cannot see the long-range structure

The second unavoidable approximation of a simulated ensemble is the treatment of long-range
forces.  Truncating a pair potential at a cutoff `rc` is exact for a short-ranged interaction
and catastrophic for a disordered polyelectrolyte, whose conformational free energy is
dominated by charge pairs that are *far apart along the chain and in space*.  This file makes
the failure exact rather than rhetorical.

* `truncEnergy_eq_zero_of_beyond` -- a conformation all of whose pair distances exceed the
  cutoff has *zero* truncated interaction energy, whatever the true potential.
* `cutoff_blind` -- hence any two such conformations receive exactly equal Boltzmann weight
  under the truncated model, at every temperature.  A cutoff model does not merely misestimate
  the long-range part of the landscape: it assigns no landscape at all beyond `rc`.
* `cutoff_loses_true_ranking` -- and the true model does rank them: two two-particle
  conformations beyond the cutoff of a Coulomb-like attraction have different true energies and
  identical truncated energies, so the truncated ensemble populates them equally and the true
  ensemble does not.
* `truncation_error_le` -- the quantitative version: with a tail bound `|u(r)| ≤ C/r` the total
  neglected energy of an `n`-particle conformation is at most `n²C/rc`.  The bound is
  *quadratic in chain length at fixed cutoff*, which is why the error of a truncated
  electrostatics model on a long disordered region cannot be absorbed into a constant and why
  a lattice-sum (Ewald/PME) treatment is not optional.

Together with `RequestProject.Box` this says that a reported ensemble depends on two
simulation parameters -- box size and cutoff -- in a direction that is known a priori
(compaction, and loss of long-range discrimination).  A model of a disordered region is
defensible only if the ensemble it reports is shown to be stationary in both.
-/
import Mathlib

set_option autoImplicit false

namespace Cutoff

open Finset

variable {n : ℕ} {E : Type*} [PseudoMetricSpace E]

/-- Total pairwise interaction energy of a conformation (each unordered pair counted twice). -/
noncomputable def pairEnergy (u : ℝ → ℝ) (x : Fin n → E) : ℝ :=
  ∑ i, ∑ j, if i = j then 0 else u (dist (x i) (x j))

/-- The potential truncated at `rc`. -/
noncomputable def truncate (rc : ℝ) (u : ℝ → ℝ) : ℝ → ℝ := fun r => if r ≤ rc then u r else 0

/-- Boltzmann weight of a conformation under a pair potential. -/
noncomputable def weight (beta : ℝ) (u : ℝ → ℝ) (x : Fin n → E) : ℝ :=
  Real.exp (-beta * pairEnergy u x)

/-- **Beyond the cutoff the model is empty.**  A conformation whose pair distances all exceed
`rc` has zero truncated energy. -/
theorem truncEnergy_eq_zero_of_beyond {rc : ℝ} (u : ℝ → ℝ) {x : Fin n → E}
    (hx : ∀ i j : Fin n, i ≠ j → rc < dist (x i) (x j)) :
    pairEnergy (truncate rc u) x = 0 := by
  unfold pairEnergy truncate
  refine Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun j _ => ?_
  by_cases hij : i = j
  · simp [hij]
  · simp only [hij, if_false]
    rw [if_neg (not_le.mpr (hx i j hij))]

/-- **A cutoff model cannot rank long-range conformations.**  Any two conformations lying
entirely beyond the cutoff get exactly the same Boltzmann weight, at every temperature. -/
theorem cutoff_blind {rc beta : ℝ} (u : ℝ → ℝ) {x y : Fin n → E}
    (hx : ∀ i j : Fin n, i ≠ j → rc < dist (x i) (x j))
    (hy : ∀ i j : Fin n, i ≠ j → rc < dist (y i) (y j)) :
    weight beta (truncate rc u) x = weight beta (truncate rc u) y := by
  unfold weight
  rw [truncEnergy_eq_zero_of_beyond u hx, truncEnergy_eq_zero_of_beyond u hy]

/-- The Coulomb-like attraction `u(r) = -1/r`. -/
noncomputable def coulombAttraction : ℝ → ℝ := fun r => -(1 / r)

/-- **The true model does rank them.**  Two conformations of a pair of charges at separations
`2` and `3`, with the cutoff at `1`: the truncated energies are equal (both zero) and the true
energies are not, so the truncated ensemble populates the two conformations equally and the
true ensemble does not. -/
theorem cutoff_loses_true_ranking :
    pairEnergy (truncate 1 coulombAttraction) (![(0 : ℝ), 2] : Fin 2 → ℝ)
        = pairEnergy (truncate 1 coulombAttraction) (![(0 : ℝ), 3] : Fin 2 → ℝ) ∧
      pairEnergy coulombAttraction (![(0 : ℝ), 2] : Fin 2 → ℝ)
        ≠ pairEnergy coulombAttraction (![(0 : ℝ), 3] : Fin 2 → ℝ) := by
  have hd2 : dist (0 : ℝ) 2 = 2 := by
    rw [Real.dist_eq]; norm_num
  have hd2' : dist (2 : ℝ) 0 = 2 := by
    rw [Real.dist_eq]; norm_num
  have hd3 : dist (0 : ℝ) 3 = 3 := by
    rw [Real.dist_eq]; norm_num
  have hd3' : dist (3 : ℝ) 0 = 3 := by
    rw [Real.dist_eq]; norm_num
  constructor
  · rw [truncEnergy_eq_zero_of_beyond, truncEnergy_eq_zero_of_beyond]
    · intro i j hij
      fin_cases i <;> fin_cases j <;> simp_all
    · intro i j hij
      fin_cases i <;> fin_cases j <;> simp_all
  · simp [pairEnergy, Fin.sum_univ_two, coulombAttraction, hd2, hd3]
    norm_num

/-- **The neglected energy is quadratic in the chain length.**  If the tail of the potential is
bounded by `C/r` beyond the cutoff, the total truncation error of an `n`-particle conformation
is at most `n²C/rc`; at fixed cutoff it grows like the square of the number of interacting
sites. -/
theorem truncation_error_le {rc C : ℝ} (hrc : 0 < rc) (hC : 0 ≤ C) (u : ℝ → ℝ)
    (hu : ∀ r : ℝ, rc < r → |u r| ≤ C / r) (x : Fin n → E) :
    |pairEnergy u x - pairEnergy (truncate rc u) x| ≤ n ^ 2 * (C / rc) := by
  have hdiff : pairEnergy u x - pairEnergy (truncate rc u) x
      = ∑ i, ∑ j, ((if i = j then 0 else u (dist (x i) (x j)))
          - (if i = j then 0 else truncate rc u (dist (x i) (x j)))) := by
    unfold pairEnergy
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => by rw [← Finset.sum_sub_distrib]
  have hbound : ∀ i j : Fin n,
      |(if i = j then 0 else u (dist (x i) (x j)))
        - (if i = j then 0 else truncate rc u (dist (x i) (x j)))| ≤ C / rc := by
    intro i j
    by_cases hij : i = j
    · simp [hij, div_nonneg hC hrc.le]
    · simp only [hij, if_false, truncate]
      by_cases hcut : dist (x i) (x j) ≤ rc
      · simp [hcut, div_nonneg hC hrc.le]
      · push_neg at hcut
        rw [if_neg (not_le.mpr hcut), sub_zero]
        exact le_trans (hu _ hcut) (div_le_div_of_nonneg_left hC hrc hcut.le)
  calc |pairEnergy u x - pairEnergy (truncate rc u) x|
      ≤ ∑ i, ∑ j, |(if i = j then 0 else u (dist (x i) (x j)))
          - (if i = j then 0 else truncate rc u (dist (x i) (x j)))| := by
        rw [hdiff]
        refine le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun i _ => ?_)
        exact Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _i : Fin n, ∑ _j : Fin n, C / rc := by
        refine Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => hbound i j
    _ = n ^ 2 * (C / rc) := by
        simp [Finset.sum_const, Finset.card_univ]
        ring

end Cutoff
