/-
# Part XIII.0  Boltzmann ensembles on a finite conformation library

Parts I-XII took the ensemble as given.  This file manufactures it from a *Hamiltonian*: given
a finite library of conformations `g : Fin m → X` and an energy `U : Fin m → ℝ`, `boltzEns`
is the equilibrium ensemble at inverse temperature `beta`.  Two facts about it are used later.

* `boltzEns_flat_same` : a landscape on which the energy does not distinguish conformations
  gives exactly the uniform ensemble -- the maximally disordered target.
* `boltz_ground_ge` / `ground_state_capacity` : if the energy minimum is attained on a set `S`
  of conformations and every other conformation lies at least `gap` above it, then each ground
  state keeps population at least `1 / (|S| + m e^{-beta·gap})`, and hence *even at zero
  temperature* a model must carry essentially `|S|` components.  Cooling does not reduce the
  capacity requirement below the ground-state degeneracy.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Statistics
import RequestProject.Metric
import RequestProject.FreeEnergy

namespace IDR.Boltz

open IDR.FreeEnergy
open scoped Classical

variable {X : Type*} {m : ℕ}

/-- **The equilibrium ensemble of a Hamiltonian.**  Conformations `g 0, …, g (m-1)` with
energies `U 0, …, U (m-1)`, populated according to Boltzmann's law at inverse temperature
`beta`. -/
noncomputable def boltzEns (hm : 0 < m) (g : Fin m → X) (beta : ℝ) (U : Fin m → ℝ) : Ens X where
  card := m
  pt := g
  w := boltz beta U
  w_nonneg := fun j => le_of_lt (boltz_pos hm beta U j)
  w_sum := boltz_sum_one hm beta U

lemma boltzEns_expect (hm : 0 < m) (g : Fin m → X) (beta : ℝ) (U : Fin m → ℝ) (f : X → ℝ) :
    (boltzEns hm g beta U).expect f = ∑ j, boltz beta U j * f (g j) := rfl

lemma boltzEns_prob (hm : 0 < m) {g : Fin m → X} (hg : Function.Injective g) (beta : ℝ)
    (U : Fin m → ℝ) (j : Fin m) :
    (boltzEns hm g beta U).prob (g j) = boltz beta U j :=
  Ens.prob_pt_of_injective (boltzEns hm g beta U) hg j

/-- Conformations outside the library carry no population. -/
lemma boltzEns_prob_not_mem (hm : 0 < m) {g : Fin m → X} {beta : ℝ} {U : Fin m → ℝ} {x : X}
    (hx : ∀ l, g l ≠ x) : (boltzEns hm g beta U).prob x = 0 := by
  classical
  simp only [Ens.prob, boltzEns_expect]
  refine Finset.sum_eq_zero fun j _ => ?_
  simp [hx j]

/-- **A flat landscape is exactly the uniform ensemble.**  When the force field does not
distinguish the conformations, equilibrium is maximal disorder, at every temperature. -/
theorem boltzEns_flat_same (hm : 0 < m) (g : Fin m → X) (beta c : ℝ) :
    (boltzEns hm g beta (fun _ => c)).Same (unif hm g) := by
  intro f
  rw [boltzEns_expect, unif_expect, Finset.sum_div]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [flat_landscape_uniform hm beta c]
  ring

/-- **Ground states keep their population as the temperature falls.**  If `U` equals its
minimum on `S` and exceeds it by at least `gap` off `S`, every ground state is populated at
least `1 / (|S| + m e^{-beta·gap})`. -/
theorem boltz_ground_ge (hm : 0 < m) {beta : ℝ} (hbeta : 0 < beta) {U : Fin m → ℝ}
    {Umin gap : ℝ} {S : Finset (Fin m)} (hSmin : ∀ j ∈ S, U j = Umin)
    (hout : ∀ j ∉ S, Umin + gap ≤ U j) {j0 : Fin m} (hj0 : j0 ∈ S) :
    1 / ((S.card : ℝ) + m * Real.exp (-beta * gap)) ≤ boltz beta U j0 := by
  have hZpos : 0 < part beta U := part_pos hm beta U
  -- split the partition function into ground states and the rest
  have hsplit : part beta U
      = ∑ j ∈ S, Real.exp (-beta * U j) + ∑ j ∈ Sᶜ, Real.exp (-beta * U j) := by
    rw [part, ← Finset.sum_add_sum_compl S]
  have hS : ∑ j ∈ S, Real.exp (-beta * U j) = (S.card : ℝ) * Real.exp (-beta * Umin) := by
    rw [Finset.sum_congr rfl (fun j hj => by rw [hSmin j hj]), Finset.sum_const, nsmul_eq_mul]
  have hcompl : ∑ j ∈ Sᶜ, Real.exp (-beta * U j)
      ≤ (m : ℝ) * (Real.exp (-beta * Umin) * Real.exp (-beta * gap)) := by
    have hterm : ∀ j ∈ Sᶜ, Real.exp (-beta * U j)
        ≤ Real.exp (-beta * Umin) * Real.exp (-beta * gap) := by
      intro j hj
      have hj' : j ∉ S := by simpa using hj
      rw [← Real.exp_add]
      exact Real.exp_le_exp.2 (by nlinarith [hout j hj'])
    calc ∑ j ∈ Sᶜ, Real.exp (-beta * U j)
        ≤ ∑ _j ∈ Sᶜ, Real.exp (-beta * Umin) * Real.exp (-beta * gap) :=
          Finset.sum_le_sum hterm
      _ = (Sᶜ.card : ℝ) * (Real.exp (-beta * Umin) * Real.exp (-beta * gap)) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (m : ℝ) * (Real.exp (-beta * Umin) * Real.exp (-beta * gap)) := by
          have : (Sᶜ.card : ℝ) ≤ (m : ℝ) := by
            have := Finset.card_le_univ Sᶜ
            simpa using (by exact_mod_cast this : (Sᶜ.card : ℝ) ≤ (Fintype.card (Fin m) : ℝ))
          have hpos : 0 < Real.exp (-beta * Umin) * Real.exp (-beta * gap) := by positivity
          exact mul_le_mul_of_nonneg_right this (le_of_lt hpos)
  have hZle : part beta U
      ≤ ((S.card : ℝ) + m * Real.exp (-beta * gap)) * Real.exp (-beta * Umin) := by
    rw [hsplit, hS]
    nlinarith [hcompl]
  have hden : 0 < (S.card : ℝ) + m * Real.exp (-beta * gap) := by
    have hm' : (0 : ℝ) < m := by exact_mod_cast hm
    have : 0 < (m : ℝ) * Real.exp (-beta * gap) := by positivity
    have : (0 : ℝ) ≤ (S.card : ℝ) := Nat.cast_nonneg _
    positivity
  have hUj0 : U j0 = Umin := hSmin j0 hj0
  rw [boltz, hUj0, div_le_div_iff₀ hden hZpos]
  calc 1 * part beta U ≤ ((S.card : ℝ) + m * Real.exp (-beta * gap)) * Real.exp (-beta * Umin) := by
        rw [one_mul]; exact hZle
    _ = Real.exp (-beta * Umin) * ((S.card : ℝ) + m * Real.exp (-beta * gap)) := by ring

/-- **A gap of `kT·log(number of conformations)` is enough to order the chain.**  If the energy
has a unique minimum and every other conformation lies at least `gap` above it with
`beta·gap ≥ log m`, that conformation holds at least half the population.  Together with
`IDR.FreeEnergy.folded_needs_entropic_gap`, which says such a gap is *necessary*, this makes
`beta·gap ≈ log (number of conformations)` the exact criterion for order. -/
theorem boltz_unique_ground_half (hm : 0 < m) {beta : ℝ} (hbeta : 0 < beta) {U : Fin m → ℝ}
    {Umin gap : ℝ} {j0 : Fin m} (hj0 : U j0 = Umin) (hout : ∀ j, j ≠ j0 → Umin + gap ≤ U j)
    (hgap : Real.log m ≤ beta * gap) : 1 / 2 ≤ boltz beta U j0 := by
  classical
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hbound : (m : ℝ) * Real.exp (-beta * gap) ≤ 1 := by
    have hle : Real.exp (-beta * gap) ≤ 1 / (m : ℝ) := by
      rw [le_div_iff₀ hmR, ← Real.exp_log hmR, ← Real.exp_add]
      exact Real.exp_le_one_iff.2 (by linarith)
    calc (m : ℝ) * Real.exp (-beta * gap) ≤ (m : ℝ) * (1 / (m : ℝ)) :=
          mul_le_mul_of_nonneg_left hle (le_of_lt hmR)
      _ = 1 := by field_simp
  have hmain := boltz_ground_ge (S := ({j0} : Finset (Fin m))) hm hbeta
    (fun j hj => by rw [Finset.mem_singleton.1 hj]; exact hj0)
    (fun j hj => hout j (by simpa using hj)) (Finset.mem_singleton_self j0)
  have hcard : (({j0} : Finset (Fin m)).card : ℝ) = 1 := by simp
  rw [hcard] at hmain
  have hden : (0 : ℝ) < 1 + (m : ℝ) * Real.exp (-beta * gap) := by positivity
  have : (1 : ℝ) / 2 ≤ 1 / (1 + (m : ℝ) * Real.exp (-beta * gap)) := by
    rw [div_le_div_iff₀ (by norm_num) hden]
    linarith
  linarith [hmain]

/-- **Ground-state degeneracy is a capacity requirement.**  A model of at most `k` components
that reproduces the equilibrium ensemble to accuracy `eps` must satisfy
`k ≥ |S| - eps·(|S| + m e^{-beta·gap})`: at low temperature, essentially one component per
ground state.  A rugged landscape with an exponentially degenerate minimum is therefore just as
expensive to model as a flat one. -/
theorem ground_state_capacity [Fintype X] (hm : 0 < m) {beta : ℝ} (hbeta : 0 < beta) {g : Fin m → X}
    (hg : Function.Injective g) {U : Fin m → ℝ} {Umin gap : ℝ} {S : Finset (Fin m)}
    (hSmin : ∀ j ∈ S, U j = Umin) (hout : ∀ j ∉ S, Umin + gap ≤ U j)
    {k : ℕ} {M : Ens X} (hM : M.card ≤ k) {eps : ℝ}
    (h : ApproxSame eps M (boltzEns hm g beta U)) :
    (S.card : ℝ) - eps * ((S.card : ℝ) + m * Real.exp (-beta * gap)) ≤ k := by
  set D : ℝ := (S.card : ℝ) + m * Real.exp (-beta * gap) with hD
  have hden : 0 < D := by
    have hm' : (0 : ℝ) < m := by exact_mod_cast hm
    have h1 : 0 < (m : ℝ) * Real.exp (-beta * gap) := by positivity
    have h2 : (0 : ℝ) ≤ (S.card : ℝ) := Nat.cast_nonneg _
    rw [hD]; positivity
  -- the ground states, enumerated
  set gS : Fin S.card → X := fun i => g ((S.equivFin.symm i : {x // x ∈ S}) : Fin m) with hgS
  have hinj : Function.Injective gS := by
    intro a b hab
    have h1 : ((S.equivFin.symm a : {x // x ∈ S}) : Fin m)
        = ((S.equivFin.symm b : {x // x ∈ S}) : Fin m) := hg hab
    have h2 : S.equivFin.symm a = S.equivFin.symm b := Subtype.ext h1
    simpa using h2
  have hprob : ∀ i : Fin S.card, 1 / D ≤ (boltzEns hm g beta U).prob (gS i) := by
    intro i
    have hmem : ((S.equivFin.symm i : {x // x ∈ S}) : Fin m) ∈ S :=
      (S.equivFin.symm i).2
    rw [hgS]
    rw [boltzEns_prob hm hg]
    exact boltz_ground_ge hm hbeta hSmin hout hmem
  have := Ens.capacity_lower_bound hM hinj (by positivity : (0:ℝ) < 1 / D) hprob h
  have hdiv : eps / (1 / D) = eps * D := by field_simp
  rw [hdiv] at this
  exact this

end IDR.Boltz
