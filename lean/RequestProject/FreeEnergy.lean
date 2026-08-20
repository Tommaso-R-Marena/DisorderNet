/-
# Part IV.1  The variational principle: what a disorder model is *trained* to minimise

`RequestProject.MaxEnt` fits an ensemble to experimental restraints.  This file treats the
other half of ensemble modelling -- fitting an ensemble to a *force field* -- and proves
the statement that makes variational training of a generative model of a disordered region
legitimate.

Over a conformational library `Fin n` with energies `U` and inverse temperature `beta`:

* `freeEnergy beta U p = ⟨U⟩_p - S[p]/beta` is the variational (Helmholtz) free energy of a
  candidate ensemble `p`: mean energy minus temperature times conformational entropy.
* `freeEnergy_eq` is the **exact decomposition**
  `F[p] = -(1/beta)·log Z + (1/beta)·KL(p‖Boltzmann)`.
  Everything else follows from it:
  - `freeEnergy_ge`, `freeEnergy_boltz` -- the **Gibbs variational principle**: the free
    energy is minimised exactly at the Boltzmann ensemble, whose value is `-(1/β) log Z`;
  - `freeEnergy_min_unique` -- and at no other ensemble;
  - `freeEnergy_gap` -- the *excess* free energy of a model is exactly `1/β` times its
    relative entropy to the truth.  So minimising variational free energy over any class of
    models is **the same optimisation** as minimising KL divergence to the true ensemble:
    variational training introduces no bias of its own, only the bias of the model class.
* `folded_needs_entropic_gap` -- the physical reason intrinsically disordered regions are
  disordered, as a theorem.  If a single conformation is to carry at least half of the
  population against `D` competing conformations of energy at most `Uu`, its energy must
  beat theirs by at least `(1/β)·log |D|`: **the energy gap must pay for the conformational
  entropy**.  A region whose accessible states are exponentially many in its length can be
  folded only by an energy gap linear in its length; absent that gap it *must* be modelled
  as a broad ensemble.
* `flat_landscape_uniform` -- the extreme case: a flat energy landscape gives exactly the
  uniform ensemble over the library, the maximally disordered target.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.EnergyModels
import RequestProject.Metric

namespace IDR

open Finset
open scoped Classical

namespace FreeEnergy

variable {n : ℕ}

/-- The partition function of a library of energies at inverse temperature `beta`. -/
noncomputable def part (beta : ℝ) (U : Fin n → ℝ) : ℝ := ∑ j, Real.exp (-beta * U j)

/-- The Boltzmann weights on the library. -/
noncomputable def boltz (beta : ℝ) (U : Fin n → ℝ) : Fin n → ℝ :=
  fun j => Real.exp (-beta * U j) / part beta U

/-- The Gibbs--Shannon entropy of a weight vector. -/
noncomputable def shannon (p : Fin n → ℝ) : ℝ := ∑ j, -(p j * Real.log (p j))

/-- The variational (Helmholtz) free energy of a candidate ensemble `p`: mean energy minus
temperature times conformational entropy. -/
noncomputable def freeEnergy (beta : ℝ) (U p : Fin n → ℝ) : ℝ :=
  (∑ j, p j * U j) - shannon p / beta

lemma part_pos (hn : 0 < n) (beta : ℝ) (U : Fin n → ℝ) : 0 < part beta U := by
  have : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  refine Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨⟨0, hn⟩, Finset.mem_univ _⟩

lemma boltz_pos (hn : 0 < n) (beta : ℝ) (U : Fin n → ℝ) (j : Fin n) : 0 < boltz beta U j :=
  div_pos (Real.exp_pos _) (part_pos hn beta U)

lemma boltz_sum_one (hn : 0 < n) (beta : ℝ) (U : Fin n → ℝ) : ∑ j, boltz beta U j = 1 := by
  simp only [boltz, ← Finset.sum_div]
  exact div_self (ne_of_gt (part_pos hn beta U))

lemma log_boltz (hn : 0 < n) (beta : ℝ) (U : Fin n → ℝ) (j : Fin n) :
    Real.log (boltz beta U j) = -beta * U j - Real.log (part beta U) := by
  rw [boltz, Real.log_div (ne_of_gt (Real.exp_pos _)) (ne_of_gt (part_pos hn beta U)),
    Real.log_exp]

/-- **The free energy is the Boltzmann free energy plus a relative entropy.**  For every
candidate ensemble `p` on the library,
`F[p] = -(1/β)·log Z + (1/β)·KL(p‖Boltzmann)`. -/
theorem freeEnergy_eq (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) {U p : Fin n → ℝ}
    (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1) :
    freeEnergy beta U p
      = -(Real.log (part beta U)) / beta + klDiv p (boltz beta U) / beta := by
  have hkl : klDiv p (boltz beta U)
      = ∑ j, (p j * Real.log (p j) + beta * (p j * U j) + p j * Real.log (part beta U)) := by
    refine Finset.sum_congr rfl fun j _ => ?_
    rcases eq_or_lt_of_le (hp j) with h0 | hpos
    · simp [← h0]
    · rw [Real.log_div (ne_of_gt hpos) (ne_of_gt (boltz_pos hn beta U j)),
        log_boltz hn beta U j]
      ring
  rw [hkl]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, hps]
  have hS : shannon p = -∑ j, p j * Real.log (p j) := by
    simp [shannon, Finset.sum_neg_distrib]
  rw [freeEnergy, hS]
  field_simp
  ring

/-- The free energy of the Boltzmann ensemble is `-(1/β)·log Z`. -/
theorem freeEnergy_boltz (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) (U : Fin n → ℝ) :
    freeEnergy beta U (boltz beta U) = -(Real.log (part beta U)) / beta := by
  have hkl : klDiv (boltz beta U) (boltz beta U) = 0 := by
    refine Finset.sum_eq_zero fun j _ => ?_
    rw [div_self (ne_of_gt (boltz_pos hn beta U j))]
    simp
  rw [freeEnergy_eq hn hbeta (fun j => le_of_lt (boltz_pos hn beta U j))
    (boltz_sum_one hn beta U), hkl]
  simp

/-- **The excess free energy is exactly the relative entropy to the truth.**  Minimising
the variational free energy over a class of models is therefore the *same* optimisation as
minimising the KL divergence to the true Boltzmann ensemble: variational training adds no
bias of its own. -/
theorem freeEnergy_gap (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) {U p : Fin n → ℝ}
    (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1) :
    freeEnergy beta U p - freeEnergy beta U (boltz beta U) = klDiv p (boltz beta U) / beta := by
  rw [freeEnergy_eq hn hbeta hp hps, freeEnergy_boltz hn hbeta]
  ring

/-- **Gibbs' variational principle.**  No ensemble has lower free energy than the Boltzmann
ensemble. -/
theorem freeEnergy_ge (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) {U p : Fin n → ℝ}
    (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1) :
    freeEnergy beta U (boltz beta U) ≤ freeEnergy beta U p := by
  have hkl := klDiv_nonneg hp hps (fun j => boltz_pos hn beta U j) (boltz_sum_one hn beta U)
  have := freeEnergy_gap (U := U) hn hbeta hp hps
  have hdiv : 0 ≤ klDiv p (boltz beta U) / beta := div_nonneg hkl (le_of_lt hbeta)
  linarith

/-- **And the Boltzmann ensemble is the only minimiser.**  A model achieving the minimal
free energy *is* the true ensemble; so a variational fit that reaches the optimum is
correct on every observable. -/
theorem freeEnergy_min_unique (hn : 0 < n) {beta : ℝ} (hbeta : 0 < beta) {U p : Fin n → ℝ}
    (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hmin : freeEnergy beta U p ≤ freeEnergy beta U (boltz beta U)) :
    p = boltz beta U := by
  have hgap := freeEnergy_gap (U := U) hn hbeta hp hps
  have hkl : klDiv p (boltz beta U) ≤ 0 := by
    have hdiv : klDiv p (boltz beta U) / beta ≤ 0 := by linarith
    have h2 : klDiv p (boltz beta U) / beta * beta ≤ 0 * beta :=
      mul_le_mul_of_nonneg_right hdiv (le_of_lt hbeta)
    rw [div_mul_cancel₀ _ (ne_of_gt hbeta)] at h2
    linarith
  have hkl0 : klDiv p (boltz beta U) = 0 :=
    le_antisymm hkl
      (klDiv_nonneg hp hps (fun j => boltz_pos hn beta U j) (boltz_sum_one hn beta U))
  exact (klDiv_eq_zero_iff hp hps (fun j => boltz_pos hn beta U j)
    (boltz_sum_one hn beta U)).1 hkl0

/-! ## Why disordered regions stay disordered: entropy has to be paid for -/

/-- **The energy gap must pay for the conformational entropy.**  If a single conformation
`j₀` carries at least half of the Boltzmann population, while `D` other conformations all
have energy at most `Uu`, then

`  Uu - U j₀  ≥  (1/β)·log |D| .`

For a region with exponentially many accessible conformations this is an energy gap
*linear in the length of the region*.  Where the force field provides no such gap -- the
defining situation of an intrinsically disordered region -- no single conformation can hold
half the population, and the target of prediction is irreducibly an ensemble. -/
theorem folded_needs_entropic_gap {beta : ℝ} (hbeta : 0 < beta) (hn : 0 < n)
    (U : Fin n → ℝ) (j0 : Fin n) (D : Finset (Fin n)) (hD : D.Nonempty) (hj0 : j0 ∉ D)
    (Uu : ℝ) (hU : ∀ j ∈ D, U j ≤ Uu) (hhalf : 1 / 2 ≤ boltz beta U j0) :
    Real.log D.card / beta ≤ Uu - U j0 := by
  have hZ : 0 < part beta U := part_pos hn beta U
  -- from `boltz j₀ ≥ 1/2`: the folded weight beats the sum of all the rest
  have hge : part beta U ≤ 2 * Real.exp (-beta * U j0) := by
    have hhalf' : (1 : ℝ) / 2 ≤ Real.exp (-beta * U j0) / part beta U := hhalf
    have h1 : (1 : ℝ) / 2 * part beta U ≤ Real.exp (-beta * U j0) :=
      (le_div_iff₀ hZ).1 hhalf'
    linarith
  have hsplit : Real.exp (-beta * U j0) + ∑ j ∈ D, Real.exp (-beta * U j) ≤ part beta U := by
    have hins : insert j0 D ⊆ (Finset.univ : Finset (Fin n)) := Finset.subset_univ _
    have hsum : ∑ j ∈ insert j0 D, Real.exp (-beta * U j) ≤ part beta U :=
      Finset.sum_le_sum_of_subset_of_nonneg hins (fun j _ _ => le_of_lt (Real.exp_pos _))
    rwa [Finset.sum_insert hj0] at hsum
  have hDsum : (D.card : ℝ) * Real.exp (-beta * Uu) ≤ ∑ j ∈ D, Real.exp (-beta * U j) := by
    have : ∀ j ∈ D, Real.exp (-beta * Uu) ≤ Real.exp (-beta * U j) := by
      intro j hj
      exact Real.exp_le_exp.2 (by nlinarith [hU j hj])
    calc (D.card : ℝ) * Real.exp (-beta * Uu) = ∑ _j ∈ D, Real.exp (-beta * Uu) := by
          rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ ∑ j ∈ D, Real.exp (-beta * U j) := Finset.sum_le_sum this
  have hkey : (D.card : ℝ) * Real.exp (-beta * Uu) ≤ Real.exp (-beta * U j0) := by
    linarith
  have hcardpos : (0 : ℝ) < D.card := by
    exact_mod_cast Finset.card_pos.2 hD
  -- take logarithms
  have hlog : Real.log ((D.card : ℝ) * Real.exp (-beta * Uu)) ≤ Real.log (Real.exp (-beta * U j0)) :=
    Real.log_le_log (by positivity) hkey
  rw [Real.log_mul (ne_of_gt hcardpos) (ne_of_gt (Real.exp_pos _)), Real.log_exp, Real.log_exp]
    at hlog
  rw [div_le_iff₀ hbeta]
  nlinarith [hlog]

/-- A flat energy landscape gives exactly the uniform ensemble: maximal disorder. -/
theorem flat_landscape_uniform (hn : 0 < n) (beta : ℝ) (c : ℝ) :
    boltz beta (fun _ : Fin n => c) = fun _ => 1 / (n : ℝ) := by
  funext j
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hpart : part beta (fun _ : Fin n => c) = (n : ℝ) * Real.exp (-beta * c) := by
    simp [part, Finset.sum_const, nsmul_eq_mul]
  rw [boltz, hpart]
  field_simp

end FreeEnergy

end IDR
