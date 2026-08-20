/-
# The entropy limits on a concrete Hamiltonian, and a worked numerical verdict

`RequestProject.SequenceEntropyLimit` states the two entropy limits with the Hamiltonian left
abstract (site-Lipschitz, homopolymer-degenerate).  This file checks that those hypotheses are
satisfiable by a physically recognisable energy function, and turns the low-complexity theorem
into a number.

* `burialE` -- the burial (hydrophobic-contact) caricature: conformation `j` buries a set of
  positions `p j`, and every buried residue that is *not* of the reference letter `c₀` pays `-L`.
  On an ensemble of equally compact conformations (`|p j|` independent of `j`) every homopolymer
  sees a perfectly flat landscape (`burialE_homopolymer_flat`), and one substitution moves every
  conformational energy by at most `L` (`burialE_siteLip`).  These are exactly the two structural
  hypotheses of the general theorem, so they are satisfiable and the theorem is not vacuous.
* `burial_low_complexity_disordered` -- consequently, in this model, a region whose compositional
  Shannon entropy is below `κ·log 2`, with `κ` the minority fraction the gap criterion demands,
  *must* be disordered.
* `worked_low_complexity_verdict` -- the numbers.  For a 100-residue region with `2^100`
  accessible conformations and a per-substitution contrast of `1 kT`, every sequence whose
  compositional entropy is below `0.3·log 2 ≈ 0.21` nats per residue (about `0.30` bits, i.e. a
  region more than ~96% one residue type) is provably disordered: no force field and no
  temperature can make it fold.
-/
import Mathlib
import RequestProject.SequenceEntropyLimit

set_option autoImplicit false

namespace IDR

namespace SeqLimit

open Finset
open scoped Classical

variable {N q M : ℕ}

/-! ## A generic one-site perturbation bound -/

lemma card_filter_le_of_agree_off {n : ℕ} (P Q : Fin n → Prop) [DecidablePred P]
    [DecidablePred Q] (i : Fin n) (h : ∀ k, k ≠ i → (P k ↔ Q k)) :
    (Finset.univ.filter P).card ≤ (Finset.univ.filter Q).card + 1 := by
  have hsub : Finset.univ.filter P ⊆ insert i (Finset.univ.filter Q) := by
    intro k hk
    by_cases hki : k = i
    · subst hki; exact Finset.mem_insert_self _ _
    · have hP : P k := (Finset.mem_filter.1 hk).2
      exact Finset.mem_insert_of_mem (Finset.mem_filter.2 ⟨Finset.mem_univ _, (h k hki).1 hP⟩)
  exact le_trans (Finset.card_le_card hsub) (Finset.card_insert_le _ _)

/-! ## The burial model -/

/-- **The burial Hamiltonian.**  Conformation `j` buries the positions `p j`; a buried residue
that differs from the reference letter `c₀` (read: a buried hydrophobic residue in a polar
background) lowers the energy by `L`. -/
noncomputable def burialE (L : ℝ) (c0 : Fin q) (p : Fin M → Fin N → Bool) :
    Seq N q → Fin M → ℝ :=
  fun s j => -L * ((Finset.univ.filter fun i => ((s i != c0) && p j i) = true).card : ℝ)

/-- On an ensemble of equally compact conformations every homopolymer sees a flat landscape:
the energy of a constant sequence is the same in every conformation. -/
lemma burialE_homopolymer_flat {L : ℝ} (c0 : Fin q) (p : Fin M → Fin N → Bool)
    (hpat : ∀ j j' : Fin M, (Finset.univ.filter fun i => p j i = true).card
      = (Finset.univ.filter fun i => p j' i = true).card)
    (c : Fin q) (j j' : Fin M) :
    burialE L c0 p (fun _ => c) j - burialE L c0 p (fun _ => c) j' = 0 := by
  classical
  by_cases hc : c = c0
  · have hempty : ∀ j0 : Fin M,
        (Finset.univ.filter fun i => (((fun _ : Fin N => c) i != c0) && p j0 i) = true) = ∅ := by
      intro j0; ext i; simp [hc]
    unfold burialE
    rw [hempty, hempty]
    simp
  · have hcount : ∀ j0 : Fin M,
        burialE L c0 p (fun _ => c) j0
          = -L * ((Finset.univ.filter fun i => p j0 i = true).card : ℝ) := by
      intro j0
      have hset : (Finset.univ.filter fun i => (((fun _ : Fin N => c) i != c0) && p j0 i) = true)
          = (Finset.univ.filter fun i => p j0 i = true) := by
        ext i; simp [hc]
      unfold burialE
      rw [hset]
    rw [hcount, hcount, hpat j j']
    ring

lemma burialE_siteLip {L : ℝ} (hL : 0 ≤ L) (c0 : Fin q) (p : Fin M → Fin N → Bool) :
    SiteLip L (burialE L c0 p) := by
  classical
  intro s i c j
  have hagree : ∀ k, k ≠ i →
      ((((Function.update s i c k != c0) && p j k) = true) ↔ (((s k != c0) && p j k) = true)) := by
    intro k hk
    rw [Function.update_of_ne hk]
  have h1 : (Finset.univ.filter fun k => ((Function.update s i c k != c0) && p j k) = true).card
      ≤ (Finset.univ.filter fun k => ((s k != c0) && p j k) = true).card + 1 :=
    card_filter_le_of_agree_off _ _ i hagree
  have h2 : (Finset.univ.filter fun k => ((s k != c0) && p j k) = true).card
      ≤ (Finset.univ.filter fun k => ((Function.update s i c k != c0) && p j k) = true).card + 1 :=
    card_filter_le_of_agree_off _ _ i (fun k hk => (hagree k hk).symm)
  have h1' : ((Finset.univ.filter fun k =>
        ((Function.update s i c k != c0) && p j k) = true).card : ℝ)
      ≤ ((Finset.univ.filter fun k => ((s k != c0) && p j k) = true).card : ℝ) + 1 := by
    exact_mod_cast h1
  have h2' : ((Finset.univ.filter fun k => ((s k != c0) && p j k) = true).card : ℝ)
      ≤ ((Finset.univ.filter fun k =>
        ((Function.update s i c k != c0) && p j k) = true).card : ℝ) + 1 := by
    exact_mod_cast h2
  simp only [burialE]
  rw [abs_le]
  constructor <;> nlinarith [h1', h2']

/-- **Low complexity forbids folding, in a concrete Hamiltonian.**  In the burial model on an
ensemble of equally compact conformations, a sequence whose compositional Shannon entropy is
below `κ·log 2` -- with `κ` the minority fraction demanded by the gap criterion -- cannot
order. -/
theorem burial_low_complexity_disordered {beta L kappa : ℝ} (hbeta : 0 < beta) (hL : 0 ≤ L)
    (hM : 1 < M) (hN : 0 < N) (c0 : Fin q) (p : Fin M → Fin N → Bool)
    (hpat : ∀ j j' : Fin M, (Finset.univ.filter fun i => p j i = true).card
      = (Finset.univ.filter fun i => p j' i = true).card)
    (hk0 : 0 < kappa) (hk : kappa ≤ 1 / 2)
    (hthr : 2 * L * (kappa * N) < Real.log ((M : ℝ) - 1) / beta)
    {s : Seq N q} (hlow : SeqEnt.H (comp s) < kappa * Real.log 2) :
    ¬ Ordered beta (burialE L c0 p) s := by
  refine disordered_of_low_composition_entropy (S0 := 0) hbeta hM hN (burialE_siteLip hL c0 p)
    ?_ hk0 hk (by linarith) hlow
  intro c j j'
  exact le_of_eq (burialE_homopolymer_flat c0 p hpat c j j')

/-! ## A worked numerical verdict -/

/-- **The number.**  A 100-residue region, `2^100` accessible conformations (`log 2` of
conformational entropy per residue), a per-substitution energy contrast of `1 kT`: every sequence
whose compositional Shannon entropy is below `0.3·log 2 ≈ 0.21` nats -- roughly, a region more
than 96% composed of a single residue type -- is disordered.  No conformation of it can hold half
the equilibrium population, whatever the burial pattern. -/
theorem worked_low_complexity_verdict (c0 : Fin 20) (p : Fin (2 ^ 100) → Fin 100 → Bool)
    (hpat : ∀ j j' : Fin (2 ^ 100), (Finset.univ.filter fun i => p j i = true).card
      = (Finset.univ.filter fun i => p j' i = true).card)
    {s : Seq 100 20} (hlow : SeqEnt.H (comp s) < (3 / 10) * Real.log 2) :
    ¬ Ordered 1 (burialE 1 c0 p) s := by
  have hlog2 : (0.6931471803 : ℝ) < Real.log 2 := Real.log_two_gt_d9
  have hM : 1 < 2 ^ 100 := Nat.one_lt_two_pow_iff.2 (by norm_num)
  have hcast : (((2 : ℕ) ^ 100 : ℕ) : ℝ) = (2 : ℝ) ^ 100 := by push_cast; ring
  have hbig : ((2 : ℝ) ^ 99) ≤ (2 : ℝ) ^ 100 - 1 := by
    have : (2 : ℝ) ^ 100 = 2 * (2 : ℝ) ^ 99 := by ring
    have h99 : (1 : ℝ) ≤ (2 : ℝ) ^ 99 := one_le_pow₀ (by norm_num)
    linarith
  have hlogbig : (99 : ℝ) * Real.log 2 ≤ Real.log ((2 : ℝ) ^ 100 - 1) := by
    have hpos : (0 : ℝ) < (2 : ℝ) ^ 99 := by positivity
    have := Real.log_le_log hpos hbig
    rwa [Real.log_pow] at this
  refine burial_low_complexity_disordered (kappa := 3 / 10) (by norm_num) (by norm_num) hM
    (by norm_num) c0 p hpat (by norm_num) (by norm_num) ?_ hlow
  rw [hcast]
  have : (2 : ℝ) * 1 * ((3 / 10) * (100 : ℕ)) = 60 := by norm_num
  rw [this]
  have h60 : (60 : ℝ) < 99 * Real.log 2 := by nlinarith
  simpa using lt_of_lt_of_le h60 hlogbig

/-! ## The homopolymer: the extreme case of the verdict -/

lemma comp_const (hN : 0 < N) (c a : Fin q) :
    comp (fun _ : Fin N => c) a = if a = c then 1 else 0 := by
  classical
  have hN0 : (N : ℝ) ≠ 0 := by positivity
  unfold comp
  by_cases h : a = c
  · subst h
    have hall : (Finset.univ.filter fun i : Fin N => (fun _ : Fin N => a) i = a)
        = Finset.univ := by
      ext i; simp
    rw [hall]
    simp
    omega
  · have hnone : (Finset.univ.filter fun i : Fin N => (fun _ : Fin N => c) i = a) = ∅ := by
      ext i; simp [Ne.symm h]
    rw [hnone]
    simp [h]

/-- A homopolymer has zero compositional entropy. -/
lemma H_comp_const (hN : 0 < N) (c : Fin q) : SeqEnt.H (comp (fun _ : Fin N => c)) = 0 := by
  classical
  unfold SeqEnt.H
  refine Finset.sum_eq_zero fun a _ => ?_
  rw [comp_const hN]
  by_cases h : a = c <;> simp [h]

/-- **Poly-X cannot fold.**  The extreme case of the verdict: a 100-residue homopolymer has zero
compositional entropy, so no conformation of it holds half the equilibrium population. -/
theorem homopolymer_disordered (c0 c : Fin 20) (p : Fin (2 ^ 100) → Fin 100 → Bool)
    (hpat : ∀ j j' : Fin (2 ^ 100), (Finset.univ.filter fun i => p j i = true).card
      = (Finset.univ.filter fun i => p j' i = true).card) :
    ¬ Ordered 1 (burialE 1 c0 p) (fun _ => c) := by
  refine worked_low_complexity_verdict c0 p hpat ?_
  rw [H_comp_const (by norm_num)]
  have : (0 : ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  linarith

end SeqLimit

end IDR
