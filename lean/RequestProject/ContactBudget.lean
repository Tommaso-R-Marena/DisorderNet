/-
# The contact budget: how many populated contacts an ensemble can afford

A transient contact reported for a disordered region -- a PRE broadening, an NOE, a
crosslink -- is an *ensemble average*.  The standard reading is "residues `p` and `q` are in
contact part of the time", and models are built by piling many such contacts into one
ensemble.  This file prices that practice.

Two elementary facts combine into a hard constraint.

* `contactPop_ge` (a Markov bound).  If the measured mean distance of the pair `(p, q)` is `M`,
  then the fraction of the ensemble in which the pair is actually within `D` is at least
  `1 - M / D`.  A short measured mean *forces* population; it cannot be explained away as a rare
  close encounter.
* `sum_contactPop_le_multiplicity` (a counting bound).  If no single conformation can satisfy
  more than `N` of the reported contacts -- for steric reasons, or because the contacts compete
  for the same partner -- then the populations of the reported contacts sum to at most `N`.

Together (`contact_budget`, `contact_panel_falsified`) they give an assumption-free
falsification: a panel of contacts with

  `∑ₖ (1 - Mₖ / D) > N`

is realised by **no** ensemble whatsoever.  The individual restraints may each be perfectly
reasonable; it is their *joint* population demand that is impossible.  `exclusive_of_hard_core`
supplies the geometric input in the simplest realistic case (two contacts competing for the
same partner, with an excluded-volume separation), and `min_ensemble_size` turns the same
counting into a lower bound on the number of distinct conformations a model must contain:
`r` mutually exclusive populated contacts require at least `r` conformers, so the size of the
ensemble is dictated by the data, not chosen for convenience.
-/
import Mathlib
import RequestProject.DistanceRealizability

namespace RequestProject.ContactBudget

open Finset RequestProject.DistanceRealizability

variable {E : Type*} [PseudoMetricSpace E] {m : ℕ}

/-- The set of conformations of the ensemble in which the labelled pair `(p, q)` is within the
contact radius `D`. -/
noncomputable def contactSet (X : Fin m → ℕ → E) (p q : ℕ) (D : ℝ) : Finset (Fin m) := by
  classical
  exact Finset.univ.filter (fun a => dist (X a p) (X a q) ≤ D)

/-- The population (total weight) of the contact `(p, q)` in the ensemble. -/
noncomputable def contactPop (w : Fin m → ℝ) (X : Fin m → ℕ → E) (p q : ℕ) (D : ℝ) : ℝ :=
  ∑ a ∈ contactSet X p q D, w a

variable {w : Fin m → ℝ} {X : Fin m → ℕ → E}

theorem contactPop_nonneg (hw : ∀ a, 0 ≤ w a) (p q : ℕ) (D : ℝ) :
    0 ≤ contactPop w X p q D :=
  Finset.sum_nonneg fun a _ => hw a

/-- **Markov bound: a short measured mean distance forces population.**  If the ensemble-mean
distance of the pair `(p, q)` is at most `M`, then a fraction at least `1 - M / D` of the
ensemble has that pair within `D`. -/
theorem contactPop_ge (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1) {p q : ℕ} {D M : ℝ}
    (hD : 0 < D) (hM : meanDist w X p q ≤ M) :
    1 - M / D ≤ contactPop w X p q D := by
  classical
  have hsplit : ∑ a ∈ (contactSet X p q D)ᶜ, w a = 1 - contactPop w X p q D := by
    have := Finset.sum_add_sum_compl (contactSet X p q D) w
    rw [contactPop]
    linarith [this, hsum]
  have hlow : D * (1 - contactPop w X p q D) ≤ meanDist w X p q := by
    rw [← hsplit, Finset.mul_sum, meanDist]
    calc ∑ a ∈ (contactSet X p q D)ᶜ, D * w a
        ≤ ∑ a ∈ (contactSet X p q D)ᶜ, w a * dist (X a p) (X a q) := by
          refine Finset.sum_le_sum fun a ha => ?_
          have hnot : ¬ dist (X a p) (X a q) ≤ D := by
            simpa [contactSet] using (Finset.mem_compl.mp ha)
          rw [mul_comm]
          exact mul_le_mul_of_nonneg_left (le_of_lt (not_le.mp hnot)) (hw a)
      _ ≤ ∑ a, w a * dist (X a p) (X a q) :=
          Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
            (fun a _ _ => mul_nonneg (hw a) dist_nonneg)
  have hle : D * (1 - contactPop w X p q D) ≤ M := le_trans hlow hM
  have hdiv : 1 - contactPop w X p q D ≤ M / D := by
    rw [le_div_iff₀ hD, mul_comm]; exact hle
  linarith

/-- **Counting bound.**  If no conformation of the ensemble realises more than `N` of the
reported contacts, then the reported populations sum to at most `N`. -/
theorem sum_contactPop_le_multiplicity {r : ℕ} (p q : Fin r → ℕ) {D : ℝ} {N : ℝ}
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hmult : ∀ a : Fin m, ((Finset.univ.filter
      (fun k : Fin r => a ∈ contactSet X (p k) (q k) D)).card : ℝ) ≤ N) :
    ∑ k, contactPop w X (p k) (q k) D ≤ N := by
  classical
  have hswap : ∑ k, contactPop w X (p k) (q k) D
      = ∑ a : Fin m, w a *
          ((Finset.univ.filter (fun k : Fin r => a ∈ contactSet X (p k) (q k) D)).card : ℝ) := by
    unfold contactPop
    rw [Finset.sum_comm']
    · refine Finset.sum_congr rfl fun a _ => ?_
      rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
    · intro k a
      simp
  rw [hswap]
  calc ∑ a : Fin m, w a * ((Finset.univ.filter
        (fun k : Fin r => a ∈ contactSet X (p k) (q k) D)).card : ℝ)
      ≤ ∑ a : Fin m, w a * N :=
        Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hmult a) (hw a)
    _ = N := by rw [← Finset.sum_mul, hsum, one_mul]

/-- **The contact budget.**  Measured mean distances `M k` for `r` reported contacts, a contact
radius `D`, and a bound `N` on how many of the contacts one conformation can realise, force

  `∑ₖ (1 - M k / D) ≤ N`. -/
theorem contact_budget {r : ℕ} (p q : Fin r → ℕ) {D N : ℝ} {M : Fin r → ℝ}
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1) (hD : 0 < D)
    (hM : ∀ k, meanDist w X (p k) (q k) ≤ M k)
    (hmult : ∀ a : Fin m, ((Finset.univ.filter
      (fun k : Fin r => a ∈ contactSet X (p k) (q k) D)).card : ℝ) ≤ N) :
    ∑ k, (1 - M k / D) ≤ N := by
  refine le_trans (Finset.sum_le_sum fun k _ => contactPop_ge hw hsum hD (hM k)) ?_
  exact sum_contactPop_le_multiplicity p q hw hsum hmult

/-- **Falsification.**  A contact panel whose population demand exceeds the multiplicity budget
is realised by no ensemble at all. -/
theorem contact_panel_falsified {r : ℕ} (p q : Fin r → ℕ) {D N : ℝ} {M : Fin r → ℝ}
    (hD : 0 < D) (hover : N < ∑ k, (1 - M k / D)) :
    ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → E),
        (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
        (∀ k, meanDist w X (p k) (q k) ≤ M k) ∧
        (∀ a : Fin m, ((Finset.univ.filter
          (fun k : Fin r => a ∈ contactSet X (p k) (q k) D)).card : ℝ) ≤ N) := by
  rintro ⟨m, w, X, hw, hsum, hM, hmult⟩
  exact absurd (contact_budget p q hw hsum hD hM hmult) (not_le.mpr hover)

/-- **Excluded volume makes competing contacts exclusive.**  If two reported contacts compete
for the same partner `i` and the two partners `j`, `k` are kept more than `2 D` apart by the
hard core of the chain, no conformation realises both. -/
theorem exclusive_of_hard_core {x : ℕ → E} {i j k : ℕ} {D sigma : ℝ}
    (hsep : sigma ≤ dist (x j) (x k)) (hbig : 2 * D < sigma)
    (h1 : dist (x i) (x j) ≤ D) (h2 : dist (x i) (x k) ≤ D) : False := by
  have := dist_triangle (x j) (x i) (x k)
  rw [dist_comm (x j) (x i)] at this
  linarith

/-- **The data set the ensemble size.**  If the reported contacts are pairwise exclusive
(no conformation realises two of them) and each is forced to be populated, then the ensemble
needs at least as many conformations as there are contacts. -/
theorem min_ensemble_size {r : ℕ} (p q : Fin r → ℕ) {D : ℝ} {M : Fin r → ℝ}
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1) (hD : 0 < D)
    (hM : ∀ k, meanDist w X (p k) (q k) ≤ M k) (hpos : ∀ k, M k < D)
    (hexcl : ∀ (a : Fin m) (k l : Fin r), k ≠ l →
      a ∈ contactSet X (p k) (q k) D → a ∉ contactSet X (p l) (q l) D) :
    r ≤ m := by
  classical
  -- each contact has a nonempty conformation set, and the sets are pairwise disjoint
  have hne : ∀ k, (contactSet X (p k) (q k) D).Nonempty := by
    intro k
    by_contra h
    rw [Finset.not_nonempty_iff_eq_empty] at h
    have hzero : contactPop w X (p k) (q k) D = 0 := by simp [contactPop, h]
    have := contactPop_ge hw hsum hD (hM k)
    rw [hzero] at this
    have : 1 ≤ M k / D := by linarith
    rw [le_div_iff₀ hD, one_mul] at this
    linarith [hpos k]
  choose f hf using hne
  have hinj : Function.Injective f := by
    intro k l hkl
    by_contra hne'
    exact hexcl (f k) k l hne' (hf k) (by rw [hkl]; exact hf l)
  calc r = Fintype.card (Fin r) := (Fintype.card_fin r).symm
    _ ≤ Fintype.card (Fin m) := Fintype.card_le_of_injective f hinj
    _ = m := Fintype.card_fin m

end RequestProject.ContactBudget
