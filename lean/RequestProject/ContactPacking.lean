/-
# Where the contact budget comes from: excluded volume caps the multiplicity

`RequestProject.ContactBudget` prices a panel of reported transient contacts against a bound
`N` on how many of them a single conformation can realise.  That bound was left as a
hypothesis.  Here it is derived from the one piece of physics no chain can escape: **two
residues cannot occupy the same place**.

* `packing_card_le` -- a hard-sphere packing bound in three-dimensional space, proved by a grid
  pigeonhole: at most `(2⌈2D/σ⌉ + 1)³` points that are pairwise at least `σ` apart fit inside a
  ball of radius `D`.
* `contact_multiplicity_le` -- consequently, in any conformation of a chain whose residues keep
  a hard-core separation `σ`, at most `(2⌈2D/σ⌉ + 1)³` distinct partners are simultaneously
  within the contact radius `D` of a given residue.
* `hard_core_contact_budget` -- and therefore the population demand of a panel of contacts that
  all report on the *same* residue is capped by that number:
  `∑ₖ (1 - Mₖ/D) ≤ (2⌈2D/σ⌉ + 1)³`, with no assumption on the ensemble beyond excluded volume.
* `hard_core_panel_falsified` -- a panel violating the cap is realised by no ensemble of chains.

This closes the chain of reasoning: an experimental contact panel is converted, using only the
excluded volume of the chain, into an inequality that a proposed ensemble either satisfies or
does not, and that the data themselves may already violate.
-/
import Mathlib
import RequestProject.ContactBudget

namespace RequestProject.ContactPacking

open Finset RequestProject.DistanceRealizability RequestProject.ContactBudget

/-- Coordinatewise closeness implies closeness in three-dimensional space. -/
theorem dist_le_of_coords {x y : EuclideanSpace ℝ (Fin 3)} {s : ℝ} (hs : 0 ≤ s)
    (h : ∀ t, |x t - y t| ≤ s) : dist x y ≤ Real.sqrt 3 * s := by
  rw [EuclideanSpace.dist_eq]
  have hsum : ∑ t : Fin 3, dist (x t) (y t) ^ 2 ≤ 3 * s ^ 2 := by
    have hterm : ∀ t : Fin 3, dist (x t) (y t) ^ 2 ≤ s ^ 2 := by
      intro t
      have ht := h t
      rw [Real.dist_eq]
      nlinarith [abs_nonneg (x t - y t)]
    calc ∑ t : Fin 3, dist (x t) (y t) ^ 2 ≤ ∑ _t : Fin 3, s ^ 2 :=
          Finset.sum_le_sum fun t _ => hterm t
      _ = 3 * s ^ 2 := by simp [Finset.sum_const]
  calc Real.sqrt (∑ t : Fin 3, dist (x t) (y t) ^ 2) ≤ Real.sqrt (3 * s ^ 2) :=
        Real.sqrt_le_sqrt hsum
    _ = Real.sqrt 3 * s := by rw [Real.sqrt_mul (by norm_num), Real.sqrt_sq hs]

/-- **Hard-sphere packing bound.**  At most `(2⌈2D/σ⌉ + 1)³` points that are pairwise at least
`σ` apart fit into a ball of radius `D` in three-dimensional space. -/
theorem packing_card_le {ι : Type*} [DecidableEq ι] (s : Finset ι)
    (x : ι → EuclideanSpace ℝ (Fin 3)) (c : EuclideanSpace ℝ (Fin 3)) {D sigma : ℝ}
    (hsig : 0 < sigma)
    (hsep : ∀ i ∈ s, ∀ j ∈ s, i ≠ j → sigma ≤ dist (x i) (x j))
    (hball : ∀ i ∈ s, dist (x i) c ≤ D) :
    s.card ≤ (2 * ⌈2 * D / sigma⌉₊ + 1) ^ 3 := by
  classical
  set B : ℕ := ⌈2 * D / sigma⌉₊ with hB
  set g : ι → (Fin 3 → ℤ) := fun i t => ⌊(x i t - c t) * (2 / sigma)⌋ with hg
  have hcoord : ∀ i ∈ s, ∀ t : Fin 3, |x i t - c t| ≤ D := by
    intro i hi t
    have h1 : |x i t - c t| ≤ dist (x i) c := by
      simpa [Real.dist_eq] using PiLp.dist_apply_le (x i) c t
    exact le_trans h1 (hball i hi)
  -- the grid map is injective on `s`
  have hinj : Set.InjOn g s := by
    intro i hi j hj hij
    by_contra hne
    have hclose : ∀ t : Fin 3, |x i t - x j t| ≤ sigma / 2 := by
      intro t
      have hfl : ⌊(x i t - c t) * (2 / sigma)⌋ = ⌊(x j t - c t) * (2 / sigma)⌋ :=
        congrFun hij t
      have habs := Int.abs_sub_lt_one_of_floor_eq_floor hfl
      have hxy : |x i t - x j t| * (2 / sigma) < 1 := by
        have : (x i t - c t) * (2 / sigma) - (x j t - c t) * (2 / sigma)
            = (x i t - x j t) * (2 / sigma) := by ring
        rw [this, abs_mul, abs_of_pos (by positivity : (0:ℝ) < 2 / sigma)] at habs
        exact habs
      have h3 := mul_lt_mul_of_pos_right hxy (show (0:ℝ) < sigma / 2 by positivity)
      have h4 : |x i t - x j t| * (2 / sigma) * (sigma / 2) = |x i t - x j t| := by
        field_simp
      rw [h4, one_mul] at h3
      exact le_of_lt h3
    have hd : dist (x i) (x j) ≤ Real.sqrt 3 * (sigma / 2) :=
      dist_le_of_coords (by positivity) hclose
    have hs3 : Real.sqrt 3 < 2 := by
      nlinarith [Real.sq_sqrt (by norm_num : (3:ℝ) ≥ 0), Real.sqrt_nonneg 3]
    have := hsep i hi j hj hne
    nlinarith
  -- and lands in a cube of side `2B + 1`
  have hmem : ∀ i ∈ s, g i ∈ Fintype.piFinset (fun _ : Fin 3 => Finset.Icc (-(B:ℤ)) (B:ℤ)) := by
    intro i hi
    rw [Fintype.mem_piFinset]
    intro t
    have hb : |x i t - c t| ≤ D := hcoord i hi t
    have hle : |(x i t - c t) * (2 / sigma)| ≤ 2 * D / sigma := by
      rw [abs_mul, abs_of_pos (by positivity : (0:ℝ) < 2 / sigma)]
      calc |x i t - c t| * (2 / sigma) ≤ D * (2 / sigma) :=
            mul_le_mul_of_nonneg_right hb (by positivity)
        _ = 2 * D / sigma := by ring
    have hBle : 2 * D / sigma ≤ (B : ℝ) := Nat.le_ceil _
    have hup : (x i t - c t) * (2 / sigma) ≤ (B : ℝ) :=
      le_trans (le_trans (le_abs_self _) hle) hBle
    have hlo : -((B : ℝ)) ≤ (x i t - c t) * (2 / sigma) := by
      have h2 : -(2 * D / sigma) ≤ (x i t - c t) * (2 / sigma) := (abs_le.mp hle).1
      linarith
    rw [Finset.mem_Icc]
    constructor
    · rw [Int.le_floor]; push_cast; exact hlo
    · rw [Int.floor_le_iff]; push_cast; linarith
  calc s.card ≤ (Fintype.piFinset (fun _ : Fin 3 => Finset.Icc (-(B:ℤ)) (B:ℤ))).card :=
        Finset.card_le_card_of_injOn g hmem hinj
    _ = (2 * B + 1) ^ 3 := by
        rw [Fintype.card_piFinset]
        simp [Int.card_Icc]
        ring_nf
        omega

/-- **Excluded volume caps how many contacts one conformation can realise.**  In a conformation
whose distinct residues are at least `sigma` apart, at most `(2⌈2D/σ⌉ + 1)³` of a family of
distinct partners lie within the contact radius `D` of a fixed residue. -/
theorem contact_multiplicity_le {r : ℕ} (y : ℕ → EuclideanSpace ℝ (Fin 3)) (i : ℕ)
    (q : Fin r → ℕ) (hq : Function.Injective q) {D sigma : ℝ} (hsig : 0 < sigma)
    (hcore : ∀ p p' : ℕ, p ≠ p' → sigma ≤ dist (y p) (y p')) :
    ((Finset.univ.filter (fun k : Fin r => dist (y i) (y (q k)) ≤ D)).card : ℝ)
      ≤ ((2 * ⌈2 * D / sigma⌉₊ + 1) ^ 3 : ℕ) := by
  classical
  have h := packing_card_le (Finset.univ.filter (fun k : Fin r => dist (y i) (y (q k)) ≤ D))
    (fun k => y (q k)) (y i) hsig
    (fun k _ l _ hkl => hcore (q k) (q l) fun h => hkl (hq h))
    (fun k hk => by
      rw [dist_comm]
      simpa using (Finset.mem_filter.mp hk).2)
  exact_mod_cast h

/-- **The hard-core contact budget.**  For a panel of contacts all reporting on the same
residue `i`, with measured mean distances `M k`, contact radius `D` and hard-core separation
`sigma`, every ensemble of chain conformations satisfies
`∑ₖ (1 - M k / D) ≤ (2⌈2D/σ⌉ + 1)³`. -/
theorem hard_core_contact_budget {m r : ℕ} {w : Fin m → ℝ}
    {X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)} (i : ℕ) (q : Fin r → ℕ)
    (hq : Function.Injective q) {D sigma : ℝ} {M : Fin r → ℝ}
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1) (hD : 0 < D) (hsig : 0 < sigma)
    (hcore : ∀ (a : Fin m) (p p' : ℕ), p ≠ p' → sigma ≤ dist (X a p) (X a p'))
    (hM : ∀ k, meanDist w X i (q k) ≤ M k) :
    ∑ k, (1 - M k / D) ≤ ((2 * ⌈2 * D / sigma⌉₊ + 1) ^ 3 : ℕ) := by
  classical
  refine contact_budget (X := X) (fun _ => i) q hw hsum hD hM ?_
  intro a
  have := contact_multiplicity_le (D := D) (X a) i q hq hsig (hcore a)
  refine le_trans (le_of_eq ?_) this
  congr 1
  apply Finset.card_nbij id <;> simp [contactSet, Set.InjOn, Set.SurjOn, Set.MapsTo]

/-- **Falsification from excluded volume alone.**  A contact panel on one residue whose
population demand exceeds the packing cap is realised by no ensemble of chains. -/
theorem hard_core_panel_falsified {r : ℕ} (i : ℕ) (q : Fin r → ℕ) (hq : Function.Injective q)
    {D sigma : ℝ} {M : Fin r → ℝ} (hD : 0 < D) (hsig : 0 < sigma)
    (hover : ((2 * ⌈2 * D / sigma⌉₊ + 1) ^ 3 : ℕ) < ∑ k, (1 - M k / D)) :
    ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
        (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
        (∀ (a : Fin m) (p p' : ℕ), p ≠ p' → sigma ≤ dist (X a p) (X a p')) ∧
        (∀ k, meanDist w X i (q k) ≤ M k) := by
  rintro ⟨m, w, X, hw, hsum, hcore, hM⟩
  exact absurd (hard_core_contact_budget i q hq hw hsum hD hsig hcore hM) (not_le.mpr hover)

end RequestProject.ContactPacking
