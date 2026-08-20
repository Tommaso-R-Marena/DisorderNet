/-
# Part XCIV.3  Test regions, not residues: where the harmonic price is paid

Part XCIV proves that under arbitrary dependence a screen must deflate its level by the harmonic
number of the number of tests, and Part XCIV.2 that the factor cannot be lowered.  The factor is
therefore a design parameter: it is fixed by *how many hypotheses the screen states*, and a screen
for intrinsically disordered regions has a choice.  Disorder is a property of a region, and an
annotation is constant along it (the block-constant error model already used in
`RequestProject.PairwiseProtocol`); a screen may state one hypothesis per residue, or one per
candidate region.

This file shows the two screens report the *same* false discovery proportion, and hence that the
region-level screen is uniformly better: identical output, smaller correction.

* `Blocks` — a partition of the `n` residues into `M` candidate regions of `b` residues each
  (disjointness and common size are all that is used).
* `lift` — a set of regions read as the set of residues it covers; `card_lift`, `lift_inter`.
* `fdp_lift` — **the invariance**: the residue-level false discovery proportion of the lifted list
  equals the region-level one.  Both numerator and denominator are multiplied by the block size,
  which cancels.  Note this is exactly the point at which region-level annotation matters: it is
  true because the truth and the report are both unions of whole regions.
* `region_screen_fdr_le` — so the region-level Benjamini–Yekutieli procedure, run at level
  `α/H_M`, controls the residue-level false discovery rate at `α` under an arbitrary joint law.
* `region_level_is_less_conservative` — and `α/H_n ≤ α/H_M`: the region screen tests at a strictly
  more permissive threshold.
* `harmonic_gain_log` — quantitatively, the saving is `H_n − H_M ≥ log b − 1`: one nat per factor
  in the block size, so aggregating residues into regions of length `b` buys back a `log b` of the
  correction.  Nothing is lost by doing so, by `fdp_lift`.

The scope note of Part XCIV applies unchanged: this is multiplicity control only, and it assumes
the region-level p-values are valid.  What this file adds is that the *unit of testing* is part of
the design, and that the correct unit for a disorder screen is the region.
-/
import Mathlib
import RequestProject.DependentScreen
import RequestProject.DependentBH

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR
namespace RegionScreen

open IDR.DepScreen IDR.DepBH

variable {Ω : Type*} [Fintype Ω] {n M b : ℕ}

/-! ## 1. Regions -/

/-- A partition of the residues into candidate regions of equal length. -/
structure Blocks (n M b : ℕ) where
  /-- the residues covered by region `j` -/
  blk : Fin M → Finset (Fin n)
  card_eq : ∀ j, (blk j).card = b
  pairwiseDisjoint : ∀ j k, j ≠ k → Disjoint (blk j) (blk k)

/-- The residues covered by a set of regions. -/
def lift (B : Blocks n M b) (S : Finset (Fin M)) : Finset (Fin n) := S.biUnion B.blk

lemma card_lift (B : Blocks n M b) (S : Finset (Fin M)) : (lift B S).card = b * S.card := by
  classical
  unfold lift
  rw [Finset.card_biUnion (fun j _ k _ hjk => B.pairwiseDisjoint j k hjk)]
  rw [Finset.sum_congr rfl (fun j _ => B.card_eq j), Finset.sum_const, smul_eq_mul, mul_comm]

lemma lift_inter (B : Blocks n M b) (S T : Finset (Fin M)) :
    lift B S ∩ lift B T = lift B (S ∩ T) := by
  classical
  ext x
  simp only [lift, Finset.mem_inter, Finset.mem_biUnion]
  constructor
  · rintro ⟨⟨j, hjS, hxj⟩, ⟨k, hkT, hxk⟩⟩
    have hjk : j = k := by
      by_contra h
      exact (Finset.disjoint_left.mp (B.pairwiseDisjoint j k h) hxj) hxk
    subst hjk
    exact ⟨j, ⟨hjS, hkT⟩, hxj⟩
  · rintro ⟨j, ⟨hjS, hjT⟩, hxj⟩
    exact ⟨⟨j, hjS, hxj⟩, ⟨j, hjT, hxj⟩⟩

/-- The standard partition of `M·b` residues into `M` consecutive regions of `b` residues, so the
hypotheses below are satisfiable and the results are not vacuous. -/
def stdBlocks (M b : ℕ) (hb : 0 < b) : Blocks (M * b) M b where
  blk := fun j => Finset.univ.filter (fun i : Fin (M * b) => (i : ℕ) / b = (j : ℕ))
  card_eq := by
    classical
    intro j
    have hbound : ∀ r : Fin b, (b * (j : ℕ) + (r : ℕ)) < M * b := by
      intro r
      have hj : (j : ℕ) + 1 ≤ M := j.2
      have hr := r.2
      nlinarith
    have h := Finset.card_bij'
      (s := Finset.univ.filter (fun i : Fin (M * b) => (i : ℕ) / b = (j : ℕ)))
      (t := (Finset.univ : Finset (Fin b)))
      (fun i _ => (⟨(i : ℕ) % b, Nat.mod_lt _ hb⟩ : Fin b))
      (fun r _ => (⟨b * (j : ℕ) + (r : ℕ), hbound r⟩ : Fin (M * b)))
      (fun i _ => Finset.mem_univ _)
      (fun r _ => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        rw [Nat.mul_add_div hb, Nat.div_eq_of_lt r.2, Nat.add_zero])
      (fun i hi => by
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi
        apply Fin.ext
        simp only
        rw [← hi, Nat.div_add_mod])
      (fun r _ => by
        apply Fin.ext
        simp only
        rw [Nat.mul_add_mod, Nat.mod_eq_of_lt r.2])
    simpa using h
  pairwiseDisjoint := by
    classical
    intro j k hjk
    rw [Finset.disjoint_left]
    intro i hij hik
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hij hik
    exact hjk (Fin.ext (by rw [← hij, ← hik]))

/-! ## 2. The two screens report the same false discovery proportion -/

omit [Fintype Ω] in
/-- **The invariance.**  Reading a region-level report at residue level multiplies both the number
of false discoveries and the size of the list by the region length, so the false discovery
proportion is unchanged. -/
lemma fdp_lift (hb : 0 < b) (B : Blocks n M b) (H0 : Finset (Fin M))
    (R : Ω → Finset (Fin M)) (ω : Ω) :
    fdp (lift B H0) (fun ω => lift B (R ω)) ω = fdp H0 R ω := by
  classical
  unfold fdp
  rw [lift_inter B (R ω) H0, card_lift, card_lift]
  by_cases h : (R ω).card = 0
  · rw [if_pos h, if_pos (by simp [h])]
  · have hbc : b * (R ω).card ≠ 0 := by
      have := Nat.pos_of_ne_zero h
      positivity
    rw [if_neg h, if_neg hbc]
    have hbR : (0 : ℝ) < b := by exact_mod_cast hb
    have hcR : (0 : ℝ) < ((R ω).card : ℝ) := by
      have := Nat.pos_of_ne_zero h
      exact_mod_cast this
    push_cast
    field_simp

/-! ## 3. The region-level screen -/

/-- **The region-level Benjamini–Yekutieli screen.**  Stating one hypothesis per candidate region
and running the corrected procedure at level `α/H_M` controls the false discovery rate of the
residue-level report at `α`, under an arbitrary joint law. -/
theorem region_screen_fdr_le (P : Law Ω) {α : ℝ} (hα : 0 ≤ α) (hM : 0 < M) (hb : 0 < b)
    (B : Blocks n M b) {p : Fin M → Ω → ℝ} (H0 : Finset (Fin M))
    (hnull : ∀ i ∈ H0, Superuniform P (p i)) :
    P.mean (fdp (lift B H0) (fun ω => lift B (bhList (α / harm M) M p ω))) ≤ α := by
  have hrw : (fdp (lift B H0) (fun ω => lift B (bhList (α / harm M) M p ω)))
      = fdp H0 (bhList (α / harm M) M p) := by
    funext ω
    exact fdp_lift hb B H0 (bhList (α / harm M) M p) ω
  rw [hrw]
  exact benjamini_yekutieli_level P hα hM H0 hnull

/-! ## 4. What the choice of unit costs -/

lemma harm_mono {M n : ℕ} (h : M ≤ n) : harm M ≤ harm n := by
  have hsub : Finset.range M ⊆ Finset.range n := by
    intro x hx
    simp only [Finset.mem_range] at *
    omega
  exact Finset.sum_le_sum_of_subset_of_nonneg hsub (fun j _ _ => by positivity)

/-- The region-level screen tests at a more permissive threshold than the residue-level screen. -/
theorem region_level_is_less_conservative {α : ℝ} (hα : 0 ≤ α) (hM : 0 < M) (h : M ≤ n) :
    α / harm n ≤ α / harm M := by
  have hM0 : 0 < harm M := harm_pos hM
  exact div_le_div_of_nonneg_left hα hM0 (harm_mono h)

/-- Quantitatively, aggregating residues into regions of length `b` buys back `log b − 1` of the
harmonic correction. -/
theorem harmonic_gain_log (hM : 0 < M) (hb : 0 < b) :
    Real.log b - 1 ≤ harm (M * b) - harm M := by
  have hMR : (0 : ℝ) < M := by exact_mod_cast hM
  have hbR : (0 : ℝ) < b := by exact_mod_cast hb
  have h1 : Real.log ((M : ℝ) * b) ≤ Real.log ((M * b : ℕ) + 1) := by
    apply Real.log_le_log (by positivity)
    push_cast
    linarith
  have h2 : Real.log (((M * b : ℕ) : ℝ) + 1) ≤ harm (M * b) := log_le_harm (M * b)
  have h3 : harm M ≤ 1 + Real.log M := harm_le_one_add_log M
  have h4 : Real.log ((M : ℝ) * b) = Real.log M + Real.log b :=
    Real.log_mul (ne_of_gt hMR) (ne_of_gt hbR)
  linarith

end RegionScreen
end IDR
