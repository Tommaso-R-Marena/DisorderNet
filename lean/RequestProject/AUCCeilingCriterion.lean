/-
# Theorem 2, completed: deciding the ceiling is a negative-cycle problem

`AUCCeiling.lean` reduces the reachability of the pooled ceiling
`w_within·AUC_within + w_between` to the feasibility of the strict difference system

  `s n − s p < b k − b l`  for every cross-protein comparison of a positive `p` of protein `k`
  against a negative `n` of protein `l`,

and `DifferenceConstraints.lean` proves that such a system is feasible exactly when every closed
walk has strictly negative weight.  This file joins the two.

* `IsGapMatrix`, `exists_gapMatrix` — the `K × K` *gap matrix* `c k l = max { s n − s p }` over
  cross-protein comparisons from protein `k` to protein `l`.  It exists as soon as every protein
  contributes at least one positive and one negative residue, and it is computed in one pass over
  the score table.
* `feasible_iff_gap` — the residue-level system and the `K × K` system have the same solutions.
* `ceiling_attainable_iff_noNonnegCycle` — **the decision procedure.**  The per-protein bias can
  push the pooled AUC up to `w_within·AUC_within(s) + w_between` if and only if the gap matrix has
  no closed walk of non-negative weight.  That is a negative-cycle test on a `K × K` matrix:
  `O(K³)` arithmetic, not a search over the `K!` orderings, and it needs the score table only
  through the `K²` maxima.
* `ceiling_unattainable_of_two_cycle` — the shortest certificate: two proteins `k`, `l` with
  `c k l + c l k ≥ 0` already put the ceiling out of reach.
-/
import RequestProject.AUCCeiling
import RequestProject.DifferenceConstraints

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Criterion

variable {I G : Type*} [Fintype I] [DecidableEq G] [Fintype G]

/-- `c` is *the gap matrix* of the data: `c k l` is the largest gap `s n − s p` over comparisons
of a positive `p` of protein `k` with a negative `n` of protein `l`. -/
def IsGapMatrix (lab : I → Bool) (grp : I → G) (s : I → ℝ) (c : G → G → ℝ) : Prop :=
  ∀ k l : G,
    (∃ q ∈ allPairs lab, grp q.1 = k ∧ grp q.2 = l ∧ c k l = s q.2 - s q.1) ∧
      (∀ q ∈ allPairs lab, grp q.1 = k → grp q.2 = l → s q.2 - s q.1 ≤ c k l)

omit [Fintype G] in
/-- The gap matrix exists whenever every protein contributes both a positive and a negative
residue. -/
theorem exists_gapMatrix (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (hP : ∀ k : G, ((posSet lab).filter (fun i => grp i = k)).Nonempty)
    (hN : ∀ k : G, ((negSet lab).filter (fun i => grp i = k)).Nonempty) :
    ∃ c : G → G → ℝ, IsGapMatrix lab grp s c := by
  classical
  have h : ∀ k l : G, ∃ r : ℝ,
      (∃ q ∈ allPairs lab, grp q.1 = k ∧ grp q.2 = l ∧ r = s q.2 - s q.1) ∧
        (∀ q ∈ allPairs lab, grp q.1 = k → grp q.2 = l → s q.2 - s q.1 ≤ r) := by
    intro k l
    obtain ⟨q, hq, hmax⟩ := Finset.exists_max_image
      (((posSet lab).filter (fun i => grp i = k)) ×ˢ ((negSet lab).filter (fun i => grp i = l)))
      (fun q => s q.2 - s q.1) (Finset.Nonempty.product (hP k) (hN l))
    rw [Finset.mem_product, Finset.mem_filter, Finset.mem_filter] at hq
    refine ⟨s q.2 - s q.1, ⟨q, ?_, hq.1.2, hq.2.2, rfl⟩, ?_⟩
    · exact Finset.mem_product.mpr ⟨hq.1.1, hq.2.1⟩
    · intro r hr hr1 hr2
      refine hmax r ?_
      rw [allPairs, Finset.mem_product] at hr
      exact Finset.mem_product.mpr
        ⟨Finset.mem_filter.mpr ⟨hr.1, hr1⟩, Finset.mem_filter.mpr ⟨hr.2, hr2⟩⟩
  choose c hc using h
  exact ⟨c, fun k l => hc k l⟩

/-- The residue-level constraint system and the `K × K` gap system have the same solutions. -/
theorem feasible_iff_gap (lab : I → Bool) (grp : I → G) (s : I → ℝ) (c : G → G → ℝ)
    (hc : IsGapMatrix lab grp s c) :
    (∃ b : G → ℝ, ∀ q ∈ betweenPairs lab grp, s q.2 - s q.1 < b (grp q.1) - b (grp q.2))
      ↔ DiffConstraints.Feasible (univ : Finset G) c := by
  constructor
  · rintro ⟨b, hb⟩
    refine ⟨b, fun k _ l _ hkl => ?_⟩
    obtain ⟨q, hq, hq1, hq2, hqc⟩ := (hc k l).1
    have hqb : q ∈ betweenPairs lab grp := by
      refine Finset.mem_filter.mpr ⟨hq, ?_⟩
      rw [hq1, hq2]
      exact hkl
    have hlt := hb q hqb
    rw [hq1, hq2] at hlt
    rw [hqc]
    exact hlt
  · rintro ⟨b, hb⟩
    refine ⟨b, fun q hq => ?_⟩
    have hq' : q ∈ allPairs lab := (Finset.mem_filter.mp hq).1
    have hne : grp q.1 ≠ grp q.2 := (Finset.mem_filter.mp hq).2
    have hle := (hc (grp q.1) (grp q.2)).2 q hq' rfl rfl
    have := hb (grp q.1) (Finset.mem_univ _) (grp q.2) (Finset.mem_univ _) hne
    linarith

/-- **The decision procedure for the ceiling.**  The per-protein bias reaches the pooled ceiling
if and only if the `K × K` gap matrix has no closed walk of non-negative weight — a negative-cycle
test, `O(K³)`. -/
theorem ceiling_attainable_iff_noNonnegCycle (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (c : G → G → ℝ) (hc : IsGapMatrix lab grp s c) :
    (∃ b : G → ℝ, U (allPairs lab) (shift grp b s)
        = U (withinPairs lab grp) s + (betweenPairs lab grp).card)
      ↔ DiffConstraints.NoNonnegCycleOn (univ : Finset G) c := by
  rw [ceiling_attainable_iff, feasible_iff_gap lab grp s c hc]
  exact DiffConstraints.feasible_iff_noNonnegCycle (univ : Finset G) c

/-- The shortest certificate that the ceiling is out of reach: two proteins whose gap matrix
entries sum to a non-negative number. -/
theorem ceiling_unattainable_of_two_cycle (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (c : G → G → ℝ) (hc : IsGapMatrix lab grp s c) (k l : G) (hkl : k ≠ l)
    (hsum : 0 ≤ c k l + c l k) :
    ¬ ∃ b : G → ℝ, U (allPairs lab) (shift grp b s)
        = U (withinPairs lab grp) s + (betweenPairs lab grp).card := by
  intro hex
  have hcyc := (ceiling_attainable_iff_noNonnegCycle lab grp s c hc).mp hex
  have := hcyc [k, l, k] (by intro v hv; exact Finset.mem_univ v)
    (by simp [List.isChain_cons_cons, hkl, Ne.symm hkl]) (by simp) (by simp)
  simp only [DiffConstraints.wsum_cons_cons, DiffConstraints.wsum_singleton, add_zero] at this
  linarith

end Criterion

end IDR.GroupedAUC
