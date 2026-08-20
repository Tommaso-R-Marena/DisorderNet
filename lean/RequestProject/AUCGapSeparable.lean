/-
# The gap matrix is separable, and the ceiling test collapses to two numbers

`AUCCeilingCriterion.lean` decides the reachability of the pooled ceiling
`w_within·AUC_within + w_between` by a negative-cycle test on the `K × K` gap matrix
`c k l = max { s n − s p : p a positive of protein k, n a negative of protein l }`.

That matrix is *separable*: the maximum splits, `c k l = maxneg l − minpos k`
(`isGapMatrix_sep`), where `minpos k` is the smallest score of a positive residue of protein `k`
and `maxneg k` the largest score of a negative residue of protein `k`.  Consequently the weight of
any closed walk `k₁ → k₂ → ⋯ → k₁` telescopes into a sum of *per-protein scalars*

  `overlap k = maxneg k − minpos k`,

the amount by which protein `k`'s negative range overtops its positive range.  So the whole test
reduces to the scalars `overlap k`:

* `feasible_iff_pairwise_overlap`, `ceiling_attainable_iff_pairwise_overlap` — **the ceiling is
  reachable iff `overlap k + overlap l < 0` for every pair of distinct proteins**.  No `K × K`
  matrix, no `O(K³)` cycle search: the two largest values of `overlap` decide it
  (`pairwise_overlap_iff_two_largest`), an `O(K)` scan (`O(K log K)` if one sorts).
* `ceiling_attained_by_centering` — and when it is reachable, the witness is explicit:
  `b k = −(minpos k + maxneg k)/2`, i.e. *centre every protein on the midpoint of its
  min-positive and max-negative score*.  One pass over the score table.
* `ceiling_unattainable_of_two_overlapping` — the practical corollary.  As soon as **two**
  proteins have `minpos k ≤ maxneg k` — a single mis-ordered within-protein comparison at the
  extremes is enough — the ceiling is out of reach.  With a within-protein AUC well below one
  this is the typical case, so the ceiling `w_within·AUC_within + w_between` is not attained and
  the crossed-matching bound of `AUCCrossedMatching.lean` is what binds.
-/
import RequestProject.AUCCeilingCriterion

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Separable

variable {I G : Type*} [Fintype I] [DecidableEq G]

/-- `minpos k` is the smallest score of a positive residue of protein `k`. -/
def IsMinPos (lab : I → Bool) (grp : I → G) (s : I → ℝ) (minpos : G → ℝ) : Prop :=
  ∀ k : G, (∃ p ∈ posSet lab, grp p = k ∧ minpos k = s p) ∧
    (∀ p ∈ posSet lab, grp p = k → minpos k ≤ s p)

/-- `maxneg k` is the largest score of a negative residue of protein `k`. -/
def IsMaxNeg (lab : I → Bool) (grp : I → G) (s : I → ℝ) (maxneg : G → ℝ) : Prop :=
  ∀ k : G, (∃ n ∈ negSet lab, grp n = k ∧ maxneg k = s n) ∧
    (∀ n ∈ negSet lab, grp n = k → s n ≤ maxneg k)

/-- Both extremes exist as soon as every protein contributes a positive and a negative residue. -/
theorem exists_minPos_maxNeg (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (hP : ∀ k : G, ((posSet lab).filter (fun i => grp i = k)).Nonempty)
    (hN : ∀ k : G, ((negSet lab).filter (fun i => grp i = k)).Nonempty) :
    ∃ minpos maxneg : G → ℝ,
      IsMinPos lab grp s minpos ∧ IsMaxNeg lab grp s maxneg := by
  classical
  have hp : ∀ k : G, ∃ r : ℝ, (∃ p ∈ posSet lab, grp p = k ∧ r = s p) ∧
      (∀ p ∈ posSet lab, grp p = k → r ≤ s p) := by
    intro k
    obtain ⟨p, hp, hmin⟩ := Finset.exists_min_image
      ((posSet lab).filter (fun i => grp i = k)) s (hP k)
    rw [Finset.mem_filter] at hp
    exact ⟨s p, ⟨p, hp.1, hp.2, rfl⟩, fun r hr hrk => hmin r (Finset.mem_filter.mpr ⟨hr, hrk⟩)⟩
  have hn : ∀ k : G, ∃ r : ℝ, (∃ n ∈ negSet lab, grp n = k ∧ r = s n) ∧
      (∀ n ∈ negSet lab, grp n = k → s n ≤ r) := by
    intro k
    obtain ⟨n, hn, hmax⟩ := Finset.exists_max_image
      ((negSet lab).filter (fun i => grp i = k)) s (hN k)
    rw [Finset.mem_filter] at hn
    exact ⟨s n, ⟨n, hn.1, hn.2, rfl⟩, fun r hr hrk => hmax r (Finset.mem_filter.mpr ⟨hr, hrk⟩)⟩
  choose mp hmp using hp
  choose mn hmn using hn
  exact ⟨mp, mn, fun k => hmp k, fun k => hmn k⟩

omit [DecidableEq G] in
/-- **The gap matrix is separable**: the largest gap between a positive of protein `k` and a
negative of protein `l` is `maxneg l − minpos k`.  So the `K × K` matrix is determined by `2K`
numbers, computed in one pass over the score table. -/
theorem isGapMatrix_sep (lab : I → Bool) (grp : I → G) (s : I → ℝ) (minpos maxneg : G → ℝ)
    (hmin : IsMinPos lab grp s minpos) (hmax : IsMaxNeg lab grp s maxneg) :
    IsGapMatrix lab grp s (fun k l => maxneg l - minpos k) := by
  intro k l
  obtain ⟨⟨p, hp, hpk, hpv⟩, hpmin⟩ := hmin k
  obtain ⟨⟨n, hn, hnl, hnv⟩, hnmax⟩ := hmax l
  refine ⟨⟨(p, n), ?_, hpk, hnl, by simp [hpv, hnv]⟩, ?_⟩
  · exact Finset.mem_product.mpr ⟨hp, hn⟩
  · intro q hq hq1 hq2
    rw [allPairs, Finset.mem_product] at hq
    have h1 := hpmin q.1 hq.1 hq1
    have h2 := hnmax q.2 hq.2 hq2
    simp only
    linarith

/-- The per-protein scalar the whole test reduces to: by how much protein `k`'s largest negative
score overtops its smallest positive score.  `overlap k < 0` says the two score ranges of protein
`k` are cleanly separated; `overlap k ≥ 0` says they interleave. -/
def overlap (minpos maxneg : G → ℝ) (k : G) : ℝ := maxneg k - minpos k

/-- The centring bias: place every protein so that the midpoint of its min-positive and
max-negative score sits at zero. -/
noncomputable def centeringBias (minpos maxneg : G → ℝ) (k : G) : ℝ :=
  -(minpos k + maxneg k) / 2

variable [Fintype G]

omit [DecidableEq G] in
/-- **The separable feasibility criterion.**  The system of difference constraints of the
separable gap matrix is feasible exactly when the per-protein overlaps are pairwise negative —
a condition on `K` scalars, not on a `K × K` matrix. -/
theorem feasible_iff_pairwise_overlap (minpos maxneg : G → ℝ) :
    DiffConstraints.Feasible (univ : Finset G) (fun k l => maxneg l - minpos k)
      ↔ ∀ k l : G, k ≠ l → overlap minpos maxneg k + overlap minpos maxneg l < 0 := by
  constructor
  · rintro ⟨b, hb⟩ k l hkl
    have h1 := hb k (Finset.mem_univ _) l (Finset.mem_univ _) hkl
    have h2 := hb l (Finset.mem_univ _) k (Finset.mem_univ _) (Ne.symm hkl)
    simp only [overlap]
    simp only at h1 h2
    linarith
  · intro h
    refine ⟨centeringBias minpos maxneg, fun k _ l _ hkl => ?_⟩
    have := h k l hkl
    simp only [overlap] at this
    simp only [centeringBias]
    linarith

/-- **The ceiling test, reduced to `K` scalars.**  A per-protein bias can push the pooled AUC to
`w_within·AUC_within + w_between` if and only if every two distinct proteins have
`overlap k + overlap l < 0`. -/
theorem ceiling_attainable_iff_pairwise_overlap (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (minpos maxneg : G → ℝ) (hmin : IsMinPos lab grp s minpos) (hmax : IsMaxNeg lab grp s maxneg) :
    (∃ b : G → ℝ, U (allPairs lab) (shift grp b s)
        = U (withinPairs lab grp) s + (betweenPairs lab grp).card)
      ↔ ∀ k l : G, k ≠ l → overlap minpos maxneg k + overlap minpos maxneg l < 0 := by
  rw [ceiling_attainable_iff,
    feasible_iff_gap lab grp s _ (isGapMatrix_sep lab grp s minpos maxneg hmin hmax),
    feasible_iff_pairwise_overlap]

omit [Fintype G] in
/-- **The witness is explicit.**  When the pairwise-overlap test passes, centring each protein on
the midpoint of its min-positive and max-negative score already attains the ceiling — an `O(n)`
computation, no optimisation. -/
theorem ceiling_attained_by_centering (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (minpos maxneg : G → ℝ) (hmin : IsMinPos lab grp s minpos) (hmax : IsMaxNeg lab grp s maxneg)
    (h : ∀ k l : G, k ≠ l → overlap minpos maxneg k + overlap minpos maxneg l < 0) :
    U (allPairs lab) (shift grp (centeringBias minpos maxneg) s)
      = U (withinPairs lab grp) s + (betweenPairs lab grp).card := by
  rw [U_shift_eq_ceiling_iff]
  intro q hq
  have hq' : q ∈ allPairs lab := (Finset.mem_filter.mp hq).1
  have hne : grp q.1 ≠ grp q.2 := (Finset.mem_filter.mp hq).2
  rw [allPairs, Finset.mem_product] at hq'
  have h1 := (hmin (grp q.1)).2 q.1 hq'.1 rfl
  have h2 := (hmax (grp q.2)).2 q.2 hq'.2 rfl
  have h3 := h (grp q.1) (grp q.2) hne
  simp only [overlap] at h3
  simp only [centeringBias]
  linarith

/-- **The practical corollary.**  Two proteins whose positive and negative score ranges interleave
at the extremes (`minpos ≤ maxneg`, i.e. one within-protein comparison at the extremes already
comes out wrong) already put the ceiling out of reach, whatever the bias. -/
theorem ceiling_unattainable_of_two_overlapping (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (minpos maxneg : G → ℝ) (hmin : IsMinPos lab grp s minpos) (hmax : IsMaxNeg lab grp s maxneg)
    (k l : G) (hkl : k ≠ l) (hk : minpos k ≤ maxneg k) (hl : minpos l ≤ maxneg l) :
    ¬ ∃ b : G → ℝ, U (allPairs lab) (shift grp b s)
        = U (withinPairs lab grp) s + (betweenPairs lab grp).card := by
  intro hex
  have h := (ceiling_attainable_iff_pairwise_overlap lab grp s minpos maxneg hmin hmax).mp hex
    k l hkl
  simp only [overlap] at h
  linarith

omit [Fintype G] in
/-- **The test is an `O(K)` scan.**  If `k₁` carries the largest overlap and `k₂` the largest
among the rest, the pairwise condition is decided by those two numbers alone: sort the `K`
overlaps (or scan for the top two) and compare their sum with zero. -/
theorem pairwise_overlap_iff_two_largest (minpos maxneg : G → ℝ) (k₁ k₂ : G) (hne : k₁ ≠ k₂)
    (h1 : ∀ j : G, overlap minpos maxneg j ≤ overlap minpos maxneg k₁)
    (h2 : ∀ j : G, j ≠ k₁ → overlap minpos maxneg j ≤ overlap minpos maxneg k₂) :
    (∀ k l : G, k ≠ l → overlap minpos maxneg k + overlap minpos maxneg l < 0)
      ↔ overlap minpos maxneg k₁ + overlap minpos maxneg k₂ < 0 := by
  constructor
  · intro h; exact h k₁ k₂ hne
  · intro h k l hkl
    by_cases hk : k = k₁
    · subst hk
      have := h2 l (Ne.symm hkl)
      linarith
    · have := h2 k hk
      have := h1 l
      linarith

end Separable

end IDR.GroupedAUC
