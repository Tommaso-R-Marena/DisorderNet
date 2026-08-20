/-
# A counting converse: how *many* comparisons a benchmark cannot resolve

`RequestProject.BenchmarkCapacityLabels` proves the existence statement: enter more methods than
the capacity `C = capacityNat R (2ν)` and *some* pair is within `2ν`, hence unresolvable
(`over_capacity_has_close_pair`).  One pair is a weak conclusion for a leaderboard of a hundred
methods.  This file proves the counting version.

The mechanism is a pigeonhole with a convexity step.  Scores live in `{0, …, R}`; cut that range
into blocks of `2ν + 1` consecutive values, of which there are exactly `C`.  Two methods in the
same block differ by at most `2ν`, so every same-block pair is a close pair.  With `k` methods in
`C` blocks, Cauchy–Schwarz forces the number of same-block ordered pairs to be at least `k²/C`:

* `card_closePairs_lower` — **`k² ≤ C · (|close ordered pairs| + k)`**, for any score function
  bounded by `R`;
* `card_closePairs_unordered_lower` — the same in the quotable form, `k² ≤ C · (2·u + k)` where `u`
  counts *unordered* comparisons;
* `benchmark_closePairs_lower`, `benchmark_unresolvable_count` — instantiated at the benchmark:
  at least that many of the leaderboard's comparisons are within `2ν` in measured score, and — with
  the room hypothesis of `unresolvable_of_close` — each of them is undecidable in the strong sense,
  two truths consistent with the annotation ordering it both ways;
* `unresolvable_count_117` — the numbers: a leaderboard of `117` methods on a benchmark of capacity
  `8` has at least `798` unresolvable comparisons out of its `6786`.

The existence statement says a leaderboard past capacity has a soft spot.  The counting statement
says the soft spot is most of the leaderboard.
-/
import Mathlib
import RequestProject.BenchmarkCapacityLabels

set_option autoImplicit false

namespace IDR
namespace CloseCount

open Finset
open IDR.LabelNoise
open IDR.BenchCapacity

/-! ## 1. Blocks -/

/-- Two scores in the same block of width `g + 1` differ by at most `g`. -/
lemma same_block_close (g x y : ℕ) (h : x / (g + 1) = y / (g + 1)) :
    x ≤ y + g ∧ y ≤ x + g := by
  have hx : (g + 1) * (x / (g + 1)) + x % (g + 1) = x := Nat.div_add_mod x (g + 1)
  have hy : (g + 1) * (y / (g + 1)) + y % (g + 1) = y := Nat.div_add_mod y (g + 1)
  rw [h] at hx
  have h1 : x % (g + 1) < g + 1 := Nat.mod_lt _ (Nat.succ_pos g)
  have h2 : y % (g + 1) < g + 1 := Nat.mod_lt _ (Nat.succ_pos g)
  generalize (g + 1) * (y / (g + 1)) = c at hx hy
  omega

/-! ## 2. The counting bound -/

variable {iota : Type*} [DecidableEq iota]

/-- The **close ordered pairs** of a family of methods: the ordered pairs of distinct methods whose
scores lie within `g` of each other. -/
def closePairs (sc : iota → ℕ) (g : ℕ) (M : Finset iota) : Finset (iota × iota) :=
  (M ×ˢ M).filter (fun q => q.1 ≠ q.2 ∧ sc q.1 ≤ sc q.2 + g ∧ sc q.2 ≤ sc q.1 + g)

/-- The **close comparisons**: the unordered version, one entry per pair of methods. -/
noncomputable def closeComparisons (sc : iota → ℕ) (g : ℕ) (M : Finset iota) :
    Finset (Sym2 iota) :=
  (closePairs sc g M).image Sym2.mk

/-- **The counting converse.**  A family of `k` methods with scores in `{0, …, N}` has at least
`k²/C − k` close ordered pairs, where `C = capacityNat N g` is the benchmark's capacity: with only
`C` blocks available, Cauchy–Schwarz forces the same-block pairs to pile up. -/
theorem card_closePairs_lower {sc : iota → ℕ} {g N : ℕ} {M : Finset iota}
    (hbd : ∀ i ∈ M, sc i ≤ N) :
    M.card ^ 2 ≤ capacityNat N g * ((closePairs sc g M).card + M.card) := by
  classical
  set f : iota → ℕ := fun i => sc i / (g + 1) with hf
  set B : Finset ℕ := M.image f with hB
  set F : ℕ → Finset iota := fun b => M.filter (fun i => f i = b) with hF
  -- there are at most `C` blocks
  have hBcard : B.card ≤ capacityNat N g := by
    have hsub : B ⊆ Finset.range (N / (g + 1) + 1) := by
      intro b hb
      obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hb
      refine Finset.mem_range.mpr ?_
      have hdiv := Nat.div_le_div_right (c := g + 1) (hbd i hi)
      simp only [hf]
      omega
    calc B.card ≤ (Finset.range (N / (g + 1) + 1)).card := Finset.card_le_card hsub
      _ = capacityNat N g := by rw [Finset.card_range, capacityNat]
  -- the same-block ordered pairs
  set SB : Finset (iota × iota) := (M ×ˢ M).filter (fun q => f q.1 = f q.2) with hSB
  have hdecomp : SB = B.biUnion (fun b => F b ×ˢ F b) := by
    ext p
    simp only [hSB, hB, hF, Finset.mem_filter, Finset.mem_product, Finset.mem_biUnion,
      Finset.mem_image]
    constructor
    · rintro ⟨⟨h1, h2⟩, h3⟩
      exact ⟨f p.1, ⟨p.1, h1, rfl⟩, ⟨h1, rfl⟩, ⟨h2, h3.symm⟩⟩
    · rintro ⟨b, _, ⟨h1, h2⟩, ⟨h3, h4⟩⟩
      exact ⟨⟨h1, h3⟩, by rw [h2, h4]⟩
  have hdisj : ∀ b ∈ B, ∀ b' ∈ B, b ≠ b' → Disjoint (F b ×ˢ F b) (F b' ×ˢ F b') := by
    intro b _ b' _ hbb
    rw [Finset.disjoint_left]
    intro p hp hp'
    have h1 := (Finset.mem_filter.mp (Finset.mem_product.mp hp).1).2
    have h2 := (Finset.mem_filter.mp (Finset.mem_product.mp hp').1).2
    exact hbb (h1 ▸ h2 ▸ rfl)
  have hSBcard : SB.card = ∑ b ∈ B, (F b).card ^ 2 := by
    rw [hdecomp, Finset.card_biUnion hdisj]
    exact Finset.sum_congr rfl (fun b _ => by rw [Finset.card_product]; ring)
  -- the fibres partition the family
  have hfib : M.card = ∑ b ∈ B, (F b).card :=
    Finset.card_eq_sum_card_fiberwise (fun i hi => Finset.mem_image_of_mem f hi)
  -- Cauchy-Schwarz
  have hcs : M.card ^ 2 ≤ B.card * SB.card := by
    rw [hfib, hSBcard]
    exact sq_sum_le_card_mul_sum_sq
  -- same block means close
  have hsub : SB ⊆ closePairs sc g M ∪ M.diag := by
    intro p hp
    have hmem := Finset.mem_filter.mp hp
    obtain ⟨h1, h2⟩ := Finset.mem_product.mp hmem.1
    by_cases hne : p.1 = p.2
    · refine Finset.mem_union_right _ ?_
      rw [Finset.mem_diag]
      exact ⟨h1, hne⟩
    · refine Finset.mem_union_left _ ?_
      rw [closePairs, Finset.mem_filter]
      exact ⟨Finset.mem_product.mpr ⟨h1, h2⟩, hne, same_block_close g _ _ hmem.2⟩
  have hSBle : SB.card ≤ (closePairs sc g M).card + M.card := by
    calc SB.card ≤ (closePairs sc g M ∪ M.diag).card := Finset.card_le_card hsub
      _ ≤ (closePairs sc g M).card + M.diag.card := Finset.card_union_le _ _
      _ = (closePairs sc g M).card + M.card := by rw [Finset.diag_card]
  calc M.card ^ 2 ≤ B.card * SB.card := hcs
    _ ≤ capacityNat N g * ((closePairs sc g M).card + M.card) :=
        Nat.mul_le_mul hBcard hSBle

/-- Each unordered comparison accounts for at most two ordered pairs. -/
lemma card_closePairs_le_two_mul (sc : iota → ℕ) (g : ℕ) (M : Finset iota) :
    (closePairs sc g M).card ≤ 2 * (closeComparisons sc g M).card := by
  classical
  refine Finset.card_le_mul_card_image _ 2 (fun b _ => ?_)
  obtain ⟨p, hp⟩ : ∃ p : iota × iota, Sym2.mk p = b := b.exists_rep
  have hsub : (closePairs sc g M).filter (fun a => Sym2.mk a = b)
      ⊆ ({p, p.swap} : Finset (iota × iota)) := by
    intro a ha
    have h := (Finset.mem_filter.mp ha).2
    rw [← hp, Sym2.mk_eq_mk_iff] at h
    rcases h with h | h
    · rw [h]; exact Finset.mem_insert_self _ _
    · rw [h]; exact Finset.mem_insert_of_mem (Finset.mem_singleton_self _)
  calc ((closePairs sc g M).filter (fun a => Sym2.mk a = b)).card
      ≤ ({p, p.swap} : Finset (iota × iota)).card := Finset.card_le_card hsub
    _ ≤ 2 := Finset.card_insert_le _ _ |>.trans (by simp)

/-- **The counting converse, in comparisons.**  A leaderboard of `k` methods on a benchmark of
capacity `C` has at least `(k² − C·k)/(2C)` unresolvably close comparisons. -/
theorem card_closePairs_unordered_lower {sc : iota → ℕ} {g N : ℕ} {M : Finset iota}
    (hbd : ∀ i ∈ M, sc i ≤ N) :
    M.card ^ 2 ≤ capacityNat N g * (2 * (closeComparisons sc g M).card + M.card) := by
  refine (card_closePairs_lower (g := g) hbd).trans ?_
  exact Nat.mul_le_mul_left _ (by
    have := card_closePairs_le_two_mul sc g M
    omega)

/-! ## 3. At the benchmark -/

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- **The counting converse at the benchmark.**  Measured against an annotation `L`, a family of
`k` methods has at least `k²/C − k` close ordered comparisons, `C = capacityNat R (2ν)`. -/
theorem benchmark_closePairs_lower {L : Finset α} {nu : ℕ} (pred : iota → Finset α)
    (M : Finset iota) :
    M.card ^ 2 ≤ capacityNat (Fintype.card α) (2 * nu) *
      ((closePairs (fun i => errors L (pred i)) (2 * nu) M).card + M.card) :=
  card_closePairs_lower (fun i _ => by
    simpa [errors] using Finset.card_le_univ (symmDiff L (pred i)))

/-- **And each of those comparisons is undecidable.**  Under the room hypothesis of
`unresolvable_of_close` — the two methods disagree with the annotation on at least `ν` residues in
each direction — every comparison counted here admits two truths consistent with the annotation,
one making each method the better.  So the count is a count of comparisons no analysis of the
benchmark's data can settle.  (The blocks are one narrower than in `benchmark_closePairs_lower`,
which is what turns the closeness into the strict inequality `unresolvable_of_close` asks for.) -/
theorem benchmark_unresolvable_count {L : Finset α} {nu : ℕ} {pred : iota → Finset α}
    {M : Finset iota} (hnu : 0 < nu)
    (hroom : ∀ i ∈ M, ∀ j ∈ M, i ≠ j →
      nu ≤ (symmDiff L (pred i) \ symmDiff L (pred j)).card) :
    M.card ^ 2 ≤ capacityNat (Fintype.card α) (2 * nu - 1) *
        ((closePairs (fun i => errors L (pred i)) (2 * nu - 1) M).card + M.card) ∧
      ∀ q ∈ closePairs (fun i => errors L (pred i)) (2 * nu - 1) M,
        (∃ T₁ : Finset α, noise T₁ L ≤ nu ∧ errors T₁ (pred q.1) < errors T₁ (pred q.2)) ∧
          (∃ T₂ : Finset α, noise T₂ L ≤ nu ∧ errors T₂ (pred q.2) < errors T₂ (pred q.1)) := by
  refine ⟨card_closePairs_lower (g := 2 * nu - 1) (fun i _ => by
    simpa [errors] using Finset.card_le_univ (symmDiff L (pred i))), ?_⟩
  intro q hq
  rw [closePairs, Finset.mem_filter] at hq
  obtain ⟨hmem, hne, h1, h2⟩ := hq
  obtain ⟨hq1, hq2⟩ := Finset.mem_product.mp hmem
  refine unresolvable_of_close (hroom q.1 hq1 q.2 hq2 hne) (hroom q.2 hq2 q.1 hq1 (Ne.symm hne))
    ?_ ?_
  · omega
  · omega

/-! ## 4. The numbers -/

/-- **The quotable form.**  A benchmark whose capacity is `8` methods, faced with a leaderboard of
`117`, cannot resolve at least `798` of the `6786` comparisons between them. -/
theorem unresolvable_count_117 {sc : iota → ℕ} {g N : ℕ} {M : Finset iota}
    (hbd : ∀ i ∈ M, sc i ≤ N) (hM : M.card = 117) (hC : capacityNat N g ≤ 8) :
    798 ≤ (closeComparisons sc g M).card := by
  have h := card_closePairs_unordered_lower (sc := sc) (g := g) (N := N) hbd
  rw [hM] at h
  have hmul : capacityNat N g * (2 * (closeComparisons sc g M).card + 117)
      ≤ 8 * (2 * (closeComparisons sc g M).card + 117) :=
    Nat.mul_le_mul_right _ hC
  omega

/-- For reference: `117` methods generate `6786` comparisons. -/
theorem comparisons_117 : (117 * 116) / 2 = 6786 := by norm_num

end CloseCount
end IDR
