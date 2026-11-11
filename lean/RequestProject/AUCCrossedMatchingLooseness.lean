/-
# The crossed-matching bound is loose by an unbounded factor

`AUCCrossedMatching.lean` bounds the reachable pooled statistic by
`U_within + |between| − M/2`, with `M` the largest matching of mutually crossed cross-protein
comparisons — a quantity a greedy scan of the sorted gap lists computes exactly, block by block.
This file settles how good that bound is: **not good at all, in the worst case.**

For every `h` there is a three-protein score table (`lab`, `grp`, `sc` below) with

* `maxCrossedCard = 2` (`maxCrossedCard_eq_two`): the crossed graph has exactly one edge, so the
  bound certifies a deficit of one single pair, and
* a true optimum of exactly `U_within + |between| − (h+2)` (`optimum_eq`): no bias comes within
  `h+2` pairs of the ceiling.

So the certified deficit is `1` while the real deficit is `h + 2`: the bound is loose by the factor
`h + 2`, which is unbounded (`crossed_bound_loose_unbounded`).

**Why.** Proteins 1 and 2 carry a single crossed pair, so the crossed graph sees one pair of
comparisons that cannot both be won.  What it cannot see is that protein 3 pins the two bias
differences `b₁ − b₃ ∈ (0,1)` and `b₂ − b₃ ∈ (h+1, h+2)` — each violation costs `2(h+2)`
half-lost comparisons — and therefore forces `b₁ − b₂ < −h`.  At such a bias every one of the
`h+1` staircase comparisons of protein 2 is lost as well.  The obstruction is a *cyclic* one,
spread over three proteins; the crossed-matching bound is a pairwise certificate and is blind to
it.
-/
import RequestProject.AUCCrossedMatching

set_option autoImplicit false

namespace IDR.GroupedAUC

namespace Looseness

open Finset

variable (h : ℕ)

/-- The residues of the family: one positive and one negative in protein 1, one positive and
`h+1` staircase negatives in protein 2, and `2(h+2)` positives and negatives in protein 3. -/
abbrev Res : Type :=
  Unit ⊕ Unit ⊕ Unit ⊕ Fin (h + 1) ⊕ Fin (2 * (h + 2)) ⊕ Fin (2 * (h + 2))

/-- The positive of protein 1. -/
def p1 : Res h := Sum.inl ()

/-- The negative of protein 1. -/
def n1 : Res h := Sum.inr (Sum.inl ())

/-- The positive of protein 2. -/
def p2 : Res h := Sum.inr (Sum.inr (Sum.inl ()))

/-- The `j`-th staircase negative of protein 2. -/
def n2 (j : Fin (h + 1)) : Res h := Sum.inr (Sum.inr (Sum.inr (Sum.inl j)))

/-- The `i`-th positive of protein 3. -/
def p3 (i : Fin (2 * (h + 2))) : Res h := Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl i))))

/-- The `i`-th negative of protein 3. -/
def n3 (i : Fin (2 * (h + 2))) : Res h := Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr i))))

/-- The labels. -/
def lab : Res h → Bool
  | Sum.inl _ => true
  | Sum.inr (Sum.inl _) => false
  | Sum.inr (Sum.inr (Sum.inl _)) => true
  | Sum.inr (Sum.inr (Sum.inr (Sum.inl _))) => false
  | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl _)))) => true
  | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr _)))) => false

/-- The proteins. -/
def grp : Res h → Fin 3
  | Sum.inl _ => 0
  | Sum.inr (Sum.inl _) => 0
  | Sum.inr (Sum.inr (Sum.inl _)) => 1
  | Sum.inr (Sum.inr (Sum.inr (Sum.inl _))) => 1
  | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl _)))) => 2
  | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr _)))) => 2

/-- The scores. -/
def sc : Res h → ℝ
  | Sum.inl _ => 0
  | Sum.inr (Sum.inl _) => (h : ℝ) + 1
  | Sum.inr (Sum.inr (Sum.inl _)) => -((h : ℝ) + 1)
  | Sum.inr (Sum.inr (Sum.inr (Sum.inl j))) => -((j : ℕ) : ℝ)
  | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inl _)))) => (h : ℝ) + 2
  | Sum.inr (Sum.inr (Sum.inr (Sum.inr (Sum.inr _)))) => 0

@[simp] theorem lab_p1 : lab h (p1 h) = true := rfl
@[simp] theorem lab_n1 : lab h (n1 h) = false := rfl
@[simp] theorem lab_p2 : lab h (p2 h) = true := rfl
@[simp] theorem lab_n2 (j : Fin (h + 1)) : lab h (n2 h j) = false := rfl
@[simp] theorem lab_p3 (i : Fin (2 * (h + 2))) : lab h (p3 h i) = true := rfl
@[simp] theorem lab_n3 (i : Fin (2 * (h + 2))) : lab h (n3 h i) = false := rfl

@[simp] theorem grp_p1 : grp h (p1 h) = 0 := rfl
@[simp] theorem grp_n1 : grp h (n1 h) = 0 := rfl
@[simp] theorem grp_p2 : grp h (p2 h) = 1 := rfl
@[simp] theorem grp_n2 (j : Fin (h + 1)) : grp h (n2 h j) = 1 := rfl
@[simp] theorem grp_p3 (i : Fin (2 * (h + 2))) : grp h (p3 h i) = 2 := rfl
@[simp] theorem grp_n3 (i : Fin (2 * (h + 2))) : grp h (n3 h i) = 2 := rfl

@[simp] theorem sc_p1 : sc h (p1 h) = 0 := rfl
@[simp] theorem sc_n1 : sc h (n1 h) = (h : ℝ) + 1 := rfl
@[simp] theorem sc_p2 : sc h (p2 h) = -((h : ℝ) + 1) := rfl
@[simp] theorem sc_n2 (j : Fin (h + 1)) : sc h (n2 h j) = -((j : ℕ) : ℝ) := rfl
@[simp] theorem sc_p3 (i : Fin (2 * (h + 2))) : sc h (p3 h i) = (h : ℝ) + 2 := rfl
@[simp] theorem sc_n3 (i : Fin (2 * (h + 2))) : sc h (n3 h i) = 0 := rfl

/-- The six kinds of cross-protein comparison in the family. -/
theorem mem_betweenPairs_iff (q : Res h × Res h) :
    q ∈ betweenPairs (lab h) (grp h) ↔
      ((∃ j, q = (p1 h, n2 h j)) ∨ (∃ i, q = (p1 h, n3 h i)) ∨ q = (p2 h, n1 h) ∨
        (∃ i, q = (p2 h, n3 h i)) ∨ (∃ i, q = (p3 h i, n1 h)) ∨
        (∃ i j, q = (p3 h i, n2 h j))) := by
  obtain ⟨a, b⟩ := q
  simp only [betweenPairs, Finset.mem_filter, allPairs, Finset.mem_product, posSet, negSet,
    Finset.mem_univ, true_and]
  constructor
  · rintro ⟨⟨ha, hb⟩, hne⟩
    rcases a with u | u | u | j | i | i <;> rcases b with v | v | v | j' | i' | i' <;>
      simp_all [lab, grp, p1, n1, p2, n2, p3, n3]
  · rintro (⟨j, hj⟩ | ⟨i, hi⟩ | he | ⟨i, hi⟩ | ⟨i, hi⟩ | ⟨i, j, hij⟩) <;>
      simp_all [lab, grp, p1, n1, p2, n2, p3, n3]

/-! ### Two counting lemmas -/

section Counting

variable {I : Type*} [Fintype I] [DecidableEq I]

omit [Fintype I] in
/-- If every comparison of a subset `S` scores at most `c`, the statistic loses `(1-c)|S|`. -/
theorem U_le_of_bad (X S : Finset (I × I)) (hS : S ⊆ X) (s' : I → ℝ) (c : ℝ)
    (hc : ∀ q ∈ S, kern (s' q.1) (s' q.2) ≤ c) :
    U X s' ≤ (X.card : ℝ) - (1 - c) * S.card := by
  classical
  have hsplit : U X s' = (∑ q ∈ X \ S, kern (s' q.1) (s' q.2))
      + ∑ q ∈ S, kern (s' q.1) (s' q.2) := by
    rw [U, ← Finset.sum_sdiff hS]
  have h1 : (∑ q ∈ X \ S, kern (s' q.1) (s' q.2)) ≤ ((X \ S).card : ℝ) := by
    calc (∑ q ∈ X \ S, kern (s' q.1) (s' q.2)) ≤ ∑ _q ∈ X \ S, (1 : ℝ) :=
          Finset.sum_le_sum (fun q _ => kern_le_one _ _)
      _ = ((X \ S).card : ℝ) := by simp
  have h2 : (∑ q ∈ S, kern (s' q.1) (s' q.2)) ≤ c * S.card := by
    calc (∑ q ∈ S, kern (s' q.1) (s' q.2)) ≤ ∑ _q ∈ S, c := Finset.sum_le_sum hc
      _ = c * S.card := by simp [mul_comm]
  have hcard : ((X \ S).card : ℝ) = (X.card : ℝ) - S.card := by
    have hsd := Finset.card_sdiff_of_subset hS
    have hle : S.card ≤ X.card := Finset.card_le_card hS
    rw [hsd]
    push_cast [Nat.cast_sub hle]
    ring
  rw [hsplit]
  have := add_le_add h1 h2
  rw [hcard] at this
  linarith

omit [Fintype I] in
/-- If every comparison outside a subset `S` is won, the statistic is at least `|X| - |S|`. -/
theorem U_ge_of_won (X S : Finset (I × I)) (hS : S ⊆ X) (s' : I → ℝ)
    (hw : ∀ q ∈ X \ S, kern (s' q.1) (s' q.2) = 1) :
    (X.card : ℝ) - S.card ≤ U X s' := by
  classical
  have hsplit : U X s' = (∑ q ∈ X \ S, kern (s' q.1) (s' q.2))
      + ∑ q ∈ S, kern (s' q.1) (s' q.2) := by
    rw [U, ← Finset.sum_sdiff hS]
  have h1 : (∑ q ∈ X \ S, kern (s' q.1) (s' q.2)) = ((X \ S).card : ℝ) := by
    rw [Finset.sum_congr rfl hw]; simp
  have h2 : (0 : ℝ) ≤ ∑ q ∈ S, kern (s' q.1) (s' q.2) :=
    Finset.sum_nonneg (fun q _ => kern_nonneg _ _)
  have hcard : ((X \ S).card : ℝ) = (X.card : ℝ) - S.card := by
    have hsd := Finset.card_sdiff_of_subset hS
    have hle : S.card ≤ X.card := Finset.card_le_card hS
    rw [hsd]
    push_cast [Nat.cast_sub hle]
    ring
  rw [hsplit, h1, hcard]
  linarith

end Counting

/-- A comparison that is not strictly won scores at most a half. -/
theorem kern_le_half {x y : ℝ} (hxy : x ≤ y) : kern x y ≤ 1 / 2 := by
  unfold kern
  split_ifs with h1 h2 <;> [linarith; norm_num; norm_num]

/-- A strictly lost comparison scores zero. -/
theorem kern_eq_zero {x y : ℝ} (hxy : x < y) : kern x y = 0 := by
  unfold kern
  split_ifs with h1 h2 <;> [linarith; linarith; rfl]

/-! ### The five ways a bias fails -/

/-- Protein 1's positive against protein 3's negatives. -/
def SA : Finset (Res h × Res h) := Finset.image (fun i => (p1 h, n3 h i)) Finset.univ

/-- Protein 3's positives against protein 1's negative. -/
def SB : Finset (Res h × Res h) := Finset.image (fun i => (p3 h i, n1 h)) Finset.univ

/-- Protein 2's positive against protein 3's negatives. -/
def SC : Finset (Res h × Res h) := Finset.image (fun i => (p2 h, n3 h i)) Finset.univ

/-- Protein 3's positives against protein 2's top negative. -/
def SD : Finset (Res h × Res h) := Finset.image (fun i => (p3 h i, n2 h 0)) Finset.univ

/-- The staircase comparisons of protein 1 against protein 2, together with the single crossed
comparison the other way. -/
def SE : Finset (Res h × Res h) :=
  insert (p2 h, n1 h) (Finset.image (fun j => (p1 h, n2 h j)) Finset.univ)

theorem SA_card : (SA h).card = 2 * (h + 2) := by
  rw [SA, Finset.card_image_of_injective _ (fun a b hab => by simpa [n3] using hab)]
  simp

theorem SB_card : (SB h).card = 2 * (h + 2) := by
  rw [SB, Finset.card_image_of_injective _ (fun a b hab => by simpa [p3] using hab)]
  simp

theorem SC_card : (SC h).card = 2 * (h + 2) := by
  rw [SC, Finset.card_image_of_injective _ (fun a b hab => by simpa [n3] using hab)]
  simp

theorem SD_card : (SD h).card = 2 * (h + 2) := by
  rw [SD, Finset.card_image_of_injective _ (fun a b hab => by simpa [p3] using hab)]
  simp

theorem SE_card : (SE h).card = h + 2 := by
  have hinj : Function.Injective (fun j : Fin (h + 1) => (p1 h, n2 h j)) := by
    intro a b hab; simpa [n2] using hab
  have hnot : (p2 h, n1 h) ∉ Finset.image (fun j : Fin (h + 1) => (p1 h, n2 h j)) Finset.univ := by
    simp only [Finset.mem_image, not_exists]
    rintro j ⟨-, hj⟩
    exact absurd (congrArg Prod.fst hj) (by simp [p1, p2])
  rw [SE, Finset.card_insert_of_notMem hnot, Finset.card_image_of_injective _ hinj]
  simp

theorem SA_subset : SA h ⊆ betweenPairs (lab h) (grp h) := by
  intro q hq
  simp only [SA, Finset.mem_image, Finset.mem_univ, true_and] at hq
  obtain ⟨i, rfl⟩ := hq
  exact (mem_betweenPairs_iff h _).2 (Or.inr (Or.inl ⟨i, rfl⟩))

theorem SB_subset : SB h ⊆ betweenPairs (lab h) (grp h) := by
  intro q hq
  simp only [SB, Finset.mem_image, Finset.mem_univ, true_and] at hq
  obtain ⟨i, rfl⟩ := hq
  exact (mem_betweenPairs_iff h _).2 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl ⟨i, rfl⟩)))))

theorem SC_subset : SC h ⊆ betweenPairs (lab h) (grp h) := by
  intro q hq
  simp only [SC, Finset.mem_image, Finset.mem_univ, true_and] at hq
  obtain ⟨i, rfl⟩ := hq
  exact (mem_betweenPairs_iff h _).2 (Or.inr (Or.inr (Or.inr (Or.inl ⟨i, rfl⟩))))

theorem SD_subset : SD h ⊆ betweenPairs (lab h) (grp h) := by
  intro q hq
  simp only [SD, Finset.mem_image, Finset.mem_univ, true_and] at hq
  obtain ⟨i, rfl⟩ := hq
  exact (mem_betweenPairs_iff h _).2 (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr ⟨i, 0, rfl⟩)))))

theorem SE_subset : SE h ⊆ betweenPairs (lab h) (grp h) := by
  intro q hq
  rw [SE, Finset.mem_insert] at hq
  rcases hq with rfl | hq
  · exact (mem_betweenPairs_iff h _).2 (Or.inr (Or.inr (Or.inl rfl)))
  · simp only [Finset.mem_image, Finset.mem_univ, true_and] at hq
    obtain ⟨j, rfl⟩ := hq
    exact (mem_betweenPairs_iff h _).2 (Or.inl ⟨j, rfl⟩)

/-! ### No bias comes within `h+2` pairs of the ceiling -/

/-- The generic step: a set of `n` comparisons each scoring at most `c` costs `(1-c)n`. -/
theorem between_le (b : Fin 3 → ℝ) (S : Finset (Res h × Res h)) (c : ℝ)
    (hsub : S ⊆ betweenPairs (lab h) (grp h))
    (hc : ∀ q ∈ S, kern (shift (grp h) b (sc h) q.1) (shift (grp h) b (sc h) q.2) ≤ c)
    (hbig : (h : ℝ) + 2 ≤ (1 - c) * S.card) :
    U (betweenPairs (lab h) (grp h)) (shift (grp h) b (sc h))
      ≤ ((betweenPairs (lab h) (grp h)).card : ℝ) - ((h : ℝ) + 2) := by
  have := U_le_of_bad (betweenPairs (lab h) (grp h)) S hsub (shift (grp h) b (sc h)) c hc
  linarith

/-- **The true deficit of the family: `h+2` pairs.**  Whatever the per-protein bias, the pooled
statistic stays `h+2` pairs below the ceiling. -/
theorem deficit_ge (b : Fin 3 → ℝ) :
    U (allPairs (lab h)) (shift (grp h) b (sc h))
      ≤ U (withinPairs (lab h) (grp h)) (sc h) + ((betweenPairs (lab h) (grp h)).card : ℝ)
        - ((h : ℝ) + 2) := by
  rw [U_shift_eq]
  have hcast : ((2 * (h + 2) : ℕ) : ℝ) = 2 * ((h : ℝ) + 2) := by push_cast; ring
  have key : U (betweenPairs (lab h) (grp h)) (shift (grp h) b (sc h))
      ≤ ((betweenPairs (lab h) (grp h)).card : ℝ) - ((h : ℝ) + 2) := by
    by_cases hA : b 0 ≤ b 2
    · refine between_le h b (SA h) (1 / 2) (SA_subset h) ?_ ?_
      · intro q hq
        simp only [SA, Finset.mem_image, Finset.mem_univ, true_and] at hq
        obtain ⟨i, rfl⟩ := hq
        exact kern_le_half (by simp [shift]; linarith)
      · rw [SA_card, hcast]; linarith
    push_neg at hA
    by_cases hB : (1 : ℝ) ≤ b 0 - b 2
    · refine between_le h b (SB h) (1 / 2) (SB_subset h) ?_ ?_
      · intro q hq
        simp only [SB, Finset.mem_image, Finset.mem_univ, true_and] at hq
        obtain ⟨i, rfl⟩ := hq
        exact kern_le_half (by simp [shift]; linarith)
      · rw [SB_card, hcast]; linarith
    push_neg at hB
    by_cases hC : b 1 - b 2 ≤ (h : ℝ) + 1
    · refine between_le h b (SC h) (1 / 2) (SC_subset h) ?_ ?_
      · intro q hq
        simp only [SC, Finset.mem_image, Finset.mem_univ, true_and] at hq
        obtain ⟨i, rfl⟩ := hq
        exact kern_le_half (by simp [shift]; linarith)
      · rw [SC_card, hcast]; linarith
    push_neg at hC
    by_cases hD : (h : ℝ) + 2 ≤ b 1 - b 2
    · refine between_le h b (SD h) (1 / 2) (SD_subset h) ?_ ?_
      · intro q hq
        simp only [SD, Finset.mem_image, Finset.mem_univ, true_and] at hq
        obtain ⟨i, rfl⟩ := hq
        have : ((0 : Fin (h + 1)) : ℕ) = 0 := rfl
        simp only [shift, sc_p3, sc_n2, grp_p3, grp_n2, this]
        refine kern_le_half ?_
        push_cast
        linarith
      · rw [SD_card, hcast]; linarith
    push_neg at hD
    -- the remaining case: the two bias differences are pinned, and the staircase is lost
    refine between_le h b (SE h) 0 (SE_subset h) ?_ ?_
    · intro q hq
      rw [SE, Finset.mem_insert] at hq
      rcases hq with rfl | hq
      · refine le_of_eq (kern_eq_zero ?_)
        simp only [shift, sc_p2, sc_n1, grp_p2, grp_n1]
        linarith
      · simp only [Finset.mem_image, Finset.mem_univ, true_and] at hq
        obtain ⟨j, rfl⟩ := hq
        refine le_of_eq (kern_eq_zero ?_)
        have hj : ((j : ℕ) : ℝ) ≤ (h : ℝ) := by
          have := j.isLt
          have : (j : ℕ) ≤ h := by omega
          exact_mod_cast this
        simp only [shift, sc_p1, sc_n2, grp_p1, grp_n2]
        linarith
    · rw [SE_card]
      push_cast
      linarith
  linarith


/-! ### The crossed-matching bound, and how loose it is -/

theorem crossed_imp (q q' : Res h × Res h)
    (hc : CrossedConfig (lab h) (grp h) (sc h) q q') :
    q = (p2 h, n1 h) ∨ q' = (p2 h, n1 h) := by
  obtain ⟨hq, hq', hne, hg1, hg2, hsum⟩ := hc
  rcases (mem_betweenPairs_iff h q).1 hq with ⟨j, rfl⟩|⟨i, rfl⟩|rfl|⟨i, rfl⟩|⟨i, rfl⟩|⟨i, j, rfl⟩ <;>
    rcases (mem_betweenPairs_iff h q').1 hq' with
      ⟨j', rfl⟩|⟨i', rfl⟩|rfl|⟨i', rfl⟩|⟨i', rfl⟩|⟨i', j', rfl⟩ <;>
    simp_all
  · exfalso
    have hj : (0 : ℝ) ≤ ((j' : ℕ) : ℝ) := Nat.cast_nonneg _
    linarith
  · exfalso
    have hj : (0 : ℝ) ≤ ((j : ℕ) : ℝ) := Nat.cast_nonneg _
    linarith

/-- The crossed graph of the family has a single edge: the largest matching has two vertices. -/
theorem maxCrossedCard_le_two : maxCrossedCard (lab h) (grp h) (sc h) ≤ 2 := by
  classical
  obtain ⟨A, hA, hcard⟩ := exists_max_crossed_matching (lab h) (grp h) (sc h)
  obtain ⟨sig, hmap, hinv, hcross⟩ := hA
  have hsub : A ⊆ {(p2 h, n1 h), sig (p2 h, n1 h)} := by
    intro q hq
    rcases crossed_imp h q (sig q) (hcross q hq) with hz | hz
    · simp [hz]
    · have hq2 : q = sig (p2 h, n1 h) := by rw [← hz, hinv q hq]
      simp [hq2]
  have h1 := Finset.card_le_card hsub
  have h2 : ({(p2 h, n1 h), sig (p2 h, n1 h)} : Finset (Res h × Res h)).card ≤ 2 := by
    refine le_trans (Finset.card_insert_le _ _) ?_
    simp
  omega

theorem two_le_maxCrossedCard : 2 ≤ maxCrossedCard (lab h) (grp h) (sc h) := by
  have hcr : CrossedConfig (lab h) (grp h) (sc h) (p1 h, n2 h 0) (p2 h, n1 h) := by
    refine ⟨(mem_betweenPairs_iff h _).2 (Or.inl ⟨0, rfl⟩),
      (mem_betweenPairs_iff h _).2 (Or.inr (Or.inr (Or.inl rfl))), ?_, rfl, rfl, ?_⟩
    · intro hcon
      exact absurd (congrArg Prod.fst hcon) (by simp [p1, p2])
    · have h0 : ((0 : Fin (h + 1)) : ℕ) = 0 := rfl
      simp only [sc_p1, sc_n2, sc_p2, sc_n1, h0]
      push_cast
      have : (0 : ℝ) ≤ (h : ℝ) := Nat.cast_nonneg _
      linarith
  have := maxCrossedCard_ge_of_pairing (lab h) (grp h) (sc h) (m := 1)
    (F := fun x => match x with
      | Sum.inl _ => (p1 h, n2 h 0)
      | Sum.inr _ => (p2 h, n1 h))
    (by
      rintro (a | a) (b | b) hab
      · rw [Subsingleton.elim a b]
      · exact absurd (congrArg Prod.fst hab) (by simp [p1, p2])
      · exact absurd (congrArg Prod.fst hab) (by simp [p1, p2])
      · rw [Subsingleton.elim a b])
    (fun _ => hcr)
  omega

/-- **The crossed-matching bound certifies a deficit of exactly one pair.** -/
theorem maxCrossedCard_eq_two : maxCrossedCard (lab h) (grp h) (sc h) = 2 :=
  le_antisymm (maxCrossedCard_le_two h) (two_le_maxCrossedCard h)



/-- The optimal bias of the family. -/
noncomputable def bopt : Fin 3 → ℝ := ![1 / 2, (h : ℝ) + 3 / 2, 0]

@[simp] theorem bopt_zero : bopt h 0 = 1 / 2 := rfl
@[simp] theorem bopt_one : bopt h 1 = (h : ℝ) + 3 / 2 := rfl
@[simp] theorem bopt_two : bopt h 2 = 0 := rfl

theorem won_outside_SE :
    ∀ q ∈ betweenPairs (lab h) (grp h) \ SE h,
      kern (shift (grp h) (bopt h) (sc h) q.1) (shift (grp h) (bopt h) (sc h) q.2) = 1 := by
  intro q hq
  rw [Finset.mem_sdiff] at hq
  obtain ⟨hq1, hq2⟩ := hq
  have hh : (0 : ℝ) ≤ (h : ℝ) := Nat.cast_nonneg _
  rcases (mem_betweenPairs_iff h q).1 hq1 with ⟨j, rfl⟩|⟨i, rfl⟩|rfl|⟨i, rfl⟩|⟨i, rfl⟩|⟨i, j, rfl⟩
  · exact absurd (by simp [SE]) hq2
  · refine kern_eq_one_of_lt ?_
    simp only [shift, sc_p1, sc_n3, grp_p1, grp_n3, bopt_zero, bopt_two]
    norm_num
  · exact absurd (by simp [SE]) hq2
  · refine kern_eq_one_of_lt ?_
    simp only [shift, sc_p2, sc_n3, grp_p2, grp_n3, bopt_one, bopt_two]
    linarith
  · refine kern_eq_one_of_lt ?_
    simp only [shift, sc_p3, sc_n1, grp_p3, grp_n1, bopt_zero, bopt_two]
    linarith
  · refine kern_eq_one_of_lt ?_
    have hj : (0 : ℝ) ≤ ((j : ℕ) : ℝ) := Nat.cast_nonneg _
    simp only [shift, sc_p3, sc_n2, grp_p3, grp_n2, bopt_one, bopt_two]
    linarith

/-- **The deficit of the family is exactly `h+2` pairs**, and it is attained. -/
theorem optimum_eq :
    U (allPairs (lab h)) (shift (grp h) (bopt h) (sc h))
      = U (withinPairs (lab h) (grp h)) (sc h) + ((betweenPairs (lab h) (grp h)).card : ℝ)
        - ((h : ℝ) + 2) := by
  have hle := deficit_ge h (bopt h)
  have hge := U_ge_of_won (betweenPairs (lab h) (grp h)) (SE h) (SE_subset h)
    (shift (grp h) (bopt h) (sc h)) (won_outside_SE h)
  rw [SE_card] at hge
  rw [U_shift_eq] at hle ⊢
  push_cast at hge
  linarith

/-- **The crossed-matching bound is loose by an unbounded factor.**

For every `C` there is a member of the family whose crossed-matching bound certifies a deficit of
exactly one comparison pair (`maxCrossedCard / 2 = 1`), while no per-protein bias comes within `C`
pairs of the ceiling — and the true deficit is exactly `h + 2`, attained at `bopt h`. -/
theorem crossed_bound_loose_unbounded (C : ℝ) :
    ∃ h : ℕ,
      (maxCrossedCard (lab h) (grp h) (sc h) : ℝ) / 2 = 1 ∧
      (∀ b : Fin 3 → ℝ, U (allPairs (lab h)) (shift (grp h) b (sc h))
          ≤ U (withinPairs (lab h) (grp h)) (sc h)
            + ((betweenPairs (lab h) (grp h)).card : ℝ) - C) ∧
      U (allPairs (lab h)) (shift (grp h) (bopt h) (sc h))
        = U (withinPairs (lab h) (grp h)) (sc h)
          + ((betweenPairs (lab h) (grp h)).card : ℝ) - ((h : ℝ) + 2) := by
  obtain ⟨n, hn⟩ := exists_nat_ge C
  refine ⟨n, ?_, ?_, optimum_eq n⟩
  · rw [maxCrossedCard_eq_two]
    norm_num
  · intro b
    have := deficit_ge n b
    linarith

end Looseness

end IDR.GroupedAUC
