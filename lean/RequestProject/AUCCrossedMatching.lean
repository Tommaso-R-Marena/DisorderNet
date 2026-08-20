/-
# The crossed-matching bound, in its maximal form

`AUCCeiling.lean` shows that two cross-protein comparisons that are *crossed* — they run between
the same two proteins in opposite directions and their gaps sum to a non-negative number — can
never both come out right, whatever the per-protein bias.  A set `A` of comparisons paired up into
such configurations therefore costs `|A|/2` pairs off the ceiling
(`ceiling_gap_of_crossed_matching`).

This file turns that into a *bound*: the largest such `A`.

* `CrossedConfig` — the crossing relation on cross-protein comparisons; symmetric
  (`crossedConfig_symm`), and, between a fixed ordered pair of proteins, a pure threshold
  condition on the two gaps (`crossedConfig_iff_threshold`).  So the crossed graph is a disjoint
  union, over unordered protein pairs, of bipartite graphs whose edges are `u + v ≥ 0`.
* `IsCrossedMatching`, `maxCrossedCard` — a matching in that graph, and the maximum size `M` of
  one.  The maximum is attained (`exists_max_crossed_matching`) and dominates every particular
  matching (`card_le_maxCrossedCard`), so the bound below is the tightest of its family.
* `ceiling_gap_of_max_crossed_matching` — **the bound.** For every per-protein bias `b`,
  `U_pooled(s+b) ≤ U_within(s) + |between| − M/2`.
* `auc_shift_le_max_crossed` — in AUC form,
  `AUC_pooled(s+b) ≤ w_within·AUC_within(s) + w_between − M / (2·|all pairs|)`.
  Unlike the bare ceiling (whose `w_between ≈ 1` makes it nearly vacuous), this one is strictly
  informative as soon as the score table contains crossed comparisons at all.
* `target_unreachable_of_max_crossed` — the decision it supports: if that number is below a
  target AUC, then **no** per-protein bias reaches the target.  Computing the bound needs only the
  score table, not the fitted bias.
* `maxCrossedCard_ge_of_pairing` — the certificate side, and what an implementation actually
  produces: any list of `m` disjoint crossed configurations, however found (greedy on the sorted
  gaps within each protein-pair block, or a maximum bipartite matching), certifies `2m ≤ M` and
  hence a valid upper bound.  Every matching gives a sound bound; the maximum gives the best one.
-/
import RequestProject.AUCCeiling
import RequestProject.ThresholdMatching

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Matching

variable {I G : Type*} [Fintype I] [DecidableEq I] [DecidableEq G]

/-- Two cross-protein comparisons `q`, `q'` are *crossed*: they are distinct, they run between the
same two proteins in opposite directions, and their gaps sum to a non-negative number.  At most one
of the two can be got right, whatever the per-protein bias. -/
def CrossedConfig (lab : I → Bool) (grp : I → G) (s : I → ℝ) (q q' : I × I) : Prop :=
  q ∈ betweenPairs lab grp ∧ q' ∈ betweenPairs lab grp ∧ q ≠ q' ∧
    grp q.1 = grp q'.2 ∧ grp q.2 = grp q'.1 ∧ 0 ≤ (s q.2 - s q.1) + (s q'.2 - s q'.1)

omit [DecidableEq I] in
/-- The crossing relation is symmetric. -/
theorem crossedConfig_symm {lab : I → Bool} {grp : I → G} {s : I → ℝ} {q q' : I × I}
    (h : CrossedConfig lab grp s q q') : CrossedConfig lab grp s q' q := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  exact ⟨h2, h1, Ne.symm h3, h5.symm, h4.symm, by linarith⟩

omit [DecidableEq I] in
/-- **The crossed graph is a threshold graph inside each protein-pair block.**  For comparisons
`q` from protein `k` to protein `l` and `q'` from `l` back to `k`, crossing is exactly the
condition `gap q + gap q' ≥ 0` on the two gaps `gap q = s q.2 − s q.1`.  Hence the graph splits
over unordered protein pairs, and inside each block it is the bipartite graph `u + v ≥ 0`, whose
matchings can be built greedily from the sorted gap lists. -/
theorem crossedConfig_iff_threshold (lab : I → Bool) (grp : I → G) (s : I → ℝ) (q q' : I × I)
    (hq : q ∈ betweenPairs lab grp) (hq' : q' ∈ betweenPairs lab grp) (hne : q ≠ q')
    (k l : G) (hk : grp q.1 = k) (hl : grp q.2 = l) (hk' : grp q'.1 = l) (hl' : grp q'.2 = k) :
    CrossedConfig lab grp s q q' ↔ 0 ≤ (s q.2 - s q.1) + (s q'.2 - s q'.1) := by
  constructor
  · rintro ⟨-, -, -, -, -, h⟩; exact h
  · intro h
    exact ⟨hq, hq', hne, by rw [hk, hl'], by rw [hl, hk'], h⟩

/-- A *crossed matching*: a set `A` of comparisons partitioned by a fixed-point-free involution
`sig` into crossed configurations. -/
def IsCrossedMatching (lab : I → Bool) (grp : I → G) (s : I → ℝ) (A : Finset (I × I)) : Prop :=
  ∃ sig : I × I → I × I, (∀ q ∈ A, sig q ∈ A) ∧ (∀ q ∈ A, sig (sig q) = q) ∧
    (∀ q ∈ A, CrossedConfig lab grp s q (sig q))

omit [DecidableEq I] in
theorem isCrossedMatching_empty (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    IsCrossedMatching lab grp s ∅ :=
  ⟨id, by simp, by simp, by simp⟩

omit [DecidableEq I] in
theorem subset_betweenPairs_of_isCrossedMatching {lab : I → Bool} {grp : I → G} {s : I → ℝ}
    {A : Finset (I × I)} (hA : IsCrossedMatching lab grp s A) : A ⊆ betweenPairs lab grp := by
  obtain ⟨_, _, _, hcross⟩ := hA
  exact fun q hq => (hcross q hq).1

/-- Every crossed matching costs half its size off the ceiling. -/
theorem ceiling_gap_of_isCrossedMatching (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    {A : Finset (I × I)} (hA : IsCrossedMatching lab grp s A) :
    U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card - (A.card : ℝ) / 2 := by
  obtain ⟨sig, hmap, hinv, hcross⟩ := hA
  exact ceiling_gap_of_crossed_matching lab grp s b A
    (subset_betweenPairs_of_isCrossedMatching ⟨sig, hmap, hinv, hcross⟩) sig hmap hinv
    (fun q hq => (hcross q hq).2.2.2.1) (fun q hq => (hcross q hq).2.2.2.2.1)
    (fun q hq => (hcross q hq).2.2.2.2.2)

/-! ### The maximum matching -/

/-- The set of sizes of crossed matchings. -/
def crossedCards (lab : I → Bool) (grp : I → G) (s : I → ℝ) : Set ℕ :=
  {n | ∃ A : Finset (I × I), IsCrossedMatching lab grp s A ∧ A.card = n}

omit [DecidableEq I] in
theorem crossedCards_nonempty (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    (crossedCards lab grp s).Nonempty :=
  ⟨0, ∅, isCrossedMatching_empty lab grp s, by simp⟩

omit [DecidableEq I] in
theorem crossedCards_bddAbove (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    BddAbove (crossedCards lab grp s) := by
  refine ⟨Fintype.card (I × I), ?_⟩
  rintro n ⟨A, -, rfl⟩
  exact Finset.card_le_univ A

/-- **The size of the largest matching of mutually crossed comparisons.**  This is the quantity
the bound below is stated with; it is the maximum-cardinality matching in the crossed graph. -/
noncomputable def maxCrossedCard (lab : I → Bool) (grp : I → G) (s : I → ℝ) : ℕ :=
  sSup (crossedCards lab grp s)

omit [DecidableEq I] in
/-- The maximum is attained by an actual matching. -/
theorem exists_max_crossed_matching (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    ∃ A : Finset (I × I), IsCrossedMatching lab grp s A ∧ A.card = maxCrossedCard lab grp s :=
  Nat.sSup_mem (crossedCards_nonempty lab grp s) (crossedCards_bddAbove lab grp s)

omit [DecidableEq I] in
/-- Every crossed matching is at most as large as the maximum: the bound stated with
`maxCrossedCard` is the tightest of its family. -/
theorem card_le_maxCrossedCard {lab : I → Bool} {grp : I → G} {s : I → ℝ} {A : Finset (I × I)}
    (hA : IsCrossedMatching lab grp s A) : A.card ≤ maxCrossedCard lab grp s :=
  le_csSup (crossedCards_bddAbove lab grp s) ⟨A, hA, rfl⟩

/-- **The crossed-matching bound.**  Whatever the per-protein bias, the pooled Mann–Whitney
statistic stays half the maximum crossed matching below the ceiling. -/
theorem ceiling_gap_of_max_crossed_matching (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (b : G → ℝ) :
    U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card
          - (maxCrossedCard lab grp s : ℝ) / 2 := by
  obtain ⟨A, hA, hcard⟩ := exists_max_crossed_matching lab grp s
  have := ceiling_gap_of_isCrossedMatching lab grp s b hA
  rw [hcard] at this
  exact this

omit [DecidableEq I] in
/-- The bound is tighter than the one from any particular matching. -/
theorem max_crossed_bound_le_matching_bound (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    {A : Finset (I × I)} (hA : IsCrossedMatching lab grp s A) :
    U (withinPairs lab grp) s + (betweenPairs lab grp).card
        - (maxCrossedCard lab grp s : ℝ) / 2
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card - (A.card : ℝ) / 2 := by
  have : (A.card : ℝ) ≤ (maxCrossedCard lab grp s : ℝ) :=
    Nat.cast_le.mpr (card_le_maxCrossedCard hA)
  linarith

/-- **The bound in AUC form.**  `AUC_pooled(s+b) ≤ w_within·AUC_within(s) + w_between −
M / (2·|all pairs|)` for every per-protein bias `b`, with `M` the maximum crossed matching. -/
theorem auc_shift_le_max_crossed (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0) :
    auc (allPairs lab) (shift grp b s)
      ≤ ((withinPairs lab grp).card / (allPairs lab).card) * auc (withinPairs lab grp) s
        + ((betweenPairs lab grp).card / (allPairs lab).card)
        - (maxCrossedCard lab grp s : ℝ) / (2 * (allPairs lab).card) := by
  have hac : (0 : ℝ) < ((allPairs lab).card : ℝ) := by
    have : 0 < (allPairs lab).card :=
      lt_of_lt_of_le (Nat.pos_of_ne_zero hb) (Finset.card_filter_le _ _)
    exact_mod_cast this
  have hwc : ((withinPairs lab grp).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hw
  have h := ceiling_gap_of_max_crossed_matching lab grp s b
  unfold auc
  rw [div_le_iff₀ hac]
  calc U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card
          - (maxCrossedCard lab grp s : ℝ) / 2 := h
    _ = (((withinPairs lab grp).card : ℝ) / (allPairs lab).card
          * (U (withinPairs lab grp) s / (withinPairs lab grp).card)
        + ((betweenPairs lab grp).card : ℝ) / (allPairs lab).card
        - (maxCrossedCard lab grp s : ℝ) / (2 * (allPairs lab).card))
          * (allPairs lab).card := by
        field_simp

/-- **The feasibility test the bound supports.**  If the crossed-matching bound falls short of a
target AUC, then no per-protein bias whatsoever reaches that target — a verdict computable from
the score table alone, before any bias is fitted. -/
theorem target_unreachable_of_max_crossed (lab : I → Bool) (grp : I → G) (s : I → ℝ) (t : ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0)
    (hlt : ((withinPairs lab grp).card / (allPairs lab).card) * auc (withinPairs lab grp) s
        + ((betweenPairs lab grp).card / (allPairs lab).card)
        - (maxCrossedCard lab grp s : ℝ) / (2 * (allPairs lab).card) < t) :
    ∀ b : G → ℝ, auc (allPairs lab) (shift grp b s) < t :=
  fun b => lt_of_le_of_lt (auc_shift_le_max_crossed lab grp s b hw hb) hlt

/-! ### Certificates: any list of disjoint crossed configurations bounds `M` from below -/

/-- **What an implementation produces.**  `m` crossed configurations with pairwise distinct
endpoints — the output of any matching procedure on the crossed graph, greedy or exact — certify
`2m ≤ M`, hence a valid upper bound on the reachable pooled AUC. -/
theorem maxCrossedCard_ge_of_pairing (lab : I → Bool) (grp : I → G) (s : I → ℝ) {m : ℕ}
    (F : Fin m ⊕ Fin m → I × I) (hinj : Function.Injective F)
    (hcross : ∀ i : Fin m, CrossedConfig lab grp s (F (Sum.inl i)) (F (Sum.inr i))) :
    2 * m ≤ maxCrossedCard lab grp s := by
  classical
  set A : Finset (I × I) := Finset.image F univ with hAdef
  set sig : I × I → I × I := fun q =>
    if h : ∃ j : Fin m ⊕ Fin m, F j = q then F (Sum.swap h.choose) else q with hsigdef
  have hsig : ∀ j : Fin m ⊕ Fin m, sig (F j) = F (Sum.swap j) := by
    intro j
    have hex : ∃ j' : Fin m ⊕ Fin m, F j' = F j := ⟨j, rfl⟩
    rw [hsigdef]
    simp only [dif_pos hex]
    congr 2
    exact hinj hex.choose_spec
  have hmemA : ∀ q ∈ A, ∃ j : Fin m ⊕ Fin m, F j = q := by
    intro q hq
    rw [hAdef, Finset.mem_image] at hq
    obtain ⟨j, -, hj⟩ := hq
    exact ⟨j, hj⟩
  have hFmem : ∀ j : Fin m ⊕ Fin m, F j ∈ A := by
    intro j
    rw [hAdef]
    exact Finset.mem_image_of_mem F (Finset.mem_univ j)
  have hmatch : IsCrossedMatching lab grp s A := by
    refine ⟨sig, ?_, ?_, ?_⟩
    · intro q hq
      obtain ⟨j, rfl⟩ := hmemA q hq
      rw [hsig j]
      exact hFmem _
    · intro q hq
      obtain ⟨j, rfl⟩ := hmemA q hq
      rw [hsig j, hsig (Sum.swap j)]
      cases j <;> rfl
    · intro q hq
      obtain ⟨j, rfl⟩ := hmemA q hq
      rw [hsig j]
      cases j with
      | inl i => exact hcross i
      | inr i => exact crossedConfig_symm (hcross i)
  have hcard : A.card = 2 * m := by
    rw [hAdef, Finset.card_image_of_injective _ hinj]
    simp [Finset.card_univ, two_mul]
  have := card_le_maxCrossedCard hmatch
  rw [hcard] at this
  exact this


/-- Any certified lower bound `M₀ ≤ M` on the crossed matching number already bounds the reachable
pooled AUC: this is what an implementation reports. -/
theorem auc_shift_le_of_le_maxCrossedCard (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0)
    (M₀ : ℕ) (hM : M₀ ≤ maxCrossedCard lab grp s) :
    auc (allPairs lab) (shift grp b s)
      ≤ ((withinPairs lab grp).card / (allPairs lab).card) * auc (withinPairs lab grp) s
        + ((betweenPairs lab grp).card / (allPairs lab).card)
        - (M₀ : ℝ) / (2 * (allPairs lab).card) := by
  have hac : (0 : ℝ) < ((allPairs lab).card : ℝ) := by
    have : 0 < (allPairs lab).card :=
      lt_of_lt_of_le (Nat.pos_of_ne_zero hb) (Finset.card_filter_le _ _)
    exact_mod_cast this
  have h := auc_shift_le_max_crossed lab grp s b hw hb
  have hle : (M₀ : ℝ) ≤ (maxCrossedCard lab grp s : ℝ) := Nat.cast_le.mpr hM
  have hdiv : (M₀ : ℝ) / (2 * (allPairs lab).card)
      ≤ (maxCrossedCard lab grp s : ℝ) / (2 * (allPairs lab).card) := by
    gcongr
  linarith

/-- **The block-level algorithm.**  Fix two proteins `k ≠ l`, list the cross-protein comparisons
from `k` to `l` as `X` and those from `l` to `k` as `Y`, and take their gaps.  Any matching of the
threshold graph `gap X + gap Y ≥ 0` — which, by `ThresholdMatching.le_maxMatch_iff_greedyFeasible`,
is maximised by sorting the two gap lists decreasingly and pairing them in reverse — yields a
crossed matching of twice its size, hence a certified upper bound on the reachable pooled AUC. -/
theorem maxCrossedCard_ge_of_block (lab : I → Bool) (grp : I → G) (s : I → ℝ) (k l : G)
    (hkl : k ≠ l) {p q : ℕ} (X : Fin p → I × I) (Y : Fin q → I × I)
    (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ i, X i ∈ allPairs lab ∧ grp (X i).1 = k ∧ grp (X i).2 = l)
    (hY : ∀ j, Y j ∈ allPairs lab ∧ grp (Y j).1 = l ∧ grp (Y j).2 = k)
    {m : ℕ} {f : Fin m → Fin p} {g : Fin m → Fin q}
    (hmatch : ThresholdMatching.IsMatch (fun i => s (X i).2 - s (X i).1)
      (fun j => s (Y j).2 - s (Y j).1) m f g) :
    2 * m ≤ maxCrossedCard lab grp s := by
  obtain ⟨hf, hg, hthr⟩ := hmatch
  have hXbet : ∀ i, X i ∈ betweenPairs lab grp := by
    intro i
    exact Finset.mem_filter.mpr ⟨(hX i).1, by rw [(hX i).2.1, (hX i).2.2]; exact hkl⟩
  have hYbet : ∀ j, Y j ∈ betweenPairs lab grp := by
    intro j
    exact Finset.mem_filter.mpr ⟨(hY j).1, by rw [(hY j).2.1, (hY j).2.2]; exact Ne.symm hkl⟩
  have hne : ∀ i j, X i ≠ Y j := by
    intro i j hij
    apply hkl
    rw [← (hX i).2.1, hij, (hY j).2.1]
  refine maxCrossedCard_ge_of_pairing lab grp s
    (Sum.elim (fun i => X (f i)) (fun i => Y (g i))) ?_ ?_
  · rintro (a | a) (c | c) hac <;> simp only [Sum.elim_inl, Sum.elim_inr] at hac
    · exact congrArg Sum.inl (hf (hXinj hac))
    · exact absurd hac (hne _ _)
    · exact absurd hac.symm (hne _ _)
    · exact congrArg Sum.inr (hg (hYinj hac))
  · intro i
    simp only [Sum.elim_inl, Sum.elim_inr]
    exact ⟨hXbet _, hYbet _, hne _ _, by rw [(hX _).2.1, (hY _).2.2],
      by rw [(hX _).2.2, (hY _).2.1], hthr i⟩


omit [DecidableEq I] in
/-- **Greedy is optimal block by block.**  If `X` enumerates all comparisons from protein `k` to
protein `l` and `Y` all comparisons from `l` to `k`, then no crossed matching can use more than
`2 · maxMatch` comparisons from this block — and `ThresholdMatching.le_maxMatch_iff_greedyFeasible`
computes `maxMatch` by sorting the two gap lists.  Together with `maxCrossedCard_ge_of_block` this
pins the block contribution exactly. -/
theorem block_card_le_two_mul_maxMatch (lab : I → Bool) (grp : I → G) (s : I → ℝ) (k l : G)
    (hkl : k ≠ l) {p q : ℕ} (X : Fin p → I × I) (Y : Fin q → I × I)
    (hXrange : ∀ z : I × I, z ∈ betweenPairs lab grp → grp z.1 = k → grp z.2 = l → ∃ i, X i = z)
    (hYrange : ∀ z : I × I, z ∈ betweenPairs lab grp → grp z.1 = l → grp z.2 = k → ∃ j, Y j = z)
    {A : Finset (I × I)} (hA : IsCrossedMatching lab grp s A) :
    (A.filter (fun z => (grp z.1 = k ∧ grp z.2 = l) ∨ (grp z.1 = l ∧ grp z.2 = k))).card
      ≤ 2 * ThresholdMatching.maxMatch (fun i => s (X i).2 - s (X i).1)
            (fun j => s (Y j).2 - s (Y j).1) := by
  classical
  obtain ⟨sig, hmap, hinv, hcross⟩ := hA
  have hAsub : A ⊆ betweenPairs lab grp :=
    subset_betweenPairs_of_isCrossedMatching ⟨sig, hmap, hinv, hcross⟩
  set C := A.filter (fun z => (grp z.1 = k ∧ grp z.2 = l) ∨ (grp z.1 = l ∧ grp z.2 = k)) with hCdef
  set Ak := C.filter (fun z => grp z.1 = k) with hAkdef
  set Al := C.filter (fun z => grp z.1 = l) with hAldef
  have hAkmem : ∀ z, z ∈ Ak ↔ (z ∈ A ∧ grp z.1 = k ∧ grp z.2 = l) := by
    intro z
    simp only [hAkdef, hCdef, Finset.mem_filter]
    constructor
    · rintro ⟨⟨hzA, hor⟩, hzk⟩
      refine ⟨hzA, hzk, ?_⟩
      rcases hor with h | h
      · exact h.2
      · exact absurd (h.1.symm.trans hzk) (Ne.symm hkl)
    · rintro ⟨hzA, hzk, hzl⟩
      exact ⟨⟨hzA, Or.inl ⟨hzk, hzl⟩⟩, hzk⟩
  have hAlmem : ∀ z, z ∈ Al ↔ (z ∈ A ∧ grp z.1 = l ∧ grp z.2 = k) := by
    intro z
    simp only [hAldef, hCdef, Finset.mem_filter]
    constructor
    · rintro ⟨⟨hzA, hor⟩, hzl⟩
      refine ⟨hzA, hzl, ?_⟩
      rcases hor with h | h
      · exact absurd (h.1.symm.trans hzl) hkl
      · exact h.2
    · rintro ⟨hzA, hzl, hzk⟩
      exact ⟨⟨hzA, Or.inr ⟨hzl, hzk⟩⟩, hzl⟩
  have hsigAk : ∀ z ∈ Ak, sig z ∈ Al := by
    intro z hz
    obtain ⟨hzA, hzk, hzl⟩ := (hAkmem z).mp hz
    have hc := hcross z hzA
    exact (hAlmem _).mpr ⟨hmap z hzA, by rw [← hc.2.2.2.2.1]; exact hzl,
      by rw [← hc.2.2.2.1]; exact hzk⟩
  have hsigAl : ∀ z ∈ Al, sig z ∈ Ak := by
    intro z hz
    obtain ⟨hzA, hzl, hzk⟩ := (hAlmem z).mp hz
    have hc := hcross z hzA
    exact (hAkmem _).mpr ⟨hmap z hzA, by rw [← hc.2.2.2.2.1]; exact hzk,
      by rw [← hc.2.2.2.1]; exact hzl⟩
  have hcardEq : Ak.card = Al.card :=
    Finset.card_bij' (fun z _ => sig z) (fun z _ => sig z) hsigAk hsigAl
      (fun z hz => hinv z ((hAkmem z).mp hz).1) (fun z hz => hinv z ((hAlmem z).mp hz).1)
  have hCcard : C.card = Ak.card + Al.card := by
    have hsplit := Finset.card_filter_add_card_filter_not (s := C) (p := fun z => grp z.1 = k)
    have heq : C.filter (fun z => ¬ (grp z.1 = k)) = Al := by
      rw [hAldef]
      apply Finset.filter_congr
      intro z hz
      rw [hCdef, Finset.mem_filter] at hz
      rcases hz.2 with h | h
      · simp [h.1, hkl]
      · simp [h.1, Ne.symm hkl]
    rw [heq] at hsplit
    rw [hAkdef]
    omega
  have hAkle : Ak.card ∈ ThresholdMatching.matchSizes (fun i => s (X i).2 - s (X i).1)
      (fun j => s (Y j).2 - s (Y j).1) := by
    have henum : ∀ t : Fin Ak.card, ((Ak.equivFin.symm t : {x // x ∈ Ak}) : I × I) ∈ Ak :=
      fun t => (Ak.equivFin.symm t).2
    have hfex : ∀ t : Fin Ak.card,
        ∃ i : Fin p, X i = ((Ak.equivFin.symm t : {x // x ∈ Ak}) : I × I) := by
      intro t
      obtain ⟨hzA, hzk, hzl⟩ := (hAkmem _).mp (henum t)
      exact hXrange _ (hAsub hzA) hzk hzl
    have hgex : ∀ t : Fin Ak.card,
        ∃ j : Fin q, Y j = sig ((Ak.equivFin.symm t : {x // x ∈ Ak}) : I × I) := by
      intro t
      obtain ⟨hzA, hzl, hzk⟩ := (hAlmem _).mp (hsigAk _ (henum t))
      exact hYrange _ (hAsub hzA) hzl hzk
    choose f hf using hfex
    choose g hg using hgex
    refine ⟨f, g, ?_, ?_, ?_⟩
    · intro a b hab
      have hz : ((Ak.equivFin.symm a : {x // x ∈ Ak}) : I × I)
          = ((Ak.equivFin.symm b : {x // x ∈ Ak}) : I × I) := by
        rw [← hf a, ← hf b, hab]
      have := congrArg Ak.equivFin (Subtype.ext hz : Ak.equivFin.symm a = Ak.equivFin.symm b)
      simpa using this
    · intro a b hab
      have hz : sig ((Ak.equivFin.symm a : {x // x ∈ Ak}) : I × I)
          = sig ((Ak.equivFin.symm b : {x // x ∈ Ak}) : I × I) := by
        rw [← hg a, ← hg b, hab]
      have hza := ((hAkmem _).mp (henum a)).1
      have hzb := ((hAkmem _).mp (henum b)).1
      have hz2 : ((Ak.equivFin.symm a : {x // x ∈ Ak}) : I × I)
          = ((Ak.equivFin.symm b : {x // x ∈ Ak}) : I × I) := by
        rw [← hinv _ hza, ← hinv _ hzb, hz]
      have := congrArg Ak.equivFin (Subtype.ext hz2 : Ak.equivFin.symm a = Ak.equivFin.symm b)
      simpa using this
    · intro t
      have hzA := ((hAkmem _).mp (henum t)).1
      have hc := hcross _ hzA
      simp only [hf t, hg t]
      exact hc.2.2.2.2.2
  have := ThresholdMatching.le_maxMatch hAkle
  omega


/-- **The bound is computed exactly in the two-protein case.**  When all cross-protein comparisons
run between the same two proteins, the maximum crossed matching is exactly twice the maximum
matching of the threshold graph on the two gap lists — which greedy computes after a sort.  For
more proteins the same two theorems bound each block from both sides. -/
theorem maxCrossedCard_eq_two_mul_maxMatch (lab : I → Bool) (grp : I → G) (s : I → ℝ) (k l : G)
    (hkl : k ≠ l) {p q : ℕ} (X : Fin p → I × I) (Y : Fin q → I × I)
    (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ i, X i ∈ allPairs lab ∧ grp (X i).1 = k ∧ grp (X i).2 = l)
    (hY : ∀ j, Y j ∈ allPairs lab ∧ grp (Y j).1 = l ∧ grp (Y j).2 = k)
    (hXrange : ∀ z : I × I, z ∈ betweenPairs lab grp → grp z.1 = k → grp z.2 = l → ∃ i, X i = z)
    (hYrange : ∀ z : I × I, z ∈ betweenPairs lab grp → grp z.1 = l → grp z.2 = k → ∃ j, Y j = z)
    (hall : ∀ z ∈ betweenPairs lab grp,
      (grp z.1 = k ∧ grp z.2 = l) ∨ (grp z.1 = l ∧ grp z.2 = k)) :
    maxCrossedCard lab grp s
      = 2 * ThresholdMatching.maxMatch (fun i => s (X i).2 - s (X i).1)
            (fun j => s (Y j).2 - s (Y j).1) := by
  classical
  refine le_antisymm ?_ ?_
  · obtain ⟨A, hA, hcard⟩ := exists_max_crossed_matching lab grp s
    have hfil : A.filter (fun z => (grp z.1 = k ∧ grp z.2 = l) ∨ (grp z.1 = l ∧ grp z.2 = k)) = A :=
      Finset.filter_true_of_mem
        (fun z hz => hall z (subset_betweenPairs_of_isCrossedMatching hA hz))
    have := block_card_le_two_mul_maxMatch lab grp s k l hkl X Y hXrange hYrange hA
    rw [hfil, hcard] at this
    exact this
  · exact maxCrossedCard_ge_of_block lab grp s k l hkl X Y hXinj hYinj hX hY
      (ThresholdMatching.maxMatch_mem _ _).choose_spec.choose_spec

end Matching

end IDR.GroupedAUC
