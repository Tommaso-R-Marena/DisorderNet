/-
# Grouped AUC: the Mann–Whitney statistic on protein-grouped residue scores

A benchmark that scores residues and reports one AUC over the pooled residue set is not
measuring what a per-protein reader of the same predictor measures.  This file sets up the
objects needed to say that precisely.

The data are: a finite set `I` of items (residues), a label `lab : I → Bool` (positive =
disordered / binding), a group map `grp : I → G` (the protein a residue belongs to), and a
score `s : I → ℝ`.

* `kern x y` is the Mann–Whitney comparison kernel, ties counted at `½`.
* `U X s` is the Mann–Whitney statistic summed over a set `X` of (positive, negative) pairs and
  `auc X s = U X s / |X|` the corresponding AUC.
* `allPairs`, `withinPairs`, `betweenPairs` are the pooled, same-protein and cross-protein pair
  sets.
* `U_split`, `card_split` — the pair set splits exactly, hence `auc_pooled_decomp`:
  `AUC_pooled = w_within · AUC_within + w_between · AUC_between`, with
  `w_within = Σ_k P_k N_k / (Σ_k P_k)(Σ_k N_k)` (`weight_within_eq`, `card_all_eq`,
  `card_within_eq`).

Everything else in the development is stated against these definitions.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

/-! ## The comparison kernel -/

/-- Mann–Whitney comparison kernel: `1` if the positive score `x` beats the negative score `y`,
`½` on a tie, `0` otherwise. -/
noncomputable def kern (x y : ℝ) : ℝ := if y < x then 1 else if x = y then 1 / 2 else 0

lemma kern_nonneg (x y : ℝ) : 0 ≤ kern x y := by
  unfold kern; split_ifs <;> norm_num

lemma kern_le_one (x y : ℝ) : kern x y ≤ 1 := by
  unfold kern; split_ifs <;> norm_num

/-- The kernel only sees the difference of its arguments: a common shift is invisible. -/
lemma kern_add_const (x y c : ℝ) : kern (x + c) (y + c) = kern x y := by
  unfold kern
  simp [add_lt_add_iff_right, add_left_inj]

lemma kern_eq_one_of_lt {x y : ℝ} (h : y < x) : kern x y = 1 := by
  unfold kern; simp [h]

lemma kern_eq_one_iff {x y : ℝ} : kern x y = 1 ↔ y < x := by
  unfold kern
  split_ifs with h1 h2
  · simp [h1]
  · norm_num [h1]
  · norm_num [h1]

/-! ## Pair sets -/

section Pairs

variable {I G : Type*} [Fintype I] [DecidableEq G]

/-- The positive items. -/
def posSet (lab : I → Bool) : Finset I := univ.filter (fun i => lab i = true)

/-- The negative items. -/
def negSet (lab : I → Bool) : Finset I := univ.filter (fun i => lab i = false)

/-- All (positive, negative) pairs: the pooled comparison set. -/
def allPairs (lab : I → Bool) : Finset (I × I) := posSet lab ×ˢ negSet lab

/-- The (positive, negative) pairs lying inside one protein. -/
def withinPairs (lab : I → Bool) (grp : I → G) : Finset (I × I) :=
  (allPairs lab).filter (fun q => grp q.1 = grp q.2)

/-- The (positive, negative) pairs straddling two different proteins. -/
def betweenPairs (lab : I → Bool) (grp : I → G) : Finset (I × I) :=
  (allPairs lab).filter (fun q => grp q.1 ≠ grp q.2)

/-- The Mann–Whitney statistic summed over a set of pairs. -/
noncomputable def U (X : Finset (I × I)) (s : I → ℝ) : ℝ := ∑ q ∈ X, kern (s q.1) (s q.2)

/-- The AUC of a set of pairs: the Mann–Whitney statistic normalised by the number of pairs. -/
noncomputable def auc (X : Finset (I × I)) (s : I → ℝ) : ℝ := U X s / X.card

omit [Fintype I] in
lemma U_nonneg (X : Finset (I × I)) (s : I → ℝ) : 0 ≤ U X s :=
  Finset.sum_nonneg fun _ _ => kern_nonneg _ _

omit [Fintype I] in
lemma U_le_card (X : Finset (I × I)) (s : I → ℝ) : U X s ≤ X.card := by
  calc U X s ≤ ∑ _q ∈ X, (1 : ℝ) := Finset.sum_le_sum fun _ _ => kern_le_one _ _
  _ = X.card := by simp

omit [Fintype I] in
lemma auc_nonneg (X : Finset (I × I)) (s : I → ℝ) : 0 ≤ auc X s :=
  div_nonneg (U_nonneg X s) (by positivity)

omit [Fintype I] in
lemma auc_le_one (X : Finset (I × I)) (s : I → ℝ) : auc X s ≤ 1 := by
  unfold auc
  rcases Nat.eq_zero_or_pos X.card with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact U_le_card X s

/-- The pooled statistic is exactly the within statistic plus the between statistic. -/
lemma U_split (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    U (allPairs lab) s = U (withinPairs lab grp) s + U (betweenPairs lab grp) s := by
  classical
  unfold U withinPairs betweenPairs
  exact (Finset.sum_filter_add_sum_filter_not (allPairs lab)
    (fun q : I × I => grp q.1 = grp q.2) _).symm

lemma card_split (lab : I → Bool) (grp : I → G) :
    (allPairs lab).card = (withinPairs lab grp).card + (betweenPairs lab grp).card := by
  classical
  unfold withinPairs betweenPairs
  exact (Finset.card_filter_add_card_filter_not
    (s := allPairs lab) (p := fun q : I × I => grp q.1 = grp q.2)).symm

/-- **The pooled AUC is the pair-count weighted average of the within and between AUCs.** -/
theorem auc_pooled_decomp (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0) :
    auc (allPairs lab) s
      = ((withinPairs lab grp).card / (allPairs lab).card) * auc (withinPairs lab grp) s
        + ((betweenPairs lab grp).card / (allPairs lab).card) * auc (betweenPairs lab grp) s := by
  have hwc : ((withinPairs lab grp).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hw
  have hbc : ((betweenPairs lab grp).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hb
  unfold auc
  rw [U_split lab grp s]
  field_simp

/-! ## The weights -/

variable [Fintype G]

/-- The number of positives in protein `k`. -/
def Pcount (lab : I → Bool) (grp : I → G) (k : G) : ℕ :=
  ((posSet lab).filter (fun i => grp i = k)).card

/-- The number of negatives in protein `k`. -/
def Ncount (lab : I → Bool) (grp : I → G) (k : G) : ℕ :=
  ((negSet lab).filter (fun i => grp i = k)).card

lemma card_all_eq (lab : I → Bool) :
    (allPairs lab).card = (posSet lab).card * (negSet lab).card := by
  simp [allPairs]

lemma sum_Pcount (lab : I → Bool) (grp : I → G) :
    ∑ k : G, Pcount lab grp k = (posSet lab).card := by
  classical
  exact (Finset.card_eq_sum_card_fiberwise
    (f := grp) (s := posSet lab) (t := (univ : Finset G)) (fun i _ => Finset.mem_univ _)).symm

lemma sum_Ncount (lab : I → Bool) (grp : I → G) :
    ∑ k : G, Ncount lab grp k = (negSet lab).card := by
  classical
  exact (Finset.card_eq_sum_card_fiberwise
    (f := grp) (s := negSet lab) (t := (univ : Finset G)) (fun i _ => Finset.mem_univ _)).symm

/-- The within-protein pairs are counted protein by protein: `Σ_k P_k N_k`. -/
lemma card_within_eq (lab : I → Bool) (grp : I → G) :
    (withinPairs lab grp).card = ∑ k : G, Pcount lab grp k * Ncount lab grp k := by
  classical
  have hfib := Finset.card_eq_sum_card_fiberwise
    (f := fun q : I × I => grp q.1) (s := withinPairs lab grp) (t := (univ : Finset G))
    (fun q _ => Finset.mem_univ _)
  rw [hfib]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  unfold Pcount Ncount
  rw [← Finset.card_product]
  congr 1
  apply Finset.ext
  rintro ⟨i, j⟩
  simp only [Finset.mem_filter, Finset.mem_product, withinPairs, allPairs, Finset.mem_filter,
    Finset.mem_product]
  constructor
  · rintro ⟨⟨⟨hi, hj⟩, hij⟩, hik⟩
    exact ⟨⟨hi, hik⟩, hj, by rw [← hij, hik]⟩
  · rintro ⟨⟨hi, hik⟩, hj, hjk⟩
    exact ⟨⟨⟨hi, hj⟩, by rw [hik, hjk]⟩, hik⟩

/-- **The within weight is `Σ_k P_k N_k / (Σ_k P_k)(Σ_k N_k)`**, the identity the benchmark code
asserts. -/
theorem weight_within_eq (lab : I → Bool) (grp : I → G) :
    ((withinPairs lab grp).card : ℝ) / (allPairs lab).card
      = (∑ k : G, (Pcount lab grp k : ℝ) * Ncount lab grp k)
          / ((∑ k : G, (Pcount lab grp k : ℝ)) * (∑ k : G, (Ncount lab grp k : ℝ))) := by
  rw [card_within_eq, card_all_eq, ← sum_Pcount lab grp, ← sum_Ncount lab grp]
  push_cast
  ring

end Pairs

end IDR.GroupedAUC
