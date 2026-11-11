/-
# Two worked instances of the ceiling

* `Example.bias_reaches_ceiling` — the five-residue predictor `A` of `AUCInversion.lean`, whose
  pooled AUC is `2/3` because it is miscalibrated across the two proteins, reaches pooled AUC `1`
  once a per-protein bias `b = (3, 0)` is added.  Its ceiling
  `w_within·AUC_within + w_between = 1` is attained: the difference constraints are feasible.

* `Crossed` — six residues in two proteins, each protein holding positives at `0` and `20` and a
  negative at `19`.  Here proteins `0` and `1` are *crossed*: the positive of protein `0` at score
  `0` sits below the negative of protein `1` at `19`, and symmetrically.  `crossed_ceiling_gap`
  shows that no bias whatever brings the pooled statistic above `5` out of `8`, while the ceiling
  `w_within·AUC_within + w_between` is `6/8`.  So the ceiling is a genuine upper bound that can be
  strictly out of reach, and the loss is exactly the one full comparison pair predicted by
  `ceiling_gap_of_crossed`.
-/
import RequestProject.AUCCeiling
import RequestProject.AUCInversion

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

namespace Example

/-- The per-protein bias that puts the two proteins of the example on a common scale. -/
def bias : Fin 2 → ℝ := ![3, 0]

@[simp] lemma bias0 : bias 0 = 3 := rfl
@[simp] lemma bias1 : bias 1 = 0 := rfl

@[simp] lemma grp0 : grp 0 = 0 := rfl
@[simp] lemma grp1 : grp 1 = 0 := rfl
@[simp] lemma grp2 : grp 2 = 0 := rfl
@[simp] lemma grp3 : grp 3 = 1 := rfl
@[simp] lemma grp4 : grp 4 = 1 := rfl

/-- With the bias, the miscalibrated predictor `A` reaches pooled AUC `1`: its ceiling is
attained. -/
theorem bias_reaches_ceiling : auc (allPairs lab) (shift grp bias sA) = 1 := by
  rw [auc, card_all, all_eq, U, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_singleton]
  norm_num [kern, shift]

end Example

/-! ## A crossed instance, where the ceiling is out of reach -/

namespace Crossed

/-- Six residues: positives `0, 1` and negative `2` in protein `0`; positives `3, 4` and negative
`5` in protein `1`. -/
def lab : Fin 6 → Bool := ![true, true, false, true, true, false]

/-- Residue to protein. -/
def grp : Fin 6 → Fin 2 := ![0, 0, 0, 1, 1, 1]

/-- Scores: each protein holds positives at `0` and `20` and a negative at `19`. -/
def sc : Fin 6 → ℝ := ![0, 20, 19, 0, 20, 19]

@[simp] lemma sc0 : sc 0 = 0 := rfl
@[simp] lemma sc1 : sc 1 = 20 := rfl
@[simp] lemma sc2 : sc 2 = 19 := rfl
@[simp] lemma sc3 : sc 3 = 0 := rfl
@[simp] lemma sc4 : sc 4 = 20 := rfl
@[simp] lemma sc5 : sc 5 = 19 := rfl

lemma within_eq :
    withinPairs lab grp = ({(0, 2), (1, 2), (3, 5), (4, 5)} : Finset (Fin 6 × Fin 6)) := by
  decide

lemma card_between : (betweenPairs lab grp).card = 4 := by decide

lemma card_all : (allPairs lab).card = 8 := by decide

lemma mem_q : ((0 : Fin 6), (5 : Fin 6)) ∈ betweenPairs lab grp := by decide

lemma mem_q' : ((3 : Fin 6), (2 : Fin 6)) ∈ betweenPairs lab grp := by decide

/-- The within-protein statistic: one of the two comparisons is right in each protein. -/
lemma U_within : U (withinPairs lab grp) sc = 2 := by
  rw [within_eq, U, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num [kern]

/-- **The ceiling is out of reach here.**  Whatever the per-protein bias, the pooled statistic
stays at or below `5`, one full pair short of the ceiling `2 + 4 = 6`. -/
theorem crossed_ceiling_gap (b : Fin 2 → ℝ) :
    U (allPairs lab) (shift grp b sc) ≤ 5 := by
  have h := ceiling_gap_of_crossed lab grp sc b (0, 5) (3, 2) mem_q mem_q' (by decide)
    (by decide) (by decide) (by norm_num)
  rw [U_within, card_between] at h
  norm_num at h
  exact h

/-- The pooled AUC after any bias is at most `5/8`, while the ceiling is `6/8`. -/
theorem crossed_auc_le : ∀ b : Fin 2 → ℝ, auc (allPairs lab) (shift grp b sc) ≤ 5 / 8 := by
  intro b
  rw [auc, card_all]
  rw [div_le_div_iff₀ (by norm_num) (by norm_num)]
  have := crossed_ceiling_gap b
  norm_num
  linarith

end Crossed

end IDR.GroupedAUC
