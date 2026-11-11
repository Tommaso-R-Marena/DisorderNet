/-
# Theorem 4 — the pooled benchmark can invert the within-protein ordering

The pooled AUC is a weighted average of the within- and between-protein AUCs, and the between
part measures something no per-protein reader ever asks for: whether residues of *different*
proteins are put on a common scale.  A predictor can therefore rank residues strictly better
inside every protein and still score lower on the benchmark.

* `auc_pooled_lt_iff` — **the exact criterion**.  For two predictors `sA`, `sB` on the same
  labelled, grouped residues, `AUC_pooled(A) < AUC_pooled(B)` iff
  `w_within·(AUC_within(B) − AUC_within(A)) + w_between·(AUC_between(B) − AUC_between(A)) > 0`.
* `auc_inversion_iff` — hence, when `AUC_within(A) > AUC_within(B)`, the benchmark reverses the
  ordering exactly when
  `w_between·(AUC_between(B) − AUC_between(A)) > w_within·(AUC_within(A) − AUC_within(B))`:
  the cross-protein advantage of `B`, weighted by the cross-protein pair fraction, must outweigh
  the within-protein advantage of `A`, weighted by the within-protein pair fraction.
* `inversion_requires_between_gap` — the quantitative form: an inversion forces a between-protein
  AUC gap of at least `(w_within/w_between)` times the within-protein gap.
* `Example.inversion` — **an explicit instance**: five residues in two proteins, two predictors
  `A`, `B` with `AUC_within(A) = 1 > 2/3 = AUC_within(B)` yet
  `AUC_pooled(A) = 2/3 < 5/6 = AUC_pooled(B)`.  So the phenomenon is not an artefact of
  estimation noise; it is exhibited by exact arithmetic on a five-residue benchmark.
-/
import RequestProject.AUCCore

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Criterion

variable {I G : Type*} [Fintype I] [DecidableEq G]

/-- The pooled comparison of two predictors is the pair-count weighted comparison of their within
and between parts. -/
theorem auc_pooled_lt_iff (lab : I → Bool) (grp : I → G) (sA sB : I → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0) :
    auc (allPairs lab) sA < auc (allPairs lab) sB
      ↔ 0 < ((withinPairs lab grp).card / (allPairs lab).card)
              * (auc (withinPairs lab grp) sB - auc (withinPairs lab grp) sA)
            + ((betweenPairs lab grp).card / (allPairs lab).card)
              * (auc (betweenPairs lab grp) sB - auc (betweenPairs lab grp) sA) := by
  rw [auc_pooled_decomp lab grp sA hw hb, auc_pooled_decomp lab grp sB hw hb]
  constructor <;> intro h <;> nlinarith [h]

/-- **Theorem 4 (characterisation).**  The pooled benchmark prefers `B` to `A` exactly when `B`'s
cross-protein advantage, weighted by the cross-protein pair fraction, exceeds `A`'s within-protein
advantage, weighted by the within-protein pair fraction.  An *inversion* is the case where the
right-hand side of the within comparison is strictly positive, i.e. `A` is strictly better inside
proteins; no such hypothesis is needed for the equivalence itself. -/
theorem auc_inversion_iff (lab : I → Bool) (grp : I → G) (sA sB : I → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0) :
    auc (allPairs lab) sA < auc (allPairs lab) sB
      ↔ ((withinPairs lab grp).card / (allPairs lab).card)
            * (auc (withinPairs lab grp) sA - auc (withinPairs lab grp) sB)
          < ((betweenPairs lab grp).card / (allPairs lab).card)
            * (auc (betweenPairs lab grp) sB - auc (betweenPairs lab grp) sA) := by
  rw [auc_pooled_lt_iff lab grp sA sB hw hb]
  constructor <;> intro h <;> nlinarith [h]

/-- The quantitative form of the inversion condition: the between-protein gap must exceed the
ratio of pair counts times the within-protein gap. -/
theorem inversion_requires_between_gap (lab : I → Bool) (grp : I → G) (sA sB : I → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0)
    (hpool : auc (allPairs lab) sA < auc (allPairs lab) sB) :
    ((withinPairs lab grp).card : ℝ) / (betweenPairs lab grp).card
        * (auc (withinPairs lab grp) sA - auc (withinPairs lab grp) sB)
      < auc (betweenPairs lab grp) sB - auc (betweenPairs lab grp) sA := by
  have hbc : (0 : ℝ) < ((betweenPairs lab grp).card : ℝ) := by
    exact_mod_cast Nat.pos_of_ne_zero hb
  have hac : (0 : ℝ) < ((allPairs lab).card : ℝ) := by
    have : 0 < (allPairs lab).card :=
      lt_of_lt_of_le (Nat.pos_of_ne_zero hb) (Finset.card_filter_le _ _)
    exact_mod_cast this
  have h := (auc_inversion_iff lab grp sA sB hw hb).mp hpool
  rw [div_mul_eq_mul_div, div_lt_iff₀ hbc]
  have h' := mul_lt_mul_of_pos_left h hac
  field_simp at h'
  nlinarith [h']

end Criterion

/-! ## An explicit inversion

Five residues in two proteins.  Protein `0` holds positives `0, 1` and negative `2`; protein `1`
holds positive `3` and negative `4`.  There are three within-protein pairs, `(0,2)`, `(1,2)`,
`(3,4)`, and three between-protein pairs, `(0,4)`, `(1,4)`, `(3,2)`.

Predictor `A` orders every protein perfectly but puts protein `1` on a much higher scale, so two
of the three cross-protein comparisons come out wrong.  Predictor `B` gets one within-protein
comparison wrong but is calibrated across proteins, so all three cross comparisons are right. -/

namespace Example

/-- Residue labels: `0, 1, 3` are positives, `2, 4` are negatives. -/
def lab : Fin 5 → Bool := ![true, true, false, true, false]

/-- Residue to protein. -/
def grp : Fin 5 → Fin 2 := ![0, 0, 0, 1, 1]

/-- Predictor `A`: perfect inside each protein, badly miscalibrated between them. -/
def sA : Fin 5 → ℝ := ![1, 2, 0, 4, 3]

/-- Predictor `B`: one within-protein error, perfectly calibrated between proteins. -/
def sB : Fin 5 → ℝ := ![1, -1, 0, 3, -2]

lemma within_eq : withinPairs lab grp = ({(0, 2), (1, 2), (3, 4)} : Finset (Fin 5 × Fin 5)) := by
  decide

lemma between_eq : betweenPairs lab grp = ({(0, 4), (1, 4), (3, 2)} : Finset (Fin 5 × Fin 5)) := by
  decide

lemma all_eq : allPairs lab
    = ({(0, 2), (0, 4), (1, 2), (1, 4), (3, 2), (3, 4)} : Finset (Fin 5 × Fin 5)) := by
  decide

@[simp] lemma sA0 : sA 0 = 1 := rfl
@[simp] lemma sA1 : sA 1 = 2 := rfl
@[simp] lemma sA2 : sA 2 = 0 := rfl
@[simp] lemma sA3 : sA 3 = 4 := rfl
@[simp] lemma sA4 : sA 4 = 3 := rfl
@[simp] lemma sB0 : sB 0 = 1 := rfl
@[simp] lemma sB1 : sB 1 = -1 := rfl
@[simp] lemma sB2 : sB 2 = 0 := rfl
@[simp] lemma sB3 : sB 3 = 3 := rfl
@[simp] lemma sB4 : sB 4 = -2 := rfl

lemma card_within : (withinPairs lab grp).card = 3 := by decide

lemma card_all : (allPairs lab).card = 6 := by decide

lemma auc_within_A : auc (withinPairs lab grp) sA = 1 := by
  rw [auc, card_within, within_eq, U, Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num [kern]

lemma auc_within_B : auc (withinPairs lab grp) sB = 2 / 3 := by
  rw [auc, card_within, within_eq, U, Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_singleton]
  norm_num [kern]

lemma auc_pooled_A : auc (allPairs lab) sA = 2 / 3 := by
  rw [auc, card_all, all_eq, U, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_singleton]
  norm_num [kern]

lemma auc_pooled_B : auc (allPairs lab) sB = 5 / 6 := by
  rw [auc, card_all, all_eq, U, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_singleton]
  norm_num [kern]

/-- **Theorem 4.**  A predictor can rank strictly better inside every protein and still lose on
the pooled benchmark. -/
theorem inversion :
    auc (withinPairs lab grp) sB < auc (withinPairs lab grp) sA
      ∧ auc (allPairs lab) sA < auc (allPairs lab) sB := by
  rw [auc_within_A, auc_within_B, auc_pooled_A, auc_pooled_B]
  norm_num

end Example

end IDR.GroupedAUC
