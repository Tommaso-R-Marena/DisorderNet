/-
# Part XLV.1  The benchmark: disorder annotation is positive-unlabelled data

Every quantitative claim about disorder *prediction* is a claim about performance on an
annotated benchmark, and the annotation has a structural property that the scoring does not
acknowledge: a residue is annotated disordered when an experiment has shown it to be, and left
unannotated otherwise, whether it was shown to be ordered or simply never looked at.  The
labels are therefore one-sided -- every labelled positive is a true positive, but an unlabelled
residue may be either.  This file proves what that does to the four numbers a benchmark
reports.

Fix a chain `Fin N`, the truth `y`, the labels `l` with `label_sound : l i → y i`, and a
predictor `p`.

* `precision_label_le_precision_truth` -- the reported precision is a *lower bound* on the true
  precision: the labels can only make a predictor look worse on this measure.
* `falsePos_label_eq` -- the reported false positives decompose exactly as the true false
  positives plus the predicted-but-unannotated true positives, so a predictor is penalised
  precisely for the discoveries the annotation has not caught up with.
* `accuracy_close` -- what the benchmark does certify: the true accuracy differs from the
  reported accuracy by at most the fraction of mislabelled residues, so with a good annotation
  the reported number is meaningful and the size of the caveat is quantitative.
* `benchmark_can_invert_ranking` -- but the ranking is not safe at any annotation quality: on a
  four-residue chain with a single unannotated disordered residue, the predictor that is right
  about the truth scores `3/4` on the benchmark while the predictor that merely reproduces the
  annotation scores `1`.  The benchmark ranks them in the opposite order to the truth.  Since
  models are selected by benchmark ranking, this is a mechanism that actively selects against
  predictors that generalise beyond the annotation.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset

namespace Benchmark

variable {N : ℕ}

/-- Residues where the predictor fires. -/
def positives (p : Fin N → Bool) : Finset (Fin N) := Finset.univ.filter fun i => p i

/-- Predicted residues that are also positive in the reference `t`. -/
def truePos (p t : Fin N → Bool) : Finset (Fin N) := Finset.univ.filter fun i => p i ∧ t i

/-- Predicted residues that are negative in the reference `t`. -/
def falsePos (p t : Fin N → Bool) : Finset (Fin N) := Finset.univ.filter fun i => p i ∧ ¬ t i

/-- Precision of `p` against the reference `t`. -/
noncomputable def precision (p t : Fin N → Bool) : ℝ :=
  ((truePos p t).card : ℝ) / (positives p).card

/-- Accuracy of `p` against the reference `t`. -/
noncomputable def accuracy (p t : Fin N → Bool) : ℝ :=
  ((Finset.univ.filter fun i : Fin N => p i = t i).card : ℝ) / N

/-! ## One-sided labels -/

/-- **The reported precision underestimates the true precision.**  Because every labelled
positive is a true positive, the numerator computed against the labels is at most the numerator
computed against the truth, and the denominator is the same. -/
theorem precision_label_le_precision_truth (p y l : Fin N → Bool)
    (hsound : ∀ i, l i → y i) : precision p l ≤ precision p y := by
  unfold precision
  have hsub : truePos p l ⊆ truePos p y := by
    intro i hi
    simp only [truePos, Finset.mem_filter, Finset.mem_univ, true_and] at hi ⊢
    exact ⟨hi.1, hsound i hi.2⟩
  have hnum : ((truePos p l).card : ℝ) ≤ (truePos p y).card := by
    exact_mod_cast Finset.card_le_card hsub
  gcongr

/-- **The reported false positives are the true ones plus the unannotated discoveries.**  A
predictor is penalised, residue by residue, exactly for the disordered residues that the
annotation has not yet recorded. -/
theorem falsePos_label_eq (p y l : Fin N → Bool) (hsound : ∀ i, l i → y i) :
    (falsePos p l).card
      = (falsePos p y).card
        + (Finset.univ.filter fun i : Fin N => p i ∧ y i ∧ ¬ l i).card := by
  classical
  have hsplit : falsePos p l
      = (falsePos p y) ∪ (Finset.univ.filter fun i : Fin N => p i ∧ y i ∧ ¬ l i) := by
    ext i
    simp only [falsePos, Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union]
    constructor
    · rintro ⟨hp, hl⟩
      by_cases hy : y i = true
      · exact Or.inr ⟨hp, hy, hl⟩
      · exact Or.inl ⟨hp, by simpa using hy⟩
    · rintro (⟨hp, hy⟩ | ⟨hp, hy, hl⟩)
      · refine ⟨hp, ?_⟩
        intro hl
        exact hy (hsound i hl)
      · exact ⟨hp, hl⟩
  have hdisj : Disjoint (falsePos p y)
      (Finset.univ.filter fun i : Fin N => p i ∧ y i ∧ ¬ l i) := by
    refine Finset.disjoint_left.mpr fun i hi hi' => ?_
    simp only [falsePos, Finset.mem_filter, Finset.mem_univ, true_and] at hi hi'
    exact hi.2 hi'.2.1
  rw [hsplit, Finset.card_union_of_disjoint hdisj]

/-! ## What the benchmark does certify -/

/-- **The reported accuracy is off by at most the annotation error rate.**  If the labels differ
from the truth at `d` residues, the true accuracy of any predictor is within `d/N` of the
accuracy the benchmark reports. -/
theorem accuracy_close (p y l : Fin N → Bool) :
    |accuracy p y - accuracy p l|
      ≤ ((Finset.univ.filter fun i : Fin N => l i ≠ y i).card : ℝ) / N := by
  classical
  set d := (Finset.univ.filter fun i : Fin N => l i ≠ y i).card with hd
  have h1 : (Finset.univ.filter fun i : Fin N => p i = y i).card
      ≤ (Finset.univ.filter fun i : Fin N => p i = l i).card + d := by
    have hsub : (Finset.univ.filter fun i : Fin N => p i = y i)
        ⊆ (Finset.univ.filter fun i : Fin N => p i = l i)
          ∪ (Finset.univ.filter fun i : Fin N => l i ≠ y i) := by
      intro i hi
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hi ⊢
      by_cases hly : l i = y i
      · exact Or.inl (by rw [hly]; exact hi)
      · exact Or.inr hly
    calc (Finset.univ.filter fun i : Fin N => p i = y i).card
        ≤ ((Finset.univ.filter fun i : Fin N => p i = l i)
            ∪ (Finset.univ.filter fun i : Fin N => l i ≠ y i)).card := Finset.card_le_card hsub
      _ ≤ _ := Finset.card_union_le _ _
  have h2 : (Finset.univ.filter fun i : Fin N => p i = l i).card
      ≤ (Finset.univ.filter fun i : Fin N => p i = y i).card + d := by
    have hsub : (Finset.univ.filter fun i : Fin N => p i = l i)
        ⊆ (Finset.univ.filter fun i : Fin N => p i = y i)
          ∪ (Finset.univ.filter fun i : Fin N => l i ≠ y i) := by
      intro i hi
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_union] at hi ⊢
      by_cases hly : l i = y i
      · exact Or.inl (by rw [← hly]; exact hi)
      · exact Or.inr hly
    calc (Finset.univ.filter fun i : Fin N => p i = l i).card
        ≤ ((Finset.univ.filter fun i : Fin N => p i = y i)
            ∪ (Finset.univ.filter fun i : Fin N => l i ≠ y i)).card := Finset.card_le_card hsub
      _ ≤ _ := Finset.card_union_le _ _
  rcases Nat.eq_zero_or_pos N with hN | hN
  · subst hN
    simp [accuracy]
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have h1' : ((Finset.univ.filter fun i : Fin N => p i = y i).card : ℝ)
      ≤ ((Finset.univ.filter fun i : Fin N => p i = l i).card : ℝ) + d := by exact_mod_cast h1
  have h2' : ((Finset.univ.filter fun i : Fin N => p i = l i).card : ℝ)
      ≤ ((Finset.univ.filter fun i : Fin N => p i = y i).card : ℝ) + d := by exact_mod_cast h2
  rw [accuracy, accuracy, div_sub_div_same, abs_div, abs_of_nonneg hNpos.le]
  gcongr
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-! ## The ranking is not safe -/

/-- The truth on a four-residue chain: two disordered residues. -/
def truth4 : Fin 4 → Bool := ![true, true, false, false]

/-- The annotation: only the first of them has been recorded. -/
def label4 : Fin 4 → Bool := ![true, false, false, false]

/-- The predictor that is right about the truth. -/
def honest4 : Fin 4 → Bool := ![true, true, false, false]

/-- The predictor that reproduces the annotation. -/
def conformist4 : Fin 4 → Bool := ![true, false, false, false]

/-- **The benchmark can inverse the ranking.**  The labels are sound -- every annotated residue
really is disordered -- yet the predictor that gets the truth exactly right scores `3/4` on the
benchmark while the predictor that merely reproduces the annotation, and is wrong about the
truth, scores a perfect `1`.  Selecting models by benchmark accuracy selects the second. -/
theorem benchmark_can_invert_ranking :
    (∀ i, label4 i → truth4 i) ∧
    accuracy honest4 truth4 = 1 ∧ accuracy conformist4 truth4 = 3 / 4 ∧
    accuracy honest4 label4 = 3 / 4 ∧ accuracy conformist4 label4 = 1 ∧
    accuracy conformist4 label4 > accuracy honest4 label4 ∧
    accuracy conformist4 truth4 < accuracy honest4 truth4 := by
  have c1 : (Finset.univ.filter fun i : Fin 4 => honest4 i = truth4 i).card = 4 := by decide
  have c2 : (Finset.univ.filter fun i : Fin 4 => conformist4 i = truth4 i).card = 3 := by decide
  have c3 : (Finset.univ.filter fun i : Fin 4 => honest4 i = label4 i).card = 3 := by decide
  have c4 : (Finset.univ.filter fun i : Fin 4 => conformist4 i = label4 i).card = 4 := by decide
  have a1 : accuracy honest4 truth4 = 1 := by rw [accuracy, c1]; norm_num
  have a2 : accuracy conformist4 truth4 = 3 / 4 := by rw [accuracy, c2]; norm_num
  have a3 : accuracy honest4 label4 = 3 / 4 := by rw [accuracy, c3]; norm_num
  have a4 : accuracy conformist4 label4 = 1 := by rw [accuracy, c4]; norm_num
  refine ⟨fun i => ?_, a1, a2, a3, a4, ?_, ?_⟩
  · fin_cases i <;> simp [truth4, label4]
  · rw [a3, a4]; norm_num
  · rw [a1, a2]; norm_num

end Benchmark

end IDR
