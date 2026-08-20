/-
# Part XLV  The benchmark: what an incomplete annotation can and cannot certify

`RequestProject.Benchmark` treats the annotation used to score disorder predictors as what it
is -- a record of the residues shown to be disordered, with everything else unlabelled -- and
derives the consequences for the four numbers a benchmark reports.

`IDR.benchmark_laws` bundles four statements:

1. the reported precision is a lower bound on the true precision;
2. the reported false positives are exactly the true false positives plus the predicted
   disordered residues the annotation has not yet recorded;
3. the reported accuracy is within the annotation error rate of the true accuracy -- what a
   benchmark does certify, quantitatively;
4. but the *ranking* it induces can be inverted: an explicit four-residue example in which the
   predictor that is exactly right about the truth is beaten, on the benchmark, by the predictor
   that merely reproduces the annotation.
-/
import Mathlib
import RequestProject.Benchmark

set_option autoImplicit false

namespace IDR

open IDR.Benchmark

/-- **The benchmark laws for disorder prediction.**

1. *Precision is underreported.*  With sound labels the reported precision never exceeds the
   true precision.
2. *False positives decompose exactly* into true false positives and unannotated discoveries.
3. *Accuracy is certified to the annotation error rate*: `|acc_true − acc_reported| ≤ d/N`.
4. *The ranking is not certified*: on a four-residue chain with sound labels, the predictor that
   matches the truth exactly scores `3/4` and the predictor that matches only the annotation
   scores `1`, so the benchmark order is the reverse of the true order. -/
theorem benchmark_laws :
    (∀ (N : ℕ) (p y l : Fin N → Bool), (∀ i, l i → y i) →
        precision p l ≤ precision p y) ∧
    (∀ (N : ℕ) (p y l : Fin N → Bool), (∀ i, l i → y i) →
        (falsePos p l).card = (falsePos p y).card
          + (Finset.univ.filter fun i : Fin N => p i ∧ y i ∧ ¬ l i).card) ∧
    (∀ (N : ℕ) (p y l : Fin N → Bool),
        |accuracy p y - accuracy p l|
          ≤ ((Finset.univ.filter fun i : Fin N => l i ≠ y i).card : ℝ) / N) ∧
    ((∀ i, label4 i → truth4 i) ∧
      accuracy honest4 truth4 = 1 ∧ accuracy conformist4 truth4 = 3 / 4 ∧
      accuracy honest4 label4 = 3 / 4 ∧ accuracy conformist4 label4 = 1 ∧
      accuracy conformist4 label4 > accuracy honest4 label4 ∧
      accuracy conformist4 truth4 < accuracy honest4 truth4) := by
  exact ⟨fun N p y l h => precision_label_le_precision_truth p y l h,
    fun N p y l h => falsePos_label_eq p y l h,
    fun N p y l => accuracy_close p y l,
    benchmark_can_invert_ranking⟩

end IDR
