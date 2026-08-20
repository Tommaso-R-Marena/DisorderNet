/-
# Part XCIV  Class imbalance, and what a benchmark number is an average of

Parts XLV and LIX price what an *incomplete* or *unsound* annotation does to a disorder
benchmark, but both count residues with equal weight and both use a single figure of merit.  The
assumptions list recorded the gap: "class imbalance, per-protein aggregation, and the particular
figures of merit used by real assessments (balanced accuracy, Matthews correlation) are not
modelled".  This file models them.

Everything is stated for a confusion matrix `CM` — four nonnegative counts `tp, fp, fn, tn` — which
is all any residue-level assessment reduces to.

* `accuracy_eq_mix` — **the identity that explains the problem.**  Accuracy is exactly the
  prevalence-weighted mixture `prevalence·sensitivity + (1 − prevalence)·specificity`.  On a
  benchmark that is 10% disordered, ninety per cent of the score is the score on ordered residues.
* `allNeg_accuracy`, `allNeg_bacc`, `allNeg_mcc`, `allPos_*` — **the trivial predictors.**  The
  predictor that calls nothing disordered has accuracy `1 − prevalence` (0.9 on a 10% benchmark),
  balanced accuracy exactly `1/2`, and Matthews correlation exactly `0`; the predictor that calls
  everything disordered has accuracy `prevalence`, and again `1/2` and `0`.  So the two
  imbalance-robust scores, and only they, assign the vacuous answer its correct value.
* `imbalance_inversion` — **and the difference is not academic.**  An explicit pair on a
  10%-disordered benchmark of 100 residues: the vacuous predictor scores accuracy `0.9`, while a
  genuine predictor that finds 8 of the 10 disordered residues scores `0.83`.  Accuracy ranks the
  vacuous one first; balanced accuracy and Matthews correlation both rank it last.  A leaderboard
  sorted on accuracy over an imbalanced benchmark is measuring the prevalence.
* `mcc_le_one`, `neg_one_le_mcc`, `mcc_eq_zero_iff` — the Matthews correlation is a genuine
  correlation: it lies in `[-1, 1]`, and it vanishes exactly when the prediction and the truth are
  independent in the confusion table (`tp·tn = fp·fn`).
* `macro_micro_inversion` — **per-protein aggregation is a separate choice, and it can invert the
  ranking on its own.**  Two predictors on the same two proteins (100 residues and 10 residues):
  the first wins the per-protein average (`0.75` against `0.5`), the second wins the pooled
  residue count (`64/110` against `60/110`).  Neither number is wrong; they answer different
  questions, and an assessment that does not say which it reports has not reported a ranking.

The reading, alongside Parts XLV and LIX: a disorder benchmark reports the annotation, weighted by
the class composition of the benchmark and by the size distribution of its proteins.  Only a
prevalence-free figure of merit measures the predictor, and only a stated aggregation rule makes
the comparison reproducible.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.Imbalance

/-- A residue-level confusion matrix. -/
structure CM where
  tp : ℝ
  fp : ℝ
  fn : ℝ
  tn : ℝ

namespace CM

/-- Total number of residues scored. -/
def total (c : CM) : ℝ := c.tp + c.fp + c.fn + c.tn

/-- Fraction of the benchmark that is genuinely disordered. -/
noncomputable def prevalence (c : CM) : ℝ := (c.tp + c.fn) / c.total

/-- Fraction of residues classified correctly. -/
noncomputable def accuracy (c : CM) : ℝ := (c.tp + c.tn) / c.total

/-- True positive rate. -/
noncomputable def sens (c : CM) : ℝ := c.tp / (c.tp + c.fn)

/-- True negative rate. -/
noncomputable def spec (c : CM) : ℝ := c.tn / (c.tn + c.fp)

/-- Balanced accuracy: the unweighted mean of the two class-conditional rates. -/
noncomputable def bacc (c : CM) : ℝ := (c.sens + c.spec) / 2

/-- Matthews correlation coefficient. -/
noncomputable def mcc (c : CM) : ℝ :=
  (c.tp * c.tn - c.fp * c.fn) /
    Real.sqrt ((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn))

/-- Pooling two per-protein confusion matrices (micro-averaging). -/
def pool (c d : CM) : CM := ⟨c.tp + d.tp, c.fp + d.fp, c.fn + d.fn, c.tn + d.tn⟩

/-- The per-protein average of the accuracies (macro-averaging). -/
noncomputable def macroAcc (c d : CM) : ℝ := (c.accuracy + d.accuracy) / 2

end CM

open CM

/-- **Accuracy is a prevalence-weighted mixture of the two class-conditional rates.**  This
identity is the whole of the imbalance problem: as the disordered class shrinks, the score is
carried by the ordered class. -/
theorem accuracy_eq_mix (c : CM) (hp : 0 < c.tp + c.fn) (hn : 0 < c.tn + c.fp)
    (ht : 0 < c.total) :
    c.accuracy = c.prevalence * c.sens + (1 - c.prevalence) * c.spec := by
  have hpne : c.tp + c.fn ≠ 0 := ne_of_gt hp
  have hnne : c.tn + c.fp ≠ 0 := ne_of_gt hn
  have htne : c.total ≠ 0 := ne_of_gt ht
  unfold CM.accuracy CM.prevalence CM.sens CM.spec CM.total at *
  field_simp
  ring

/-! ### The vacuous predictors -/

/-- The predictor that calls no residue disordered, on a benchmark with `P` disordered and `N`
ordered residues. -/
def allNeg (P N : ℝ) : CM := ⟨0, 0, P, N⟩

/-- The predictor that calls every residue disordered. -/
def allPos (P N : ℝ) : CM := ⟨P, N, 0, 0⟩

theorem allNeg_prevalence (P N : ℝ) : (allNeg P N).prevalence = P / (P + N) := by
  unfold CM.prevalence CM.total allNeg
  simp

/-- **The vacuous predictor's accuracy is one minus the prevalence.** -/
theorem allNeg_accuracy {P N : ℝ} (h : 0 < P + N) :
    (allNeg P N).accuracy = 1 - (allNeg P N).prevalence := by
  have hne : P + N ≠ 0 := ne_of_gt h
  unfold CM.accuracy CM.prevalence CM.total allNeg
  simp
  field_simp
  ring

/-- **Balanced accuracy sees through it: exactly one half, at every prevalence.** -/
theorem allNeg_bacc (P : ℝ) {N : ℝ} (hN : 0 < N) : (allNeg P N).bacc = 1 / 2 := by
  unfold CM.bacc CM.sens CM.spec allNeg
  simp
  rw [div_self (ne_of_gt hN)]
  norm_num

/-- **And so does the Matthews correlation: exactly zero.** -/
theorem allNeg_mcc (P N : ℝ) : (allNeg P N).mcc = 0 := by
  unfold CM.mcc allNeg
  simp

theorem allPos_accuracy (P N : ℝ) :
    (allPos P N).accuracy = (allPos P N).prevalence := by
  unfold CM.accuracy CM.prevalence CM.total allPos
  simp

theorem allPos_bacc {P : ℝ} (hP : 0 < P) (N : ℝ) : (allPos P N).bacc = 1 / 2 := by
  unfold CM.bacc CM.sens CM.spec allPos
  simp
  rw [div_self (ne_of_gt hP)]
  norm_num

theorem allPos_mcc (P N : ℝ) : (allPos P N).mcc = 0 := by
  unfold CM.mcc allPos
  simp

/-! ### The Matthews correlation is a correlation -/

private lemma mcc_num_sq_le (c : CM) (h1 : 0 ≤ c.tp) (h2 : 0 ≤ c.fp) (h3 : 0 ≤ c.fn)
    (h4 : 0 ≤ c.tn) :
    (c.tp * c.tn - c.fp * c.fn) ^ 2
      ≤ (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn) := by
  nlinarith [mul_nonneg h1 h2, mul_nonneg h1 h3, mul_nonneg h1 h4, mul_nonneg h2 h3,
    mul_nonneg h2 h4, mul_nonneg h3 h4, sq_nonneg (c.tp * c.tn - c.fp * c.fn),
    sq_nonneg (c.tp * c.tn + c.fp * c.fn), mul_nonneg (mul_nonneg h1 h2) h3,
    mul_nonneg (mul_nonneg h1 h2) h4, mul_nonneg (mul_nonneg h1 h3) h4,
    mul_nonneg (mul_nonneg h2 h3) h4]

theorem mcc_le_one (c : CM) (h1 : 0 ≤ c.tp) (h2 : 0 ≤ c.fp) (h3 : 0 ≤ c.fn) (h4 : 0 ≤ c.tn) :
    c.mcc ≤ 1 := by
  set D : ℝ := (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn) with hD
  have hDnn : 0 ≤ D := by positivity
  rcases eq_or_lt_of_le hDnn with h | h
  · unfold CM.mcc
    rw [← hD, ← h]
    simp
  · have hs : 0 < Real.sqrt D := Real.sqrt_pos.mpr h
    have hnum : c.tp * c.tn - c.fp * c.fn ≤ Real.sqrt D := by
      have := mcc_num_sq_le c h1 h2 h3 h4
      have habs : |c.tp * c.tn - c.fp * c.fn| ≤ Real.sqrt D := by
        rw [← Real.sqrt_sq_eq_abs]
        exact Real.sqrt_le_sqrt (by rw [hD]; exact this)
      exact (le_abs_self _).trans habs
    unfold CM.mcc
    rw [← hD, div_le_one hs]
    exact hnum

theorem neg_one_le_mcc (c : CM) (h1 : 0 ≤ c.tp) (h2 : 0 ≤ c.fp) (h3 : 0 ≤ c.fn)
    (h4 : 0 ≤ c.tn) : -1 ≤ c.mcc := by
  set D : ℝ := (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn) with hD
  have hDnn : 0 ≤ D := by positivity
  rcases eq_or_lt_of_le hDnn with h | h
  · unfold CM.mcc
    rw [← hD, ← h]
    simp
  · have hs : 0 < Real.sqrt D := Real.sqrt_pos.mpr h
    have hnum : -(Real.sqrt D) ≤ c.tp * c.tn - c.fp * c.fn := by
      have := mcc_num_sq_le c h1 h2 h3 h4
      have habs : |c.tp * c.tn - c.fp * c.fn| ≤ Real.sqrt D := by
        rw [← Real.sqrt_sq_eq_abs]
        exact Real.sqrt_le_sqrt (by rw [hD]; exact this)
      have := (abs_le.mp habs).1
      linarith
    unfold CM.mcc
    rw [← hD, le_div_iff₀ hs]
    linarith

/-- The Matthews correlation vanishes exactly when the confusion table factorises: the prediction
carries no information about the truth. -/
theorem mcc_eq_zero_iff (c : CM)
    (hD : 0 < (c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)) :
    c.mcc = 0 ↔ c.tp * c.tn = c.fp * c.fn := by
  have hs : 0 < Real.sqrt ((c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn)) :=
    Real.sqrt_pos.mpr hD
  unfold CM.mcc
  rw [div_eq_zero_iff]
  constructor
  · rintro (h | h)
    · linarith
    · exact absurd h (ne_of_gt hs)
  · intro h
    left
    linarith

/-! ### The inversion -/

/-- A genuine predictor on a 100-residue benchmark that is 10% disordered: it finds 8 of the 10
disordered residues at the cost of 15 false positives. -/
def realistic : CM := ⟨8, 15, 2, 75⟩

/-- **Accuracy ranks the vacuous predictor above a genuine one; the imbalance-robust scores rank
it below.**  On a 100-residue benchmark with 10 disordered residues, the predictor that calls
nothing disordered scores accuracy `9/10`, balanced accuracy `1/2` and Matthews correlation `0`;
the predictor that finds 8 of the 10 scores accuracy `83/100 < 9/10`, but balanced accuracy
`49/60 > 1/2` and a strictly positive Matthews correlation. -/
theorem imbalance_inversion :
    realistic.accuracy < (allNeg 10 90).accuracy ∧
    (allNeg 10 90).bacc < realistic.bacc ∧
    (allNeg 10 90).mcc < realistic.mcc := by
  refine ⟨?_, ?_, ?_⟩
  · unfold CM.accuracy CM.total realistic allNeg
    norm_num
  · rw [allNeg_bacc 10 (by norm_num : (0:ℝ) < 90)]
    unfold CM.bacc CM.sens CM.spec realistic
    norm_num
  · rw [allNeg_mcc]
    unfold CM.mcc realistic
    apply div_pos (by norm_num)
    apply Real.sqrt_pos.mpr
    norm_num

/-! ### Aggregation -/

/-- Predictor A on a 100-residue protein and a 10-residue protein. -/
def protA1 : CM := ⟨50, 50, 0, 0⟩
def protA2 : CM := ⟨10, 0, 0, 0⟩

/-- Predictor B on the same two proteins. -/
def protB1 : CM := ⟨60, 40, 0, 0⟩
def protB2 : CM := ⟨4, 6, 0, 0⟩

/-- **The aggregation rule alone can invert the ranking.**  Predictor A wins the per-protein
average of accuracies, predictor B wins the pooled residue count — on the very same two proteins,
with the very same predictions. -/
theorem macro_micro_inversion :
    CM.macroAcc protB1 protB2 < CM.macroAcc protA1 protA2 ∧
    (CM.pool protA1 protA2).accuracy < (CM.pool protB1 protB2).accuracy := by
  constructor
  · unfold CM.macroAcc CM.accuracy CM.total protA1 protA2 protB1 protB2
    norm_num
  · unfold CM.pool CM.accuracy CM.total protA1 protA2 protB1 protB2
    norm_num

end IDR.Imbalance
