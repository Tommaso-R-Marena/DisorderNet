/-
# Feasibility geometry of a measured distance panel

Pairwise-distance data are the backbone of experimental work on disordered regions:
paramagnetic relaxation enhancement, nuclear Overhauser effects, single-molecule FRET and
crosslinking all report, for a handful of labelled residue pairs `(i, j)`, a number that is an
*average over the ensemble* of a distance between two sites of the chain.  A model of the
region is fitted to such a panel, and the panel is the first thing against which it is checked.

This file asks the prior question: **which panels of numbers can be the mean-distance panel of
an ensemble at all?**  The answer is a set of constraints that a measured panel must satisfy
before any model is even attempted, and which therefore also give a falsification test that is
free of modelling assumptions:

* `meanDist_triangle` -- an ensemble average of distances still satisfies the triangle
  inequality, even though the ensemble is disordered and no single conformation is being
  described.  Averaging destroys many structural constraints; this one survives.
* `meanDist_le_chain` -- connectivity: if consecutive residues are at most `b` apart in every
  conformation, the mean distance between residues `i ≤ j` is at most `b * (j - i)`.  This is
  the contour bound, and it holds for the *average*, so it is directly comparable with data.
* `panel_triangle_bound_of_error` / `no_ensemble_of_triangle_defect` -- with error bars: if the
  measured panel exceeds the triangle inequality by more than three times the measurement
  error, **no** ensemble of conformations whatsoever reproduces the panel within its error
  bars.  The data are then internally inconsistent, and no amount of model flexibility can
  repair them.
* `panel_contour_defect_falsifies` -- likewise for the contour bound.
* `triangle_precision_design_rule` -- the corresponding instrument specification: a triangle
  defect `D` can be turned into a falsification only if the error bar satisfies `e < D / 3`.

Companion file `RequestProject.DistanceCutEnsembles` shows that these necessary conditions are
*not* sufficient, and by a wide margin.
-/
import Mathlib

namespace RequestProject.DistanceRealizability

open Finset

variable {E : Type*} [PseudoMetricSpace E]

/-- A conformational ensemble: `m` conformations `X a : ℕ → E` giving the position of each
residue, with weights `w a`.  `meanDist w X i j` is the ensemble-averaged distance between the
labelled sites `i` and `j` -- the quantity a distance measurement on a disordered region
reports. -/
noncomputable def meanDist {m : ℕ} (w : Fin m → ℝ) (X : Fin m → ℕ → E) (i j : ℕ) : ℝ :=
  ∑ a, w a * dist (X a i) (X a j)

variable {m : ℕ} {w : Fin m → ℝ} {X : Fin m → ℕ → E}

theorem meanDist_nonneg (hw : ∀ a, 0 ≤ w a) (i j : ℕ) : 0 ≤ meanDist w X i j :=
  Finset.sum_nonneg fun a _ => mul_nonneg (hw a) dist_nonneg

theorem meanDist_self (i : ℕ) : meanDist w X i i = 0 := by
  simp [meanDist]

theorem meanDist_comm (i j : ℕ) : meanDist w X i j = meanDist w X j i := by
  simp [meanDist, dist_comm]

/-- **The triangle inequality survives ensemble averaging.**  Every measured panel of mean
distances must satisfy it, whatever the ensemble is. -/
theorem meanDist_triangle (hw : ∀ a, 0 ≤ w a) (i j k : ℕ) :
    meanDist w X i k ≤ meanDist w X i j + meanDist w X j k := by
  rw [meanDist, meanDist, meanDist, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun a _ => ?_
  rw [← mul_add]
  exact mul_le_mul_of_nonneg_left (dist_triangle _ _ _) (hw a)

/-- Connectivity in a single conformation: consecutive residues at most `b` apart forces
`dist (x i) (x j) ≤ b * (j - i)`. -/
theorem dist_le_chain {x : ℕ → E} {b : ℝ} (hb : ∀ k, dist (x k) (x (k + 1)) ≤ b)
    {i j : ℕ} (hij : i ≤ j) : dist (x i) (x j) ≤ b * (j - i : ℕ) := by
  induction j, hij using Nat.le_induction with
  | base => simp
  | succ j hij ih =>
      have h1 : dist (x i) (x (j + 1)) ≤ dist (x i) (x j) + dist (x j) (x (j + 1)) :=
        dist_triangle _ _ _
      have h2 : (j + 1 - i : ℕ) = (j - i : ℕ) + 1 := by omega
      rw [h2]
      push_cast
      nlinarith [hb j, ih]

/-- **The contour bound holds for the average.**  If every conformation of the ensemble has
consecutive residues at most `b` apart, then the *measured* mean distance between residues
`i ≤ j` cannot exceed `b * (j - i)`. -/
theorem meanDist_le_chain {b : ℝ} (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hb : ∀ a k, dist (X a k) (X a (k + 1)) ≤ b) {i j : ℕ} (hij : i ≤ j) :
    meanDist w X i j ≤ b * (j - i : ℕ) := by
  have h : ∀ a ∈ Finset.univ, w a * dist (X a i) (X a j) ≤ w a * (b * (j - i : ℕ)) := by
    intro a _
    exact mul_le_mul_of_nonneg_left (dist_le_chain (hb a) hij) (hw a)
  calc meanDist w X i j ≤ ∑ a, w a * (b * (j - i : ℕ)) := Finset.sum_le_sum h
    _ = b * (j - i : ℕ) := by rw [← Finset.sum_mul, hsum, one_mul]

/-!
## Falsification with error bars

A measurement reports `M i j` with a stated uncertainty `e`.  The triangle inequality then
becomes a test with a threshold: a defect larger than `3 e` is not attributable to noise.
-/

/-- If a panel `M` is within `e` of the mean-distance panel of *some* ensemble, it satisfies the
triangle inequality up to `3 e`. -/
theorem panel_triangle_bound_of_error {M : ℕ → ℕ → ℝ} {e : ℝ} (hw : ∀ a, 0 ≤ w a)
    {i j k : ℕ}
    (hik : |M i k - meanDist w X i k| ≤ e) (hij : |M i j - meanDist w X i j| ≤ e)
    (hjk : |M j k - meanDist w X j k| ≤ e) :
    M i k ≤ M i j + M j k + 3 * e := by
  have t := meanDist_triangle (X := X) hw i j k
  have h1 := abs_le.mp hik
  have h2 := abs_le.mp hij
  have h3 := abs_le.mp hjk
  linarith [h1.1, h1.2, h2.1, h2.2, h3.1, h3.2]

/-- **A triangle defect larger than three error bars falsifies every ensemble.**  No model, of
any flexibility, and no ensemble of conformations in any metric space, reproduces such a panel
within its error bars: the data themselves are inconsistent. -/
theorem no_ensemble_of_triangle_defect {M : ℕ → ℕ → ℝ} {e : ℝ} {i j k : ℕ}
    (hdef : M i j + M j k + 3 * e < M i k) :
    ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → E),
        (∀ a, 0 ≤ w a) ∧ |M i k - meanDist w X i k| ≤ e ∧
        |M i j - meanDist w X i j| ≤ e ∧ |M j k - meanDist w X j k| ≤ e := by
  rintro ⟨m, w, X, hw, hik, hij, hjk⟩
  exact absurd (panel_triangle_bound_of_error hw hik hij hjk) (by linarith)

/-- **Instrument specification.**  To convert an observed triangle defect `D > 0` into a
falsification, the measurement error must satisfy `e < D / 3`; conversely any `e` below that
threshold suffices. -/
theorem triangle_precision_design_rule {M : ℕ → ℕ → ℝ} {e D : ℝ} {i j k : ℕ}
    (hD : M i k - (M i j + M j k) = D) (he : e < D / 3) :
    ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → E),
        (∀ a, 0 ≤ w a) ∧ |M i k - meanDist w X i k| ≤ e ∧
        |M i j - meanDist w X i j| ≤ e ∧ |M j k - meanDist w X j k| ≤ e :=
  no_ensemble_of_triangle_defect (by linarith)

/-- **The contour bound is also a falsification test.**  A mean distance exceeding
`b * (j - i)` by more than the error bar is incompatible with any ensemble of chains whose
consecutive residues are at most `b` apart. -/
theorem panel_contour_defect_falsifies {M : ℕ → ℕ → ℝ} {e b : ℝ} {i j : ℕ} (hij : i ≤ j)
    (hdef : b * (j - i : ℕ) + e < M i j) :
    ¬ ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → E),
        (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧ (∀ a k, dist (X a k) (X a (k + 1)) ≤ b) ∧
        |M i j - meanDist w X i j| ≤ e := by
  rintro ⟨m, w, X, hw, hsum, hb, hM⟩
  have h := meanDist_le_chain hw hsum hb hij
  have h2 := (abs_le.mp hM).2
  linarith

end RequestProject.DistanceRealizability
