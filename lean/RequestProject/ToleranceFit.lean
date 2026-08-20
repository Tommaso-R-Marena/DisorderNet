/-
# Part XCVIII  Fitting within the error bars

Part XLIV proves what an *exact* fit of an ensemble to data is worth: Carathéodory says `n`
measurements are reproduced exactly by `n+1` structures, and any larger pool leaves a continuum of
exact fits.  No real fit is exact.  The assumptions list recorded the gap: "Part XLIV treats
[...] exact agreement rather than agreement within error bars; the practical statement — that the
data leave a large set of ensembles admissible — is the same, but the theorems are proved for the
exact-fit version."  This file proves the error-bar version, and makes the "same" quantitative.

A **tolerance fit** (`IsFitTol`) is a genuine ensemble whose back-calculated averages lie within
the reported error bars `eps i` of the data — which is what every published ensemble is.

* `isFitTol_of_isFit`, `isFitTol_mono` — exact fits are tolerance fits, and loosening the error
  bars can only enlarge the admissible set.  Everything Part XLIV proves about exact fits is
  therefore inherited.
* `fitTol_convex` — the admissible set is convex: agreement within error bars never isolates a
  model.
* `tol_fit_perturb` — **the quantitative statement.**  Around an interior exact fit, every
  direction `u` in weight space that the measurements see with gain at most `G` can be followed a
  distance `t = min(d, eps/G)` and still fit within the error bars.  Where an exact fit is pinned
  by the measured directions, a tolerance fit is free in *every* direction to first order in the
  error bar: the ambiguity of Part XLIV grows by `eps/G`, and it is the error bar divided by the
  sensitivity — the same combination as the precision floor of Part LXXI.
* `tol_fit_not_unique` — hence, as soon as one such direction is nonzero, the fit is not unique:
  an explicitly exhibited second ensemble fits every datum within its error bar.
* `tol_fit_without_exact_fit` — and the two notions genuinely differ in the other direction too:
  data admitting **no** exact fit at all can admit a tolerance fit.  Agreement within error bars is
  strictly weaker evidence than exact agreement, and a `chi²` near one is not a statement that the
  model reproduces the data.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR

open Finset

namespace ToleranceFit

variable {n k : ℕ}

/-- `w` is an exact fit of the data `b`. -/
def IsFit (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) (w : Fin k → ℝ) : Prop :=
  (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ ∀ i, ∑ j, w j * A i j = b i

/-- `w` fits the data to within the reported error bars `eps`. -/
def IsFitTol (A : Fin n → Fin k → ℝ) (b eps : Fin n → ℝ) (w : Fin k → ℝ) : Prop :=
  (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ ∀ i, |(∑ j, w j * A i j) - b i| ≤ eps i

theorem isFitTol_of_isFit (A : Fin n → Fin k → ℝ) (b eps : Fin n → ℝ) (heps : ∀ i, 0 ≤ eps i)
    {w : Fin k → ℝ} (h : IsFit A b w) : IsFitTol A b eps w := by
  refine ⟨h.1, h.2.1, fun i => ?_⟩
  rw [h.2.2 i, sub_self, abs_zero]
  exact heps i

theorem isFitTol_mono (A : Fin n → Fin k → ℝ) (b eps eps' : Fin n → ℝ) (hle : ∀ i, eps i ≤ eps' i)
    {w : Fin k → ℝ} (h : IsFitTol A b eps w) : IsFitTol A b eps' w :=
  ⟨h.1, h.2.1, fun i => (h.2.2 i).trans (hle i)⟩

/-- The admissible set of a fit within error bars is convex. -/
theorem fitTol_convex (A : Fin n → Fin k → ℝ) (b eps : Fin n → ℝ) :
    Convex ℝ {w : Fin k → ℝ | IsFitTol A b eps w} := by
  intro w1 h1 w2 h2 a c ha hc hac
  refine ⟨fun j => ?_, ?_, fun i => ?_⟩
  · have := mul_nonneg ha (h1.1 j)
    have := mul_nonneg hc (h2.1 j)
    simpa using by positivity
  · have : ∑ j, (a * w1 j + c * w2 j) = a * (∑ j, w1 j) + c * (∑ j, w2 j) := by
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
    simpa [h1.2.1, h2.2.1, hac] using this
  · have hlin : ∑ j, (a * w1 j + c * w2 j) * A i j
        = a * (∑ j, w1 j * A i j) + c * (∑ j, w2 j * A i j) := by
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    have hsplit : (∑ j, (a * w1 j + c * w2 j) * A i j) - b i
        = a * ((∑ j, w1 j * A i j) - b i) + c * ((∑ j, w2 j * A i j) - b i) := by
      rw [hlin]
      have : a * b i + c * b i = b i := by rw [← add_mul, hac, one_mul]
      linarith
    have hb : |(∑ j, (a * w1 j + c * w2 j) * A i j) - b i| ≤ eps i := by
      rw [hsplit]
      calc |a * ((∑ j, w1 j * A i j) - b i) + c * ((∑ j, w2 j * A i j) - b i)|
          ≤ |a * ((∑ j, w1 j * A i j) - b i)| + |c * ((∑ j, w2 j * A i j) - b i)| :=
            abs_add_le _ _
        _ = a * |(∑ j, w1 j * A i j) - b i| + c * |(∑ j, w2 j * A i j) - b i| := by
            rw [abs_mul, abs_mul, abs_of_nonneg ha, abs_of_nonneg hc]
        _ ≤ a * eps i + c * eps i := by
            have := h1.2.2 i
            have := h2.2.2 i
            nlinarith [abs_nonneg ((∑ j, w1 j * A i j) - b i),
              abs_nonneg ((∑ j, w2 j * A i j) - b i)]
        _ = eps i := by rw [← add_mul, hac, one_mul]
    simpa using hb

/-- **The error bar buys a move of size `eps/G` in every direction.**  Around an interior exact
fit, any weight direction `u` that the measurements see with gain at most `G` can be followed for
a distance `t ≤ min(d, eps/G)` without leaving the admissible set. -/
theorem tol_fit_perturb (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) {w u : Fin k → ℝ}
    (hfit : IsFit A b w) {d G eps t : ℝ} (hd : ∀ j, d ≤ w j)
    (hu0 : ∑ j, u j = 0) (hu1 : ∀ j, |u j| ≤ 1)
    (hG : ∀ i, |∑ j, u j * A i j| ≤ G) (ht : 0 ≤ t) (htd : t ≤ d) (htG : t * G ≤ eps) :
    IsFitTol A b (fun _ => eps) (fun j => w j + t * u j) := by
  refine ⟨fun j => ?_, ?_, fun i => ?_⟩
  · have h1 : -1 ≤ u j := (abs_le.mp (hu1 j)).1
    have h2 : d ≤ w j := hd j
    nlinarith
  · rw [Finset.sum_add_distrib, hfit.2.1, ← Finset.mul_sum, hu0, mul_zero, add_zero]
  · have hlin : ∑ j, (w j + t * u j) * A i j
        = (∑ j, w j * A i j) + t * ∑ j, u j * A i j := by
      simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hlin, hfit.2.2 i]
    have : (b i + t * ∑ j, u j * A i j) - b i = t * ∑ j, u j * A i j := by ring
    rw [this, abs_mul, abs_of_nonneg ht]
    calc t * |∑ j, u j * A i j| ≤ t * G := mul_le_mul_of_nonneg_left (hG i) ht
      _ ≤ eps := htG

/-- **So the fit is not unique within the error bars**: an explicitly different ensemble fits
every datum to within its error bar. -/
theorem tol_fit_not_unique (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) {w u : Fin k → ℝ}
    (hfit : IsFit A b w) {d G eps t : ℝ} (hd : ∀ j, d ≤ w j)
    (hu0 : ∑ j, u j = 0) (hu1 : ∀ j, |u j| ≤ 1)
    (hG : ∀ i, |∑ j, u j * A i j| ≤ G) (ht : 0 < t) (htd : t ≤ d) (htG : t * G ≤ eps)
    {j0 : Fin k} (hj0 : u j0 ≠ 0) :
    ∃ w' : Fin k → ℝ, IsFitTol A b (fun _ => eps) w' ∧ w' j0 ≠ w j0 := by
  refine ⟨fun j => w j + t * u j, tol_fit_perturb A b hfit hd hu0 hu1 hG ht.le htd htG, ?_⟩
  simp only []
  intro hEq
  have : t * u j0 = 0 := by linarith
  rcases mul_eq_zero.mp this with h | h
  · exact absurd h ht.ne'
  · exact hj0 h

/-! ### Tolerance is strictly weaker than exact agreement -/

/-- A one-conformation pool with one observable that vanishes on it, against a datum of `1/2`. -/
def demoA : Fin 1 → Fin 1 → ℝ := fun _ _ => 0
noncomputable def demoB : Fin 1 → ℝ := fun _ => 1/2
noncomputable def demoEps : Fin 1 → ℝ := fun _ => 1/2
def demoW : Fin 1 → ℝ := fun _ => 1

/-- **Data with no exact fit can have a tolerance fit.**  Agreement within error bars is strictly
weaker evidence than exact agreement: the admissible set can be nonempty when the data are not
reproducible at all. -/
theorem tol_fit_without_exact_fit :
    IsFitTol demoA demoB demoEps demoW ∧ ∀ w : Fin 1 → ℝ, ¬ IsFit demoA demoB w := by
  constructor
  · refine ⟨fun j => by norm_num [demoW], by norm_num [demoW], fun i => ?_⟩
    simp [demoA, demoB, demoEps, demoW]
  · intro w hw
    have := hw.2.2 0
    simp [demoA, demoB] at this

end ToleranceFit

end IDR
