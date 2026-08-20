/-
# Part IV.4  What an experiment can and cannot resolve: contraction under measurement

`RequestProject.FiniteData` shows that no finite family of observables determines an
ensemble, and `RequestProject.CoarseGraining` that coarse descriptors are never sufficient.
Those are qualitative.  This file supplies the quantitative mechanism behind them: an
*experiment* is a stochastic channel from conformation space to outcome space, and **every
channel contracts**.  Information about the ensemble can only be destroyed by measuring it,
at a rate that the channel itself fixes.

* `IsChannel`, `push` -- a measurement channel `K` (conformation `j` gives outcome `y` with
  probability `K j y`: a SAXS profile, an NOE, a FRET efficiency histogram, or a
  coarse-graining map, which is the deterministic case) and the push-forward of an ensemble
  through it.
* `log_sum_ineq` -- the log-sum inequality, proved from Gibbs' inequality.
* `dataProcessing` -- **the data-processing inequality**: `KL(Kp ‖ Kq) ≤ KL(p ‖ q)`.  Two
  candidate ensembles are never easier to tell apart *after* an experiment than before, so
  no amount of processing, fitting or post-hoc analysis of the data can recover ensemble
  information the experiment did not transmit.
* `ell1_push_le` -- the same statement in the operational metric of
  `RequestProject.Metric`: measurement contracts the `ℓ¹` distance.
* `ell1_push_contraction` -- **Dobrushin's contraction**, the quantitative form.  If every
  conformation has probability at least `alpha` of producing the *same* outcome
  distribution `nu` -- the overlap that makes an experiment insensitive -- then a distance
  `d` between candidate ensembles is squeezed to at most `(1-alpha)·d` in the data.
* `indistinguishable_radius` -- the design consequence.  With experimental precision `eps`,
  every pair of ensembles closer than `eps/(1-alpha)` is indistinguishable, so an
  experiment defines a *ball of underdetermination* whose radius grows as its overlap grows.
  This is why an ensemble model must report a prior (`RequestProject.MaxEnt`) and why
  fitting an ensemble to data can never be posed as recovering a unique answer.
* `blind_channel_loses_everything` -- the extreme case: a channel whose outcome does not
  depend on the conformation transmits nothing at all.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.Metric

namespace IDR

open Finset
open scoped Classical

namespace Channel

variable {n p : ℕ}

/-- A measurement channel: conformation `j` produces outcome `y` with probability `K j y`.
A deterministic coarse-graining is the special case where each row is a point mass. -/
structure IsChannel (K : Fin n → Fin p → ℝ) : Prop where
  nonneg : ∀ j y, 0 ≤ K j y
  rows : ∀ j, ∑ y, K j y = 1

/-- The distribution of outcomes produced by an ensemble: the push-forward of the
population vector `w` through the channel `K`. -/
def push (K : Fin n → Fin p → ℝ) (w : Fin n → ℝ) : Fin p → ℝ := fun y => ∑ j, w j * K j y

lemma push_nonneg {K : Fin n → Fin p → ℝ} (hK : IsChannel K) {w : Fin n → ℝ}
    (hw : ∀ j, 0 ≤ w j) (y : Fin p) : 0 ≤ push K w y :=
  Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (hK.nonneg j y)

lemma push_sum_one {K : Fin n → Fin p → ℝ} (hK : IsChannel K) {w : Fin n → ℝ}
    (hw : ∑ j, w j = 1) : ∑ y, push K w y = 1 := by
  simp only [push]
  rw [Finset.sum_comm]
  rw [Finset.sum_congr rfl (fun j _ => by rw [← Finset.mul_sum, hK.rows j, mul_one])]
  exact hw

/-! ## The log-sum inequality -/

/-- **The log-sum inequality**, from Gibbs' inequality: pooling terms can only decrease the
relative-entropy contribution. -/
theorem log_sum_ineq {I : Type*} (s : Finset I) (a b : I → ℝ)
    (ha : ∀ i ∈ s, 0 ≤ a i) (hb : ∀ i ∈ s, 0 < b i) (hA : 0 < ∑ i ∈ s, a i) :
    (∑ i ∈ s, a i) * Real.log ((∑ i ∈ s, a i) / (∑ i ∈ s, b i))
      ≤ ∑ i ∈ s, a i * Real.log (a i / b i) := by
  classical
  set A : ℝ := ∑ i ∈ s, a i with hAdef
  have hsne : s.Nonempty := by
    rcases Finset.eq_empty_or_nonempty s with rfl | h
    · simp [hAdef] at hA
    · exact h
  set B : ℝ := ∑ i ∈ s, b i with hBdef
  have hB : 0 < B := Finset.sum_pos hb hsne
  have hgibbs := gibbs_finset s (fun i => a i / A) (fun i => b i / B)
    (fun i hi => div_nonneg (ha i hi) (le_of_lt hA))
    (fun i hi => div_pos (hb i hi) hB)
    (by rw [← Finset.sum_div, ← hAdef]; exact div_self (ne_of_gt hA))
    (by rw [← Finset.sum_div, ← hBdef]; exact div_self (ne_of_gt hB))
  have hterm : ∀ i ∈ s, A * (a i / A * Real.log (a i / A / (b i / B)))
      = a i * Real.log (a i / b i) - a i * Real.log (A / B) := by
    intro i hi
    rcases eq_or_lt_of_le (ha i hi) with h0 | hpos
    · simp [← h0]
    · have hbi := hb i hi
      have hrw : a i / A / (b i / B) = a i / b i * (B / A) := by
        field_simp
      rw [hrw, Real.log_mul (by positivity) (by positivity)]
      have hAB : Real.log (B / A) = -Real.log (A / B) := by
        rw [← Real.log_inv]
        congr 1
        field_simp
      rw [hAB]
      field_simp
      ring
  have hmul : 0 ≤ A * ∑ i ∈ s, a i / A * Real.log (a i / A / (b i / B)) :=
    mul_nonneg (le_of_lt hA) hgibbs
  rw [Finset.mul_sum, Finset.sum_congr rfl hterm, Finset.sum_sub_distrib, ← Finset.sum_mul,
    ← hAdef] at hmul
  linarith

/-! ## Data processing -/

/-- **The data-processing inequality.**  Measuring can only make two candidate ensembles
harder to tell apart: the relative entropy between the outcome distributions never exceeds
the relative entropy between the ensembles themselves.  No analysis of the data can recover
ensemble information the experiment did not transmit. -/
theorem dataProcessing {K : Fin n → Fin p → ℝ} (hK : IsChannel K) {u v : Fin n → ℝ}
    (hu : ∀ j, 0 ≤ u j) (hv : ∀ j, 0 < v j) :
    klDiv (push K u) (push K v) ≤ klDiv u v := by
  classical
  have hkey : ∀ y : Fin p,
      push K u y * Real.log (push K u y / push K v y)
        ≤ ∑ j, u j * K j y * Real.log (u j / v j) := by
    intro y
    set S : Finset (Fin n) := Finset.univ.filter (fun j => 0 < K j y) with hS
    have hzero : ∀ j ∈ (Finset.univ : Finset (Fin n)) \ S, K j y = 0 := by
      intro j hj
      have := (Finset.mem_sdiff.1 hj).2
      have hnot : ¬ (0 < K j y) := by simpa [hS] using this
      exact le_antisymm (not_lt.1 hnot) (hK.nonneg j y)
    have hpushu : push K u y = ∑ j ∈ S, u j * K j y := by
      rw [push]
      refine (Finset.sum_subset (Finset.subset_univ S) ?_).symm
      intro j hj hjS
      rw [hzero j (Finset.mem_sdiff.2 ⟨hj, hjS⟩)]
      ring
    have hpushv : push K v y = ∑ j ∈ S, v j * K j y := by
      rw [push]
      refine (Finset.sum_subset (Finset.subset_univ S) ?_).symm
      intro j hj hjS
      rw [hzero j (Finset.mem_sdiff.2 ⟨hj, hjS⟩)]
      ring
    have hrhs : ∑ j, u j * K j y * Real.log (u j / v j)
        = ∑ j ∈ S, u j * K j y * Real.log (u j / v j) := by
      refine (Finset.sum_subset (Finset.subset_univ S) ?_).symm
      intro j hj hjS
      rw [hzero j (Finset.mem_sdiff.2 ⟨hj, hjS⟩)]
      ring
    rcases eq_or_lt_of_le (push_nonneg hK hu y) with h0 | hpos
    · -- no population reaches outcome `y`
      have hall : ∀ j ∈ S, u j * K j y = 0 := by
        intro j hjS
        have hnn : ∀ j' ∈ S, 0 ≤ u j' * K j' y :=
          fun j' _ => mul_nonneg (hu j') (hK.nonneg j' y)
        have hsum0 : ∑ j' ∈ S, u j' * K j' y = 0 := by rw [← hpushu, ← h0]
        exact (Finset.sum_eq_zero_iff_of_nonneg hnn).1 hsum0 j hjS
      rw [← h0, hrhs, Finset.sum_eq_zero (fun j hj => by rw [hall j hj]; ring)]
      simp
    · have hAS : 0 < ∑ j ∈ S, u j * K j y := by rw [← hpushu]; exact hpos
      have hBS : ∀ j ∈ S, 0 < v j * K j y := by
        intro j hj
        have : 0 < K j y := by simpa [hS] using (Finset.mem_filter.1 hj).2
        exact mul_pos (hv j) this
      have hlog := log_sum_ineq S (fun j => u j * K j y) (fun j => v j * K j y)
        (fun j _ => mul_nonneg (hu j) (hK.nonneg j y)) hBS hAS
      rw [← hpushu, ← hpushv] at hlog
      refine le_trans hlog ?_
      rw [hrhs]
      refine le_of_eq (Finset.sum_congr rfl fun j hj => ?_)
      have hKpos : 0 < K j y := by simpa [hS] using (Finset.mem_filter.1 hj).2
      have : u j * K j y / (v j * K j y) = u j / v j := by
        field_simp
      rw [this]
  calc klDiv (push K u) (push K v)
      = ∑ y, push K u y * Real.log (push K u y / push K v y) := rfl
    _ ≤ ∑ y, ∑ j, u j * K j y * Real.log (u j / v j) := Finset.sum_le_sum fun y _ => hkey y
    _ = klDiv u v := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun j _ => ?_
        have : ∀ y : Fin p, u j * K j y * Real.log (u j / v j)
            = (u j * Real.log (u j / v j)) * K j y := fun y => by ring
        rw [Finset.sum_congr rfl (fun y _ => this y), ← Finset.mul_sum, hK.rows j, mul_one]

/-! ## Contraction in the operational metric -/

/-- Measurement contracts the `ℓ¹` distance: the discrepancy visible in the data never
exceeds the discrepancy between the ensembles. -/
theorem ell1_push_le {K : Fin n → Fin p → ℝ} (hK : IsChannel K) (u v : Fin n → ℝ) :
    ∑ y, |push K u y - push K v y| ≤ ∑ j, |u j - v j| := by
  have hstep : ∀ y : Fin p, |push K u y - push K v y| ≤ ∑ j, |u j - v j| * K j y := by
    intro y
    have hdiff : push K u y - push K v y = ∑ j, (u j - v j) * K j y := by
      simp only [push, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hdiff]
    calc |∑ j, (u j - v j) * K j y| ≤ ∑ j, |(u j - v j) * K j y| :=
          Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, |u j - v j| * K j y := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [abs_mul, abs_of_nonneg (hK.nonneg j y)]
  calc ∑ y, |push K u y - push K v y| ≤ ∑ y, ∑ j, |u j - v j| * K j y :=
        Finset.sum_le_sum fun y _ => hstep y
    _ = ∑ j, |u j - v j| := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [← Finset.mul_sum, hK.rows j, mul_one]

/-- **Dobrushin contraction.**  If every conformation has probability at least `alpha` of
producing one and the same outcome distribution `nu` -- the *insensitivity* of the
experiment -- then the distance between two candidate ensembles is squeezed by the factor
`1 - alpha` on its way into the data. -/
theorem ell1_push_contraction {K : Fin n → Fin p → ℝ} (hK : IsChannel K) {alpha : ℝ}
    {nu : Fin p → ℝ} (hnusum : ∑ y, nu y = 1)
    (hmin : ∀ j y, alpha * nu y ≤ K j y) {u v : Fin n → ℝ} (hsum : ∑ j, u j = ∑ j, v j) :
    ∑ y, |push K u y - push K v y| ≤ (1 - alpha) * ∑ j, |u j - v j| := by
  have hd : ∑ j, (u j - v j) = 0 := by
    rw [Finset.sum_sub_distrib, hsum]
    ring
  have hstep : ∀ y : Fin p,
      |push K u y - push K v y| ≤ ∑ j, |u j - v j| * (K j y - alpha * nu y) := by
    intro y
    have hdiff : push K u y - push K v y = ∑ j, (u j - v j) * (K j y - alpha * nu y) := by
      have h1 : ∑ j, (u j - v j) * (K j y - alpha * nu y)
          = (∑ j, (u j - v j) * K j y) - (∑ j, (u j - v j)) * (alpha * nu y) := by
        rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun j _ => by ring
      rw [h1, hd, zero_mul, sub_zero]
      simp only [push]
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => by ring
    rw [hdiff]
    calc |∑ j, (u j - v j) * (K j y - alpha * nu y)|
        ≤ ∑ j, |(u j - v j) * (K j y - alpha * nu y)| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, |u j - v j| * (K j y - alpha * nu y) := by
          refine Finset.sum_congr rfl fun j _ => ?_
          rw [abs_mul,
            abs_of_nonneg (show (0:ℝ) ≤ K j y - alpha * nu y by linarith [hmin j y])]
  calc ∑ y, |push K u y - push K v y|
      ≤ ∑ y, ∑ j, |u j - v j| * (K j y - alpha * nu y) := Finset.sum_le_sum fun y _ => hstep y
    _ = (1 - alpha) * ∑ j, |u j - v j| := by
        rw [Finset.sum_comm, Finset.mul_sum]
        refine Finset.sum_congr rfl fun j _ => ?_
        have hin : ∀ y : Fin p, |u j - v j| * (K j y - alpha * nu y)
            = |u j - v j| * K j y - |u j - v j| * alpha * nu y := fun y => by ring
        rw [Finset.sum_congr rfl (fun y _ => hin y), Finset.sum_sub_distrib, ← Finset.mul_sum,
          hK.rows j, ← Finset.mul_sum, hnusum]
        ring

/-- **The ball of underdetermination of an experiment.**  With outcome precision `eps`, an
experiment of insensitivity `alpha` cannot separate any two candidate ensembles within
`ℓ¹` distance `eps / (1 - alpha)` of each other.  The more invariant the measurement, the
larger the set of ensembles that fit it equally well -- which is why an ensemble model must
report the prior that selects among them. -/
theorem indistinguishable_radius {K : Fin n → Fin p → ℝ} (hK : IsChannel K) {alpha eps : ℝ}
    {nu : Fin p → ℝ} (hnusum : ∑ y, nu y = 1) (halpha1 : alpha < 1)
    (hmin : ∀ j y, alpha * nu y ≤ K j y) {u v : Fin n → ℝ} (hsum : ∑ j, u j = ∑ j, v j)
    (hclose : ∑ j, |u j - v j| ≤ eps / (1 - alpha)) :
    ∑ y, |push K u y - push K v y| ≤ eps := by
  have h1 := ell1_push_contraction hK hnusum hmin hsum
  have hpos : 0 < 1 - alpha := by linarith
  have h2 : (1 - alpha) * ∑ j, |u j - v j| ≤ (1 - alpha) * (eps / (1 - alpha)) :=
    mul_le_mul_of_nonneg_left hclose (le_of_lt hpos)
  rw [mul_div_cancel₀ _ (ne_of_gt hpos)] at h2
  linarith

/-- The extreme case: an experiment whose outcome distribution does not depend on the
conformation transmits nothing -- every ensemble produces the same data. -/
theorem blind_channel_loses_everything {nu : Fin p → ℝ} {u v : Fin n → ℝ}
    (hsum : ∑ j, u j = ∑ j, v j) :
    push (fun _ y => nu y) u = push (fun _ y => nu y) v := by
  funext y
  simp only [push, ← Finset.sum_mul, hsum]

end Channel

end IDR
