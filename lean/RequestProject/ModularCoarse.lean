/-
# Modular ensembles V: coarse validation hides the price

A modular model is usually judged on coarse readouts — a per-fragment descriptor, a
secondary-structure content, a binned distance — rather than on the full conformational
distribution.  This file shows that such a test is systematically optimistic.

`coarse f p` is the ensemble obtained by describing the first segment only through
`f : X → X'`.  Coarse-graining commutes with the modular construction
(`glue_coarse`: the glued model of the coarse data is the coarse-graining of the glued
model), and

* `cmi_coarse_le` — the seam information of the coarse-grained ensemble is at most that of
  the true one.  The price of modularity measured on coarse data is a *lower* bound on the
  real price, never an upper bound.
* `coarse_validation_can_hide_everything` — and the gap can be total: for the long-range
  contact of `RequestProject.ModularCoupling`, collapsing the description of one terminus
  makes the seam information vanish, while the true seam information is `log 2`.  A modular
  model can pass every coarse test and still be maximally wrong.
-/
import Mathlib
import RequestProject.ModularGluing
import RequestProject.ModularOptimality
import RequestProject.ModularCoupling

namespace RequestProject.Modular

open Finset IDR.Pinsker

/-- The log-sum inequality on an arbitrary finite set of indices. -/
theorem log_sum_inequality_finset {ι : Type*} (s : Finset ι) {a b : ι → ℝ}
    (ha : ∀ i ∈ s, 0 ≤ a i) (hb : ∀ i ∈ s, 0 < b i) :
    (∑ i ∈ s, a i) * Real.log ((∑ i ∈ s, a i) / (∑ i ∈ s, b i))
      ≤ ∑ i ∈ s, a i * Real.log (a i / b i) := by
  rcases s.eq_empty_or_nonempty with rfl | hne
  · simp
  have hBpos : 0 < ∑ i ∈ s, b i := Finset.sum_pos hb hne
  have hAnn : 0 ≤ ∑ i ∈ s, a i := Finset.sum_nonneg ha
  rcases eq_or_lt_of_le hAnn with hA0 | hApos
  · have hzero : ∀ i ∈ s, a i = 0 := fun i hi =>
      le_antisymm (hA0 ▸ Finset.single_le_sum (f := a) ha hi) (ha i hi)
    rw [← hA0]
    simp only [zero_mul]
    refine Finset.sum_nonneg fun i hi => ?_
    rw [hzero i hi, zero_mul]
  · have hR : 0 < (∑ i ∈ s, a i) / (∑ i ∈ s, b i) := div_pos hApos hBpos
    have hRB : ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * (∑ i ∈ s, b i) = ∑ i ∈ s, a i := by
      field_simp
    have key : ∀ i ∈ s, a i * Real.log ((∑ i ∈ s, a i) / (∑ i ∈ s, b i))
        + (a i - ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * b i) ≤ a i * Real.log (a i / b i) := by
      intro i hi
      rcases eq_or_lt_of_le (ha i hi) with h0 | hpos
      · have hgpos : 0 < ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * b i := mul_pos hR (hb i hi)
        rw [← h0]
        simp only [zero_mul, zero_sub, zero_add]
        linarith
      · have hg : 0 < ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * b i := mul_pos hR (hb i hi)
        have hstep := mul_log_div_ge hpos hg
        have hlog : Real.log (a i / (((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * b i))
            = Real.log (a i / b i) - Real.log ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) := by
          rw [show a i / (((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * b i)
              = (a i / b i) / ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) by field_simp,
            Real.log_div (div_ne_zero (ne_of_gt hpos) (ne_of_gt (hb i hi))) (ne_of_gt hR)]
        rw [hlog, mul_sub] at hstep
        linarith
    have hsum := Finset.sum_le_sum key
    have hL : ∑ i ∈ s, (a i * Real.log ((∑ i ∈ s, a i) / (∑ i ∈ s, b i))
        + (a i - ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) * b i))
        = (∑ i ∈ s, a i) * Real.log ((∑ i ∈ s, a i) / (∑ i ∈ s, b i)) := by
      rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hRB]
      ring
    linarith [hsum, hL]

variable {X X' Y Z : Type*} [Fintype X] [Fintype X'] [DecidableEq X'] [Fintype Y] [Fintype Z]

/-- The fibre of the coarse description `f` over a coarse state. -/
def fibre (f : X → X') (x' : X') : Finset X := Finset.univ.filter (fun x => f x = x')

/-- Describing the first segment only through `f`. -/
noncomputable def coarse (f : X → X') (p : X → Y → Z → ℝ) : X' → Y → Z → ℝ :=
  fun x' y z => ∑ x ∈ fibre f x', p x y z

variable {p : X → Y → Z → ℝ} {f : X → X'}

omit [Fintype X'] [Fintype Y] [Fintype Z] in
lemma coarse_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (x' : X') (y : Y) (z : Z) :
    0 ≤ coarse f p x' y z :=
  Finset.sum_nonneg fun _ _ => hp _ _ _

/-- Reordering a triple sum so the first segment is summed last. -/
lemma sum_triple_swap (g : X → Y → Z → ℝ) :
    ∑ x, ∑ y, ∑ z, g x y z = ∑ y, ∑ z, ∑ x, g x y z := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun y _ => Finset.sum_comm

lemma sum_fibre (g : X → ℝ) : ∑ x' : X', ∑ x ∈ fibre f x', g x = ∑ x, g x :=
  Finset.sum_fiberwise Finset.univ f g

omit [Fintype X'] [Fintype Y] in
lemma margXY_coarse (x' : X') (y : Y) :
    margXY (coarse f p) x' y = ∑ x ∈ fibre f x', margXY p x y := by
  simp only [margXY, coarse]
  exact Finset.sum_comm

omit [Fintype Y] [Fintype Z] in
lemma margYZ_coarse (y : Y) (z : Z) : margYZ (coarse f p) y z = margYZ p y z := by
  simp only [margYZ, coarse]
  exact sum_fibre (fun x => p x y z)

omit [Fintype Y] in
lemma margY_coarse (y : Y) : margY (coarse f p) y = margY p y := by
  rw [margY_eq_sum_margYZ, margY_eq_sum_margYZ]
  exact Finset.sum_congr rfl fun z _ => margYZ_coarse y z

omit [Fintype Y] in
/-- Coarse-graining commutes with gluing: the modular model built from coarse fragment data
is the coarse-graining of the modular model built from the full data. -/
lemma glue_coarse (x' : X') (y : Y) (z : Z) :
    glue (coarse f p) x' y z = ∑ x ∈ fibre f x', glue p x y z := by
  rw [glue, margXY_coarse, margYZ_coarse, margY_coarse, Finset.sum_mul, Finset.sum_div]
  exact Finset.sum_congr rfl fun x _ => rfl

/-- **Coarse validation understates the price of modularity.**  The seam information seen
through a coarse description of one segment is at most the true seam information. -/
theorem cmi_coarse_le (hp : ∀ x y z, 0 < p x y z) : cmi (coarse f p) ≤ cmi p := by
  have hp' : ∀ x y z, 0 ≤ p x y z := fun x y z => le_of_lt (hp x y z)
  have hgpos : ∀ x y z, 0 < glue p x y z := fun x y z => glue_pos hp' (hp x y z)
  have hcpos : ∀ x' y z, 0 ≤ coarse f p x' y z := coarse_nonneg hp'
  rw [cmi_eq_sum_log_ratio hcpos, cmi_eq_sum_log_ratio hp',
    sum_triple_swap (fun x' y z => coarse f p x' y z * Real.log
      (coarse f p x' y z / glue (coarse f p) x' y z)),
    sum_triple_swap (fun x y z => p x y z * Real.log (p x y z / glue p x y z))]
  refine Finset.sum_le_sum fun y _ => Finset.sum_le_sum fun z _ => ?_
  have hfib : ∑ x' : X', ∑ x ∈ fibre f x', p x y z * Real.log (p x y z / glue p x y z)
      = ∑ x, p x y z * Real.log (p x y z / glue p x y z) :=
    sum_fibre (fun x => p x y z * Real.log (p x y z / glue p x y z))
  rw [← hfib]
  refine Finset.sum_le_sum fun x' _ => ?_
  have h := log_sum_inequality_finset (fibre f x') (a := fun x => p x y z)
    (b := fun x => glue p x y z) (fun x _ => hp' x y z) (fun x _ => hgpos x y z)
  rw [show (∑ x ∈ fibre f x', p x y z) = coarse f p x' y z from rfl,
    show (∑ x ∈ fibre f x', glue p x y z) = glue (coarse f p) x' y z from
      (glue_coarse x' y z).symm] at h
  exact h

/-! ## The gap can be total -/

/-- Describing a terminus only by "it is there" — the completely coarse readout. -/
def blind : Bool → Unit := fun _ => ()

/-- Under a blind readout of the first terminus the long-range contact ensemble is
conditionally independent across the seam, so its coarse seam information is zero. -/
theorem coarse_longRange_cmi : cmi (coarse blind longRange) = 0 := by
  have hnn : ∀ x' y z, 0 ≤ coarse blind longRange x' y z :=
    coarse_nonneg longRange_nonneg
  have hsum : ∑ x', ∑ y, ∑ z, coarse blind longRange x' y z = 1 := by
    simp [coarse, fibre, blind, longRange]
  refine (cmi_eq_zero_iff_condIndep hnn hsum).2 ?_
  intro x' y z
  have hXY : margXY (coarse blind longRange) x' y = 1 := by
    rw [margXY_coarse]
    simp [fibre, blind, longRange_margXY]
  have hY : margY (coarse blind longRange) y = 1 := by
    rw [margY_coarse]; exact longRange_margY y
  have hYZ : margYZ (coarse blind longRange) y z = coarse blind longRange x' y z := by
    simp [margYZ]
  rw [hXY, hY, hYZ, mul_one, one_mul]

/-- **Coarse validation can hide everything.**  The blind readout of one terminus reports a
seam information of zero — the modular model looks exactly right — while the true seam
information is one full bit and no modular model is within `log 2` of the truth. -/
theorem coarse_validation_can_hide_everything :
    cmi (coarse blind longRange) = 0 ∧ cmi longRange = Real.log 2 ∧ 0 < Real.log 2 :=
  ⟨coarse_longRange_cmi, longRange_cmi, Real.log_pos (by norm_num)⟩

end RequestProject.Modular
