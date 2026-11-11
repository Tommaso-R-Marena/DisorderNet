/-
# Modular ensembles II: the glued model is the best modular model

`RequestProject.ModularGluing` showed that joining two fragment panels through their shared
seam costs exactly the conditional mutual information across the seam.  One could hope that
a cleverer modular construction — some other ensemble that is conditionally independent
across the seam, fitted by whatever means — would do better.  It cannot.

* `log_sum_inequality` — the log-sum inequality, proved from Gibbs' pointwise bound.
* `klXY_ge_klY` — relative entropy decreases under marginalisation (data processing for the
  seam variable).
* `no_modular_model_beats_seam_information` — for **every** strictly positive ensemble `q`
  that is conditionally independent across the seam, `cmi p ≤ KL(p‖q)`.  The seam
  information is a floor on the error of every modular model, not an artefact of the
  particular gluing rule; and by `klG_glue_eq_cmi` the glued model attains it.
-/
import Mathlib
import RequestProject.ModularGluing

namespace RequestProject.Modular

open Finset IDR.Pinsker

/-- **The log-sum inequality.**  Merging two nonnegative weight vectors can only decrease
their relative entropy. -/
theorem log_sum_inequality {ι : Type*} [Fintype ι] [Nonempty ι] {a b : ι → ℝ}
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 < b i) :
    (∑ i, a i) * Real.log ((∑ i, a i) / (∑ i, b i)) ≤ ∑ i, a i * Real.log (a i / b i) := by
  have hBpos : 0 < ∑ i, b i :=
    Finset.sum_pos (fun i _ => hb i) ⟨Classical.arbitrary ι, Finset.mem_univ _⟩
  have hAnn : 0 ≤ ∑ i, a i := Finset.sum_nonneg fun i _ => ha i
  rcases eq_or_lt_of_le hAnn with hA0 | hApos
  · have hzero : ∀ i, a i = 0 := fun i =>
      le_antisymm (hA0 ▸ Finset.single_le_sum (f := a) (fun j _ => ha j) (Finset.mem_univ i))
        (ha i)
    simp [hzero]
  · have hR : 0 < (∑ i, a i) / (∑ i, b i) := div_pos hApos hBpos
    have hRB : ((∑ i, a i) / (∑ i, b i)) * (∑ i, b i) = ∑ i, a i := by field_simp
    have key : ∀ i, a i * Real.log ((∑ i, a i) / (∑ i, b i))
        + (a i - ((∑ i, a i) / (∑ i, b i)) * b i) ≤ a i * Real.log (a i / b i) := by
      intro i
      rcases eq_or_lt_of_le (ha i) with h0 | hpos
      · have hgpos : 0 < ((∑ i, a i) / (∑ i, b i)) * b i := mul_pos hR (hb i)
        rw [← h0]
        simp only [zero_mul, zero_sub, zero_add]
        linarith
      · have hg : 0 < ((∑ i, a i) / (∑ i, b i)) * b i := mul_pos hR (hb i)
        have hstep := mul_log_div_ge hpos hg
        have hlog : Real.log (a i / (((∑ i, a i) / (∑ i, b i)) * b i))
            = Real.log (a i / b i) - Real.log ((∑ i, a i) / (∑ i, b i)) := by
          rw [show a i / (((∑ i, a i) / (∑ i, b i)) * b i)
              = (a i / b i) / ((∑ i, a i) / (∑ i, b i)) by
            field_simp, Real.log_div (div_ne_zero (ne_of_gt hpos) (ne_of_gt (hb i))) (ne_of_gt hR)]
        rw [hlog, mul_sub] at hstep
        linarith
    have hsum := Finset.sum_le_sum fun i (_ : i ∈ Finset.univ) => key i
    have hL : ∑ i, (a i * Real.log ((∑ i, a i) / (∑ i, b i))
        + (a i - ((∑ i, a i) / (∑ i, b i)) * b i))
        = (∑ i, a i) * Real.log ((∑ i, a i) / (∑ i, b i)) := by
      rw [Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_sub_distrib, ← Finset.mul_sum,
        hRB]
      ring
    linarith [hsum, hL]

/-- The logarithmic bookkeeping behind the excess of a modular model over the glued one. -/
lemma log_ratio_identity {a pX pZ pY qX qZ qY : ℝ}
    (ha : 0 < a) (hpX : 0 < pX) (hpZ : 0 < pZ) (hpY : 0 < pY)
    (hqX : 0 < qX) (hqZ : 0 < qZ) (hqY : 0 < qY) :
    Real.log (a / (qX * qZ / qY))
      = Real.log (a * pY / (pX * pZ))
        + (Real.log (pX / qX) + Real.log (pZ / qZ) - Real.log (pY / qY)) := by
  rw [Real.log_div (ne_of_gt ha) (div_ne_zero (mul_ne_zero (ne_of_gt hqX) (ne_of_gt hqZ))
      (ne_of_gt hqY)),
    Real.log_div (mul_ne_zero (ne_of_gt hqX) (ne_of_gt hqZ)) (ne_of_gt hqY),
    Real.log_mul (ne_of_gt hqX) (ne_of_gt hqZ),
    Real.log_div (mul_ne_zero (ne_of_gt ha) (ne_of_gt hpY))
      (mul_ne_zero (ne_of_gt hpX) (ne_of_gt hpZ)),
    Real.log_mul (ne_of_gt ha) (ne_of_gt hpY), Real.log_mul (ne_of_gt hpX) (ne_of_gt hpZ),
    Real.log_div (ne_of_gt hpX) (ne_of_gt hqX), Real.log_div (ne_of_gt hpZ) (ne_of_gt hqZ),
    Real.log_div (ne_of_gt hpY) (ne_of_gt hqY)]
  ring

variable {X Y Z : Type*} [Fintype X] [Fintype Y] [Fintype Z]
variable [Nonempty X] [Nonempty Y] [Nonempty Z]
variable {p q : X → Y → Z → ℝ}

omit [Fintype X] [Fintype Y] [Nonempty X] [Nonempty Y] in
lemma margXY_pos_of_pos (hq : ∀ x y z, 0 < q x y z) (x : X) (y : Y) : 0 < margXY q x y :=
  Finset.sum_pos (fun z _ => hq x y z) ⟨Classical.arbitrary Z, Finset.mem_univ _⟩

omit [Fintype Y] [Fintype Z] [Nonempty Y] [Nonempty Z] in
lemma margYZ_pos_of_pos (hq : ∀ x y z, 0 < q x y z) (y : Y) (z : Z) : 0 < margYZ q y z :=
  Finset.sum_pos (fun x _ => hq x y z) ⟨Classical.arbitrary X, Finset.mem_univ _⟩

omit [Fintype Y] [Nonempty Y] in
lemma margY_pos_of_pos (hq : ∀ x y z, 0 < q x y z) (y : Y) : 0 < margY q y :=
  lt_of_lt_of_le (margXY_pos_of_pos hq (Classical.arbitrary X) y)
    (margXY_le_margY (fun x y z => le_of_lt (hq x y z)) _ y)

omit [Nonempty Y] in
/-- Relative entropy of the fragment panels dominates that of the seam laws: marginalising
away a segment can only lose discrimination. -/
theorem klXY_ge_klY (hp : ∀ x y z, 0 ≤ p x y z) (hq : ∀ x y z, 0 < q x y z) :
    ∑ y, margY p y * Real.log (margY p y / margY q y)
      ≤ ∑ y, ∑ x, margXY p x y * Real.log (margXY p x y / margXY q x y) := by
  refine Finset.sum_le_sum fun y _ => ?_
  have h := log_sum_inequality (a := fun x => margXY p x y) (b := fun x => margXY q x y)
    (fun x => margXY_nonneg hp x y) (fun x => margXY_pos_of_pos hq x y)
  rw [← margY_eq_sum_margXY, ← margY_eq_sum_margXY] at h
  exact h

omit [Nonempty X] [Nonempty Y] [Nonempty Z] in
/-- Rewriting a triple sum as a sum over the second fragment's panel. -/
lemma sum_prod_margYZ (p : X → Y → Z → ℝ) :
    ∑ t : Y × Z, margYZ p t.1 t.2 = ∑ x, ∑ y, ∑ z, p x y z := by
  rw [Fintype.sum_prod_type]
  have h1 : ∀ y : Y, ∑ z, margYZ p y z = ∑ x, ∑ z, p x y z := by
    intro y
    simp only [margYZ]
    exact Finset.sum_comm
  rw [Finset.sum_congr rfl fun y _ => h1 y, Finset.sum_comm]

omit [Nonempty Y] [Nonempty Z] in
/-- Gibbs' inequality for the second fragment's panel. -/
theorem klYZ_nonneg (hp : ∀ x y z, 0 ≤ p x y z) (hq : ∀ x y z, 0 < q x y z)
    (hps : ∑ x, ∑ y, ∑ z, p x y z = 1) (hqs : ∑ x, ∑ y, ∑ z, q x y z = 1) :
    0 ≤ ∑ y, ∑ z, margYZ p y z * Real.log (margYZ p y z / margYZ q y z) := by
  have hsum : ∑ t : Y × Z, margYZ p t.1 t.2 = 1 := by rw [sum_prod_margYZ p, hps]
  have hsumq : ∑ t : Y × Z, margYZ q t.1 t.2 = 1 := by rw [sum_prod_margYZ q, hqs]
  have hnn := klG_nonneg (p := fun t : Y × Z => margYZ p t.1 t.2)
    (q := fun t : Y × Z => margYZ q t.1 t.2)
    (fun t => margYZ_nonneg hp t.1 t.2)
    (fun t => margYZ_pos_of_pos hq t.1 t.2) hsum hsumq
  rw [klG, Fintype.sum_prod_type] at hnn
  exact hnn

omit [Nonempty X] [Nonempty Y] [Nonempty Z] in
/-- Collapsing a triple sum weighted by `p` onto the first fragment's panel. -/
lemma sum_triple_margXY (p : X → Y → Z → ℝ) (f : X → Y → ℝ) :
    ∑ x, ∑ y, ∑ z, p x y z * f x y = ∑ x, ∑ y, margXY p x y * f x y := by
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  rw [margXY, Finset.sum_mul]

omit [Nonempty X] [Nonempty Y] [Nonempty Z] in
/-- Collapsing a triple sum weighted by `p` onto the second fragment's panel. -/
lemma sum_triple_margYZ (p : X → Y → Z → ℝ) (f : Y → Z → ℝ) :
    ∑ x, ∑ y, ∑ z, p x y z * f y z = ∑ y, ∑ z, margYZ p y z * f y z := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun z _ => ?_
  rw [margYZ, Finset.sum_mul]

omit [Nonempty X] [Nonempty Y] [Nonempty Z] in
/-- Collapsing a triple sum weighted by `p` onto the seam law. -/
lemma sum_triple_margY (p : X → Y → Z → ℝ) (f : Y → ℝ) :
    ∑ x, ∑ y, ∑ z, p x y z * f y = ∑ y, margY p y * f y := by
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [margY, Finset.sum_mul]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.sum_mul]

omit [Nonempty Y] in
/-- **No modular model beats the seam information.**  Every strictly positive ensemble that
is conditionally independent across the seam — that is, every ensemble a modular
construction can produce, however it was fitted — is at relative entropy at least `cmi p`
from the truth.  The glued model attains the floor (`klG_glue_eq_cmi`), so the price of
modularity is intrinsic to the cut, not to the joining rule. -/
theorem no_modular_model_beats_seam_information
    (hp : ∀ x y z, 0 ≤ p x y z) (hq : ∀ x y z, 0 < q x y z)
    (hps : ∑ x, ∑ y, ∑ z, p x y z = 1) (hqs : ∑ x, ∑ y, ∑ z, q x y z = 1)
    (hci : CondIndep q) :
    cmi p ≤ ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / q x y z) := by
  have hqXY : ∀ x y, 0 < margXY q x y := margXY_pos_of_pos hq
  have hqYZ : ∀ y z, 0 < margYZ q y z := margYZ_pos_of_pos hq
  have hqY : ∀ y, 0 < margY q y := margY_pos_of_pos hq
  have hsplit : ∀ x y z, p x y z * Real.log (p x y z / q x y z)
      = p x y z * Real.log (p x y z * margY p y / (margXY p x y * margYZ p y z))
        + (p x y z * Real.log (margXY p x y / margXY q x y)
          + p x y z * Real.log (margYZ p y z / margYZ q y z)
          - p x y z * Real.log (margY p y / margY q y)) := by
    intro x y z
    rcases eq_or_lt_of_le (hp x y z) with h0 | hpos
    · simp [← h0]
    · obtain ⟨hxy, hyz, hy⟩ := pos_of_pos hp hpos
      have hqeq : q x y z = margXY q x y * margYZ q y z / margY q y := by
        rw [eq_div_iff (ne_of_gt (hqY y))]
        exact hci x y z
      rw [hqeq, log_ratio_identity hpos hxy hyz hy (hqXY x y) (hqYZ y z) (hqY y)]
      ring
  have hKL : ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / q x y z)
      = cmi p + ((∑ x, ∑ y, margXY p x y * Real.log (margXY p x y / margXY q x y))
        + (∑ y, ∑ z, margYZ p y z * Real.log (margYZ p y z / margYZ q y z))
        - ∑ y, margY p y * Real.log (margY p y / margY q y)) := by
    have e1 : ∑ x, ∑ y, ∑ z, p x y z * Real.log (p x y z / q x y z)
        = (∑ x, ∑ y, ∑ z, p x y z
            * Real.log (p x y z * margY p y / (margXY p x y * margYZ p y z)))
          + ((∑ x, ∑ y, ∑ z, p x y z * Real.log (margXY p x y / margXY q x y))
            + (∑ x, ∑ y, ∑ z, p x y z * Real.log (margYZ p y z / margYZ q y z))
            - ∑ x, ∑ y, ∑ z, p x y z * Real.log (margY p y / margY q y)) := by
      rw [Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ =>
        Finset.sum_congr rfl fun z _ => hsplit x y z]
      simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [e1, cmi, sum_triple_margXY p (fun x y => Real.log (margXY p x y / margXY q x y)),
      sum_triple_margYZ p (fun y z => Real.log (margYZ p y z / margYZ q y z)),
      sum_triple_margY p (fun y => Real.log (margY p y / margY q y))]
  rw [hKL]
  have h1 := klXY_ge_klY (p := p) (q := q) hp hq
  have h2 := klYZ_nonneg (p := p) (q := q) hp hq hps hqs
  have h1' : ∑ y, margY p y * Real.log (margY p y / margY q y)
      ≤ ∑ x, ∑ y, margXY p x y * Real.log (margXY p x y / margXY q x y) := by
    rw [Finset.sum_comm]
    exact h1
  linarith

end RequestProject.Modular
