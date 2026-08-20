/-
# Part V.1  The price of reweighting: chi-squared divergence and effective sample size

`RequestProject.MaxEnt` shows *what* an experimentally refined ensemble is (the minimum
relative-entropy tilt of the reference ensemble).  It says nothing about whether that
answer can be *computed* from a finite simulation.  In practice one has `N` frames drawn
from a force-field ensemble `q` and wants averages under the refined ensemble `p`; these
are obtained by importance reweighting, with weights `w j = p j / q j`.

This file proves the exact statistical cost of that operation.

* `chiSq p q` is the chi-squared divergence, and `expect_iw`, `expect_iw_sq` compute the
  first two moments of the importance weight under the reference ensemble: `1` and
  `1 + chiSq p q`.
* `essFrac_eq`: Kish's effective sample size fraction is **exactly** `1 / (1 + chiSq p q)`.
  So `N` frames of `q` are worth `N / (1 + chiSq p q)` frames of `p`.
* `kl_le_log_one_add_chiSq` and `essFrac_le_exp_neg_kl`: the effective fraction is at most
  `exp (-KL(p‖q))`.  Reweighting therefore decays **exponentially in the relative entropy**
  between the refined and the reference ensemble: an experiment that moves the ensemble by
  `K` nats costs a factor `e^K` in simulation length (`frames_needed`).
* `ell1_sq_le_chiSq`: the cost is already visible in the operational (ℓ¹) metric of
  `RequestProject.Metric`.
* `chiSq_unif_dirac`: for a maximally disordered reference the cost of shifting weight onto
  a single conformation is linear in the size of the library, `chiSq = m - 1`.  This is the
  quantitative form of the folklore statement that reweighting cannot rescue a prior that
  does not already populate the relevant conformations -- the qualitative version of which
  is `no_reweight_of_unpopulated` in `RequestProject.Binding`.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.Metric

namespace IDR

open Finset
open scoped Classical

namespace Reweight

variable {m : ℕ}

/-- The chi-squared divergence of the target `p` from the reference `q`. -/
noncomputable def chiSq (p q : Fin m → ℝ) : ℝ := ∑ j, (p j - q j) ^ 2 / q j

/-- The importance weight of a library conformation. -/
noncomputable def iw (p q : Fin m → ℝ) : Fin m → ℝ := fun j => p j / q j

@[simp] lemma iw_apply (p q : Fin m → ℝ) (j : Fin m) : iw p q j = p j / q j := rfl

lemma chiSq_nonneg {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) : 0 ≤ chiSq p q :=
  Finset.sum_nonneg fun j _ => div_nonneg (sq_nonneg _) (hq j).le

/-- The mean importance weight under the reference ensemble is one: reweighting is
unbiased. -/
lemma expect_iw {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1) :
    ∑ j, q j * iw p q j = 1 := by
  rw [← hps]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [iw_apply, mul_comm, div_mul_cancel₀ _ (hq j).ne']

/-- `∑ p²/q = 1 + χ²`. -/
lemma sum_sq_div {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1)
    (hqs : ∑ j, q j = 1) :
    ∑ j, p j ^ 2 / q j = 1 + chiSq p q := by
  have key : ∀ j : Fin m, p j ^ 2 / q j = (p j - q j) ^ 2 / q j + (2 * p j - q j) := by
    intro j
    have h := (hq j).ne'
    field_simp
    ring
  rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => key j, Finset.sum_add_distrib,
    ← chiSq, Finset.sum_sub_distrib, ← Finset.mul_sum, hps, hqs]
  ring

/-- The second moment of the importance weight is `1 + χ²`. -/
lemma expect_iw_sq {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1)
    (hqs : ∑ j, q j = 1) :
    ∑ j, q j * (iw p q j) ^ 2 = 1 + chiSq p q := by
  rw [← sum_sq_div hq hps hqs]
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [iw_apply, div_pow]
  field_simp

/-- Kish's effective sample size fraction of an importance-reweighted ensemble: the ratio
of the squared mean weight to the mean squared weight. -/
noncomputable def essFrac (p q : Fin m → ℝ) : ℝ :=
  (∑ j, q j * iw p q j) ^ 2 / (∑ j, q j * (iw p q j) ^ 2)

/-- **The effective sample size is exactly `N / (1 + χ²)`.**  Reweighting `N` frames of the
reference ensemble `q` onto the target `p` leaves `N · essFrac` independent samples. -/
theorem essFrac_eq {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hps : ∑ j, p j = 1)
    (hqs : ∑ j, q j = 1) :
    essFrac p q = 1 / (1 + chiSq p q) := by
  rw [essFrac, expect_iw hq hps, expect_iw_sq hq hps hqs]
  norm_num

lemma one_add_chiSq_pos {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) : 0 < 1 + chiSq p q := by
  have := chiSq_nonneg (p := p) hq
  linarith

/-- **The relative entropy is at most `log (1 + χ²)`.**  (The Rényi ordering
`KL ≤ log(1+χ²)`, proved from the elementary bound `log x ≤ x - 1`.) -/
theorem kl_le_log_one_add_chiSq {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    klDiv p q ≤ Real.log (1 + chiSq p q) := by
  set C : ℝ := 1 + chiSq p q with hC
  have hCpos : 0 < C := one_add_chiSq_pos (p := p) hq
  -- termwise: `p log (p / (q C)) ≤ p * (p / (q C) - 1)`
  have hterm : ∀ j : Fin m,
      p j * Real.log (p j / q j) - p j * Real.log C ≤ p j ^ 2 / (q j * C) - p j := by
    intro j
    rcases eq_or_lt_of_le (hp j) with h0 | h0
    · simp [← h0]
    · have hqj := hq j
      have hx : 0 < p j / (q j * C) := div_pos h0 (mul_pos hqj hCpos)
      have hlog := Real.log_le_sub_one_of_pos hx
      have hsplit : Real.log (p j / (q j * C))
          = Real.log (p j / q j) - Real.log C := by
        rw [Real.log_div h0.ne' (mul_pos hqj hCpos).ne',
          Real.log_div h0.ne' hqj.ne', Real.log_mul hqj.ne' hCpos.ne']
        ring
      rw [hsplit] at hlog
      have hmul := mul_le_mul_of_nonneg_left hlog h0.le
      have hrw : p j * (p j / (q j * C) - 1) = p j ^ 2 / (q j * C) - p j := by
        field_simp
      calc p j * Real.log (p j / q j) - p j * Real.log C
          = p j * (Real.log (p j / q j) - Real.log C) := by ring
        _ ≤ p j * (p j / (q j * C) - 1) := hmul
        _ = p j ^ 2 / (q j * C) - p j := hrw
  have hsum := Finset.sum_le_sum fun j (_ : j ∈ Finset.univ) => hterm j
  have hA : ∑ j, p j ^ 2 / (q j * C) = 1 := by
    have : ∑ j, p j ^ 2 / (q j * C) = (∑ j, p j ^ 2 / q j) / C := by
      rw [Finset.sum_div]
      exact Finset.sum_congr rfl fun j _ => by
        rw [div_div]
    rw [this, sum_sq_div hq hps hqs, ← hC, div_self hCpos.ne']
  have hB : ∑ j, p j * Real.log C = Real.log C := by
    rw [← Finset.sum_mul, hps, one_mul]
  rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, hA, hB, hps] at hsum
  simpa [klDiv] using by linarith [hsum]

/-- **The reweighting cost is exponential in the relative entropy.**  The effective sample
size fraction obeys `essFrac ≤ exp (-KL(p‖q))`. -/
theorem essFrac_le_exp_neg_kl {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    essFrac p q ≤ Real.exp (-(klDiv p q)) := by
  have hCpos : (0:ℝ) < 1 + chiSq p q := one_add_chiSq_pos (p := p) hq
  have hkl := kl_le_log_one_add_chiSq hp hq hps hqs
  have hexp : Real.exp (klDiv p q) ≤ 1 + chiSq p q := by
    calc Real.exp (klDiv p q) ≤ Real.exp (Real.log (1 + chiSq p q)) := Real.exp_le_exp.2 hkl
      _ = 1 + chiSq p q := Real.exp_log hCpos
  rw [essFrac_eq hq hps hqs, Real.exp_neg, inv_eq_one_div]
  gcongr

/-- **How long a simulation must be.**  To retain `neff` effective frames after reweighting
a reference ensemble `q` onto a target `p`, the simulation must contain at least
`neff · exp (KL(p‖q))` frames. -/
theorem frames_needed {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) {N neff : ℝ} (hN : 0 ≤ N)
    (h : neff ≤ N * essFrac p q) :
    neff * Real.exp (klDiv p q) ≤ N := by
  have h1 : N * essFrac p q ≤ N * Real.exp (-(klDiv p q)) :=
    mul_le_mul_of_nonneg_left (essFrac_le_exp_neg_kl hp hq hps hqs) hN
  have h2 : neff ≤ N * Real.exp (-(klDiv p q)) := le_trans h h1
  have hpos : (0:ℝ) < Real.exp (klDiv p q) := Real.exp_pos _
  have h3 := mul_le_mul_of_nonneg_right h2 hpos.le
  rw [mul_assoc, Real.exp_neg, inv_mul_cancel₀ hpos.ne'] at h3
  simpa using h3

/-- The chi-squared divergence dominates the square of the operational (ℓ¹) distance. -/
theorem ell1_sq_le_chiSq {p q : Fin m → ℝ} (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1) :
    (∑ j, |p j - q j|) ^ 2 ≤ chiSq p q := by
  have h1 : ∑ j, |p j - q j| / Real.sqrt (q j) * Real.sqrt (q j) = ∑ j, |p j - q j| := by
    refine Finset.sum_congr rfl fun j _ => ?_
    field_simp [Real.sqrt_ne_zero'.2 (hq j)]
  have h2 : ∑ j, (|p j - q j| / Real.sqrt (q j)) ^ 2 = chiSq p q := by
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [div_pow, sq_abs, Real.sq_sqrt (hq j).le]
  have h3 : ∑ j, (Real.sqrt (q j)) ^ 2 = 1 := by
    rw [← hqs]
    exact Finset.sum_congr rfl fun j _ => Real.sq_sqrt (hq j).le
  have key : (∑ j, |p j - q j| / Real.sqrt (q j) * Real.sqrt (q j)) ^ 2
      ≤ (∑ j, (|p j - q j| / Real.sqrt (q j)) ^ 2) * (∑ j, (Real.sqrt (q j)) ^ 2) :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  rw [h1, h2, h3, mul_one] at key
  exact key

/-- On a uniform reference ensemble over a library of `m` conformations, concentrating the
population on a single conformation costs `χ² = m - 1`: the effective sample size fraction
is `1/m`.  Reweighting cannot manufacture a structure the prior did not sample. -/
theorem chiSq_unif_dirac {m : ℕ} (hm : 0 < m) (j₀ : Fin m) :
    chiSq (fun j => if j = j₀ then (1:ℝ) else 0) (fun _ => (1:ℝ)/m) = (m : ℝ) - 1 := by
  have hmpos : (0:ℝ) < m := by exact_mod_cast hm
  simp only [chiSq]
  rw [Finset.sum_eq_add_sum_diff_singleton (Finset.mem_univ j₀)]
  have hother : ∀ j ∈ Finset.univ \ ({j₀} : Finset (Fin m)),
      ((if j = j₀ then (1:ℝ) else 0) - 1/(m:ℝ)) ^ 2 / (1/(m:ℝ)) = 1/(m:ℝ) := by
    intro j hj
    have hne : j ≠ j₀ := by simpa using (Finset.mem_sdiff.1 hj).2
    rw [if_neg hne]
    field_simp
    norm_num
  rw [Finset.sum_congr rfl hother, if_pos rfl, Finset.sum_const, nsmul_eq_mul]
  have hcard : ((Finset.univ \ ({j₀} : Finset (Fin m))).card : ℝ) = (m : ℝ) - 1 := by
    have : (Finset.univ \ ({j₀} : Finset (Fin m))).card = m - 1 := by
      rw [Finset.card_sdiff]; simp
    rw [this, Nat.cast_sub hm]
    simp
  rw [hcard]
  field_simp
  ring

end Reweight

end IDR
