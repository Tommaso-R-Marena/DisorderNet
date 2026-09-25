/-
# Part V.4  Achievability: the empirical ensemble matches the lower bound

`RequestProject.SampleComplexity` proves that *no* estimator can learn a conformational
ensemble from few samples.  A lower bound alone would be a half-theory: it leaves open
whether the obstruction is information or ingenuity.  This file closes the loop by
analysing the simplest possible estimator -- count the sampled conformations -- and showing
that it already attains the same order.

* `sum_prod_coord`, `expect_coord`, `expect_coord_pair` are the independence calculus of the
  `n`-fold product law: expectations of one- and two-coordinate functions factorise.
* `expect_emp`, `variance_emp` -- the empirical populations are unbiased and have variance
  exactly `p j (1 - p j) / n`, the classical multinomial variance, proved here from scratch
  for the product law on `Fin n → Fin m`.
* `empirical_risk_le` -- **achievability**: the empirical ensemble has expected `ℓ¹` risk at
  most `sqrt (m / n)`.  Together with the Le Cam bound this brackets the sample cost of an
  ensemble: `n ≳ 1/eps` is necessary and `n ≤ m/eps²` suffices.
* `ensemble_sample_complexity` states the two together.

Read physically: the data cost of an ensemble model grows with the *number of populated
conformations*, i.e. with `exp` of the conformational entropy of the disordered region --
the same quantity that controls model capacity in `RequestProject.Metric`.
-/
import Mathlib
import RequestProject.SampleComplexity

namespace IDR

open Finset
open scoped Classical

namespace Learn

variable {m n : ℕ}

/-! ## The independence calculus of the product law -/

/-- The sum over samples of a product of coordinate functions factorises. -/
lemma sum_prod_coord (F : Fin n → Fin m → ℝ) :
    ∑ s : Fin n → Fin m, ∏ i, F i (s i) = ∏ i, ∑ a, F i a := by
  have h := Finset.prod_univ_sum (ι := Fin n) (κ := fun _ => Fin m)
    (fun _ => (Finset.univ : Finset (Fin m))) (fun i a => F i a)
  rw [Fintype.piFinset_univ] at h
  exact h.symm

/-- One-coordinate marginal: the `i₀`-th draw has law `p`. -/
lemma expect_coord {p : Fin m → ℝ} (hps : ∑ j, p j = 1) (i0 : Fin n) (g : Fin m → ℝ) :
    ∑ s : Fin n → Fin m, prodP p s * g (s i0) = ∑ a, p a * g a := by
  classical
  set F : Fin n → Fin m → ℝ := fun i => if i = i0 then (fun a => p a * g a) else p with hFdef
  have hF : ∀ s : Fin n → Fin m, prodP p s * g (s i0) = ∏ i, F i (s i) := by
    intro s
    have h1 : ∏ i, F i (s i) = F i0 (s i0) * ∏ i ∈ Finset.univ.erase i0, F i (s i) :=
      (Finset.mul_prod_erase _ _ (Finset.mem_univ i0)).symm
    have h2 : ∏ i, p (s i) = p (s i0) * ∏ i ∈ Finset.univ.erase i0, p (s i) :=
      (Finset.mul_prod_erase _ _ (Finset.mem_univ i0)).symm
    have h3 : ∏ i ∈ Finset.univ.erase i0, F i (s i) = ∏ i ∈ Finset.univ.erase i0, p (s i) :=
      Finset.prod_congr rfl fun i hi => by
        have hne : i ≠ i0 := Finset.ne_of_mem_erase hi
        simp [hFdef, hne]
    rw [h1, h3, prodP, h2]
    simp [hFdef]
    ring
  rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => hF s, sum_prod_coord]
  rw [Finset.prod_eq_single i0]
  · simp [hFdef]
  · intro i _ hi
    simp [hFdef, hi, hps]
  · intro h; exact absurd (Finset.mem_univ i0) h

/-- Two-coordinate marginal: distinct draws are independent. -/
lemma expect_coord_pair {p : Fin m → ℝ} (hps : ∑ j, p j = 1) {i0 i1 : Fin n}
    (hne : i0 ≠ i1) (g h : Fin m → ℝ) :
    ∑ s : Fin n → Fin m, prodP p s * (g (s i0) * h (s i1))
      = (∑ a, p a * g a) * (∑ a, p a * h a) := by
  classical
  set F : Fin n → Fin m → ℝ :=
    fun i => if i = i0 then (fun a => p a * g a)
      else if i = i1 then (fun a => p a * h a) else p with hFdef
  have hF : ∀ s : Fin n → Fin m, prodP p s * (g (s i0) * h (s i1)) = ∏ i, F i (s i) := by
    intro s
    have h1 : ∏ i, F i (s i) = F i0 (s i0) * ∏ i ∈ Finset.univ.erase i0, F i (s i) :=
      (Finset.mul_prod_erase _ _ (Finset.mem_univ i0)).symm
    have hmem : i1 ∈ Finset.univ.erase i0 := Finset.mem_erase.2 ⟨Ne.symm hne, Finset.mem_univ _⟩
    have h2 : ∏ i ∈ Finset.univ.erase i0, F i (s i)
        = F i1 (s i1) * ∏ i ∈ (Finset.univ.erase i0).erase i1, F i (s i) :=
      (Finset.mul_prod_erase _ _ hmem).symm
    have h3 : ∏ i ∈ (Finset.univ.erase i0).erase i1, F i (s i)
        = ∏ i ∈ (Finset.univ.erase i0).erase i1, p (s i) :=
      Finset.prod_congr rfl fun i hi => by
        have hne1 : i ≠ i1 := Finset.ne_of_mem_erase hi
        have hne0 : i ≠ i0 := Finset.ne_of_mem_erase (Finset.mem_of_mem_erase hi)
        simp [hFdef, hne0, hne1]
    have h4 : ∏ i, p (s i) = p (s i0) * ∏ i ∈ Finset.univ.erase i0, p (s i) :=
      (Finset.mul_prod_erase _ _ (Finset.mem_univ i0)).symm
    have h5 : ∏ i ∈ Finset.univ.erase i0, p (s i)
        = p (s i1) * ∏ i ∈ (Finset.univ.erase i0).erase i1, p (s i) :=
      (Finset.mul_prod_erase _ _ hmem).symm
    rw [h1, h2, h3, prodP, h4, h5]
    have hF0 : F i0 (s i0) = p (s i0) * g (s i0) := by simp [hFdef]
    have hF1 : F i1 (s i1) = p (s i1) * h (s i1) := by simp [hFdef, Ne.symm hne]
    rw [hF0, hF1]
    ring
  rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => hF s, sum_prod_coord]
  have hc0 : ∑ a, F i0 a = ∑ a, p a * g a := by simp [hFdef]
  have hc1 : ∑ a, F i1 a = ∑ a, p a * h a := by simp [hFdef, Ne.symm hne]
  have hrest : ∀ i ∈ (Finset.univ.erase i0).erase i1, ∑ a, F i a = 1 := by
    intro i hi
    have hne1 : i ≠ i1 := Finset.ne_of_mem_erase hi
    have hne0 : i ≠ i0 := Finset.ne_of_mem_erase (Finset.mem_of_mem_erase hi)
    simp [hFdef, hne0, hne1, hps]
  have hmem : i1 ∈ Finset.univ.erase i0 := Finset.mem_erase.2 ⟨Ne.symm hne, Finset.mem_univ _⟩
  rw [← Finset.mul_prod_erase _ _ (Finset.mem_univ i0),
    ← Finset.mul_prod_erase _ _ hmem, Finset.prod_congr rfl hrest]
  rw [hc0, hc1]
  simp

/-! ## The empirical ensemble -/

/-- The indicator of a library conformation. -/
noncomputable def ind (j : Fin m) : Fin m → ℝ := fun a => if a = j then 1 else 0

/-- The empirical populations of a sample: the estimator that just counts. -/
noncomputable def emp (s : Fin n → Fin m) : Fin m → ℝ :=
  fun j => (∑ i, ind j (s i)) / n

lemma sum_ind {p : Fin m → ℝ} (j : Fin m) : ∑ a, p a * ind j a = p j := by
  simp [ind, Finset.sum_ite_eq']

lemma sum_ind_sq {p : Fin m → ℝ} (j : Fin m) : ∑ a, p a * (ind j a * ind j a) = p j := by
  have : ∀ a : Fin m, p a * (ind j a * ind j a) = p a * ind j a := by
    intro a; by_cases h : a = j <;> simp [ind, h]
  rw [Finset.sum_congr rfl fun a (_ : a ∈ Finset.univ) => this a, sum_ind]

/-- The empirical populations are unbiased. -/
lemma expect_emp {p : Fin m → ℝ} (hps : ∑ j, p j = 1) (hn : 0 < n) (j : Fin m) :
    ∑ s : Fin n → Fin m, prodP p s * emp s j = p j := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast hn
  have hstep : ∀ s : Fin n → Fin m, prodP p s * emp s j
      = (∑ i, prodP p s * ind j (s i)) / n := by
    intro s
    have he : emp s j = (∑ i, ind j (s i)) / n := rfl
    rw [he, ← mul_div_assoc, Finset.mul_sum]
  rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => hstep s, ← Finset.sum_div,
    Finset.sum_comm]
  have hinner : ∀ i : Fin n, ∑ s : Fin n → Fin m, prodP p s * ind j (s i) = p j := by
    intro i
    rw [expect_coord hps i (ind j), sum_ind]
  rw [Finset.sum_congr rfl fun i (_ : i ∈ Finset.univ) => hinner i, Finset.sum_const,
    nsmul_eq_mul, Finset.card_univ, Fintype.card_fin]
  field_simp

/-- **The multinomial variance.**  The empirical population of a conformation has variance
exactly `p j (1 - p j) / n`. -/
lemma variance_emp {p : Fin m → ℝ} (hps : ∑ j, p j = 1) (hn : 0 < n) (j : Fin m) :
    ∑ s : Fin n → Fin m, prodP p s * (emp s j - p j) ^ 2 = p j * (1 - p j) / n := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast hn
  -- expand the square
  have hexp : ∀ s : Fin n → Fin m, prodP p s * (emp s j - p j) ^ 2
      = prodP p s * (emp s j)^2 - 2 * p j * (prodP p s * emp s j) + p j ^ 2 * prodP p s := by
    intro s; ring
  have hsq : ∀ s : Fin n → Fin m,
      prodP p s * (emp s j)^2
        = (∑ i, ∑ k, prodP p s * (ind j (s i) * ind j (s k))) / (n^2 : ℝ) := by
    intro s
    have h1 : (emp s j)^2 = ((∑ i, ind j (s i)) * (∑ k, ind j (s k))) / (n^2 : ℝ) := by
      have he : emp s j = (∑ i, ind j (s i)) / n := rfl
      rw [he]
      field_simp
    have h2 : prodP p s * ((∑ i, ind j (s i)) * (∑ k, ind j (s k)))
        = ∑ i, ∑ k, prodP p s * (ind j (s i) * ind j (s k)) := by
      rw [Finset.sum_mul_sum, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by rw [Finset.mul_sum]
    rw [h1, ← mul_div_assoc, h2]
  -- expectation of the double sum
  have hpair : ∀ i k : Fin n,
      ∑ s : Fin n → Fin m, prodP p s * (ind j (s i) * ind j (s k))
        = if i = k then p j else p j ^ 2 := by
    intro i k
    by_cases h : i = k
    · subst h
      rw [expect_coord hps i (fun a => ind j a * ind j a), sum_ind_sq]
      simp
    · rw [expect_coord_pair hps h (ind j) (ind j), sum_ind, if_neg h, sq]
  have hdouble : ∑ s : Fin n → Fin m, prodP p s * (emp s j)^2
      = ((n : ℝ) * p j + ((n:ℝ)^2 - n) * p j ^ 2) / (n^2 : ℝ) := by
    rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => hsq s, ← Finset.sum_div]
    congr 1
    have hswap : ∑ s : Fin n → Fin m, ∑ i : Fin n, ∑ k : Fin n,
          prodP p s * (ind j (s i) * ind j (s k))
        = ∑ i : Fin n, ∑ k : Fin n, ∑ s : Fin n → Fin m,
          prodP p s * (ind j (s i) * ind j (s k)) := by
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_comm
    rw [hswap]
    have hval : ∑ i : Fin n, ∑ k : Fin n,
        (∑ s : Fin n → Fin m, prodP p s * (ind j (s i) * ind j (s k)))
        = ∑ i : Fin n, ∑ k : Fin n, (if i = k then p j else p j ^ 2) :=
      Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun k _ => hpair i k
    rw [hval]
    have hinner : ∀ i : Fin n, ∑ k : Fin n, (if i = k then p j else p j ^ 2)
        = p j + ((n:ℝ) - 1) * p j ^ 2 := by
      intro i
      have hrw : ∀ k : Fin n, (if i = k then p j else p j ^ 2)
          = p j ^ 2 + (if i = k then p j - p j ^ 2 else 0) := by
        intro k; by_cases h : i = k <;> simp [h]
      rw [Finset.sum_congr rfl fun k (_ : k ∈ Finset.univ) => hrw k, Finset.sum_add_distrib,
        Finset.sum_const, nsmul_eq_mul, Finset.sum_ite_eq]
      simp
      ring
    rw [Finset.sum_congr rfl fun i (_ : i ∈ Finset.univ) => hinner i, Finset.sum_const,
      nsmul_eq_mul, Finset.card_univ, Fintype.card_fin]
    ring
  have hmean : ∑ s : Fin n → Fin m, prodP p s * emp s j = p j := expect_emp hps hn j
  have hone : ∑ s : Fin n → Fin m, prodP p s = 1 := prodP_sum_one hps
  rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => hexp s]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
    hmean, hone, hdouble]
  field_simp
  ring

/-! ## The risk of the empirical ensemble -/

/-- Jensen / Cauchy--Schwarz: the mean absolute deviation is at most the standard
deviation. -/
lemma mean_abs_le_sqrt {p : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (X : (Fin n → Fin m) → ℝ) :
    ∑ s : Fin n → Fin m, prodP p s * |X s|
      ≤ Real.sqrt (∑ s : Fin n → Fin m, prodP p s * (X s)^2) := by
  have hPnn : ∀ s : Fin n → Fin m, 0 ≤ prodP p s := fun s => prodP_nonneg hp s
  have hCS : (∑ s : Fin n → Fin m, Real.sqrt (prodP p s) * (Real.sqrt (prodP p s) * |X s|))^2
      ≤ (∑ s : Fin n → Fin m, (Real.sqrt (prodP p s))^2)
        * (∑ s : Fin n → Fin m, (Real.sqrt (prodP p s) * |X s|)^2) :=
    Finset.sum_mul_sq_le_sq_mul_sq _ _ _
  have h1 : ∀ s : Fin n → Fin m,
      Real.sqrt (prodP p s) * (Real.sqrt (prodP p s) * |X s|) = prodP p s * |X s| := by
    intro s
    rw [← mul_assoc, Real.mul_self_sqrt (hPnn s)]
  have h2 : ∑ s : Fin n → Fin m, (Real.sqrt (prodP p s))^2 = 1 := by
    rw [← prodP_sum_one (p := p) (n := n) hps]
    exact Finset.sum_congr rfl fun s _ => Real.sq_sqrt (hPnn s)
  have h3 : ∀ s : Fin n → Fin m, (Real.sqrt (prodP p s) * |X s|)^2 = prodP p s * (X s)^2 := by
    intro s
    rw [mul_pow, Real.sq_sqrt (hPnn s), sq_abs]
  rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => h1 s, h2, one_mul,
    Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => h3 s] at hCS
  have hnn : 0 ≤ ∑ s : Fin n → Fin m, prodP p s * |X s| :=
    Finset.sum_nonneg fun s _ => mul_nonneg (hPnn s) (abs_nonneg _)
  have hV : 0 ≤ ∑ s : Fin n → Fin m, prodP p s * (X s)^2 :=
    Finset.sum_nonneg fun s _ => mul_nonneg (hPnn s) (sq_nonneg _)
  exact (Real.le_sqrt hnn hV).2 hCS

/-- **Achievability.**  The empirical ensemble -- simply counting the sampled conformations
-- has expected `ℓ¹` risk at most `sqrt (m/n)`.  Accuracy `eps` is therefore reached with
`n = m/eps²` samples, matching the `1/eps` lower bound up to the library size. -/
theorem empirical_risk_le {p : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hn : 0 < n) : risk p (emp (n := n)) ≤ Real.sqrt ((m : ℝ) / n) := by
  have hnpos : (0:ℝ) < n := by exact_mod_cast hn
  -- exchange the sum over conformations with the sum over samples
  have hrisk : risk p (emp (n := n))
      = ∑ j : Fin m, ∑ s : Fin n → Fin m, prodP p s * |emp s j - p j| := by
    rw [risk, Finset.sum_comm]
    exact Finset.sum_congr rfl fun s _ => by
      rw [l1, Finset.mul_sum]
  have hterm : ∀ j : Fin m,
      ∑ s : Fin n → Fin m, prodP p s * |emp s j - p j| ≤ Real.sqrt (p j / n) := by
    intro j
    have h1 := mean_abs_le_sqrt (n := n) hp hps (fun s => emp s j - p j)
    have h2 : ∑ s : Fin n → Fin m, prodP p s * (emp s j - p j)^2 = p j * (1 - p j) / n :=
      variance_emp hps hn j
    rw [h2] at h1
    refine le_trans h1 (Real.sqrt_le_sqrt ?_)
    have hpj := hp j
    have hle : p j * (1 - p j) ≤ p j := by nlinarith [hp j, sq_nonneg (p j)]
    gcongr
  have hsum : ∑ j : Fin m, ∑ s : Fin n → Fin m, prodP p s * |emp s j - p j|
      ≤ ∑ j : Fin m, Real.sqrt (p j / n) :=
    Finset.sum_le_sum fun j _ => hterm j
  -- Cauchy--Schwarz over the library
  have hCS : (∑ j : Fin m, Real.sqrt (p j))^2 ≤ (m : ℝ) := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin m))
      (fun _ => (1:ℝ)) (fun j => Real.sqrt (p j))
    simp only [one_mul, one_pow, Finset.sum_const, nsmul_eq_mul, mul_one, Finset.card_univ,
      Fintype.card_fin] at h
    have h2 : ∑ j : Fin m, (Real.sqrt (p j))^2 = 1 := by
      rw [← hps]
      exact Finset.sum_congr rfl fun j _ => Real.sq_sqrt (hp j)
    rw [h2, mul_one] at h
    exact h
  have hfin : ∑ j : Fin m, Real.sqrt (p j / n) ≤ Real.sqrt ((m : ℝ)/ n) := by
    have hsplit : ∀ j : Fin m, Real.sqrt (p j / n) = Real.sqrt (p j) / Real.sqrt n := by
      intro j; rw [Real.sqrt_div (hp j)]
    rw [Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => hsplit j, ← Finset.sum_div,
      Real.sqrt_div (by positivity)]
    have hsq : ∑ j : Fin m, Real.sqrt (p j) ≤ Real.sqrt (m : ℝ) := by
      have hnn : 0 ≤ ∑ j : Fin m, Real.sqrt (p j) :=
        Finset.sum_nonneg fun j _ => Real.sqrt_nonneg _
      exact (Real.le_sqrt hnn (by positivity)).2 hCS
    gcongr
  rw [hrisk]
  exact le_trans hsum hfin

/-- **The sample cost of an ensemble, from both sides.**  Any estimator that is
`eps`-accurate on a hard pair needs `n ≥ 1/(4·eps)` samples; and the empirical ensemble
is accurate to `sqrt (m/n)`, so `n = m/eps²` samples always suffice. -/
theorem ensemble_sample_complexity {m : ℕ} (hm : 2 ≤ m) {eps : ℝ} (heps : 0 < eps)
    (heps' : eps ≤ 1/4) :
    (∃ p q : Fin m → ℝ, (∀ j, 0 ≤ p j) ∧ (∀ j, 0 ≤ q j) ∧ (∑ j, p j = 1) ∧ (∑ j, q j = 1)
        ∧ l1 p q = 4 * eps
        ∧ ∀ (n : ℕ) (T : (Fin n → Fin m) → (Fin m → ℝ)),
            risk p T ≤ eps → risk q T ≤ eps → 1 / (4 * eps) ≤ (n : ℝ))
      ∧ (∀ (n : ℕ), 0 < n → ∀ p : Fin m → ℝ, (∀ j, 0 ≤ p j) → (∑ j, p j = 1) →
            risk p (emp (n := n)) ≤ Real.sqrt ((m : ℝ) / n)) := by
  obtain ⟨p, q, hp, hq, hps, hqs, hD⟩ := exists_hard_pair hm heps heps'
  refine ⟨⟨p, q, hp, hq, hps, hqs, hD, ?_⟩, ?_⟩
  · intro n T hTp hTq
    exact sample_complexity_two_point hp hq hps hqs T heps hD hTp hTq
  · intro n hn p hp hps
    exact empirical_risk_le hp hps hn

end Learn

end IDR
