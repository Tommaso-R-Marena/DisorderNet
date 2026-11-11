/-
# Part V.6  The sharp data requirement: `n ≳ 1/eps²`

`RequestProject.SampleComplexity` gets `n ≥ 1/(4·eps)` out of Le Cam's lemma with the crude
tensorisation bound `TV(p^{⊗n}, q^{⊗n}) ≤ n·TV(p,q)`.  The true rate is `1/eps²`, and this
file proves it, by replacing the crude bound with the information-theoretic one: relative
entropy is **additive** over independent draws (`klG_prodP`), and Pinsker's inequality
(Part V.5) converts it back into total variation, costing only a square root.

* `klG_prodP` -- `KL(p^{⊗n} ‖ q^{⊗n}) = n · KL(p‖q)`.
* `tv_prodP_le` -- hence `‖p^{⊗n} - q^{⊗n}‖₁ ≤ sqrt (2 n KL(p‖q))`.
* `bernoulli_kl_le` -- the relative entropy of the hard pair `(1/2 ± tau)` is at most
  `24·tau²`, quadratic in the separation: nearby ensembles are quadratically hard to tell
  apart, which is exactly why the data requirement is quadratic in the tolerance.
* `sharp_sample_complexity` -- **for every estimator whatsoever**, `eps`-accuracy in
  expected `ℓ¹` error on both members of the hard pair forces `n ≥ 1/(48·eps²)` samples.
  With `RequestProject.Estimation`'s `empirical_risk_le` (`risk ≤ sqrt (m/n)`, i.e.
  `n = m/eps²` suffices) the dependence on the tolerance is now matched: the data cost of an
  ensemble is `Θ(1/eps²)`, with a library-size factor between the two sides.
-/
import Mathlib
import RequestProject.SampleComplexity
import RequestProject.Estimation
import RequestProject.Pinsker

namespace IDR

open Finset
open scoped Classical

namespace Learn

variable {m n : ℕ}

/-! ## Relative entropy is additive over independent draws -/

/-- **Additivity of relative entropy.**  `n` independent draws multiply the relative entropy
by `n`. -/
theorem klG_prodP {p q : Fin m → ℝ} (hp : ∀ j, 0 < p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) :
    Pinsker.klG (prodP (n := n) p) (prodP (n := n) q) = n * Pinsker.klG p q := by
  have hlog : ∀ s : Fin n → Fin m,
      Real.log (prodP p s / prodP q s) = ∑ i, Real.log (p (s i) / q (s i)) := by
    intro s
    have hP : 0 < prodP p s := Finset.prod_pos fun i _ => hp _
    have hQ : 0 < prodP q s := Finset.prod_pos fun i _ => hq _
    rw [Real.log_div hP.ne' hQ.ne']
    simp only [prodP]
    rw [Real.log_prod (fun i _ => (hp (s i)).ne'),
      Real.log_prod (fun i _ => (hq (s i)).ne'), ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun i _ => (Real.log_div (hp (s i)).ne' (hq (s i)).ne').symm
  have hstep : ∀ s : Fin n → Fin m,
      prodP p s * Real.log (prodP p s / prodP q s)
        = ∑ i, prodP p s * Real.log (p (s i) / q (s i)) := by
    intro s
    rw [hlog s, Finset.mul_sum]
  have hswap : ∑ s : Fin n → Fin m, ∑ i : Fin n, prodP p s * Real.log (p (s i) / q (s i))
      = ∑ i : Fin n, ∑ s : Fin n → Fin m, prodP p s * Real.log (p (s i) / q (s i)) :=
    Finset.sum_comm
  have hcoord : ∀ i : Fin n,
      ∑ s : Fin n → Fin m, prodP p s * Real.log (p (s i) / q (s i)) = Pinsker.klG p q := by
    intro i
    rw [expect_coord hps i (fun a => Real.log (p a / q a))]
    rfl
  calc Pinsker.klG (prodP (n := n) p) (prodP (n := n) q)
      = ∑ s : Fin n → Fin m, ∑ i : Fin n, prodP p s * Real.log (p (s i) / q (s i)) := by
        unfold Pinsker.klG
        exact Finset.sum_congr rfl fun s _ => hstep s
    _ = ∑ i : Fin n, ∑ s : Fin n → Fin m, prodP p s * Real.log (p (s i) / q (s i)) := hswap
    _ = ∑ _i : Fin n, Pinsker.klG p q :=
        Finset.sum_congr rfl fun i _ => hcoord i
    _ = n * Pinsker.klG p q := by
        rw [Finset.sum_const, nsmul_eq_mul, Finset.card_univ, Fintype.card_fin]

/-- Pinsker plus additivity: `n` draws separate two ensembles by at most
`sqrt (2 n KL(p‖q))` in `ℓ¹`. -/
theorem tv_prodP_le {p q : Fin m → ℝ} (hp : ∀ j, 0 < p j) (hq : ∀ j, 0 < q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    l1 (prodP (n := n) p) (prodP (n := n) q)
      ≤ Real.sqrt (2 * (n * Pinsker.klG p q)) := by
  have hPnn : ∀ s : Fin n → Fin m, 0 ≤ prodP p s := fun s => prodP_nonneg (fun j => (hp j).le) s
  have hQpos : ∀ s : Fin n → Fin m, 0 < prodP q s := fun s => Finset.prod_pos fun i _ => hq _
  have hPs : ∑ s : Fin n → Fin m, prodP p s = 1 := prodP_sum_one hps
  have hQs : ∑ s : Fin n → Fin m, prodP q s = 1 := prodP_sum_one hqs
  have h := Pinsker.ell1_le_sqrt_two_klG hPnn hQpos hPs hQs
  rwa [klG_prodP hp hq hps] at h

/-! ## The hard pair, and its relative entropy -/

/-- The hard pair on a library of two conformations. -/
noncomputable def bern (tau : ℝ) : Fin 2 → ℝ := ![1/2 + tau, 1/2 - tau]

lemma bern_pos {tau : ℝ} (h0 : 0 < tau) (h1 : tau < 1/2) (j : Fin 2) : 0 < bern tau j := by
  fin_cases j <;> simp [bern] <;> linarith

lemma bern_sum (tau : ℝ) : ∑ j, bern tau j = 1 := by
  simp [bern, Fin.sum_univ_two]
  ring

lemma bern_l1 {tau : ℝ} (h0 : 0 ≤ tau) : l1 (bern tau) (bern (-tau)) = 4 * tau := by
  simp only [l1, Fin.sum_univ_two, bern]
  norm_num
  rw [abs_of_nonneg (by linarith), abs_of_nonpos (by linarith)]
  ring

/-- **The relative entropy of the hard pair is quadratic in the separation.**  This is why
the data requirement is quadratic in the tolerance. -/
theorem bernoulli_kl_le {tau : ℝ} (h0 : 0 < tau) (h1 : tau ≤ 1/4) :
    Pinsker.klG (bern tau) (bern (-tau)) ≤ 24 * tau ^ 2 := by
  have ha : (0:ℝ) < 1/2 + tau := by linarith
  have hb : (0:ℝ) < 1/2 - tau := by linarith
  -- `log x ≤ x - 1` on both terms
  have hlog1 : Real.log ((1/2 + tau)/(1/2 - tau)) ≤ (1/2 + tau)/(1/2 - tau) - 1 :=
    Real.log_le_sub_one_of_pos (div_pos ha hb)
  have hlog2 : Real.log ((1/2 - tau)/(1/2 + tau)) ≤ (1/2 - tau)/(1/2 + tau) - 1 :=
    Real.log_le_sub_one_of_pos (div_pos hb ha)
  have hkl : Pinsker.klG (bern tau) (bern (-tau))
      = (1/2 + tau) * Real.log ((1/2 + tau)/(1/2 - tau))
        + (1/2 - tau) * Real.log ((1/2 - tau)/(1/2 + tau)) := by
    have e1 : (1:ℝ)/2 + -tau = 1/2 - tau := by ring
    have e2 : (1:ℝ)/2 - -tau = 1/2 + tau := by ring
    simp only [Pinsker.klG, Fin.sum_univ_two, bern, Matrix.cons_val_zero, Matrix.cons_val_one, e1, e2]
  have hb1 : (1/2 + tau) * Real.log ((1/2 + tau)/(1/2 - tau))
      ≤ (1/2 + tau) * ((1/2 + tau)/(1/2 - tau) - 1) :=
    mul_le_mul_of_nonneg_left hlog1 ha.le
  have hb2 : (1/2 - tau) * Real.log ((1/2 - tau)/(1/2 + tau))
      ≤ (1/2 - tau) * ((1/2 - tau)/(1/2 + tau) - 1) :=
    mul_le_mul_of_nonneg_left hlog2 hb.le
  have hval : (1/2 + tau) * ((1/2 + tau)/(1/2 - tau) - 1)
      + (1/2 - tau) * ((1/2 - tau)/(1/2 + tau) - 1)
      = 4 * tau^2 / (1/4 - tau^2) := by
    have key : ∀ a b : ℝ, a ≠ 0 → b ≠ 0 →
        a * (a/b - 1) + b * (b/a - 1) = (a - b)^2 * (a + b) / (a * b) := by
      intro a b ha hb; field_simp; ring
    rw [key _ _ (ne_of_gt ha) (ne_of_gt hb)]
    congr 1 <;> ring
  have hden : (0:ℝ) < 1/4 - tau^2 := by nlinarith
  have htau2 : tau^2 ≤ 1/16 := by nlinarith
  have hfrac : 4 * tau^2 / (1/4 - tau^2) ≤ 24 * tau^2 := by
    rw [div_le_iff₀ hden]
    nlinarith [sq_nonneg tau, htau2]
  rw [hkl]
  linarith [hb1, hb2, hval, hfrac]

/-! ## The sharp lower bound -/

/-- **The sharp data requirement.**  Any estimator -- of any kind, with any prior knowledge
of the model class -- whose expected `ℓ¹` error is at most `eps` at both members of the hard
pair `(1/2 ± eps)` must have seen at least `1/(48·eps²)` independent conformations. -/
theorem sharp_sample_complexity {eps : ℝ} (heps : 0 < eps) (heps' : eps ≤ 1/4)
    (T : (Fin n → Fin 2) → (Fin 2 → ℝ))
    (hTp : risk (bern eps) T ≤ eps) (hTq : risk (bern (-eps)) T ≤ eps) :
    1 / (48 * eps^2) ≤ (n : ℝ) := by
  have hppos : ∀ j, 0 < bern eps j := bern_pos heps (by linarith)
  have hqpos : ∀ j, 0 < bern (-eps) j := by
    intro j
    have : (0:ℝ) < -(-eps) := by linarith
    fin_cases j <;> simp [bern] <;> linarith
  have hps : ∑ j, bern eps j = 1 := bern_sum eps
  have hqs : ∑ j, bern (-eps) j = 1 := bern_sum (-eps)
  have hlc := le_cam (fun j => (hppos j).le) (fun j => (hqpos j).le) hps hqs T
  have hD : l1 (bern eps) (bern (-eps)) = 4 * eps := bern_l1 heps.le
  -- the sample laws are close, by additivity plus Pinsker
  have hkl : Pinsker.klG (bern eps) (bern (-eps)) ≤ 24 * eps^2 := bernoulli_kl_le heps heps'
  have hklnn : 0 ≤ Pinsker.klG (bern eps) (bern (-eps)) :=
    Pinsker.klG_nonneg (fun j => (hppos j).le) hqpos hps hqs
  have htv := tv_prodP_le (n := n) hppos hqpos hps hqs
  have hnnn : (0:ℝ) ≤ n := Nat.cast_nonneg n
  have hmono : Real.sqrt (2 * ((n:ℝ) * Pinsker.klG (bern eps) (bern (-eps))))
      ≤ Real.sqrt (48 * n * eps^2) := by
    apply Real.sqrt_le_sqrt
    nlinarith [hkl, hnnn]
  have hsqrt : Real.sqrt (48 * n * eps^2) = eps * Real.sqrt (48 * n) := by
    rw [show (48:ℝ) * n * eps^2 = (48 * n) * eps^2 by ring,
      Real.sqrt_mul (by positivity), Real.sqrt_sq heps.le]
    ring
  have hclose : l1 (prodP (n := n) (bern eps)) (prodP (n := n) (bern (-eps)))
      ≤ eps * Real.sqrt (48 * n) := by
    rw [← hsqrt]
    exact le_trans htv hmono
  -- Le Cam
  have hrisk : risk (bern eps) T + risk (bern (-eps)) T ≤ 2 * eps := by linarith
  have hstep : 4 * eps * (1 - eps * Real.sqrt (48 * (n:ℝ)) / 2) ≤ 2 * eps := by
    have hmono2 : 4 * eps * (1 - eps * Real.sqrt (48 * (n:ℝ)) / 2)
        ≤ l1 (bern eps) (bern (-eps))
          * (1 - l1 (prodP (n := n) (bern eps)) (prodP (n := n) (bern (-eps))) / 2) := by
      rw [hD]
      have : 1 - l1 (prodP (n := n) (bern eps)) (prodP (n := n) (bern (-eps))) / 2
          ≥ 1 - eps * Real.sqrt (48 * (n:ℝ)) / 2 := by linarith
      nlinarith [heps]
    linarith [hlc]
  -- conclude
  have hsq : 1 ≤ eps * Real.sqrt (48 * (n:ℝ)) := by nlinarith [hstep, heps]
  have hsqnn : 0 ≤ Real.sqrt (48 * (n:ℝ)) := Real.sqrt_nonneg _
  have hsq2 : 1 ≤ eps^2 * (48 * (n:ℝ)) := by
    have h1 : (1:ℝ) ≤ (eps * Real.sqrt (48 * (n:ℝ)))^2 := by nlinarith [hsq]
    have h2 : (eps * Real.sqrt (48 * (n:ℝ)))^2 = eps^2 * (48 * (n:ℝ)) := by
      rw [mul_pow, Real.sq_sqrt (by positivity)]
    linarith [h1, h2 ▸ h1]
  rw [div_le_iff₀ (by positivity)]
  nlinarith [hsq2]

end Learn

end IDR
