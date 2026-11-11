/-
# Modelling intrinsically disordered regions of proteins

A *conformation* of a protein with `n` geometric degrees of freedom (torsion angles,
or the coordinates of the backbone atoms) is a point of `Conf n = Fin n → ℝ`.
A protein in solution is not one conformation but a thermodynamic *ensemble*: a
probability distribution over conformations.  A degree of freedom `i` belongs to an
*intrinsically disordered region* when its marginal in that ensemble is non-degenerate,
i.e. it has strictly positive variance.

This file formalises three things.

* **Expressivity (negative side).**  A single-structure predictor -- a model that outputs
  one conformation `x` -- induces the point mass at `x`, whose every coordinate variance is
  zero.  Consequently no single-structure model can reproduce an ensemble with a disordered
  degree of freedom, and its squared error is bounded below by the total variance
  (bias-variance decomposition), which is strictly positive on a disordered ensemble.

* **Expressivity (positive side).**  A latent-variable ensemble model
  (a latent distribution together with a decoder into conformation space) represents
  *every* ensemble exactly, in the strong sense that all observables agree.  Moreover a
  model capturing a disordered degree of freedom must genuinely use at least two latent
  states with different decoded geometry -- disorder forces the ensemble picture.

* **Estimation.**  Fitting the ensemble weights by maximum likelihood (equivalently, by
  minimising cross-entropy) is consistent: the cross-entropy is minimised exactly at the
  true weights, since the Kullback-Leibler divergence is nonnegative and vanishes only at
  equality.  By contrast the best single-structure fit is the mean conformation and it
  retains an irreducible error equal to the total variance.
-/
import Mathlib

namespace IDR

open Finset

/-- A conformation: the values of the `n` geometric degrees of freedom. -/
abbrev Conf (n : ℕ) := Fin n → ℝ

/-- A conformational ensemble: finitely many conformations carried with probability
weights. -/
structure Ensemble (n : ℕ) where
  /-- number of conformations in the ensemble -/
  card : ℕ
  /-- the conformations -/
  conf : Fin card → Conf n
  /-- their statistical weights -/
  w : Fin card → ℝ
  w_nonneg : ∀ j, 0 ≤ w j
  w_sum : ∑ j, w j = 1

namespace Ensemble

variable {n : ℕ} (E : Ensemble n)

/-- The ensemble average of an observable. -/
def expect (f : Conf n → ℝ) : ℝ := ∑ j, E.w j * f (E.conf j)

@[simp] lemma expect_const (a : ℝ) : E.expect (fun _ => a) = a := by
  simp [expect, ← Finset.sum_mul, E.w_sum]

lemma expect_add (f g : Conf n → ℝ) :
    E.expect (fun c => f c + g c) = E.expect f + E.expect g := by
  simp [expect, mul_add, Finset.sum_add_distrib]

lemma expect_smul (a : ℝ) (f : Conf n → ℝ) :
    E.expect (fun c => a * f c) = a * E.expect f := by
  simp only [expect, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

lemma expect_nonneg {f : Conf n → ℝ} (hf : ∀ c, 0 ≤ f c) : 0 ≤ E.expect f :=
  Finset.sum_nonneg fun j _ => mul_nonneg (E.w_nonneg j) (hf _)

/-- The mean value of degree of freedom `i`. -/
def mean (i : Fin n) : ℝ := E.expect (fun c => c i)

/-- The variance of degree of freedom `i`. -/
def variance (i : Fin n) : ℝ := E.expect (fun c => (c i - E.mean i) ^ 2)

/-- Total (trace) variance of the ensemble. -/
def totalVariance : ℝ := ∑ i, E.variance i

/-- Degree of freedom `i` is *disordered* in `E` when its marginal is non-degenerate. -/
def Disordered (i : Fin n) : Prop := 0 < E.variance i

/-- The mean squared deviation of the ensemble from a single predicted structure `x`. -/
def pointLoss (x : Conf n) : ℝ := E.expect (fun c => ∑ i, (c i - x i) ^ 2)

end Ensemble

/-- A latent-variable ensemble model: a distribution over latent states together with a
decoder producing a conformation for each state. -/
structure LatentModel (n : ℕ) where
  /-- number of latent states -/
  k : ℕ
  /-- latent distribution -/
  p : Fin k → ℝ
  p_nonneg : ∀ z, 0 ≤ p z
  p_sum : ∑ z, p z = 1
  /-- decoder -/
  decode : Fin k → Conf n

namespace LatentModel

variable {n : ℕ} (M : LatentModel n)

/-- Model average of an observable. -/
def expect (f : Conf n → ℝ) : ℝ := ∑ z, M.p z * f (M.decode z)

/-- The model reproduces the ensemble: every observable has the same average. -/
def Captures (E : Ensemble n) : Prop := ∀ f : Conf n → ℝ, M.expect f = E.expect f

/-- A *single-structure* (point) predictor: the decoder is constant, so the model always
returns the same conformation. -/
def IsPoint : Prop := ∃ x : Conf n, ∀ z, M.decode z = x

end LatentModel

/-! ## Basic facts about ensemble averages -/

variable {n : ℕ}

lemma Ensemble.expect_centered (E : Ensemble n) (i : Fin n) :
    E.expect (fun c => c i - E.mean i) = 0 := by
  have h : E.expect (fun c => c i - E.mean i)
      = E.expect (fun c => c i) + E.expect (fun _ => -E.mean i) := by
    simpa [sub_eq_add_neg] using E.expect_add (fun c => c i) (fun _ => -E.mean i)
  rw [h, E.expect_const]
  simp [Ensemble.mean]

lemma Ensemble.variance_nonneg (E : Ensemble n) (i : Fin n) : 0 ≤ E.variance i :=
  E.expect_nonneg fun _ => sq_nonneg _

lemma Ensemble.totalVariance_nonneg (E : Ensemble n) : 0 ≤ E.totalVariance :=
  Finset.sum_nonneg fun i _ => E.variance_nonneg i

lemma Ensemble.variance_le_totalVariance (E : Ensemble n) (i : Fin n) :
    E.variance i ≤ E.totalVariance :=
  Finset.single_le_sum (f := fun i => E.variance i) (fun i _ => E.variance_nonneg i)
    (Finset.mem_univ i)

/-! ## Single-structure predictors: bias-variance and irreducible error -/

/-- **Bias-variance decomposition.** The squared error of predicting the single structure
`x` splits into the ensemble's total variance plus the squared bias. -/
theorem pointLoss_eq (E : Ensemble n) (x : Conf n) :
    E.pointLoss x = E.totalVariance + ∑ i, (E.mean i - x i) ^ 2 := by
  simp only [Ensemble.pointLoss, Ensemble.expect, Ensemble.totalVariance, Ensemble.variance,
    Finset.mul_sum]
  rw [Finset.sum_comm, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hc : ∑ j, E.w j * (E.conf j i - E.mean i) = 0 := by
    have := E.expect_centered i
    simpa [Ensemble.expect] using this
  have hexp : ∀ j : Fin E.card, E.w j * (E.conf j i - x i) ^ 2
      = E.w j * (E.conf j i - E.mean i) ^ 2
        + 2 * (E.mean i - x i) * (E.w j * (E.conf j i - E.mean i))
        + (E.mean i - x i) ^ 2 * E.w j := fun j => by ring
  rw [Finset.sum_congr rfl (fun j _ => hexp j), Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, hc, E.w_sum]
  ring

/-- The best single structure is the mean conformation, and even there the squared error
equals the total variance of the ensemble. -/
theorem pointLoss_mean (E : Ensemble n) :
    E.pointLoss (fun i => E.mean i) = E.totalVariance := by
  rw [pointLoss_eq]
  simp

/-- No single structure does better than the mean. -/
theorem pointLoss_min (E : Ensemble n) (x : Conf n) :
    E.pointLoss (fun i => E.mean i) ≤ E.pointLoss x := by
  rw [pointLoss_mean, pointLoss_eq]
  have : 0 ≤ ∑ i, (E.mean i - x i) ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  linarith

/-- Every single structure has squared error at least the variance of any one degree of
freedom. -/
theorem variance_le_pointLoss (E : Ensemble n) (i : Fin n) (x : Conf n) :
    E.variance i ≤ E.pointLoss x := by
  rw [pointLoss_eq]
  have h1 : E.variance i ≤ E.totalVariance := E.variance_le_totalVariance i
  have h2 : 0 ≤ ∑ i, (E.mean i - x i) ^ 2 := Finset.sum_nonneg fun i _ => sq_nonneg _
  linarith

/-- **Irreducible error of single-structure prediction.** If some degree of freedom is
disordered, then *every* single-structure predictor has strictly positive squared error. -/
theorem pointLoss_pos_of_disordered (E : Ensemble n) {i : Fin n} (hi : E.Disordered i)
    (x : Conf n) : 0 < E.pointLoss x :=
  lt_of_lt_of_le hi (variance_le_pointLoss E i x)

/-! ## Ordered ensembles are exactly the point masses -/

/-- An ensemble has all its variances equal to zero **iff** it is observationally a single
structure, i.e. some conformation `x` computes every observable.  Thus single-structure
models capture precisely the fully ordered ensembles. -/
theorem variance_all_zero_iff_dirac (E : Ensemble n) :
    (∀ i, E.variance i = 0) ↔ ∃ x : Conf n, ∀ f : Conf n → ℝ, E.expect f = f x := by
  constructor
  · intro h
    refine ⟨fun i => E.mean i, fun f => ?_⟩
    have key : ∀ j : Fin E.card, E.w j ≠ 0 → E.conf j = fun i => E.mean i := by
      intro j hj
      funext i
      have h0 : ∑ j : Fin E.card, E.w j * (E.conf j i - E.mean i) ^ 2 = 0 := by
        have := h i
        simpa [Ensemble.variance, Ensemble.expect] using this
      have hterm : E.w j * (E.conf j i - E.mean i) ^ 2 = 0 :=
        (Finset.sum_eq_zero_iff_of_nonneg
          (fun j _ => mul_nonneg (E.w_nonneg j) (sq_nonneg _))).1 h0 j (Finset.mem_univ j)
      rcases mul_eq_zero.1 hterm with h1 | h2
      · exact absurd h1 hj
      · have : E.conf j i - E.mean i = 0 := by
          exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h2
        linarith
    have step : ∀ j ∈ (Finset.univ : Finset (Fin E.card)),
        E.w j * f (E.conf j) = E.w j * f (fun i => E.mean i) := by
      intro j _
      rcases eq_or_lt_of_le (E.w_nonneg j) with hw | hw
      · rw [← hw]; ring
      · rw [key j (ne_of_gt hw)]
    simp only [Ensemble.expect]
    rw [Finset.sum_congr rfl step, ← Finset.sum_mul, E.w_sum, one_mul]
  · rintro ⟨x, hx⟩ i
    have hm : E.mean i = x i := hx (fun c => c i)
    have := hx (fun c => (c i - E.mean i) ^ 2)
    simpa [Ensemble.variance, hm] using this

/-! ## Latent-variable ensemble models -/

/-- **Universality of ensemble models.** Every conformational ensemble is captured exactly
by a latent-variable model (all observables agree). -/
theorem exists_latentModel_captures (E : Ensemble n) :
    ∃ M : LatentModel n, M.Captures E :=
  ⟨⟨E.card, E.w, E.w_nonneg, E.w_sum, E.conf⟩, fun _ => rfl⟩

/-- A single-structure model reproduces only fully ordered ensembles: it can never capture
a disordered degree of freedom. -/
theorem not_isPoint_of_captures_disordered (E : Ensemble n) (M : LatentModel n)
    (hM : M.IsPoint) (hcap : M.Captures E) {i : Fin n} : ¬ E.Disordered i := by
  obtain ⟨x, hx⟩ := hM
  have hdirac : ∀ f : Conf n → ℝ, E.expect f = f x := by
    intro f
    rw [← hcap f]
    simp only [LatentModel.expect, hx, ← Finset.sum_mul, M.p_sum, one_mul]
  have := (variance_all_zero_iff_dirac E).2 ⟨x, hdirac⟩ i
  simp [Ensemble.Disordered, this]

/-- **Disorder forces a genuine ensemble.** A latent-variable model that captures an
ensemble with a disordered degree of freedom must place positive probability on two latent
states whose decoded conformations differ in that degree of freedom. -/
theorem captures_disordered_two_states (E : Ensemble n) (M : LatentModel n)
    (hcap : M.Captures E) {i : Fin n} (hi : E.Disordered i) :
    ∃ z₁ z₂ : Fin M.k, 0 < M.p z₁ ∧ 0 < M.p z₂ ∧ M.decode z₁ i ≠ M.decode z₂ i := by
  by_contra hcon
  push_neg at hcon
  -- pick a latent state of positive probability
  obtain ⟨z₀, -, hz₀⟩ : ∃ z ∈ (Finset.univ : Finset (Fin M.k)), M.p z ≠ 0 := by
    refine Finset.exists_ne_zero_of_sum_ne_zero ?_
    rw [M.p_sum]; norm_num
  have hz₀pos : 0 < M.p z₀ := lt_of_le_of_ne (M.p_nonneg z₀) (Ne.symm hz₀)
  set a : ℝ := M.decode z₀ i with ha
  have hsame : ∀ z : Fin M.k, M.p z * M.decode z i = M.p z * a := by
    intro z
    rcases eq_or_lt_of_le (M.p_nonneg z) with hz | hz
    · rw [← hz]; ring
    · rw [hcon z z₀ hz hz₀pos]
  have hmean : E.mean i = a := by
    have := hcap (fun c => c i)
    rw [Ensemble.mean, ← this]
    simp only [LatentModel.expect]
    rw [Finset.sum_congr rfl (fun z _ => hsame z), ← Finset.sum_mul, M.p_sum, one_mul]
  have hvar : E.variance i = 0 := by
    have := hcap (fun c => (c i - E.mean i) ^ 2)
    rw [Ensemble.variance, ← this]
    simp only [LatentModel.expect]
    refine Finset.sum_eq_zero fun z _ => ?_
    rcases eq_or_lt_of_le (M.p_nonneg z) with hz | hz
    · rw [← hz]; ring
    · rw [hmean, hcon z z₀ hz hz₀pos, ← ha]
      simp
  exact absurd hvar (ne_of_gt hi)

/-- In particular such a model needs at least two latent states. -/
theorem two_le_k_of_captures_disordered (E : Ensemble n) (M : LatentModel n)
    (hcap : M.Captures E) {i : Fin n} (hi : E.Disordered i) : 2 ≤ M.k := by
  obtain ⟨z₁, z₂, -, -, hne⟩ := captures_disordered_two_states E M hcap hi
  have hz : z₁ ≠ z₂ := fun h => hne (by rw [h])
  have h1 := z₁.isLt
  have h2 := z₂.isLt
  have : (z₁ : ℕ) ≠ (z₂ : ℕ) := fun h => hz (Fin.ext h)
  omega

/-! ## Estimation: maximum likelihood recovers the ensemble weights -/

variable {m : ℕ}

/-- Kullback-Leibler divergence of the fitted weights `q` from the true weights `p`. -/
noncomputable def klDiv (p q : Fin m → ℝ) : ℝ := ∑ j, p j * Real.log (p j / q j)

/-- Expected negative log-likelihood (cross-entropy) of the fitted weights `q` under the
true weights `p`. -/
noncomputable def crossEntropy (p q : Fin m → ℝ) : ℝ := -∑ j, p j * Real.log (q j)

/-- Termwise Gibbs bound: `p - q ≤ p log (p / q)`. -/
lemma klDiv_term_le {a b : ℝ} (ha : 0 ≤ a) (hb : 0 < b) :
    a - b ≤ a * Real.log (a / b) := by
  rcases eq_or_lt_of_le ha with h0 | h0
  · rw [← h0]; simp; linarith
  · have hlog : Real.log (b / a) ≤ b / a - 1 :=
      Real.log_le_sub_one_of_pos (div_pos hb h0)
    have hswap : Real.log (a / b) = -Real.log (b / a) := by
      rw [← Real.log_inv]
      congr 1
      field_simp
    rw [hswap]
    have : a * (b / a - 1) = b - a := by field_simp
    nlinarith [hlog, h0]

/-- Strict termwise Gibbs bound when `a ≠ b`. -/
lemma klDiv_term_lt {a b : ℝ} (ha : 0 ≤ a) (hb : 0 < b) (hab : a ≠ b) :
    a - b < a * Real.log (a / b) := by
  rcases eq_or_lt_of_le ha with h0 | h0
  · rw [← h0]; simp; linarith
  · have hne : b / a ≠ 1 := by
      intro h
      apply hab
      field_simp at h
      linarith
    have hlog : Real.log (b / a) < b / a - 1 :=
      Real.log_lt_sub_one_of_pos (div_pos hb h0) hne
    have hswap : Real.log (a / b) = -Real.log (b / a) := by
      rw [← Real.log_inv]
      congr 1
      field_simp
    rw [hswap]
    have : a * (b / a - 1) = b - a := by field_simp
    nlinarith [hlog, h0]

/-- **Gibbs' inequality.**  The Kullback-Leibler divergence of a (strictly positive)
fitted weight vector from the true weights is nonnegative. -/
theorem klDiv_nonneg {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1) : 0 ≤ klDiv p q := by
  have hle : ∑ j, (p j - q j) ≤ klDiv p q :=
    Finset.sum_le_sum fun j _ => klDiv_term_le (hp j) (hq j)
  rw [Finset.sum_sub_distrib, hps, hqs] at hle
  simpa using hle

/-- The KL divergence is strictly positive unless the fitted weights are the true ones. -/
theorem klDiv_pos_of_ne {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1) (hne : p ≠ q) : 0 < klDiv p q := by
  obtain ⟨j₀, hj₀⟩ : ∃ j, p j ≠ q j := by
    by_contra h
    push_neg at h
    exact hne (funext h)
  have hlt : ∑ j, (p j - q j) < klDiv p q := by
    refine Finset.sum_lt_sum (fun j _ => klDiv_term_le (hp j) (hq j))
      ⟨j₀, Finset.mem_univ j₀, klDiv_term_lt (hp j₀) (hq j₀) hj₀⟩
  rw [Finset.sum_sub_distrib, hps, hqs] at hlt
  simpa using hlt

/-- The KL divergence vanishes exactly at the true weights. -/
theorem klDiv_eq_zero_iff {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1) : klDiv p q = 0 ↔ p = q := by
  constructor
  · intro h
    by_contra hne
    exact absurd h (ne_of_gt (klDiv_pos_of_ne hp hps hq hqs hne))
  · rintro rfl
    refine Finset.sum_eq_zero fun j _ => ?_
    rcases eq_or_lt_of_le (hp j) with h0 | h0
    · rw [← h0]; ring
    · rw [div_self (ne_of_gt h0)]
      simp

lemma crossEntropy_sub_crossEntropy {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j)
    (hq : ∀ j, 0 < q j) : crossEntropy p q - crossEntropy p p = klDiv p q := by
  simp only [crossEntropy, klDiv, ← Finset.sum_neg_distrib, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  rcases eq_or_lt_of_le (hp j) with h0 | h0
  · rw [← h0]; ring
  · rw [Real.log_div (ne_of_gt h0) (ne_of_gt (hq j))]
    ring

/-- **Consistency of maximum-likelihood ensemble fitting.** Over a fixed library of
conformations, the expected negative log-likelihood is minimised by the true weights, and
by no other weight vector. -/
theorem crossEntropy_min_iff {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hps : ∑ j, p j = 1)
    (hq : ∀ j, 0 < q j) (hqs : ∑ j, q j = 1) :
    crossEntropy p p ≤ crossEntropy p q ∧ (crossEntropy p q = crossEntropy p p ↔ p = q) := by
  have hdiff := crossEntropy_sub_crossEntropy hp hq
  have hnn := klDiv_nonneg hp hps hq hqs
  refine ⟨by linarith, ?_⟩
  constructor
  · intro h
    have : klDiv p q = 0 := by linarith
    exact (klDiv_eq_zero_iff hp hps hq hqs).1 this
  · rintro rfl
    rfl

/-! ## Summary theorem -/

/-- **How to model an intrinsically disordered region.**  Let `E` be a conformational
ensemble with a disordered degree of freedom `i`.  Then

1. no single-structure predictor reproduces `E`, and every single structure incurs a
   squared error at least `E.variance i > 0` (at best exactly the total variance);
2. some latent-variable ensemble model reproduces `E` exactly on every observable, and any
   such model uses at least two latent states with distinct geometry at `i`;
3. fitting the ensemble weights by maximum likelihood over the conformational library is
   consistent: the true weights are the unique minimiser of the expected negative
   log-likelihood.
-/
theorem disordered_region_model (E : Ensemble n) {i : Fin n} (hi : E.Disordered i) :
    (∀ x : Conf n, E.variance i ≤ E.pointLoss x ∧ 0 < E.pointLoss x) ∧
    E.pointLoss (fun i => E.mean i) = E.totalVariance ∧
    (¬ ∃ x : Conf n, ∀ f : Conf n → ℝ, E.expect f = f x) ∧
    (∃ M : LatentModel n, M.Captures E) ∧
    (∀ M : LatentModel n, M.Captures E → 2 ≤ M.k) ∧
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∀ j, 0 ≤ p j) → ∑ j, p j = 1 → (∀ j, 0 < q j) →
      ∑ j, q j = 1 → crossEntropy p p ≤ crossEntropy p q ∧
        (crossEntropy p q = crossEntropy p p ↔ p = q)) := by
  refine ⟨fun x => ⟨variance_le_pointLoss E i x, pointLoss_pos_of_disordered E hi x⟩,
    pointLoss_mean E, ?_, exists_latentModel_captures E,
    fun M hM => two_le_k_of_captures_disordered E M hM hi,
    fun m p q hp hps hq hqs => crossEntropy_min_iff hp hps hq hqs⟩
  rintro hdirac
  exact absurd ((variance_all_zero_iff_dirac E).2 hdirac i) (ne_of_gt hi)

/-! ## A concrete disordered ensemble (non-vacuity check) -/

/-- A two-state ensemble of a single degree of freedom: the values `0` and `1` with equal
weights. -/
noncomputable def twoState : Ensemble 1 where
  card := 2
  conf := ![fun _ => 0, fun _ => 1]
  w := ![1 / 2, 1 / 2]
  w_nonneg := by
    intro j
    fin_cases j <;> norm_num
  w_sum := by norm_num [Fin.sum_univ_succ]

lemma twoState_mean : twoState.mean 0 = 1 / 2 := by
  show ∑ j, twoState.w j * twoState.conf j 0 = 1 / 2
  simp only [twoState]
  norm_num [Fin.sum_univ_succ]

/-- The two-state ensemble really is disordered, so the hypotheses above are satisfiable. -/
lemma twoState_disordered : twoState.Disordered 0 := by
  have h : twoState.variance 0 = 1 / 4 := by
    show ∑ j, twoState.w j * (twoState.conf j 0 - twoState.mean 0) ^ 2 = 1 / 4
    rw [twoState_mean]
    simp only [twoState]
    norm_num [Fin.sum_univ_succ]
  rw [Ensemble.Disordered, h]
  norm_num

end IDR
