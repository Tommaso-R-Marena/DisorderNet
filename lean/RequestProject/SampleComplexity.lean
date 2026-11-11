/-
# Part V.3  How much data does an ensemble cost?  A minimax lower bound

Earlier parts bound the *size* a correct model must have (`RequestProject.Metric`,
`RequestProject.Quantization`).  This file bounds the amount of *data* needed to identify
the target at all -- a statement about the world, not about any particular architecture:
it holds for **every** estimator, however clever, that sees `n` independent conformations
drawn from the true ensemble.

* `prodP` is the law of `n` independent draws, and `prodP_sum_one` says it is a probability
  distribution on the sample space `Fin n → Fin m`.
* `l1_prodP_le` is the **tensorisation bound** `‖p^{⊗n} - q^{⊗n}‖₁ ≤ n ‖p - q‖₁`: `n` draws
  can separate two ensembles only in proportion to `n` times their single-draw distance.
* `le_cam` is **Le Cam's two-point bound**: for any estimator `T` and any two candidate
  ensembles, the sum of the two expected `ℓ¹` risks is at least
  `‖p-q‖₁ · (1 - TV(p^{⊗n}, q^{⊗n}))`.  No estimator can be accurate at both members of an
  indistinguishable pair.
* `sample_complexity` and `sample_complexity_two_point` combine them: an estimator that is
  `eps`-accurate in expectation on every ensemble needs `n ≥ 1/(4·eps)` samples, and
  `exists_hard_pair` shows the hypotheses are satisfiable whenever the library has at least
  two conformations.  Accuracy therefore costs data *at least inversely* in the tolerance,
  independently of model class -- the statistical counterpart of the capacity laws.
* `support_honest_needs_coverage`: for the class of models whose output is supported on the
  conformations actually observed -- every reweighting-style ensemble model, every
  "weighted frames" model -- the requirement is far more brutal: reaching `ℓ¹` accuracy
  `eps` on a uniform target over `m` conformations requires `n ≥ m(1 - eps)` samples.  Since
  `m` is exponential in the length of a disordered region (`RequestProject.Chain`), such
  models cannot be trained to accuracy by sampling alone.
-/
import Mathlib
import RequestProject.DisorderedRegions
import RequestProject.EnsembleCore
import RequestProject.Metric

namespace IDR

open Finset
open scoped Classical

namespace Learn

/-- The `ℓ¹` distance between two weight vectors on an arbitrary finite index type. -/
noncomputable def l1 {ι : Type*} [Fintype ι] (u v : ι → ℝ) : ℝ := ∑ i, |u i - v i|

variable {ι : Type*} [Fintype ι]

lemma l1_nonneg (u v : ι → ℝ) : 0 ≤ l1 u v :=
  Finset.sum_nonneg fun _ _ => abs_nonneg _

lemma l1_comm (u v : ι → ℝ) : l1 u v = l1 v u :=
  Finset.sum_congr rfl fun _ _ => abs_sub_comm _ _

lemma l1_triangle (u v w : ι → ℝ) : l1 u w ≤ l1 u v + l1 v w := by
  rw [l1, l1, l1, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  calc |u i - w i| = |(u i - v i) + (v i - w i)| := by ring_nf
    _ ≤ |u i - v i| + |v i - w i| := abs_add_le _ _

/-! ## The law of `n` independent draws -/

variable {m n : ℕ}

/-- The law of `n` independent conformations drawn from the ensemble `p`. -/
noncomputable def prodP (p : Fin m → ℝ) (s : Fin n → Fin m) : ℝ := ∏ i, p (s i)

lemma prodP_nonneg {p : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (s : Fin n → Fin m) :
    0 ≤ prodP p s :=
  Finset.prod_nonneg fun _ _ => hp _

lemma prodP_sum_one {p : Fin m → ℝ} (hps : ∑ j, p j = 1) :
    ∑ s : Fin n → Fin m, prodP p s = 1 := by
  have h := Finset.prod_univ_sum (ι := Fin n) (κ := fun _ => Fin m)
    (fun _ => (Finset.univ : Finset (Fin m))) (fun _ j => p j)
  rw [Fintype.piFinset_univ] at h
  simp [hps] at h
  simpa [prodP] using h.symm

lemma prodP_cons (p : Fin m → ℝ) (a : Fin m) (t : Fin n → Fin m) :
    prodP p (Fin.cons a t) = p a * prodP p t := by
  simp [prodP, Fin.prod_univ_succ]

lemma sum_cons_split (F : (Fin (n+1) → Fin m) → ℝ) :
    ∑ s : Fin (n+1) → Fin m, F s = ∑ a : Fin m, ∑ t : Fin n → Fin m, F (Fin.cons a t) := by
  have h1 : ∑ s : Fin (n+1) → Fin m, F s
      = ∑ x : Fin m × (Fin n → Fin m), F (Fin.cons x.1 x.2) :=
    (Fintype.sum_equiv (Fin.consEquiv (fun _ => Fin m)) _ _ (fun _ => rfl)).symm
  rw [h1, Fintype.sum_prod_type]

/-- **Tensorisation.**  `n` independent draws separate two ensembles by at most `n` times
their single-draw `ℓ¹` distance.  Data accumulate information only linearly. -/
theorem l1_prodP_le {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 ≤ q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) (n : ℕ) :
    l1 (prodP (n := n) p) (prodP (n := n) q) ≤ n * l1 p q := by
  induction n with
  | zero =>
      have : l1 (prodP (n := 0) p) (prodP (n := 0) q) = 0 := by
        simp [l1, prodP]
      rw [this]
      exact mul_nonneg (by positivity) (l1_nonneg _ _)
  | succ n ih =>
      have hstep : l1 (prodP (n := n+1) p) (prodP (n := n+1) q)
          ≤ l1 p q + l1 (prodP (n := n) p) (prodP (n := n) q) := by
        have hsplit : l1 (prodP (n := n+1) p) (prodP (n := n+1) q)
            = ∑ a : Fin m, ∑ t : Fin n → Fin m,
                |p a * prodP p t - q a * prodP q t| := by
          simp only [l1]
          rw [sum_cons_split (fun s => |prodP p s - prodP q s|)]
          exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun t _ => by
            rw [prodP_cons, prodP_cons]
        have hbound : ∀ a : Fin m, ∀ t : Fin n → Fin m,
            |p a * prodP p t - q a * prodP q t|
              ≤ |p a - q a| * prodP p t + q a * |prodP p t - prodP q t| := by
          intro a t
          have hrw : p a * prodP p t - q a * prodP q t
              = (p a - q a) * prodP p t + q a * (prodP p t - prodP q t) := by ring
          calc |p a * prodP p t - q a * prodP q t|
              = |(p a - q a) * prodP p t + q a * (prodP p t - prodP q t)| := by rw [hrw]
            _ ≤ |(p a - q a) * prodP p t| + |q a * (prodP p t - prodP q t)| := abs_add_le _ _
            _ = |p a - q a| * prodP p t + q a * |prodP p t - prodP q t| := by
                rw [abs_mul, abs_mul, abs_of_nonneg (prodP_nonneg hp t), abs_of_nonneg (hq a)]
        have hsum : ∑ a : Fin m, ∑ t : Fin n → Fin m,
              |p a * prodP p t - q a * prodP q t|
            ≤ ∑ a : Fin m, ∑ t : Fin n → Fin m,
              (|p a - q a| * prodP p t + q a * |prodP p t - prodP q t|) :=
          Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun t _ => hbound a t
        have hval : ∑ a : Fin m, ∑ t : Fin n → Fin m,
              (|p a - q a| * prodP p t + q a * |prodP p t - prodP q t|)
            = l1 p q + l1 (prodP (n := n) p) (prodP (n := n) q) := by
          have hinner : ∀ a : Fin m, ∑ t : Fin n → Fin m,
              (|p a - q a| * prodP p t + q a * |prodP p t - prodP q t|)
              = |p a - q a| + q a * l1 (prodP (n := n) p) (prodP (n := n) q) := by
            intro a
            rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
              prodP_sum_one hps, mul_one]
            rfl
          rw [Finset.sum_congr rfl fun a (_ : a ∈ Finset.univ) => hinner a,
            Finset.sum_add_distrib, ← Finset.sum_mul, hqs, one_mul]
          rfl
        rw [hsplit]
        calc ∑ a : Fin m, ∑ t : Fin n → Fin m, |p a * prodP p t - q a * prodP q t|
            ≤ ∑ a : Fin m, ∑ t : Fin n → Fin m,
              (|p a - q a| * prodP p t + q a * |prodP p t - prodP q t|) := hsum
          _ = l1 p q + l1 (prodP (n := n) p) (prodP (n := n) q) := hval
      have : (n : ℝ) * l1 p q + l1 p q = ((n : ℝ) + 1) * l1 p q := by ring
      push_cast
      linarith [ih]

/-! ## Le Cam's two-point bound -/

/-- The expected `ℓ¹` risk of an estimator `T` at the ensemble `p`, given `n` independent
draws. -/
noncomputable def risk (p : Fin m → ℝ) (T : (Fin n → Fin m) → (Fin m → ℝ)) : ℝ :=
  ∑ s, prodP p s * l1 (T s) p

/-- The total overlap of the two sample laws is `1 - TV`. -/
lemma sum_min_prodP {p q : Fin m → ℝ} (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) :
    ∑ s : Fin n → Fin m, min (prodP p s) (prodP q s)
      = 1 - l1 (prodP (n := n) p) (prodP (n := n) q) / 2 := by
  have hmin : ∀ s : Fin n → Fin m, min (prodP p s) (prodP q s)
      = (prodP p s + prodP q s - |prodP p s - prodP q s|) / 2 := by
    intro s
    rcases le_total (prodP p s) (prodP q s) with h | h
    · rw [min_eq_left h, abs_of_nonpos (by linarith)]; ring
    · rw [min_eq_right h, abs_of_nonneg (by linarith)]; ring
  rw [Finset.sum_congr rfl fun s (_ : s ∈ Finset.univ) => hmin s]
  have h1 : ∑ s : Fin n → Fin m, (prodP p s + prodP q s - |prodP p s - prodP q s|) / 2
      = ((∑ s : Fin n → Fin m, prodP p s) + (∑ s : Fin n → Fin m, prodP q s)
          - ∑ s : Fin n → Fin m, |prodP p s - prodP q s|) / 2 := by
    rw [← Finset.sum_div]
    congr 1
    rw [Finset.sum_sub_distrib, Finset.sum_add_distrib]
  rw [h1, prodP_sum_one hps, prodP_sum_one hqs]
  rw [l1]
  ring

/-- **Le Cam's two-point lemma.**  For every estimator `T`, the two risks cannot both be
small at an indistinguishable pair of ensembles. -/
theorem le_cam {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 ≤ q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) (T : (Fin n → Fin m) → (Fin m → ℝ)) :
    l1 p q * (1 - l1 (prodP (n := n) p) (prodP (n := n) q) / 2) ≤ risk p T + risk q T := by
  have hsep : ∀ s : Fin n → Fin m, l1 p q ≤ l1 (T s) p + l1 (T s) q := by
    intro s
    calc l1 p q ≤ l1 p (T s) + l1 (T s) q := l1_triangle _ _ _
      _ = l1 (T s) p + l1 (T s) q := by rw [l1_comm p (T s)]
  have hlow : ∀ s : Fin n → Fin m,
      min (prodP p s) (prodP q s) * l1 p q
        ≤ prodP p s * l1 (T s) p + prodP q s * l1 (T s) q := by
    intro s
    have h1 : min (prodP p s) (prodP q s) * l1 (T s) p ≤ prodP p s * l1 (T s) p :=
      mul_le_mul_of_nonneg_right (min_le_left _ _) (l1_nonneg _ _)
    have h2 : min (prodP p s) (prodP q s) * l1 (T s) q ≤ prodP q s * l1 (T s) q :=
      mul_le_mul_of_nonneg_right (min_le_right _ _) (l1_nonneg _ _)
    have h3 : min (prodP p s) (prodP q s) * l1 p q
        ≤ min (prodP p s) (prodP q s) * (l1 (T s) p + l1 (T s) q) := by
      refine mul_le_mul_of_nonneg_left (hsep s) ?_
      exact le_min (prodP_nonneg hp s) (prodP_nonneg hq s)
    nlinarith [h1, h2, h3]
  have hsum := Finset.sum_le_sum fun s (_ : s ∈ Finset.univ) => hlow s
  rw [← Finset.sum_mul, sum_min_prodP hps hqs] at hsum
  rw [risk, risk, ← Finset.sum_add_distrib, mul_comm]
  exact hsum

/-- **Sample complexity.**  If an estimator has expected `ℓ¹` risk at most `eps` at both
members of a pair of ensembles at distance `D`, then `n ≥ (D - 2·eps)·2/D²`. -/
theorem sample_complexity {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j) (hq : ∀ j, 0 ≤ q j)
    (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1) (T : (Fin n → Fin m) → (Fin m → ℝ))
    {eps : ℝ} (hTp : risk p T ≤ eps) (hTq : risk q T ≤ eps) :
    l1 p q * (1 - n * l1 p q / 2) ≤ 2 * eps := by
  have hlc := le_cam hp hq hps hqs T
  have htens := l1_prodP_le hp hq hps hqs n
  have hD : 0 ≤ l1 p q := l1_nonneg _ _
  have hmono : l1 p q * (1 - (n : ℝ) * l1 p q / 2)
      ≤ l1 p q * (1 - l1 (prodP (n := n) p) (prodP (n := n) q) / 2) := by
    have : l1 (prodP (n := n) p) (prodP (n := n) q) / 2 ≤ (n : ℝ) * l1 p q / 2 := by
      linarith
    nlinarith
  linarith

/-- **Two-point form.**  An estimator that is `eps`-accurate in expectation on both members
of a pair at distance exactly `4·eps` needs at least `1/(4·eps)` samples. -/
theorem sample_complexity_two_point {p q : Fin m → ℝ} (hp : ∀ j, 0 ≤ p j)
    (hq : ∀ j, 0 ≤ q j) (hps : ∑ j, p j = 1) (hqs : ∑ j, q j = 1)
    (T : (Fin n → Fin m) → (Fin m → ℝ)) {eps : ℝ} (heps : 0 < eps)
    (hD : l1 p q = 4 * eps) (hTp : risk p T ≤ eps) (hTq : risk q T ≤ eps) :
    1 / (4 * eps) ≤ (n : ℝ) := by
  have h := sample_complexity hp hq hps hqs T hTp hTq
  rw [hD] at h
  -- `4 eps (1 - 2 n eps) ≤ 2 eps`  gives  `4 n eps² ≥ eps`
  have h2 : eps ≤ 4 * (n : ℝ) * eps ^ 2 := by nlinarith
  have hpos : (0:ℝ) < 4 * eps := by linarith
  rw [div_le_iff₀ hpos]
  nlinarith

/-- A two-point weight vector on the library. -/
noncomputable def twoPt (j0 j1 : Fin m) (a b : ℝ) : Fin m → ℝ :=
  fun j => (if j = j0 then a else 0) + (if j = j1 then b else 0)

lemma sum_twoPt {j0 j1 : Fin m} (a b : ℝ) : ∑ j, twoPt j0 j1 a b j = a + b := by
  simp [twoPt, Finset.sum_add_distrib]

/-- The hard pairs exist: on a library of at least two conformations there is, for every
tolerance `eps ≤ 1/4`, a pair of legitimate ensembles at `ℓ¹` distance exactly `4·eps`. -/
theorem exists_hard_pair {m : ℕ} (hm : 2 ≤ m) {eps : ℝ} (heps : 0 < eps) (heps' : eps ≤ 1/4) :
    ∃ p q : Fin m → ℝ, (∀ j, 0 ≤ p j) ∧ (∀ j, 0 ≤ q j) ∧ (∑ j, p j = 1) ∧ (∑ j, q j = 1)
      ∧ l1 p q = 4 * eps := by
  have h0 : (0 : ℕ) < m := by omega
  have h1 : (1 : ℕ) < m := by omega
  set j0 : Fin m := ⟨0, h0⟩ with hj0
  set j1 : Fin m := ⟨1, h1⟩ with hj1
  have hne : j0 ≠ j1 := by simp [hj0, hj1, Fin.ext_iff]
  refine ⟨twoPt j0 j1 (1/2 + eps) (1/2 - eps), twoPt j0 j1 (1/2 - eps) (1/2 + eps),
    ?_, ?_, ?_, ?_, ?_⟩
  · intro j
    by_cases ha : j = j0
    · have hb : j ≠ j1 := by rw [ha]; exact hne
      simp only [twoPt, if_pos ha, if_neg hb, add_zero]
      linarith
    · by_cases hb : j = j1
      · simp only [twoPt, if_neg ha, if_pos hb, zero_add]
        linarith
      · simp [twoPt, ha, hb]
  · intro j
    by_cases ha : j = j0
    · have hb : j ≠ j1 := by rw [ha]; exact hne
      simp only [twoPt, if_pos ha, if_neg hb, add_zero]
      linarith
    · by_cases hb : j = j1
      · simp only [twoPt, if_neg ha, if_pos hb, zero_add]
        linarith
      · simp [twoPt, ha, hb]
  · rw [sum_twoPt]; ring
  · rw [sum_twoPt]; ring
  · have hdiff : ∀ j : Fin m,
        |twoPt j0 j1 (1/2 + eps) (1/2 - eps) j - twoPt j0 j1 (1/2 - eps) (1/2 + eps) j|
          = twoPt j0 j1 (2 * eps) (2 * eps) j := by
      intro j
      by_cases ha : j = j0
      · have hb : j ≠ j1 := by rw [ha]; exact hne
        simp only [twoPt, if_pos ha, if_neg hb, add_zero]
        rw [show (1/2 + eps) - (1/2 - eps) = 2 * eps by ring, abs_of_nonneg (by linarith)]
      · by_cases hb : j = j1
        · simp only [twoPt, if_neg ha, if_pos hb, zero_add]
          rw [show (1/2 - eps) - (1/2 + eps) = -(2 * eps) by ring, abs_neg,
            abs_of_nonneg (by linarith)]
        · simp [twoPt, ha, hb]
    rw [l1, Finset.sum_congr rfl fun j (_ : j ∈ Finset.univ) => hdiff j, sum_twoPt]
    ring

/-! ## Models that can only reuse what they have seen -/

/-- A *support-honest* model returns weight only on conformations that appeared in the
sample: this is exactly what reweighting-style and "weighted frames" ensemble models do. -/
def SupportHonest (T : (Fin n → Fin m) → (Fin m → ℝ)) : Prop :=
  ∀ (s : Fin n → Fin m) (j : Fin m), (∀ i, s i ≠ j) → T s j = 0

/-- The uniform ensemble on a library of `m` conformations: the maximally disordered
target. -/
noncomputable def unifW (m : ℕ) : Fin m → ℝ := fun _ => 1 / (m : ℝ)

/-- A support-honest model is off by at least the population of the conformations it has
not seen. -/
theorem support_honest_error {T : (Fin n → Fin m) → (Fin m → ℝ)} (hT : SupportHonest T)
    (hm : 0 < m) (s : Fin n → Fin m) :
    ((m : ℝ) - n) / m ≤ l1 (T s) (unifW m) := by
  have hmpos : (0:ℝ) < m := by exact_mod_cast hm
  classical
  set seen : Finset (Fin m) := Finset.image s Finset.univ with hseen
  have hcard : seen.card ≤ n := by
    calc seen.card ≤ (Finset.univ : Finset (Fin n)).card := Finset.card_image_le
      _ = n := by simp
  have hunseen : ∀ j ∈ (Finset.univ \ seen), |T s j - unifW m j| = 1 / (m:ℝ) := by
    intro j hj
    have hnot : j ∉ seen := (Finset.mem_sdiff.1 hj).2
    have hzero : T s j = 0 := by
      refine hT s j fun i hi => hnot ?_
      rw [hseen]
      exact Finset.mem_image.2 ⟨i, Finset.mem_univ i, hi⟩
    rw [hzero, unifW]
    rw [zero_sub, abs_neg, abs_of_nonneg (le_of_lt (one_div_pos.2 hmpos))]
  have hsub : ∑ j ∈ (Finset.univ \ seen), |T s j - unifW m j| ≤ l1 (T s) (unifW m) := by
    refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) ?_
    intro j _ _
    exact abs_nonneg _
  rw [Finset.sum_congr rfl hunseen, Finset.sum_const, nsmul_eq_mul] at hsub
  have hcard' : ((m : ℝ) - n) ≤ ((Finset.univ \ seen).card : ℝ) := by
    have h1 : (Finset.univ \ seen).card = m - seen.card := by
      rw [Finset.card_sdiff]
      simp
    have h2 : seen.card ≤ m := by
      calc seen.card ≤ (Finset.univ : Finset (Fin m)).card := Finset.card_le_univ _
        _ = m := by simp
    rw [h1]
    have : ((m - seen.card : ℕ) : ℝ) = (m : ℝ) - seen.card := by
      rw [Nat.cast_sub h2]
    rw [this]
    have : (seen.card : ℝ) ≤ n := by exact_mod_cast hcard
    linarith
  have : ((m : ℝ) - n) / m ≤ ((Finset.univ \ seen).card : ℝ) * (1 / (m:ℝ)) := by
    rw [div_eq_mul_one_div]
    exact mul_le_mul_of_nonneg_right hcard' (by positivity)
  linarith

/-- **Coverage is unavoidable for support-honest models.**  Reaching expected `ℓ¹` accuracy
`eps` on the uniform target over `m` conformations requires `n ≥ m(1 - eps)` samples.  With
`m` exponential in the length of a disordered region, sampling-based ensemble models cannot
reach accuracy by data alone. -/
theorem support_honest_needs_coverage {T : (Fin n → Fin m) → (Fin m → ℝ)}
    (hT : SupportHonest T) (hm : 0 < m) {eps : ℝ} (hrisk : risk (unifW m) T ≤ eps) :
    (m : ℝ) * (1 - eps) ≤ n := by
  have hmpos : (0:ℝ) < m := by exact_mod_cast hm
  have hunif_nonneg : ∀ j, 0 ≤ unifW m j :=
    fun _ => div_nonneg zero_le_one (Nat.cast_nonneg m)
  have hunif_sum : ∑ j, unifW m j = 1 := by
    simp [unifW, Finset.sum_const, nsmul_eq_mul]
    field_simp
  have hbound : ((m : ℝ) - n) / m ≤ risk (unifW m) T := by
    have hpoint : ∀ s : Fin n → Fin m,
        prodP (unifW m) s * (((m : ℝ) - n) / m) ≤ prodP (unifW m) s * l1 (T s) (unifW m) :=
      fun s => mul_le_mul_of_nonneg_left (support_honest_error hT hm s)
        (prodP_nonneg hunif_nonneg s)
    have hsum := Finset.sum_le_sum fun s (_ : s ∈ Finset.univ) => hpoint s
    rw [← Finset.sum_mul, prodP_sum_one hunif_sum, one_mul] at hsum
    exact hsum
  have h : ((m : ℝ) - n) / m ≤ eps := le_trans hbound hrisk
  rw [div_le_iff₀ hmpos] at h
  nlinarith

end Learn

end IDR
