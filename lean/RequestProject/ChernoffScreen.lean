/-
# Part XCI.3  An exponential-tail p-value, and a screen only logarithmic in the proteome

The quadratic budget of `RequestProject.ScreenBudget` is a property of Chebyshev's inequality,
not of the experiment.  The read-out law of Part XC is an explicit product of independent
Bernoulli reads, so its moment generating function is available exactly, and Markov's inequality
applied to it gives a tail that decays exponentially rather than quadratically.

* `sum_recProb_exp` — the exact moment generating function `E[e^{λ·count}] = (1 - q + q e^λ)ⁿ`,
  proved by induction on the number of molecules.
* `markov_tail`, `exp_quad_bound`, `chernoff_tail` — Markov on the exponential, the elementary
  bound `e^x ≤ 1 + x + ¾x²` on `[0,1]`, and the resulting Chernoff bound
  `P(count ≥ n·q + a) ≤ exp(-a²/4n)`, valid at every finite `n` with no asymptotics.
* `expP`, `expP_superuniform` — the p-value `exp(-(count - n·q₀)²/4n)`, and its validity.
* `expP_power`, `expSamples`, `expSamples_spec` — its power, at a cost of
  `(16·log(1/u) + 1/α)/Δ²` molecules per candidate: *logarithmic* in the demanded level where the
  two-moment design paid `1/u`.
* `screen_fdr_control_exp` — the screen built from these p-values controls the false discovery
  rate exactly as before, since Benjamini–Hochberg cares only about superuniformity.
* `expSamples_screen`, `screen_exp_cheaper` — and the separation: at the worst-case BH threshold
  `q/N` the exponential design costs `⌈(16·log(N/q) + 1/α)/Δ²⌉` per candidate, so a screen large
  enough is cheaper by the whole factor between `log N` and `N`.

The moral is not that the physics got easier.  It is that in a proteome-scale screen a large part
of the molecule budget is decided by which inequality the analysis is prepared to prove.
-/
import Mathlib
import RequestProject.NoisyDetection
import RequestProject.FalseDiscovery
import RequestProject.ScreenBudget

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR
namespace Chernoff

open IDR.Noisy IDR.FDR IDR.Screen

/-! ## 1. The moment generating function of the read-out law -/

lemma cnt_le (n : ℕ) (s : Fin n → Bool) : cnt s ≤ (n : ℝ) := by
  unfold cnt
  calc ∑ i, (if s i then (1:ℝ) else 0) ≤ ∑ _i : Fin n, (1:ℝ) := by
        refine Finset.sum_le_sum (fun i _ => ?_)
        by_cases h : s i <;> simp [h]
    _ = (n : ℝ) := by simp

/-- **The exact moment generating function.**  `E[exp(λ·count)] = (1 - q + q·e^λ)ⁿ`. -/
lemma sum_recProb_exp (q lam : ℝ) (n : ℕ) :
    ∑ s : Fin n → Bool, recProb q s * Real.exp (lam * cnt s)
      = (1 - q + q * Real.exp lam) ^ n := by
  induction n with
  | zero => simp [recProb, cnt]
  | succ n ih =>
    rw [sum_cons (fun s => recProb q s * Real.exp (lam * cnt s))]
    have hstep : ∀ b : Bool,
        (∑ t : Fin n → Bool, recProb q (Fin.cons b t) * Real.exp (lam * cnt (Fin.cons b t)))
          = (bern q b * Real.exp (lam * (if b then (1:ℝ) else 0))) *
            (1 - q + q * Real.exp lam) ^ n := by
      intro b
      rw [← ih, Finset.mul_sum]
      refine Finset.sum_congr rfl (fun t _ => ?_)
      rw [recProb_cons, cnt_cons, mul_add, Real.exp_add]
      ring
    rw [Fintype.sum_bool, hstep true, hstep false]
    simp only [bern, if_pos, Bool.false_eq_true, if_false]
    rw [pow_succ]
    ring_nf
    rw [mul_comm]
    ring_nf
    rw [Real.exp_zero]
    ring

/-! ## 2. Markov's inequality on the exponential -/

lemma markov_tail {q lam c : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (hlam : 0 ≤ lam) (n : ℕ) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s), recProb q s
      ≤ Real.exp (-lam * c) * (1 - q + q * Real.exp lam) ^ n := by
  have hkey : ∀ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s),
      recProb q s ≤ Real.exp (-lam * c) * (recProb q s * Real.exp (lam * cnt s)) := by
    intro s hs
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hs
    have hp := recProb_nonneg h0 h1 s
    have h1' : Real.exp (-lam * c) * Real.exp (lam * cnt s) = Real.exp (lam * (cnt s - c)) := by
      rw [← Real.exp_add]; ring_nf
    have h2 : (1:ℝ) ≤ Real.exp (lam * (cnt s - c)) :=
      Real.one_le_exp (by nlinarith)
    calc recProb q s = recProb q s * 1 := by ring
      _ ≤ recProb q s * Real.exp (lam * (cnt s - c)) := by
          exact mul_le_mul_of_nonneg_left h2 hp
      _ = Real.exp (-lam * c) * (recProb q s * Real.exp (lam * cnt s)) := by
          rw [← h1']; ring
  calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s), recProb q s
      ≤ ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s),
          Real.exp (-lam * c) * (recProb q s * Real.exp (lam * cnt s)) := Finset.sum_le_sum hkey
    _ ≤ ∑ s : Fin n → Bool, Real.exp (-lam * c) * (recProb q s * Real.exp (lam * cnt s)) := by
        refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) ?_
        intro s _ _
        have := recProb_nonneg h0 h1 s
        positivity
    _ = Real.exp (-lam * c) * (1 - q + q * Real.exp lam) ^ n := by
        rw [← Finset.mul_sum, sum_recProb_exp]

/-! ## 3. The Chernoff tail bound -/

lemma exp_quad_bound {x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ 1) :
    Real.exp x ≤ 1 + x + (3 / 4) * x ^ 2 := by
  have h := Real.exp_bound (x := x) (by rw [abs_of_nonneg h0]; exact h1) (n := 2) (by norm_num)
  simp [Finset.sum_range_succ] at h
  cases abs_le.mp h with
  | intro _ hr => nlinarith

/-- **Chernoff's bound for the read-out law.**  The chance that the positive count exceeds its
mean by `a` decays exponentially, `exp(-a²/4n)` — no normal approximation, no asymptotics. -/
theorem chernoff_tail {q a : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ} (hn : 0 < n)
    (ha0 : 0 ≤ a) (ha1 : a ≤ n) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => (n : ℝ) * q + a ≤ cnt s), recProb q s
      ≤ Real.exp (-(a ^ 2) / (4 * n)) := by
  have hnpos : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  set lam : ℝ := a / n with hlam
  have hlam0 : 0 ≤ lam := by rw [hlam]; positivity
  have hlam1 : lam ≤ 1 := by
    rw [hlam, div_le_one hnpos]; exact ha1
  refine le_trans (markov_tail h0 h1 hlam0 n) ?_
  have hbase : 1 - q + q * Real.exp lam ≤ Real.exp (q * (Real.exp lam - 1)) := by
    have := Real.add_one_le_exp (q * (Real.exp lam - 1))
    linarith
  have hbase0 : (0:ℝ) ≤ 1 - q + q * Real.exp lam := by
    have := Real.exp_pos lam
    nlinarith
  have hpow : (1 - q + q * Real.exp lam) ^ n ≤ Real.exp ((n : ℝ) * (q * (Real.exp lam - 1))) := by
    calc (1 - q + q * Real.exp lam) ^ n ≤ (Real.exp (q * (Real.exp lam - 1))) ^ n :=
          pow_le_pow_left₀ hbase0 hbase n
      _ = Real.exp ((n : ℝ) * (q * (Real.exp lam - 1))) := by
          rw [← Real.exp_nat_mul]
  have hexp : Real.exp lam - 1 ≤ lam + (3 / 4) * lam ^ 2 := by
    have := exp_quad_bound hlam0 hlam1
    linarith
  calc Real.exp (-lam * ((n : ℝ) * q + a)) * (1 - q + q * Real.exp lam) ^ n
      ≤ Real.exp (-lam * ((n : ℝ) * q + a)) * Real.exp ((n : ℝ) * (q * (Real.exp lam - 1))) := by
        exact mul_le_mul_of_nonneg_left hpow (Real.exp_pos _).le
    _ = Real.exp (-lam * ((n : ℝ) * q + a) + (n : ℝ) * (q * (Real.exp lam - 1))) := by
        rw [← Real.exp_add]
    _ ≤ Real.exp (-(a ^ 2) / (4 * n)) := by
        refine Real.exp_le_exp.mpr ?_
        have hq : (n : ℝ) * (q * (Real.exp lam - 1)) ≤ (n : ℝ) * (q * (lam + (3 / 4) * lam ^ 2)) := by
          have hnq : (0:ℝ) ≤ (n : ℝ) * q := by positivity
          nlinarith
        have hlamn : lam * (n : ℝ) = a := by
          rw [hlam]; field_simp
        have hkey : -lam * ((n : ℝ) * q + a) + (n : ℝ) * (q * (lam + (3 / 4) * lam ^ 2))
            = -(a * lam) + (3 / 4) * q * ((n : ℝ) * lam ^ 2) := by
          ring
        have hlamsq : (n : ℝ) * lam ^ 2 = a ^ 2 / n := by
          rw [hlam]; field_simp
        have halam : a * lam = a ^ 2 / n := by
          rw [hlam]; field_simp
        have : -(a * lam) + (3 / 4) * q * ((n : ℝ) * lam ^ 2) ≤ -(a ^ 2) / (4 * n) := by
          rw [hlamsq, halam]
          have hq1 : (3 / 4) * q ≤ 3 / 4 := by linarith
          have hnn : (0:ℝ) ≤ a ^ 2 / n := by positivity
          have : (3 / 4) * q * (a ^ 2 / n) ≤ (3 / 4) * (a ^ 2 / n) := by
            exact mul_le_mul_of_nonneg_right hq1 hnn
          have hfour : -(a ^ 2) / (4 * n) = -(1 / 4) * (a ^ 2 / n) := by
            field_simp
          rw [hfour]
          linarith
        linarith [hq, this]

/-! ## 4. The exponential-tail p-value -/

/-- The one-sided p-value from the Chernoff bound: `exp(-(count - n·q₀)²/4n)`, truncated at one,
and one when the count is at or below the baseline mean. -/
noncomputable def expP (n : ℕ) (q₀ : ℝ) (s : Fin n → Bool) : ℝ :=
  if cnt s ≤ n * q₀ then 1 else min 1 (Real.exp (-((cnt s - n * q₀) ^ 2) / (4 * n)))

lemma expP_le_one (n : ℕ) (q₀ : ℝ) (s : Fin n → Bool) : expP n q₀ s ≤ 1 := by
  unfold expP
  split
  · exact le_rfl
  · exact min_le_left _ _

lemma expP_nonneg (n : ℕ) (q₀ : ℝ) (s : Fin n → Bool) : 0 ≤ expP n q₀ s := by
  unfold expP
  split
  · norm_num
  · exact le_min (by norm_num) (Real.exp_pos _).le

lemma expP_superuniform_pos {q₀ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (n : ℕ) {t : ℝ}
    (ht : 0 < t) (ht1 : t < 1) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t), recProb q₀ s ≤ t := by
  rcases Nat.eq_zero_or_pos n with hn | hn
  · -- no molecules: nothing can be rejected
    subst hn
    have hempty : Finset.univ.filter (fun s : Fin 0 → Bool => expP 0 q₀ s ≤ t) = ∅ := by
      refine Finset.filter_eq_empty_iff.mpr ?_
      intro s _
      have hc : cnt s ≤ (0 : ℕ) * q₀ := by
        have := cnt_le 0 s
        simpa using this
      unfold expP
      rw [if_pos hc]
      linarith
    rw [hempty, Finset.sum_empty]
    exact ht.le
  have hnpos : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  set L : ℝ := Real.log (1 / t) with hL
  have hLpos : 0 < L := by
    rw [hL, one_div, Real.log_inv]
    have := Real.log_neg ht ht1
    linarith
  -- on the rejection set the count exceeds the mean and the exponent is at least `L`
  have hmem : ∀ s : Fin n → Bool, expP n q₀ s ≤ t →
      (n : ℝ) * q₀ < cnt s ∧ 4 * n * L ≤ (cnt s - n * q₀) ^ 2 := by
    intro s hs
    unfold expP at hs
    by_cases hc : cnt s ≤ (n : ℝ) * q₀
    · rw [if_pos hc] at hs; linarith
    · rw [if_neg hc] at hs
      push_neg at hc
      refine ⟨hc, ?_⟩
      have hmin : Real.exp (-((cnt s - n * q₀) ^ 2) / (4 * n)) ≤ t := by
        rcases min_cases (1 : ℝ) (Real.exp (-((cnt s - n * q₀) ^ 2) / (4 * n))) with
          ⟨he, _⟩ | ⟨he, _⟩
        · rw [he] at hs; linarith
        · rw [he] at hs; exact hs
      have hlog : -((cnt s - (n : ℝ) * q₀) ^ 2) / (4 * n) ≤ Real.log t := by
        have := Real.log_le_log (Real.exp_pos _) hmin
        rwa [Real.log_exp] at this
      have hlt : Real.log t = -L := by rw [hL, one_div, Real.log_inv, neg_neg]
      rw [hlt] at hlog
      rw [div_le_iff₀ (by positivity)] at hlog
      nlinarith
  set a : ℝ := Real.sqrt (4 * n * L) with ha
  have ha0 : 0 ≤ a := Real.sqrt_nonneg _
  have hasq : a ^ 2 = 4 * n * L := Real.sq_sqrt (by positivity)
  rcases le_or_gt a (n : ℝ) with han | han
  · have hsub : Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t)
        ⊆ Finset.univ.filter (fun s : Fin n → Bool => (n : ℝ) * q₀ + a ≤ cnt s) := by
      intro s hsm
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm ⊢
      obtain ⟨hgt, hsq⟩ := hmem s hsm
      have hd : 0 ≤ cnt s - (n : ℝ) * q₀ := by linarith
      have : a ≤ cnt s - (n : ℝ) * q₀ := by nlinarith
      linarith
    calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t), recProb q₀ s
        ≤ ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => (n : ℝ) * q₀ + a ≤ cnt s),
            recProb q₀ s := by
          refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
          intro s _ _
          exact recProb_nonneg h0 h1 s
      _ ≤ Real.exp (-(a ^ 2) / (4 * n)) := chernoff_tail h0 h1 hn ha0 han
      _ = t := by
          rw [hasq]
          have : -(4 * (n : ℝ) * L) / (4 * n) = -L := by field_simp
          rw [this, hL, one_div, Real.log_inv, neg_neg, Real.exp_log ht]
  · -- the demanded deviation exceeds the number of molecules: nothing can be rejected
    have hempty : Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t) = ∅ := by
      refine Finset.filter_eq_empty_iff.mpr ?_
      intro s _ hs
      obtain ⟨hgt, hsq⟩ := hmem s hs
      have hcn : cnt s ≤ (n : ℝ) := cnt_le n s
      have hd : cnt s - (n : ℝ) * q₀ ≤ (n : ℝ) := by nlinarith
      have hd0 : 0 ≤ cnt s - (n : ℝ) * q₀ := by linarith
      have : (cnt s - (n : ℝ) * q₀) ^ 2 ≤ (n : ℝ) ^ 2 := by nlinarith
      have hna : (n : ℝ) ^ 2 < a ^ 2 := by nlinarith
      rw [hasq] at hna
      linarith
    rw [hempty, Finset.sum_empty]
    exact ht.le

/-- **The exponential-tail p-value is valid.**  Under the baseline read-out law, `P(p ≤ t) ≤ t` at
every level — again with no distributional approximation, only the exact moment generating
function and Markov's inequality. -/
theorem expP_superuniform {q₀ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (n : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t), recProb q₀ s ≤ t := by
  rcases le_or_gt 1 t with hge | hlt
  · calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t), recProb q₀ s
        ≤ ∑ s : Fin n → Bool, recProb q₀ s := by
          refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) ?_
          intro s _ _
          exact recProb_nonneg h0 h1 s
      _ = 1 := sum_recProb q₀ n
      _ ≤ t := hge
  · rcases eq_or_lt_of_le ht with ht0 | htpos
    · subst_vars
      by_contra hcon
      push_neg at hcon
      set M : ℝ := ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ 0),
        recProb q₀ s with hM
      set t' : ℝ := min (M / 2) (1 / 2) with ht'
      have ht'pos : 0 < t' := lt_min (by linarith) (by norm_num)
      have ht'lt : t' < 1 := lt_of_le_of_lt (min_le_right _ _) (by norm_num)
      have hsub : Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ 0)
          ⊆ Finset.univ.filter (fun s : Fin n → Bool => expP n q₀ s ≤ t') := by
        intro s hsm
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm ⊢
        linarith
      have hle : M ≤ t' := by
        refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg hsub ?_) ?_
        · intro s _ _
          exact recProb_nonneg h0 h1 s
        · exact expP_superuniform_pos h0 h1 n ht'pos ht'lt
      have : t' ≤ M / 2 := min_le_left _ _
      linarith
    · exact expP_superuniform_pos h0 h1 n htpos hlt

/-! ## 5. Power, and the molecules the exponential design needs -/

/-- Molecules per candidate for the exponential-tail p-value: `(16·log(1/u) + 1/α)/Δ²`.  The
dependence on the demanded p-value `u` is *logarithmic*, where the two-moment design of
`RequestProject.ScreenBudget` pays `1/u`. -/
noncomputable def expSamples (u alpha delta : ℝ) : ℕ :=
  ⌈(16 * Real.log (1 / u) + 1 / alpha) / delta ^ 2⌉₊

lemma expSamples_spec {u alpha delta : ℝ} (hu : 0 < u) (hu1 : u ≤ 1) (ha : 0 < alpha)
    (hdelta : 0 < delta) {n : ℕ} (hn : expSamples u alpha delta ≤ n) :
    16 * Real.log (1 / u) ≤ (n : ℝ) * delta ^ 2 ∧ 1 ≤ alpha * n * delta ^ 2 := by
  have hLnn : 0 ≤ Real.log (1 / u) := by
    rw [one_div]
    exact Real.log_nonneg (by rw [le_inv_comm₀ (by norm_num) hu]; simpa using hu1)
  have hceil : (16 * Real.log (1 / u) + 1 / alpha) / delta ^ 2
      ≤ (expSamples u alpha delta : ℝ) := Nat.le_ceil _
  have hnle : (expSamples u alpha delta : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hkey : 16 * Real.log (1 / u) + 1 / alpha ≤ (n : ℝ) * delta ^ 2 := by
    have h1 : (16 * Real.log (1 / u) + 1 / alpha) / delta ^ 2 ≤ (n : ℝ) := le_trans hceil hnle
    rw [div_le_iff₀ (by positivity)] at h1
    linarith
  have hinv : 0 < 1 / alpha := by positivity
  refine ⟨by linarith, ?_⟩
  have : 1 / alpha ≤ (n : ℝ) * delta ^ 2 := by linarith
  rw [div_le_iff₀ ha] at this
  nlinarith

/-- **The exponential-tail p-value has power.**  A candidate whose reporter contrast is `Δ`
returns a p-value above `u` with probability at most `α`, once the molecule count satisfies
`16·log(1/u) ≤ n·Δ²` and `1 ≤ α·n·Δ²`. -/
theorem expP_power {q₀ q₁ u alpha : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (hlt : q₀ < q₁)
    (hu : 0 < u) {n : ℕ} (hn : 0 < n)
    (hL : 16 * Real.log (1 / u) ≤ (n : ℝ) * (q₁ - q₀) ^ 2)
    (hna : 1 ≤ alpha * n * (q₁ - q₀) ^ 2) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => ¬ (expP n q₀ s ≤ u)), recProb q₁ s
      ≤ alpha := by
  set d : ℝ := q₁ - q₀ with hd
  have hdpos : 0 < d := by rw [hd]; linarith
  have hnpos : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  set r : ℝ := (n : ℝ) * d / 2 with hr
  have hrpos : 0 < r := by rw [hr]; positivity
  have hclose : ∀ s : Fin n → Bool, |cnt s - (n : ℝ) * q₁| < r → expP n q₀ s ≤ u := by
    intro s hs
    have habs := abs_lt.mp hs
    have hgap : cnt s - (n : ℝ) * q₀ > r := by
      have hsplit : cnt s - (n : ℝ) * q₀ = (cnt s - (n : ℝ) * q₁) + (n : ℝ) * d := by
        rw [hd]; ring
      rw [hsplit, hr]
      have h := habs.1
      rw [hr] at h
      linarith
    have hgt : ¬ (cnt s ≤ (n : ℝ) * q₀) := by intro hcon; linarith
    unfold expP
    rw [if_neg hgt]
    refine le_trans (min_le_right _ _) ?_
    have hsq : r ^ 2 ≤ (cnt s - (n : ℝ) * q₀) ^ 2 := by nlinarith
    have hlog : -((cnt s - (n : ℝ) * q₀) ^ 2) / (4 * n) ≤ Real.log u := by
      rw [div_le_iff₀ (by positivity)]
      have hlogu : Real.log (1 / u) = -Real.log u := by rw [one_div, Real.log_inv]
      rw [hlogu] at hL
      have hr2 : r ^ 2 = (n : ℝ) ^ 2 * d ^ 2 / 4 := by rw [hr]; ring
      nlinarith
    calc Real.exp (-((cnt s - (n : ℝ) * q₀) ^ 2) / (4 * n))
        ≤ Real.exp (Real.log u) := Real.exp_le_exp.mpr hlog
      _ = u := Real.exp_log hu
  have hsub : Finset.univ.filter (fun s : Fin n → Bool => ¬ (expP n q₀ s ≤ u))
      ⊆ devSet n ((n : ℝ) * q₁) r := by
    intro s hsm
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm
    rw [mem_devSet]
    by_contra hcon
    push_neg at hcon
    exact hsm (hclose s hcon)
  calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => ¬ (expP n q₀ s ≤ u)), recProb q₁ s
      ≤ ∑ s ∈ devSet n ((n : ℝ) * q₁) r, recProb q₁ s := by
        refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
        intro s _ _
        exact recProb_nonneg h10 h11 s
    _ ≤ (n : ℝ) * q₁ * (1 - q₁) / r ^ 2 := chebyshev h10 h11 n hrpos
    _ ≤ alpha := by
        have hq4 : q₁ * (1 - q₁) ≤ 1 / 4 := by nlinarith [sq_nonneg (q₁ - 1 / 2)]
        have hv : (n : ℝ) * q₁ * (1 - q₁) ≤ (n : ℝ) / 4 := by nlinarith [hnpos.le]
        have hr2 : r ^ 2 = (n : ℝ) ^ 2 * d ^ 2 / 4 := by rw [hr]; ring
        rw [div_le_iff₀ (by positivity), hr2]
        nlinarith

/-! ## 6. The screen, and its cost -/

/-- **The screen with exponential-tail p-values also controls the false discovery rate.** -/
theorem screen_fdr_control_exp {N : ℕ} (n : Fin N → ℕ) (q0 rate : Fin N → ℝ)
    (h0 : ∀ i, 0 ≤ q0 i) (h1 : ∀ i, q0 i ≤ 1)
    (hr0 : ∀ i, 0 ≤ rate i) (hr1 : ∀ i, rate i ≤ 1)
    {q : ℝ} (hq : 0 ≤ q) (H₀ : Finset (Fin N)) (hnull : ∀ i ∈ H₀, rate i = q0 i) :
    EE (V := fun i => (Fin (n i) → Bool)) (fun i s => recProb (rate i) s)
        (fun ω => fdp N q (pvec (fun i s => expP (n i) (q0 i) s) ω) H₀) ≤ q := by
  refine bh_fdr_le (V := fun i => (Fin (n i) → Bool)) hq
    (fun i s => recProb_nonneg (hr0 i) (hr1 i) s)
    (fun i => sum_recProb (rate i) (n i))
    (fun i s => expP_nonneg (n i) (q0 i) s) H₀ ?_
  intro i hi t ht
  have := expP_superuniform (h0 i) (h1 i) (n i) (t := t) ht
  rw [hnull i hi]
  exact this

/-- At the worst-case BH threshold `q/N`, the exponential design costs
`⌈(16·log(N/q) + 1/α)/Δ²⌉` molecules per candidate: logarithmic in the size of the screen. -/
lemma expSamples_screen (N : ℕ) (q alpha delta : ℝ) :
    expSamples (q / N) alpha delta
      = ⌈(16 * Real.log ((N : ℝ) / q) + 1 / alpha) / delta ^ 2⌉₊ := by
  unfold expSamples
  congr 2
  rw [one_div_div]

/-- **The separation.**  Whenever the logarithmic cost per candidate is below the two-moment cost,
the whole exponential-tail screen is cheaper than the two-moment screen — and since the former
grows like `log N` and the latter like `N`, this holds for every screen large enough. -/
theorem screen_exp_cheaper {N : ℕ} (hN : 0 < N) {q alpha delta : ℝ} (hq : 0 < q) (hq1 : q ≤ 1)
    (halpha : 0 < alpha) (hdelta : 0 < delta) (hle : q / N ≤ alpha)
    (hcmp : (16 * Real.log ((N : ℝ) / q) + 1 / alpha) / delta ^ 2 + 1
      ≤ (N : ℝ) / (q * delta ^ 2)) :
    (N : ℝ) * (expSamples (q / N) alpha delta : ℝ)
      ≤ (N : ℝ) * (screenSamples (q / N) alpha delta : ℝ) := by
  have hNpos : (0:ℝ) < (N : ℝ) := by exact_mod_cast hN
  have hexp : (expSamples (q / N) alpha delta : ℝ)
      ≤ (16 * Real.log ((N : ℝ) / q) + 1 / alpha) / delta ^ 2 + 1 := by
    rw [expSamples_screen N q alpha delta]
    have hNq : (1:ℝ) ≤ (N : ℝ) / q := by
      rw [le_div_iff₀ hq]
      have : (1:ℝ) ≤ (N : ℝ) := by exact_mod_cast hN
      linarith
    have hlog : 0 ≤ Real.log ((N : ℝ) / q) := Real.log_nonneg hNq
    have hnn : 0 ≤ (16 * Real.log ((N : ℝ) / q) + 1 / alpha) / delta ^ 2 := by positivity
    exact (Nat.ceil_lt_add_one hnn).le
  have hcheb : (N : ℝ) / (q * delta ^ 2) ≤ (screenSamples (q / N) alpha delta : ℝ) := by
    have hmin : min (q / N) alpha = q / N := min_eq_left hle
    have hceil : (1 : ℝ) / (min (q / N) alpha * delta ^ 2)
        ≤ (screenSamples (q / N) alpha delta : ℝ) := Nat.le_ceil _
    rw [hmin] at hceil
    refine le_trans (le_of_eq ?_) hceil
    field_simp
  exact mul_le_mul_of_nonneg_left (le_trans (le_trans hexp hcmp) hcheb) hNpos.le

end Chernoff
end IDR
