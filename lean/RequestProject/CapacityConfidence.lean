/-
# Capacity under a *measured* annotation error rate

`RequestProject.BenchmarkCapacityLabels` prices a benchmark against a *worst-case* noise budget:
feed it a rate `ε` and it returns the number of methods that can be certifiably ordered.  A paper
does not have a worst-case budget — it has a rate measured on a finite sample of structures, and
wants to apply the resulting number to a leaderboard.  The join between the two is the statistical
half, and this file supplies it.

The statistical statement is a one-sided test, which is the honest form: the measurement cannot put
an upper bound on the noise (a run of luck can always make good labels look bad), but it can put a
*lower* bound, and a lower bound on the noise is exactly what shrinks the capacity.

* `tail_le_of_rate_le` — **the uniform tail bound.**  For any true rate `q ≤ q₀` and any
  `λ ∈ [0, 1]`, the chance that the count of mislabelled reads reaches `c` is at most
  `exp(−λc + n q₀ (e^λ − 1))`.  It is uniform over the null region `q ≤ q₀`, which is what makes it
  a valid test, and it is sharper than the sub-Gaussian bound `exp(−a²/4n)` of
  `RequestProject.ChernoffScreen` in the small-rate regime the application lives in.
* `capacity_or_rare_event` — **the join.**  Either the true rate is at least `q₀`, in which case the
  capacity bound at `q₀` applies to the leaderboard verbatim; or the true rate is below `q₀`, in
  which case the observed count was a rare event, with probability at most the tail bound.
* `rate_test_2746`, `capacity_at_95_confidence` — **the numbers.**  With `2746` measured structure
  pairs, a count of at least `160` disagreements has probability at most `1/20` under *any* true
  rate below `4%`.  So at 95% confidence the rate is at least `4%`, and a benchmark at that rate
  certifiably orders at most `13` methods, however many are entered.

The three together are what lets a paper write "at 95% confidence, this benchmark can order at most
`K` methods" from a rate measured on its own calibration data.
-/
import Mathlib
import RequestProject.ChernoffScreen
import RequestProject.BenchmarkCapacityLabels

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace CapacityConfidence

open Finset
open scoped Classical
open IDR.Noisy IDR.Chernoff IDR.BenchCapacity IDR.LabelNoise

/-! ## 1. A tail bound uniform over the null region -/

/-- **The tail bound of the test.**  Under any true rate `q ≤ q₀`, the chance that the count of
positive reads reaches `c` is at most `exp(−λc + n q₀ (e^λ − 1))`, for every `λ ≥ 0`.  Uniformity in
`q` over the null region is what makes this a test rather than a statement about a known rate. -/
theorem tail_le_of_rate_le {q q0 lam c : ℝ} (h0 : 0 ≤ q) (hq0 : q ≤ q0) (hq01 : q0 ≤ 1)
    (hlam : 0 ≤ lam) (n : ℕ) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s), recProb q s
      ≤ Real.exp (-lam * c + n * q0 * (Real.exp lam - 1)) := by
  have h1 : q ≤ 1 := le_trans hq0 hq01
  have hstep := markov_tail (q := q) (lam := lam) (c := c) h0 h1 hlam n
  have hexp1 : (0:ℝ) ≤ Real.exp lam - 1 := by
    have := Real.one_le_exp hlam
    linarith
  have hbase : 1 - q + q * Real.exp lam ≤ Real.exp (q * (Real.exp lam - 1)) := by
    have := Real.add_one_le_exp (q * (Real.exp lam - 1))
    nlinarith
  have hbase0 : (0:ℝ) ≤ 1 - q + q * Real.exp lam := by nlinarith
  have hpow : (1 - q + q * Real.exp lam) ^ n ≤ Real.exp ((n : ℝ) * q0 * (Real.exp lam - 1)) := by
    calc (1 - q + q * Real.exp lam) ^ n ≤ (Real.exp (q * (Real.exp lam - 1))) ^ n :=
          pow_le_pow_left₀ hbase0 hbase n
      _ = Real.exp ((n : ℝ) * (q * (Real.exp lam - 1))) := by rw [← Real.exp_nat_mul]
      _ ≤ Real.exp ((n : ℝ) * q0 * (Real.exp lam - 1)) := by
          refine Real.exp_le_exp.mpr ?_
          have hn : (0:ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
          have hstep2 : q * (Real.exp lam - 1) ≤ q0 * (Real.exp lam - 1) :=
            mul_le_mul_of_nonneg_right hq0 hexp1
          calc (n : ℝ) * (q * (Real.exp lam - 1)) ≤ (n : ℝ) * (q0 * (Real.exp lam - 1)) :=
                mul_le_mul_of_nonneg_left hstep2 hn
            _ = (n : ℝ) * q0 * (Real.exp lam - 1) := by ring
  calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s), recProb q s
      ≤ Real.exp (-lam * c) * (1 - q + q * Real.exp lam) ^ n := hstep
    _ ≤ Real.exp (-lam * c) * Real.exp ((n : ℝ) * q0 * (Real.exp lam - 1)) :=
        mul_le_mul_of_nonneg_left hpow (Real.exp_pos _).le
    _ = Real.exp (-lam * c + (n : ℝ) * q0 * (Real.exp lam - 1)) := by rw [← Real.exp_add]

/-! ## 2. The join with the capacity theorem -/

variable {α : Type*} [Fintype α] [DecidableEq α]

/-- **Either the capacity bound holds, or the measurement was a rare event.**  This is the join the
paper needs: the capacity theorem takes a rate, the experiment provides a test of a rate, and the
disjunction below is what they prove together.  Nothing is assumed about the true rate except that
it is a rate. -/
theorem capacity_or_rare_event {iota : Type*} {T : Finset α} {pred : iota → Finset α}
    {M : Finset iota} {nu : ℕ} {q q0 lam c : ℝ} {n : ℕ}
    (hR : 0 < Fintype.card α) (h0 : 0 ≤ q) (hq00 : 0 < q0) (hq01 : q0 ≤ 1) (hlam : 0 ≤ lam)
    (hbudget : q * (Fintype.card α : ℝ) ≤ nu) (hcert : Certified T nu pred M) :
    (q0 ≤ q → M.card ≤ benchCapacity (Fintype.card α) q0 0) ∧
      (q ≤ q0 → ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s), recProb q s
        ≤ Real.exp (-lam * c + n * q0 * (Real.exp lam - 1))) := by
  constructor
  · intro hq
    refine card_le_benchCapacity_of_rate hR hq00.le ?_ hcert
    have hRR : (0:ℝ) ≤ (Fintype.card α : ℝ) := Nat.cast_nonneg _
    nlinarith
  · intro hq
    exact tail_le_of_rate_le h0 hq hq01 hlam n

/-! ## 3. The numbers -/

private lemma exp_half_le : Real.exp (1 / 2) ≤ 165 / 100 := by
  have hpos : (0:ℝ) < Real.exp (1 / 2) := Real.exp_pos _
  have hsq : Real.exp (1 / 2) ^ 2 = Real.exp 1 := by
    rw [← Real.exp_nat_mul]
    norm_num
  have he : Real.exp 1 < 2.7182818286 := Real.exp_one_lt_d9
  nlinarith

private lemma twenty_le_exp_four : (20:ℝ) ≤ Real.exp 4 := by
  have he : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  have h4 : Real.exp 4 = (Real.exp 1) ^ (4 : ℕ) := by
    rw [← Real.exp_nat_mul]; norm_num
  rw [h4]
  nlinarith [pow_le_pow_left₀ (by norm_num : (0:ℝ) ≤ 2.7182818283) he.le 4]

/-- **The test, on the measured sample.**  With `2746` structure pairs read, seeing at least `160`
disagreements has probability at most `1/20` under *every* true error rate of at most `4%`.  So a
measurement of that size rejects "the annotation is at most 4% wrong" at the 95% level. -/
theorem rate_test_2746 {q : ℝ} (h0 : 0 ≤ q) (hq : q ≤ 4 / 100) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin 2746 → Bool => 160 ≤ cnt s), recProb q s ≤ 1 / 20 := by
  have hbound := tail_le_of_rate_le (q := q) (q0 := 4 / 100) (lam := 1 / 2) (c := 160)
    h0 hq (by norm_num) (by norm_num) 2746
  refine hbound.trans ?_
  have hexp : Real.exp (1 / 2) - 1 ≤ 65 / 100 := by
    have := exp_half_le
    linarith
  have harg : -(1 / 2 : ℝ) * 160 + (2746 : ℕ) * (4 / 100) * (Real.exp (1 / 2) - 1) ≤ -4 := by
    have hpos : (0:ℝ) ≤ (2746 : ℕ) * (4 / 100 : ℝ) := by positivity
    have : ((2746 : ℕ) : ℝ) * (4 / 100) * (Real.exp (1 / 2) - 1)
        ≤ ((2746 : ℕ) : ℝ) * (4 / 100) * (65 / 100) := by
      refine mul_le_mul_of_nonneg_left hexp ?_
      positivity
    push_cast at this ⊢
    linarith
  calc Real.exp (-(1 / 2 : ℝ) * 160 + (2746 : ℕ) * (4 / 100) * (Real.exp (1 / 2) - 1))
      ≤ Real.exp (-4) := Real.exp_le_exp.mpr harg
    _ ≤ 1 / 20 := by
        rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 4) (by norm_num)]
        simpa using twenty_le_exp_four

omit [DecidableEq α] in
/-- The capacity at a `4%` error rate: at most `13` methods. -/
theorem capacity_at_four_percent (hR : 0 < Fintype.card α) :
    benchCapacity (Fintype.card α) (4 / 100) 0 ≤ 13 := by
  refine (benchCapacity_noise_only (Fintype.card α) hR (by norm_num)).trans ?_
  have hceil : ⌈(1 : ℝ) / (2 * (4 / 100))⌉₊ = 13 := by
    rw [show (1 : ℝ) / (2 * (4 / 100)) = 25 / 2 by norm_num, Nat.ceil_eq_iff (by norm_num)]
    norm_num
  rw [hceil]
  norm_num

/-- **The statement the paper can make.**  Read `2746` structure pairs against the annotation.  Then
either the leaderboard's certifiably ordered family has at most `13` methods — however many are
entered — or the measurement produced at least `160` disagreements while the true error rate was
below `4%`, an event of probability at most `1/20`.  In short: at 95% confidence, at most `13`
methods can be ordered. -/
theorem capacity_at_95_confidence {iota : Type*} {T : Finset α} {pred : iota → Finset α}
    {M : Finset iota} {nu : ℕ} {q : ℝ}
    (hR : 0 < Fintype.card α) (h0 : 0 ≤ q)
    (hbudget : q * (Fintype.card α : ℝ) ≤ nu) (hcert : Certified T nu pred M) :
    (4 / 100 ≤ q → M.card ≤ 13) ∧
      (q ≤ 4 / 100 →
        ∑ s ∈ Finset.univ.filter (fun s : Fin 2746 → Bool => 160 ≤ cnt s), recProb q s
          ≤ 1 / 20) := by
  refine ⟨fun hq => ?_, fun hq => rate_test_2746 h0 hq⟩
  have h := (capacity_or_rare_event (n := 2746) (lam := 1 / 2) (c := 160) (q0 := 4 / 100)
    hR h0 (by norm_num) (by norm_num) (by norm_num) hbudget hcert).1 hq
  exact h.trans (capacity_at_four_percent hR)

end CapacityConfidence
end IDR
