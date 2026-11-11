/-
# Part XCII.2  The other half of the sequential test: does it ever stop?

`RequestProject.Sequential` proves that watching a run and stopping it when the likelihood ratio
crosses `1/α` cannot inflate the false-refutation rate: that is the type-I half, and it holds at
every horizon, for every stopping rule.  A test that never stops has type-I error zero and is
worthless, so the design is only finished when the *other* half is proved: when the model really
does omit population, the run stops, and it stops after a stated number of molecules.

This file proves that, again with no asymptotics and no martingale library.

* `m2`, `klVar` — the second moment and the variance of the log likelihood ratio of one read.
* `expected_logWealth_sq`, `variance_logWealth` — the exact second moment of the accumulated log
  wealth after `n` molecules, hence its variance: exactly `n · klVar`.  Evidence has mean `n·KL`
  and standard deviation `√(n·klVar)`, so the signal-to-noise of the run grows like `√n`.
* `cheb_words` — Chebyshev's inequality on the read-out law.
* `sequential_power` — **the run stops.** Under the alternative, the probability that the wealth
  fails to reach `c` within `n` molecules is at most `n·klVar / (n·KL − log c)²`, at every finite
  `n`: an explicit, non-asymptotic type-II bound.
* `powerSamples`, `powerSamples_spec` — hence an explicit molecule count: `2·log(1/α)/KL`
  molecules to overcome the threshold and `4·klVar/(β·KL²)` to overcome the fluctuations, whose
  maximum delivers power `1 − β` at level `α`.

Together with `IDR.Seq.anytime_valid` this is a complete design: a stopping rule, a proved level,
a proved power, and a molecule count — for an experiment whose length is *not* fixed in advance.
-/
import Mathlib
import RequestProject.Sequential
import RequestProject.Pinsker

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace Seq

open Finset

/-! ## 1. The second moment of the evidence -/

/-- The second moment of the log likelihood ratio of a single read, under the alternative. -/
noncomputable def m2 (q₁ q₀ : ℝ) : ℝ :=
  q₁ * Real.log (q₁ / q₀) ^ 2 + (1 - q₁) * Real.log ((1 - q₁) / (1 - q₀)) ^ 2

/-- The variance of the log likelihood ratio of a single read, under the alternative. -/
noncomputable def klVar (q₁ q₀ : ℝ) : ℝ := m2 q₁ q₀ - kl2 q₁ q₀ ^ 2

/-- The exact second moment of the accumulated log wealth after `n` molecules. -/
theorem expected_logWealth_sq (q₁ q₀ : ℝ) (n : ℕ) :
    ∑ l ∈ words n, probL q₁ l * logWealth q₁ q₀ l ^ 2
      = n * m2 q₁ q₀ + n * ((n : ℝ) - 1) * kl2 q₁ q₀ ^ 2 := by
  induction n with
  | zero => simp [words]
  | succ n ih =>
      rw [sum_words_succ]
      have e : ∀ b : Bool, ∀ l ∈ words n,
          probL q₁ (b :: l) * logWealth q₁ q₀ (b :: l) ^ 2
            = Noisy.bern q₁ b * Real.log (lrOne q₁ q₀ b) ^ 2 * probL q₁ l
              + 2 * (Noisy.bern q₁ b * Real.log (lrOne q₁ q₀ b))
                  * (probL q₁ l * logWealth q₁ q₀ l)
              + Noisy.bern q₁ b * (probL q₁ l * logWealth q₁ q₀ l ^ 2) := by
        intro b l _
        simp only [probL_cons, logWealth_cons]
        ring
      rw [Finset.sum_congr rfl (e true), Finset.sum_congr rfl (e false)]
      simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [sum_probL, expected_logWealth, ih]
      have hbt : Noisy.bern q₁ true = q₁ := by simp [Noisy.bern]
      have hbf : Noisy.bern q₁ false = 1 - q₁ := by simp [Noisy.bern]
      rw [hbt, hbf, lrOne_true, lrOne_false]
      unfold m2 kl2
      push_cast
      ring

/-- The variance of the accumulated log wealth is exactly `n` times the single-read variance. -/
theorem variance_logWealth (q₁ q₀ : ℝ) (n : ℕ) :
    ∑ l ∈ words n, probL q₁ l * (logWealth q₁ q₀ l - n * kl2 q₁ q₀) ^ 2
      = n * klVar q₁ q₀ := by
  have expand : ∀ l ∈ words n,
      probL q₁ l * (logWealth q₁ q₀ l - n * kl2 q₁ q₀) ^ 2
        = probL q₁ l * logWealth q₁ q₀ l ^ 2
          + (-2 * ((n : ℝ) * kl2 q₁ q₀)) * (probL q₁ l * logWealth q₁ q₀ l)
          + ((n : ℝ) * kl2 q₁ q₀) ^ 2 * probL q₁ l := by
    intro l _; ring
  rw [Finset.sum_congr rfl expand]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [expected_logWealth_sq, expected_logWealth, sum_probL]
  unfold klVar
  ring

/-- The evidence variance in closed form: the variance of a two-valued log likelihood ratio. -/
lemma klVar_eq (q₁ q₀ : ℝ) :
    klVar q₁ q₀
      = q₁ * (1 - q₁) * (Real.log (q₁ / q₀) - Real.log ((1 - q₁) / (1 - q₀))) ^ 2 := by
  unfold klVar m2 kl2
  ring

/-- Every molecule carries a genuinely fluctuating amount of evidence: the variance is strictly
positive whenever the alternative read-out rate differs from the null one. -/
lemma klVar_pos {q₁ q₀ : ℝ} (h10 : 0 < q₁) (h11 : q₁ < 1) (h00 : 0 < q₀) (h01 : q₀ < 1)
    (hne : q₁ ≠ q₀) : 0 < klVar q₁ q₀ := by
  have hx : (0:ℝ) < q₁ / q₀ := by positivity
  have hy : (0:ℝ) < (1 - q₁) / (1 - q₀) := by
    have h1 : (0:ℝ) < 1 - q₁ := by linarith
    have h2 : (0:ℝ) < 1 - q₀ := by linarith
    positivity
  have hne' : q₁ / q₀ ≠ (1 - q₁) / (1 - q₀) := by
    intro h
    apply hne
    have h1 : (1:ℝ) - q₀ ≠ 0 := by intro hh; linarith
    field_simp at h
    linarith
  have hlog : Real.log (q₁ / q₀) ≠ Real.log ((1 - q₁) / (1 - q₀)) := by
    intro h
    exact hne' (by
      have := congrArg Real.exp h
      rwa [Real.exp_log hx, Real.exp_log hy] at this)
  rw [klVar_eq]
  have hsq : 0 < (Real.log (q₁ / q₀) - Real.log ((1 - q₁) / (1 - q₀))) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (sub_ne_zero.mpr hlog)))
  have h1q : (0:ℝ) < 1 - q₁ := by linarith
  positivity

/-- **Pinsker for the reporter.**  The evidence rate is at least twice the squared rate
contrast, so the sequential horizon is at most `log(1/α)/(2Δ²)` — the same `Δ²` the fixed-`n`
design of Part XC pays, but multiplied by `log(1/α)` instead of `1/α`. -/
lemma kl2_ge_two_sq {q₁ q₀ : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < q₀) (h01 : q₀ < 1) :
    2 * (q₁ - q₀) ^ 2 ≤ kl2 q₁ q₀ := by
  have hp : ∀ b : Bool, 0 ≤ Noisy.bern q₁ b := fun b => Noisy.bern_nonneg h10 h11 b
  have hq : ∀ b : Bool, 0 < Noisy.bern q₀ b := by
    intro b
    cases b
    · show (0:ℝ) < 1 - q₀
      linarith
    · exact h00
  have hps : ∑ b : Bool, Noisy.bern q₁ b = 1 := by
    simp [Noisy.bern]
  have hqs : ∑ b : Bool, Noisy.bern q₀ b = 1 := by
    simp [Noisy.bern]
  have hpin := Pinsker.pinskerG (ι := Bool) hp hq hps hqs
  have habs : ∑ b : Bool, |Noisy.bern q₁ b - Noisy.bern q₀ b| = 2 * |q₁ - q₀| := by
    simp [Noisy.bern]
    rw [abs_sub_comm q₀ q₁]
    ring
  have hklG : Pinsker.klG (fun b : Bool => Noisy.bern q₁ b) (fun b : Bool => Noisy.bern q₀ b)
      = kl2 q₁ q₀ := by
    simp [Pinsker.klG, Noisy.bern, kl2]
  rw [habs, hklG] at hpin
  nlinarith [sq_abs (q₁ - q₀), hpin]

/-- The sequential horizon expressed in the reporter's own currency: at most
`⌈log(1/α)/(2Δ²)⌉` molecules, where `Δ` is the rate contrast. -/
theorem sequentialHorizon_le_of_contrast {q₁ q₀ α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1)
    (h00 : 0 < q₀) (h01 : q₀ < 1) (hne : q₁ ≠ q₀) (hα : 0 < α) (hα1 : α ≤ 1) :
    sequentialHorizon α (kl2 q₁ q₀) ≤ ⌈Real.log (1 / α) / (2 * (q₁ - q₀) ^ 2)⌉₊ := by
  have hd : 0 < 2 * (q₁ - q₀) ^ 2 := by
    have : (q₁ - q₀) ≠ 0 := sub_ne_zero.mpr hne
    positivity
  have hkl : 2 * (q₁ - q₀) ^ 2 ≤ kl2 q₁ q₀ := kl2_ge_two_sq h10 h11 h00 h01
  have hL : 0 ≤ Real.log (1 / α) := Real.log_nonneg (by rw [le_div_iff₀ hα]; linarith)
  refine Nat.ceil_le_ceil ?_
  exact div_le_div_of_nonneg_left hL hd hkl

/-! ## 2. Chebyshev's inequality on the read-out law -/

open Classical in
/-- Chebyshev's inequality for the law of a run of `n` reads. -/
lemma cheb_words {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (n : ℕ) (X : List Bool → ℝ) (μ t : ℝ)
    (ht : 0 < t) :
    ∑ l ∈ words n, probL q l * (if t ≤ |X l - μ| then 1 else 0)
      ≤ (∑ l ∈ words n, probL q l * (X l - μ) ^ 2) / t ^ 2 := by
  rw [Finset.sum_div]
  refine Finset.sum_le_sum ?_
  intro l _
  have hp : 0 ≤ probL q l := probL_nonneg h0 h1 l
  by_cases hcase : t ≤ |X l - μ|
  · rw [if_pos hcase]
    have hsq : t ^ 2 ≤ (X l - μ) ^ 2 := by
      nlinarith [sq_abs (X l - μ), abs_nonneg (X l - μ), hcase, ht]
    rw [mul_one, le_div_iff₀ (by positivity)]
    nlinarith [hp]
  · rw [if_neg hcase, mul_zero]
    positivity

/-! ## 3. The power of the sequential test -/

/-- **The sequential test stops.** Under the alternative rate `q₁`, the probability that the
wealth never reaches `c` during a run of `n` molecules is at most `n·klVar/(n·KL − log c)²`. -/
theorem sequential_power {q₁ q₀ c : ℝ} (h10 : 0 < q₁) (h11 : q₁ < 1) (h00 : 0 < q₀)
    (h01 : q₀ < 1) (n : ℕ) (hgap : Real.log c < n * kl2 q₁ q₀) :
    ∑ l ∈ words n, probL q₁ l * (1 - crossInd q₁ q₀ c 1 l)
      ≤ (n * klVar q₁ q₀) / ((n : ℝ) * kl2 q₁ q₀ - Real.log c) ^ 2 := by
  classical
  set μ : ℝ := (n : ℝ) * kl2 q₁ q₀ with hμ
  set t : ℝ := μ - Real.log c with hts
  have ht : 0 < t := by simp only [hts]; linarith
  -- a run that fails to cross has small log wealth
  have hdev : ∀ l ∈ words n,
      probL q₁ l * (1 - crossInd q₁ q₀ c 1 l)
        ≤ probL q₁ l * (if t ≤ |logWealth q₁ q₀ l - μ| then 1 else 0) := by
    intro l _
    have hp : 0 ≤ probL q₁ l := probL_nonneg (le_of_lt h10) (le_of_lt h11) l
    by_cases hcr : Crossed q₁ q₀ c 1 l
    · have : crossInd q₁ q₀ c 1 l = 1 := by unfold crossInd; simp [hcr]
      rw [this]
      simp only [sub_self, mul_zero]
      positivity
    · have hci : crossInd q₁ q₀ c 1 l = 0 := by unfold crossInd; simp [hcr]
      have hwlt : wealth q₁ q₀ 1 l < c := by
        by_contra hcon
        exact hcr (crossed_of_final l 1 (le_of_not_gt hcon))
      have hwpos : 0 < wealth q₁ q₀ 1 l := by
        rw [wealth_eq_ratio h00 h01]
        have h1 : 0 < probL q₁ l := by
          rcases lt_or_eq_of_le (probL_nonneg (le_of_lt h10) (le_of_lt h11) l) with h | h
          · exact h
          · exact absurd h.symm (probL_ne_zero h10 h11 l)
        have h2 : 0 < probL q₀ l := by
          rcases lt_or_eq_of_le (probL_nonneg (le_of_lt h00) (le_of_lt h01) l) with h | h
          · exact h
          · exact absurd h.symm (probL_ne_zero h00 h01 l)
        positivity
      have hlog : logWealth q₁ q₀ l < Real.log c := by
        rw [logWealth_eq_log_wealth h10 h11 h00 h01 l]
        exact Real.log_lt_log hwpos hwlt
      have : t ≤ |logWealth q₁ q₀ l - μ| := by
        have : logWealth q₁ q₀ l - μ ≤ -t := by simp only [hts]; linarith
        have habs : t ≤ -(logWealth q₁ q₀ l - μ) := by linarith
        exact le_trans habs (neg_le_abs _)
      rw [hci, if_pos this]
      simp
  refine le_trans (Finset.sum_le_sum hdev) ?_
  refine le_trans (cheb_words (le_of_lt h10) (le_of_lt h11) n (logWealth q₁ q₀) μ t ht) ?_
  rw [variance_logWealth]

/-- The number of molecules that gives the sequential test power `1 − β` against the alternative
`q₁` at level `α`: enough to clear the threshold `log(1/α)` twice over, and enough to make the
fluctuations of the evidence negligible beside it. -/
noncomputable def powerSamples (α β kl v : ℝ) : ℕ :=
  max ⌈2 * Real.log (1 / α) / kl⌉₊ ⌈4 * v / (β * kl ^ 2)⌉₊

/-- **Level and power together.** At `n = powerSamples α β KL klVar` molecules the sequential
test refutes an under-capacity model with probability at least `1 − β`, while
`IDR.Seq.anytime_valid` bounds its false-refutation rate by `α` at that and every other
horizon. -/
theorem powerSamples_spec {q₁ q₀ α β : ℝ} (h10 : 0 < q₁) (h11 : q₁ < 1) (h00 : 0 < q₀)
    (h01 : q₀ < 1) (hα : 0 < α) (hα1 : α < 1) (hβ : 0 < β)
    (hkl : 0 < kl2 q₁ q₀) (hv : 0 < klVar q₁ q₀)
    {n : ℕ} (hn : powerSamples α β (kl2 q₁ q₀) (klVar q₁ q₀) ≤ n) :
    ∑ l ∈ words n, probL q₁ l * (1 - crossInd q₁ q₀ (1 / α) 1 l) ≤ β := by
  have hL : 0 < Real.log (1 / α) := Real.log_pos (by rw [lt_div_iff₀ hα]; linarith)
  have hn1 : ⌈2 * Real.log (1 / α) / kl2 q₁ q₀⌉₊ ≤ n :=
    le_trans (le_max_left _ _) hn
  have hn2 : ⌈4 * klVar q₁ q₀ / (β * kl2 q₁ q₀ ^ 2)⌉₊ ≤ n :=
    le_trans (le_max_right _ _) hn
  have hb1 : 2 * Real.log (1 / α) / kl2 q₁ q₀ ≤ (n : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast Nat.cast_le.2 hn1)
  have hb2 : 4 * klVar q₁ q₀ / (β * kl2 q₁ q₀ ^ 2) ≤ (n : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast Nat.cast_le.2 hn2)
  have hnpos : 0 < (n : ℝ) := lt_of_lt_of_le (by positivity) hb1
  -- the threshold is cleared twice over
  have hthresh : 2 * Real.log (1 / α) ≤ (n : ℝ) * kl2 q₁ q₀ := by
    rw [div_le_iff₀ hkl] at hb1; linarith
  have hgap : Real.log (1 / α) < (n : ℝ) * kl2 q₁ q₀ := by linarith
  have hhalf : Real.log (1 / α) ≤ (n : ℝ) * kl2 q₁ q₀ - Real.log (1 / α) := by linarith
  have hbound := sequential_power h10 h11 h00 h01 (c := 1 / α) n hgap
  refine le_trans hbound ?_
  -- the gap is at least half the drift, so the Chebyshev bound is at most `4·klVar/(n·KL²)`
  have hgap2 : (n : ℝ) * kl2 q₁ q₀ / 2 ≤ (n : ℝ) * kl2 q₁ q₀ - Real.log (1 / α) := by linarith
  have hgappos : 0 < (n : ℝ) * kl2 q₁ q₀ - Real.log (1 / α) := by linarith
  have hsq : ((n : ℝ) * kl2 q₁ q₀ / 2) ^ 2 ≤ ((n : ℝ) * kl2 q₁ q₀ - Real.log (1 / α)) ^ 2 := by
    have h0 : 0 ≤ (n : ℝ) * kl2 q₁ q₀ / 2 := by positivity
    nlinarith
  have hstep : (n * klVar q₁ q₀) / ((n : ℝ) * kl2 q₁ q₀ - Real.log (1 / α)) ^ 2
      ≤ (n * klVar q₁ q₀) / ((n : ℝ) * kl2 q₁ q₀ / 2) ^ 2 := by
    apply div_le_div_of_nonneg_left (by positivity) (by positivity) hsq
  refine le_trans hstep ?_
  -- and `n` is large enough for that to be at most `β`
  have hfinal : 4 * klVar q₁ q₀ ≤ β * kl2 q₁ q₀ ^ 2 * (n : ℝ) := by
    rw [div_le_iff₀ (by positivity)] at hb2; linarith
  rw [div_le_iff₀ (by positivity)]
  nlinarith [hnpos, hfinal, sq_nonneg (kl2 q₁ q₀)]

end Seq
end IDR
