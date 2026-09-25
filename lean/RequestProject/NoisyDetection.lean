/-
# Part XC.1  A real instrument: read-out noise, finite samples, and calibration

`RequestProject.DetectionPower` prices the capacity test under an idealisation that no
experiment satisfies: that the observer can *see which conformation* a molecule is in, so that
a single molecule found in the omitted set drives the under-capacity model's likelihood to
exactly zero.  Real read-outs are binary reporters with errors: a contact probe, a distance
window, a labelled pair fires with probability `se` (sensitivity) when the molecule is in the
omitted set and with probability `1 - sp` (one minus specificity) when it is not.  Under such a
reporter no likelihood is ever exactly zero, refutation becomes an inference about a rate, and
the sample size is no longer `log(1/α)/τ`.

This file redoes the power calculation for that instrument, from scratch and with finite-sample
rigour — no asymptotics, no normal approximation, no measure theory: an explicit product law on
`Fin n → Bool`, its exact first and second moments, Chebyshev's inequality proved from them, and
a test with a proved error bound at every finite `n`.

* `sum_recProb`, `sum_recProb_cnt`, `sum_recProb_cnt_sq`, `sum_recProb_var` — the read-out law is
  a probability distribution with mean `n·q` and variance exactly `n·q·(1-q)`.
* `chebyshev` — the deviation bound, proved from those moments.
* `readRate_sub` — **noise shrinks the signal by Youden's index**: the positive rate exceeds the
  disorder-free baseline by exactly `τ·(se + sp - 1)`, not by `τ`.
* `test_error_truth`, `test_error_baseline` — the midpoint counting test has both error
  probabilities at most `1/(n·Δ²)` where `Δ` is the rate contrast, at every finite `n`.
* `samplesFor`, `samplesFor_spec`, `reporter_power` — hence `⌈1/(α·(τ·J)²)⌉` observations
  suffice: **the price of a noisy reporter is quadratic in the contrast, where a perfect
  reporter paid a logarithm.**
* `samplesFor_antitone` — planning against a *lower* bound on the contrast is conservative, which
  is what makes the number usable when `τ` and `J` are themselves estimates.
* `calibration_confound`, `no_power_of_uncalibrated` — the hard limit: a system with population
  `τ` read by a perfectly calibrated reporter and a system with *no* such population read by a
  reporter whose specificity is lower by `τ·J` generate the *identical* law on data.  No test, at
  any sample size, separates them.  So the reporter's specificity must be known to better than
  `τ·J` — a requirement on the instrument that no amount of counting can replace.
* `rate_estimate_error`, `calibrationSamples_spec` — the reporter's own rates must themselves be
  measured, and the same inequality prices that: `⌈1/(4αε²)⌉` calibration observations pin a rate
  to `±ε` with confidence `1-α`.
* `pilotContrast_le`, `design_from_pilot` — and because the contrast is never known in advance,
  the design must be sized from a *lower confidence bound* on it; doing so is sound, and doing so
  can only inflate the sample size.

Nothing here is a measurement.  What is proved is the arithmetic that turns measured reporter
characteristics into a sample size, and the theorem that says which part of the problem sample
size cannot fix.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace Noisy

open Finset
open scoped Classical

/-! ## 1. The law of `n` independent binary reads

A run of the instrument on `n` molecules is a word `s : Fin n → Bool`; `recProb q s` is its
probability when each molecule reads positive with probability `q`, and `cnt s` is the number of
positive reads.  Everything below is an identity or an inequality about these finite sums. -/

/-- The Bernoulli weight of a single read at positive-rate `q`. -/
noncomputable def bern (q : ℝ) (b : Bool) : ℝ := if b then q else 1 - q

/-- The probability of the data record `s` under `n` independent reads at rate `q`. -/
noncomputable def recProb (q : ℝ) {n : ℕ} (s : Fin n → Bool) : ℝ := ∏ i, bern q (s i)

/-- The number of positive reads in the record `s`. -/
noncomputable def cnt {n : ℕ} (s : Fin n → Bool) : ℝ := ∑ i, if s i then (1 : ℝ) else 0

lemma bern_nonneg {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (b : Bool) : 0 ≤ bern q b := by
  cases b <;> simp [bern] <;> linarith

lemma recProb_nonneg {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ} (s : Fin n → Bool) :
    0 ≤ recProb q s :=
  Finset.prod_nonneg fun i _ => bern_nonneg h0 h1 (s i)

lemma cnt_nonneg {n : ℕ} (s : Fin n → Bool) : 0 ≤ cnt s :=
  Finset.sum_nonneg fun i _ => by positivity

lemma sum_cons {n : ℕ} (f : (Fin (n + 1) → Bool) → ℝ) :
    ∑ s : Fin (n + 1) → Bool, f s = ∑ b : Bool, ∑ t : Fin n → Bool, f (Fin.cons b t) := by
  rw [← (Fin.consEquiv (fun _ : Fin (n + 1) => Bool)).sum_comp f, Fintype.sum_prod_type]
  rfl

lemma recProb_cons (q : ℝ) {n : ℕ} (b : Bool) (t : Fin n → Bool) :
    recProb q (Fin.cons b t) = bern q b * recProb q t := by
  simp [recProb, Fin.prod_univ_succ]

lemma cnt_cons {n : ℕ} (b : Bool) (t : Fin n → Bool) :
    cnt (Fin.cons b t) = (if b then (1 : ℝ) else 0) + cnt t := by
  simp only [cnt, Fin.sum_univ_succ, Fin.cons_zero, Fin.cons_succ]

/-- The read-out law is a probability distribution. -/
lemma sum_recProb (q : ℝ) (n : ℕ) : ∑ s : Fin n → Bool, recProb q s = 1 := by
  induction n with
  | zero => simp [recProb]
  | succ n ih => rw [sum_cons]; simp [recProb_cons, ← Finset.mul_sum, ih, bern]

/-- **Exact first moment**: the expected number of positive reads is `n·q`. -/
lemma sum_recProb_cnt (q : ℝ) (n : ℕ) :
    ∑ s : Fin n → Bool, recProb q s * cnt s = n * q := by
  induction n with
  | zero => simp [recProb, cnt]
  | succ n ih =>
      rw [sum_cons]
      have h : ∀ b : Bool, ∑ t : Fin n → Bool, recProb q (Fin.cons b t) * cnt (Fin.cons b t)
          = bern q b * ((if b then (1 : ℝ) else 0) + n * q) := by
        intro b
        have hstep : ∀ t : Fin n → Bool,
            recProb q (Fin.cons b t) * cnt (Fin.cons b t)
              = bern q b * ((if b then (1 : ℝ) else 0) * recProb q t + recProb q t * cnt t) := by
          intro t; rw [recProb_cons, cnt_cons]; ring
        rw [Finset.sum_congr rfl (fun t _ => hstep t), ← Finset.mul_sum, Finset.sum_add_distrib,
          ← Finset.mul_sum, sum_recProb, ih, mul_one]
      rw [Fintype.sum_bool, h, h]
      push_cast
      simp [bern]
      ring

/-- **Exact second moment** of the number of positive reads. -/
lemma sum_recProb_cnt_sq (q : ℝ) (n : ℕ) :
    ∑ s : Fin n → Bool, recProb q s * cnt s ^ 2 = n * q * (1 - q) + (n * q) ^ 2 := by
  induction n with
  | zero => simp [recProb, cnt]
  | succ n ih =>
      rw [sum_cons]
      have h : ∀ b : Bool, ∑ t : Fin n → Bool, recProb q (Fin.cons b t) * cnt (Fin.cons b t) ^ 2
          = bern q b * ((if b then (1 : ℝ) else 0) + 2 * (if b then (1 : ℝ) else 0) * (n * q)
              + (n * q * (1 - q) + (n * q) ^ 2)) := by
        intro b
        have hstep : ∀ t : Fin n → Bool,
            recProb q (Fin.cons b t) * cnt (Fin.cons b t) ^ 2
              = bern q b * ((if b then (1 : ℝ) else 0) * recProb q t
                  + 2 * (if b then (1 : ℝ) else 0) * (recProb q t * cnt t)
                  + recProb q t * cnt t ^ 2) := by
          intro t
          cases b <;> · rw [recProb_cons, cnt_cons]; simp; ring
        rw [Finset.sum_congr rfl (fun t _ => hstep t), ← Finset.mul_sum]
        congr 1
        rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
          sum_recProb, ih, sum_recProb_cnt, mul_one]
      rw [Fintype.sum_bool, h, h]
      push_cast
      simp [bern]
      ring

/-- **Exact variance**: `n·q·(1-q)`. -/
lemma sum_recProb_var (q : ℝ) (n : ℕ) :
    ∑ s : Fin n → Bool, recProb q s * (cnt s - n * q) ^ 2 = n * q * (1 - q) := by
  have hexp : ∀ s : Fin n → Bool, recProb q s * (cnt s - n * q) ^ 2
      = recProb q s * cnt s ^ 2 - 2 * (n * q) * (recProb q s * cnt s)
        + (n * q) ^ 2 * recProb q s := by
    intro s; ring
  rw [Finset.sum_congr rfl (fun s _ => hexp s), Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, sum_recProb, sum_recProb_cnt, sum_recProb_cnt_sq]
  ring

/-! ## 2. Chebyshev's inequality for the read-out law

Proved from the exact moments above, so the bound below holds at every finite `n` with no
approximation anywhere. -/

/-- The records whose positive count deviates from `m` by at least `r`. -/
noncomputable def devSet (n : ℕ) (m r : ℝ) : Finset (Fin n → Bool) :=
  Finset.univ.filter (fun s => r ≤ |cnt s - m|)

lemma mem_devSet {n : ℕ} {m r : ℝ} {s : Fin n → Bool} :
    s ∈ devSet n m r ↔ r ≤ |cnt s - m| := by
  simp [devSet]

/-- **Chebyshev.**  The probability that the positive count deviates from `n·q` by at least `r`
is at most `n·q·(1-q)/r²`. -/
theorem chebyshev {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (n : ℕ) {r : ℝ} (hr : 0 < r) :
    ∑ s ∈ devSet n (n * q) r, recProb q s ≤ n * q * (1 - q) / r ^ 2 := by
  have hkey : ∀ s ∈ devSet n (n * q) r,
      recProb q s ≤ recProb q s * (cnt s - n * q) ^ 2 / r ^ 2 := by
    intro s hs
    rw [mem_devSet] at hs
    have hsq : r ^ 2 ≤ (cnt s - n * q) ^ 2 := by
      nlinarith [sq_abs (cnt s - (n : ℝ) * q), abs_nonneg (cnt s - (n : ℝ) * q), hr.le]
    have hp := recProb_nonneg h0 h1 s
    rw [le_div_iff₀ (by positivity)]
    nlinarith
  calc ∑ s ∈ devSet n (n * q) r, recProb q s
      ≤ ∑ s ∈ devSet n (n * q) r, recProb q s * (cnt s - n * q) ^ 2 / r ^ 2 :=
        Finset.sum_le_sum hkey
    _ ≤ ∑ s : Fin n → Bool, recProb q s * (cnt s - n * q) ^ 2 / r ^ 2 := by
        refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) ?_
        intro s _ _
        have := recProb_nonneg h0 h1 s
        positivity
    _ = n * q * (1 - q) / r ^ 2 := by
        rw [← Finset.sum_div, sum_recProb_var]

/-! ## 3. The instrument: sensitivity, specificity, and the contrast that survives them

A binary reporter fires with probability `se` on a molecule in the omitted set and with
probability `1 - sp` otherwise.  What the counting experiment sees is not `τ` but the rate
`readRate τ se sp`, and what it can act on is the *difference* between that rate and the rate the
same reporter produces on a system with no such population. -/

/-- The rate of positive reads on a system whose omitted set carries population `tau`, read by a
reporter of sensitivity `se` and specificity `sp`. -/
noncomputable def readRate (tau se sp : ℝ) : ℝ := tau * se + (1 - tau) * (1 - sp)

/-- Youden's index of a reporter: `se + sp - 1`.  Zero for a reporter that is pure noise. -/
noncomputable def youden (se sp : ℝ) : ℝ := se + sp - 1

lemma readRate_zero (se sp : ℝ) : readRate 0 se sp = 1 - sp := by simp [readRate]

/-- **The signal available to a counting experiment is `τ·J`, not `τ`.** -/
theorem readRate_sub (tau se sp : ℝ) :
    readRate tau se sp - readRate 0 se sp = tau * youden se sp := by
  simp only [readRate, youden]; ring

/-- A reporter can only shrink the contrast: with `se, sp ≤ 1` the available signal `τ·J` is at
most the population `τ` itself, with equality only for a perfect reporter. -/
theorem youden_le_one {se sp : ℝ} (hse : se ≤ 1) (hsp : sp ≤ 1) : youden se sp ≤ 1 := by
  simp only [youden]; linarith

lemma readRate_nonneg {tau se sp : ℝ} (ht0 : 0 ≤ tau) (ht1 : tau ≤ 1) (hse : 0 ≤ se)
    (hsp : sp ≤ 1) : 0 ≤ readRate tau se sp := by
  have h₁ : 0 ≤ (1 - tau) * (1 - sp) := mul_nonneg (by linarith) (by linarith)
  have h₂ : 0 ≤ tau * se := mul_nonneg ht0 hse
  simp only [readRate]; linarith

lemma readRate_le_one {tau se sp : ℝ} (ht0 : 0 ≤ tau) (ht1 : tau ≤ 1) (hse : se ≤ 1)
    (hsp : 0 ≤ sp) : readRate tau se sp ≤ 1 := by
  simp only [readRate]; nlinarith

/-! ## 4. The counting test and its finite-sample error bounds

The test is the obvious one: reject the disorder-free baseline when the positive count exceeds
the midpoint `n·(q₀+q₁)/2` between the two predicted rates.  Both of its error probabilities are
bounded at every finite `n`. -/

/-- The records on which the test rejects the baseline. -/
noncomputable def rejects (n : ℕ) (c : ℝ) : Finset (Fin n → Bool) :=
  Finset.univ.filter (fun s => c < cnt s)

/-- The records on which the test fails to reject the baseline. -/
noncomputable def accepts (n : ℕ) (c : ℝ) : Finset (Fin n → Bool) :=
  Finset.univ.filter (fun s => cnt s ≤ c)

/-- **Type II error.**  Under the true rate `q₁`, the probability that the midpoint test fails to
reject the baseline is at most `1/(n·Δ²)`, where `Δ = q₁ - q₀` is the rate contrast. -/
theorem test_error_truth {q₀ q₁ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₁ ≤ 1) (hlt : q₀ < q₁) {n : ℕ}
    (hn : 0 < n) :
    ∑ s ∈ accepts n (n * (q₀ + q₁) / 2), recProb q₁ s ≤ 1 / (n * (q₁ - q₀) ^ 2) := by
  have hΔpos : 0 < q₁ - q₀ := by linarith
  have hq₁0 : 0 ≤ q₁ := le_trans h0 hlt.le
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hr : 0 < (n : ℝ) * (q₁ - q₀) / 2 := by positivity
  have hsub : accepts n (n * (q₀ + q₁) / 2)
      ⊆ devSet n ((n : ℝ) * q₁) ((n : ℝ) * (q₁ - q₀) / 2) := by
    intro s hs
    simp only [accepts, Finset.mem_filter, Finset.mem_univ, true_and] at hs
    rw [mem_devSet, le_abs]
    exact Or.inr (by linarith)
  calc ∑ s ∈ accepts n (n * (q₀ + q₁) / 2), recProb q₁ s
      ≤ ∑ s ∈ devSet n ((n : ℝ) * q₁) ((n : ℝ) * (q₁ - q₀) / 2), recProb q₁ s := by
        refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
        intro s _ _
        exact recProb_nonneg hq₁0 h1 s
    _ ≤ (n : ℝ) * q₁ * (1 - q₁) / ((n : ℝ) * (q₁ - q₀) / 2) ^ 2 := chebyshev hq₁0 h1 n hr
    _ ≤ 1 / ((n : ℝ) * (q₁ - q₀) ^ 2) := by
        rw [div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith [sq_nonneg (1 - 2 * q₁), hnR.le, sq_nonneg ((n : ℝ) * (q₁ - q₀))]

/-- **Type I error.**  Under the disorder-free baseline rate `q₀`, the probability that the
midpoint test rejects it is at most `1/(n·Δ²)`. -/
theorem test_error_baseline {q₀ q₁ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₁ ≤ 1) (hlt : q₀ < q₁) {n : ℕ}
    (hn : 0 < n) :
    ∑ s ∈ rejects n (n * (q₀ + q₁) / 2), recProb q₀ s ≤ 1 / (n * (q₁ - q₀) ^ 2) := by
  have hΔpos : 0 < q₁ - q₀ := by linarith
  have hq₀1 : q₀ ≤ 1 := le_trans hlt.le h1
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hr : 0 < (n : ℝ) * (q₁ - q₀) / 2 := by positivity
  have hsub : rejects n (n * (q₀ + q₁) / 2)
      ⊆ devSet n ((n : ℝ) * q₀) ((n : ℝ) * (q₁ - q₀) / 2) := by
    intro s hs
    simp only [rejects, Finset.mem_filter, Finset.mem_univ, true_and] at hs
    rw [mem_devSet, le_abs]
    exact Or.inl (by linarith)
  calc ∑ s ∈ rejects n (n * (q₀ + q₁) / 2), recProb q₀ s
      ≤ ∑ s ∈ devSet n ((n : ℝ) * q₀) ((n : ℝ) * (q₁ - q₀) / 2), recProb q₀ s := by
        refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
        intro s _ _
        exact recProb_nonneg h0 hq₀1 s
    _ ≤ (n : ℝ) * q₀ * (1 - q₀) / ((n : ℝ) * (q₁ - q₀) / 2) ^ 2 := chebyshev h0 hq₀1 n hr
    _ ≤ 1 / ((n : ℝ) * (q₁ - q₀) ^ 2) := by
        rw [div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith [sq_nonneg (1 - 2 * q₀), hnR.le, sq_nonneg ((n : ℝ) * (q₁ - q₀))]

/-! ## 5. The sample size, and how it must be planned -/

/-- The number of molecules to observe: `⌈1/(α·Δ²)⌉`, for contrast `Δ` and error level `α`. -/
noncomputable def samplesFor (alpha delta : ℝ) : ℕ := ⌈1 / (alpha * delta ^ 2)⌉₊

lemma samplesFor_pos {alpha delta : ℝ} (ha : 0 < alpha) (hd : 0 < delta) :
    0 < samplesFor alpha delta := by
  simp only [samplesFor]
  exact Nat.ceil_pos.mpr (by positivity)

/-- With `samplesFor α Δ` observations (or more), both error probabilities of the midpoint test
are at most `α`. -/
theorem samplesFor_spec {q₀ q₁ alpha : ℝ} (h0 : 0 ≤ q₀) (h1 : q₁ ≤ 1) (hlt : q₀ < q₁)
    (ha : 0 < alpha) {n : ℕ} (hn : samplesFor alpha (q₁ - q₀) ≤ n) :
    ∑ s ∈ accepts n (n * (q₀ + q₁) / 2), recProb q₁ s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (q₀ + q₁) / 2), recProb q₀ s ≤ alpha := by
  have hΔpos : 0 < q₁ - q₀ := by linarith
  have hpos : 0 < samplesFor alpha (q₁ - q₀) := samplesFor_pos ha hΔpos
  have hn0 : 0 < n := lt_of_lt_of_le hpos hn
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn0
  have hge : 1 / (alpha * (q₁ - q₀) ^ 2) ≤ (n : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast hn)
  have hbound : 1 / ((n : ℝ) * (q₁ - q₀) ^ 2) ≤ alpha := by
    rw [div_le_iff₀ (by positivity)]
    rw [div_le_iff₀ (by positivity)] at hge
    nlinarith
  exact ⟨le_trans (test_error_truth h0 h1 hlt hn0) hbound,
    le_trans (test_error_baseline h0 h1 hlt hn0) hbound⟩

/-- **Planning against a lower bound on the contrast is safe**: the sample size is antitone in
the contrast, so substituting any conservative (smaller) estimate of `τ·J` can only increase the
number of molecules the protocol demands. -/
theorem samplesFor_antitone {alpha d d' : ℝ} (ha : 0 < alpha) (hd : 0 < d') (hle : d' ≤ d) :
    samplesFor alpha d ≤ samplesFor alpha d' := by
  simp only [samplesFor]
  refine Nat.ceil_le_ceil ?_
  gcongr

/-- The sample size for the realistic instrument, in the quantities a practitioner measures: the
omitted population `τ`, the sensitivity and the specificity. -/
noncomputable def samplesForReporter (alpha tau se sp : ℝ) : ℕ :=
  samplesFor alpha (tau * youden se sp)

/-- **The realistic power theorem.**  If an under-capacity model gives population zero to a set
that truly carries `τ`, and that set is read by a reporter with Youden index `J > 0`, then
`⌈1/(α·(τJ)²)⌉` observed molecules make both errors of the counting test at most `α`. -/
theorem reporter_power {tau se sp alpha : ℝ} (ht0 : 0 < tau) (ht1 : tau ≤ 1) (hse : se ≤ 1)
    (hsp0 : 0 ≤ sp) (hsp1 : sp ≤ 1) (hJ : 0 < youden se sp) (ha : 0 < alpha)
    {n : ℕ} (hn : samplesForReporter alpha tau se sp ≤ n) :
    ∑ s ∈ accepts n (n * (readRate 0 se sp + readRate tau se sp) / 2),
        recProb (readRate tau se sp) s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (readRate 0 se sp + readRate tau se sp) / 2),
        recProb (readRate 0 se sp) s ≤ alpha := by
  have hgap : readRate tau se sp - readRate 0 se sp = tau * youden se sp := readRate_sub _ _ _
  have hlt : readRate 0 se sp < readRate tau se sp := by
    have : 0 < tau * youden se sp := mul_pos ht0 hJ
    linarith
  have h0 : 0 ≤ readRate 0 se sp := by rw [readRate_zero]; linarith
  have h1 : readRate tau se sp ≤ 1 := readRate_le_one ht0.le ht1 hse hsp0
  refine samplesFor_spec h0 h1 hlt ha ?_
  rw [hgap]
  exact hn

/-! ## 6. What sample size cannot buy: the calibration confound

The following is the sharpest statement in this file, and it is a limitation, not a capability. -/

/-- **Exact confound.**  A system whose omitted set carries population `tau`, read by a reporter
of specificity `sp`, produces exactly the same positive rate as a system with *no* such
population read by a reporter whose specificity is lower by `tau·J`. -/
theorem readRate_confound (tau se sp : ℝ) :
    readRate tau se sp = readRate 0 se (sp - tau * youden se sp) := by
  simp only [readRate, youden]; ring

/-- The records a decision rule `T` rejects on. -/
noncomputable def ruleSet {n : ℕ} (T : (Fin n → Bool) → Bool) : Finset (Fin n → Bool) :=
  Finset.univ.filter (fun s => T s)

/-- Consequently the two situations generate the *identical* law on data records, so **every**
decision rule — at every sample size, however large — assigns them exactly the same probability
of rejection.  Counting more molecules cannot separate a real population from a calibration
error of the same size. -/
theorem calibration_confound (tau se sp : ℝ) (n : ℕ) (T : (Fin n → Bool) → Bool) :
    ∑ s ∈ ruleSet T, recProb (readRate tau se sp) s
      = ∑ s ∈ ruleSet T, recProb (readRate 0 se (sp - tau * youden se sp)) s := by
  rw [readRate_confound tau se sp]

/-- **The instrument requirement.**  If the reporter's specificity is only known to within `eta`
and the available contrast `tau·J` is at most `eta`, then a disorder-free system with an
admissible specificity is observationally indistinguishable from the real one: every test has
identical behaviour on the two.  The reporter must therefore be calibrated to better than
`tau·J`; this is a requirement on the hardware, and no sample size substitutes for it. -/
theorem no_power_of_uncalibrated {tau se sp eta : ℝ} (ht : 0 ≤ tau) (hJ : 0 ≤ youden se sp)
    (hsmall : tau * youden se sp ≤ eta) :
    ∃ sp' : ℝ, |sp' - sp| ≤ eta ∧ readRate 0 se sp' = readRate tau se sp ∧
      ∀ (n : ℕ) (T : (Fin n → Bool) → Bool),
        ∑ s ∈ ruleSet T, recProb (readRate tau se sp) s
          = ∑ s ∈ ruleSet T, recProb (readRate 0 se sp') s := by
  refine ⟨sp - tau * youden se sp, ?_, (readRate_confound tau se sp).symm, ?_⟩
  · have hnn : 0 ≤ tau * youden se sp := mul_nonneg ht hJ
    rw [abs_le]
    constructor <;> linarith
  · intro n T
    rw [← readRate_confound tau se sp]

/-! ## 7. The reporter's own numbers must be measured, and that costs too -/

/-- The records whose observed frequency of positive reads misses the rate `q` by `eps` or
more. -/
noncomputable def estErrSet (n : ℕ) (q eps : ℝ) : Finset (Fin n → Bool) :=
  Finset.univ.filter (fun s => eps ≤ |cnt s / n - q|)

/-- **Estimating a rate.**  The probability that the observed frequency of positive reads misses
the true rate by `eps` or more is at most `1/(4nε²)`. -/
theorem rate_estimate_error {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ} (hn : 0 < n) {eps : ℝ}
    (heps : 0 < eps) :
    ∑ s ∈ estErrSet n q eps, recProb q s ≤ 1 / (4 * n * eps ^ 2) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have hsub : estErrSet n q eps ⊆ devSet n ((n : ℝ) * q) ((n : ℝ) * eps) := by
    intro s hs
    simp only [estErrSet, Finset.mem_filter, Finset.mem_univ, true_and] at hs
    rw [mem_devSet]
    have hrw : cnt s - (n : ℝ) * q = (n : ℝ) * (cnt s / n - q) := by field_simp
    rw [hrw, abs_mul, abs_of_pos hnR]
    exact mul_le_mul_of_nonneg_left hs hnR.le
  calc ∑ s ∈ estErrSet n q eps, recProb q s
      ≤ ∑ s ∈ devSet n ((n : ℝ) * q) ((n : ℝ) * eps), recProb q s := by
        refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
        intro s _ _
        exact recProb_nonneg h0 h1 s
    _ ≤ (n : ℝ) * q * (1 - q) / ((n : ℝ) * eps) ^ 2 := chebyshev h0 h1 n (by positivity)
    _ ≤ 1 / (4 * (n : ℝ) * eps ^ 2) := by
        rw [div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith [sq_nonneg (1 - 2 * q), hnR.le, sq_nonneg ((n : ℝ) * eps)]

/-- The number of calibration observations that pin a reporter rate to `±eps` with confidence
`1 - alpha`. -/
noncomputable def calibrationSamples (alpha eps : ℝ) : ℕ := ⌈1 / (4 * alpha * eps ^ 2)⌉₊

/-- `calibrationSamples α ε` observations suffice. -/
theorem calibrationSamples_spec {q alpha eps : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (ha : 0 < alpha)
    (heps : 0 < eps) {n : ℕ} (hn : calibrationSamples alpha eps ≤ n) (hn0 : 0 < n) :
    ∑ s ∈ estErrSet n q eps, recProb q s ≤ alpha := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn0
  have hge : 1 / (4 * alpha * eps ^ 2) ≤ (n : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast hn)
  have hbound : 1 / (4 * (n : ℝ) * eps ^ 2) ≤ alpha := by
    rw [div_le_iff₀ (by positivity)]
    rw [div_le_iff₀ (by positivity)] at hge
    nlinarith
  exact le_trans (rate_estimate_error h0 h1 hn0 heps) hbound

/-! ## 8. Designing from a pilot, when the contrast is not known in advance

The sample size of §5 is computed from a contrast that a real study does not know before it
starts: `τ` is an estimate and the reporter's rates are estimates.  Plugging a point estimate
into `samplesFor` is not sound — an optimistic contrast silently under-powers the run.  The
sound procedure is to plug in a *lower confidence bound*, and the two ingredients for that are
already proved: `rate_estimate_error` says how far a pilot frequency can be from the truth, and
`samplesFor_antitone` says that under-estimating the contrast can only inflate the sample
size. -/

/-- The conservative contrast a pilot run licenses: the observed positive frequency, reduced by
the pilot's guaranteed precision, minus the calibrated baseline rate. -/
noncomputable def pilotContrast (freq eps q₀ : ℝ) : ℝ := freq - eps - q₀

/-- **A pilot within its stated precision never overstates the contrast.**  (The probability
that it is within that precision is bounded by `rate_estimate_error`.) -/
theorem pilotContrast_le {freq eps q₀ q₁ : ℝ} (h : |freq - q₁| ≤ eps) :
    pilotContrast freq eps q₀ ≤ q₁ - q₀ := by
  have h' : freq - q₁ ≤ eps := (abs_le.mp h).2
  simp only [pilotContrast]
  linarith

/-- **The two-stage design is sound.**  Run a pilot, take the conservative contrast it licenses,
and size the main run from that: whenever the pilot has not overstated the contrast, both error
probabilities of the counting test are at most `alpha`. -/
theorem design_from_pilot {q₀ q₁ freq eps alpha : ℝ} (h0 : 0 ≤ q₀) (h1 : q₁ ≤ 1)
    (ha : 0 < alpha) (hpos : 0 < pilotContrast freq eps q₀)
    (hcons : pilotContrast freq eps q₀ ≤ q₁ - q₀) {n : ℕ}
    (hn : samplesFor alpha (pilotContrast freq eps q₀) ≤ n) :
    ∑ s ∈ accepts n (n * (q₀ + q₁) / 2), recProb q₁ s ≤ alpha ∧
      ∑ s ∈ rejects n (n * (q₀ + q₁) / 2), recProb q₀ s ≤ alpha := by
  have hlt : q₀ < q₁ := by linarith
  refine samplesFor_spec h0 h1 hlt ha (le_trans ?_ hn)
  exact samplesFor_antitone ha hpos hcons

end Noisy
end IDR
