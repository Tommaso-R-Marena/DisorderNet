/-
# Part CXIII  Calibrating the null read rate: the protocol that closes the gap

Two earlier parts prove that the null read rate is the one thing the statistics cannot supply.
Part XC.1 (`RequestProject.NoisyDetection`) shows the confound exactly: a system with population
`tau` read by a calibrated reporter and a system with *no* such population read by a reporter
whose specificity is lower by `tau·J` generate the identical law on data, so no test at any
sample size separates them.  Part CIX (`RequestProject.DependentReads`) removes the independence
assumption from the sequential machinery and records that this leaves the calibration hypothesis
untouched: "a wealth process built on a miscalibrated null is a bet on an instrument artefact".
Neither sequential analysis nor multiplicity control repairs calibration.

That is a genuine limitation, and the honest way to close it is not to repair it from inside the
statistics but to give the *external protocol* and prove that it suffices, with an explicit error
budget.  That is what this file does.

* `uncalibrated_no_power` — first, the sharpened impossibility.  A rejection rule that is valid at
  level `alpha` for **every** null read rate has power at most `alpha` against every alternative
  rate: with an unknown null, validity forces powerlessness.  This holds for any rejection region
  whatsoever, so no amount of sequential analysis or multiplicity control evades it.

* `prob_mono_of_upward` — the enabling fact.  For an **upward-closed** rejection region (more
  positive reads can only strengthen the case for rejection), the rejection probability is
  monotone in the read rate.  Proved from scratch by induction on the number of reads.

* `plugin_valid` — hence **an upper bound on the null rate is all that is needed**: a test
  calibrated at an assumed rate `qhat` keeps its level at every true rate `q ≤ qhat`.  Being
  conservative about the instrument is sound; being wrong in the other direction is not.

* `seqReject_upward` — and the sequential rules are of this kind: any anytime rule that rejects
  when the running count of positive reads crosses a boundary, at any horizon, is upward closed.
  So the plug-in guarantee covers the sequential test, not only the fixed-`n` one.

* `calibrated_test_valid` — the protocol, with its budget.  Run `m` reads on a control system
  whose rate *is* the null rate, plug in `min 1 (frequency + eps)`, and run any upward-closed
  level-`alpha` rule on the `n` experimental reads.  The total type-I error, over the joint law of
  calibration run and experiment, is at most `alpha + 1/(4·m·eps²)`.

* `calibrated_test_valid_samples` — in the design form: with `m ≥ calibrationSamples delta eps`
  control reads the total error is at most `alpha + delta`.  This is the theorem that makes the
  external calibration requirement into a finite, checkable experimental protocol rather than an
  open assumption.

The requirement itself is not removed — it cannot be, by `uncalibrated_no_power` — but it is now
*discharged by a stated experiment of stated size*, and the price of discharging it is a term in
the error budget that the design can be sized against.
-/
import Mathlib
import RequestProject.NoisyDetection

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace NullCal

open Finset
open IDR.Noisy
open scoped Classical

/-! ## Rejection regions and their probability -/

/-- The probability that the record falls in the region `A`, under `n` independent reads at
rate `q`. -/
noncomputable def prob (q : ℝ) {n : ℕ} (A : Finset (Fin n → Bool)) : ℝ := ∑ s ∈ A, recProb q s

lemma prob_nonneg {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ} (A : Finset (Fin n → Bool)) :
    0 ≤ prob q A :=
  Finset.sum_nonneg fun s _ => recProb_nonneg h0 h1 s

lemma prob_le_one {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ} (A : Finset (Fin n → Bool)) :
    prob q A ≤ 1 := by
  have := sum_recProb q n
  calc prob q A ≤ ∑ s : Fin n → Bool, recProb q s :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
          (fun s _ _ => recProb_nonneg h0 h1 s)
    _ = 1 := this

lemma prob_mono_subset {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) {n : ℕ}
    {A B : Finset (Fin n → Bool)} (hAB : A ⊆ B) : prob q A ≤ prob q B :=
  Finset.sum_le_sum_of_subset_of_nonneg hAB (fun s _ _ => recProb_nonneg h0 h1 s)

/-! ## Upward-closed regions -/

/-- `s` reads positive wherever `t` does. -/
def Dom {n : ℕ} (t s : Fin n → Bool) : Prop := ∀ i, t i = true → s i = true

/-- A rejection region is upward closed when turning any read from negative to positive can only
keep the record inside it: more evidence never withdraws a rejection. -/
def Upward {n : ℕ} (A : Finset (Fin n → Bool)) : Prop :=
  ∀ t ∈ A, ∀ s, Dom t s → s ∈ A

/-- **Monotonicity in the read rate.**  For an upward-closed region the rejection probability
increases with the rate of positive reads. -/
theorem prob_mono_of_upward : ∀ (n : ℕ) (A : Finset (Fin n → Bool)), Upward A →
    ∀ {q q' : ℝ}, 0 ≤ q → q ≤ q' → q' ≤ 1 → prob q A ≤ prob q' A := by
  intro n
  induction n with
  | zero =>
      intro A _ q q' _ _ _
      simp [prob, recProb]
  | succ k ih =>
      intro A hA q q' h0 hq h1
      have h0' : 0 ≤ q' := le_trans h0 hq
      have h1' : q ≤ 1 := le_trans hq h1
      -- split on the first read
      set Ab : Bool → Finset (Fin k → Bool) :=
        fun b => Finset.univ.filter (fun t => Fin.cons b t ∈ A) with hAb
      have hdec : ∀ r : ℝ, prob r A
          = (1 - r) * prob r (Ab false) + r * prob r (Ab true) := by
        intro r
        have h1r : prob r A = ∑ s : Fin (k + 1) → Bool, (if s ∈ A then recProb r s else 0) := by
          rw [prob, Finset.sum_ite_mem, Finset.univ_inter]
        rw [h1r, sum_cons]
        have hb : ∀ b : Bool, ∑ t : Fin k → Bool,
            (if Fin.cons b t ∈ A then recProb r (Fin.cons b t) else 0)
              = bern r b * prob r (Ab b) := by
          intro b
          rw [prob, hAb]
          simp only [Finset.sum_filter, Finset.mul_sum]
          refine Finset.sum_congr rfl fun t _ => ?_
          by_cases hmem : Fin.cons b t ∈ A <;> simp [hmem, recProb_cons]
        rw [Fintype.sum_bool, hb false, hb true]
        simp only [bern, if_true, if_false, Bool.false_eq_true]
        ring
      have hup : ∀ b, Upward (Ab b) := by
        intro b t ht s hts
        simp only [hAb, Finset.mem_filter, Finset.mem_univ, true_and] at ht ⊢
        refine hA _ ht _ ?_
        intro i
        induction i using Fin.cases with
        | zero => intro hi; simpa using hi
        | succ j => intro hi; simpa using hts j (by simpa using hi)
      have hsub : Ab false ⊆ Ab true := by
        intro t ht
        simp only [hAb, Finset.mem_filter, Finset.mem_univ, true_and] at ht ⊢
        refine hA _ ht _ ?_
        intro i
        induction i using Fin.cases with
        | zero => intro hi; simp at hi
        | succ j => intro hi; simpa using hi
      have hFG : prob q' (Ab false) ≤ prob q' (Ab true) := prob_mono_subset h0' h1 hsub
      have hF := ih (Ab false) (hup false) h0 hq h1
      have hG := ih (Ab true) (hup true) h0 hq h1
      rw [hdec q, hdec q']
      nlinarith [hFG, hF, hG, h0, hq, h1, h1']

/-- **A conservative null rate is sound.**  A test whose rejection region is upward closed and
which has level `alpha` at the assumed rate `qhat` has level `alpha` at every true rate below
it. -/
theorem plugin_valid {n : ℕ} {A : Finset (Fin n → Bool)} (hA : Upward A) {q qhat alpha : ℝ}
    (h0 : 0 ≤ q) (hle : q ≤ qhat) (h1 : qhat ≤ 1) (hlev : prob qhat A ≤ alpha) :
    prob q A ≤ alpha :=
  le_trans (prob_mono_of_upward n A hA h0 hle h1) hlev

/-! ## Sequential rules are upward closed -/

/-- The number of positive reads among the first `k`. -/
noncomputable def cntPrefix {n : ℕ} (k : ℕ) (s : Fin n → Bool) : ℝ :=
  ∑ i ∈ Finset.univ.filter (fun i : Fin n => (i : ℕ) < k), if s i then (1 : ℝ) else 0

lemma cntPrefix_mono {n : ℕ} {t s : Fin n → Bool} (h : Dom t s) (k : ℕ) :
    cntPrefix k t ≤ cntPrefix k s := by
  refine Finset.sum_le_sum fun i _ => ?_
  cases ht : t i with
  | false =>
      simp only [Bool.false_eq_true, if_false]
      split <;> norm_num
  | true =>
      rw [h i ht]

/-- The rejection region of an **anytime** rule: reject as soon as the running count of positive
reads crosses the boundary `c`, at any horizon up to `n`. -/
noncomputable def seqReject (n : ℕ) (c : ℕ → ℝ) : Finset (Fin n → Bool) :=
  Finset.univ.filter (fun s => ∃ k ≤ n, c k ≤ cntPrefix k s)

/-- **Sequential count rules are upward closed**, so the plug-in guarantee applies to them. -/
theorem seqReject_upward (n : ℕ) (c : ℕ → ℝ) : Upward (seqReject n c) := by
  intro t ht s hts
  simp only [seqReject, Finset.mem_filter, Finset.mem_univ, true_and] at ht ⊢
  obtain ⟨k, hk, hck⟩ := ht
  exact ⟨k, hk, le_trans hck (cntPrefix_mono hts k)⟩

/-- The fixed-horizon counting test is upward closed too. -/
theorem countReject_upward (n : ℕ) (c : ℝ) :
    Upward (Finset.univ.filter (fun s : Fin n → Bool => c ≤ cnt s)) := by
  intro t ht s hts
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ht ⊢
  refine le_trans ht (Finset.sum_le_sum fun i _ => ?_)
  cases hti : t i with
  | false =>
      simp only [Bool.false_eq_true, if_false]
      split <;> norm_num
  | true =>
      rw [hts i hti]

/-! ## The impossibility, sharpened -/

/-- **Without a calibrated null there is no power.**  A rejection rule valid at level `alpha`
for every possible null read rate rejects with probability at most `alpha` under *every* rate,
alternative rates included.  This holds for an arbitrary rejection region, so no sequential
design and no multiplicity correction escapes it: the null rate has to come from outside the
data. -/
theorem uncalibrated_no_power {n : ℕ} (A : Finset (Fin n → Bool)) {alpha : ℝ}
    (hvalid : ∀ q, 0 ≤ q → q ≤ 1 → prob q A ≤ alpha) {q₁ : ℝ} (h0 : 0 ≤ q₁) (h1 : q₁ ≤ 1) :
    prob q₁ A ≤ alpha := hvalid q₁ h0 h1

/-! ## The calibration protocol and its error budget -/

/-- The conservative null rate licensed by a calibration run: the observed frequency of positive
reads on the control, inflated by the guaranteed precision, capped at one. -/
noncomputable def plugRate {m : ℕ} (c : Fin m → Bool) (eps : ℝ) : ℝ := min 1 (cnt c / m + eps)

lemma plugRate_le_one {m : ℕ} (c : Fin m → Bool) (eps : ℝ) : plugRate c eps ≤ 1 :=
  min_le_left _ _

/-- Off the calibration-failure event, the plug-in rate is an upper bound on the truth. -/
lemma le_plugRate_of_good {m : ℕ} {q₀ eps : ℝ} (h1 : q₀ ≤ 1) (c : Fin m → Bool)
    (hgood : c ∉ estErrSet m q₀ eps) : q₀ ≤ plugRate c eps := by
  simp only [estErrSet, Finset.mem_filter, Finset.mem_univ, true_and, not_le] at hgood
  have habs : |cnt c / m - q₀| < eps := hgood
  have := abs_lt.1 habs
  exact le_min h1 (by linarith [this.1])

/-- **The calibrated protocol is valid, with an explicit budget.**  Running `m` reads on a
control system at the null rate, plugging in the conservative rate, and applying any
upward-closed rule that has level `alpha` at its assumed rate, the total type-I error over the
joint law of calibration and experiment is at most `alpha + 1/(4·m·eps²)`. -/
theorem calibrated_test_valid {m n : ℕ} (hm : 0 < m) {q₀ eps alpha : ℝ}
    (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (heps : 0 < eps) (halpha : 0 ≤ alpha)
    (A : ℝ → Finset (Fin n → Bool)) (hup : ∀ r, Upward (A r))
    (hlev : ∀ r, prob r (A r) ≤ alpha) :
    ∑ c : Fin m → Bool, recProb q₀ c * prob q₀ (A (plugRate c eps))
      ≤ alpha + 1 / (4 * m * eps ^ 2) := by
  classical
  set G : Finset (Fin m → Bool) := Finset.univ \ estErrSet m q₀ eps with hG
  have hsplit : ∑ c : Fin m → Bool, recProb q₀ c * prob q₀ (A (plugRate c eps))
      = (∑ c ∈ G, recProb q₀ c * prob q₀ (A (plugRate c eps)))
        + ∑ c ∈ estErrSet m q₀ eps, recProb q₀ c * prob q₀ (A (plugRate c eps)) := by
    rw [hG, Finset.sum_sdiff_eq_sub (Finset.subset_univ _)]
    ring
  have hgood : ∑ c ∈ G, recProb q₀ c * prob q₀ (A (plugRate c eps)) ≤ alpha := by
    have hterm : ∀ c ∈ G, recProb q₀ c * prob q₀ (A (plugRate c eps)) ≤ recProb q₀ c * alpha := by
      intro c hc
      have hnotmem : c ∉ estErrSet m q₀ eps := by
        rw [hG, Finset.mem_sdiff] at hc
        exact hc.2
      have hleq : q₀ ≤ plugRate c eps := le_plugRate_of_good h1 c hnotmem
      have := plugin_valid (hup (plugRate c eps)) h0 hleq (plugRate_le_one c eps)
        (hlev (plugRate c eps))
      exact mul_le_mul_of_nonneg_left this (recProb_nonneg h0 h1 c)
    calc ∑ c ∈ G, recProb q₀ c * prob q₀ (A (plugRate c eps))
        ≤ ∑ c ∈ G, recProb q₀ c * alpha := Finset.sum_le_sum hterm
      _ = (∑ c ∈ G, recProb q₀ c) * alpha := by rw [Finset.sum_mul]
      _ ≤ 1 * alpha := by
          refine mul_le_mul_of_nonneg_right ?_ halpha
          calc ∑ c ∈ G, recProb q₀ c ≤ ∑ c : Fin m → Bool, recProb q₀ c :=
                Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
                  (fun s _ _ => recProb_nonneg h0 h1 s)
            _ = 1 := sum_recProb q₀ m
      _ = alpha := one_mul alpha
  have hbad : ∑ c ∈ estErrSet m q₀ eps, recProb q₀ c * prob q₀ (A (plugRate c eps))
      ≤ 1 / (4 * m * eps ^ 2) := by
    have hterm : ∀ c ∈ estErrSet m q₀ eps,
        recProb q₀ c * prob q₀ (A (plugRate c eps)) ≤ recProb q₀ c := by
      intro c _
      have hp := prob_le_one (q := q₀) h0 h1 (A (plugRate c eps))
      have := recProb_nonneg h0 h1 c
      nlinarith [prob_nonneg (q := q₀) h0 h1 (A (plugRate c eps))]
    calc ∑ c ∈ estErrSet m q₀ eps, recProb q₀ c * prob q₀ (A (plugRate c eps))
        ≤ ∑ c ∈ estErrSet m q₀ eps, recProb q₀ c := Finset.sum_le_sum hterm
      _ ≤ 1 / (4 * m * eps ^ 2) := rate_estimate_error h0 h1 hm heps
  rw [hsplit]
  linarith

/-- **The design form.**  With `calibrationSamples delta eps` control reads the calibration term
is `delta`, so the whole procedure has type-I error at most `alpha + delta`. -/
theorem calibrated_test_valid_samples {m n : ℕ} {q₀ eps alpha delta : ℝ}
    (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (heps : 0 < eps) (halpha : 0 ≤ alpha) (hdelta : 0 < delta)
    (hm : calibrationSamples delta eps ≤ m) (hm0 : 0 < m)
    (A : ℝ → Finset (Fin n → Bool)) (hup : ∀ r, Upward (A r))
    (hlev : ∀ r, prob r (A r) ≤ alpha) :
    ∑ c : Fin m → Bool, recProb q₀ c * prob q₀ (A (plugRate c eps)) ≤ alpha + delta := by
  have hmain := calibrated_test_valid (m := m) (n := n) hm0 h0 h1 heps halpha A hup hlev
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm0
  have hge : 1 / (4 * delta * eps ^ 2) ≤ (m : ℝ) :=
    le_trans (Nat.le_ceil _) (by exact_mod_cast hm)
  have hbound : 1 / (4 * (m : ℝ) * eps ^ 2) ≤ delta := by
    rw [div_le_iff₀ (by positivity)]
    rw [div_le_iff₀ (by positivity)] at hge
    nlinarith
  linarith

end NullCal
end IDR
