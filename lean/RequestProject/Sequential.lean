/-
# Part XCII.1  Sequential counting: when may the run be stopped?

`RequestProject.NoisyDetection` prices the capacity test for a *fixed* number of molecules: the
run length `n` is chosen in advance from `α`, the omitted population `τ` and the reporter's
Youden index `J`, and the data may only be scored once, at the end.  Real runs are not like
that.  Molecules arrive one at a time, the experimenter watches the counter, and the run stops
when the answer looks clear or when the sample runs out.  Under the fixed-`n` theory that
behaviour is fatal: a test whose stopping time depends on the data has no proved error bound at
all, and looking twice at level `α` already exceeds level `α` (`peeking_inflates_error` below).

This file rebuilds the test so that stopping is free.  The statistic is the wealth of a bet on
the alternative — the running likelihood ratio — and the theorem is Ville's inequality, proved
here from scratch by induction over the read-out word, with no measure theory and no
martingale library:

* `words`, `probL`, `sum_probL` — the law of a run of `n` binary reads, as an explicit finite
  sum over words.
* `wealth`, `wealth_eq_ratio` — the wealth process is exactly the likelihood ratio of the
  alternative rate against the null rate on the reads seen so far.
* `ville` — **the crossing bound.** Under the null, the probability that the wealth *ever*
  reaches `c`, at any point of a run of any length, is at most `w₀/c`.
* `anytime_valid` — hence stopping the run the moment the likelihood ratio exceeds `1/α`, and
  calling that a refutation, has type-I error at most `α`, *whatever the run length and however
  often the data are inspected*.  `anytime_valid_all_horizons` states that uniformity over `n`.
* `peeking_inflates_error` — the contrast: two looks, each at level `1/2` in the fixed-`n`
  theory, give a combined null error of `3/4`.  Optional stopping is only free for the wealth
  test.
* `kl2`, `expected_logWealth`, `kl2_pos` — the price paid for that freedom: the expected log
  wealth after `n` molecules is exactly `n · KL(q₁‖q₀)`, and the Kullback–Leibler rate is
  strictly positive whenever the alternative rate differs from the null rate.
* `sequentialHorizon`, `sequentialHorizon_spec` — so `⌈log(1/α)/KL⌉` molecules bring the
  expected evidence to the threshold: **logarithmic in `α`**, where the fixed-`n` Chebyshev
  design of Part XC paid `1/(α·Δ²)`.
* `sequential_cheaper` — an explicit inequality making that comparison, with no asymptotics.

Nothing here is a measurement.  What is proved is that a run may be watched and stopped at will
without inflating the error, provided the quantity watched is the likelihood ratio, and what
that costs in molecules.
-/
import Mathlib
import RequestProject.NoisyDetection

set_option autoImplicit false

namespace IDR
namespace Seq

open Finset

/-! ## 1. Runs of binary reads as words -/

/-- All read-out records of exactly `n` binary reads. -/
def words : ℕ → Finset (List Bool)
  | 0 => {[]}
  | n + 1 => (words n).image (fun l => true :: l) ∪ (words n).image (fun l => false :: l)

/-- The probability of the record `l` when each read is positive with probability `q`. -/
noncomputable def probL (q : ℝ) : List Bool → ℝ
  | [] => 1
  | b :: t => Noisy.bern q b * probL q t

@[simp] lemma probL_nil (q : ℝ) : probL q [] = 1 := rfl

@[simp] lemma probL_cons (q : ℝ) (b : Bool) (t : List Bool) :
    probL q (b :: t) = Noisy.bern q b * probL q t := rfl

lemma probL_nonneg {q : ℝ} (h0 : 0 ≤ q) (h1 : q ≤ 1) (l : List Bool) : 0 ≤ probL q l := by
  induction l with
  | nil => simp
  | cons b t ih =>
      exact mul_nonneg (Noisy.bern_nonneg h0 h1 b) ih

/-- Every word in `words n` has length `n`. -/
lemma length_of_mem_words : ∀ {n : ℕ} {l : List Bool}, l ∈ words n → l.length = n := by
  intro n
  induction n with
  | zero => intro l hl; simp [words] at hl; simp [hl]
  | succ n ih =>
      intro l hl
      simp only [words, Finset.mem_union, Finset.mem_image] at hl
      rcases hl with ⟨t, ht, rfl⟩ | ⟨t, ht, rfl⟩ <;> simp [ih ht]

/-- Splitting a sum over runs of `n+1` reads on the first read. -/
lemma sum_words_succ (n : ℕ) (f : List Bool → ℝ) :
    ∑ l ∈ words (n + 1), f l =
      (∑ l ∈ words n, f (true :: l)) + ∑ l ∈ words n, f (false :: l) := by
  have hdisj :
      Disjoint ((words n).image (fun l => true :: l)) ((words n).image (fun l => false :: l)) := by
    refine Finset.disjoint_left.mpr ?_
    intro a ha hb
    simp only [Finset.mem_image] at ha hb
    obtain ⟨t, _, rfl⟩ := ha
    obtain ⟨u, _, hu⟩ := hb
    simp at hu
  have hinj1 : Set.InjOn (fun l : List Bool => true :: l) (words n) := by
    intro a _ b _ h; simpa using h
  have hinj2 : Set.InjOn (fun l : List Bool => false :: l) (words n) := by
    intro a _ b _ h; simpa using h
  rw [words, Finset.sum_union hdisj, Finset.sum_image hinj1, Finset.sum_image hinj2]

/-- The read-out law is a probability distribution on runs of `n` reads. -/
lemma sum_probL (q : ℝ) (n : ℕ) : ∑ l ∈ words n, probL q l = 1 := by
  induction n with
  | zero => simp [words]
  | succ n ih =>
      rw [sum_words_succ]
      simp only [probL_cons, Noisy.bern]
      rw [← Finset.mul_sum, ← Finset.mul_sum, ih]
      norm_num

/-! ## 2. The wealth process

Betting on the alternative rate `q₁` against the null rate `q₀`, a positive read multiplies the
wealth by `q₁/q₀` and a negative read by `(1-q₁)/(1-q₀)`.  This is the likelihood ratio, and it
is a fair bet under the null: its expected multiplier is exactly `1`. -/

/-- The likelihood ratio contributed by a single read. -/
noncomputable def lrOne (q₁ q₀ : ℝ) (b : Bool) : ℝ := Noisy.bern q₁ b / Noisy.bern q₀ b

@[simp] lemma lrOne_true (q₁ q₀ : ℝ) : lrOne q₁ q₀ true = q₁ / q₀ := by simp [lrOne, Noisy.bern]

@[simp] lemma lrOne_false (q₁ q₀ : ℝ) : lrOne q₁ q₀ false = (1 - q₁) / (1 - q₀) := by
  simp [lrOne, Noisy.bern]

/-- The wealth after reading the record `l`, starting from wealth `w`. -/
noncomputable def wealth (q₁ q₀ : ℝ) : ℝ → List Bool → ℝ
  | w, [] => w
  | w, b :: t => wealth q₁ q₀ (w * lrOne q₁ q₀ b) t

@[simp] lemma wealth_nil (q₁ q₀ w : ℝ) : wealth q₁ q₀ w [] = w := rfl

@[simp] lemma wealth_cons (q₁ q₀ w : ℝ) (b : Bool) (t : List Bool) :
    wealth q₁ q₀ w (b :: t) = wealth q₁ q₀ (w * lrOne q₁ q₀ b) t := rfl

lemma bern_ne_zero {q : ℝ} (h0 : 0 < q) (h1 : q < 1) (b : Bool) : Noisy.bern q b ≠ 0 := by
  cases b
  · show (1 - q) ≠ 0
    intro h; linarith
  · show q ≠ 0
    intro h; linarith

lemma probL_ne_zero {q : ℝ} (h0 : 0 < q) (h1 : q < 1) (l : List Bool) : probL q l ≠ 0 := by
  induction l with
  | nil => simp
  | cons b t ih => simpa using mul_ne_zero (bern_ne_zero h0 h1 b) ih

/-- The wealth is the likelihood ratio of the two read-out laws on the data seen so far. -/
lemma wealth_eq_ratio {q₁ q₀ : ℝ} (h0 : 0 < q₀) (h1 : q₀ < 1) (w : ℝ) (l : List Bool) :
    wealth q₁ q₀ w l = w * (probL q₁ l / probL q₀ l) := by
  induction l generalizing w with
  | nil => simp
  | cons b t ih =>
      have hb : Noisy.bern q₀ b ≠ 0 := bern_ne_zero h0 h1 b
      have hpt : probL q₀ t ≠ 0 := probL_ne_zero h0 h1 t
      rw [wealth_cons, ih]
      simp only [probL_cons, lrOne]
      field_simp

/-! ## 3. Crossing, and Ville's inequality -/

/-- `Crossed q₁ q₀ c w l` : starting from wealth `w`, the wealth reaches `c` at some point
during the run `l` — including before any read, and including at the end. -/
def Crossed (q₁ q₀ c : ℝ) : ℝ → List Bool → Prop
  | w, [] => c ≤ w
  | w, b :: t => c ≤ w ∨ Crossed q₁ q₀ c (w * lrOne q₁ q₀ b) t

open Classical in
/-- The indicator of the crossing event. -/
noncomputable def crossInd (q₁ q₀ c w : ℝ) (l : List Bool) : ℝ :=
  if Crossed q₁ q₀ c w l then 1 else 0

lemma crossInd_nonneg (q₁ q₀ c w : ℝ) (l : List Bool) : 0 ≤ crossInd q₁ q₀ c w l := by
  unfold crossInd; split <;> norm_num

lemma crossInd_of_le {q₁ q₀ c w : ℝ} (h : c ≤ w) (l : List Bool) :
    crossInd q₁ q₀ c w l = 1 := by
  unfold crossInd
  have : Crossed q₁ q₀ c w l := by cases l with
    | nil => exact h
    | cons b t => exact Or.inl h
  simp [this]

lemma crossInd_cons_of_not_le {q₁ q₀ c w : ℝ} (h : ¬ c ≤ w) (b : Bool) (t : List Bool) :
    crossInd q₁ q₀ c w (b :: t) = crossInd q₁ q₀ c (w * lrOne q₁ q₀ b) t := by
  unfold crossInd
  have hiff : Crossed q₁ q₀ c w (b :: t) ↔ Crossed q₁ q₀ c (w * lrOne q₁ q₀ b) t :=
    ⟨fun hc => hc.resolve_left h, fun hc => Or.inr hc⟩
  by_cases hc : Crossed q₁ q₀ c (w * lrOne q₁ q₀ b) t
  · rw [if_pos (hiff.mpr hc), if_pos hc]
  · rw [if_neg (fun hx => hc (hiff.mp hx)), if_neg hc]

/-- **Ville's inequality.** Under the null rate `q₀`, the probability that the wealth of the
likelihood-ratio bet ever reaches `c` during a run of `n` reads is at most `w/c`, where `w` is
the starting wealth.  No stopping rule, however chosen, evades this bound. -/
theorem ville {q₁ q₀ : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < q₀) (h01 : q₀ < 1)
    {c : ℝ} (hc : 0 < c) :
    ∀ (n : ℕ) (w : ℝ), 0 ≤ w → ∑ l ∈ words n, probL q₀ l * crossInd q₁ q₀ c w l ≤ w / c := by
  intro n
  induction n with
  | zero =>
      intro w hw
      simp only [words, Finset.sum_singleton, probL_nil, one_mul]
      by_cases h : c ≤ w
      · rw [crossInd_of_le h]
        rw [le_div_iff₀ hc]; linarith
      · unfold crossInd
        have : ¬ Crossed q₁ q₀ c w ([] : List Bool) := h
        simp [this]
        positivity
  | succ n ih =>
      intro w hw
      rw [sum_words_succ]
      by_cases h : c ≤ w
      · have e1 : ∀ b : Bool, ∀ l ∈ words n,
            probL q₀ (b :: l) * crossInd q₁ q₀ c w (b :: l) = Noisy.bern q₀ b * probL q₀ l := by
          intro b l _
          rw [crossInd_of_le h, probL_cons]; ring
        rw [Finset.sum_congr rfl (e1 true), Finset.sum_congr rfl (e1 false),
          ← Finset.mul_sum, ← Finset.mul_sum, sum_probL]
        simp only [Noisy.bern]
        rw [le_div_iff₀ hc]
        norm_num
        linarith
      · have hlr1 : 0 ≤ lrOne q₁ q₀ true := by
          rw [lrOne_true]; positivity
        have hlr0 : 0 ≤ lrOne q₁ q₀ false := by
          rw [lrOne_false]
          have : (0:ℝ) ≤ 1 - q₁ := by linarith
          have : (0:ℝ) < 1 - q₀ := by linarith
          positivity
        have e1 : ∀ b : Bool, ∀ l ∈ words n,
            probL q₀ (b :: l) * crossInd q₁ q₀ c w (b :: l) =
              Noisy.bern q₀ b * (probL q₀ l * crossInd q₁ q₀ c (w * lrOne q₁ q₀ b) l) := by
          intro b l _
          rw [crossInd_cons_of_not_le h, probL_cons]; ring
        rw [Finset.sum_congr rfl (e1 true), Finset.sum_congr rfl (e1 false),
          ← Finset.mul_sum, ← Finset.mul_sum]
        have b1 := ih (w * lrOne q₁ q₀ true) (by positivity)
        have b0 := ih (w * lrOne q₁ q₀ false) (by positivity)
        have hq0 : (0:ℝ) ≤ Noisy.bern q₀ true := by simp [Noisy.bern]; linarith
        have hq1 : (0:ℝ) ≤ Noisy.bern q₀ false := by simp [Noisy.bern]; linarith
        have step :
            Noisy.bern q₀ true * (∑ l ∈ words n, probL q₀ l * crossInd q₁ q₀ c (w * lrOne q₁ q₀ true) l)
              + Noisy.bern q₀ false * (∑ l ∈ words n, probL q₀ l * crossInd q₁ q₀ c (w * lrOne q₁ q₀ false) l)
              ≤ Noisy.bern q₀ true * ((w * lrOne q₁ q₀ true) / c)
                + Noisy.bern q₀ false * ((w * lrOne q₁ q₀ false) / c) := by
          gcongr
        refine step.trans_eq ?_
        have hbt : Noisy.bern q₀ true = q₀ := by simp [Noisy.bern]
        have hbf : Noisy.bern q₀ false = 1 - q₀ := by simp [Noisy.bern]
        rw [hbt, hbf, lrOne_true, lrOne_false]
        have hne0 : q₀ ≠ 0 := ne_of_gt h00
        have hne1 : (1 : ℝ) - q₀ ≠ 0 := by linarith
        field_simp
        ring

/-- **Anytime validity.** Starting from wealth `1` and stopping the run as soon as the
likelihood ratio reaches `1/α`, the null probability of ever stopping — at any point of a run of
`n` molecules — is at most `α`. -/
theorem anytime_valid {q₁ q₀ α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < q₀) (h01 : q₀ < 1)
    (hα : 0 < α) (n : ℕ) :
    ∑ l ∈ words n, probL q₀ l * crossInd q₁ q₀ (1 / α) 1 l ≤ α := by
  have hc : 0 < 1 / α := by positivity
  have := ville h10 h11 h00 h01 hc n 1 zero_le_one
  calc ∑ l ∈ words n, probL q₀ l * crossInd q₁ q₀ (1 / α) 1 l ≤ 1 / (1 / α) := this
    _ = α := by field_simp

/-- The bound is uniform in the horizon: it holds for every run length at once, so the
experimenter need not fix `n` in advance. -/
theorem anytime_valid_all_horizons {q₁ q₀ α : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (h00 : 0 < q₀)
    (h01 : q₀ < 1) (hα : 0 < α) :
    ∀ n : ℕ, ∑ l ∈ words n, probL q₀ l * crossInd q₁ q₀ (1 / α) 1 l ≤ α :=
  fun n => anytime_valid h10 h11 h00 h01 hα n

/-- A run that ends above the threshold has crossed it, so the fixed-horizon likelihood-ratio
test is a special case of the sequential one. -/
lemma crossed_of_final {q₁ q₀ c : ℝ} :
    ∀ (l : List Bool) (w : ℝ), c ≤ wealth q₁ q₀ w l → Crossed q₁ q₀ c w l := by
  intro l
  induction l with
  | nil => intro w h; exact h
  | cons b t ih => intro w h; exact Or.inr (ih _ (by simpa using h))

/-! ## 4. What the freedom costs: the evidence rate -/

/-- The Kullback–Leibler divergence between two Bernoulli read-out rates. -/
noncomputable def kl2 (q₁ q₀ : ℝ) : ℝ :=
  q₁ * Real.log (q₁ / q₀) + (1 - q₁) * Real.log ((1 - q₁) / (1 - q₀))

/-- The log wealth accumulated over a record. -/
noncomputable def logWealth (q₁ q₀ : ℝ) : List Bool → ℝ
  | [] => 0
  | b :: t => Real.log (lrOne q₁ q₀ b) + logWealth q₁ q₀ t

@[simp] lemma logWealth_nil (q₁ q₀ : ℝ) : logWealth q₁ q₀ [] = 0 := rfl

@[simp] lemma logWealth_cons (q₁ q₀ : ℝ) (b : Bool) (t : List Bool) :
    logWealth q₁ q₀ (b :: t) = Real.log (lrOne q₁ q₀ b) + logWealth q₁ q₀ t := rfl

lemma logWealth_eq_log_wealth {q₁ q₀ : ℝ} (h10 : 0 < q₁) (h11 : q₁ < 1) (h00 : 0 < q₀)
    (h01 : q₀ < 1) (l : List Bool) :
    logWealth q₁ q₀ l = Real.log (wealth q₁ q₀ 1 l) := by
  have hpos : ∀ b : Bool, 0 < lrOne q₁ q₀ b := by
    intro b
    cases b
    · rw [lrOne_false]
      have h1 : (0:ℝ) < 1 - q₁ := by linarith
      have h2 : (0:ℝ) < 1 - q₀ := by linarith
      positivity
    · rw [lrOne_true]; positivity
  have key : ∀ (l : List Bool) (w : ℝ), 0 < w →
      Real.log (wealth q₁ q₀ w l) = Real.log w + logWealth q₁ q₀ l := by
    intro l
    induction l with
    | nil => intro w _; simp
    | cons b t ih =>
        intro w hw
        rw [wealth_cons, ih _ (by exact mul_pos hw (hpos b)),
          Real.log_mul (ne_of_gt hw) (ne_of_gt (hpos b))]
        simp [logWealth]
        ring
  rw [key l 1 one_pos]
  simp

/-- **The evidence rate.** Under the alternative rate `q₁`, the expected log wealth after `n`
molecules is exactly `n` times the Kullback–Leibler divergence of the alternative from the null.
Evidence accumulates linearly in the number of molecules, at a rate fixed by the instrument. -/
theorem expected_logWealth (q₁ q₀ : ℝ) (n : ℕ) :
    ∑ l ∈ words n, probL q₁ l * logWealth q₁ q₀ l = n * kl2 q₁ q₀ := by
  induction n with
  | zero => simp [words]
  | succ n ih =>
      rw [sum_words_succ]
      have e : ∀ b : Bool, ∀ l ∈ words n,
          probL q₁ (b :: l) * logWealth q₁ q₀ (b :: l) =
            Noisy.bern q₁ b * Real.log (lrOne q₁ q₀ b) * probL q₁ l
              + Noisy.bern q₁ b * (probL q₁ l * logWealth q₁ q₀ l) := by
        intro b l _; simp [probL, logWealth]; ring
      rw [Finset.sum_congr rfl (e true), Finset.sum_congr rfl (e false)]
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
        ← Finset.mul_sum, ← Finset.mul_sum, sum_probL, ih]
      simp only [Noisy.bern, lrOne_true, lrOne_false]
      push_cast
      unfold kl2
      ring

/-- The evidence rate is strictly positive whenever the alternative read-out rate differs from
the null one: every molecule carries information. -/
theorem kl2_pos {q₁ q₀ : ℝ} (h10 : 0 < q₁) (h11 : q₁ < 1) (h00 : 0 < q₀) (h01 : q₀ < 1)
    (hne : q₁ ≠ q₀) : 0 < kl2 q₁ q₀ := by
  have hx : (0:ℝ) < q₀ / q₁ := by positivity
  have hy : (0:ℝ) < (1 - q₀) / (1 - q₁) := by
    have h1 : (0:ℝ) < 1 - q₁ := by linarith
    have h2 : (0:ℝ) < 1 - q₀ := by linarith
    positivity
  have hxne : q₀ / q₁ ≠ 1 := by
    intro h
    apply hne
    field_simp at h
    linarith
  have hlx : Real.log (q₀ / q₁) < q₀ / q₁ - 1 := Real.log_lt_sub_one_of_pos hx hxne
  have hly : Real.log ((1 - q₀) / (1 - q₁)) ≤ (1 - q₀) / (1 - q₁) - 1 :=
    Real.log_le_sub_one_of_pos hy
  have e1 : Real.log (q₁ / q₀) = - Real.log (q₀ / q₁) := by
    rw [← Real.log_inv]
    congr 1
    field_simp
  have e2 : Real.log ((1 - q₁) / (1 - q₀)) = - Real.log ((1 - q₀) / (1 - q₁)) := by
    rw [← Real.log_inv]
    congr 1
    have h1 : (1:ℝ) - q₁ ≠ 0 := by intro h; linarith
    have h2 : (1:ℝ) - q₀ ≠ 0 := by intro h; linarith
    field_simp
  have hq1 : (0:ℝ) < 1 - q₁ := by linarith
  have s1 : q₁ * Real.log (q₀ / q₁) < q₁ * (q₀ / q₁ - 1) :=
    mul_lt_mul_of_pos_left hlx h10
  have s2 : (1 - q₁) * Real.log ((1 - q₀) / (1 - q₁)) ≤ (1 - q₁) * ((1 - q₀) / (1 - q₁) - 1) :=
    by exact mul_le_mul_of_nonneg_left hly (le_of_lt hq1)
  have hsum : q₁ * (q₀ / q₁ - 1) + (1 - q₁) * ((1 - q₀) / (1 - q₁) - 1) = 0 := by
    field_simp
    ring
  unfold kl2
  rw [e1, e2]
  nlinarith [s1, s2, hsum]

/-- The number of molecules at which the expected evidence reaches the threshold `log(1/α)`. -/
noncomputable def sequentialHorizon (α kl : ℝ) : ℕ := ⌈Real.log (1 / α) / kl⌉₊

/-- At the sequential horizon the expected log wealth is at least `log(1/α)`: the run is
expected to have crossed the anytime-valid threshold. -/
theorem sequentialHorizon_spec {α q₁ q₀ : ℝ}
    (hkl : 0 < kl2 q₁ q₀) {n : ℕ} (hn : sequentialHorizon α (kl2 q₁ q₀) ≤ n) :
    Real.log (1 / α) ≤ ∑ l ∈ words n, probL q₁ l * logWealth q₁ q₀ l := by
  rw [expected_logWealth]
  have h1 : Real.log (1 / α) / kl2 q₁ q₀ ≤ (n : ℝ) := by
    refine le_trans (Nat.le_ceil _) ?_
    exact_mod_cast Nat.cast_le.2 hn
  rw [div_le_iff₀ hkl] at h1
  linarith

/-- **The sequential design is cheaper than the fixed-`n` one at small `α`.** The sequential
horizon `log(1/α)/kl` (rounded up, hence the `+1`) is below the Chebyshev sample size
`1/(α·Δ²)` of Part XC once `α` is small — precisely, once `√α ≤ kl/(4Δ²)`.  The two designs
scale differently in the confidence level: logarithmically against reciprocally. -/
theorem sequential_cheaper {kl Δ α : ℝ} (hkl : 0 < kl) (hΔ0 : 0 < Δ) (hΔ1 : Δ ≤ 1)
    (hα0 : 0 < α) (hα1 : α ≤ 1 / 2) (hsmall : Real.sqrt α ≤ kl / (4 * Δ ^ 2)) :
    Real.log (1 / α) / kl + 1 ≤ 1 / (α * Δ ^ 2) := by
  set s := Real.sqrt α with hs_def
  have hs : 0 < s := Real.sqrt_pos.2 hα0
  have hs2 : s ^ 2 = α := Real.sq_sqrt (le_of_lt hα0)
  -- `log (1/α) ≤ 2/s`
  have hlog : Real.log (1 / α) ≤ 2 / s := by
    have h1 : (1 : ℝ) / α = (1 / s) ^ 2 := by
      rw [div_pow, one_pow, hs2]
    have h2 : Real.log (1 / α) = 2 * Real.log (1 / s) := by
      rw [h1, Real.log_pow]; push_cast; ring
    have h3 : Real.log (1 / s) ≤ 1 / s - 1 := Real.log_le_sub_one_of_pos (by positivity)
    have : 2 * Real.log (1 / s) ≤ 2 * (1 / s - 1) := by linarith
    have h4 : 2 * (1 / s - 1) ≤ 2 / s := by
      rw [mul_sub, mul_one_div]
      linarith
    linarith [h2 ▸ this]
  have hΔ2 : 0 < Δ ^ 2 := by positivity
  -- the two halves of the budget
  have hhalf1 : 2 / (s * kl) ≤ 1 / (2 * (α * Δ ^ 2)) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    have h : 4 * Δ ^ 2 * s ≤ kl := by
      rw [le_div_iff₀ (by positivity)] at hsmall
      linarith [hsmall]
    nlinarith [mul_le_mul_of_nonneg_left h (le_of_lt hs), hs2]
  have hhalf2 : (1 : ℝ) ≤ 1 / (2 * (α * Δ ^ 2)) := by
    rw [le_div_iff₀ (by positivity)]
    nlinarith [sq_nonneg Δ]
  have hdiv : Real.log (1 / α) / kl ≤ 2 / (s * kl) := by
    rw [div_le_div_iff₀ hkl (by positivity)]
    have : Real.log (1 / α) * s ≤ 2 / s * s := by
      exact mul_le_mul_of_nonneg_right hlog (le_of_lt hs)
    rw [div_mul_cancel₀ _ (ne_of_gt hs)] at this
    nlinarith [this]
  have hsum : 1 / (2 * (α * Δ ^ 2)) + 1 / (2 * (α * Δ ^ 2)) = 1 / (α * Δ ^ 2) := by
    field_simp
    ring
  linarith

/-! ## 5. Why the wealth statistic, and not a repeated fixed-`n` test -/

lemma sum_words_two (f : List Bool → ℝ) :
    ∑ l ∈ words 2, f l =
      f [true, true] + f [true, false] + (f [false, true] + f [false, false]) := by
  rw [sum_words_succ, sum_words_succ, sum_words_succ]
  simp [words]

/-- **Peeking inflates the error.** With a fair null coin, "reject if the first read is
positive" and "reject if the second read is positive" each have null error exactly `1/2`, but
running both — that is, looking after one molecule and again after two — has null error `3/4`.
A fixed-`n` test does not survive being watched; the wealth test of `anytime_valid` does. -/
theorem peeking_inflates_error :
    (∑ l ∈ words 2, probL (1/2) l * (if l.head? = some true then 1 else 0) = 1/2) ∧
    (∑ l ∈ words 2, probL (1/2) l * (if l.tail.head? = some true then 1 else 0) = 1/2) ∧
    (∑ l ∈ words 2, probL (1/2) l * (if l.any id then 1 else 0) = 3/4) := by
  refine ⟨?_, ?_, ?_⟩ <;>
  · rw [sum_words_two]
    norm_num [probL, Noisy.bern]

end Seq
end IDR
