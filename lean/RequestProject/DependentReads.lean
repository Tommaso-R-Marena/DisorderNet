/-
# Part CIX  Stopping when you like, with reads that are not independent

Part XCII makes stopping free: the wealth of a likelihood-ratio bet obeys Ville's inequality, so a
run may be watched and stopped at will.  The assumptions list recorded the two hypotheses it did
not remove — "the reads are independent, and the null read rate `1 − sp` is known".  The first is
false in every real experiment: molecules arrive in bursts, the instrument drifts, and a positive
read makes the next read more likely.  This file removes it.

The null is now specified not by a single rate but by a **predictable conditional rate**
`q₀ : List Bool → ℝ`: after any history of reads, the probability that the next read is positive
may depend on that history in an arbitrary way.  Nothing is assumed about the dependence — no
mixing, no exchangeability, no bound on the memory.  The alternative `q₁` is another such
conditional rate, so the bet may itself be history dependent.

* `sum_probD` — the conditional rates define a probability law on runs of any length, from any
  history.
* `wealthD_eq_ratio` — the wealth is exactly the likelihood ratio of the two conditional laws on
  the data seen so far, as in the independent case.
* `villeD` — **Ville's inequality survives the dependence.**  Under the null, the probability that
  the wealth ever reaches `c`, at any point of a run of any length, is at most `w₀/c`.
* `anytime_validD` — hence the test that stops the moment the likelihood ratio exceeds `1/α` has
  type-I error at most `α`, for arbitrarily dependent reads, at every horizon
  (`anytime_validD_all_horizons`).
* `ville_bursty_reads` — an instance with reads that are strongly dependent by construction: a
  positive read makes the next read positive with probability `9/10`, a negative read with
  probability `1/10`.  Bursty data of exactly the kind that breaks a fixed-`n` design leaves the
  sequential guarantee untouched.

What is *not* repaired is the second hypothesis: the conditional null rates must be known.  The
calibration floor of Part XC is unaffected by this part, exactly as it is unaffected by the
multiplicity control of Part XCIII — a wealth process built on a miscalibrated null is a bet on an
instrument artefact, however the reads are correlated.
-/
import Mathlib
import RequestProject.Sequential

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace DepSeq

open Finset
open IDR.Seq (words sum_words_succ)

/-! ## Runs with a predictable conditional rate -/

/-- The probability of the record `l` when, after the history `h` (most recent read first), the
next read is positive with probability `q h`. -/
noncomputable def probD (q : List Bool → ℝ) : List Bool → List Bool → ℝ
  | _, [] => 1
  | h, b :: t => Noisy.bern (q h) b * probD q (b :: h) t

@[simp] lemma probD_nil (q : List Bool → ℝ) (h : List Bool) : probD q h [] = 1 := rfl

@[simp] lemma probD_cons (q : List Bool → ℝ) (h : List Bool) (b : Bool) (t : List Bool) :
    probD q h (b :: t) = Noisy.bern (q h) b * probD q (b :: h) t := rfl

lemma probD_nonneg {q : List Bool → ℝ} (h0 : ∀ h, 0 ≤ q h) (h1 : ∀ h, q h ≤ 1)
    (h : List Bool) (l : List Bool) : 0 ≤ probD q h l := by
  induction l generalizing h with
  | nil => simp
  | cons b t ih =>
      exact mul_nonneg (Noisy.bern_nonneg (h0 h) (h1 h) b) (ih (b :: h))

/-- **The conditional rates define a probability law on runs of `n` reads**, from any history. -/
lemma sum_probD (q : List Bool → ℝ) (n : ℕ) :
    ∀ h : List Bool, ∑ l ∈ words n, probD q h l = 1 := by
  induction n with
  | zero => intro h; simp [words]
  | succ n ih =>
      intro h
      rw [sum_words_succ]
      simp only [probD_cons, Noisy.bern]
      rw [← Finset.mul_sum, ← Finset.mul_sum, ih (true :: h), ih (false :: h)]
      norm_num

/-! ## The wealth process -/

/-- The likelihood ratio contributed by one read, at the current history. -/
noncomputable def lrD (q₁ q₀ : List Bool → ℝ) (h : List Bool) (b : Bool) : ℝ :=
  Noisy.bern (q₁ h) b / Noisy.bern (q₀ h) b

/-- The wealth after reading the record `l` from history `h`, starting from wealth `w`. -/
noncomputable def wealthD (q₁ q₀ : List Bool → ℝ) : ℝ → List Bool → List Bool → ℝ
  | w, _, [] => w
  | w, h, b :: t => wealthD q₁ q₀ (w * lrD q₁ q₀ h b) (b :: h) t

@[simp] lemma wealthD_nil (q₁ q₀ : List Bool → ℝ) (w : ℝ) (h : List Bool) :
    wealthD q₁ q₀ w h [] = w := rfl

@[simp] lemma wealthD_cons (q₁ q₀ : List Bool → ℝ) (w : ℝ) (h : List Bool) (b : Bool)
    (t : List Bool) :
    wealthD q₁ q₀ w h (b :: t) = wealthD q₁ q₀ (w * lrD q₁ q₀ h b) (b :: h) t := rfl

lemma bernD_ne_zero {r : ℝ} (h0 : 0 < r) (h1 : r < 1) (b : Bool) : Noisy.bern r b ≠ 0 := by
  cases b
  · show (1 - r) ≠ 0
    intro hh; linarith
  · show r ≠ 0
    intro hh; linarith

lemma probD_ne_zero {q : List Bool → ℝ} (h0 : ∀ h, 0 < q h) (h1 : ∀ h, q h < 1)
    (h : List Bool) (l : List Bool) : probD q h l ≠ 0 := by
  induction l generalizing h with
  | nil => simp
  | cons b t ih =>
      exact mul_ne_zero (bernD_ne_zero (h0 h) (h1 h) b) (ih (b :: h))

/-- **The wealth is the likelihood ratio of the two conditional laws** on the data seen so far. -/
lemma wealthD_eq_ratio {q₁ q₀ : List Bool → ℝ} (h0 : ∀ h, 0 < q₀ h) (h1 : ∀ h, q₀ h < 1)
    (w : ℝ) (h : List Bool) (l : List Bool) :
    wealthD q₁ q₀ w h l = w * (probD q₁ h l / probD q₀ h l) := by
  induction l generalizing w h with
  | nil => simp
  | cons b t ih =>
      have hb : Noisy.bern (q₀ h) b ≠ 0 := bernD_ne_zero (h0 h) (h1 h) b
      have hpt : probD q₀ (b :: h) t ≠ 0 := probD_ne_zero h0 h1 (b :: h) t
      rw [wealthD_cons, ih]
      simp only [probD_cons, lrD]
      field_simp

/-! ## Ville's inequality under arbitrary dependence -/

/-- `CrossedD q₁ q₀ c w h l` : starting from wealth `w` at history `h`, the wealth reaches `c` at
some point during the run `l`. -/
def CrossedD (q₁ q₀ : List Bool → ℝ) (c : ℝ) : ℝ → List Bool → List Bool → Prop
  | w, _, [] => c ≤ w
  | w, h, b :: t => c ≤ w ∨ CrossedD q₁ q₀ c (w * lrD q₁ q₀ h b) (b :: h) t

open Classical in
/-- The indicator of the crossing event. -/
noncomputable def crossIndD (q₁ q₀ : List Bool → ℝ) (c w : ℝ) (h l : List Bool) : ℝ :=
  if CrossedD q₁ q₀ c w h l then 1 else 0

lemma crossIndD_nonneg (q₁ q₀ : List Bool → ℝ) (c w : ℝ) (h l : List Bool) :
    0 ≤ crossIndD q₁ q₀ c w h l := by
  unfold crossIndD; split <;> norm_num

lemma crossIndD_of_le {q₁ q₀ : List Bool → ℝ} {c w : ℝ} (hle : c ≤ w) (h l : List Bool) :
    crossIndD q₁ q₀ c w h l = 1 := by
  unfold crossIndD
  have : CrossedD q₁ q₀ c w h l := by
    cases l with
    | nil => exact hle
    | cons b t => exact Or.inl hle
  simp [this]

lemma crossIndD_cons_of_not_le {q₁ q₀ : List Bool → ℝ} {c w : ℝ} (hn : ¬ c ≤ w)
    (h : List Bool) (b : Bool) (t : List Bool) :
    crossIndD q₁ q₀ c w h (b :: t) = crossIndD q₁ q₀ c (w * lrD q₁ q₀ h b) (b :: h) t := by
  unfold crossIndD
  have hiff : CrossedD q₁ q₀ c w h (b :: t)
      ↔ CrossedD q₁ q₀ c (w * lrD q₁ q₀ h b) (b :: h) t :=
    ⟨fun hc => hc.resolve_left hn, fun hc => Or.inr hc⟩
  by_cases hc : CrossedD q₁ q₀ c (w * lrD q₁ q₀ h b) (b :: h) t
  · rw [if_pos (hiff.mpr hc), if_pos hc]
  · rw [if_neg (fun hx => hc (hiff.mp hx)), if_neg hc]

/-- **Ville's inequality with arbitrarily dependent reads.**  Under a null specified by predictable
conditional rates, the probability that the wealth of the likelihood-ratio bet ever reaches `c`
during a run of `n` reads is at most `w/c`. -/
theorem villeD {q₁ q₀ : List Bool → ℝ} (h10 : ∀ h, 0 ≤ q₁ h) (h11 : ∀ h, q₁ h ≤ 1)
    (h00 : ∀ h, 0 < q₀ h) (h01 : ∀ h, q₀ h < 1) {c : ℝ} (hc : 0 < c) :
    ∀ (n : ℕ) (w : ℝ), 0 ≤ w →
      ∀ h : List Bool, ∑ l ∈ words n, probD q₀ h l * crossIndD q₁ q₀ c w h l ≤ w / c := by
  intro n
  induction n with
  | zero =>
      intro w hw h
      simp only [words, Finset.sum_singleton, probD_nil, one_mul]
      by_cases hle : c ≤ w
      · rw [crossIndD_of_le hle]
        rw [le_div_iff₀ hc]; linarith
      · unfold crossIndD
        have hnc : ¬ CrossedD q₁ q₀ c w h ([] : List Bool) := hle
        simp [hnc]
        positivity
  | succ n ih =>
      intro w hw h
      rw [sum_words_succ]
      by_cases hle : c ≤ w
      · have e1 : ∀ b : Bool, ∀ l ∈ words n,
            probD q₀ h (b :: l) * crossIndD q₁ q₀ c w h (b :: l)
              = Noisy.bern (q₀ h) b * probD q₀ (b :: h) l := by
          intro b l _
          rw [crossIndD_of_le hle, probD_cons]; ring
        rw [Finset.sum_congr rfl (e1 true), Finset.sum_congr rfl (e1 false),
          ← Finset.mul_sum, ← Finset.mul_sum, sum_probD q₀ n (true :: h),
          sum_probD q₀ n (false :: h)]
        simp only [Noisy.bern]
        rw [le_div_iff₀ hc]
        norm_num
        linarith
      · have hlr1 : 0 ≤ lrD q₁ q₀ h true := by
          have hn : (0:ℝ) ≤ q₁ h := h10 h
          have hd : (0:ℝ) < q₀ h := h00 h
          simp only [lrD, Noisy.bern]
          positivity
        have hlr0 : 0 ≤ lrD q₁ q₀ h false := by
          have hn : (0:ℝ) ≤ 1 - q₁ h := by linarith [h11 h]
          have hd : (0:ℝ) < 1 - q₀ h := by linarith [h01 h]
          simp only [lrD, Noisy.bern]
          positivity
        have e1 : ∀ b : Bool, ∀ l ∈ words n,
            probD q₀ h (b :: l) * crossIndD q₁ q₀ c w h (b :: l)
              = Noisy.bern (q₀ h) b
                  * (probD q₀ (b :: h) l * crossIndD q₁ q₀ c (w * lrD q₁ q₀ h b) (b :: h) l) := by
          intro b l _
          rw [crossIndD_cons_of_not_le hle, probD_cons]; ring
        rw [Finset.sum_congr rfl (e1 true), Finset.sum_congr rfl (e1 false),
          ← Finset.mul_sum, ← Finset.mul_sum]
        have b1 := ih (w * lrD q₁ q₀ h true) (by positivity) (true :: h)
        have b0 := ih (w * lrD q₁ q₀ h false) (by positivity) (false :: h)
        have hq0 : (0:ℝ) ≤ Noisy.bern (q₀ h) true := by
          show (0:ℝ) ≤ q₀ h
          linarith [h00 h]
        have hq1 : (0:ℝ) ≤ Noisy.bern (q₀ h) false := by
          show (0:ℝ) ≤ 1 - q₀ h
          linarith [h01 h]
        have step :
            Noisy.bern (q₀ h) true
                * (∑ l ∈ words n,
                    probD q₀ (true :: h) l * crossIndD q₁ q₀ c (w * lrD q₁ q₀ h true) (true :: h) l)
              + Noisy.bern (q₀ h) false
                * (∑ l ∈ words n,
                    probD q₀ (false :: h) l
                      * crossIndD q₁ q₀ c (w * lrD q₁ q₀ h false) (false :: h) l)
              ≤ Noisy.bern (q₀ h) true * ((w * lrD q₁ q₀ h true) / c)
                + Noisy.bern (q₀ h) false * ((w * lrD q₁ q₀ h false) / c) := by
          gcongr
        refine step.trans ?_
        have hbt : Noisy.bern (q₀ h) true = q₀ h := rfl
        have hbf : Noisy.bern (q₀ h) false = 1 - q₀ h := rfl
        have hq0ne : q₀ h ≠ 0 := (h00 h).ne'
        have hq1ne : (1 : ℝ) - q₀ h ≠ 0 := by linarith [h01 h]
        have hlrtdef : lrD q₁ q₀ h true = q₁ h / q₀ h := rfl
        have hlrfdef : lrD q₁ q₀ h false = (1 - q₁ h) / (1 - q₀ h) := rfl
        have hlrt : Noisy.bern (q₀ h) true * lrD q₁ q₀ h true = q₁ h := by
          rw [hbt, hlrtdef]
          field_simp
        have hlrf : Noisy.bern (q₀ h) false * lrD q₁ q₀ h false = 1 - q₁ h := by
          rw [hbf, hlrfdef]
          field_simp
        have hexp :
            Noisy.bern (q₀ h) true * ((w * lrD q₁ q₀ h true) / c)
              + Noisy.bern (q₀ h) false * ((w * lrD q₁ q₀ h false) / c)
              = w * (Noisy.bern (q₀ h) true * lrD q₁ q₀ h true
                  + Noisy.bern (q₀ h) false * lrD q₁ q₀ h false) / c := by
          field_simp
        rw [hexp, hlrt, hlrf]
        have : q₁ h + (1 - q₁ h) = 1 := by ring
        rw [this, mul_one]

/-- **Anytime validity with dependent reads.**  Stopping the run the moment the likelihood ratio
exceeds `1/α` and calling that a refutation has type-I error at most `α`. -/
theorem anytime_validD {q₁ q₀ : List Bool → ℝ} (h10 : ∀ h, 0 ≤ q₁ h) (h11 : ∀ h, q₁ h ≤ 1)
    (h00 : ∀ h, 0 < q₀ h) (h01 : ∀ h, q₀ h < 1) {alpha : ℝ} (ha : 0 < alpha) (n : ℕ) :
    ∑ l ∈ words n, probD q₀ [] l * crossIndD q₁ q₀ (1 / alpha) 1 [] l ≤ alpha := by
  have hc : 0 < 1 / alpha := by positivity
  have := villeD h10 h11 h00 h01 hc n 1 zero_le_one []
  calc ∑ l ∈ words n, probD q₀ [] l * crossIndD q₁ q₀ (1 / alpha) 1 [] l
      ≤ 1 / (1 / alpha) := this
    _ = alpha := by field_simp

/-- The guarantee is uniform over the horizon. -/
theorem anytime_validD_all_horizons {q₁ q₀ : List Bool → ℝ} (h10 : ∀ h, 0 ≤ q₁ h)
    (h11 : ∀ h, q₁ h ≤ 1) (h00 : ∀ h, 0 < q₀ h) (h01 : ∀ h, q₀ h < 1) {alpha : ℝ}
    (ha : 0 < alpha) :
    ∀ n : ℕ, ∑ l ∈ words n, probD q₀ [] l * crossIndD q₁ q₀ (1 / alpha) 1 [] l ≤ alpha :=
  fun n => anytime_validD h10 h11 h00 h01 ha n

/-! ## Bursty reads -/

/-- A strongly dependent read law: after a positive read the next read is positive with
probability `9/10`, after a negative one with probability `1/10`. -/
noncomputable def burstyRate : List Bool → ℝ := fun h =>
  match h with
  | true :: _ => 9 / 10
  | _ => 1 / 10

/-- **Bursty, strongly correlated reads leave the sequential guarantee untouched.** -/
theorem ville_bursty_reads {q₁ : List Bool → ℝ} (h10 : ∀ h, 0 ≤ q₁ h) (h11 : ∀ h, q₁ h ≤ 1)
    {alpha : ℝ} (ha : 0 < alpha) (n : ℕ) :
    ∑ l ∈ words n, probD burstyRate [] l * crossIndD q₁ burstyRate (1 / alpha) 1 [] l ≤ alpha := by
  refine anytime_validD h10 h11 (fun h => ?_) (fun h => ?_) ha n <;>
    · unfold burstyRate
      match h with
      | true :: _ => norm_num
      | false :: _ => norm_num
      | [] => norm_num

end DepSeq
end IDR
