/-
# Part XC.4  A converse: how few molecules can *never* be enough

Everything in `RequestProject.NoisyDetection` is an upper bound: a particular test, the midpoint
count, achieves error `α` at `⌈1/(α·Δ²)⌉` molecules.  An upper bound invites the obvious
objection — perhaps a cleverer analysis, a likelihood ratio, a neural read-out, a Bayesian
posterior, would need far fewer.  This file closes that gap from the other side, and the
statement is about *every* analysis whatsoever.

The tool is the total-variation distance between the two data laws, bounded by the hybrid
(telescoping) argument, all of it finite and explicit:

* `l1_dist_le` — the `ℓ¹` distance between `n`-fold product read-out laws at rates `q₀` and `q₁`
  is at most `2·n·|q₁ - q₀|`.  Proved by induction on `n` from the one-read distance; no coupling
  machinery, no measure theory.
* `test_error_sum_ge` — hence for **any** decision rule `T` on the data, at any sample size, the
  sum of its two error probabilities is at least `1 - n·Δ`.  No test, however sophisticated,
  escapes this: it is a property of the data, not of the analysis.
* `molecules_lower_bound` — consequently a study that claims both errors at most `α < 1/2` must
  have observed at least `(1 - 2α)/Δ` molecules.  With a realistic reporter `Δ = τ·(se+sp-1)`, so
  the floor is `(1-2α)/(τ·J)`: **the same Youden index that degrades the achievable design also
  raises the information-theoretic floor**, and no analysis method recovers what the reporter
  threw away.
* `detection_window` — the two bounds side by side: somewhere between `(1-2α)/Δ` and
  `⌈1/(α·Δ²)⌉` molecules, the question is decidable; below the first number it is not decidable
  by anyone.

This is the piece that makes the sample sizes of Part XC a statement about the experiment rather
than about one particular statistic.
-/
import Mathlib
import RequestProject.NoisyDetection

set_option autoImplicit false

namespace IDR
namespace Lower

open Finset Noisy

/-! ## 1. The `ℓ¹` distance between the two data laws -/

/-- The `ℓ¹` distance between the `n`-fold read-out laws at rates `q₀` and `q₁` (twice their
total-variation distance). -/
noncomputable def l1Dist (q₀ q₁ : ℝ) (n : ℕ) : ℝ :=
  ∑ s : Fin n → Bool, |recProb q₁ s - recProb q₀ s|

/-- One read: the `ℓ¹` distance is exactly twice the rate difference. -/
lemma l1Dist_one (q₀ q₁ : ℝ) : ∑ b : Bool, |bern q₁ b - bern q₀ b| = 2 * |q₁ - q₀| := by
  have h : |(1 - q₁) - (1 - q₀)| = |q₁ - q₀| := by
    rw [show (1 - q₁) - (1 - q₀) = -(q₁ - q₀) by ring, abs_neg]
  rw [Fintype.sum_bool]
  simp only [bern, if_pos, Bool.false_eq_true, if_false]
  rw [h]
  ring

/-- **The hybrid bound.**  Independent repetition can multiply the distinguishability of two
rates by at most the number of reads. -/
theorem l1_dist_le {q₀ q₁ : ℝ} (h₀0 : 0 ≤ q₀) (h₀1 : q₀ ≤ 1) (h₁0 : 0 ≤ q₁) (h₁1 : q₁ ≤ 1)
    (n : ℕ) : l1Dist q₀ q₁ n ≤ 2 * n * |q₁ - q₀| := by
  induction n with
  | zero => simp [l1Dist, recProb]
  | succ n ih =>
      have hstep : l1Dist q₀ q₁ (n + 1) ≤ 2 * |q₁ - q₀| + l1Dist q₀ q₁ n := by
        simp only [l1Dist]
        rw [sum_cons]
        have hterm : ∀ b : Bool, ∀ t : Fin n → Bool,
            |recProb q₁ (Fin.cons b t) - recProb q₀ (Fin.cons b t)|
              ≤ |bern q₁ b - bern q₀ b| * recProb q₁ t
                + bern q₀ b * |recProb q₁ t - recProb q₀ t| := by
          intro b t
          rw [recProb_cons, recProb_cons]
          have hsplit : bern q₁ b * recProb q₁ t - bern q₀ b * recProb q₀ t
              = (bern q₁ b - bern q₀ b) * recProb q₁ t
                + bern q₀ b * (recProb q₁ t - recProb q₀ t) := by ring
          rw [hsplit]
          refine le_trans (abs_add_le _ _) ?_
          rw [abs_mul, abs_mul, abs_of_nonneg (recProb_nonneg h₁0 h₁1 t),
            abs_of_nonneg (bern_nonneg h₀0 h₀1 b)]
        calc ∑ b : Bool, ∑ t : Fin n → Bool,
              |recProb q₁ (Fin.cons b t) - recProb q₀ (Fin.cons b t)|
            ≤ ∑ b : Bool, ∑ t : Fin n → Bool,
                (|bern q₁ b - bern q₀ b| * recProb q₁ t
                  + bern q₀ b * |recProb q₁ t - recProb q₀ t|) :=
              Finset.sum_le_sum fun b _ => Finset.sum_le_sum fun t _ => hterm b t
          _ = 2 * |q₁ - q₀| + l1Dist q₀ q₁ n := by
              have hb : ∀ b : Bool, ∑ t : Fin n → Bool,
                  (|bern q₁ b - bern q₀ b| * recProb q₁ t
                    + bern q₀ b * |recProb q₁ t - recProb q₀ t|)
                  = |bern q₁ b - bern q₀ b| + bern q₀ b * l1Dist q₀ q₁ n := by
                intro b
                rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
                  sum_recProb, mul_one]
                rfl
              rw [Finset.sum_congr rfl fun b _ => hb b, Finset.sum_add_distrib,
                ← Finset.sum_mul, l1Dist_one]
              have hsum : ∑ b : Bool, bern q₀ b = 1 := by
                rw [Fintype.sum_bool]
                simp only [bern, if_pos, Bool.false_eq_true, if_false]
                ring
              rw [hsum, one_mul]
      have hnR : (0 : ℝ) ≤ n := Nat.cast_nonneg n
      calc l1Dist q₀ q₁ (n + 1) ≤ 2 * |q₁ - q₀| + l1Dist q₀ q₁ n := hstep
        _ ≤ 2 * |q₁ - q₀| + 2 * n * |q₁ - q₀| := by linarith
        _ = 2 * ((n : ℝ) + 1) * |q₁ - q₀| := by ring
        _ = 2 * ((n + 1 : ℕ) : ℝ) * |q₁ - q₀| := by push_cast; ring

/-- A subset can capture at most half of the `ℓ¹` distance between two probability laws. -/
lemma sum_sub_le_half_l1 (q₀ q₁ : ℝ) {n : ℕ} (A : Finset (Fin n → Bool)) :
    ∑ s ∈ A, recProb q₁ s - ∑ s ∈ A, recProb q₀ s ≤ l1Dist q₀ q₁ n / 2 := by
  classical
  set f : (Fin n → Bool) → ℝ := fun s => recProb q₁ s - recProb q₀ s with hf
  have htot : ∑ s : Fin n → Bool, f s = 0 := by
    simp only [hf, Finset.sum_sub_distrib, sum_recProb, sub_self]
  have hsplit : ∑ s ∈ A, f s + ∑ s ∈ Finset.univ \ A, f s = 0 := by
    have := Finset.sum_sdiff (f := f) (Finset.subset_univ A)
    linarith [this, htot]
  have hA : ∑ s ∈ A, f s ≤ ∑ s ∈ A, |f s| :=
    Finset.sum_le_sum fun s _ => le_abs_self (f s)
  have hAc : -(∑ s ∈ Finset.univ \ A, f s) ≤ ∑ s ∈ Finset.univ \ A, |f s| := by
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_le_sum fun s _ => neg_le_abs (f s)
  have hfull : ∑ s ∈ A, |f s| + ∑ s ∈ Finset.univ \ A, |f s| = l1Dist q₀ q₁ n := by
    have := Finset.sum_sdiff (f := fun s => |f s|) (Finset.subset_univ A)
    simp only [l1Dist, hf]
    linarith [this]
  have hkey : ∑ s ∈ A, f s = -(∑ s ∈ Finset.univ \ A, f s) := by linarith
  have h2 : 2 * ∑ s ∈ A, f s ≤ l1Dist q₀ q₁ n := by
    have := hA
    have h' : ∑ s ∈ A, f s ≤ ∑ s ∈ Finset.univ \ A, |f s| := by
      rw [hkey]; exact hAc
    linarith
  simp only [hf, Finset.sum_sub_distrib] at h2 ⊢
  linarith

/-! ## 2. The converse -/

/-- The records a decision rule accepts (fails to reject the baseline on). -/
noncomputable def acceptSet {n : ℕ} (T : (Fin n → Bool) → Bool) : Finset (Fin n → Bool) :=
  Finset.univ \ ruleSet T

/-- **No analysis escapes the data.**  For every decision rule `T` on the record of `n`
independent reads, the sum of its two error probabilities — rejecting the baseline when the rate
is `q₀`, and failing to reject when the rate is `q₁` — is at least `1 - n·|q₁ - q₀|`. -/
theorem test_error_sum_ge {q₀ q₁ : ℝ} (h₀0 : 0 ≤ q₀) (h₀1 : q₀ ≤ 1) (h₁0 : 0 ≤ q₁) (h₁1 : q₁ ≤ 1)
    {n : ℕ} (T : (Fin n → Bool) → Bool) :
    1 - n * |q₁ - q₀|
      ≤ ∑ s ∈ ruleSet T, recProb q₀ s + ∑ s ∈ acceptSet T, recProb q₁ s := by
  classical
  have hcompl : ∑ s ∈ acceptSet T, recProb q₁ s = 1 - ∑ s ∈ ruleSet T, recProb q₁ s := by
    have hsd := Finset.sum_sdiff (f := fun s => recProb q₁ s) (Finset.subset_univ (ruleSet T))
    have htot : ∑ s : Fin n → Bool, recProb q₁ s = 1 := sum_recProb q₁ n
    simp only [acceptSet]
    linarith [hsd, htot]
  have hhalf := sum_sub_le_half_l1 q₀ q₁ (ruleSet T)
  have hl1 := l1_dist_le h₀0 h₀1 h₁0 h₁1 n
  rw [hcompl]
  linarith

/-- **The molecule floor.**  A study that claims both error probabilities at most `α < 1/2` on
a rate contrast `Δ` must have observed at least `(1 - 2α)/Δ` molecules — whatever statistic it
used. -/
theorem molecules_lower_bound {q₀ q₁ alpha : ℝ} (h₀0 : 0 ≤ q₀) (h₀1 : q₀ ≤ 1) (h₁0 : 0 ≤ q₁)
    (h₁1 : q₁ ≤ 1) (hlt : q₀ < q₁) {n : ℕ} (T : (Fin n → Bool) → Bool)
    (hI : ∑ s ∈ ruleSet T, recProb q₀ s ≤ alpha)
    (hII : ∑ s ∈ acceptSet T, recProb q₁ s ≤ alpha) :
    1 - 2 * alpha ≤ n * (q₁ - q₀) := by
  have h := test_error_sum_ge h₀0 h₀1 h₁0 h₁1 T
  have habs : |q₁ - q₀| = q₁ - q₀ := abs_of_pos (by linarith)
  rw [habs] at h
  linarith

/-- **The detection window.**  For a realistic reporter with contrast `Δ = τ·J`: below
`(1-2α)/Δ` molecules no analysis whatsoever attains both errors `α`, and at `⌈1/(α·Δ²)⌉`
molecules the explicit midpoint counting test attains them.  The experiment is decidable
somewhere in between, and the lower end of the window degrades with the reporter exactly as the
upper end does. -/
theorem detection_window {q₀ q₁ alpha : ℝ} (h₀0 : 0 ≤ q₀) (h₀1 : q₀ ≤ 1) (h₁0 : 0 ≤ q₁)
    (h₁1 : q₁ ≤ 1) (hlt : q₀ < q₁) (ha : 0 < alpha) :
    (∀ (n : ℕ) (T : (Fin n → Bool) → Bool),
        ∑ s ∈ ruleSet T, recProb q₀ s ≤ alpha →
        ∑ s ∈ acceptSet T, recProb q₁ s ≤ alpha →
        1 - 2 * alpha ≤ n * (q₁ - q₀)) ∧
      (∀ n : ℕ, samplesFor alpha (q₁ - q₀) ≤ n →
        ∑ s ∈ accepts n (n * (q₀ + q₁) / 2), recProb q₁ s ≤ alpha ∧
          ∑ s ∈ rejects n (n * (q₀ + q₁) / 2), recProb q₀ s ≤ alpha) :=
  ⟨fun _ T hI hII => molecules_lower_bound h₀0 h₀1 h₁0 h₁1 hlt T hI hII,
    fun _ hn => samplesFor_spec h₀0 h₁1 hlt ha hn⟩

/-! ## 3. Molecules, not reads

The bound above is stated in *independent* reads.  A single-molecule experiment usually does not
deliver those: it delivers many reads of the same molecule — photon bins of one burst, frames of
one trajectory, repeated interrogations of one immobilised complex — and while the molecule
stays in the same conformational state those reads carry one state, not many.  The extreme case
is exact and is worth stating, because it is the case a photon count silently assumes away. -/

/-- The read-generation map of a clustered experiment: `g` independently sampled molecules, each
contributing `c` reads that report the same state. -/
def replicate (g c : ℕ) (t : Fin g → Bool) : Fin (g * c) → Bool :=
  fun i => t ⟨(i : ℕ) / c,
    Nat.div_lt_of_lt_mul (lt_of_lt_of_le i.isLt (le_of_eq (Nat.mul_comm g c)))⟩

/-- **Data processing.**  Whatever the instrument does to the `g` independently sampled molecule
states before a decision rule sees the result, the two error probabilities still sum to at least
`1 - g·Δ`: processing cannot create distinguishability. -/
theorem processing_lower_bound {q₀ q₁ : ℝ} (h₀0 : 0 ≤ q₀) (h₀1 : q₀ ≤ 1) (h₁0 : 0 ≤ q₁)
    (h₁1 : q₁ ≤ 1) {g n : ℕ} (R : (Fin g → Bool) → (Fin n → Bool))
    (T : (Fin n → Bool) → Bool) :
    1 - g * |q₁ - q₀|
      ≤ ∑ s ∈ ruleSet (fun t => T (R t)), recProb q₀ s
        + ∑ s ∈ acceptSet (fun t => T (R t)), recProb q₁ s :=
  test_error_sum_ge h₀0 h₀1 h₁0 h₁1 (fun t => T (R t))

/-- **Repeat reads of the same molecule buy nothing.**  Observe `g` molecules and read each of
them `c` times; the record has `g·c` entries, but the two error probabilities of *any* rule
applied to it still sum to at least `1 - g·Δ`.  The molecule count, not the read count, is what
the sample-size formulas of Part XC must be fed. -/
theorem replicate_no_help {q₀ q₁ : ℝ} (h₀0 : 0 ≤ q₀) (h₀1 : q₀ ≤ 1) (h₁0 : 0 ≤ q₁) (h₁1 : q₁ ≤ 1)
    {g c : ℕ} (T : (Fin (g * c) → Bool) → Bool) :
    1 - g * |q₁ - q₀|
      ≤ ∑ s ∈ ruleSet (fun t => T (replicate g c t)), recProb q₀ s
        + ∑ s ∈ acceptSet (fun t => T (replicate g c t)), recProb q₁ s :=
  processing_lower_bound h₀0 h₀1 h₁0 h₁1 (replicate g c) T

end Lower
end IDR
