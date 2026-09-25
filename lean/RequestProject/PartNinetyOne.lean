/-
# Part XCI  What a discovery list means when the whole proteome is screened

Capstone for `RequestProject.FalseDiscovery`, `RequestProject.ScreenBudget` and
`RequestProject.ChernoffScreen`.

Part XC priced the test of the capacity law on *one* system read by *one* realistic reporter.
But no one tests one disordered region.  A claim of the form "structure-aware models omit
conformational states that these regions occupy" is made, if it is made at all, by screening
thousands of candidate regions at once, and reporting the ones that fired.  That report is a
*discovery list*, and a list has an error rate of its own that no per-system calculation
controls.  Part LXXXIX.4 answered the conservative version of the question — run each system at
level `alpha/n` and the family-wise error is `alpha` — but a Bonferroni split over twenty
thousand candidates demands a per-candidate level of `2.5·10⁻⁶`, which is not an experiment
anyone runs.  The currency that a screen can actually afford is the *false discovery rate*: the
expected fraction of the reported list that is wrong.

This part builds that theory, from the read-out law upwards, with no asymptotics anywhere.

1. **Benjamini–Hochberg, proved** (`IDR.FDR.bh_fdr_control`).  In the finite product experiment —
   one independent coordinate per candidate region — with every null p-value superuniform,
   the expected false discovery proportion of the BH list at level `q` is at most `|H₀|·q/N`,
   hence at most `q`.  The proof is the classical leave-one-out argument, made explicit:
   `IDR.FDR.bhR_eq_iff_pZero` says that on the event that candidate `i`'s p-value is below the
   `k`-th threshold, the BH stopping index is unchanged by *setting that p-value to zero* — which
   turns `{BH stops at k}` into an event about the other candidates only, and lets independence
   factorise the term.  Nothing about the alternatives is assumed: they may be arbitrarily
   distributed and arbitrarily many.

2. **And BH never loses to Bonferroni** (`IDR.FDR.bh_dominates_bonferroni`).  Deterministically,
   every candidate the Bonferroni rule rejects is on the BH list.  The gain in power is free;
   what is paid is the weaker guarantee (a controlled *fraction* of the list, not a controlled
   *probability of any* error).

3. **A p-value the instrument can actually produce** (`IDR.Screen.chebP_superuniform`).  A
   counting reporter has an exact mean and an exact variance (Part XC) and nothing else that is
   free of modelling assumptions.  The Chebyshev statistic
   `p = n·q₀·(1-q₀)/(count - n·q₀)²` truncated at one is superuniform at every level, so it is a
   legitimate input to BH — with no normal approximation and no exact binomial tail.

4. **What that costs** (`IDR.Screen.screen_budget_quadratic`).  A two-moment p-value must reach
   `q/N` to be discoverable in the worst case, and Chebyshev buys tail probability only
   quadratically, so the screen costs at least `N²/(q·Δ²)` molecules in total: quadratic in the
   number of candidates.

5. **Unless the reporter's tail is used** (`IDR.Chernoff.chernoff_tail`,
   `IDR.Chernoff.expP_superuniform`).  The read-out law's moment generating function is exactly
   `(1 - q + q·e^λ)ⁿ`; Markov's inequality on it, with `λ` equal to the standardised deviation and
   the elementary bound `e^x ≤ 1 + x + ¾x²` on `[0,1]`, gives `P(count ≥ n·q + a) ≤ exp(-a²/4n)`
   at every finite `n`.  The resulting p-value is superuniform too, and its cost is only
   `(16·log(1/u) + 1/α)/Δ²` molecules — *logarithmic* in the demanded level, hence in the size of
   the screen (`IDR.Chernoff.screen_exp_cheaper`).

6. **The numbers** (`worked_cheb_cost`, `worked_exp_cost`).  Twenty thousand candidate regions,
   `q = 0.05`, per-candidate power `1 - 0.05`, reporter contrast `Δ = 0.1`: the two-moment design
   needs `4·10⁷` molecules per candidate; the exponential-tail design needs at most `22 800`.
   The same experiment, the same guarantee, a factor of about `1 750` in cost — and the whole
   difference is which inequality the analysis is willing to prove.

What Part XCI does *not* claim.  Independence across candidate regions is assumed, and it is a
real assumption: shared reagents, shared calibration and shared batch effects couple candidates,
and under arbitrary dependence BH needs the harmonic correction, which is not proved here.
Superuniformity of the nulls requires each candidate's baseline rate to be *known*, which is
exactly the calibration floor of Part XC — and that floor is per candidate, so multiplicity
control does not soften it: a screen whose reporters are miscalibrated by `τ·J` produces a
discovery list whose false discovery rate is uncontrolled no matter what `q` is set to.
-/
import Mathlib
import RequestProject.NoisyDetection
import RequestProject.FalseDiscovery
import RequestProject.ScreenBudget
import RequestProject.ChernoffScreen

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace PartXCI

open Finset IDR.Noisy IDR.FDR IDR.Screen IDR.Chernoff

/-- **The laws of a proteome-scale screen.**

`N` candidate disordered regions are screened; region `i` is scored on `n i` molecules by its own
binary reporter, which fires at the (unknown) rate `rate i`, and its p-value is computed against
its calibrated disorder-free baseline `q0 i`.  A region is *null* when its reporter fires at the
baseline rate.  Then, at Benjamini–Hochberg level `q`:

* the expected fraction of the discovery list that is null is at most `q`, whether the p-values
  are the two-moment ones or the exponential-tail ones;
* it is in fact at most `|H₀|·q/N`, so a screen in which most candidates are genuinely disordered
  is even better protected;
* and the list always contains everything the Bonferroni rule at the same level would report. -/
theorem screen_laws {N : ℕ} (n : Fin N → ℕ) (q0 rate : Fin N → ℝ)
    (h0 : ∀ i, 0 ≤ q0 i) (h1 : ∀ i, q0 i ≤ 1)
    (hr0 : ∀ i, 0 ≤ rate i) (hr1 : ∀ i, rate i ≤ 1)
    {q : ℝ} (hq : 0 ≤ q) (H₀ : Finset (Fin N)) (hnull : ∀ i ∈ H₀, rate i = q0 i) :
    (EE (V := fun i => (Fin (n i) → Bool)) (fun i s => recProb (rate i) s)
        (fun ω => fdp N q (pvec (fun i s => chebP (n i) (q0 i) s) ω) H₀) ≤ q)
    ∧ (EE (V := fun i => (Fin (n i) → Bool)) (fun i s => recProb (rate i) s)
        (fun ω => fdp N q (pvec (fun i s => expP (n i) (q0 i) s) ω) H₀) ≤ q)
    ∧ (EE (V := fun i => (Fin (n i) → Bool)) (fun i s => recProb (rate i) s)
        (fun ω => fdp N q (pvec (fun i s => expP (n i) (q0 i) s) ω) H₀)
        ≤ (H₀.card : ℝ) * q / N)
    ∧ (∀ p : Fin N → ℝ, ∀ i : Fin N, p i ≤ q / N → i ∈ bhRej N q p) := by
  refine ⟨screen_fdr_control n q0 rate h0 h1 hr0 hr1 hq H₀ hnull,
    screen_fdr_control_exp n q0 rate h0 h1 hr0 hr1 hq H₀ hnull, ?_, ?_⟩
  · refine bh_fdr_control (V := fun i => (Fin (n i) → Bool)) hq
      (fun i s => recProb_nonneg (hr0 i) (hr1 i) s)
      (fun i => sum_recProb (rate i) (n i))
      (fun i s => expP_nonneg (n i) (q0 i) s) H₀ ?_
    intro i hi t ht
    have := expP_superuniform (h0 i) (h1 i) (n i) (t := t) ht
    rw [hnull i hi]
    exact this
  · intro p i hi
    exact bh_dominates_bonferroni N hq p hi

/-! ## The worked record

Twenty thousand candidate regions at Benjamini–Hochberg level `q = 0.05`, per-candidate power
`1 - 0.05`, reporter contrast `Δ = 0.1`. -/

/-- The two-moment design: forty million molecules per candidate. -/
theorem worked_cheb_cost :
    screenSamples ((0.05 : ℝ) / 20000) 0.05 0.1 = 40000000 := by
  unfold screenSamples
  norm_num

/-- Logarithm bound used by the worked record. -/
lemma log_four_hundred_thousand : Real.log 400000 ≤ 13 := by
  rw [Real.log_le_iff_le_exp (by norm_num)]
  have h : (2.7182818283 : ℝ) ≤ Real.exp 1 := le_of_lt Real.exp_one_gt_d9
  calc (400000 : ℝ) ≤ (2.7182818283 : ℝ) ^ 13 := by norm_num
    _ ≤ (Real.exp 1) ^ 13 := by gcongr
    _ = Real.exp 13 := by rw [← Real.exp_nat_mul]; norm_num

/-- The exponential-tail design: at most twenty-two thousand eight hundred molecules per
candidate, for the same guarantee on the same experiment. -/
theorem worked_exp_cost :
    expSamples ((0.05 : ℝ) / 20000) 0.05 0.1 ≤ 22800 := by
  unfold expSamples
  refine Nat.ceil_le.mpr ?_
  have hlog : Real.log (1 / ((0.05 : ℝ) / 20000)) ≤ 13 := by
    have h : (1 : ℝ) / ((0.05 : ℝ) / 20000) = 400000 := by norm_num
    rw [h]
    exact log_four_hundred_thousand
  rw [div_le_iff₀ (by norm_num)]
  push_cast
  nlinarith [hlog]

/-- The ratio the two records express: the assumption-free two-moment analysis costs more than a
thousand times as many molecules per candidate as the exponential-tail analysis of the very same
read-out law. -/
theorem worked_cost_ratio :
    1750 * (expSamples ((0.05 : ℝ) / 20000) 0.05 0.1)
      ≤ screenSamples ((0.05 : ℝ) / 20000) 0.05 0.1 := by
  have h1 := worked_exp_cost
  have h2 := worked_cheb_cost
  omega

end PartXCI
end IDR
