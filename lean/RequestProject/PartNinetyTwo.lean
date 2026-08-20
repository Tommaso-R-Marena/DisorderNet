/-
# Part XCII  A run that may be watched: sequential tests of the capacity law

Capstone for `RequestProject.Sequential` and `RequestProject.SequentialPower`.

Part XC priced the capacity test for a *fixed* number of molecules.  That design has a defect
nobody in a laboratory can avoid: it forbids looking at the data.  A run whose stopping time
depends on what has been seen so far has no proved error rate under the fixed-`n` theory, and
the inflation is not small — two looks at level `1/2` already give a null error of `3/4`
(`IDR.Seq.peeking_inflates_error`).  The usual remedies (correct the level for the number of
looks, or promise not to look) are either wasteful or unenforceable.

This part rebuilds the test so that stopping is free, and proves both halves of it.

1. **What is watched.**  Not the count of positive reads, but the *wealth* of a bet on the
   alternative read-out rate: the running likelihood ratio `IDR.Seq.wealth`, which
   `IDR.Seq.wealth_eq_ratio` identifies with `probL q₁ / probL q₀` on the reads seen so far.

2. **The level, at every horizon** (`IDR.Seq.ville`, `IDR.Seq.anytime_valid`).  Ville's
   inequality, proved here from scratch by induction over the read-out word: under the null the
   probability that the wealth *ever* reaches `c` is at most `1/c`.  Stopping the run the moment
   the likelihood ratio exceeds `1/α` therefore has type-I error at most `α`, whatever the run
   length, however often the counter is inspected, and whatever rule is used to decide to stop.

3. **The power, and hence a molecule count** (`IDR.Seq.sequential_power`,
   `IDR.Seq.powerSamples_spec`).  Under the alternative the expected log wealth after `n`
   molecules is exactly `n·KL(q₁‖q₀)` (`IDR.Seq.expected_logWealth`) and its variance is exactly
   `n·klVar` (`IDR.Seq.variance_logWealth`); Chebyshev on those two exact moments gives a
   finite-`n` type-II bound `n·klVar/(n·KL − log c)²`, and `powerSamples` turns it into an
   explicit number of molecules delivering power `1 − β` at level `α`.

4. **What it costs against the fixed-`n` design** (`IDR.Seq.sequential_cheaper`).  The sequential
   horizon scales like `log(1/α)/KL`, the Chebyshev design of Part XC like `1/(α·Δ²)`: at small
   `α` the sequential run is shorter, and it is never obliged to run longer than it needs to.

5. **Instantiated on the capacity test** (`sequential_capacity_laws` below).  The null is the
   under-capacity model's own prediction — the reporter fires at its false-positive rate `1 − sp`
   — and the alternative is the truth, `1 − sp + τ·J`, the omitted population times the probe's
   Youden index.  All four statements above then hold of the actual experiment.

6. **Every stopping rule** (`IDR.Seq.stopping_rule_valid`, `capacity_test_under_any_stopping_rule`
   below).  A run stopped by an arbitrary rule — any function of the reads seen so far — declares
   a refutation under the null with probability at most `α`, so the guarantee covers the rule an
   experimenter actually uses and not only "stop at the first crossing".

7. **An estimate, not only a verdict** (`IDR.Seq.confSeq_coverage`,
   `sequential_population_estimate` below).  Running the same test against every candidate value
   of the omitted population and reporting the candidates not yet excluded gives a confidence
   sequence: nested, so it only shrinks as molecules accumulate, and covering the true value at
   every horizon with probability at least `1 − α`.

What Part XCII does *not* claim.  The reads are independent and their null rate `1 − sp` is
known: the calibration confound of Part XC is untouched by sequential analysis, and a reporter
miscalibrated by `τ·J` defeats the sequential test exactly as it defeats the fixed-`n` one — the
wealth process is then a bet on a rate difference that is an instrument artefact.  The power
bound uses two moments only, so it is conservative; and `klVar` is a property of the reporter,
not of the model under test.
-/
import Mathlib
import RequestProject.Sequential
import RequestProject.SequentialPower
import RequestProject.StoppingRules
import RequestProject.ConfidenceSequence

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace PartXCII

open IDR.Noisy IDR.Seq

/-- **The sequential capacity test.**

A binary reporter with sensitivity `se` and specificity `sp` scores molecules one at a time.  The
model under test omits conformational states carrying population `τ`; under the model the
reporter fires at its false-positive rate `q₀ = 1 − sp`, and in truth it fires at
`q₁ = readRate τ se sp`.  Then:

* the whole signal is the omitted population times the probe's Youden index, `q₁ − q₀ = τ·J`;
* **level:** stopping the run whenever the likelihood ratio of `q₁` against `q₀` reaches `1/α`
  has probability at most `α` of ever firing when the model is right — simultaneously at every
  horizon `n`, so the experimenter may watch the counter and stop at will;
* **evidence:** each molecule contributes `KL(q₁‖q₀) > 0` in expectation to the log wealth, with
  strictly positive variance;
* **power:** after `powerSamples α β KL klVar` molecules the run has failed to stop with
  probability at most `β`.

Level and power together, for an experiment of unfixed length. -/
theorem sequential_capacity_laws
    {tau se sp alpha beta : ℝ}
    (hq0 : 0 < 1 - sp) (hq0' : 1 - sp < 1)
    (hq1 : 0 < readRate tau se sp) (hq1' : readRate tau se sp < 1)
    (hgap : 0 < tau * youden se sp)
    (halpha : 0 < alpha) (halpha1 : alpha < 1) (hbeta : 0 < beta) :
    (readRate tau se sp - (1 - sp) = tau * youden se sp)
    ∧ (∀ n : ℕ, ∑ l ∈ words n,
        probL (1 - sp) l * crossInd (readRate tau se sp) (1 - sp) (1 / alpha) 1 l ≤ alpha)
    ∧ 0 < kl2 (readRate tau se sp) (1 - sp)
    ∧ 0 < klVar (readRate tau se sp) (1 - sp)
    ∧ (∀ n : ℕ, powerSamples alpha beta (kl2 (readRate tau se sp) (1 - sp))
          (klVar (readRate tau se sp) (1 - sp)) ≤ n →
        ∑ l ∈ words n,
          probL (readRate tau se sp) l
            * (1 - crossInd (readRate tau se sp) (1 - sp) (1 / alpha) 1 l) ≤ beta) := by
  have hsub : readRate tau se sp - (1 - sp) = tau * youden se sp := by
    unfold readRate youden
    ring
  have hne : readRate tau se sp ≠ 1 - sp := by
    intro h
    rw [h, sub_self] at hsub
    linarith [hgap, hsub.symm]
  refine ⟨hsub, ?_, ?_, ?_, ?_⟩
  · intro n
    exact anytime_valid (le_of_lt hq1) (le_of_lt hq1') hq0 hq0' halpha n
  · exact kl2_pos hq1 hq1' hq0 hq0' hne
  · exact klVar_pos hq1 hq1' hq0 hq0' hne
  · intro n hn
    exact powerSamples_spec hq1 hq1' hq0 hq0' halpha halpha1 hbeta
      (kl2_pos hq1 hq1' hq0 hq0' hne) (klVar_pos hq1 hq1' hq0 hq0' hne) hn

/-- **The run length in the reporter's own currency.**  Because the evidence rate is at least
twice the squared rate contrast (Pinsker), and the contrast is `τ·J`, the sequential horizon is at
most `⌈log(1/α)/(2(τJ)²)⌉` molecules — the same `(τJ)²` the fixed-`n` design of Part XC pays,
with `log(1/α)` in place of `1/α`. -/
theorem sequential_horizon_from_contrast
    {tau se sp alpha : ℝ}
    (hq0 : 0 < 1 - sp) (hq0' : 1 - sp < 1)
    (hq1 : 0 ≤ readRate tau se sp) (hq1' : readRate tau se sp ≤ 1)
    (hgap : 0 < tau * youden se sp)
    (halpha : 0 < alpha) (halpha1 : alpha ≤ 1) :
    sequentialHorizon alpha (kl2 (readRate tau se sp) (1 - sp))
      ≤ ⌈Real.log (1 / alpha) / (2 * (tau * youden se sp) ^ 2)⌉₊ := by
  have hsub : readRate tau se sp - (1 - sp) = tau * youden se sp := by
    unfold readRate youden; ring
  have hne : readRate tau se sp ≠ 1 - sp := by
    intro h
    rw [h, sub_self] at hsub
    linarith [hgap, hsub.symm]
  have := sequentialHorizon_le_of_contrast hq1 hq1' hq0 hq0' hne halpha halpha1
  rwa [hsub] at this

/-- **The level holds for the rule the experimenter actually uses.**  A run of the capacity test
may be stopped by any rule whatsoever — a data-dependent one, a reagent-limited one, a rule
chosen after the fact — and the probability that it declares a refutation when the model is right
is still at most `alpha`. -/
theorem capacity_test_under_any_stopping_rule
    {tau se sp alpha : ℝ}
    (hq0 : 0 < 1 - sp) (hq0' : 1 - sp < 1)
    (hq1 : 0 ≤ readRate tau se sp) (hq1' : readRate tau se sp ≤ 1)
    (halpha : 0 < alpha) :
    ∀ (n : ℕ) (R : Rule), ∑ l ∈ words n,
        probL (1 - sp) l * declInd (readRate tau se sp) (1 - sp) (1 / alpha) R 1 l ≤ alpha :=
  fun n R => stopping_rule_valid hq1 hq1' hq0 hq0' halpha n R

/-- **From a refutation to an estimate.**  Running the same wealth test against every candidate
value of the omitted population and reporting the candidates not yet excluded gives a confidence
sequence for the omitted population: the reported set only shrinks as molecules accumulate, and
the true value is dropped from it, at any horizon, with probability at most `alpha`. -/
theorem sequential_population_estimate
    {tau se sp alpha : ℝ} {q₁ : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1)
    (h00 : 0 < readRate tau se sp) (h01 : readRate tau se sp < 1) (halpha : 0 < alpha) :
    (∀ (l m : List Bool), Excluded q₁ (readRate tau se sp) alpha l →
        Excluded q₁ (readRate tau se sp) alpha (l ++ m))
    ∧ (∀ n : ℕ, ∑ l ∈ words n,
        probL (readRate tau se sp) l * exclInd q₁ (readRate tau se sp) alpha l ≤ alpha) :=
  ⟨fun l m h => excluded_of_prefix l m h,
    population_confSeq_coverage h10 h11 h00 h01 halpha⟩

/-- The hypotheses of `sequential_capacity_laws` are satisfiable: a probe with sensitivity `0.9`
and specificity `0.95` reading a model that omits ten per cent of the population. -/
theorem worked_record_admissible :
    0 < 1 - (19/20 : ℝ) ∧ (1 - (19/20 : ℝ)) < 1
    ∧ 0 < readRate (1/10) (9/10) (19/20) ∧ readRate (1/10) (9/10) (19/20) < 1
    ∧ 0 < (1/10 : ℝ) * youden (9/10) (19/20)
    ∧ readRate (1/10 : ℝ) (9/10) (19/20) - (1 - 19/20) = (1/10) * youden (9/10) (19/20) := by
  refine ⟨by norm_num, by norm_num, ?_, ?_, ?_, ?_⟩ <;>
    simp [readRate, youden] <;> norm_num

/-- **Why the wealth statistic and not the count.**  A fixed-`n` counting test loses its level as
soon as the data are inspected more than once: two looks, each at level `1/2`, give a combined
null error of `3/4`.  The sequential test of `sequential_capacity_laws` has no such clause. -/
theorem watching_costs_nothing_only_for_the_wealth_test :
    (∑ l ∈ words 2, probL (1/2) l * (if l.head? = some true then 1 else 0) = 1/2) ∧
    (∑ l ∈ words 2, probL (1/2) l * (if l.tail.head? = some true then 1 else 0) = 1/2) ∧
    (∑ l ∈ words 2, probL (1/2) l * (if l.any id then 1 else 0) = 3/4) :=
  peeking_inflates_error

end PartXCII
end IDR
