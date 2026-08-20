/-
# Part LXXXIV  The moment problem: what a finite list of averages can and cannot fix

`RequestProject.MomentProblem` asks the question every refinement protocol in this development
presupposes an answer to.  Each experiment returns a *moment* of a conformational observable -- a
mean distance, a mean square, a sixth power, an arrival-time width.  Given `k` such numbers, what
is determined?

`IDR.moment_problem_laws` bundles four statements.

1. *Nothing, without a support.*  For every `k` there are two ensembles over the same `k+2`
   distinct conformations -- nonnegative weights, each summing to one, and **sharing no
   conformation at all** -- whose moments of every order `p ≤ k` are identical.  A protocol
   restrained by `k` moments cannot claim to have measured the distribution, however small its
   residuals: an ensemble disjoint from the truth fits the same data exactly as well.
2. *The smallest instance.*  Half a population at `0` and half at `2` against all of it at `1`:
   the same mean, no shared conformation, separated only by the second moment.
3. *Everything, with one.*  Once the candidate conformations are fixed and distinct, the moments
   of orders `0,…,k` determine the populations of `k+1` conformations uniquely (Lagrange
   interpolation).  Structural prior knowledge is not a convenience but the ingredient that turns
   moment data into an ensemble -- and the honest report must therefore state the support it
   assumed.
4. *And what moments certify with no prior at all is a bound.*  Markov: at most `⟨x⟩/a` of the
   ensemble has `x ≥ a`.  Chebyshev: at most `Var/a²` of it deviates from the mean by `a`.  These
   population bounds are the part of a moment measurement that survives the indeterminacy of
   statement 1, and they are the form in which a fitted model's agreement with an average should
   be reported.  `markov_sharp` shows the first of them is attained, so nothing sharper can be
   claimed from a mean.
-/
import Mathlib
import RequestProject.MomentProblem

set_option autoImplicit false

namespace IDR

open Finset IDR.Moments

/-- **The moment-problem laws.**

1. for every `k`, two ensembles with disjoint supports share all moments of order `≤ k`;
2. the smallest instance: `{0, 2}` at equal weights versus `{1}`, same mean, different variance;
3. on a fixed distinct support of `k+1` conformations the moments of order `≤ k` determine the
   weights uniquely;
4. Markov's and Chebyshev's population bounds, the part of a moment that is certified outright;
5. and the attainment of Markov's bound, so no sharper population claim follows from a mean. -/
theorem moment_problem_laws :
    (∀ k : ℕ, ∃ x wA wB : Fin (k + 2) → ℝ,
        Function.Injective x ∧
        (∀ j, 0 ≤ wA j) ∧ (∀ j, 0 ≤ wB j) ∧
        (∑ j, wA j = 1) ∧ (∑ j, wB j = 1) ∧
        (∀ j, wA j = 0 ∨ wB j = 0) ∧ wA ≠ wB ∧
        ∀ p ≤ k, mom wA x p = mom wB x p) ∧
    (mom (wEven 1) (pts 1) 1 = mom (wOdd 1) (pts 1) 1 ∧
      mom (wEven 1) (pts 1) 2 ≠ mom (wOdd 1) (pts 1) 2) ∧
    (∀ (k : ℕ) (x u v : Fin (k + 1) → ℝ), Function.Injective x →
        (∀ p ≤ k, mom u x p = mom v x p) → u = v) ∧
    (∀ (N : ℕ) (w x : Fin N → ℝ), (∀ j, 0 ≤ w j) →
        ((∀ j, 0 ≤ x j) → ∀ a : ℝ, 0 < a → pop w x a ≤ mom w x 1 / a) ∧
        (∀ mu a : ℝ, 0 < a → popDev w x mu a ≤ (∑ j, w j * (x j - mu) ^ 2) / a ^ 2)) ∧
    (∀ a m : ℝ, 0 < a → 0 ≤ m → m ≤ a →
        ∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∀ j, 0 ≤ x j) ∧ (∑ j, w j = 1) ∧
          mom w x 1 = m ∧ pop w x a = m / a) :=
  ⟨moment_indeterminacy, two_state_instance,
    fun _ x u v hx h => weights_eq_of_moments_eq x u v hx h,
    fun _ w x hw =>
      ⟨fun hx _ ha => markov_bound w x hw hx ha, fun _ _ ha => chebyshev_bound w x hw ha⟩,
    fun _ _ ha hm0 hma => markov_sharp ha hm0 hma⟩

end IDR
