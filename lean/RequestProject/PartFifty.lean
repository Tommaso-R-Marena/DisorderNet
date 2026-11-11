/-
# Part L  A barrier is neither necessary nor sufficient for slow kinetics

Parts XLVIII and XLIX show that the equilibrium profile of a coordinate determines neither the
rate nor the mechanism.  `RequestProject.BarrierRate` states the consequence in the form in which
it is used, and isolates the extra hypothesis that makes the familiar Arrhenius reading correct.

`IDR.barrier_rate_laws` bundles three statements:

1. *A flat landscape can be arbitrarily slow*: for every `M` there is a detailed-balanced hopping
   model on a perfectly flat three-state profile -- every state equally populated, no barrier
   anywhere -- whose mean first-passage time exceeds `M`;
2. *a barrier can be arbitrarily fast*: for every barrier height `B` and every `eps > 0` there is
   a model whose profile has a barrier of exactly height `B` (`p 0 / p 1 = exp B`) and whose mean
   first-passage time is exactly `eps`;
3. *the Arrhenius bracket*: once the kinetic prefactor is bounded, `kmin ≤ kp i ≤ kmax`, the time
   is bracketed between `p 0 / (p b · kmax)` -- the Arrhenius factor of any intermediate state --
   and `(Σ_{i<n} 1/p i) / kmin`.

So "there is a barrier, therefore the transition is slow" and "the transition is slow, therefore
there is a barrier" are both invalid as stated; they become valid exactly when the model also
commits to a bounded diffusion profile.  For a model of a disordered region the practical reading
is that a reported free-energy landscape is not a kinetic prediction, and a measured relaxation
time is not a measurement of a barrier.
-/
import Mathlib
import RequestProject.BarrierRate

set_option autoImplicit false

namespace IDR

open IDR.FirstPassage
open IDR.BarrierRate

/-- **What a barrier does and does not imply.**

1. *Flat but slow*: an explicit detailed-balanced model on the flat profile with time `≥ M`.
2. *Barrier but fast*: an explicit detailed-balanced model with barrier `exp B` and time `eps`.
3. *The Arrhenius bracket* for a bounded kinetic prefactor. -/
theorem barrier_rate_laws :
    (∀ M : ℝ, ∃ d : ℝ, 0 < d ∧ DetailedBalance 2 flatP (slowRate d) (slowRateBack d) ∧
        (∀ i, i ≤ 2 → flatP i = 1) ∧ (∀ i, i < 2 → 0 < slowRate d i) ∧ slowRateBack d 0 = 0 ∧
        M ≤ mfptFormula 2 flatP (slowRate d)) ∧
    (∀ (B eps : ℝ), 0 < eps → ∃ R : ℝ, 0 < R ∧
        DetailedBalance 2 (barrierP B) (fastRate R) (fastRateBack B R) ∧
        barrierP B 0 / barrierP B 1 = Real.exp B ∧
        mfptFormula 2 (barrierP B) (fastRate R) = eps) ∧
    (∀ (n : ℕ) (p kp : ℕ → ℝ) (kmin kmax : ℝ) (b : ℕ), b < n → (∀ i, i ≤ n → 0 < p i) →
        0 < kmin → (∀ i, i < n → kmin ≤ kp i) → (∀ i, i < n → kp i ≤ kmax) →
        (∀ i, i < n → cum p i ≤ 1) →
          p 0 / (p b * kmax) ≤ mfptFormula n p kp ∧
            mfptFormula n p kp ≤ (∑ i ∈ Finset.range n, 1 / p i) / kmin) :=
  ⟨flatLandscape_arbitrarily_slow,
    fun B _ heps => barrier_arbitrarily_fast B heps,
    fun _ _ _ _ _ _ hb hp hkmin hlo hhi hnorm => barrier_brackets_rate hb hp hkmin hlo hhi hnorm⟩

end IDR
