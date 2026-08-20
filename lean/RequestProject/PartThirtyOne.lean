/-
# Part XXXI  The scaling exponent is fitted, not measured

The Flory scaling exponent is the single number most often quoted for a disordered region, and
it is a *derived* quantity: inferred from a handful of chain lengths through a relation that
carries corrections to scaling.  `RequestProject.Scaling` computes what the standard two-point
log--log estimator returns when the truth is `R_g² = A N^{2ν}(1 + B/N)`.

* `nuHat_eq` -- the fitted exponent is `ν` plus a term built entirely from the correction;
* `nuHat_lt_of_pos_correction`, `nuHat_gt_of_neg_correction` -- the bias has a determined sign;
* `nuHat_bias_le` -- and a determined size, `B/(2 N₁ log(N₂/N₁))`, controlled by the *shortest*
  chain used;
* `two_point_fit_exact` -- and it is undetectable: a pure power law with the fitted exponent
  reproduces both measurements exactly, so no goodness-of-fit argument at two lengths can defend
  the reported exponent.

`IDR.scaling_laws` bundles the four.  A model of a disordered region should be compared with the
radii actually measured, forward-modelled at the lengths studied, and not with a fitted `ν`.
-/
import Mathlib
import RequestProject.ScalingExponent

set_option autoImplicit false

namespace IDR

/-- **The design laws of a reported scaling exponent.**

1. *The estimator is exponent plus correction*: an exact identity.
2. *The bias has a sign*: positive correction amplitude biases the apparent exponent down,
   negative biases it up.
3. *The bias has a size*: at most `B/(2 N₁ log(N₂/N₁))`.
4. *It is invisible at two lengths*: a pure power law with the fitted exponent fits both
   measurements exactly. -/
theorem scaling_laws :
    -- 1  what the two-point estimator returns
    (∀ A nu B N1 N2 : ℝ, 0 < A → 0 < N1 → 0 < N2 → Real.log N1 < Real.log N2 →
        0 < 1 + B / N1 → 0 < 1 + B / N2 →
        Scaling.nuHat A nu B N1 N2
          = nu + (Real.log (1 + B / N2) - Real.log (1 + B / N1))
              / (2 * (Real.log N2 - Real.log N1))) ∧
    -- 2  the sign of the bias
    ((∀ A nu B N1 N2 : ℝ, 0 < A → 0 < N1 → N1 < N2 → Real.log N1 < Real.log N2 → 0 < B →
        Scaling.nuHat A nu B N1 N2 < nu) ∧
      (∀ A nu B N1 N2 : ℝ, 0 < A → 0 < N1 → N1 < N2 → Real.log N1 < Real.log N2 → B < 0 →
        0 < 1 + B / N1 → nu < Scaling.nuHat A nu B N1 N2)) ∧
    -- 3  the size of the bias
    (∀ A nu B N1 N2 : ℝ, 0 < A → 0 < N1 → N1 < N2 → Real.log N1 < Real.log N2 → 0 ≤ B →
        |Scaling.nuHat A nu B N1 N2 - nu| ≤ B / N1 / (2 * (Real.log N2 - Real.log N1))) ∧
    -- 4  and it cannot be detected from two lengths
    (∀ A nu B N1 N2 : ℝ, 0 < A → 0 < N1 → 0 < N2 → Real.log N1 < Real.log N2 →
        0 < 1 + B / N1 → 0 < 1 + B / N2 →
        ∃ A' : ℝ, 0 < A' ∧
          A' * N1 ^ (2 * Scaling.nuHat A nu B N1 N2) = Scaling.rg2 A nu B N1 ∧
          A' * N2 ^ (2 * Scaling.nuHat A nu B N1 N2) = Scaling.rg2 A nu B N2) :=
  ⟨fun _ _ _ _ _ hA hN1 hN2 h12 hc1 hc2 => Scaling.nuHat_eq hA hN1 hN2 h12 hc1 hc2,
    ⟨fun _ _ _ _ _ hA hN1 hN12 h12 hB => Scaling.nuHat_lt_of_pos_correction hA hN1 hN12 h12 hB,
      fun _ _ _ _ _ hA hN1 hN12 h12 hB hc1 =>
        Scaling.nuHat_gt_of_neg_correction hA hN1 hN12 h12 hB hc1⟩,
    fun _ _ _ _ _ hA hN1 hN12 h12 hB => Scaling.nuHat_bias_le hA hN1 hN12 h12 hB,
    fun _ _ _ _ _ hA hN1 hN2 h12 hc1 hc2 => Scaling.two_point_fit_exact hA hN1 hN2 h12 hc1 hc2⟩

end IDR
