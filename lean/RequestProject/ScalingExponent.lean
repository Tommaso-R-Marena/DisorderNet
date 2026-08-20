/-
# Part XXXI  The scaling exponent is fitted, not measured

The single number most often quoted for a disordered region is its Flory scaling exponent `ν`,
read off from the dependence of the radius of gyration on chain length, `R_g² = A N^{2ν}`.  It is
never measured directly: it is inferred from a small number of lengths, and the underlying
relation carries corrections to scaling, `R_g² = A N^{2ν}(1 + B/N)`, which are not small at the
lengths of real disordered regions.  This file computes what the standard two-point log--log
estimator actually returns.

* `nuHat_eq` -- the exact identity: the fitted exponent is
  `ν + (log(1 + B/N₂) - log(1 + B/N₁)) / (2 log(N₂/N₁))`.  The estimator is the true exponent plus
  a term that is entirely a property of the correction, not of the polymer statistics.
* `nuHat_lt_of_pos_correction` and `nuHat_gt_of_neg_correction` -- the bias has a determined
  *sign*: a positive correction amplitude biases the apparent exponent *down* (the chain looks
  less swollen than it is), a negative one biases it up.  Reporting `ν = 0.54` rather than `0.6`
  from two lengths is what a positive correction to scaling looks like.
* `nuHat_bias_le` -- and a determined size: the bias is at most `B/(2 N₁ log(N₂/N₁))`, so it
  shrinks only with the *shortest* chain used and with the lever arm in log length.
* `two_point_fit_exact` -- worst of all, it is undetectable at two lengths: a pure power law with
  the fitted exponent and no correction reproduces both measurements exactly.  Goodness of fit
  on two lengths carries no information about the correction, so an exponent quoted from two
  lengths cannot be defended by the quality of its fit.

Design consequence: a scaling exponent reported for a disordered region is a statement about the
lengths used and the correction model assumed.  A model of the region should be compared with the
*measurements* — the radii at the lengths actually studied, forward-modelled — and not with a
fitted `ν`, which is a derived quantity with a sign-determined, undetectable bias.
-/
import Mathlib

set_option autoImplicit false

namespace Scaling

/-- Radius of gyration squared with the leading correction to scaling. -/
noncomputable def rg2 (A nu B N : ℝ) : ℝ := A * N ^ (2 * nu) * (1 + B / N)

/-- The two-point log--log estimator of the scaling exponent. -/
noncomputable def nuHat (A nu B N1 N2 : ℝ) : ℝ :=
  (Real.log (rg2 A nu B N2) - Real.log (rg2 A nu B N1)) / (2 * (Real.log N2 - Real.log N1))

lemma log_rg2 {A nu B N : ℝ} (hA : 0 < A) (hN : 0 < N) (hcorr : 0 < 1 + B / N) :
    Real.log (rg2 A nu B N) = Real.log A + 2 * nu * Real.log N + Real.log (1 + B / N) := by
  have hpow : (0 : ℝ) < N ^ (2 * nu) := Real.rpow_pos_of_pos hN _
  rw [rg2, Real.log_mul (by positivity) (ne_of_gt hcorr),
    Real.log_mul (ne_of_gt hA) (ne_of_gt hpow), Real.log_rpow hN]

/-- **What the two-point estimator returns.** -/
theorem nuHat_eq {A nu B N1 N2 : ℝ} (hA : 0 < A) (hN1 : 0 < N1) (hN2 : 0 < N2)
    (h12 : Real.log N1 < Real.log N2) (hc1 : 0 < 1 + B / N1) (hc2 : 0 < 1 + B / N2) :
    nuHat A nu B N1 N2
      = nu + (Real.log (1 + B / N2) - Real.log (1 + B / N1))
          / (2 * (Real.log N2 - Real.log N1)) := by
  have hL : 0 < Real.log N2 - Real.log N1 := by linarith
  rw [nuHat, log_rg2 hA hN2 hc2, log_rg2 hA hN1 hc1]
  field_simp
  ring

/-- **A positive correction to scaling biases the apparent exponent down.** -/
theorem nuHat_lt_of_pos_correction {A nu B N1 N2 : ℝ} (hA : 0 < A) (hN1 : 0 < N1)
    (hN12 : N1 < N2) (h12 : Real.log N1 < Real.log N2) (hB : 0 < B) :
    nuHat A nu B N1 N2 < nu := by
  have hN2 : 0 < N2 := lt_trans hN1 hN12
  have hlt : B / N2 < B / N1 := by
    exact div_lt_div_of_pos_left hB hN1 hN12
  have hc1 : 0 < 1 + B / N1 := by positivity
  have hc2 : 0 < 1 + B / N2 := by positivity
  have hlog : Real.log (1 + B / N2) < Real.log (1 + B / N1) :=
    Real.log_lt_log hc2 (by linarith)
  have hL : 0 < 2 * (Real.log N2 - Real.log N1) := by linarith
  rw [nuHat_eq hA hN1 hN2 h12 hc1 hc2]
  have : (Real.log (1 + B / N2) - Real.log (1 + B / N1))
      / (2 * (Real.log N2 - Real.log N1)) < 0 := div_neg_of_neg_of_pos (by linarith) hL
  linarith

/-- **A negative correction biases it up.** -/
theorem nuHat_gt_of_neg_correction {A nu B N1 N2 : ℝ} (hA : 0 < A) (hN1 : 0 < N1)
    (hN12 : N1 < N2) (h12 : Real.log N1 < Real.log N2) (hB : B < 0) (hc1 : 0 < 1 + B / N1) :
    nu < nuHat A nu B N1 N2 := by
  have hN2 : 0 < N2 := lt_trans hN1 hN12
  have hlt : B / N1 < B / N2 := by
    have hinv : 1 / N2 < 1 / N1 := one_div_lt_one_div_of_lt hN1 hN12
    have hmul : B * (1 / N1) < B * (1 / N2) := mul_lt_mul_of_neg_left hinv hB
    have e1 : B / N1 = B * (1 / N1) := by ring
    have e2 : B / N2 = B * (1 / N2) := by ring
    rw [e1, e2]
    exact hmul
  have hc2 : 0 < 1 + B / N2 := by linarith
  have hlog : Real.log (1 + B / N1) < Real.log (1 + B / N2) :=
    Real.log_lt_log hc1 (by linarith)
  have hL : 0 < 2 * (Real.log N2 - Real.log N1) := by linarith
  rw [nuHat_eq hA hN1 hN2 h12 hc1 hc2]
  have : 0 < (Real.log (1 + B / N2) - Real.log (1 + B / N1))
      / (2 * (Real.log N2 - Real.log N1)) := div_pos (by linarith) hL
  linarith

/-- **The size of the bias.**  It shrinks only with the shortest chain used and with the lever
arm in log length. -/
theorem nuHat_bias_le {A nu B N1 N2 : ℝ} (hA : 0 < A) (hN1 : 0 < N1) (hN12 : N1 < N2)
    (h12 : Real.log N1 < Real.log N2) (hB : 0 ≤ B) :
    |nuHat A nu B N1 N2 - nu| ≤ B / N1 / (2 * (Real.log N2 - Real.log N1)) := by
  have hN2 : 0 < N2 := lt_trans hN1 hN12
  have hle : B / N2 ≤ B / N1 := by
    exact div_le_div_of_nonneg_left hB hN1 hN12.le
  have hc1 : 0 < 1 + B / N1 := by positivity
  have hc2 : 0 < 1 + B / N2 := by positivity
  have hL : 0 < 2 * (Real.log N2 - Real.log N1) := by linarith
  have hlog2 : Real.log (1 + B / N2) ≤ Real.log (1 + B / N1) :=
    Real.log_le_log hc2 (by linarith)
  have hnonneg : 0 ≤ Real.log (1 + B / N2) :=
    Real.log_nonneg (by have : 0 ≤ B / N2 := by positivity
                        linarith)
  have hupper : Real.log (1 + B / N1) ≤ B / N1 := by
    have := Real.add_one_le_exp (B / N1)
    have hxx : (1 : ℝ) + B / N1 ≤ Real.exp (B / N1) := by linarith
    calc Real.log (1 + B / N1) ≤ Real.log (Real.exp (B / N1)) := Real.log_le_log hc1 hxx
      _ = B / N1 := Real.log_exp _
  rw [nuHat_eq hA hN1 hN2 h12 hc1 hc2]
  have hval : nu + (Real.log (1 + B / N2) - Real.log (1 + B / N1))
      / (2 * (Real.log N2 - Real.log N1)) - nu
      = (Real.log (1 + B / N2) - Real.log (1 + B / N1))
        / (2 * (Real.log N2 - Real.log N1)) := by ring
  rw [hval, abs_div, abs_of_pos hL, abs_sub_comm,
    abs_of_nonneg (by linarith : (0:ℝ) ≤ Real.log (1 + B / N1) - Real.log (1 + B / N2))]
  exact (div_le_div_iff_of_pos_right hL).mpr (by linarith)

/-- **The correction is invisible at two lengths.**  A pure power law with the fitted exponent
and no correction term reproduces both measured radii exactly, so the quality of a two-point fit
says nothing about the correction to scaling. -/
theorem two_point_fit_exact {A nu B N1 N2 : ℝ} (hA : 0 < A) (hN1 : 0 < N1) (hN2 : 0 < N2)
    (h12 : Real.log N1 < Real.log N2) (hc1 : 0 < 1 + B / N1) (hc2 : 0 < 1 + B / N2) :
    ∃ A' : ℝ, 0 < A' ∧
      A' * N1 ^ (2 * nuHat A nu B N1 N2) = rg2 A nu B N1 ∧
      A' * N2 ^ (2 * nuHat A nu B N1 N2) = rg2 A nu B N2 := by
  have hL : 0 < Real.log N2 - Real.log N1 := by linarith
  set nh := nuHat A nu B N1 N2 with hnh
  have hr1 : (0 : ℝ) < rg2 A nu B N1 := by
    have : (0 : ℝ) < N1 ^ (2 * nu) := Real.rpow_pos_of_pos hN1 _
    unfold rg2; positivity
  have hr2 : (0 : ℝ) < rg2 A nu B N2 := by
    have : (0 : ℝ) < N2 ^ (2 * nu) := Real.rpow_pos_of_pos hN2 _
    unfold rg2; positivity
  have hp1 : (0 : ℝ) < N1 ^ (2 * nh) := Real.rpow_pos_of_pos hN1 _
  have hp2 : (0 : ℝ) < N2 ^ (2 * nh) := Real.rpow_pos_of_pos hN2 _
  refine ⟨rg2 A nu B N1 / N1 ^ (2 * nh), by positivity, by field_simp, ?_⟩
  -- the second point is matched because `nh` was defined by the log-ratio
  have hkey : Real.log (rg2 A nu B N2) - Real.log (rg2 A nu B N1)
      = 2 * nh * (Real.log N2 - Real.log N1) := by
    rw [hnh, nuHat]
    field_simp
  have hlogeq : Real.log (rg2 A nu B N1 / N1 ^ (2 * nh) * N2 ^ (2 * nh))
      = Real.log (rg2 A nu B N2) := by
    rw [Real.log_mul (by positivity) (ne_of_gt hp2), Real.log_div (ne_of_gt hr1) (ne_of_gt hp1),
      Real.log_rpow hN1, Real.log_rpow hN2]
    linarith [hkey]
  have hposl : (0 : ℝ) < rg2 A nu B N1 / N1 ^ (2 * nh) * N2 ^ (2 * nh) := by positivity
  exact Real.log_injOn_pos (Set.mem_Ioi.mpr hposl) (Set.mem_Ioi.mpr hr2) hlogeq

end Scaling
