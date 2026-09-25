/-
# Part VIII.1  Time: how long must the sampling run?

Every earlier part treats the target as an *equilibrium* ensemble and the training data as
independent draws from it.  Neither is free.  A disordered region that interconverts slowly
-- because two of its states are separated by a free-energy barrier -- is sampled by a
simulation or a single-molecule experiment only after a time set by that barrier, and the
frames of a trajectory are not independent until they are separated by the same time.  This
file proves both statements on the smallest system that has them: a two-state exchange.

* `twoStep`, `twoStep_iterate_sub_stat` -- the exact solution of a two-state chain with
  forward rate `a` and backward rate `b`: the deviation from the stationary population
  `b/(a+b)` decays as `lam^t` with `lam = 1 - a - b`.
* `relaxation_lower_bound` -- **the run must be at least as long as the relaxation time.**
  If a run of `t` steps started from a population `d` away from equilibrium reports the
  populations to accuracy `eps`, then `1 - eps/d ≤ t·(1 - lam)`, i.e. `t` is at least
  `(1 - eps/d)` relaxation times `1/(1-lam)`.
* `barrier_cost` -- and since the exchange rate of an activated process is exponentially
  small in the barrier, the required run length is **exponential in the barrier height**:
  the sampling cost of a rugged disordered landscape is not a matter of engineering.
* `ell1_error_of_unequilibrated` -- an unequilibrated run does not merely converge slowly;
  it reports populations that are wrong by `2·lam^t·d` in the operational `ℓ¹` metric of
  Part III, so it enters the error budget on the same footing as a modelling error.
* `autocorrelation` and `frames_correlated_within_relaxation_time` -- the two-time
  correlation of the state decays as `lam^t` as well, and frames closer together than half a
  relaxation time retain at least half the full variance in covariance.  So the `n`
  independent frames assumed by the tensorisation law of Part VI must be counted as
  `n = T/(2·tau)`, not as the number of stored snapshots.
-/
import Mathlib
import RequestProject.Repeats

namespace IDR

namespace Relax

/-- One step of a two-state exchange: `p` is the population of the first state, `a` the
probability of leaving it per step and `b` the probability of entering it. -/
def twoStep (a b p : ℝ) : ℝ := b + (1 - a - b) * p

/-- The stationary population of the first state. -/
noncomputable def stat (a b : ℝ) : ℝ := b / (a + b)

lemma twoStep_stat {a b : ℝ} (hab : a + b ≠ 0) : twoStep a b (stat a b) = stat a b := by
  simp only [twoStep, stat]
  field_simp
  ring

/-- **Exact relaxation of a two-state exchange.**  The deviation from the stationary
population decays geometrically with ratio `lam = 1 - a - b`. -/
theorem twoStep_iterate_sub_stat {a b : ℝ} (hab : a + b ≠ 0) (p : ℝ) (t : ℕ) :
    (twoStep a b)^[t] p - stat a b = (1 - a - b) ^ t * (p - stat a b) := by
  induction t with
  | zero => simp
  | succ n ih =>
    rw [Function.iterate_succ_apply', twoStep, pow_succ]
    have h : b + (1 - a - b) * ((twoStep a b)^[n] p) - stat a b
        = (1 - a - b) * ((twoStep a b)^[n] p - stat a b) := by
      have hs : twoStep a b (stat a b) = stat a b := twoStep_stat hab
      simp only [twoStep] at hs
      nlinarith [hs]
    rw [h, ih]
    ring

/-- The elementary Bernoulli bound behind the relaxation-time law. -/
lemma one_sub_mul_le_pow {lam : ℝ} (h0 : 0 ≤ lam) (t : ℕ) : 1 - t * (1 - lam) ≤ lam ^ t := by
  have h := one_add_mul_le_pow (a := lam - 1) (by linarith) t
  calc 1 - (t:ℝ) * (1 - lam) = 1 + t * (lam - 1) := by ring
    _ ≤ lam ^ t := by simpa using h

/-- **The run must be at least as long as the relaxation time.**  If a trajectory of `t`
steps, started a distance `d > 0` from the equilibrium population, reports the populations to
accuracy `eps`, then `1 - eps/d ≤ t·(1 - lam)`: the number of steps is at least
`(1 - eps/d)` relaxation times.  No estimator, smoothing or reweighting scheme applied to
the same trajectory can evade this, because the trajectory simply has not visited the other
state. -/
theorem relaxation_lower_bound {lam d eps : ℝ} (hlam : 0 ≤ lam) (hd : 0 < d) (t : ℕ)
    (h : lam ^ t * d ≤ eps) : 1 - eps / d ≤ t * (1 - lam) := by
  have hpow : lam ^ t ≤ eps / d := by
    rw [le_div_iff₀ hd]
    exact h
  have hb := one_sub_mul_le_pow hlam t
  linarith

/-- **The cost of a barrier.**  Writing the exchange rate of an activated process as
`1 - lam = exp (-B)` with `B` the barrier in units of `kT`, a run that reports the
populations to accuracy `eps` must be at least `(1 - eps/d)·exp B` steps long: **the
sampling cost is exponential in the barrier height.** -/
theorem barrier_cost {B d eps : ℝ} (hd : 0 < d) (hB : 0 ≤ B) (t : ℕ)
    (h : (1 - Real.exp (-B)) ^ t * d ≤ eps) :
    (1 - eps / d) * Real.exp B ≤ t := by
  have hexp0 : 0 < Real.exp (-B) := Real.exp_pos _
  have hexp1 : Real.exp (-B) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    linarith
  have hlam : 0 ≤ 1 - Real.exp (-B) := by linarith
  have hkey := relaxation_lower_bound hlam hd t h
  have hrate : 1 - (1 - Real.exp (-B)) = Real.exp (-B) := by ring
  rw [hrate] at hkey
  have hpos : 0 < Real.exp B := Real.exp_pos _
  have hmul : (1 - eps / d) * Real.exp B ≤ (t * Real.exp (-B)) * Real.exp B :=
    mul_le_mul_of_nonneg_right hkey hpos.le
  rwa [mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, mul_one] at hmul

/-- **An unequilibrated run is wrong in the operational metric.**  Its reported populations
are at `ℓ¹` distance `2·lam^t·d` from the truth, so a too-short trajectory enters the error
budget exactly like a modelling error. -/
theorem ell1_error_of_unequilibrated {a b : ℝ} (hab : a + b ≠ 0) (p : ℝ) (t : ℕ) :
    |(twoStep a b)^[t] p - stat a b| + |(1 - (twoStep a b)^[t] p) - (1 - stat a b)|
      = 2 * |(1 - a - b) ^ t| * |p - stat a b| := by
  have h := twoStep_iterate_sub_stat hab p t
  have h2 : (1 - (twoStep a b)^[t] p) - (1 - stat a b)
      = -((twoStep a b)^[t] p - stat a b) := by ring
  rw [h2, abs_neg, h, abs_mul]
  ring

/-- **The two-time correlation decays at the same rate.**  Started in the first state, the
excess probability of being found there after `t` steps is `lam^t·(1 - pi)`, so the
covariance of the state with itself at lag `t` is `pi(1-pi)·lam^t`. -/
theorem autocorrelation {a b : ℝ} (hab : a + b ≠ 0) (t : ℕ) :
    stat a b * ((twoStep a b)^[t] 1 - stat a b)
      = stat a b * (1 - stat a b) * (1 - a - b) ^ t := by
  rw [twoStep_iterate_sub_stat hab 1 t]
  ring

/-- **Frames closer than half a relaxation time are not independent.**  If `t·(1-lam) ≤ 1/2`
then the lag-`t` covariance is still at least half the variance.  The independent frames
assumed by the tensorisation law of Part VI must therefore be counted in units of the
relaxation time, not in stored snapshots. -/
theorem frames_correlated_within_relaxation_time {lam sig : ℝ} (hlam : 0 ≤ lam) (hsig : 0 ≤ sig)
    (t : ℕ) (ht : t * (1 - lam) ≤ 1/2) : sig / 2 ≤ sig * lam ^ t := by
  have hb := one_sub_mul_le_pow hlam t
  have hhalf : (1:ℝ)/2 ≤ lam ^ t := by linarith
  calc sig / 2 = sig * (1/2) := by ring
    _ ≤ sig * lam ^ t := by exact mul_le_mul_of_nonneg_left hhalf hsig

end Relax

end IDR
