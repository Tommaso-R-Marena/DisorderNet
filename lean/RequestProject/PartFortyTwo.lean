/-
# Part XLII  Synthesis: two errors a model of the mature chain makes about the nascent one

`RequestProject.Cotranslational` treats the chain as the cell actually produces it: emerging
from the ribosome one residue at a time, its accessible ensemble changing at every elongation
step, and relaxing towards that moving target only as fast as its own dynamics allow.

`IDR.cotranslational_laws` bundles the five statements:

1. *the adiabatic bound* -- with per-step contraction `delta < 1` and per-step equilibrium
   drift at most `d`, the chain lags behind its own equilibrium by at most
   `delta^t·(initial lag) + d(1 − delta^t)/(1 − delta)`, hence at most `(initial lag) + d/(1 − delta)`;
2. *the bound is attained* -- a chain relaxing by exactly `delta` per step behind a uniformly
   drifting equilibrium lags by exactly `d(1 − delta^t)/(1 − delta)`, converging to
   `d/(1 − delta)`, and satisfies each hypothesis with equality;
3. *slow synthesis is quasi-static* -- with `delta = e^{-lambda·tau}` the steady bound tends to
   `d` as the codon time `tau` grows;
4. *fast synthesis is not* -- the same bound is at least `d/(lambda·tau)`, which grows without
   limit as the codon time shrinks;
5. *and even at infinite slowness the ensembles differ* -- the fragment's equilibrium
   population without a contact partner that exists only in the mature chain differs from the
   mature population by exactly `(e^{beta·eps} − 1)/(2(e^{beta·eps} + 1))`, positive for every
   stabilising contact and tending to `1/2` for a strong one, so a model that reproduces the
   mature ensemble exactly carries that whole difference as error on the nascent chain.
-/
import Mathlib
import RequestProject.Cotranslational

set_option autoImplicit false

namespace IDR

open IDR.Cotranslational

/-- **The co-translational laws.**

1. *Lag.*  Under per-step contraction `delta ∈ [0,1)` towards the current equilibrium and
   per-step equilibrium drift at most `d ≥ 0`, the distance to equilibrium after `t` residues
   is at most `delta^t·(initial lag) + d/(1 − delta)`.
2. *Sharpness.*  There is a chain meeting every hypothesis with equality whose lag is exactly
   `d(1 − delta^t)/(1 − delta)` and converges to `d/(1 − delta)`.
3. *Quasi-static limit.*  With `delta = e^{-lambda·tau}` the steady bound tends to `d` as the
   codon time grows.
4. *Fast synthesis.*  The same bound is at least `d/(lambda·tau)`.
5. *Vectorial context.*  The nascent and mature equilibrium populations of one fragment
   conformation differ by exactly `(e^{beta·eps} − 1)/(2(e^{beta·eps} + 1)) > 0`, tending to
   `1/2`; a model exact on the mature chain carries that as its error on the nascent one. -/
theorem cotranslational_laws :
    (∀ {X : Type} [PseudoMetricSpace X] (p pi : ℕ → X) (delta d : ℝ), 0 ≤ delta → delta < 1 →
        0 ≤ d →
        (∀ t, dist (p (t + 1)) (pi t) ≤ delta * dist (p t) (pi t)) →
        (∀ t, dist (pi t) (pi (t + 1)) ≤ d) →
        ∀ t, dist (p t) (pi t) ≤ delta ^ t * dist (p 0) (pi 0) + d / (1 - delta)) ∧
    (∀ delta d : ℝ, 0 ≤ delta → delta < 1 → 0 ≤ d →
        ((∀ t, dist (sharpP delta d (t + 1)) (sharpPi d t)
            = delta * dist (sharpP delta d t) (sharpPi d t)) ∧
         (∀ t, dist (sharpPi d t) (sharpPi d (t + 1)) = d) ∧
         dist (sharpP delta d 0) (sharpPi d 0) = 0 ∧
         (∀ t, dist (sharpP delta d t) (sharpPi d t) = d * (1 - delta ^ t) / (1 - delta))) ∧
        Filter.Tendsto (fun t : ℕ => dist (sharpP delta d t) (sharpPi d t)) Filter.atTop
          (nhds (d / (1 - delta)))) ∧
    (∀ lam d : ℝ, 0 < lam →
        Filter.Tendsto (fun tau : ℝ => d / (1 - Real.exp (-(lam * tau)))) Filter.atTop
          (nhds d)) ∧
    (∀ lam tau d : ℝ, 0 < lam → 0 < tau → 0 ≤ d →
        d / (lam * tau) ≤ d / (1 - Real.exp (-(lam * tau)))) ∧
    (∀ beta eps : ℝ, 0 < beta * eps →
        matureFraction beta eps - nascentFraction
            = (Real.exp (beta * eps) - 1) / (2 * (Real.exp (beta * eps) + 1)) ∧
        0 < matureFraction beta eps - nascentFraction ∧
        |matureFraction beta eps - nascentFraction|
            = (Real.exp (beta * eps) - 1) / (2 * (Real.exp (beta * eps) + 1))) ∧
    (∀ beta : ℝ, 0 < beta →
        Filter.Tendsto (fun eps : ℝ => matureFraction beta eps - nascentFraction) Filter.atTop
          (nhds (1 / 2))) := by
  refine ⟨fun p pi delta d h0 h1 hd hrelax hdrift t =>
      tracking_error_le_steady p pi h0 h1 hd hrelax hdrift t,
    fun delta d h0 h1 hd => ⟨sharp_saturates h0 h1 hd, sharp_tracking_error_tendsto h0 h1 hd⟩,
    fun lam d hlam => quasi_static_limit hlam,
    fun lam tau d hlam htau hd => lag_bound_ge_of_fast_synthesis hlam htau hd,
    fun beta eps h => ⟨vectorial_gap beta eps, vectorial_gap_pos h,
      (mature_model_error_on_nascent h rfl).1⟩,
    fun beta hbeta => vectorial_gap_tendsto_half hbeta⟩

end IDR
