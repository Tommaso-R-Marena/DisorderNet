/-
# Part XXXVII  Aggregation kinetics: the lag time is a logarithm

`RequestProject.Aggregation` treats the fate that makes disordered regions worth modelling in the
first place — aggregation — and the number extracted from a thioflavin curve: the lag time.

In the exactly solvable early-time nucleation–elongation model `M'' = κ²M`, `M(0) = 0`,
`M'(0) = v`, the mass is `(v/κ)·sinh(κt)`.  It is strictly positive at every positive time, so the
lag phase is a detection threshold and not a waiting period; the model carries the exact invariant
`(M')² − κ²M² = v²`, which is what a curve determines; the time to cross a threshold `Mc` is
`arsinh(κMc/v)/κ`, a logarithm of the nucleation flux; multiplying either the threshold or the
nucleation rate by `r` moves the apparent lag by at most `log r/κ`, and by at least
`(log r − log(3/2))/κ` once the threshold is at or above `v/κ`; and for *every* growth rate there
is a nucleation flux reproducing an observed lag time exactly, so the lag time alone determines
neither.

`IDR.aggregation_laws` bundles the seven statements.
-/
import Mathlib
import RequestProject.Aggregation

set_option autoImplicit false

namespace IDR

/-- **The design laws of an aggregation curve.**

1. *The model*: `mass` solves `M'' = κ²M` with `M(0) = 0` and `M'(0) = v`.
2. *No lag phase*: the mass is strictly positive at every positive time.
3. *The invariant*: `(M')² − κ²M² = v²` at all times.
4. *The threshold time*: the mass reaches `Mc` exactly at `lagTime v κ Mc = arsinh(κMc/v)/κ`.
5. *Logarithmic in the threshold and in the nucleation rate*: scaling either by `r ≥ 1` moves the
   apparent lag time by at most `log r / κ`.
6. *And no faster than logarithmic*: for a threshold at or above `v/κ` the shift is at least
   `(log r − log(3/2))/κ`.
7. *The lag time determines neither parameter*: every growth rate `κ` admits a nucleation flux
   reproducing a given lag time, and two such fits differ in that flux. -/
theorem aggregation_laws :
    -- 1  the initial value problem
    (∀ v κ : ℝ, κ ≠ 0 → (Agg.mass v κ 0 = 0 ∧
        ∀ t : ℝ, HasDerivAt (Agg.mass v κ) (v * Real.cosh (κ * t)) t ∧
          HasDerivAt (fun s => v * Real.cosh (κ * s)) (κ ^ 2 * Agg.mass v κ t) t)) ∧
    -- 2  there is no lag phase
    (∀ v κ t : ℝ, 0 < v → 0 < κ → 0 < t → 0 < Agg.mass v κ t) ∧
    -- 3  the constant of the motion
    (∀ v κ : ℝ, κ ≠ 0 → ∀ t : ℝ,
        (v * Real.cosh (κ * t)) ^ 2 - κ ^ 2 * (Agg.mass v κ t) ^ 2 = v ^ 2) ∧
    -- 4  the threshold crossing time
    (∀ v κ Mc : ℝ, v ≠ 0 → κ ≠ 0 → Agg.mass v κ (Agg.lagTime v κ Mc) = Mc) ∧
    -- 5  logarithmic sensitivity, from above
    (∀ v κ Mc r : ℝ, 0 < v → 0 < κ → 0 ≤ Mc → 1 ≤ r →
        Agg.lagTime v κ (r * Mc) - Agg.lagTime v κ Mc ≤ Real.log r / κ) ∧
    (∀ v κ Mc r : ℝ, 0 < v → 0 < κ → 0 ≤ Mc → 1 ≤ r →
        Agg.lagTime v κ Mc - Agg.lagTime (r * v) κ Mc ≤ Real.log r / κ) ∧
    -- 6  and from below
    (∀ v κ Mc r : ℝ, 0 < v → 0 < κ → v / κ ≤ Mc → 1 ≤ r →
        (Real.log r - Real.log (3/2)) / κ ≤ Agg.lagTime v κ (r * Mc) - Agg.lagTime v κ Mc) ∧
    -- 7  the lag time alone determines nothing
    (∀ κ Mc T : ℝ, 0 < κ → 0 < Mc → 0 < T →
        ∃ v : ℝ, 0 < v ∧ v = κ * Mc / Real.sinh (κ * T) ∧ Agg.lagTime v κ Mc = T) ∧
    (Agg.lagTime (1 / Real.sinh 1) 1 1 = 1 ∧ Agg.lagTime (2 / Real.sinh 2) 2 1 = 1 ∧
      2 / Real.sinh 2 < 1 / Real.sinh 1) := by
  refine ⟨fun v κ hκ => ⟨Agg.mass_zero v κ,
      fun t => ⟨Agg.mass_hasDerivAt v κ hκ t, Agg.mass_second_deriv v κ hκ t⟩⟩,
    fun v κ t hv hκ ht => Agg.mass_pos hv hκ ht,
    fun v κ hκ t => Agg.mass_invariant v κ hκ t,
    fun v κ Mc hv hκ => Agg.mass_lagTime hv hκ,
    fun v κ Mc r hv hκ hMc hr => Agg.lag_shift_le hv hκ hMc hr,
    fun v κ Mc r hv hκ hMc hr => Agg.lag_rate_shift_le hv hκ hMc hr,
    fun v κ Mc r hv hκ hMc hr => Agg.lag_shift_ge hv hκ hMc hr,
    fun κ Mc T hκ hMc hT => Agg.lag_underdetermined hκ hMc hT,
    Agg.two_models_one_lag⟩

end IDR
