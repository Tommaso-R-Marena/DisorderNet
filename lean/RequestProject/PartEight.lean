/-
# Part VIII capstone: the temporal design laws

Parts I--VII treat the target as an equilibrium ensemble and the data as independent draws
from it.  Part VIII prices those two idealisations on the smallest system that has them, a
two-state exchange, and the conclusion is that both costs are set by one number: the
relaxation time of the slowest interconversion in the disordered region.

`temporal_design_laws` bundles five clauses:

1. **Exact relaxation.**  The deviation of a two-state exchange from its stationary
   population decays geometrically with ratio `lam = 1 - a - b`.
2. **The run must be at least as long as the relaxation time.**  A trajectory started a
   distance `d` from equilibrium that reports the populations to accuracy `eps` must satisfy
   `1 - eps/d ≤ t·(1 - lam)`.
3. **The cost of a barrier is exponential.**  Writing the exchange rate as `exp (-B)` for a
   barrier `B` in units of `kT`, the required run length is at least `(1 - eps/d)·exp B`.
4. **An unequilibrated run is a modelling error.**  Its populations are at `ℓ¹` distance
   `2·lam^t·d` from the truth, in the same operational metric that Part III uses to size the
   model.
5. **Frames are not independent inside a relaxation time.**  The two-time correlation decays
   at the same rate `lam^t`, and at lags shorter than half a relaxation time at least half
   the variance survives -- so the independent frames counted by the tensorisation law of
   Part VI are `T/(2·tau)`, not the number of stored snapshots.

With `IDR.model_must_be` and `IDR.model_cannot_be` (Part II),
`IDR.quantitative_design_laws` (Part III), `IDR.physical_design_laws` (Part IV),
`IDR.statistical_design_laws` (Part V), `IDR.precision_design_laws` (Part VI) and
`IDR.collective_design_laws` (Part VII), the specification now also carries the cost of
obtaining the data in the first place.
-/
import Mathlib
import RequestProject.Relaxation
import RequestProject.PartSeven

namespace IDR

/-- **The temporal design laws for a model of an intrinsically disordered region.**
Each clause is an instance of a theorem proved in Part VIII. -/
theorem temporal_design_laws :
    -- (1) exact geometric relaxation of a two-state exchange
    (∀ a b : ℝ, a + b ≠ 0 → ∀ (p : ℝ) (t : ℕ),
        (Relax.twoStep a b)^[t] p - Relax.stat a b
          = (1 - a - b) ^ t * (p - Relax.stat a b)) ∧
    -- (2) the run must be at least as long as the relaxation time
    (∀ lam d eps : ℝ, 0 ≤ lam → 0 < d → ∀ t : ℕ,
        lam ^ t * d ≤ eps → 1 - eps / d ≤ t * (1 - lam)) ∧
    -- (3) and exponentially long in the barrier height
    (∀ B d eps : ℝ, 0 < d → 0 ≤ B → ∀ t : ℕ,
        (1 - Real.exp (-B)) ^ t * d ≤ eps → (1 - eps / d) * Real.exp B ≤ t) ∧
    -- (4) an unequilibrated run is wrong in the operational metric
    (∀ a b : ℝ, a + b ≠ 0 → ∀ (p : ℝ) (t : ℕ),
        |(Relax.twoStep a b)^[t] p - Relax.stat a b|
            + |(1 - (Relax.twoStep a b)^[t] p) - (1 - Relax.stat a b)|
          = 2 * |(1 - a - b) ^ t| * |p - Relax.stat a b|) ∧
    -- (5) the correlation decays at the same rate, so frames must be spaced by it
    ((∀ a b : ℝ, a + b ≠ 0 → ∀ t : ℕ,
        Relax.stat a b * ((Relax.twoStep a b)^[t] 1 - Relax.stat a b)
          = Relax.stat a b * (1 - Relax.stat a b) * (1 - a - b) ^ t) ∧
      (∀ lam sig : ℝ, 0 ≤ lam → 0 ≤ sig → ∀ t : ℕ,
        t * (1 - lam) ≤ 1/2 → sig / 2 ≤ sig * lam ^ t)) := by
  exact ⟨fun a b hab p t => Relax.twoStep_iterate_sub_stat hab p t,
    fun lam d eps hlam hd t h => Relax.relaxation_lower_bound hlam hd t h,
    fun B d eps hd hB t h => Relax.barrier_cost hd hB t h,
    fun a b hab p t => Relax.ell1_error_of_unequilibrated hab p t,
    ⟨fun a b hab t => Relax.autocorrelation hab t,
      fun lam sig hlam hsig t ht => Relax.frames_correlated_within_relaxation_time hlam hsig t ht⟩⟩

end IDR
