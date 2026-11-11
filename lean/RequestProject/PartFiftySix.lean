/-
# Part LVI  Aggregation after the early-time regime: depletion, plateau and sigmoid

Part XXXVII solved the nucleation--elongation kinetics in its unsaturated form and stated the
limitation: nothing about monomer depletion, nothing about the plateau.  A measured thioflavin
trace is a sigmoid, and the numbers taken from it -- the lag, the maximal slope, the end point --
live in the saturated part of the curve.  `RequestProject.Depletion` solves the saturated model
exactly.

`IDR.depletion_laws` bundles five statements about the depletion-limited curve
`M(t) = m₀/(1 + e^{-κ(t - t½)})`:

1. *It solves the model*: `M' = κ M (1 - M/m₀)`, so the parameters mean what they are said to
   mean, and the curve is positive, strictly increasing and bounded by the total protein.
2. *The plateau*: `M(t) → m₀`.
3. *Depletion only slows growth*: `M(t) ≤ m₀ e^{κ(t-t½)}` at every time, so the early-time
   treatment of Part XXXVII is an upper bound of known sign.
4. *The threshold identity*: the fraction `f` is reached at `t½ + log(f/(1-f))/κ`, and two
   different detection thresholds give genuinely different crossing times.  A quoted lag time
   is a statement about the instrument as much as about the sample.
5. *The tangent lag determines neither rate*: the steepest tangent (slope `κm₀/4` at `t½`)
   meets the baseline at `t½ - 2/κ`, and for any lag time and any pair of distinct growth rates
   there are half-times giving that same lag with curves that differ.

For a disordered region -- the part of a proteome that aggregates -- the practical reading is
that a lag time is one number extracted from a two-parameter family through a
threshold-dependent construction.  Reporting `κ` and `t½` (or the whole curve) is the only way
to state what was measured.
-/
import Mathlib
import RequestProject.Depletion

set_option autoImplicit false

namespace IDR

open IDR.Depletion
open Filter Topology

/-- **The saturated aggregation laws.**

1. the logistic curve solves `M' = κM(1 - M/m₀)`, and is positive, increasing and below `m₀`;
2. it tends to the plateau `m₀`;
3. it never exceeds the unsaturated exponential;
4. the threshold identity, and the threshold dependence of a lag;
5. the tangent construction, and the fact that a lag time fixes neither rate. -/
theorem depletion_laws :
    (∀ m0 kap th : ℝ, m0 ≠ 0 → ∀ t : ℝ,
        HasDerivAt (fibrilMass m0 kap th)
          (kap * fibrilMass m0 kap th t * (1 - fibrilMass m0 kap th t / m0)) t) ∧
    (∀ m0 kap th : ℝ, 0 < m0 → 0 < kap →
        (∀ t : ℝ, 0 < fibrilMass m0 kap th t ∧ fibrilMass m0 kap th t < m0) ∧
          StrictMono (fibrilMass m0 kap th) ∧
          Tendsto (fibrilMass m0 kap th) atTop (𝓝 m0)) ∧
    (∀ m0 kap th t : ℝ, 0 < m0 →
        fibrilMass m0 kap th t ≤ m0 * Real.exp (kap * (t - th))) ∧
    (∀ m0 kap th f : ℝ, kap ≠ 0 → 0 < f → f < 1 →
        fibrilMass m0 kap th (th + Real.log (f / (1 - f)) / kap) = f * m0) ∧
    (∀ kap f1 f2 : ℝ, 0 < kap → 0 < f1 → f1 < 1 → 0 < f2 → f2 < 1 → f1 ≠ f2 →
        Real.log (f1 / (1 - f1)) / kap ≠ Real.log (f2 / (1 - f2)) / kap) ∧
    (∀ m0 kap th : ℝ, m0 ≠ 0 → kap ≠ 0 →
        HasDerivAt (fibrilMass m0 kap th) (kap * m0 / 4) th ∧
          m0 / 2 + kap * m0 / 4 * (tangentLag kap th - th) = 0) ∧
    (∀ m0 L kap1 kap2 : ℝ, 0 < m0 → 0 < kap1 → 0 < kap2 → kap1 ≠ kap2 →
        ∃ th1 th2 : ℝ, tangentLag kap1 th1 = L ∧ tangentLag kap2 th2 = L ∧
          fibrilMass m0 kap1 th1 th1 ≠ fibrilMass m0 kap2 th2 th1) :=
  ⟨fun _ _ _ hm t => fibrilMass_hasDerivAt hm t,
    fun _ _ _ hm hk =>
      ⟨fun _ => ⟨fibrilMass_pos hm, fibrilMass_lt_total hm⟩, fibrilMass_strictMono hm hk,
        tendsto_fibrilMass_total _ hk⟩,
    fun _ _ _ _ hm => fibrilMass_le_exp hm,
    fun _ _ _ _ hk hf0 hf1 => crossing_eq hk hf0 hf1,
    fun _ _ _ hk hf1 hf1' hf2 hf2' hne => crossing_threshold_shift hk hf1 hf1' hf2 hf2' hne,
    fun _ _ _ hm hk => ⟨tangent_slope _ hm, tangentLag_eq hk⟩,
    fun _ _ _ _ hm hk1 hk2 hne => lag_does_not_determine_rate hm hk1 hk2 hne⟩

end IDR
