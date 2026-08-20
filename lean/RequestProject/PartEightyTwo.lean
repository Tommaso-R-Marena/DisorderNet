/-
# Part LXXXII  Heterogeneous kinetics: what a single rate constant reports

`RequestProject.Heterokinetics` treats the kinetics of a disordered region the way the earlier
parts treat its structure: as a *distribution*.  Proteolysis, degradation, chemical modification
and labelling proceed from whichever conformations expose the site, so the decaying object is a
mixture of substates with weights `w` and first-order rates `k`, and the surviving fraction is
`surv w k t = sum_j w j exp (-(k j t))`.

`IDR.heterogeneous_kinetics_laws` bundles five statements.

1. *It is a survival curve.*  `surv 0 = 1` and the curve decreases in time.
2. *Heterogeneity looks like stability.*  The mixture always survives at least as well as a
   homogeneous population decaying at the mean rate: a broad rate distribution mimics protection
   with no protective mechanism present.
3. *Only the initial slope reports the average.*  At `t = 0` the apparent rate is exactly the
   population mean rate.
4. *And the apparent rate then falls, monotonically, always*, staying between the slowest and the
   fastest substate rate; the survival curve is log-convex.  The fragile conformations are consumed
   first, so the measured "rate constant" is a property of the ensemble *and* of the measurement
   window.
5. *An explicit instance.*  Half a population at rate `1` and half at rate `1/100` has initial
   apparent rate `101/200`, but by `t = 10` its apparent rate is below `1/10` -- a fivefold change
   in the fitted rate constant with no change in the sample.

The design conclusion matches the structural one: reporting the kinetics of a disordered region
requires reporting a rate *distribution*, just as reporting its structure requires reporting a
conformational distribution.  A single fitted rate constant is not a property of the region.
-/
import Mathlib
import RequestProject.Heterokinetics

set_option autoImplicit false

namespace IDR

open Finset IDR.Hetero

/-- **The heterogeneous-kinetics laws.**

1. the decay curve is a survival function;
2. a mixture survives at least as well as a homogeneous population at the mean rate;
3. the initial apparent rate is the mean rate;
4. the apparent rate decreases with time, is bracketed by the extreme substate rates, and the
   survival curve is log-convex;
5. an explicit two-state instance whose fitted rate constant falls fivefold between `t = 0` and
   `t = 10`. -/
theorem heterogeneous_kinetics_laws {m : ℕ} :
    (∀ w k : Fin m → ℝ, (∀ j, 0 ≤ w j) → (∀ j, 0 ≤ k j) → ∑ j, w j = 1 →
        surv w k 0 = 1 ∧ ∀ t₁ t₂ : ℝ, t₁ ≤ t₂ → surv w k t₂ ≤ surv w k t₁) ∧
    (∀ w k : Fin m → ℝ, (∀ j, 0 ≤ w j) → ∑ j, w j = 1 → ∀ t : ℝ,
        Real.exp (-((∑ j, w j * k j) * t)) ≤ surv w k t) ∧
    (∀ w k : Fin m → ℝ, ∑ j, w j = 1 → apparentRate w k 0 = ∑ j, w j * k j) ∧
    (∀ w k : Fin m → ℝ, (∀ j, 0 ≤ w j) → ∑ j, w j = 1 →
        (∀ t₁ t₂ : ℝ, t₁ ≤ t₂ → apparentRate w k t₂ ≤ apparentRate w k t₁) ∧
        (∀ c : ℝ, (∀ j, c ≤ k j) → ∀ t, c ≤ apparentRate w k t) ∧
        (∀ c : ℝ, (∀ j, k j ≤ c) → ∀ t, apparentRate w k t ≤ c) ∧
        (∀ (t₁ t₂ a : ℝ), 0 ≤ a → a ≤ 1 →
          surv w k (a * t₁ + (1 - a) * t₂) ≤ surv w k t₁ ^ a * surv w k t₂ ^ (1 - a))) ∧
    (apparentRate twoW twoK 0 = 101 / 200 ∧ apparentRate twoW twoK 10 < 1 / 10) :=
  ⟨fun _ _ hw hk hws => ⟨surv_zero hws, fun _ _ h => surv_antitone hw hk h⟩,
    fun _ _ hw hws t => surv_ge_exp_mean hw hws t,
    fun _ _ hws => apparentRate_zero hws,
    fun _ _ hw hws =>
      ⟨fun _ _ h => apparentRate_antitone hw hws h,
        fun _ hc t => apparentRate_ge_min hw hws hc t,
        fun _ hc t => apparentRate_le_max hw hws hc t,
        fun t₁ t₂ _ ha0 ha1 => surv_log_convex hw hws t₁ t₂ ha0 ha1⟩,
    ⟨twoState_rate_zero, twoState_rate_ten⟩⟩

end IDR
