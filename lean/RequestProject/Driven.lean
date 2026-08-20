/-
# Part XVIII  Driven: in a living cell the target is not a Gibbs ensemble

Every equilibrium statement in the earlier parts -- Boltzmann populations, the variational free
energy, exponential tilting, linkage -- presupposes detailed balance.  A cell does not supply
it: ATP-driven kinases, chaperones and translocation maintain cycles that run in one direction.
This file proves what that costs, in the smallest system that can show it: a three-state
conformational cycle stepping forward with probability `a` and backward with probability `b`.

* `cycleP_stochastic`, `cycle_stationary_unif` -- the driven cycle is a legitimate kinetics with
  a perfectly ordinary stationary distribution, so the observation theory of Part XV survives
  and `Trajectory.timeAvg_stationary` still applies to it.
* `cycle_current` -- but the steady state carries a probability current `(a − b)/3` around the
  cycle: it is stationary without being at rest.
* `no_detailed_balance_of_driven` -- **and no strictly positive distribution whatsoever
  satisfies detailed balance with it** when `a ≠ b` (Kolmogorov's criterion in its smallest
  instance).  In particular the driven steady state is the reversible equilibrium of no energy
  function at all, so the Boltzmann/free-energy/tilting apparatus of the equilibrium parts does
  not apply to it.
* `driven_needs_kinetics` -- the design statement: for a driven region a model must predict the
  stationary distribution *of a kinetics*, not the equilibrium of a landscape.
-/
import Mathlib
import RequestProject.Kinetics
import RequestProject.Trajectory

namespace IDR

open Finset
open scoped Classical

namespace Driven

/-- A three-state conformational cycle: from each state the chain steps forward with
probability `a`, backward with probability `b`, and stays put with the rest. -/
noncomputable def cycleP (a b : ℝ) : Fin 3 → Fin 3 → ℝ :=
  ![![1 - a - b, a, b], ![b, 1 - a - b, a], ![a, b, 1 - a - b]]

/-- The uniform distribution on the cycle. -/
noncomputable def unif3 : Fin 3 → ℝ := fun _ => 1 / 3

theorem cycleP_stochastic {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b ≤ 1) :
    Kinetics.IsStochastic (cycleP a b) := by
  constructor
  · intro i j
    fin_cases i <;> fin_cases j <;> simp [cycleP] <;> linarith
  · intro i
    fin_cases i
    all_goals simp [cycleP, Fin.sum_univ_three]
    all_goals ring_nf

/-- The uniform distribution is stationary for the driven cycle, whatever the drive. -/
theorem cycle_stationary_unif (a b : ℝ) : Kinetics.Stationary (cycleP a b) unif3 := by
  funext j
  simp only [Kinetics.evolve, unif3, Fin.sum_univ_three]
  fin_cases j <;> simp [cycleP] <;> ring

/-- The steady state carries a current: the net probability flow along each edge of the cycle
is `(a − b)/3`, nonzero exactly when the cycle is driven. -/
theorem cycle_current (a b : ℝ) :
    unif3 0 * cycleP a b 0 1 - unif3 1 * cycleP a b 1 0 = (a - b) / 3 ∧
      unif3 1 * cycleP a b 1 2 - unif3 2 * cycleP a b 2 1 = (a - b) / 3 ∧
      unif3 2 * cycleP a b 2 0 - unif3 0 * cycleP a b 0 2 = (a - b) / 3 := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [cycleP, unif3] <;> ring

/-- **Kolmogorov's criterion, smallest instance.**  If the cycle is driven (`a ≠ b`) then *no*
strictly positive distribution satisfies detailed balance with it: the steady state is the
reversible equilibrium of no energy function, and the entire equilibrium apparatus --
Boltzmann weights, variational free energy, exponential tilting -- does not apply to it. -/
theorem no_detailed_balance_of_driven {a b : ℝ} (hne : a ≠ b) :
    ¬ ∃ pi : Fin 3 → ℝ, (∀ i, 0 < pi i) ∧ Kinetics.DetailedBalance (cycleP a b) pi := by
  rintro ⟨pi, hpos, hDB⟩
  have h01 : pi 0 * a = pi 1 * b := by
    have h := hDB 0 1
    simpa [cycleP] using h
  have h12 : pi 1 * a = pi 2 * b := by
    have h := hDB 1 2
    simpa [cycleP] using h
  have h20 : pi 2 * a = pi 0 * b := by
    have h := hDB 2 0
    simpa [cycleP] using h
  have hposprod : 0 < pi 0 * pi 1 * pi 2 :=
    mul_pos (mul_pos (hpos 0) (hpos 1)) (hpos 2)
  have hmul : (pi 0 * a) * ((pi 1 * a) * (pi 2 * a))
      = (pi 1 * b) * ((pi 2 * b) * (pi 0 * b)) := by
    rw [h01, h12, h20]
  have hprod : (pi 0 * pi 1 * pi 2) * a ^ 3 = (pi 0 * pi 1 * pi 2) * b ^ 3 := by
    linear_combination hmul
  have hcube : a ^ 3 = b ^ 3 := mul_left_cancel₀ hposprod.ne' hprod
  exact hne ((Odd.strictMono_pow (R := ℝ) (n := 3) (by decide)).injective hcube)

/-- **The design statement for a driven region.**  The driven cycle is a perfectly good
kinetics with a stationary distribution -- so time averages still report it
(`Trajectory.timeAvg_stationary`) -- but that distribution is stationary *with a current* and
is the reversible equilibrium of no energy function.  A model of a disordered region in a
living cell must therefore parameterise the kinetics, not only a landscape. -/
theorem driven_needs_kinetics {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b ≤ 1)
    (hne : a ≠ b) :
    Kinetics.IsStochastic (cycleP a b) ∧
      Kinetics.Stationary (cycleP a b) unif3 ∧
      (∀ f : Fin 3 → ℝ, ∀ T : ℕ, 0 < T →
        Trajectory.timeAvg (cycleP a b) unif3 f T = ∑ j, unif3 j * f j) ∧
      unif3 0 * cycleP a b 0 1 - unif3 1 * cycleP a b 1 0 ≠ 0 ∧
      ¬ ∃ pi : Fin 3 → ℝ, (∀ i, 0 < pi i) ∧ Kinetics.DetailedBalance (cycleP a b) pi := by
  refine ⟨cycleP_stochastic ha hb hab, cycle_stationary_unif a b,
    fun f T hT => Trajectory.timeAvg_stationary (cycle_stationary_unif a b) f hT,
    ?_, no_detailed_balance_of_driven hne⟩
  rw [(cycle_current a b).1]
  intro h
  exact hne (by linarith [(div_eq_zero_iff.mp h).resolve_right (by norm_num)])

end Driven

end IDR
