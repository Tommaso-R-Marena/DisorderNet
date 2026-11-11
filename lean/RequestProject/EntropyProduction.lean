/-
# Part XIX  What the drive costs: entropy production

Part XVIII shows that a driven conformational cycle has a perfectly ordinary stationary
distribution which is nevertheless the reversible equilibrium of no energy function.  This
file quantifies the gap.  For a kinetics `P` with stationary distribution `pi` the
*entropy production rate* is the relative entropy between the forward and the time-reversed
one-step process,

  `epRate P pi = ∑ i ∑ j  pi i * P i j * log (pi i * P i j / (pi j * P j i))`,

written here without division as `flux * (log flux − log reversed flux)`.

* `epRate_nonneg` -- it is never negative (the second law for a Markov chain), proved by
  symmetrising the double sum into `∑ (x − y)(log x − log y)`.
* `epRate_eq_zero_iff_detailedBalance` -- and it vanishes *exactly* at detailed balance.
  Dissipation is not an extra modelling assumption: it is the precise obstruction to
  describing the region by a landscape.
* `cycle_epRate` -- for the three-state cycle of Part XVIII the rate is exactly
  `(a − b) * log (a / b)`, strictly positive whenever the cycle is driven
  (`cycle_epRate_pos`).
* `dissipation_not_determined_by_populations` -- two kinetics with *the same* stationary
  populations, one reversible and one dissipating: the populations a model reports do not
  determine the cell's energy budget.
* `driven_model_needs_a_dissipation_parameter` -- the design statement.
-/
import Mathlib
import RequestProject.Kinetics
import RequestProject.Driven

namespace IDR

open Finset
open scoped Classical

namespace EntropyProduction

variable {n : ℕ}

/-- The one-step probability flux from `i` to `j` in the steady state. -/
noncomputable def flux (P : Fin n → Fin n → ℝ) (pi : Fin n → ℝ) (i j : Fin n) : ℝ :=
  pi i * P i j

/-- The entropy production rate: the relative entropy of the forward one-step process
against its time reverse, in units of `k_B` per step. -/
noncomputable def epRate (P : Fin n → Fin n → ℝ) (pi : Fin n → ℝ) : ℝ :=
  ∑ i, ∑ j, flux P pi i j * (Real.log (flux P pi i j) - Real.log (flux P pi j i))

/-- The elementary inequality behind the second law: `(x − y)(log x − log y) ≥ 0`. -/
lemma sub_mul_log_sub_nonneg {x y : ℝ} (hx : 0 < x) (hy : 0 < y) :
    0 ≤ (x - y) * (Real.log x - Real.log y) := by
  rcases le_total x y with h | h
  · have : Real.log x ≤ Real.log y := Real.log_le_log hx h
    nlinarith
  · have : Real.log y ≤ Real.log x := Real.log_le_log hy h
    exact mul_nonneg (by linarith) (by linarith)

/-- ... and it is strict unless `x = y`. -/
lemma sub_mul_log_sub_pos {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hne : x ≠ y) :
    0 < (x - y) * (Real.log x - Real.log y) := by
  rcases lt_or_gt_of_ne hne with h | h
  · have : Real.log x < Real.log y := Real.log_lt_log hx h
    exact mul_pos_of_neg_of_neg (by linarith) (by linarith)
  · have : Real.log y < Real.log x := Real.log_lt_log hy h
    exact mul_pos (by linarith) (by linarith)

/-- Symmetrising the double sum: twice the entropy production is a sum of manifestly
nonnegative pair terms, one for each ordered pair of states. -/
theorem two_mul_epRate (P : Fin n → Fin n → ℝ) (pi : Fin n → ℝ) :
    2 * epRate P pi =
      ∑ i, ∑ j, (flux P pi i j - flux P pi j i) *
        (Real.log (flux P pi i j) - Real.log (flux P pi j i)) := by
  have hcomm :
      ∑ i, ∑ j, flux P pi j i * (Real.log (flux P pi j i) - Real.log (flux P pi i j))
        = epRate P pi := by
    rw [Finset.sum_comm]
    rfl
  have h1 :
      ∑ i, ∑ j, (flux P pi i j - flux P pi j i) *
          (Real.log (flux P pi i j) - Real.log (flux P pi j i))
        = (∑ i, ∑ j, flux P pi i j * (Real.log (flux P pi i j) - Real.log (flux P pi j i)))
          + ∑ i, ∑ j, flux P pi j i * (Real.log (flux P pi j i) - Real.log (flux P pi i j)) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [h1, hcomm]
  simp only [epRate]
  ring

/-- **The second law for a conformational kinetics.**  The entropy production rate of a
strictly positive kinetics in a strictly positive steady state is never negative. -/
theorem epRate_nonneg {P : Fin n → Fin n → ℝ} {pi : Fin n → ℝ}
    (hP : ∀ i j, 0 < P i j) (hpi : ∀ i, 0 < pi i) : 0 ≤ epRate P pi := by
  have hflux : ∀ i j, 0 < flux P pi i j := fun i j => mul_pos (hpi i) (hP i j)
  have h : 0 ≤ 2 * epRate P pi := by
    rw [two_mul_epRate]
    exact Finset.sum_nonneg fun i _ => Finset.sum_nonneg fun j _ =>
      sub_mul_log_sub_nonneg (hflux i j) (hflux j i)
  linarith

/-- **Dissipation is exactly the failure of detailed balance.**  The entropy production
rate vanishes if and only if the kinetics is reversible with respect to its steady state --
that is, if and only if the region *is* described by a landscape. -/
theorem epRate_eq_zero_iff_detailedBalance {P : Fin n → Fin n → ℝ} {pi : Fin n → ℝ}
    (hP : ∀ i j, 0 < P i j) (hpi : ∀ i, 0 < pi i) :
    epRate P pi = 0 ↔ Kinetics.DetailedBalance P pi := by
  have hflux : ∀ i j, 0 < flux P pi i j := fun i j => mul_pos (hpi i) (hP i j)
  constructor
  · intro h0 i j
    have h2 : ∑ i, ∑ j, (flux P pi i j - flux P pi j i) *
        (Real.log (flux P pi i j) - Real.log (flux P pi j i)) = 0 := by
      rw [← two_mul_epRate, h0, mul_zero]
    have hinner : ∀ i ∈ (Finset.univ : Finset (Fin n)),
        0 ≤ ∑ j, (flux P pi i j - flux P pi j i) *
          (Real.log (flux P pi i j) - Real.log (flux P pi j i)) := fun i _ =>
      Finset.sum_nonneg fun j _ => sub_mul_log_sub_nonneg (hflux i j) (hflux j i)
    have hi := (Finset.sum_eq_zero_iff_of_nonneg hinner).mp h2 i (Finset.mem_univ i)
    have hij := (Finset.sum_eq_zero_iff_of_nonneg
      (fun j (_ : j ∈ (Finset.univ : Finset (Fin n))) =>
        sub_mul_log_sub_nonneg (hflux i j) (hflux j i))).mp hi j (Finset.mem_univ j)
    by_contra hne
    have : flux P pi i j ≠ flux P pi j i := hne
    exact absurd hij (sub_mul_log_sub_pos (hflux i j) (hflux j i) this).ne'
  · intro hDB
    refine Finset.sum_eq_zero fun i _ => Finset.sum_eq_zero fun j _ => ?_
    have : flux P pi i j = flux P pi j i := hDB i j
    rw [this, sub_self, mul_zero]

/-! ## The three-state driven cycle of Part XVIII -/

/-- The entropy production of the driven cycle is exactly `(a − b) * log (a / b)`: the
housekeeping heat the cell must pay, per step, to hold the region away from equilibrium. -/
theorem cycle_epRate {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    epRate (Driven.cycleP a b) Driven.unif3 = (a - b) * (Real.log a - Real.log b) := by
  have la : Real.log (1 / 3 * a) = Real.log (1 / 3) + Real.log a :=
    Real.log_mul (by norm_num) ha.ne'
  have lb : Real.log (1 / 3 * b) = Real.log (1 / 3) + Real.log b :=
    Real.log_mul (by norm_num) hb.ne'
  have hf : ∀ i j : Fin 3, flux (Driven.cycleP a b) Driven.unif3 i j
      = 1 / 3 * Driven.cycleP a b i j := fun _ _ => rfl
  have e00 : Driven.cycleP a b 0 0 = 1 - a - b := rfl
  have e01 : Driven.cycleP a b 0 1 = a := rfl
  have e02 : Driven.cycleP a b 0 2 = b := rfl
  have e10 : Driven.cycleP a b 1 0 = b := rfl
  have e11 : Driven.cycleP a b 1 1 = 1 - a - b := rfl
  have e12 : Driven.cycleP a b 1 2 = a := rfl
  have e20 : Driven.cycleP a b 2 0 = a := rfl
  have e21 : Driven.cycleP a b 2 1 = b := rfl
  have e22 : Driven.cycleP a b 2 2 = 1 - a - b := rfl
  simp only [epRate, Fin.sum_univ_three, hf, e00, e01, e02, e10, e11, e12, e20, e21, e22,
    la, lb]
  ring

/-- A driven cycle dissipates at a strictly positive rate. -/
theorem cycle_epRate_pos {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hne : a ≠ b) :
    0 < epRate (Driven.cycleP a b) Driven.unif3 := by
  rw [cycle_epRate ha hb]
  exact sub_mul_log_sub_pos ha hb hne

/-- An undriven cycle dissipates nothing. -/
theorem cycle_epRate_eq_zero {c : ℝ} (hc : 0 < c) :
    epRate (Driven.cycleP c c) Driven.unif3 = 0 := by
  rw [cycle_epRate hc hc]; ring

/-- **Populations do not determine dissipation.**  Two kinetics on the same three states,
both stochastic, both with the *same* stationary distribution -- and one of them reversible
with zero entropy production while the other dissipates at a strictly positive rate.  No
functional of the reported populations can recover the energy budget. -/
theorem dissipation_not_determined_by_populations :
    ∃ P Q : Fin 3 → Fin 3 → ℝ,
      Kinetics.IsStochastic P ∧ Kinetics.IsStochastic Q ∧
      Kinetics.Stationary P Driven.unif3 ∧ Kinetics.Stationary Q Driven.unif3 ∧
      epRate P Driven.unif3 = 0 ∧ 0 < epRate Q Driven.unif3 := by
  refine ⟨Driven.cycleP (1 / 4) (1 / 4), Driven.cycleP (1 / 2) (1 / 4),
    Driven.cycleP_stochastic (by norm_num) (by norm_num) (by norm_num),
    Driven.cycleP_stochastic (by norm_num) (by norm_num) (by norm_num),
    Driven.cycle_stationary_unif _ _, Driven.cycle_stationary_unif _ _,
    cycle_epRate_eq_zero (by norm_num),
    cycle_epRate_pos (by norm_num) (by norm_num) (by norm_num)⟩

/-- **The design statement.**  For a region held out of equilibrium the kinetics carries a
parameter that no equilibrium description contains: the entropy production rate.  It is
nonnegative always, zero exactly at detailed balance -- so a landscape model is committed to
predicting zero dissipation -- and the stationary populations do not determine it. -/
theorem driven_model_needs_a_dissipation_parameter :
    (∀ (m : ℕ) (P : Fin m → Fin m → ℝ) (pi : Fin m → ℝ),
        (∀ i j, 0 < P i j) → (∀ i, 0 < pi i) → 0 ≤ epRate P pi) ∧
      (∀ (m : ℕ) (P : Fin m → Fin m → ℝ) (pi : Fin m → ℝ),
        (∀ i j, 0 < P i j) → (∀ i, 0 < pi i) →
          (epRate P pi = 0 ↔ Kinetics.DetailedBalance P pi)) ∧
      (∀ a b : ℝ, 0 < a → 0 < b → a ≠ b →
        0 < epRate (Driven.cycleP a b) Driven.unif3) ∧
      (∃ P Q : Fin 3 → Fin 3 → ℝ,
        Kinetics.IsStochastic P ∧ Kinetics.IsStochastic Q ∧
        Kinetics.Stationary P Driven.unif3 ∧ Kinetics.Stationary Q Driven.unif3 ∧
        epRate P Driven.unif3 = 0 ∧ 0 < epRate Q Driven.unif3) :=
  ⟨fun _ _ _ hP hpi => epRate_nonneg hP hpi,
    fun _ _ _ hP hpi => epRate_eq_zero_iff_detailedBalance hP hpi,
    fun _ _ ha hb hne => cycle_epRate_pos ha hb hne,
    dissipation_not_determined_by_populations⟩

end EntropyProduction

end IDR
