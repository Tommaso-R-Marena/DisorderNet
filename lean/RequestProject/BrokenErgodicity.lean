/-
# Part XXV.3  Broken ergodicity: a finite simulation of a slow isomerisation is not the
# Boltzmann ensemble

The ensemble of Part XXIV is the equilibrium (Boltzmann-Gibbs) measure.  A sampler returns
it only in the limit of infinite time, and a disordered region has degrees of freedom -- the
classic one is cis/trans isomerisation of a prolyl bond -- whose interconversion rate is
many orders of magnitude slower than any accessible simulation.  This file makes the
statement exact for the two-state kinetics such a degree of freedom obeys.

* `popB` solves the master equation `p' = k1 (1 - p) - k2 p` exactly (`popB_hasDerivAt`,
  `popB_zero`), and converges to the Boltzmann population `k1/(k1+k2)`
  (`popB_tendsto_equilibrium`).
* `popB_stuck` -- the quantitative failure: whenever the total rate times the horizon is at
  most `delta`, the state at the end of the run still carries a fraction `1 - delta` of its
  initial deviation from equilibrium.
* `timeAverage_eq` computes the running average exactly, and `timeAverage_stuck` shows the
  *average over the whole trajectory* is stuck by the same argument: no post-processing of a
  short run recovers the equilibrium population.
* `finite_run_not_boltzmann` -- for every horizon and every tolerance there is a barrier slow
  enough that the run misreports the population by nearly the whole gap.
-/
import Mathlib

namespace IDR

namespace BrokenErgodicity

open Real Filter Topology intervalIntegral

variable {k1 k2 : ℝ}

/-- The derivative of `s ↦ exp (c s)`. -/
lemma exp_lin_hasDerivAt (c t : ℝ) :
    HasDerivAt (fun s : ℝ => Real.exp (c * s)) (c * Real.exp (c * t)) t := by
  have h := ((hasDerivAt_id t).const_mul c).exp
  simpa [mul_comm] using h

/-- The equilibrium (Boltzmann) population of state `B` for rates `k1 : A -> B` and
`k2 : B -> A`. -/
noncomputable def peq (k1 k2 : ℝ) : ℝ := k1 / (k1 + k2)

/-- The solution of the two-state master equation with initial population `p0` of `B`. -/
noncomputable def popB (k1 k2 p0 t : ℝ) : ℝ :=
  peq k1 k2 + (p0 - peq k1 k2) * Real.exp (-(k1 + k2) * t)

@[simp] lemma popB_zero (k1 k2 p0 : ℝ) : popB k1 k2 p0 0 = p0 := by
  unfold popB; simp

/-- **The master equation is solved exactly.** -/
theorem popB_hasDerivAt (hk : 0 < k1 + k2) (p0 t : ℝ) :
    HasDerivAt (popB k1 k2 p0)
      (k1 * (1 - popB k1 k2 p0 t) - k2 * popB k1 k2 p0 t) t := by
  have hbase := exp_lin_hasDerivAt (-(k1 + k2)) t
  have h := (hbase.const_mul (p0 - peq k1 k2)).const_add (peq k1 k2)
  have hval : k1 * (1 - popB k1 k2 p0 t) - k2 * popB k1 k2 p0 t
      = (p0 - peq k1 k2) * (-(k1 + k2) * Real.exp (-(k1 + k2) * t)) := by
    unfold popB peq
    field_simp
    ring
  rw [hval]
  exact h

/-- The population converges to the Boltzmann value -- in infinite time. -/
theorem popB_tendsto_equilibrium (hk : 0 < k1 + k2) (p0 : ℝ) :
    Tendsto (popB k1 k2 p0) atTop (𝓝 (peq k1 k2)) := by
  have hexp : Tendsto (fun t : ℝ => Real.exp (-(k1 + k2) * t)) atTop (𝓝 0) := by
    have hlin : Tendsto (fun t : ℝ => (k1 + k2) * t) atTop atTop :=
      Filter.Tendsto.const_mul_atTop hk tendsto_id
    have hneg : Tendsto (fun t : ℝ => -(k1 + k2) * t) atTop atBot :=
      (tendsto_neg_atTop_atBot.comp hlin).congr (fun t => by simp; ring)
    exact Real.tendsto_exp_atBot.comp hneg
  have hmain := (hexp.const_mul (p0 - peq k1 k2)).const_add (peq k1 k2)
  unfold popB
  simpa using hmain

/-- **A slow degree of freedom does not relax.**  If the total rate times the horizon is at
most `delta`, the deviation from equilibrium at the end of the run is at least `1 - delta`
times the initial deviation. -/
theorem popB_stuck (p0 : ℝ) {T delta : ℝ}
    (hd : (k1 + k2) * T ≤ delta) :
    (1 - delta) * |p0 - peq k1 k2| ≤ |popB k1 k2 p0 T - peq k1 k2| := by
  have hx : Real.exp (-(k1 + k2) * T) ≥ 1 - (k1 + k2) * T := by
    have := Real.add_one_le_exp (-(k1 + k2) * T)
    linarith
  have hpos : (0:ℝ) < Real.exp (-(k1 + k2) * T) := Real.exp_pos _
  have habs : |popB k1 k2 p0 T - peq k1 k2|
      = |p0 - peq k1 k2| * Real.exp (-(k1 + k2) * T) := by
    unfold popB
    rw [add_sub_cancel_left, abs_mul, abs_of_pos hpos]
  rw [habs]
  have hge : 1 - delta ≤ Real.exp (-(k1 + k2) * T) := by linarith
  calc (1 - delta) * |p0 - peq k1 k2| = |p0 - peq k1 k2| * (1 - delta) := by ring
    _ ≤ |p0 - peq k1 k2| * Real.exp (-(k1 + k2) * T) :=
        mul_le_mul_of_nonneg_left hge (abs_nonneg _)

/-- The trajectory average of the population over `[0, T]`, computed exactly. -/
theorem timeAverage_eq (hk : 0 < k1 + k2) (p0 : ℝ) {T : ℝ} (hT : 0 < T) :
    (∫ t in (0:ℝ)..T, popB k1 k2 p0 t) / T
      = peq k1 k2 + (p0 - peq k1 k2) * (1 - Real.exp (-(k1 + k2) * T)) / ((k1 + k2) * T) := by
  have hprim : ∀ t : ℝ, HasDerivAt
      (fun s : ℝ => peq k1 k2 * s
        + (p0 - peq k1 k2) * (-(Real.exp (-(k1 + k2) * s)) / (k1 + k2)))
      (popB k1 k2 p0 t) t := by
    intro t
    have hbase := exp_lin_hasDerivAt (-(k1 + k2)) t
    have h1 : HasDerivAt (fun s : ℝ => peq k1 k2 * s) (peq k1 k2) t := by
      simpa using (hasDerivAt_id t).const_mul (peq k1 k2)
    have h2 : HasDerivAt
        (fun s : ℝ => (p0 - peq k1 k2) * (-(Real.exp (-(k1 + k2) * s)) / (k1 + k2)))
        ((p0 - peq k1 k2) * (-(-(k1 + k2) * Real.exp (-(k1 + k2) * t)) / (k1 + k2))) t :=
      ((hbase.neg.div_const (k1 + k2)).const_mul (p0 - peq k1 k2))
    have hsum := h1.add h2
    have hval : peq k1 k2
        + (p0 - peq k1 k2) * (-(-(k1 + k2) * Real.exp (-(k1 + k2) * t)) / (k1 + k2))
        = popB k1 k2 p0 t := by
      unfold popB
      field_simp
    rwa [hval] at hsum
  have hcont : IntervalIntegrable (popB k1 k2 p0) MeasureTheory.volume 0 T := by
    apply Continuous.intervalIntegrable
    unfold popB
    fun_prop
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt (fun t _ => hprim t) hcont]
  simp only [mul_zero, Real.exp_zero]
  field_simp
  ring

/-- **Averaging does not help.**  The trajectory average over a short run is stuck at the
initial population by the same bound, since `(1 - e^{-x})/x >= e^{-x} >= 1 - x`. -/
theorem timeAverage_stuck (hk : 0 < k1 + k2) (p0 : ℝ) {T delta : ℝ} (hT : 0 < T)
    (hd : (k1 + k2) * T ≤ delta) :
    (1 - delta) * |p0 - peq k1 k2|
      ≤ |(∫ t in (0:ℝ)..T, popB k1 k2 p0 t) / T - peq k1 k2| := by
  set x : ℝ := (k1 + k2) * T with hx
  have hxpos : 0 < x := mul_pos hk hT
  have hlin : 1 - x ≤ Real.exp (-x) := by
    have := Real.add_one_le_exp (-x)
    linarith
  have hquot : Real.exp (-x) ≤ (1 - Real.exp (-x)) / x := by
    have h1 : x + 1 ≤ Real.exp x := Real.add_one_le_exp x
    have hee : Real.exp (-x) * Real.exp x = 1 := by
      rw [← Real.exp_add]; simp
    have hpos : 0 < Real.exp (-x) := Real.exp_pos _
    rw [le_div_iff₀ hxpos]
    nlinarith [hpos, h1, hee]
  have hrw := timeAverage_eq hk p0 hT
  rw [hrw]
  have hsimp : peq k1 k2 + (p0 - peq k1 k2) * (1 - Real.exp (-x)) / x - peq k1 k2
      = (p0 - peq k1 k2) * ((1 - Real.exp (-x)) / x) := by ring
  rw [show -(k1 + k2) * T = -x by rw [hx]; ring, hsimp, abs_mul]
  have hqpos : 0 < (1 - Real.exp (-x)) / x := by
    have : Real.exp (-x) < 1 := by
      rw [show (1:ℝ) = Real.exp 0 by simp]
      exact Real.exp_lt_exp.mpr (by linarith)
    exact div_pos (by linarith) hxpos
  rw [abs_of_pos hqpos]
  have hge : 1 - delta ≤ (1 - Real.exp (-x)) / x := by
    have hxd : x ≤ delta := hd
    linarith
  calc (1 - delta) * |p0 - peq k1 k2| = |p0 - peq k1 k2| * (1 - delta) := by ring
    _ ≤ |p0 - peq k1 k2| * ((1 - Real.exp (-x)) / x) :=
        mul_le_mul_of_nonneg_left hge (abs_nonneg _)

/-- **A finite run of a slow isomerisation misreports the population.**  For every horizon
`T` and every tolerance there are rates -- an arbitrarily high barrier -- for which the
population at the end of the run still differs from the Boltzmann population by nearly the
whole initial gap. -/
theorem finite_run_not_boltzmann {T eps : ℝ} (hT : 0 < T) (heps : 0 < eps) :
    ∃ k1 k2 : ℝ, 0 < k1 ∧ 0 < k2 ∧ peq k1 k2 = 1/2 ∧
      (1/2 - eps) ≤ |popB k1 k2 1 T - peq k1 k2| := by
  obtain ⟨k, hk0, hk⟩ : ∃ k : ℝ, 0 < k ∧ 2 * k * T ≤ 2 * eps := by
    refine ⟨min 1 (eps / T), lt_min one_pos (div_pos heps hT), ?_⟩
    have h1 : min 1 (eps / T) ≤ eps / T := min_le_right _ _
    have : min 1 (eps / T) * T ≤ eps := by
      rw [← le_div_iff₀ hT]; exact h1
    linarith
  refine ⟨k, k, hk0, hk0, ?_, ?_⟩
  · unfold peq
    rw [div_eq_iff (by linarith)]
    ring
  · have hstuck := popB_stuck (k1 := k) (k2 := k) 1 (T := T)
      (delta := 2 * eps) (by linarith)
    have hpeq : peq k k = 1/2 := by
      unfold peq; rw [div_eq_iff (by linarith)]; ring
    rw [hpeq] at hstuck ⊢
    have habs : |(1:ℝ) - 1/2| = 1/2 := by rw [show (1:ℝ) - 1/2 = 1/2 by norm_num]; norm_num
    rw [habs] at hstuck
    linarith

end BrokenErgodicity

end IDR
