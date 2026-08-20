/-
# Part LVI.1  Beyond early times: monomer depletion, the plateau, and the sigmoid

Part XXXVII solves the nucleation--elongation kinetics in its *unsaturated* form, `M'' = κ²M`,
and says so: it "says nothing about the plateau, about monomer depletion, or about fibril
fragmentation".  A thioflavin trace, however, is a sigmoid, and everything a reader takes from it
-- the lag, the maximal slope, the end point -- is read off the saturated part of the curve.
This file treats the saturated model exactly.

The model is the standard depletion-limited one: the fibril mass `M` grows at a rate
proportional to the mass already present and to the monomer left,

  `M' = κ M (1 - M/m₀)`,

whose solution is the logistic curve `fibrilMass m₀ κ t½ t = m₀/(1 + e^{-κ(t - t½)})`.

* `fibrilMass_hasDerivAt` -- the closed form solves the equation exactly, so `κ` and `t½` mean
  what they are said to mean; `fibrilMass_pos`, `fibrilMass_lt_total`, `fibrilMass_strictMono`
  and `tendsto_fibrilMass_total` give positivity, the bound by the total protein, monotonicity
  and the plateau.
* `fibrilMass_le_exp` -- **depletion only ever slows growth**: the saturated mass is bounded by
  the exponential of the unsaturated model at every time, so the early-time analysis of Part
  XXXVII is an upper bound and not an approximation of unknown sign.
* `crossing_eq` -- **the threshold identity**: the curve reaches the fraction `f` of the total
  at `t½ + log(f/(1-f))/κ`, exactly.
* `crossing_threshold_shift` -- so a quoted lag time is a statement about the *detection
  threshold* as much as about the sample: two thresholds give crossing times differing by
  `log(f₁(1-f₂)/(f₂(1-f₁)))/κ`, which is nonzero whenever the thresholds differ.
* `tangent_slope`, `tangentLag_eq` -- the standard construction: the maximal slope is `κm₀/4`,
  attained at `t½`, and the tangent there meets the baseline at `t½ - 2/κ`.  That is the
  quantity most papers call the lag time.
* `lag_does_not_determine_rate` -- **and it determines neither rate.**  For every lag time and
  every `κ` there is a half-time reproducing it; two models with different `κ` can share a lag
  time exactly and still differ at a time where the curves are compared.  The lag is one number;
  the model has two.

Together with Part XXXVII the position is complete: in the early-time regime the lag is a
logarithm of the nucleation rate, and in the saturated regime it is a two-parameter shadow of the
whole curve.  Neither licenses reading a lag time as a rate.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Depletion

open Real Filter Topology

/-- The depletion-limited (logistic) fibril mass: total protein `m0`, growth rate `kap`,
half-time `th`. -/
noncomputable def fibrilMass (m0 kap th t : ℝ) : ℝ := m0 / (1 + Real.exp (-(kap * (t - th))))

lemma denom_pos (kap th t : ℝ) : 0 < 1 + Real.exp (-(kap * (t - th))) := by
  have := Real.exp_pos (-(kap * (t - th)))
  linarith

/-- **The closed form solves the depletion equation** `M' = κ M (1 - M/m₀)`. -/
theorem fibrilMass_hasDerivAt {m0 kap th : ℝ} (hm : m0 ≠ 0) (t : ℝ) :
    HasDerivAt (fibrilMass m0 kap th)
      (kap * fibrilMass m0 kap th t * (1 - fibrilMass m0 kap th t / m0)) t := by
  have hd : HasDerivAt (fun s : ℝ => -(kap * (s - th))) (-kap) t := by
    simpa using ((hasDerivAt_id t).sub_const th).const_mul kap |>.neg
  have he : HasDerivAt (fun s : ℝ => Real.exp (-(kap * (s - th))))
      (Real.exp (-(kap * (t - th))) * -kap) t := hd.exp
  have h1 : HasDerivAt (fun s : ℝ => 1 + Real.exp (-(kap * (s - th))))
      (Real.exp (-(kap * (t - th))) * -kap) t := by simpa using he.const_add 1
  have hne : (1 + Real.exp (-(kap * (t - th)))) ≠ 0 := ne_of_gt (denom_pos kap th t)
  have hq := (hasDerivAt_const t m0).div h1 hne
  convert hq using 1
  have hE : (0:ℝ) < Real.exp (-(kap * (t - th))) := Real.exp_pos _
  simp only [fibrilMass]
  field_simp
  ring

/-- The mass is positive at every time. -/
theorem fibrilMass_pos {m0 kap th t : ℝ} (hm : 0 < m0) : 0 < fibrilMass m0 kap th t :=
  div_pos hm (denom_pos kap th t)

/-- The mass never reaches the total protein concentration. -/
theorem fibrilMass_lt_total {m0 kap th t : ℝ} (hm : 0 < m0) : fibrilMass m0 kap th t < m0 := by
  rw [fibrilMass, div_lt_iff₀ (denom_pos kap th t)]
  have := Real.exp_pos (-(kap * (t - th)))
  nlinarith

/-- The curve is strictly increasing. -/
theorem fibrilMass_strictMono {m0 kap th : ℝ} (hm : 0 < m0) (hk : 0 < kap) :
    StrictMono (fibrilMass m0 kap th) := by
  intro a b hab
  have hexp : Real.exp (-(kap * (b - th))) < Real.exp (-(kap * (a - th))) := by
    apply Real.exp_lt_exp.mpr
    nlinarith
  rw [fibrilMass, fibrilMass]
  apply div_lt_div_of_pos_left hm (denom_pos kap th b)
  linarith

/-- **The plateau**: the mass tends to the total protein concentration. -/
theorem tendsto_fibrilMass_total {m0 kap : ℝ} (th : ℝ) (hk : 0 < kap) :
    Tendsto (fibrilMass m0 kap th) atTop (𝓝 m0) := by
  have harg : Tendsto (fun t : ℝ => -(kap * (t - th))) atTop atBot := by
    have h : Tendsto (fun t : ℝ => kap * (t - th)) atTop atTop :=
      Filter.Tendsto.const_mul_atTop hk (tendsto_atTop_add_const_right _ (-th) tendsto_id)
    exact tendsto_neg_atTop_atBot.comp h
  have hexp : Tendsto (fun t : ℝ => Real.exp (-(kap * (t - th)))) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp harg
  have hden : Tendsto (fun t : ℝ => 1 + Real.exp (-(kap * (t - th)))) atTop (𝓝 1) := by
    simpa using hexp.const_add 1
  have := (tendsto_const_nhds (x := m0) (f := atTop (α := ℝ))).div hden (by norm_num)
  simpa [fibrilMass] using this

/-- **Depletion only slows growth**: the saturated mass never exceeds the exponential of the
unsaturated model. -/
theorem fibrilMass_le_exp {m0 kap th t : ℝ} (hm : 0 < m0) :
    fibrilMass m0 kap th t ≤ m0 * Real.exp (kap * (t - th)) := by
  rw [fibrilMass, div_le_iff₀ (denom_pos kap th t)]
  have hE : Real.exp (-(kap * (t - th))) * Real.exp (kap * (t - th)) = 1 := by
    rw [← Real.exp_add]; simp
  have hpos : 0 < Real.exp (kap * (t - th)) := Real.exp_pos _
  nlinarith

/-- **The threshold identity**: the fraction `f` of the total is reached at
`t½ + log(f/(1-f))/κ`. -/
theorem crossing_eq {m0 kap th f : ℝ} (hk : kap ≠ 0) (hf0 : 0 < f) (hf1 : f < 1) :
    fibrilMass m0 kap th (th + Real.log (f / (1 - f)) / kap) = f * m0 := by
  have hfr : 0 < f / (1 - f) := by
    apply div_pos hf0; linarith
  have harg : -(kap * (th + Real.log (f / (1 - f)) / kap - th))
      = -Real.log (f / (1 - f)) := by field_simp; ring
  rw [fibrilMass, harg, ← Real.log_inv, Real.exp_log (by positivity)]
  have h1 : (f / (1 - f))⁻¹ = (1 - f) / f := by
    rw [inv_div]
  rw [h1]
  have hne : f ≠ 0 := ne_of_gt hf0
  field_simp
  ring

/-- **A quoted lag time is a statement about the detection threshold.**  Two thresholds give
crossing times that differ, by an amount fixed by the rate. -/
theorem crossing_threshold_shift {kap f1 f2 : ℝ} (hk : 0 < kap) (hf1 : 0 < f1) (hf1' : f1 < 1)
    (hf2 : 0 < f2) (hf2' : f2 < 1) (hne : f1 ≠ f2) :
    Real.log (f1 / (1 - f1)) / kap ≠ Real.log (f2 / (1 - f2)) / kap := by
  intro h
  have hlog : Real.log (f1 / (1 - f1)) = Real.log (f2 / (1 - f2)) := by
    have h2 := congrArg (fun x : ℝ => x * kap) h
    simpa [div_mul_cancel₀ _ (ne_of_gt hk)] using h2
  have hp1 : 0 < f1 / (1 - f1) := div_pos hf1 (by linarith)
  have hp2 : 0 < f2 / (1 - f2) := div_pos hf2 (by linarith)
  have heq := Real.log_injOn_pos (Set.mem_Ioi.mpr hp1) (Set.mem_Ioi.mpr hp2) hlog
  apply hne
  have h1 : (1:ℝ) - f1 ≠ 0 := by linarith
  have h2 : (1:ℝ) - f2 ≠ 0 := by linarith
  field_simp at heq
  linarith

/-- The tangent construction: the lag time is where the steepest tangent meets the baseline. -/
noncomputable def tangentLag (kap th : ℝ) : ℝ := th - 2 / kap

/-- **The maximal slope** `κm₀/4` is attained at the half-time. -/
theorem tangent_slope {m0 kap : ℝ} (th : ℝ) (hm : m0 ≠ 0) :
    HasDerivAt (fibrilMass m0 kap th) (kap * m0 / 4) th := by
  have h := fibrilMass_hasDerivAt (m0 := m0) (kap := kap) (th := th) hm th
  have hval : fibrilMass m0 kap th th = m0 / 2 := by
    simp [fibrilMass]
    norm_num
  rw [hval] at h
  convert h using 1
  field_simp
  ring

/-- **The tangent at the half-time meets the baseline at** `t½ - 2/κ`. -/
theorem tangentLag_eq {m0 kap th : ℝ} (hk : kap ≠ 0) :
    m0 / 2 + kap * m0 / 4 * (tangentLag kap th - th) = 0 := by
  simp only [tangentLag]
  field_simp
  ring

/-- **The lag time determines neither rate.**  For every lag time and every growth rate there is
a half-time reproducing it exactly; two models with different rates share the lag and differ as
curves. -/
theorem lag_does_not_determine_rate {m0 L kap1 kap2 : ℝ} (hm : 0 < m0) (hk1 : 0 < kap1)
    (hk2 : 0 < kap2) (hne : kap1 ≠ kap2) :
    ∃ th1 th2 : ℝ, tangentLag kap1 th1 = L ∧ tangentLag kap2 th2 = L ∧
      fibrilMass m0 kap1 th1 th1 ≠ fibrilMass m0 kap2 th2 th1 := by
  refine ⟨L + 2 / kap1, L + 2 / kap2, by simp [tangentLag], by simp [tangentLag], ?_⟩
  have hthne : L + 2 / kap1 ≠ L + 2 / kap2 := by
    intro h
    apply hne
    have h2 : (2:ℝ) / kap1 = 2 / kap2 := by linarith
    field_simp at h2
    linarith
  have hhalf : fibrilMass m0 kap1 (L + 2 / kap1) (L + 2 / kap1) = m0 / 2 := by
    simp [fibrilMass]
    norm_num
  rw [hhalf]
  intro hcon
  have hmono := fibrilMass_strictMono (m0 := m0) (kap := kap2) (th := L + 2 / kap2) hm hk2
  have hhalf2 : fibrilMass m0 kap2 (L + 2 / kap2) (L + 2 / kap2) = m0 / 2 := by
    simp [fibrilMass]
    norm_num
  rcases lt_trichotomy (L + 2 / kap1) (L + 2 / kap2) with hlt | heq | hgt
  · have := hmono hlt
    rw [hhalf2] at this
    linarith
  · exact hthne heq
  · have := hmono hgt
    rw [hhalf2] at this
    linarith

end Depletion

end IDR
