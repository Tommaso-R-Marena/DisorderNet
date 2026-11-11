/-
# Part IX.4  Temperature: heat capacity, van 't Hoff, and cold denaturation

A disordered region is not a fixed object: its ensemble is a function of temperature, and
the temperature dependence is the most-measured thermodynamic signature it has.  This file
adds the temperature axis to the development.

**Energy fluctuations are the heat capacity.**  Taking the inverse temperature `beta` as the
tilting parameter of `RequestProject.Response` makes the Boltzmann family a one-parameter
exponential family in `beta`, so the exact response identity of that file becomes the
standard fluctuation formula:

* `hasDerivAt_meanE` -- `d⟨E⟩/dβ = -Var(E)`;
* `heatCapacity_eq_variance`, `heatCapacity_nonneg` -- `C = k β²·Var(E) ≥ 0`, thermodynamic
  stability as an *identity*, not an assumption;
* `meanE_antitone` -- the mean energy rises with temperature;
* `rigid_zero_heatCapacity` and `heatCapacity_pos_of_disordered` -- a model with a single
  conformation (or any model that gets the energy fluctuation wrong) predicts zero (wrong)
  heat capacity.  The calorimetric signal of a disordered region is exactly its
  conformational fluctuation, so calorimetry is a direct measurement of the quantity a model
  of a disordered region exists to reproduce.

**van 't Hoff.**  `log_pop_ratio` -- populations of two conformations satisfy
`log(p_i/p_j) = -β(E_i - E_j)` exactly, and `vantHoff` -- the slope of the van 't Hoff plot
is minus the energy gap.  So a two-state analysis of any experiment returns an energy
difference, never a structure.

**Cold denaturation.**  Burial of hydrophobic surface gives a *positive* heat-capacity change
`ΔCp`, and the Gibbs--Helmholtz function

  `ΔG(T) = ΔH₀ - T·ΔS₀ + ΔCp·[(T - T₀) - T·log(T/T₀)]`   (`gibbsHelmholtz`)

is then strictly concave in `T` (`gibbsHelmholtz_strictConcaveOn`).  Two consequences, both
proved:

* `denaturation_two_temperatures` -- if the state is stable at some `Tm` and unstable at a
  lower and at a higher temperature, there are transition temperatures on *both* sides: cold
  denaturation is forced, not an anomaly;
* `no_three_transition_temperatures` -- and there are never three.

The design conclusion is that a model of a disordered region must take the temperature as an
input and reproduce a *curve*; a model trained at one temperature is not a model of the
region, and the curvature of that curve is set by the same fluctuations that make the region
disordered.
-/
import Mathlib
import RequestProject.Response

namespace IDR

open Finset

namespace Thermo

variable {n : ℕ}

/-! ## Energy fluctuations and the heat capacity -/

/-- Reversing the sign of an observable reverses the sign of its mean. -/
lemma meanObs_neg (q A U : Fin n → ℝ) (lam : ℝ) :
    Response.meanObs q A (fun j => -U j) lam = -Response.meanObs q A U lam := by
  simp only [Response.meanObs]
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl (fun i _ => by ring)

/-- Reversing the sign of an observable does not change its variance. -/
lemma var_neg (q A U : Fin n → ℝ) (lam : ℝ) :
    Response.var q A (fun j => -U j) lam = Response.var q A U lam := by
  simp only [Response.var, Response.cov, meanObs_neg]
  exact Finset.sum_congr rfl (fun j _ => by ring)

/-- The covariance of an observable with its own negative is minus its variance. -/
lemma cov_neg_self (q A U : Fin n → ℝ) (lam : ℝ) :
    Response.cov q A U (fun j => -U j) lam = -Response.var q A U lam := by
  simp only [Response.var, Response.cov, meanObs_neg]
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl (fun j _ => by ring)

/-- The uniform reference ensemble on the conformational library. -/
noncomputable def unif (n : ℕ) : Fin n → ℝ := fun _ => 1 / (n : ℝ)

lemma unif_pos {n : ℕ} (hn : 0 < n) : ∀ j, 0 < unif n j := by
  intro j
  have : (0 : ℝ) < n := by exact_mod_cast hn
  simpa [unif] using (by positivity : (0:ℝ) < 1 / (n:ℝ))

/-- The Boltzmann mean energy at inverse temperature `beta`, as a function of `beta`. -/
noncomputable def meanE (U : Fin n → ℝ) (beta : ℝ) : ℝ :=
  Response.meanObs (unif n) (fun j => -U j) U beta

/-- The Boltzmann energy variance at inverse temperature `beta`. -/
noncomputable def varE (U : Fin n → ℝ) (beta : ℝ) : ℝ :=
  Response.var (unif n) (fun j => -U j) U beta

/-- The Boltzmann family in `beta` is the exponential tilt of the uniform ensemble by the
energy, so at `beta` its weights are the Boltzmann weights of `U`. -/
lemma tilted_eq_boltz (hn : 0 < n) (U : Fin n → ℝ) (beta : ℝ) :
    Response.tilted (unif n) (fun j => -U j) beta = FreeEnergy.boltz beta U := by
  have h := Response.tilted_eq_boltz_of_unif (n := n) hn beta U
  funext j
  have hpart : Response.part (unif n) (fun j => -U j) beta
      = Response.part (fun _ => 1 / (n:ℝ)) (fun j => -beta * U j) 1 := by
    simp only [Response.part, unif, one_mul]
    exact Finset.sum_congr rfl (fun i _ => by ring_nf)
  have : Response.tilted (unif n) (fun j => -U j) beta j
      = Response.tilted (fun _ => 1 / (n:ℝ)) (fun j => -beta * U j) 1 j := by
    simp only [Response.tilted, unif, hpart, one_mul]
    congr 2
    ring_nf
  rw [this, h]

/-- **The exact fluctuation formula.**  `d⟨E⟩/dβ = -Var(E)`. -/
theorem hasDerivAt_meanE (hn : 0 < n) (U : Fin n → ℝ) (beta : ℝ) :
    HasDerivAt (meanE U) (-varE U beta) beta := by
  have h := Response.linear_response (q := unif n) (A := fun j => -U j) (f := U) hn
    (unif_pos hn) beta
  rw [cov_neg_self] at h
  exact h

/-- The heat capacity in units of Boltzmann's constant, `C/k = β²·Var(E)`. -/
noncomputable def heatCapacity (U : Fin n → ℝ) (beta : ℝ) : ℝ := beta ^ 2 * varE U beta

/-- **The heat capacity is the energy fluctuation.**  `C/k = -β²·d⟨E⟩/dβ = β²·Var(E)`. -/
theorem heatCapacity_eq_variance (hn : 0 < n) (U : Fin n → ℝ) (beta : ℝ) :
    heatCapacity U beta = -beta ^ 2 * deriv (meanE U) beta := by
  rw [(hasDerivAt_meanE hn U beta).deriv, heatCapacity]
  ring

/-- **Thermodynamic stability.**  The heat capacity is nonnegative -- as an identity, not an
assumption. -/
theorem heatCapacity_nonneg (hn : 0 < n) (U : Fin n → ℝ) (beta : ℝ) :
    0 ≤ heatCapacity U beta := by
  have hv : 0 ≤ varE U beta := Response.var_nonneg hn (unif_pos hn) beta
  unfold heatCapacity
  positivity

/-- The mean energy decreases with `beta`, i.e. increases with temperature. -/
theorem meanE_antitone (hn : 0 < n) (U : Fin n → ℝ) : Antitone (meanE U) := by
  have hderiv : ∀ b : ℝ, HasDerivAt (meanE U) (-varE U b) b := hasDerivAt_meanE hn U
  have hdiff : Differentiable ℝ (meanE U) := fun b => (hderiv b).differentiableAt
  refine antitone_of_deriv_nonpos hdiff (fun b => ?_)
  rw [(hderiv b).deriv]
  simpa using Response.var_nonneg (q := unif n) (A := fun j => -U j) (f := U) hn (unif_pos hn) b

/-- A rigid (single-energy) model has no heat capacity at all. -/
theorem rigid_zero_heatCapacity (hn : 0 < n) {U : Fin n → ℝ} {c : ℝ} (hconst : ∀ j, U j = c)
    (beta : ℝ) : heatCapacity U beta = 0 := by
  have h : Response.var (unif n) (fun j => -U j) (fun j => -U j) beta = 0 :=
    Response.rigid_no_response hn (unif_pos hn) beta (-c) (fun j => by rw [hconst j])
  rw [var_neg] at h
  unfold heatCapacity varE
  rw [h, mul_zero]

/-- **Calorimetry sees the disorder.**  A library with two conformations of different energy
has a strictly positive heat capacity at every temperature. -/
theorem heatCapacity_pos_of_disordered (hn : 0 < n) {U : Fin n → ℝ} {j₁ j₂ : Fin n}
    (hne : U j₁ ≠ U j₂) {beta : ℝ} (hbeta : beta ≠ 0) : 0 < heatCapacity U beta := by
  have hA : (fun j => -U j) j₁ ≠ (fun j => -U j) j₂ := by
    simpa using fun h => hne (by linarith [h])
  have h := Response.response_of_disordered (q := unif n) (A := fun j => -U j) hn
    (unif_pos hn) beta hA
  rw [var_neg] at h
  unfold heatCapacity varE
  have : 0 < beta ^ 2 := by positivity
  exact mul_pos this h

/-! ## van 't Hoff -/

/-- **The exact van 't Hoff relation.**  The log ratio of two Boltzmann populations is minus
`beta` times their energy gap. -/
theorem log_pop_ratio (hn : 0 < n) (beta : ℝ) (U : Fin n → ℝ) (i j : Fin n) :
    Real.log (FreeEnergy.boltz beta U i / FreeEnergy.boltz beta U j)
      = -beta * (U i - U j) := by
  have hi : 0 < FreeEnergy.boltz beta U i := FreeEnergy.boltz_pos hn beta U i
  have hj : 0 < FreeEnergy.boltz beta U j := FreeEnergy.boltz_pos hn beta U j
  rw [Real.log_div hi.ne' hj.ne', FreeEnergy.log_boltz hn, FreeEnergy.log_boltz hn]
  ring

/-- The van 't Hoff slope: differentiating the log population ratio in `beta` returns the
energy gap, and nothing else about the two conformations. -/
theorem vantHoff (hn : 0 < n) (U : Fin n → ℝ) (i j : Fin n) (beta : ℝ) :
    HasDerivAt (fun b => Real.log (FreeEnergy.boltz b U i / FreeEnergy.boltz b U j))
      (-(U i - U j)) beta := by
  have hfun : (fun b => Real.log (FreeEnergy.boltz b U i / FreeEnergy.boltz b U j))
      = fun b => -b * (U i - U j) := by
    funext b
    exact log_pop_ratio hn b U i j
  rw [hfun]
  simpa using ((hasDerivAt_id beta).neg.mul_const (U i - U j))

/-! ## Cold denaturation -/

/-- The Gibbs--Helmholtz stability curve with a constant heat-capacity change `dCp`. -/
noncomputable def gibbsHelmholtz (dH0 dS0 dCp T0 : ℝ) (T : ℝ) : ℝ :=
  dH0 - T * dS0 + dCp * ((T - T0) - T * Real.log (T / T0))

/-- The stability curve is an affine function of temperature plus `dCp` times the (strictly
concave) entropy function `-T log T`. -/
lemma gibbsHelmholtz_eq {dH0 dS0 dCp T0 T : ℝ} (hT0 : 0 < T0) (hT : 0 < T) :
    gibbsHelmholtz dH0 dS0 dCp T0 T
      = (dH0 - dCp * T0) + (dCp + dCp * Real.log T0 - dS0) * T
        + dCp * Real.negMulLog T := by
  unfold gibbsHelmholtz Real.negMulLog
  rw [Real.log_div hT.ne' hT0.ne']
  ring

/-- An affine function plus a positive multiple of `negMulLog` is strictly concave. -/
lemma strictConcaveOn_affine_add_negMulLog {A B c : ℝ} (hc : 0 < c) :
    StrictConcaveOn ℝ (Set.Ici (0 : ℝ)) (fun T => A + B * T + c * Real.negMulLog T) := by
  refine ⟨convex_Ici 0, ?_⟩
  intro x hx y hy hxy a b ha hb hab
  have h := Real.strictConcaveOn_negMulLog.2 hx hy hxy ha hb hab
  simp only [smul_eq_mul] at h ⊢
  have h2 : c * (a * Real.negMulLog x + b * Real.negMulLog y)
      < c * Real.negMulLog (a * x + b * y) := by
    exact mul_lt_mul_of_pos_left h hc
  calc a * (A + B * x + c * Real.negMulLog x) + b * (A + B * y + c * Real.negMulLog y)
      = (a + b) * A + B * (a * x + b * y)
        + c * (a * Real.negMulLog x + b * Real.negMulLog y) := by ring
    _ < (a + b) * A + B * (a * x + b * y) + c * Real.negMulLog (a * x + b * y) := by
        linarith
    _ = A + B * (a * x + b * y) + c * Real.negMulLog (a * x + b * y) := by
        rw [hab]; ring

/-- **The stability curve is strictly concave when `ΔCp > 0`.** -/
theorem gibbsHelmholtz_strictConcaveOn {dH0 dS0 dCp T0 : ℝ} (hT0 : 0 < T0) (hCp : 0 < dCp) :
    StrictConcaveOn ℝ (Set.Ioi (0 : ℝ)) (gibbsHelmholtz dH0 dS0 dCp T0) := by
  have hsub : StrictConcaveOn ℝ (Set.Ioi (0 : ℝ))
      (fun T => (dH0 - dCp * T0) + (dCp + dCp * Real.log T0 - dS0) * T
        + dCp * Real.negMulLog T) :=
    (strictConcaveOn_affine_add_negMulLog hCp).subset Set.Ioi_subset_Ici_self (convex_Ioi 0)
  refine ⟨convex_Ioi 0, fun x hx y hy hxy a b ha hb hab => ?_⟩
  have hx' : (0:ℝ) < x := hx
  have hy' : (0:ℝ) < y := hy
  have hmix : (0:ℝ) < a * x + b * y := by nlinarith
  have h := hsub.2 hx hy hxy ha hb hab
  simp only [smul_eq_mul] at h ⊢
  rw [gibbsHelmholtz_eq hT0 hx', gibbsHelmholtz_eq hT0 hy', gibbsHelmholtz_eq hT0 hmix]
  exact h

/-- A strictly concave function has at most two zeros. -/
theorem no_three_zeros {f : ℝ → ℝ} {s : Set ℝ} (hf : StrictConcaveOn ℝ s f)
    {x y z : ℝ} (hx : x ∈ s) (hz : z ∈ s) (hxy : x < y) (hyz : y < z)
    (h0x : f x = 0) (h0y : f y = 0) (h0z : f z = 0) : False := by
  have hxz : x < z := lt_trans hxy hyz
  set a := (z - y) / (z - x) with ha_def
  set b := (y - x) / (z - x) with hb_def
  have hzx : 0 < z - x := by linarith
  have ha : 0 < a := div_pos (by linarith) hzx
  have hb : 0 < b := div_pos (by linarith) hzx
  have hab : a + b = 1 := by
    rw [ha_def, hb_def, ← add_div, div_eq_one_iff_eq (ne_of_gt hzx)]
    ring
  have hcomb : a • x + b • z = y := by
    simp only [smul_eq_mul, ha_def, hb_def]
    field_simp
    ring
  have h := hf.2 hx hz (ne_of_lt hxz) ha hb hab
  rw [hcomb, h0x, h0z, h0y] at h
  simp at h

/-- **Never three transition temperatures.** -/
theorem no_three_transition_temperatures {dH0 dS0 dCp T0 : ℝ} (hT0 : 0 < T0) (hCp : 0 < dCp)
    {T1 T2 T3 : ℝ} (h1 : 0 < T1) (h12 : T1 < T2) (h23 : T2 < T3)
    (z1 : gibbsHelmholtz dH0 dS0 dCp T0 T1 = 0)
    (z2 : gibbsHelmholtz dH0 dS0 dCp T0 T2 = 0)
    (z3 : gibbsHelmholtz dH0 dS0 dCp T0 T3 = 0) : False :=
  no_three_zeros (gibbsHelmholtz_strictConcaveOn hT0 hCp) (Set.mem_Ioi.mpr h1)
    (Set.mem_Ioi.mpr (by linarith)) h12 h23 z1 z2 z3

lemma gibbsHelmholtz_continuousOn {dH0 dS0 dCp T0 : ℝ} (hT0 : 0 < T0) {a b : ℝ} (ha : 0 < a) :
    ContinuousOn (gibbsHelmholtz dH0 dS0 dCp T0) (Set.Icc a b) := by
  unfold gibbsHelmholtz
  refine ContinuousOn.add (ContinuousOn.sub continuousOn_const
    (continuousOn_id.mul continuousOn_const)) ?_
  refine continuousOn_const.mul (ContinuousOn.sub
    (continuousOn_id.sub continuousOn_const) (continuousOn_id.mul ?_))
  refine ContinuousOn.log (continuousOn_id.div continuousOn_const (fun x _ => hT0.ne')) ?_
  intro x hx
  have hx' : a ≤ x := hx.1
  have : 0 < x := lt_of_lt_of_le ha hx'
  positivity

/-- **Cold denaturation is forced.**  If the state is stable at `Tm` and unstable at a lower
temperature `T1` and at a higher temperature `T2`, then there is a transition temperature on
each side: a cold denaturation temperature in `(T1, Tm)` and a heat denaturation temperature
in `(Tm, T2)`.  A positive heat-capacity change -- the thermodynamic signature of burying
hydrophobic surface -- therefore *forces* the ordered state to melt at both ends of the
temperature axis. -/
theorem denaturation_two_temperatures {dH0 dS0 dCp T0 : ℝ} (hT0 : 0 < T0)
    {T1 Tm T2 : ℝ} (h1 : 0 < T1) (h1m : T1 < Tm) (hm2 : Tm < T2)
    (hlow : gibbsHelmholtz dH0 dS0 dCp T0 T1 < 0)
    (hmid : 0 < gibbsHelmholtz dH0 dS0 dCp T0 Tm)
    (hhigh : gibbsHelmholtz dH0 dS0 dCp T0 T2 < 0) :
    (∃ Tc ∈ Set.Ioo T1 Tm, gibbsHelmholtz dH0 dS0 dCp T0 Tc = 0) ∧
      (∃ Th ∈ Set.Ioo Tm T2, gibbsHelmholtz dH0 dS0 dCp T0 Th = 0) := by
  constructor
  · have hcont : ContinuousOn (gibbsHelmholtz dH0 dS0 dCp T0) (Set.Icc T1 Tm) :=
      gibbsHelmholtz_continuousOn hT0 h1
    have hsub := intermediate_value_Ioo (le_of_lt h1m) hcont
    have hmem : (0 : ℝ) ∈ Set.Ioo (gibbsHelmholtz dH0 dS0 dCp T0 T1)
        (gibbsHelmholtz dH0 dS0 dCp T0 Tm) := ⟨hlow, hmid⟩
    obtain ⟨Tc, hTc, hval⟩ := hsub hmem
    exact ⟨Tc, hTc, hval⟩
  · have hcont : ContinuousOn (gibbsHelmholtz dH0 dS0 dCp T0) (Set.Icc Tm T2) :=
      gibbsHelmholtz_continuousOn hT0 (lt_trans h1 h1m)
    have hsub := intermediate_value_Ioo' (le_of_lt hm2) hcont
    have hmem : (0 : ℝ) ∈ Set.Ioo (gibbsHelmholtz dH0 dS0 dCp T0 T2)
        (gibbsHelmholtz dH0 dS0 dCp T0 Tm) := ⟨hhigh, hmid⟩
    obtain ⟨Th, hTh, hval⟩ := hsub hmem
    exact ⟨Th, hTh, hval⟩

end Thermo

end IDR
