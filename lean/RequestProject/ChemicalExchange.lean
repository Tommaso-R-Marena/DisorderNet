/-
# Part XXXII.2  Relaxation dispersion: the invisible state and what it hides

The other half of the NMR dynamics toolkit is *chemical exchange*.  A disordered region that
transiently visits a minority conformation — a partly folded state, a bound-like state, a
`cis` proline — carries the exchange into the transverse relaxation rate, and a CPMG
relaxation-dispersion experiment measures the effective rate `R₂,eff` as a function of the
refocusing cycle time `t_cp`.  In the fast-exchange limit that measurement is the
Luz--Meiboom profile

  `R₂,eff = R₂⁰ + (Φ_ex / k_ex)·(1 − 2·tanh(k_ex t_cp/2)/(k_ex t_cp))`,
  `Φ_ex = p_A p_B Δω²`   (`R2eff`, `phiEx`, `disp`),

with `p_B` the population of the minority ("invisible") state, `Δω` its chemical-shift
difference from the major state, and `k_ex` the exchange rate.  *This forward model is a
hypothesis of the file, not a theorem*: it is the standard fast-exchange approximation to the
two-site Bloch--McConnell equations, and everything below is a statement about what the
experiment reports under it.

Shape of the profile:

* `disp_nonneg`, `disp_lt_one`, `R2eff_ge`, `R2eff_lt_plateau` -- the dispersion is a strictly
  positive excess relaxation bounded by the plateau `Φ_ex/k_ex`; that plateau, and nothing
  more, is the total amplitude available to the experiment.
* `R2eff_sub_le_short_cycle` -- at short cycle times the exchange is refocused, and the excess
  vanishes linearly in `t_cp`: it is at most `(Φ_ex/k_ex)·(k_ex t_cp/2)·e^{k_ex t_cp/2}`.
* `plateau_gap_le` -- at long cycle times the profile approaches the plateau, within
  `2Φ_ex/(k_ex² t_cp)`.

What the profile determines, and what it does not:

* `population_lower_bound` -- a positive result: with the chemical-shift difference bounded by
  the spectral range, a measured `Φ_ex` bounds the minority population from *below*,
  `p_B ≥ Φ_ex/Δω_max²`.  Exchange broadening is genuine evidence that a minority state exists.
* `R2eff_congr_of_phiEx` and `invisible_state_population_unidentifiable` -- but it says nothing
  about *how much* of it there is: the population and the shift enter only through the product
  `p_A p_B Δω²`, so for *every* population in `(0, 1/2]` there is a shift difference
  reproducing the entire measured profile at every cycle time and every field.  A quoted `p_B`
  is a consequence of the assumed `Δω`, not a measurement.
* `fast_exchange_erases_dispersion` and `arbitrarily_populated_invisible_state` -- and a state
  can be invisible outright: at fixed `Φ_ex` the whole profile is squeezed into a band of
  height `Φ_ex/k_ex`, so for any population, any shift difference and any detection threshold
  there is an exchange rate at which the state contributes less than the threshold at every
  cycle time.  Disordered regions, whose interconversions are fast, are exactly the regime in
  which this happens.

Design consequence: a model of a disordered region is not entitled to the populations quoted
from dispersion data, and it is not contradicted by flat dispersion data.  It must predict the
*profile* — `Φ_ex` and `k_ex` together — and be scored on it.
-/
import Mathlib

set_option autoImplicit false

namespace Exchange

open Real

/-- `sinh y ≤ y cosh y` for `y ≥ 0`, the inequality behind `tanh_le_self`. -/
private lemma sinh_le_self_mul_cosh {y : ℝ} (hy : 0 ≤ y) : Real.sinh y ≤ y * Real.cosh y := by
  set f : ℝ → ℝ := fun t => t * Real.cosh t - Real.sinh t with hf
  have hderiv : ∀ t : ℝ, HasDerivAt f (t * Real.sinh t) t := by
    intro t
    have h1 : HasDerivAt (fun t : ℝ => t * Real.cosh t) (1 * Real.cosh t + t * Real.sinh t) t :=
      (hasDerivAt_id t).mul (Real.hasDerivAt_cosh t)
    simpa using h1.sub (Real.hasDerivAt_sinh t)
  have hmono : MonotoneOn f (Set.Ici 0) := by
    apply monotoneOn_of_deriv_nonneg (convex_Ici 0)
    · exact Continuous.continuousOn (by fun_prop)
    · intro t _
      exact (hderiv t).differentiableAt.differentiableWithinAt
    · intro t ht
      simp only [interior_Ici, Set.mem_Ioi] at ht
      rw [(hderiv t).deriv]
      exact mul_nonneg ht.le (Real.sinh_nonneg_iff.mpr ht.le)
  have h0 : f 0 = 0 := by simp [hf]
  have hle := hmono Set.self_mem_Ici (Set.mem_Ici.mpr hy) hy
  rw [h0] at hle
  simp only [hf] at hle
  linarith

/-- `tanh y ≤ y` for `y ≥ 0`. -/
private lemma tanh_le_self {y : ℝ} (hy : 0 ≤ y) : Real.tanh y ≤ y := by
  rw [Real.tanh_eq_sinh_div_cosh, div_le_iff₀ (Real.cosh_pos y)]
  exact sinh_le_self_mul_cosh hy

private lemma tanh_pos {y : ℝ} (hy : 0 < y) : 0 < Real.tanh y := by
  rw [Real.tanh_eq_sinh_div_cosh]
  exact div_pos (Real.sinh_pos_iff.mpr hy) (Real.cosh_pos y)

/-- `y / cosh y ≤ tanh y`, from `y ≤ sinh y`. -/
private lemma self_div_cosh_le_tanh {y : ℝ} (hy : 0 ≤ y) : y / Real.cosh y ≤ Real.tanh y := by
  rw [Real.tanh_eq_sinh_div_cosh]
  have hs : y ≤ Real.sinh y := by
    rcases eq_or_lt_of_le hy with h | h
    · simp [← h]
    · exact (Real.self_lt_sinh_iff.mpr h).le
  gcongr

private lemma cosh_le_exp {y : ℝ} (hy : 0 ≤ y) : Real.cosh y ≤ Real.exp y := by
  rw [Real.cosh_eq]
  have : Real.exp (-y) ≤ Real.exp y := Real.exp_le_exp.mpr (by linarith)
  linarith

/-! ### The Luz--Meiboom dispersion profile -/

/-- The CPMG dispersion function of the dimensionless cycle time `x = k_ex t_cp`. -/
noncomputable def disp (x : ℝ) : ℝ := 1 - 2 * Real.tanh (x / 2) / x

/-- The exchange amplitude `Φ_ex = p_A p_B Δω²`. -/
noncomputable def phiEx (pB dw : ℝ) : ℝ := (1 - pB) * pB * dw ^ 2

/-- The measured effective transverse relaxation rate, fast-exchange (Luz--Meiboom) form. -/
noncomputable def R2eff (R20 pB dw kex tcp : ℝ) : ℝ :=
  R20 + phiEx pB dw / kex * disp (kex * tcp)

lemma phiEx_nonneg {pB dw : ℝ} (h0 : 0 ≤ pB) (h1 : pB ≤ 1) : 0 ≤ phiEx pB dw := by
  have : 0 ≤ 1 - pB := by linarith
  unfold phiEx
  positivity

lemma phiEx_pos {pB dw : ℝ} (h0 : 0 < pB) (h1 : pB < 1) (hdw : dw ≠ 0) : 0 < phiEx pB dw := by
  have h2 : 0 < 1 - pB := by linarith
  have : 0 < dw ^ 2 := by positivity
  unfold phiEx
  positivity

/-- The dispersion function is nonnegative: exchange never refocuses below the intrinsic rate. -/
theorem disp_nonneg {x : ℝ} (hx : 0 < x) : 0 ≤ disp x := by
  have h := tanh_le_self (y := x / 2) (by linarith)
  rw [disp, sub_nonneg, div_le_one hx]
  linarith

/-- And is strictly below one: the plateau is approached, never attained. -/
theorem disp_lt_one {x : ℝ} (hx : 0 < x) : disp x < 1 := by
  have h := tanh_pos (y := x / 2) (by linarith)
  have : 0 < 2 * Real.tanh (x / 2) / x := by positivity
  rw [disp]
  linarith

/-- The measured rate is never below the intrinsic rate. -/
theorem R2eff_ge {R20 pB dw kex tcp : ℝ} (h0 : 0 ≤ pB) (h1 : pB ≤ 1) (hk : 0 < kex)
    (ht : 0 < tcp) : R20 ≤ R2eff R20 pB dw kex tcp := by
  have hx : 0 < kex * tcp := by positivity
  have := mul_nonneg (div_nonneg (phiEx_nonneg (dw := dw) h0 h1) hk.le) (disp_nonneg hx)
  simp only [R2eff]
  linarith

/-- **The whole amplitude available to the experiment is `Φ_ex/k_ex`.** -/
theorem R2eff_lt_plateau {R20 pB dw kex tcp : ℝ} (h0 : 0 < pB) (h1 : pB < 1) (hdw : dw ≠ 0)
    (hk : 0 < kex) (ht : 0 < tcp) :
    R2eff R20 pB dw kex tcp < R20 + phiEx pB dw / kex := by
  have hx : 0 < kex * tcp := by positivity
  have hphi : 0 < phiEx pB dw / kex := div_pos (phiEx_pos h0 h1 hdw) hk
  have := (mul_lt_mul_of_pos_left (disp_lt_one hx) hphi)
  simp only [R2eff]
  linarith [this]

/-- **At long cycle times the profile reaches the plateau**, within `2Φ_ex/(k_ex² t_cp)`. -/
theorem plateau_gap_le {R20 pB dw kex tcp : ℝ} (h0 : 0 ≤ pB) (h1 : pB ≤ 1) (hk : 0 < kex)
    (ht : 0 < tcp) :
    (R20 + phiEx pB dw / kex) - R2eff R20 pB dw kex tcp
      ≤ 2 * phiEx pB dw / (kex ^ 2 * tcp) := by
  have hx : 0 < kex * tcp := by positivity
  have hphi : 0 ≤ phiEx pB dw := phiEx_nonneg h0 h1
  have htanh : Real.tanh (kex * tcp / 2) < 1 := Real.tanh_lt_one _
  have hgap : 1 - disp (kex * tcp) ≤ 2 / (kex * tcp) := by
    rw [disp]
    have h2 : 2 * Real.tanh (kex * tcp / 2) ≤ 2 := by linarith
    have : 2 * Real.tanh (kex * tcp / 2) / (kex * tcp) ≤ 2 / (kex * tcp) := by
      gcongr
    linarith
  have hmul : phiEx pB dw / kex * (1 - disp (kex * tcp))
      ≤ phiEx pB dw / kex * (2 / (kex * tcp)) :=
    mul_le_mul_of_nonneg_left hgap (div_nonneg hphi hk.le)
  have hrw : phiEx pB dw / kex * (2 / (kex * tcp)) = 2 * phiEx pB dw / (kex ^ 2 * tcp) := by
    field_simp
  simp only [R2eff]
  rw [hrw] at hmul
  nlinarith [hmul]

/-- **At short cycle times the exchange is refocused.**  The excess relaxation vanishes
linearly in the cycle time. -/
theorem R2eff_sub_le_short_cycle {R20 pB dw kex tcp : ℝ} (h0 : 0 ≤ pB) (h1 : pB ≤ 1)
    (hk : 0 < kex) (ht : 0 < tcp) :
    R2eff R20 pB dw kex tcp - R20
      ≤ phiEx pB dw / kex * ((kex * tcp / 2) * Real.exp (kex * tcp / 2)) := by
  have hx : 0 < kex * tcp := by positivity
  have hy : 0 < kex * tcp / 2 := by linarith
  have hphi : 0 ≤ phiEx pB dw := phiEx_nonneg h0 h1
  set y : ℝ := kex * tcp / 2 with hydef
  have hcosh : 0 < Real.cosh y := Real.cosh_pos y
  -- `disp x ≤ 1 - 1/cosh(x/2)`
  have harg : kex * tcp / 2 = y := by rw [hydef]
  have hxy : kex * tcp = 2 * y := by rw [hydef]; ring
  have hkey : 1 / Real.cosh y ≤ 2 * Real.tanh y / (2 * y) := by
    rw [div_le_div_iff₀ hcosh (by positivity)]
    have h3 := mul_le_mul_of_nonneg_right (self_div_cosh_le_tanh hy.le) hcosh.le
    rw [div_mul_cancel₀ _ (ne_of_gt hcosh)] at h3
    linarith
  have hstep : disp (kex * tcp) ≤ 1 - 1 / Real.cosh y := by
    unfold disp
    rw [harg, hxy]
    linarith
  -- and `1 - 1/cosh y ≤ y·e^y`
  have hcexp : Real.cosh y ≤ Real.exp y := cosh_le_exp hy.le
  have hone : 1 ≤ Real.cosh y := Real.one_le_cosh y
  have hfinal : 1 - 1 / Real.cosh y ≤ y * Real.exp y := by
    have h1' : 1 - 1 / Real.cosh y ≤ Real.cosh y - 1 := by
      have hinv : (2 : ℝ) - Real.cosh y ≤ 1 / Real.cosh y := by
        rw [le_div_iff₀ hcosh]
        nlinarith [sq_nonneg (Real.cosh y - 1)]
      linarith
    have h2' : Real.cosh y - 1 ≤ Real.exp y - 1 := by linarith
    have h3' : Real.exp y - 1 ≤ y * Real.exp y := by
      have := Real.add_one_le_exp (-y)
      have hpos : 0 < Real.exp y := Real.exp_pos y
      have hinv : Real.exp (-y) = 1 / Real.exp y := by
        rw [Real.exp_neg]; ring
      rw [hinv] at this
      have := mul_le_mul_of_nonneg_right this hpos.le
      rw [div_mul_cancel₀] at this
      · linarith
      · exact ne_of_gt hpos
    linarith
  have hmul : phiEx pB dw / kex * disp (kex * tcp)
      ≤ phiEx pB dw / kex * (y * Real.exp y) :=
    mul_le_mul_of_nonneg_left (le_trans hstep hfinal) (div_nonneg hphi hk.le)
  simp only [R2eff]
  linarith

/-! ### What the profile determines -/

/-- **A positive result: exchange broadening bounds the minority population from below.**  If
the chemical-shift difference cannot exceed the spectral range `dwmax`, a measured amplitude
`Φ_ex` forces `p_B ≥ Φ_ex/dwmax²`. -/
theorem population_lower_bound {pB dw dwmax : ℝ} (h0 : 0 ≤ pB)
    (hdw : |dw| ≤ dwmax) (hmax : 0 < dwmax) :
    phiEx pB dw / dwmax ^ 2 ≤ pB := by
  have hsq : dw ^ 2 ≤ dwmax ^ 2 := by
    have := abs_nonneg dw
    nlinarith [sq_abs dw]
  have hstep : phiEx pB dw ≤ pB * dwmax ^ 2 := by
    unfold phiEx
    nlinarith [sq_nonneg dw, mul_nonneg h0 (sq_nonneg dw),
      mul_le_mul_of_nonneg_left hsq h0]
  rw [div_le_iff₀ (by positivity)]
  linarith

/-- The profile sees the minority state only through `Φ_ex = p_A p_B Δω²`. -/
theorem R2eff_congr_of_phiEx {R20 pB dw pB' dw' kex : ℝ} (h : phiEx pB dw = phiEx pB' dw') :
    ∀ tcp, R2eff R20 pB dw kex tcp = R2eff R20 pB' dw' kex tcp := by
  intro tcp
  simp [R2eff, h]

/-- **The population of the invisible state is not identifiable.**  For *every* population in
`(0, 1/2]` there is a chemical-shift difference reproducing a given dispersion profile exactly,
at every cycle time. -/
theorem invisible_state_population_unidentifiable {R20 pB0 dw0 kex : ℝ} (h00 : 0 < pB0)
    (h01 : pB0 < 1) (hdw0 : 0 < dw0) {pB : ℝ} (hp0 : 0 < pB) (hp1 : pB ≤ 1 / 2) :
    ∃ dw : ℝ, 0 < dw ∧ ∀ tcp, R2eff R20 pB dw kex tcp = R2eff R20 pB0 dw0 kex tcp := by
  have hden : 0 < (1 - pB) * pB := by
    have : 0 < 1 - pB := by linarith
    positivity
  have hnum : 0 < phiEx pB0 dw0 := phiEx_pos h00 h01 (ne_of_gt hdw0)
  refine ⟨Real.sqrt (phiEx pB0 dw0 / ((1 - pB) * pB)), Real.sqrt_pos.mpr (by positivity), ?_⟩
  apply R2eff_congr_of_phiEx
  unfold phiEx
  rw [Real.sq_sqrt (by positivity)]
  have hne : (1 : ℝ) - pB ≠ 0 := by linarith
  field_simp [hne]

/-- **Fast exchange erases the dispersion.**  At fixed amplitude the entire profile lies within
`Φ_ex/k_ex` of the intrinsic rate. -/
theorem fast_exchange_erases_dispersion {R20 pB dw kex tcp : ℝ} (h0 : 0 ≤ pB) (h1 : pB ≤ 1)
    (hk : 0 < kex) (ht : 0 < tcp) :
    |R2eff R20 pB dw kex tcp - R20| ≤ phiEx pB dw / kex := by
  have hx : 0 < kex * tcp := by positivity
  have hphi : 0 ≤ phiEx pB dw := phiEx_nonneg h0 h1
  have hd0 : 0 ≤ disp (kex * tcp) := disp_nonneg hx
  have hd1 : disp (kex * tcp) < 1 := disp_lt_one hx
  have hlow : 0 ≤ phiEx pB dw / kex * disp (kex * tcp) :=
    mul_nonneg (div_nonneg hphi hk.le) hd0
  have hhigh : phiEx pB dw / kex * disp (kex * tcp) ≤ phiEx pB dw / kex := by
    nlinarith [div_nonneg hphi hk.le]
  simp only [R2eff]
  rw [abs_le]
  constructor <;> linarith

/-- **A state of any population can be invisible.**  Given any population, any chemical-shift
difference and any detection threshold, there is an exchange rate at which the state changes the
measured rate by less than the threshold at every cycle time. -/
theorem arbitrarily_populated_invisible_state {pB dw eps : ℝ} (h0 : 0 ≤ pB) (h1 : pB ≤ 1)
    (heps : 0 < eps) :
    ∃ kex : ℝ, 0 < kex ∧ ∀ R20 tcp : ℝ, 0 < tcp → |R2eff R20 pB dw kex tcp - R20| < eps := by
  have hphi0 : 0 ≤ phiEx pB dw := phiEx_nonneg (dw := dw) h0 h1
  have hk : 0 < (phiEx pB dw + 1) / eps := by positivity
  refine ⟨(phiEx pB dw + 1) / eps, hk, ?_⟩
  intro R20 tcp ht
  have hbound := fast_exchange_erases_dispersion (R20 := R20) (pB := pB) (dw := dw)
    (kex := (phiEx pB dw + 1) / eps) (tcp := tcp) h0 h1 hk ht
  have hphi : 0 ≤ phiEx pB dw := phiEx_nonneg h0 h1
  have : phiEx pB dw / ((phiEx pB dw + 1) / eps) < eps := by
    rw [div_div_eq_mul_div, div_lt_iff₀ (by positivity)]
    nlinarith
  linarith

end Exchange
