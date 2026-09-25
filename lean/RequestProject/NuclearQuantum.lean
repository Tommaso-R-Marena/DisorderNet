/-
# Part LII.1  Nuclear quantum effects: what a classical force field cannot report

Part XXIV's model, like every molecular-mechanics model of a disordered region, is *classical*:
its ensemble is the Boltzmann measure of a potential energy over configuration space.  Part XXV
named "quantum-mechanical bond making and breaking" as outside the development.  The commoner
and more embarrassing gap is smaller and is entirely computable: even with no chemistry
happening, the nuclei of a chain are quantum oscillators, and a classical model gets their
thermodynamics wrong by a definite amount.

This file treats one harmonic mode of energy quantum `w = ħω` at inverse temperature `b = 1/kT`.

* `qFree`, `clFree`, `qEnergy` -- the exact quantum free energy `(1/b) log(2 sinh(bw/2))`, the
  classical one `(1/b) log(bw)`, and the quantum mean energy `(w/2) coth(bw/2)`.
* `qFree_eq_zpe_add` -- the exact decomposition `F_q = w/2 + (1/b) log(1 - e^{-bw})`: a
  zero-point term plus a thermal one.
* `clFree_lt_qFree`, `qFree_lt_zpe` -- the classical free energy is always *below* the quantum
  one, and the quantum one is always below the zero-point energy: `F_cl < F_q < w/2`.  The
  first inequality is `sinh x > x`; there is no regime, and no parameterisation, in which a
  classical harmonic mode has the right free energy.
* `qEnergy_gt_zpe`, `qEnergy_gt_kT` -- equipartition fails in the same direction: the mode
  holds more than `kT`, and more than its zero-point energy, at every temperature.  (`kT` is
  the classical answer for a harmonic mode.)
* `tendsto_partition_ratio_one` -- and the model is not wrong everywhere: as `w → 0` at fixed
  temperature the quantum and classical partition functions agree in the limit.  Soft
  collective modes of a disordered chain are classical; the C-H, N-H and O-H stretches that
  carry a large part of the vibrational free energy are not.
* `classical_isotope_independent` -- **the sharp statement.**  In a classical model the free
  energy difference between two states of a chain is *independent of the nuclear masses*: the
  mass enters every frequency by the same factor and cancels.  A classical force field
  therefore predicts exactly zero equilibrium isotope effect, for every observable, at every
  temperature.
* `quantum_isotope_effect` -- whereas the quantum model does not: for every mass scaling
  `0 < s < 1` and every pair of distinct mode frequencies there is a temperature at which the
  two free energy differences differ, because at low temperature each tends to its zero-point
  value and the zero-point values scale with `s`.

So a measured H/D equilibrium effect is not a parameterisation error of a classical model; it
is outside its range.  The repair -- a path-integral (ring polymer) treatment of the nuclei --
is standard, and what the theorems above price is the decision not to make it.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace NuclearQuantum

open Real Filter Topology

/-- The exact free energy of one harmonic mode of energy quantum `w = ħω` at inverse
temperature `b`: `(1/b) log(2 sinh(bw/2))`. -/
noncomputable def qFree (b w : ℝ) : ℝ := (1 / b) * Real.log (2 * Real.sinh (b * w / 2))

/-- The classical free energy of the same mode, `(1/b) log(bw)`. -/
noncomputable def clFree (b w : ℝ) : ℝ := (1 / b) * Real.log (b * w)

/-- The exact mean energy of the mode, `(w/2) coth(bw/2)`. -/
noncomputable def qEnergy (b w : ℝ) : ℝ :=
  (w / 2) * (Real.cosh (b * w / 2) / Real.sinh (b * w / 2))

/-- The quantum partition function of the mode. -/
noncomputable def qPartition (b w : ℝ) : ℝ := 1 / (2 * Real.sinh (b * w / 2))

/-- The classical partition function of the mode. -/
noncomputable def clPartition (b w : ℝ) : ℝ := 1 / (b * w)

lemma two_sinh_eq (x : ℝ) : 2 * Real.sinh x = Real.exp x * (1 - Real.exp (-(2 * x))) := by
  rw [Real.sinh_eq]
  have h : Real.exp x * Real.exp (-(2 * x)) = Real.exp (-x) := by
    rw [← Real.exp_add]; ring_nf
  rw [mul_sub, mul_one, h]; ring

/-- **Zero-point energy plus a thermal term.**  The exact quantum free energy of a harmonic
mode splits as `w/2 + (1/b) log(1 - e^{-bw})`. -/
theorem qFree_eq_zpe_add {b w : ℝ} (hb : 0 < b) (hw : 0 < w) :
    qFree b w = w / 2 + (1 / b) * Real.log (1 - Real.exp (-(b * w))) := by
  have hbw : 0 < b * w := by positivity
  have hu : Real.exp (-(b * w)) < 1 := by rw [Real.exp_lt_one_iff]; linarith
  have h1 : (0 : ℝ) < 1 - Real.exp (-(b * w)) := by linarith
  rw [qFree, two_sinh_eq, show -(2 * (b * w / 2)) = -(b * w) by ring,
    Real.log_mul (Real.exp_ne_zero _) (ne_of_gt h1), Real.log_exp]
  field_simp

/-- **The quantum free energy is below the zero-point energy.** -/
theorem qFree_lt_zpe {b w : ℝ} (hb : 0 < b) (hw : 0 < w) : qFree b w < w / 2 := by
  have hbw : 0 < b * w := by positivity
  have hu : Real.exp (-(b * w)) < 1 := by rw [Real.exp_lt_one_iff]; linarith
  have hupos : 0 < Real.exp (-(b * w)) := Real.exp_pos _
  have hlog : Real.log (1 - Real.exp (-(b * w))) < 0 :=
    Real.log_neg (by linarith) (by linarith)
  rw [qFree_eq_zpe_add hb hw]
  have : (1 / b) * Real.log (1 - Real.exp (-(b * w))) < 0 :=
    mul_neg_of_pos_of_neg (by positivity) hlog
  linarith

/-- **The classical free energy is below the quantum one, always.**  The gap is
`(1/b) log(sinh(bw/2)/(bw/2))`, and it is strictly positive at every temperature and every
frequency. -/
theorem clFree_lt_qFree {b w : ℝ} (hb : 0 < b) (hw : 0 < w) : clFree b w < qFree b w := by
  have hx : 0 < b * w / 2 := by positivity
  have hs : b * w / 2 < Real.sinh (b * w / 2) := Real.self_lt_sinh_iff.mpr hx
  have hlog : Real.log (b * w) < Real.log (2 * Real.sinh (b * w / 2)) := by
    apply Real.log_lt_log (by positivity)
    linarith
  exact mul_lt_mul_of_pos_left hlog (by positivity)

/-- **Zero-point energy is a strict lower bound on the mean energy.** -/
theorem qEnergy_gt_zpe {b w : ℝ} (hb : 0 < b) (hw : 0 < w) : w / 2 < qEnergy b w := by
  have hx : 0 < b * w / 2 := by positivity
  have hsp : 0 < Real.sinh (b * w / 2) := Real.sinh_pos_iff.mpr hx
  have hlt : Real.sinh (b * w / 2) < Real.cosh (b * w / 2) := Real.sinh_lt_cosh _
  have hratio : 1 < Real.cosh (b * w / 2) / Real.sinh (b * w / 2) :=
    (one_lt_div hsp).mpr hlt
  have hw2 : 0 < w / 2 := by positivity
  calc w / 2 = (w / 2) * 1 := by ring
    _ < (w / 2) * (Real.cosh (b * w / 2) / Real.sinh (b * w / 2)) :=
        mul_lt_mul_of_pos_left hratio hw2
    _ = qEnergy b w := rfl

/-- `sinh x < x cosh x` for `x > 0`: the elementary inequality behind the failure of
equipartition. -/
lemma sinh_lt_mul_cosh {x : ℝ} (hx : 0 < x) : Real.sinh x < x * Real.cosh x := by
  have hmono : StrictMonoOn (fun y : ℝ => y * Real.cosh y - Real.sinh y) (Set.Ici 0) := by
    apply strictMonoOn_of_deriv_pos (convex_Ici 0)
    · fun_prop
    · intro y hy
      rw [interior_Ici] at hy
      have hy' : (0 : ℝ) < y := hy
      have hd : HasDerivAt (fun z : ℝ => z * Real.cosh z - Real.sinh z)
          (1 * Real.cosh y + y * Real.sinh y - Real.cosh y) y :=
        ((hasDerivAt_id y).mul (Real.hasDerivAt_cosh y)).sub (Real.hasDerivAt_sinh y)
      rw [hd.deriv]
      have hs : 0 < Real.sinh y := Real.sinh_pos_iff.mpr hy'
      nlinarith [mul_pos hy' hs]
  have h := hmono (by simp : (0 : ℝ) ∈ Set.Ici (0 : ℝ)) (le_of_lt hx : x ∈ Set.Ici (0 : ℝ)) hx
  simp at h
  linarith

/-- **Equipartition fails.**  A harmonic mode holds strictly more than the classical `kT` at
every temperature: the classical heat capacity is an overestimate exactly where the quantum
energy is. -/
theorem qEnergy_gt_kT {b w : ℝ} (hb : 0 < b) (hw : 0 < w) : 1 / b < qEnergy b w := by
  have hx : 0 < b * w / 2 := by positivity
  have hsp : 0 < Real.sinh (b * w / 2) := Real.sinh_pos_iff.mpr hx
  have hkey : Real.sinh (b * w / 2) < (b * w / 2) * Real.cosh (b * w / 2) :=
    sinh_lt_mul_cosh hx
  have hR : (w / 2) * (Real.cosh (b * w / 2) / Real.sinh (b * w / 2))
      = ((w / 2) * Real.cosh (b * w / 2)) / Real.sinh (b * w / 2) := by ring
  rw [qEnergy, hR, lt_div_iff₀ hsp, div_mul_eq_mul_div, div_lt_iff₀ hb]
  nlinarith [Real.cosh_pos (b * w / 2)]

/-- `x / sinh x → 1` as `x → 0`: the classical limit, in its bare form. -/
lemma tendsto_x_div_sinh : Tendsto (fun x : ℝ => x / Real.sinh x) (𝓝[≠] 0) (𝓝 1) := by
  have h := Real.hasDerivAt_sinh 0
  rw [hasDerivAt_iff_tendsto_slope] at h
  simp only [Real.cosh_zero] at h
  have h2 : Tendsto (fun x : ℝ => Real.sinh x / x) (𝓝[≠] 0) (𝓝 1) := by
    refine h.congr (fun x => ?_)
    simp [slope_def_field, div_eq_inv_mul]
  have h3 := h2.inv₀ (by norm_num)
  rw [inv_one] at h3
  refine h3.congr (fun x => ?_)
  rw [inv_div]

/-- **The classical limit is exact.**  At fixed temperature the ratio of the quantum to the
classical partition function of a mode tends to `1` as the mode softens. -/
theorem tendsto_partition_ratio_one {b : ℝ} (hb : 0 < b) :
    Tendsto (fun w : ℝ => qPartition b w / clPartition b w) (𝓝[>] (0 : ℝ)) (𝓝 1) := by
  have hmap : Tendsto (fun w : ℝ => b * w / 2) (𝓝[>] (0 : ℝ)) (𝓝[≠] 0) := by
    apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
    · have hc : Continuous (fun w : ℝ => b * w / 2) := by continuity
      simpa using (hc.tendsto 0).mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with w hw
      have hw' : 0 < w := hw
      simp only [Set.mem_compl_iff, Set.mem_singleton_iff]
      positivity
  have hcomp := tendsto_x_div_sinh.comp hmap
  refine hcomp.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with w hw
  have hw' : 0 < w := hw
  have hs : 0 < Real.sinh (b * w / 2) := Real.sinh_pos_iff.mpr (by positivity)
  simp only [Function.comp, qPartition, clPartition]
  rw [div_div]
  field_simp

/-! ### The isotope effect -/

/-- **A classical model has no equilibrium isotope effect.**  Changing every nuclear mass
scales every frequency by the same factor `s`, and the factor cancels from the classical free
energy difference of two states: the classical prediction is exactly zero effect. -/
theorem classical_isotope_independent {b s wA wB : ℝ} (hb : 0 < b) (hs : 0 < s)
    (hA : 0 < wA) (hB : 0 < wB) :
    clFree b (s * wA) - clFree b (s * wB) = clFree b wA - clFree b wB := by
  have hbs : (0:ℝ) < b * s := by positivity
  simp only [clFree]
  rw [show b * (s * wA) = (b * s) * wA by ring, show b * (s * wB) = (b * s) * wB by ring,
    Real.log_mul (ne_of_gt hbs) (ne_of_gt hA), Real.log_mul (ne_of_gt hbs) (ne_of_gt hB),
    Real.log_mul (ne_of_gt hb) (ne_of_gt hA), Real.log_mul (ne_of_gt hb) (ne_of_gt hB)]
  ring

/-- `-log(1-u) ≤ 2u` for `0 < u ≤ 1/2`. -/
lemma log_one_sub_ge {u : ℝ} (hu : 0 < u) (hu2 : u ≤ 1 / 2) : -(2 * u) ≤ Real.log (1 - u) := by
  have hexp : Real.exp (-(2 * u)) ≤ 1 - u := by
    have h2 : 1 + 2 * u ≤ Real.exp (2 * u) := by linarith [Real.add_one_le_exp (2 * u)]
    rw [Real.exp_neg, inv_le_iff_one_le_mul₀ (by positivity)]
    nlinarith [Real.exp_pos (2 * u)]
  calc -(2 * u) = Real.log (Real.exp (-(2 * u))) := by rw [Real.log_exp]
    _ ≤ Real.log (1 - u) := Real.log_le_log (Real.exp_pos _) hexp

/-- **The quantum free energy tends to the zero-point energy** as the temperature falls, with
the explicit rate `2/b`. -/
lemma abs_qFree_sub_zpe_le {b w : ℝ} (hb : 0 < b) (hw : 0 < w) (h1 : 1 ≤ b * w) :
    |qFree b w - w / 2| ≤ 2 / b := by
  rw [qFree_eq_zpe_add hb hw]
  have hsimp : w / 2 + (1 / b) * Real.log (1 - Real.exp (-(b * w))) - w / 2
      = (1 / b) * Real.log (1 - Real.exp (-(b * w))) := by ring
  rw [hsimp]
  set u := Real.exp (-(b * w)) with hudef
  have hupos : 0 < u := Real.exp_pos _
  have hle : u ≤ 1 / 2 := by
    have hmono : u ≤ Real.exp (-1) := by
      rw [hudef]; exact Real.exp_le_exp.mpr (by linarith)
    have he : Real.exp (-1 : ℝ) ≤ 1 / 2 := by
      rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos 1) (by norm_num)]
      linarith [Real.exp_one_gt_d9]
    linarith
  have hlog_le : Real.log (1 - u) ≤ 0 := Real.log_nonpos (by linarith) (by linarith)
  have hlog_ge : -(2 * u) ≤ Real.log (1 - u) := log_one_sub_ge hupos hle
  rw [abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / b)]
  have habs : |Real.log (1 - u)| ≤ 2 * u := by rw [abs_of_nonpos hlog_le]; linarith
  calc (1 / b) * |Real.log (1 - u)| ≤ (1 / b) * (2 * u) :=
        mul_le_mul_of_nonneg_left habs (by positivity)
    _ ≤ (1 / b) * 2 := mul_le_mul_of_nonneg_left (by linarith) (by positivity)
    _ = 2 / b := by ring

/-- The quantum free energy of a mode tends to its zero-point energy as the temperature falls. -/
theorem tendsto_qFree_zpe {w : ℝ} (hw : 0 < w) :
    Tendsto (fun b : ℝ => qFree b w) atTop (𝓝 (w / 2)) := by
  have hzero : Tendsto (fun b : ℝ => qFree b w - w / 2) atTop (𝓝 0) := by
    apply squeeze_zero_norm' (a := fun b : ℝ => 2 / b)
    · filter_upwards [eventually_ge_atTop (max 1 (1 / w))] with b hb
      have hb1 : (1:ℝ) ≤ b := le_trans (le_max_left _ _) hb
      have hbw : 1 ≤ b * w := by
        have : 1 / w ≤ b := le_trans (le_max_right _ _) hb
        rw [div_le_iff₀ hw] at this
        linarith
      simpa [Real.norm_eq_abs] using abs_qFree_sub_zpe_le (lt_of_lt_of_le one_pos hb1) hw hbw
    · exact Filter.Tendsto.div_atTop tendsto_const_nhds tendsto_id
  have := hzero.add_const (w / 2)
  simpa using this

/-- **The quantum model does have an equilibrium isotope effect.**  For every mass scaling
`0 < s < 1` and every pair of distinct frequencies there is a temperature at which the free
energy difference of the two states is changed by the substitution: the low-temperature limits
are the zero-point differences `(wA - wB)/2` and `s(wA - wB)/2`. -/
theorem quantum_isotope_effect {s wA wB : ℝ} (hs0 : 0 < s) (hs1 : s < 1) (hB : 0 < wB)
    (hAB : wB < wA) :
    ∃ b : ℝ, 0 < b ∧
      qFree b wA - qFree b wB ≠ qFree b (s * wA) - qFree b (s * wB) := by
  by_contra hcon
  push_neg at hcon
  have hA : 0 < wA := lt_trans hB hAB
  have hlight : Tendsto (fun b : ℝ => qFree b wA - qFree b wB) atTop (𝓝 ((wA - wB) / 2)) := by
    have := (tendsto_qFree_zpe hA).sub (tendsto_qFree_zpe hB)
    simpa [sub_div] using this
  have hheavy : Tendsto (fun b : ℝ => qFree b (s * wA) - qFree b (s * wB)) atTop
      (𝓝 ((s * wA - s * wB) / 2)) := by
    have := (tendsto_qFree_zpe (by positivity : (0:ℝ) < s * wA)).sub
      (tendsto_qFree_zpe (by positivity : (0:ℝ) < s * wB))
    simpa [sub_div] using this
  have heq : Tendsto (fun b : ℝ => qFree b wA - qFree b wB) atTop
      (𝓝 ((s * wA - s * wB) / 2)) := by
    refine hheavy.congr' ?_
    filter_upwards [eventually_gt_atTop (0:ℝ)] with b hb
    exact (hcon b hb).symm
  have := tendsto_nhds_unique hlight heq
  have hne : (wA - wB) / 2 ≠ (s * wA - s * wB) / 2 := by
    intro h
    have : (1 - s) * (wA - wB) = 0 := by nlinarith
    rcases mul_eq_zero.mp this with h1 | h2
    · linarith
    · linarith
  exact hne this

end NuclearQuantum

end IDR
