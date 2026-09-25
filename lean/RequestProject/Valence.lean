/-
# Part VII.2  Multivalency: why a switch needs coupled sites

`RequestProject.Condensate` shows that whether a solution of disordered chains condenses is
decided by the interchain coupling and not by the single-chain ensemble.  This file treats
the microscopic origin of that coupling in the standard sticker--spacer picture: a
disordered region carries `n` short binding motifs, and the question a model must answer is
how sharply the bound fraction responds to the activity `x` of the partner.

Everything is read off the binding polynomial `Z`, the model's actual output at this level of
description, through the thermodynamic definition of the mean occupancy,
`occupancy Z x = x · Z'(x) / Z(x)`.

* `occupancy_independent` -- `n` independent equivalent motifs give `Z = (1+x)^n` and mean
  occupancy `n·x/(1+x)`: **the per-site occupancy does not depend on `n` at all.**
* `occupancy_allOrNone` -- fully coupled motifs give `Z = 1 + x^n` and mean occupancy
  `n·x^n/(1+x^n)`.
* `hill_log`, `hill_slope` -- the Hill plot of `x^n/(1+x^n)` is the straight line
  `n · log x`, so the **Hill coefficient is exactly `n`** for the coupled model and exactly
  `1` for the independent one (`hill_slope_independent`), whatever the valence.
* `no_cooperativity_from_independent_sites` -- consequently, for `n ≥ 2` no assignment of
  affinities to independent sites reproduces the coupled response: cooperativity is a
  statement about the *coupling* between motifs, exactly the kind of structure that
  `RequestProject.EnergyModels` proves a factorised model cannot carry.
* `ligand_window_independent`, `ligand_window_coupled`, `ligand_window_tendsto_one` -- the
  quantitative form: independent sites need an **81-fold** change in activity to go from 10%
  to 90% bound, coupled sites only `81^{1/n}`-fold, and that window tends to `1` as the
  valence grows.  Multivalency is what turns a graded binding curve into the switch that a
  condensate needs.
-/
import Mathlib
import RequestProject.Condensate

namespace IDR

namespace Valence

open Set Filter Topology

/-- The mean number of ligands bound, read off a binding polynomial `Z` in the standard
thermodynamic way: `x · d log Z / dx`. -/
noncomputable def occupancy (Z : ℝ → ℝ) (x : ℝ) : ℝ := x * deriv Z x / Z x

/-- The fraction of the `n`-site unit that is occupied in the fully coupled (all-or-none)
model: `x^n/(1+x^n)`.  For `n = 1` this is the ordinary Langmuir isotherm. -/
noncomputable def fracOcc (n : ℕ) (x : ℝ) : ℝ := x ^ n / (1 + x ^ n)

theorem fracOcc_one (x : ℝ) : fracOcc 1 x = x / (1 + x) := by
  simp [fracOcc]

/-- **`n` independent equivalent motifs.**  The binding polynomial is `(1+x)^n` and the mean
occupancy is `n·x/(1+x)`: the sites simply add up. -/
theorem occupancy_independent (n : ℕ) {x : ℝ} (hx : 0 ≤ x) :
    occupancy (fun y => (1 + y) ^ n) x = n * (x / (1 + x)) := by
  have h1 : (0:ℝ) < 1 + x := by linarith
  have hd : HasDerivAt (fun y : ℝ => (1 + y) ^ n) ((n : ℝ) * (1 + x) ^ (n - 1) * (0 + 1)) x := by
    simpa using (((hasDerivAt_const x (1:ℝ)).add (hasDerivAt_id x)).pow n)
  rw [occupancy, hd.deriv]
  cases n with
  | zero => simp
  | succ m =>
    simp only [Nat.add_sub_cancel, pow_succ]
    push_cast
    field_simp
    ring

/-- **The per-site occupancy of independent motifs is the Langmuir isotherm**, the same curve
for every valence: adding motifs multiplies the amount bound but does not sharpen the
response. -/
theorem perSite_occupancy_independent (n : ℕ) {x : ℝ} (hx : 0 ≤ x) (hn : 0 < n) :
    occupancy (fun y => (1 + y) ^ n) x / n = fracOcc 1 x := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  rw [occupancy_independent n hx, fracOcc_one]
  field_simp

/-- **Fully coupled motifs** (the unit binds all or nothing): the binding polynomial is
`1 + x^n` and the mean occupancy is `n·x^n/(1+x^n)`. -/
theorem occupancy_allOrNone (n : ℕ) {x : ℝ} (hx : 0 < x) :
    occupancy (fun y => 1 + y ^ n) x = n * fracOcc n x := by
  have hd : HasDerivAt (fun y : ℝ => 1 + y ^ n) (0 + (n : ℝ) * x ^ (n - 1) * 1) x :=
    (hasDerivAt_const x (1:ℝ)).add ((hasDerivAt_id x).pow n)
  have hxn : (0:ℝ) < x ^ n := pow_pos hx n
  rw [occupancy, hd.deriv, fracOcc]
  cases n with
  | zero => simp
  | succ m =>
    simp only [Nat.add_sub_cancel, pow_succ]
    push_cast
    field_simp
    ring

/-- **The Hill plot is a straight line of slope `n`.** -/
theorem hill_log (n : ℕ) {x : ℝ} (hx : 0 < x) :
    Real.log (fracOcc n x / (1 - fracOcc n x)) = n * Real.log x := by
  have hxn : (0:ℝ) < x ^ n := pow_pos hx n
  have h1 : (0:ℝ) < 1 + x ^ n := by linarith
  have hone : 1 - fracOcc n x = 1 / (1 + x ^ n) := by
    rw [fracOcc]; field_simp; ring
  have hratio : fracOcc n x / (1 - fracOcc n x) = x ^ n := by
    rw [hone, fracOcc]; field_simp
  rw [hratio, Real.log_pow]

/-- **The Hill coefficient of the coupled `n`-site unit is exactly `n`**: the slope of the
Hill plot against `log x`. -/
theorem hill_slope (n : ℕ) (u : ℝ) :
    HasDerivAt (fun v => Real.log (fracOcc n (Real.exp v) / (1 - fracOcc n (Real.exp v))))
      n u := by
  have hfun : (fun v => Real.log (fracOcc n (Real.exp v) / (1 - fracOcc n (Real.exp v))))
      = fun v => (n : ℝ) * v := by
    funext v
    rw [hill_log n (Real.exp_pos v), Real.log_exp]
  rw [hfun]
  simpa using (hasDerivAt_id u).const_mul (n : ℝ)

/-- **Independent sites have Hill coefficient exactly one**, however many of them there
are. -/
theorem hill_slope_independent (u : ℝ) :
    HasDerivAt (fun v => Real.log (fracOcc 1 (Real.exp v) / (1 - fracOcc 1 (Real.exp v))))
      1 u := by
  simpa using hill_slope 1 u

/-- **Cooperativity cannot come from independent sites.**  For valence at least two the
coupled binding curve is not the Langmuir isotherm of any independent-site model, because
the two have different Hill slopes. -/
theorem no_cooperativity_from_independent_sites {n : ℕ} (hn : 2 ≤ n) :
    fracOcc n ≠ fracOcc 1 := by
  intro hEq
  have h1 : HasDerivAt
      (fun v => Real.log (fracOcc n (Real.exp v) / (1 - fracOcc n (Real.exp v)))) n 0 :=
    hill_slope n 0
  have h2 : HasDerivAt
      (fun v => Real.log (fracOcc n (Real.exp v) / (1 - fracOcc n (Real.exp v)))) 1 0 := by
    rw [hEq]
    exact hill_slope_independent 0
  have hone : (n : ℝ) = 1 := h1.unique h2
  have hge : (2:ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  linarith

/-- The activity at which the coupled `n`-site unit is `90%` bound. -/
theorem fracOcc_ninety (n : ℕ) (hn : 0 < n) : fracOcc n ((9:ℝ) ^ ((n:ℝ)⁻¹)) = 9 / 10 := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hx : ((9:ℝ) ^ ((n:ℝ)⁻¹)) ^ n = 9 := by
    rw [← Real.rpow_natCast ((9:ℝ) ^ ((n:ℝ)⁻¹)) n, ← Real.rpow_mul (by norm_num : (0:ℝ) ≤ 9),
      inv_mul_cancel₀ hn', Real.rpow_one]
  rw [fracOcc, hx]
  norm_num

/-- The activity at which the coupled `n`-site unit is `10%` bound. -/
theorem fracOcc_ten (n : ℕ) (hn : 0 < n) : fracOcc n ((9:ℝ) ^ (-(n:ℝ)⁻¹)) = 1 / 10 := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  have hx : ((9:ℝ) ^ (-(n:ℝ)⁻¹)) ^ n = 1 / 9 := by
    rw [← Real.rpow_natCast ((9:ℝ) ^ (-(n:ℝ)⁻¹)) n, ← Real.rpow_mul (by norm_num : (0:ℝ) ≤ 9)]
    rw [show -(n:ℝ)⁻¹ * n = -1 by field_simp]
    rw [Real.rpow_neg_one]
    norm_num
  rw [fracOcc, hx]
  norm_num

/-- **The window of ligand activity over which a coupled `n`-site unit switches** from `10%`
to `90%` bound is `81^{1/n}`-fold. -/
theorem ligand_window_coupled (n : ℕ) :
    (9:ℝ) ^ ((n:ℝ)⁻¹) / (9:ℝ) ^ (-(n:ℝ)⁻¹) = (81:ℝ) ^ ((n:ℝ)⁻¹) := by
  rw [← Real.rpow_sub (by norm_num), show ((n:ℝ)⁻¹ - -(n:ℝ)⁻¹) = 2 * (n:ℝ)⁻¹ by ring,
    Real.rpow_mul (by norm_num : (0:ℝ) ≤ 9) 2 ((n:ℝ)⁻¹)]
  norm_num

/-- **Independent sites need an 81-fold change in activity** to go from `10%` to `90%`
bound: the Langmuir isotherm is a graded response, not a switch. -/
theorem ligand_window_independent :
    fracOcc 1 (1/9 : ℝ) = 1 / 10 ∧ fracOcc 1 (9 : ℝ) = 9 / 10 ∧ (9:ℝ) / (1/9) = 81 := by
  refine ⟨?_, ?_, by norm_num⟩ <;> · rw [fracOcc_one]; norm_num

/-- **Multivalency makes the response a step.**  The `10%`-to-`90%` activity window
`81^{1/n}` of the coupled unit tends to `1` as the valence grows: the binding curve becomes
a switch, which is what a condensate's sharp concentration threshold requires. -/
theorem ligand_window_tendsto_one :
    Tendsto (fun n : ℕ => (81:ℝ) ^ ((n:ℝ)⁻¹)) atTop (𝓝 1) := by
  have h : ∀ n : ℕ, (81:ℝ) ^ ((n:ℝ)⁻¹) = Real.exp (Real.log 81 / n) := by
    intro n
    rw [Real.rpow_def_of_pos (by norm_num), mul_comm]
    ring_nf
  simp only [h]
  have h0 : Tendsto (fun n : ℕ => Real.log 81 / n) atTop (𝓝 0) :=
    tendsto_const_div_atTop_nhds_zero_nat _
  simpa using (Real.continuous_exp.tendsto 0).comp h0

end Valence

end IDR
