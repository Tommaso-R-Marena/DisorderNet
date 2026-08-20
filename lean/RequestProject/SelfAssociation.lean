/-
# Part XVII.1  Concentration: a reported ensemble is a concentration, not a molecule

Disordered regions are weakly self-associating, and the concentrations at which they must be
studied (NMR, SAXS) are far above the concentrations at which they often work.  Any measured
ensemble is therefore an average over the association states present at the measurement
concentration.  This file makes that exact for the simplest and most common case, a
monomer--dimer equilibrium with association constant `K`.

Solving mass action `c = m + 2·K·m²` for the monomer concentration gives the closed form
`monoConc K c = 2c / (1 + sqrt (1 + 8·K·c))` (`mass_action`), so the monomer *fraction* is
`monoFrac K c = 2 / (1 + sqrt (1 + 8·K·c))`.

* `monoFrac_pos`, `monoFrac_le_one`, `monoFrac_zero` -- it is a genuine fraction, equal to one
  at infinite dilution.
* `monoFrac_strictAnti` -- and strictly decreasing in the total concentration at every
  concentration: there is no plateau at which the sample is "monomeric enough".
* `monoFrac_tendsto_zero` -- at high concentration the monomer disappears.
* `apparentObs_strictMono` / `measured_observable_depends_on_concentration` -- consequently any
  observable that distinguishes monomer from dimer drifts strictly with concentration, so a
  reported ensemble is a property of the sample, not of the molecule, unless the concentration
  is reported with it (or the infinite-dilution value is used:
  `apparentObs_at_zero_eq_monomer`).
-/
import Mathlib
import RequestProject.EnsembleCore

namespace IDR

open Filter
open scoped Topology

namespace SelfAssociation

variable {K c : ℝ}

/-- The monomer concentration at total concentration `c`, from mass action. -/
noncomputable def monoConc (K c : ℝ) : ℝ := 2 * c / (1 + Real.sqrt (1 + 8 * K * c))

/-- The monomer fraction at total concentration `c`. -/
noncomputable def monoFrac (K c : ℝ) : ℝ := 2 / (1 + Real.sqrt (1 + 8 * K * c))

lemma sqrt_ge_one (hK : 0 ≤ K) (hc : 0 ≤ c) : 1 ≤ Real.sqrt (1 + 8 * K * c) := by
  have h : (1:ℝ) ≤ 1 + 8 * K * c := by nlinarith
  simpa using Real.sqrt_le_sqrt h

lemma sq_sqrt_arg (hK : 0 ≤ K) (hc : 0 ≤ c) :
    Real.sqrt (1 + 8 * K * c) ^ 2 = 1 + 8 * K * c := by
  have h : (0:ℝ) ≤ 1 + 8 * K * c := by nlinarith
  rw [sq, Real.mul_self_sqrt h]

/-- **Mass action.**  The closed form really is the root of `c = m + 2·K·m²`. -/
theorem mass_action (hK : 0 ≤ K) (hc : 0 ≤ c) :
    monoConc K c + 2 * K * monoConc K c ^ 2 = c := by
  set s := Real.sqrt (1 + 8 * K * c) with hs
  have hs1 : 1 ≤ s := sqrt_ge_one hK hc
  have hsq : s ^ 2 = 1 + 8 * K * c := sq_sqrt_arg hK hc
  have hne : (1 + s) ≠ 0 := by linarith
  simp only [monoConc, ← hs]
  field_simp
  nlinarith [hsq, hs1]

lemma monoConc_eq (c : ℝ) : monoConc K c = c * monoFrac K c := by
  simp only [monoConc, monoFrac]
  ring

lemma monoFrac_pos (hK : 0 ≤ K) (hc : 0 ≤ c) : 0 < monoFrac K c := by
  have hs1 : 1 ≤ Real.sqrt (1 + 8 * K * c) := sqrt_ge_one hK hc
  simp only [monoFrac]
  positivity

lemma monoFrac_le_one (hK : 0 ≤ K) (hc : 0 ≤ c) : monoFrac K c ≤ 1 := by
  have hs1 : 1 ≤ Real.sqrt (1 + 8 * K * c) := sqrt_ge_one hK hc
  rw [monoFrac, div_le_one (by linarith)]
  linarith

@[simp] lemma monoFrac_zero (K : ℝ) : monoFrac K 0 = 1 := by
  simp [monoFrac]
  norm_num

/-- **No sample is dilute enough.**  The monomer fraction strictly decreases with total
concentration, at every concentration. -/
theorem monoFrac_strictAnti (hK : 0 < K) : StrictAntiOn (monoFrac K) (Set.Ici 0) := by
  intro a ha b hb hab
  have ha0 : (0:ℝ) ≤ a := ha
  have hb0 : (0:ℝ) ≤ b := hb
  have hsa : 1 ≤ Real.sqrt (1 + 8 * K * a) := sqrt_ge_one hK.le ha0
  have hsb : 1 ≤ Real.sqrt (1 + 8 * K * b) := sqrt_ge_one hK.le hb0
  have hlt : Real.sqrt (1 + 8 * K * a) < Real.sqrt (1 + 8 * K * b) := by
    refine Real.sqrt_lt_sqrt (by nlinarith) ?_
    nlinarith
  simp only [monoFrac]
  apply div_lt_div_of_pos_left (by norm_num) (by linarith)
  linarith

/-- At high concentration nothing is monomeric. -/
theorem monoFrac_tendsto_zero (hK : 0 < K) :
    Tendsto (monoFrac K) atTop (𝓝 0) := by
  have h1 : Tendsto (fun c : ℝ => 1 + 8 * K * c) atTop atTop := by
    apply Filter.tendsto_atTop_add_const_left
    exact (tendsto_id.const_mul_atTop (by positivity : (0:ℝ) < 8 * K))
  have h2 : Tendsto (fun c : ℝ => Real.sqrt (1 + 8 * K * c)) atTop atTop :=
    Real.tendsto_sqrt_atTop.comp h1
  have h3 : Tendsto (fun c : ℝ => 1 + Real.sqrt (1 + 8 * K * c)) atTop atTop :=
    Filter.tendsto_atTop_add_const_left _ 1 h2
  simpa [monoFrac] using h3.inv_tendsto_atTop.const_mul (2:ℝ)

/-! ## What the experiment reports -/

/-- The apparent value of an observable at total concentration `c`, when the monomer and the
dimer have values `xm` and `xd`: the mass-weighted average. -/
noncomputable def apparentObs (K c xm xd : ℝ) : ℝ :=
  monoFrac K c * xm + (1 - monoFrac K c) * xd

lemma apparentObs_eq (K c xm xd : ℝ) :
    apparentObs K c xm xd = xd + monoFrac K c * (xm - xd) := by
  simp only [apparentObs]; ring

/-- **A measured observable drifts with concentration.**  If the dimer differs from the
monomer in the observable at all, its apparent value changes strictly with the total
concentration -- monotonically, in the direction of the dimer value. -/
theorem measured_observable_depends_on_concentration (hK : 0 < K) {xm xd : ℝ} (hne : xm ≠ xd)
    {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a < b) :
    apparentObs K a xm xd ≠ apparentObs K b xm xd := by
  have hlt := monoFrac_strictAnti hK ha hb hab
  simp only [apparentObs_eq]
  intro hEq
  have : (monoFrac K a - monoFrac K b) * (xm - xd) = 0 := by linarith [hEq]
  rcases mul_eq_zero.mp this with h | h
  · linarith
  · exact hne (by linarith)

/-- The direction of the drift, stated as strict monotonicity: an observable larger in the
monomer than in the dimer is reported strictly smaller as the sample is concentrated. -/
theorem apparentObs_strictAnti (hK : 0 < K) {xm xd : ℝ} (hgt : xd < xm) :
    StrictAntiOn (fun c => apparentObs K c xm xd) (Set.Ici 0) := by
  intro a ha b hb hab
  have hlt := monoFrac_strictAnti hK ha hb hab
  simp only [apparentObs_eq]
  nlinarith

/-- Only the infinite-dilution limit is a property of the molecule. -/
theorem apparentObs_at_zero_eq_monomer (K xm xd : ℝ) :
    apparentObs K 0 xm xd = xm := by
  simp [apparentObs]

end SelfAssociation

end IDR
