/-
# Part CXL  What a pressure experiment can buy: resolution, blindness, extrapolation

Part CXXXVIII built the pressure axis and proved the exact laws it obeys.  This part asks the
design question that every earlier probe in this project was put through: given a real pressure
cell — a bounded pressure range and a finite energy resolution — *which* features of a model of a
disordered region does the experiment actually constrain?

Three answers, all exact.

* **Resolution.**  A volume difference enters the free energy only as `p·ΔV`, so over a range
  `[0, P]` read to `eps` it is invisible unless `|ΔV| ≥ eps/P` (`volume_difference_invisible`),
  and visible — at the top pressure, which is the optimal condition — as soon as it is
  (`volume_difference_detectable`).  The floor is exact in both directions.  In laboratory units
  the arithmetic is friendly, because one megapascal times one millilitre is one joule:
  `demo_volume_floor` shows that a `200 MPa` cell read to `10⁻² kT` cannot see a volume change of
  `0.1 mL/mol`, which is of the order of a single buried water molecule per chain.
* **Blindness.**  Pressure couples to nothing but volume.  Formally, an observable whose
  covariance with the partial molar volume vanishes has zero pressure response
  (`pressure_blind_of_cov_zero`), and — exactly, not just to first order — an ensemble whose
  conformers share a volume has *every* average independent of pressure
  (`mean_const_of_volumes_equal`).  A pressure series adds information about a coordinate only in
  so far as that coordinate is correlated with solvent-excluded volume; it is the natural
  complement to a size probe, and no substitute for one.
* **Extrapolation.**  `two_point_extrapolation_arbitrary` is the sharpest statement of Part
  CXXXVIII's identifiability boundary: from data at two pressures, the free-energy difference at
  *any* third pressure can be made to take *any* value whatever by a second-order model that fits
  both data points exactly.  Two-point pressure data therefore support no extrapolation at all —
  in particular none back to atmospheric pressure, which is the number usually quoted.
* `pressure_design_report` collects the clauses.
-/
import Mathlib
import RequestProject.PressureEnsemble

namespace RequestProject.PressureDesign

open RequestProject.PressureEnsemble

/-! ## Resolution: the volume floor of a bounded pressure range -/

/-- **Below the floor.**  If `|ΔV|·P < eps` then two models whose volume changes differ by `ΔV`
predict free-energy differences closer than the resolution at *every* accessible pressure. -/
theorem volume_difference_invisible {dV eps P : ℝ} (h : |dV| * P < eps) {p : ℝ}
    (hp0 : 0 ≤ p) (hpP : p ≤ P) (dG0 : ℝ) :
    |(dG0 + p * (dV)) - (dG0 + p * 0)| < eps := by
  have hred : (dG0 + p * dV) - (dG0 + p * 0) = p * dV := by ring
  have habs : |p * dV| = p * |dV| := by
    rw [abs_mul, abs_of_nonneg hp0]
  have hmono : p * |dV| ≤ P * |dV| :=
    mul_le_mul_of_nonneg_right hpP (abs_nonneg dV)
  have hcomm : |dV| * P = P * |dV| := by ring
  rw [hred, habs]
  linarith [hcomm ▸ h]

/-- **Above the floor.**  If `|ΔV|·P ≥ eps` and the resolution is positive, the top of the
pressure range separates the two models. -/
theorem volume_difference_detectable {dV eps P : ℝ} (hP : 0 ≤ P) (h : eps ≤ |dV| * P)
    (dG0 : ℝ) : eps ≤ |(dG0 + P * dV) - (dG0 + P * 0)| := by
  have hred : (dG0 + P * dV) - (dG0 + P * 0) = P * dV := by ring
  have habs : |P * dV| = P * |dV| := by rw [abs_mul, abs_of_nonneg hP]
  rw [hred, habs]
  linarith [h]

/-- **The floor in laboratory units.**  Pressures in megapascals, volumes in millilitres per
mole, energies in joules per mole (`1 MPa · 1 mL = 1 J`), and `kT = 2494 J/mol` at 300 K.  A cell
reaching `200 MPa`, read to `10⁻² kT`, cannot detect a volume change of `0.1 mL/mol` at any
accessible pressure. -/
theorem demo_volume_floor {dV p : ℝ} (hdV : |dV| ≤ 1 / 10) (hp0 : 0 ≤ p) (hpP : p ≤ 200)
    (dG0 : ℝ) : |(dG0 + p * dV) - (dG0 + p * 0)| < (1 / 100) * 2494 := by
  refine volume_difference_invisible ?_ hp0 hpP dG0
  have : |dV| * 200 ≤ (1 / 10) * 200 := by
    exact mul_le_mul_of_nonneg_right hdV (by norm_num)
  norm_num at this ⊢
  linarith

/-! ## Blindness: pressure couples to volume and to nothing else -/

variable {ι : Type*} [Fintype ι] [Nonempty ι]

/-- An observable uncorrelated with the partial molar volume has zero pressure response. -/
theorem pressure_blind_of_cov_zero (E : Ensemble ι) (p : ℝ) (f : ι → ℝ)
    (h : cov E p E.V f = 0) : HasDerivAt (fun p => mean E p f) 0 p := by
  have := hasDerivAt_mean E p f
  rwa [h, neg_zero] at this

omit [Nonempty ι] in
/-- **Exactly blind.**  If all conformers have the same partial molar volume, then *every*
ensemble average is independent of pressure: such an ensemble is invisible to a pressure
experiment, however precise and however wide its range. -/
theorem mean_const_of_volumes_equal (E : Ensemble ι) (v : ℝ) (hV : ∀ i, E.V i = v)
    (f : ι → ℝ) (p q : ℝ) : mean E p f = mean E q f := by
  have hshift : ∀ (r : ℝ) (i : ι), wt E r i = Real.exp (-(r * v)) * wt E 0 i := by
    intro r i
    simp only [wt, hV i, ← Real.exp_add]
    ring_nf
  have hnum : ∀ r : ℝ, ∑ i, wt E r i * f i
      = Real.exp (-(r * v)) * ∑ i, wt E 0 i * f i := by
    intro r
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by rw [hshift r i]; ring
  have hden : ∀ r : ℝ, Z E r = Real.exp (-(r * v)) * Z E 0 := by
    intro r
    rw [Z, Z, Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => hshift r i
  have key : ∀ r : ℝ, mean E r f = mean E 0 f := by
    intro r
    rw [mean, mean, hnum r, hden r, mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [key p, key q]

/-! ## Extrapolation: two pressures constrain nothing off the two points -/

/-- **No extrapolation from two pressures.**  Given free-energy differences measured at `p₁` and
`p₂`, and *any* desired value `y` at a third pressure `p₃`, there is a second-order pressure
model reproducing both measurements exactly and predicting `y` at `p₃`.  The free energy away
from the measured pressures — in particular the extrapolation to ambient pressure — is therefore
entirely a modelling choice, not a measurement. -/
theorem two_point_extrapolation_arbitrary (a b c p₁ p₂ p₃ y : ℝ)
    (h₁₃ : p₃ ≠ p₁) (h₂₃ : p₃ ≠ p₂) :
    ∃ a' b' c' : ℝ, dGq a' b' c' p₁ = dGq a b c p₁ ∧ dGq a' b' c' p₂ = dGq a b c p₂ ∧
      dGq a' b' c' p₃ = y := by
  have hn₁ : p₃ - p₁ ≠ 0 := sub_ne_zero.2 h₁₃
  have hn₂ : p₃ - p₂ ≠ 0 := sub_ne_zero.2 h₂₃
  have hne : (p₃ - p₁) * (p₃ - p₂) ≠ 0 := mul_ne_zero hn₁ hn₂
  set k : ℝ := -(y - dGq a b c p₃) / ((p₃ - p₁) * (p₃ - p₂)) with hk
  obtain ⟨a', b', e₁, e₂, e₃⟩ :=
    pressure_curve_two_point_underdetermined a b c (c + 2 * k) p₁ p₂
  refine ⟨a', b', c + 2 * k, e₁, e₂, ?_⟩
  have h3 := e₃ p₃
  have hcoef : ((c + 2 * k) - c) / 2 = k := by ring
  rw [hcoef] at h3
  have : -k * (p₃ - p₁) * (p₃ - p₂) = y - dGq a b c p₃ := by
    rw [hk]
    field_simp
  linarith [h3, this]

omit [Nonempty ι] in
/-- **The pressure design report.**  (1) a volume difference below `eps/P` is invisible over the
whole range, (2) at or above it the top pressure separates, (3) an ensemble of equal-volume
conformers has no pressure response whatever, and (4) two-pressure data determine the free energy
at no other pressure. -/
theorem pressure_design_report (E : Ensemble ι) :
    (∀ dV eps P p dG0 : ℝ, |dV| * P < eps → 0 ≤ p → p ≤ P →
      |(dG0 + p * dV) - (dG0 + p * 0)| < eps) ∧
    (∀ dV eps P dG0 : ℝ, 0 ≤ P → eps ≤ |dV| * P →
      eps ≤ |(dG0 + P * dV) - (dG0 + P * 0)|) ∧
    (∀ v : ℝ, (∀ i, E.V i = v) → ∀ (f : ι → ℝ) (p q : ℝ), mean E p f = mean E q f) ∧
    (∀ a b c p₁ p₂ p₃ y : ℝ, p₃ ≠ p₁ → p₃ ≠ p₂ → ∃ a' b' c' : ℝ,
      dGq a' b' c' p₁ = dGq a b c p₁ ∧ dGq a' b' c' p₂ = dGq a b c p₂ ∧
      dGq a' b' c' p₃ = y) :=
  ⟨fun _ _ _ _ dG0 h hp0 hpP => volume_difference_invisible h hp0 hpP dG0,
   fun _ _ _ dG0 hP h => volume_difference_detectable hP h dG0,
   fun v hV f p q => mean_const_of_volumes_equal E v hV f p q,
   fun a b c p₁ p₂ p₃ y h₁₃ h₂₃ => two_point_extrapolation_arbitrary a b c p₁ p₂ p₃ y h₁₃ h₂₃⟩

end RequestProject.PressureDesign
