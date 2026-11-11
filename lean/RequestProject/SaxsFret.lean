/-
# Part XLVII.1  Which moment does the experiment weigh?

The two experiments that dominate the structural characterisation of disordered regions
disagree, in a way that is not an experimental artefact.  Small-angle scattering reports a
radius of gyration, a *second* moment of the distance distribution; single-molecule FRET reports
a mean transfer efficiency, and the transfer efficiency of a conformation at donor--acceptor
distance `r` is `R₀⁶ / (R₀⁶ + r⁶)`, so the average is dominated by the *sixth* moment and by the
short distances.  Converting the measured efficiency to a distance in the usual way -- solving
`eff R₀ d = E` for `d` -- therefore does not return any mean of the distances present.

This file makes that precise for a finite weighted ensemble.

* `eff_antitone`, `meanEff_pos`, `meanEff_le_one` -- the forward model and its range.
* `apparentSixth`, `apparentDistance`, `apparentDistance_pow_six` -- the standard inversion, in
  a form that avoids roots: the apparent distance to the sixth power is `R₀⁶(1−E)/E`.
* `meanEff_ge` -- **dynamic averaging biases the measured efficiency upward**: the efficiency of
  an ensemble is at least the efficiency of the conformation whose sixth power of distance is
  the ensemble mean.  Proved from the Cauchy--Schwarz (Engel) inequality, not asymptotically.
* `apparentSixth_le_meanSixth` -- equivalently, **the FRET-inferred distance never exceeds the
  sixth-moment mean distance**: heterogeneity makes a chain look more compact than it is.
* `apparentSixth_eq_of_homogeneous` -- the bias vanishes exactly for a homogeneous ensemble, so
  it is a statement about the width of the ensemble, not an offset of the method.
* `apparentSixth_mem_Icc` -- quantitatively, the inferred distance is bracketed by the extreme
  distances present: the discrepancy is bounded by the spread of the ensemble.
* `meanSq_cube_le_meanSixth` -- the second and sixth moments are ordered (power mean), which is
  the only general relation between the two experiments.
* `saxs_blind_to_fret`, `fret_blind_to_saxs` -- and it is only an inequality: two explicit
  two-state ensembles with the *same* mean square distance and different transfer efficiencies,
  and two with the *same* transfer efficiency and different mean square distances.  Neither
  experiment determines the other; a model must be compared with both, through their forward
  models.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset

namespace SaxsFret

variable {n : ℕ}

/-- The FRET transfer efficiency of a conformation with donor--acceptor distance `r` and Förster
radius `R0`. -/
noncomputable def eff (R0 r : ℝ) : ℝ := R0 ^ 6 / (R0 ^ 6 + r ^ 6)

/-- The ensemble-mean transfer efficiency: what a FRET experiment measures. -/
noncomputable def meanEff (R0 : ℝ) (w r : Fin n → ℝ) : ℝ := ∑ i, w i * eff R0 (r i)

/-- The mean sixth power of the distance. -/
noncomputable def meanSixth (w r : Fin n → ℝ) : ℝ := ∑ i, w i * r i ^ 6

/-- The mean square distance: the moment a scattering experiment reports. -/
noncomputable def meanSq (w r : Fin n → ℝ) : ℝ := ∑ i, w i * r i ^ 2

/-! ## The forward model -/

/-- The transfer efficiency is strictly positive for a positive Förster radius. -/
theorem eff_pos {R0 r : ℝ} (hR : 0 < R0) : 0 < eff R0 r := by
  have h6 : (0:ℝ) < R0 ^ 6 := by positivity
  have hden : (0:ℝ) < R0 ^ 6 + r ^ 6 := by positivity
  exact div_pos h6 hden

/-- The transfer efficiency never exceeds `1`. -/
theorem eff_le_one {R0 r : ℝ} (hR : 0 < R0) : eff R0 r ≤ 1 := by
  have hden : (0:ℝ) < R0 ^ 6 + r ^ 6 := by positivity
  have hr : (0:ℝ) ≤ r ^ 6 := by positivity
  rw [eff, div_le_one hden]
  linarith

/-- **The transfer efficiency decreases with distance.** -/
theorem eff_antitone {R0 r s : ℝ} (hR : 0 < R0) (hr : 0 ≤ r) (hrs : r ≤ s) :
    eff R0 s ≤ eff R0 r := by
  have h1 : (0:ℝ) < R0 ^ 6 + r ^ 6 := by positivity
  have h2 : (0:ℝ) < R0 ^ 6 + s ^ 6 := by positivity
  have hpow : r ^ 6 ≤ s ^ 6 := pow_le_pow_left₀ hr hrs 6
  rw [eff, eff, div_le_div_iff₀ h2 h1]
  nlinarith [pow_pos hR 6]

/-- The measured efficiency of an ensemble is positive. -/
theorem meanEff_pos {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ i, 0 < w i)
    (hn : 0 < n) : 0 < meanEff R0 w r := by
  refine sum_pos (fun i _ => mul_pos (hw i) (eff_pos hR)) ?_
  simpa [Finset.univ_nonempty_iff, ← Fin.pos_iff_nonempty] using hn

/-- The measured efficiency of an ensemble never exceeds `1`. -/
theorem meanEff_le_one {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ i, 0 ≤ w i)
    (hsum : ∑ i, w i = 1) : meanEff R0 w r ≤ 1 := by
  calc meanEff R0 w r ≤ ∑ i, w i * 1 :=
        sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (eff_le_one hR) (hw i)
    _ = 1 := by simpa using hsum

/-! ## The inversion -/

/-- The sixth power of the distance a FRET experiment reports: the solution `d` of
`eff R0 d = E`, raised to the sixth power. -/
noncomputable def apparentSixth (R0 E : ℝ) : ℝ := R0 ^ 6 * (1 - E) / E

/-- The distance a FRET experiment reports. -/
noncomputable def apparentDistance (R0 E : ℝ) : ℝ := R0 * ((1 - E) / E) ^ ((1:ℝ)/6)

/-- The apparent distance really is the sixth root of `apparentSixth`. -/
theorem apparentDistance_pow_six {R0 E : ℝ} (hE : 0 ≤ (1 - E) / E) :
    apparentDistance R0 E ^ 6 = apparentSixth R0 E := by
  have h : (((1 - E) / E) ^ ((1:ℝ)/6)) ^ (6:ℕ) = (1 - E) / E := by
    rw [← Real.rpow_natCast (((1 - E) / E) ^ ((1:ℝ)/6)) 6, ← Real.rpow_mul hE]
    norm_num
  rw [apparentDistance, mul_pow, h, apparentSixth, mul_div_assoc]

/-- The inversion is exact on a single conformation. -/
theorem apparentSixth_eff {R0 s : ℝ} (hR : 0 < R0) : apparentSixth R0 (eff R0 s) = s ^ 6 := by
  have hden : (0:ℝ) < R0 ^ 6 + s ^ 6 := by positivity
  have h6 : (0:ℝ) < R0 ^ 6 := by positivity
  rw [apparentSixth, eff]
  field_simp
  ring

/-- The inversion is decreasing in the measured efficiency. -/
theorem apparentSixth_antitone {R0 E F : ℝ} (hR : 0 < R0) (hE : 0 < E) (hEF : E ≤ F) :
    apparentSixth R0 F ≤ apparentSixth R0 E := by
  have hF : 0 < F := lt_of_lt_of_le hE hEF
  have h6 : (0:ℝ) < R0 ^ 6 := by positivity
  rw [apparentSixth, apparentSixth, div_le_div_iff₀ hF hE]
  nlinarith

/-! ## The bias of the inversion -/

/-- **Dynamic averaging biases the measured efficiency upward.**  The efficiency of a
heterogeneous ensemble is at least the efficiency of the single conformation whose sixth power
of distance equals the ensemble mean: short distances dominate the average.  An exact inequality
for every ensemble, from Cauchy--Schwarz. -/
theorem meanEff_ge {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i, w i = 1) :
    R0 ^ 6 / (R0 ^ 6 + meanSixth w r) ≤ meanEff R0 w r := by
  have h6 : (0:ℝ) < R0 ^ 6 := by positivity
  set g : Fin n → ℝ := fun i => w i * ((R0 ^ 6 + r i ^ 6) / R0 ^ 6) with hg
  have hgpos : ∀ i ∈ (univ : Finset (Fin n)), 0 < g i := by
    intro i _
    have hden : (0:ℝ) < R0 ^ 6 + r i ^ 6 := by positivity
    exact mul_pos (hw i) (div_pos hden h6)
  have key := Finset.sq_sum_div_le_sum_sq_div (univ : Finset (Fin n)) w hgpos
  have hgsum : ∑ i, g i = (R0 ^ 6 + meanSixth w r) / R0 ^ 6 := by
    have : ∀ i : Fin n, g i = (w i * R0 ^ 6 + w i * r i ^ 6) / R0 ^ 6 := by
      intro i; rw [hg]; field_simp
    rw [Finset.sum_congr rfl (fun i _ => this i), ← Finset.sum_div, Finset.sum_add_distrib,
      ← Finset.sum_mul, hsum, one_mul, meanSixth]
  have hterm : ∀ i : Fin n, w i ^ 2 / g i = w i * eff R0 (r i) := by
    intro i
    have hden : (0:ℝ) < R0 ^ 6 + r i ^ 6 := by positivity
    have hwi := (hw i).ne'
    rw [hg, eff]
    field_simp
  rw [hgsum, hsum, Finset.sum_congr rfl (fun i _ => hterm i)] at key
  have hMnn : (0:ℝ) ≤ meanSixth w r :=
    Finset.sum_nonneg fun i _ => mul_nonneg (hw i).le (by positivity)
  have hden : (0:ℝ) < R0 ^ 6 + meanSixth w r := by linarith
  calc R0 ^ 6 / (R0 ^ 6 + meanSixth w r)
      = (1:ℝ) ^ 2 / ((R0 ^ 6 + meanSixth w r) / R0 ^ 6) := by
        rw [one_pow, one_div_div]
    _ ≤ meanEff R0 w r := key

/-- **The FRET-inferred distance never exceeds the sixth-moment mean distance.**  A
heterogeneous ensemble is reported as more compact than it is; the deficit is exactly the price
of averaging an `r⁻⁶`-weighted observable. -/
theorem apparentSixth_le_meanSixth {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i, w i = 1) :
    apparentSixth R0 (meanEff R0 w r) ≤ meanSixth w r := by
  have h6 : (0:ℝ) < R0 ^ 6 := by positivity
  have hMnn : (0:ℝ) ≤ meanSixth w r :=
    Finset.sum_nonneg fun i _ => mul_nonneg (hw i).le (by positivity)
  have hden : (0:ℝ) < R0 ^ 6 + meanSixth w r := by linarith
  have hpos : 0 < R0 ^ 6 / (R0 ^ 6 + meanSixth w r) := div_pos h6 hden
  have hle := apparentSixth_antitone (R0 := R0) hR hpos (meanEff_ge hR hw hsum)
  have hval : apparentSixth R0 (R0 ^ 6 / (R0 ^ 6 + meanSixth w r)) = meanSixth w r := by
    rw [apparentSixth]
    field_simp
    ring
  linarith [hle, hval.le, hval.ge]

/-- **No bias for a homogeneous ensemble.**  If every conformation has the same
donor--acceptor distance, the inversion returns it exactly. -/
theorem apparentSixth_eq_of_homogeneous {R0 s : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0)
    (hsum : ∑ i, w i = 1) (hconst : ∀ i, r i = s) :
    apparentSixth R0 (meanEff R0 w r) = s ^ 6 := by
  have hE : meanEff R0 w r = eff R0 s := by
    rw [meanEff, Finset.sum_congr rfl (fun i _ => by rw [hconst i]), ← Finset.sum_mul, hsum,
      one_mul]
  rw [hE, apparentSixth_eff hR]

/-- **The inferred distance is bracketed by the distances present.**  If every conformation has
donor--acceptor distance between `a` and `b`, so does the FRET-inferred distance; the
discrepancy between the experiment and any mean of the ensemble is bounded by its spread. -/
theorem apparentSixth_mem_Icc {R0 a b : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ i, 0 < w i)
    (hsum : ∑ i, w i = 1) (hn : 0 < n) (ha : 0 ≤ a) (hab : ∀ i, a ≤ r i ∧ r i ≤ b) :
    a ^ 6 ≤ apparentSixth R0 (meanEff R0 w r) ∧
      apparentSixth R0 (meanEff R0 w r) ≤ b ^ 6 := by
  have hE : 0 < meanEff R0 w r := meanEff_pos hR hw hn
  have hupper : meanEff R0 w r ≤ eff R0 a := by
    calc meanEff R0 w r ≤ ∑ i, w i * eff R0 a :=
          sum_le_sum fun i _ =>
            mul_le_mul_of_nonneg_left (eff_antitone hR ha (hab i).1) (hw i).le
      _ = eff R0 a := by rw [← Finset.sum_mul, hsum, one_mul]
  have hlower : eff R0 b ≤ meanEff R0 w r := by
    calc eff R0 b = ∑ i, w i * eff R0 b := by rw [← Finset.sum_mul, hsum, one_mul]
      _ ≤ meanEff R0 w r :=
          sum_le_sum fun i _ =>
            mul_le_mul_of_nonneg_left
              (eff_antitone hR (le_trans ha (hab i).1) (hab i).2) (hw i).le
  constructor
  · have := apparentSixth_antitone (R0 := R0) hR hE hupper
    rwa [apparentSixth_eff hR] at this
  · have := apparentSixth_antitone (R0 := R0) hR (eff_pos hR) hlower
    rwa [apparentSixth_eff hR] at this

/-! ## The two moments -/

/-- **Power-mean ordering.**  The mean square distance cubed never exceeds the mean sixth power:
the sixth-moment scale a FRET experiment reports is at least the root-mean-square scale a
scattering experiment reports. -/
theorem meanSq_cube_le_meanSixth {w r : Fin n → ℝ} (hw : ∀ i, 0 ≤ w i) (hsum : ∑ i, w i = 1) :
    meanSq w r ^ 3 ≤ meanSixth w r := by
  have hconv := (convexOn_pow (𝕜 := ℝ) 3).map_sum_le (t := (univ : Finset (Fin n)))
    (w := w) (p := fun i => r i ^ 2) (fun i _ => hw i) hsum
    (fun i _ => by simp [Set.mem_Ici]; positivity)
  simp only [smul_eq_mul] at hconv
  calc meanSq w r ^ 3 = (∑ i, w i * r i ^ 2) ^ 3 := rfl
    _ ≤ ∑ i, w i * (r i ^ 2) ^ 3 := hconv
    _ = meanSixth w r := by
        refine Finset.sum_congr rfl fun i _ => ?_
        ring

/-! ## Neither experiment determines the other -/

/-- Uniform weights on two conformations. -/
noncomputable def half : Fin 2 → ℝ := ![1/2, 1/2]

theorem half_sum : ∑ i, half i = 1 := by
  simp only [half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
  norm_num

theorem half_pos : ∀ i, 0 < half i := by
  intro i
  fin_cases i <;> norm_num [half]

/-- **Scattering does not determine FRET.**  Two two-state ensembles with the same mean square
distance -- the same second moment, hence the same contribution of the labelled pair to a
scattering curve -- have transfer efficiencies `1/2` and `5/9`. -/
theorem saxs_blind_to_fret :
    meanSq half ![1, 1] = meanSq half ![0, Real.sqrt 2] ∧
      meanEff 1 half ![1, 1] = 1/2 ∧
      meanEff 1 half ![0, Real.sqrt 2] = 5/9 := by
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have h6 : Real.sqrt 2 ^ 6 = 8 := by
    rw [show Real.sqrt 2 ^ 6 = (Real.sqrt 2 ^ 2) ^ 3 by ring, h2]; norm_num
  refine ⟨?_, ?_, ?_⟩
  · simp only [meanSq, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      h2]
    norm_num
  · simp only [meanEff, eff, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    norm_num
  · simp only [meanEff, eff, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      h6]
    norm_num

/-- The cube root of `2`: the two distances of the second ensemble below are its square root and
the square root of its reciprocal. -/
noncomputable def tcube : ℝ := (2:ℝ) ^ ((1:ℝ)/3)

theorem tcube_pos : 0 < tcube := Real.rpow_pos_of_pos (by norm_num) _

theorem tcube_cube : tcube ^ 3 = 2 := by
  rw [tcube, ← Real.rpow_natCast ((2:ℝ) ^ ((1:ℝ)/3)) 3, ← Real.rpow_mul (by norm_num)]
  norm_num

theorem tcube_ne_one : tcube ≠ 1 := by
  intro h
  have h3 := tcube_cube
  rw [h] at h3
  norm_num at h3

/-- **FRET does not determine scattering.**  Two two-state ensembles with exactly the same mean
transfer efficiency `1/2` have different mean square distances: the second is strictly larger.
An `r⁻⁶`-weighted average is compatible with a range of second moments. -/
theorem fret_blind_to_saxs :
    meanEff 1 half ![1, 1] = meanEff 1 half ![Real.sqrt tcube, Real.sqrt tcube⁻¹] ∧
      meanSq half ![1, 1] < meanSq half ![Real.sqrt tcube, Real.sqrt tcube⁻¹] := by
  have ht : 0 < tcube := tcube_pos
  have hs2 : Real.sqrt tcube ^ 2 = tcube := Real.sq_sqrt ht.le
  have hi2 : Real.sqrt tcube⁻¹ ^ 2 = tcube⁻¹ := Real.sq_sqrt (by positivity)
  have hs6 : Real.sqrt tcube ^ 6 = 2 := by
    rw [show Real.sqrt tcube ^ 6 = (Real.sqrt tcube ^ 2) ^ 3 by ring, hs2, tcube_cube]
  have hi6 : Real.sqrt tcube⁻¹ ^ 6 = 1/2 := by
    rw [show Real.sqrt tcube⁻¹ ^ 6 = (Real.sqrt tcube⁻¹ ^ 2) ^ 3 by ring, hi2, inv_pow,
      tcube_cube]
    norm_num
  have hA : meanEff 1 half ![1, 1] = 1/2 := by
    simp only [meanEff, eff, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    norm_num
  have hB : meanEff 1 half ![Real.sqrt tcube, Real.sqrt tcube⁻¹] = 1/2 := by
    simp only [meanEff, eff, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      hs6, hi6]
    norm_num
  have hSA : meanSq half ![1, 1] = 1 := by
    simp only [meanSq, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    norm_num
  have hSB : meanSq half ![Real.sqrt tcube, Real.sqrt tcube⁻¹] = (tcube + tcube⁻¹) / 2 := by
    simp only [meanSq, half, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
      hs2, hi2]
    ring
  have hsq : 0 < (tcube - 1) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr tcube_ne_one)
  have key : 2 < tcube + tcube⁻¹ := by
    have hexp : tcube + tcube⁻¹ - 2 = (tcube - 1) ^ 2 / tcube := by
      field_simp; ring
    have := div_pos hsq ht
    linarith
  refine ⟨by rw [hA, hB], ?_⟩
  rw [hSA, hSB]
  linarith

end SaxsFret

end IDR
