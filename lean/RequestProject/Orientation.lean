/-
# Part LIII.1  The orientation factor: a FRET efficiency is not a distance

Part XLVII compares what small-angle scattering and single-molecule FRET weigh, and states
plainly that the *orientation factor* was left out: the efficiency was taken to be
`R₀⁶/(R₀⁶ + r⁶)` with `R₀` a constant.  It is not a constant.  The Förster radius carries the
factor `κ²`, where

  `κ = d·a - 3 (d·r̂)(a·r̂)`

for donor and acceptor transition dipoles `d`, `a` and the unit separation `r̂`.  Assuming the
"isotropic average" `κ² = 2/3` is the single most common unstated step in the interpretation of
a FRET experiment on a disordered region -- exactly the system in which the linkers are short,
the dyes are sticky, and the rotational averaging cannot be assumed complete.

This file removes the idealisation.

* `kappaSq_nonneg`, `kappaSq_le_four` -- the exact range `0 ≤ κ² ≤ 4`, proved from the
  Gram determinant of the three unit vectors (`gram_identity`, a polynomial identity whose
  right-hand side is the squared triple product, so its nonnegativity is not an assumption).
* `meanKappaSq_eq` -- a finite orientational model that reproduces the textbook average
  exactly: with donor and acceptor each uniform on the six axis directions, `⟨κ²⟩ = 2/3`.
  So the number `2/3` is *correct as an average* -- and that is all it is.
* `meanEff_ne_effMean` -- **the bias.**  The efficiency is a nonlinear function of `κ²`, so
  averaging the orientation and averaging the efficiency are different operations.  For the
  same model, at the distance where the `κ² = 2/3` formula returns `E = 2/5`, the true mean
  efficiency is `1/5`: a factor of two, at the correct mean orientation factor.
* `apparentSixth_of_meanEff` -- read as a distance, that measurement returns `r⁶ = 8/3` when
  the true `r⁶ = 1`: an 18% error in `r`, in a technique whose quoted precision is better.
* `apparentSixth_ratio_six` -- and the residual uncertainty is not removable by better
  photon statistics: with `κ²` known only to lie in its range, the inferred `r⁶` moves by a
  factor of `6` (a factor `6^{1/6} ≈ 1.35` in `r`) at fixed efficiency.

The design consequence for a model of a disordered region is concrete: an ensemble is not
comparable with a FRET efficiency unless the model also carries the dye orientational
distribution, or the analysis reports the κ²-bracket rather than a distance.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

namespace Orientation

open Finset

/-- The Euclidean inner product in three dimensions. -/
def dot (u v : Fin 3 → ℝ) : ℝ := ∑ i, u i * v i

/-- A unit vector. -/
def IsUnitVec (u : Fin 3 → ℝ) : Prop := dot u u = 1

/-- The scalar triple product. -/
def triple (u v w : Fin 3 → ℝ) : ℝ :=
  u 0 * (v 1 * w 2 - v 2 * w 1) - u 1 * (v 0 * w 2 - v 2 * w 0)
    + u 2 * (v 0 * w 1 - v 1 * w 0)

/-- The orientation factor of a donor dipole `d`, an acceptor dipole `a` and the unit vector
`r` along the separation. -/
def kappa (d a r : Fin 3 → ℝ) : ℝ := dot d a - 3 * dot d r * dot a r

/-- The Gram determinant of three vectors is the square of their triple product: the fact that
makes the range of `κ²` a theorem rather than an assumption. -/
theorem gram_identity (d a r : Fin 3 → ℝ) :
    dot d d * dot a a * dot r r + 2 * (dot d a * (dot a r * dot d r))
        - dot d d * dot a r ^ 2 - dot a a * dot d r ^ 2 - dot r r * dot d a ^ 2
      = triple d a r ^ 2 := by
  simp only [dot, triple, Fin.sum_univ_three]
  ring

/-- The Lagrange identity in three dimensions: Cauchy--Schwarz with an explicit remainder. -/
theorem lagrange_identity (u v : Fin 3 → ℝ) :
    dot u u * dot v v - dot u v ^ 2
      = (u 0 * v 1 - u 1 * v 0) ^ 2 + (u 0 * v 2 - u 2 * v 0) ^ 2
        + (u 1 * v 2 - u 2 * v 1) ^ 2 := by
  simp only [dot, Fin.sum_univ_three]
  ring

/-- The orientation factor squared is nonnegative. -/
theorem kappaSq_nonneg (d a r : Fin 3 → ℝ) : 0 ≤ kappa d a r ^ 2 := sq_nonneg _

/-- **The exact range of the orientation factor.**  For unit dipoles and a unit separation
direction, `κ² ≤ 4`, and the bound is attained by collinear dipoles along the separation. -/
theorem kappaSq_le_four {d a r : Fin 3 → ℝ} (hd : IsUnitVec d) (ha : IsUnitVec a)
    (hr : IsUnitVec r) : kappa d a r ^ 2 ≤ 4 := by
  set p := dot d r with hp
  set q := dot a r with hq
  set s := dot d a with hs
  have hgram : 1 + 2 * (s * (q * p)) - q ^ 2 - p ^ 2 - s ^ 2 = triple d a r ^ 2 := by
    have h := gram_identity d a r
    rw [hd, ha, hr] at h
    simpa [hp, hq, hs, mul_comm] using h
  have hgram' : 0 ≤ 1 + 2 * (s * (q * p)) - q ^ 2 - p ^ 2 - s ^ 2 := by
    rw [hgram]; exact sq_nonneg _
  have hp1 : p ^ 2 ≤ 1 := by
    have h := lagrange_identity d r
    rw [hd, hr] at h
    nlinarith [sq_nonneg (d 0 * r 1 - d 1 * r 0), sq_nonneg (d 0 * r 2 - d 2 * r 0),
      sq_nonneg (d 1 * r 2 - d 2 * r 1)]
  have hq1 : q ^ 2 ≤ 1 := by
    have h := lagrange_identity a r
    rw [ha, hr] at h
    nlinarith [sq_nonneg (a 0 * r 1 - a 1 * r 0), sq_nonneg (a 0 * r 2 - a 2 * r 0),
      sq_nonneg (a 1 * r 2 - a 2 * r 1)]
  -- write `s = pq + e`; then `e² ≤ (1-p²)(1-q²)` and `κ = e - 2pq`
  have hkap : kappa d a r = (s - p * q) - 2 * (p * q) := by
    simp only [kappa, ← hp, ← hq, ← hs]; ring
  set e := s - p * q with he
  have heB : e ^ 2 ≤ (1 - p ^ 2) * (1 - q ^ 2) := by nlinarith
  rcases eq_or_lt_of_le (by nlinarith : q ^ 2 ≤ 1) with hqe | hqlt
  · -- `a` is parallel to `r`; then `e = 0`
    have he0 : e = 0 := by nlinarith
    rw [hkap, he0]
    nlinarith
  · have hB : 0 < 1 - q ^ 2 := by linarith
    rw [hkap]
    nlinarith [sq_nonneg (e * q + 2 * p * (1 - q ^ 2)), mul_pos hB hB]

/-! ### A finite orientational model with the textbook average -/

/-- The six axis directions: a finite orientational distribution. -/
def axisVec : Fin 6 → (Fin 3 → ℝ) :=
  ![![1, 0, 0], ![-1, 0, 0], ![0, 1, 0], ![0, -1, 0], ![0, 0, 1], ![0, 0, -1]]

/-- The separation direction, taken along `z`. -/
def ez : Fin 3 → ℝ := ![0, 0, 1]

theorem axisVec_unit (i : Fin 6) : IsUnitVec (axisVec i) := by
  fin_cases i <;> simp [IsUnitVec, dot, Fin.sum_univ_three, axisVec]

theorem ez_unit : IsUnitVec ez := by
  simp [IsUnitVec, dot, Fin.sum_univ_three, ez]

/-- The mean of `κ²` over the finite orientational model. -/
noncomputable def meanKappaSq : ℝ :=
  (1 / 36) * ∑ i, ∑ j, kappa (axisVec i) (axisVec j) ez ^ 2

/-- **The textbook value is exact, as an average.**  With donor and acceptor dipoles uniform
over the six axis directions, `⟨κ²⟩ = 2/3`. -/
theorem meanKappaSq_eq : meanKappaSq = 2 / 3 := by
  simp only [meanKappaSq, kappa, dot, ez, axisVec, Fin.sum_univ_six, Fin.sum_univ_three,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
  norm_num

/-! ### Averaging the orientation is not averaging the efficiency -/

/-- The transfer efficiency at orientation factor `k`, separation `r` and spectroscopic
constant `c` (so that the Förster radius satisfies `R₀⁶ = k c`). -/
noncomputable def orientEff (c k r : ℝ) : ℝ := k * c / (k * c + r ^ 6)

/-- The mean efficiency over the finite orientational model. -/
noncomputable def meanOrientEff (c r : ℝ) : ℝ :=
  (1 / 36) * ∑ i, ∑ j, orientEff c (kappa (axisVec i) (axisVec j) ez ^ 2) r

/-- **The bias.**  At `c = r = 1` the model whose mean orientation factor is exactly `2/3`
transfers with mean efficiency `1/5`, whereas the `κ² = 2/3` formula returns `2/5`. -/
theorem meanEff_ne_effMean :
    meanOrientEff 1 1 = 1 / 5 ∧ orientEff 1 (2 / 3) 1 = 2 / 5 ∧
      meanOrientEff 1 1 ≠ orientEff 1 (2 / 3) 1 := by
  have h1 : meanOrientEff 1 1 = 1 / 5 := by
    simp only [meanOrientEff, orientEff, kappa, dot, ez, axisVec, Fin.sum_univ_six,
      Fin.sum_univ_three, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    norm_num
  have h2 : orientEff 1 (2 / 3) 1 = 2 / 5 := by norm_num [orientEff]
  exact ⟨h1, h2, by rw [h1, h2]; norm_num⟩

/-- The sixth power of the distance inferred from an efficiency `E` under an assumed
orientation factor `k`. -/
noncomputable def apparentSixth (c k E : ℝ) : ℝ := k * c * (1 - E) / E

/-- The inference is exact when the assumed orientation factor is the true one. -/
theorem apparentSixth_orientEff {c k r : ℝ} (hc : 0 < c) (hk : 0 < k) :
    apparentSixth c k (orientEff c k r) = r ^ 6 := by
  have hkc : 0 < k * c := mul_pos hk hc
  have hden : k * c + r ^ 6 ≠ 0 := by positivity
  have hE : orientEff c k r ≠ 0 := by
    simp only [orientEff]
    exact div_ne_zero (ne_of_gt hkc) hden
  simp only [apparentSixth, orientEff]
  field_simp
  ring

/-- **The distance is wrong too.**  Reading the true mean efficiency `1/5` with the assumed
`κ² = 2/3` returns `r⁶ = 8/3`, while the true `r⁶` is `1`. -/
theorem apparentSixth_of_meanEff :
    apparentSixth 1 (2 / 3) (meanOrientEff 1 1) = 8 / 3 := by
  rw [meanEff_ne_effMean.1]
  norm_num [apparentSixth]

/-- **The κ² bracket does not shrink with better data.**  At any fixed efficiency the distance
inferred with the extreme orientation factor `κ² = 4` and with the isotropic value `κ² = 2/3`
differ by a factor of exactly `6` in `r⁶`. -/
theorem apparentSixth_ratio_six (c E : ℝ) :
    apparentSixth c 4 E = 6 * apparentSixth c (2 / 3) E := by
  simp only [apparentSixth]
  ring

end Orientation

end IDR
