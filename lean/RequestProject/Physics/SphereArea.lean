import RequestProject.Physics.Laplacian

/-!
# Part CXXXIX — The two-dimensional Hausdorff measure of the unit sphere in `ℝ³`

This file supplies the geometric groundwork needed to treat solvent-accessible surface area
as an honest Hausdorff measure.  We exhibit an explicit Lipschitz parametrisation of the unit
sphere by a compact subset of the plane (spherical coordinates) and an explicit Lipschitz
projection of the unit sphere onto the unit disc, and deduce that
`μH[2] (sphere (0 : ℝ³) 1)` is positive and finite.  No value of the constant is assumed;
only the two facts that make the SASA theory non-degenerate are proved.
-/

noncomputable section
namespace RequestProject.Physics
open MeasureTheory Metric Set
open scoped ENNReal NNReal

/-- Spherical-coordinate parametrisation of the unit sphere. -/
def sphParam (p : Sp 2) : Sp 3 :=
  (WithLp.equiv 2 (Fin 3 → ℝ)).symm
    ![Real.sin (p 0) * Real.cos (p 1), Real.sin (p 0) * Real.sin (p 1), Real.cos (p 0)]

/-- Orthogonal projection onto the first two coordinates. -/
def proj12 (x : Sp 3) : Sp 2 := (WithLp.equiv 2 (Fin 2 → ℝ)).symm ![x 0, x 1]

@[simp] lemma sphParam_apply0 (p : Sp 2) : sphParam p 0 = Real.sin (p 0) * Real.cos (p 1) := rfl
@[simp] lemma sphParam_apply1 (p : Sp 2) : sphParam p 1 = Real.sin (p 0) * Real.sin (p 1) := rfl
@[simp] lemma sphParam_apply2 (p : Sp 2) : sphParam p 2 = Real.cos (p 0) := rfl
@[simp] lemma proj12_apply0 (x : Sp 3) : proj12 x 0 = x 0 := rfl
@[simp] lemma proj12_apply1 (x : Sp 3) : proj12 x 1 = x 1 := rfl

lemma dist_sq_sp2 (p q : Sp 2) : dist p q ^ 2 = (p 0 - q 0) ^ 2 + (p 1 - q 1) ^ 2 := by
  rw [EuclideanSpace.dist_eq, Real.sq_sqrt (by positivity)]
  simp [Fin.sum_univ_two, Real.dist_eq, sq_abs]

lemma dist_sq_sp3 (x y : Sp 3) :
    dist x y ^ 2 = (x 0 - y 0) ^ 2 + (x 1 - y 1) ^ 2 + (x 2 - y 2) ^ 2 := by
  rw [EuclideanSpace.dist_eq, Real.sq_sqrt (by positivity)]
  simp [Fin.sum_univ_three, Real.dist_eq, sq_abs]

lemma le_of_sq_le_sq_nonneg {a b : ℝ} (hb : 0 ≤ b) (h : a ^ 2 ≤ b ^ 2) : a ≤ b := by
  nlinarith [sq_nonneg (a + b), sq_nonneg (a - b)]

lemma norm_sq_sp3 (x : Sp 3) : ‖x‖ ^ 2 = (x 0) ^ 2 + (x 1) ^ 2 + (x 2) ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  simp [Fin.sum_univ_three, sq_abs]

lemma norm_sq_sp2 (x : Sp 2) : ‖x‖ ^ 2 = (x 0) ^ 2 + (x 1) ^ 2 := by
  rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
  simp [Fin.sum_univ_two, sq_abs]

lemma lipschitz_proj12 : LipschitzWith 1 proj12 := by
  refine LipschitzWith.of_dist_le_mul fun x y => ?_
  push_cast
  refine le_of_sq_le_sq_nonneg (by positivity) ?_
  rw [dist_sq_sp2]
  simp only [proj12_apply0, proj12_apply1, one_mul]
  rw [dist_sq_sp3]
  nlinarith [sq_nonneg (x 2 - y 2)]

private lemma abs_mul_sub_mul_le {u v w z : ℝ} (hu : |u| ≤ 1) (hz : |z| ≤ 1) :
    |u * v - w * z| ≤ |u - w| + |v - z| := by
  have h1 : u * v - w * z = u * (v - z) + (u - w) * z := by ring
  have h2 : |u| * |v - z| ≤ 1 * |v - z| := mul_le_mul_of_nonneg_right hu (abs_nonneg _)
  have h3 : |u - w| * |z| ≤ |u - w| * 1 := mul_le_mul_of_nonneg_left hz (abs_nonneg _)
  calc |u * v - w * z| = |u * (v - z) + (u - w) * z| := by rw [h1]
    _ ≤ |u * (v - z)| + |(u - w) * z| := abs_add_le _ _
    _ = |u| * |v - z| + |u - w| * |z| := by rw [abs_mul, abs_mul]
    _ ≤ |u - w| + |v - z| := by linarith

lemma lipschitz_sphParam : LipschitzWith 3 sphParam := by
  refine LipschitzWith.of_dist_le_mul fun p q => ?_
  push_cast
  refine le_of_sq_le_sq_nonneg (by positivity) ?_
  have hs0 : |Real.sin (p 0) - Real.sin (q 0)| ≤ |p 0 - q 0| := by
    simpa [Real.dist_eq] using Real.lipschitzWith_sin.dist_le_mul (p 0) (q 0)
  have hc0 : |Real.cos (p 0) - Real.cos (q 0)| ≤ |p 0 - q 0| := by
    simpa [Real.dist_eq] using Real.lipschitzWith_cos.dist_le_mul (p 0) (q 0)
  have hs1 : |Real.sin (p 1) - Real.sin (q 1)| ≤ |p 1 - q 1| := by
    simpa [Real.dist_eq] using Real.lipschitzWith_sin.dist_le_mul (p 1) (q 1)
  have hc1 : |Real.cos (p 1) - Real.cos (q 1)| ≤ |p 1 - q 1| := by
    simpa [Real.dist_eq] using Real.lipschitzWith_cos.dist_le_mul (p 1) (q 1)
  have hd0 : |Real.sin (p 0) * Real.cos (p 1) - Real.sin (q 0) * Real.cos (q 1)|
      ≤ |p 0 - q 0| + |p 1 - q 1| :=
    le_trans (abs_mul_sub_mul_le (Real.abs_sin_le_one _) (Real.abs_cos_le_one _))
      (by linarith)
  have hd1 : |Real.sin (p 0) * Real.sin (p 1) - Real.sin (q 0) * Real.sin (q 1)|
      ≤ |p 0 - q 0| + |p 1 - q 1| :=
    le_trans (abs_mul_sub_mul_le (Real.abs_sin_le_one _) (Real.abs_sin_le_one _))
      (by linarith)
  rw [dist_sq_sp3, show (3 * dist p q) ^ 2 = 9 * (dist p q ^ 2) by ring, dist_sq_sp2]
  simp only [sphParam_apply0, sphParam_apply1, sphParam_apply2]
  nlinarith [sq_abs (Real.sin (p 0) * Real.cos (p 1) - Real.sin (q 0) * Real.cos (q 1)),
    sq_abs (Real.sin (p 0) * Real.sin (p 1) - Real.sin (q 0) * Real.sin (q 1)),
    sq_abs (Real.cos (p 0) - Real.cos (q 0)), sq_abs (p 0 - q 0), sq_abs (p 1 - q 1),
    abs_nonneg (p 0 - q 0), abs_nonneg (p 1 - q 1),
    sq_nonneg (|p 0 - q 0| - |p 1 - q 1|), hd0, hd1, hc0,
    abs_nonneg (Real.sin (p 0) * Real.cos (p 1) - Real.sin (q 0) * Real.cos (q 1)),
    abs_nonneg (Real.sin (p 0) * Real.sin (p 1) - Real.sin (q 0) * Real.sin (q 1)),
    abs_nonneg (Real.cos (p 0) - Real.cos (q 0))]

/-- `μH[2]` is an additive Haar measure on the Euclidean plane. -/
instance hausdorff2_isAddHaar : (μH[(2 : ℝ)] : Measure (Sp 2)).IsAddHaarMeasure := by
  have h : Module.finrank ℝ (Sp 2) = 2 := by simp
  have hinst : (μH[(Module.finrank ℝ (Sp 2) : ℝ)] : Measure (Sp 2)).IsAddHaarMeasure :=
    MeasureTheory.isAddHaarMeasure_hausdorffMeasure
  rw [h] at hinst
  simpa using hinst

lemma unitDisc_subset_proj_sphere :
    ball (0 : Sp 2) 1 ⊆ proj12 '' (sphere (0 : Sp 3) 1) := by
  intro v hv
  have hv1 : ‖v‖ < 1 := by simpa [mem_ball, dist_eq_norm] using hv
  have hnn : (0 : ℝ) ≤ 1 - ‖v‖ ^ 2 := by nlinarith [norm_nonneg v]
  refine ⟨(WithLp.equiv 2 (Fin 3 → ℝ)).symm ![v 0, v 1, Real.sqrt (1 - ‖v‖ ^ 2)], ?_, ?_⟩
  · have hx : ‖(WithLp.equiv 2 (Fin 3 → ℝ)).symm
        ![v 0, v 1, Real.sqrt (1 - ‖v‖ ^ 2)]‖ ^ 2 = 1 := by
      rw [norm_sq_sp3]
      have h0 : ((WithLp.equiv 2 (Fin 3 → ℝ)).symm
          ![v 0, v 1, Real.sqrt (1 - ‖v‖ ^ 2)] : Sp 3) 0 = v 0 := rfl
      have h1 : ((WithLp.equiv 2 (Fin 3 → ℝ)).symm
          ![v 0, v 1, Real.sqrt (1 - ‖v‖ ^ 2)] : Sp 3) 1 = v 1 := rfl
      have h2 : ((WithLp.equiv 2 (Fin 3 → ℝ)).symm
          ![v 0, v 1, Real.sqrt (1 - ‖v‖ ^ 2)] : Sp 3) 2 = Real.sqrt (1 - ‖v‖ ^ 2) := rfl
      rw [h0, h1, h2, Real.sq_sqrt hnn, norm_sq_sp2 v]
      ring
    have hnn' : (0 : ℝ) ≤ ‖(WithLp.equiv 2 (Fin 3 → ℝ)).symm
        ![v 0, v 1, Real.sqrt (1 - ‖v‖ ^ 2)]‖ := norm_nonneg _
    simp only [mem_sphere_iff_norm, sub_zero]
    nlinarith
  · ext i
    fin_cases i <;> rfl

lemma sphere_subset_sphParam_image :
    sphere (0 : Sp 3) 1 ⊆ sphParam '' (closedBall (0 : Sp 2) 10) := by
  intro x hx
  have hnorm : ‖x‖ = 1 := by simpa [mem_sphere_iff_norm] using hx
  have hsq : (x 0) ^ 2 + (x 1) ^ 2 + (x 2) ^ 2 = 1 := by
    rw [← norm_sq_sp3, hnorm]; norm_num
  set c := x 2 with hc
  have hc1 : c ^ 2 ≤ 1 := by nlinarith [sq_nonneg (x 0), sq_nonneg (x 1)]
  have hcle : -1 ≤ c ∧ c ≤ 1 := abs_le.mp (by rw [← Real.sqrt_sq_eq_abs]; nlinarith [Real.sq_sqrt (le_of_lt (by nlinarith : (0:ℝ) < 1)), Real.sqrt_le_sqrt hc1, Real.sqrt_one, Real.sqrt_sq_eq_abs c, abs_nonneg c])
  set th := Real.arccos c with hth
  have hcosth : Real.cos th = c := Real.cos_arccos hcle.1 hcle.2
  have hsinth : Real.sin th = Real.sqrt (1 - c ^ 2) := Real.sin_arccos c
  set s := Real.sqrt (1 - c ^ 2) with hs
  have hsnn : 0 ≤ s := Real.sqrt_nonneg _
  have hs2 : s ^ 2 = (x 0) ^ 2 + (x 1) ^ 2 := by
    rw [hs, Real.sq_sqrt (by nlinarith)]
    nlinarith
  have hthbd : |th| ≤ 4 := by
    rw [abs_of_nonneg (Real.arccos_nonneg c)]
    exact le_trans (Real.arccos_le_pi c) (by linarith [Real.pi_le_four])
  -- the azimuthal angle
  obtain ⟨phi, hphibd, hcos, hsin⟩ :
      ∃ phi : ℝ, |phi| ≤ 4 ∧ s * Real.cos phi = x 0 ∧ s * Real.sin phi = x 1 := by
    rcases eq_or_lt_of_le hsnn with hs0 | hs0
    · refine ⟨0, by norm_num, ?_, ?_⟩ <;> · rw [← hs0]; nlinarith [hs2, sq_nonneg (x 0), sq_nonneg (x 1)]
    · have hu : (x 0 / s) ^ 2 ≤ 1 := by
        rw [div_pow]
        rw [div_le_one (by positivity)]
        nlinarith [sq_nonneg (x 1)]
      have hule : -1 ≤ x 0 / s ∧ x 0 / s ≤ 1 := abs_le.mp (by
        rw [← Real.sqrt_sq_eq_abs]
        nlinarith [Real.sqrt_le_sqrt hu, Real.sqrt_one, Real.sqrt_nonneg ((x 0 / s) ^ 2)])
      set al := Real.arccos (x 0 / s) with hal
      have hcosal : Real.cos al = x 0 / s := Real.cos_arccos hule.1 hule.2
      have hsinal : Real.sin al = Real.sqrt (1 - (x 0 / s) ^ 2) := Real.sin_arccos _
      have halbd : |al| ≤ 4 := by
        rw [abs_of_nonneg (Real.arccos_nonneg _)]
        exact le_trans (Real.arccos_le_pi _) (by linarith [Real.pi_le_four])
      have hsq2 : Real.sqrt (1 - (x 0 / s) ^ 2) = |x 1| / s := by
        rw [show (1 : ℝ) - (x 0 / s) ^ 2 = (x 1 / s) ^ 2 by field_simp; nlinarith]
        rw [Real.sqrt_sq_eq_abs, abs_div, abs_of_pos hs0]
      by_cases hb : 0 ≤ x 1
      · refine ⟨al, halbd, ?_, ?_⟩
        · rw [hcosal]; field_simp
        · rw [hsinal, hsq2, abs_of_nonneg hb]; field_simp
      · refine ⟨-al, by simpa using halbd, ?_, ?_⟩
        · rw [Real.cos_neg, hcosal]; field_simp
        · rw [Real.sin_neg, hsinal, hsq2, abs_of_neg (not_le.mp hb)]; field_simp
  refine ⟨(WithLp.equiv 2 (Fin 2 → ℝ)).symm ![th, phi], ?_, ?_⟩
  · have h0 : ((WithLp.equiv 2 (Fin 2 → ℝ)).symm ![th, phi] : Sp 2) 0 = th := rfl
    have h1 : ((WithLp.equiv 2 (Fin 2 → ℝ)).symm ![th, phi] : Sp 2) 1 = phi := rfl
    simp only [mem_closedBall, dist_eq_norm, sub_zero]
    refine le_of_sq_le_sq_nonneg (by norm_num) ?_
    rw [norm_sq_sp2, h0, h1]
    nlinarith [abs_nonneg th, abs_nonneg phi, sq_abs th, sq_abs phi, hthbd, hphibd]
  · have h0 : ((WithLp.equiv 2 (Fin 2 → ℝ)).symm ![th, phi] : Sp 2) 0 = th := rfl
    have h1 : ((WithLp.equiv 2 (Fin 2 → ℝ)).symm ![th, phi] : Sp 2) 1 = phi := rfl
    ext i
    fin_cases i
    · show Real.sin th * Real.cos phi = _
      rw [h0, h1] at *
      rw [hsinth]; exact hcos
    · show Real.sin th * Real.sin phi = _
      rw [h0, h1] at *
      rw [hsinth]; exact hsin
    · show Real.cos th = _
      rw [h0] at *
      rw [hcosth]; rfl

theorem unit_sphere_area_pos : 0 < μH[(2 : ℝ)] (sphere (0 : Sp 3) 1) := by
  have h1 : μH[(2 : ℝ)] (ball (0 : Sp 2) 1) ≤ μH[(2 : ℝ)] (proj12 '' (sphere (0 : Sp 3) 1)) :=
    measure_mono unitDisc_subset_proj_sphere
  have h2 : μH[(2 : ℝ)] (proj12 '' (sphere (0 : Sp 3) 1))
      ≤ ((1 : ℝ≥0) : ℝ≥0∞) ^ (2 : ℝ) * μH[(2 : ℝ)] (sphere (0 : Sp 3) 1) :=
    lipschitz_proj12.hausdorffMeasure_image_le (by norm_num) _
  have h3 : 0 < μH[(2 : ℝ)] (ball (0 : Sp 2) 1) := measure_ball_pos _ _ one_pos
  have h4 : ((1 : ℝ≥0) : ℝ≥0∞) ^ (2 : ℝ) = 1 := by simp
  rw [h4, one_mul] at h2
  exact lt_of_lt_of_le h3 (h1.trans h2)

theorem unit_sphere_area_ne_top : μH[(2 : ℝ)] (sphere (0 : Sp 3) 1) ≠ ⊤ := by
  have h1 : μH[(2 : ℝ)] (sphere (0 : Sp 3) 1)
      ≤ μH[(2 : ℝ)] (sphParam '' closedBall (0 : Sp 2) 10) :=
    measure_mono sphere_subset_sphParam_image
  have h2 : μH[(2 : ℝ)] (sphParam '' closedBall (0 : Sp 2) 10)
      ≤ ((3 : ℝ≥0) : ℝ≥0∞) ^ (2 : ℝ) * μH[(2 : ℝ)] (closedBall (0 : Sp 2) 10) :=
    lipschitz_sphParam.hausdorffMeasure_image_le (by norm_num) _
  have h3 : μH[(2 : ℝ)] (closedBall (0 : Sp 2) 10) < ⊤ :=
    (isCompact_closedBall _ _).measure_lt_top
  have h4 : ((3 : ℝ≥0) : ℝ≥0∞) ^ (2 : ℝ) ≠ ⊤ := by
    rw [← ENNReal.coe_rpow_of_ne_zero (by norm_num)]
    exact ENNReal.coe_ne_top
  have : μH[(2 : ℝ)] (sphere (0 : Sp 3) 1) < ⊤ :=
    lt_of_le_of_lt (h1.trans h2) (ENNReal.mul_lt_top (lt_top_iff_ne_top.mpr h4) h3)
  exact this.ne

end RequestProject.Physics
