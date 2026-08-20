import RequestProject.Physics.SphereArea

/-!
# Part CXL — Solvent-accessible surface area as a Hausdorff measure

The solvent-accessible surface area (SASA) of atom `i` in a molecule is the two-dimensional
Hausdorff measure of the part of its (solvent-inflated) sphere that is not swallowed by any
other atom's ball.  No smoothness, no triangulation and no numerical quadrature is assumed:
the object is the exact Hausdorff measure of a highly irregular set cut out of a sphere by
finitely many intersecting balls.

We prove the exact scaling law of the sphere measure, positivity and finiteness of the unit
sphere area, and then the structural theorems that any faithful SASA model must satisfy:
the buried atom has zero area, the isolated atom has the full sphere area, SASA is
monotone (antitone) under growing neighbours, invariant under rigid motions, homogeneous of
degree two under dilation, and obeys the two-sided occlusion bounds used by every practical
SASA algorithm.
-/

noncomputable section

namespace RequestProject.Physics

open MeasureTheory Metric Set
open scoped ENNReal NNReal Pointwise

variable {N : ℕ}

/-- The exposed part of atom `i`'s sphere. -/
def sasaSet (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N) : Set (Sp 3) :=
  sphere (c i) (rad i) \ ⋃ j ∈ ({i}ᶜ : Set (Fin N)), ball (c j) (rad j)

/-- The solvent-accessible surface area of atom `i`, as a two-dimensional Hausdorff
measure. -/
def sasa (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N) : ℝ≥0∞ :=
  μH[2] (sasaSet c rad i)

/-- Area of the unit sphere of three-dimensional space, as a Hausdorff measure. -/
def unitSphereArea : ℝ≥0∞ := μH[2] (sphere (0 : Sp 3) 1)

/-- The area of a sphere of radius `r`. -/
def sphereArea (r : ℝ) : ℝ≥0∞ := ENNReal.ofReal (r ^ 2) * unitSphereArea

/-- **Exact scaling law**: the two-dimensional Hausdorff measure of a sphere of radius `r`
is `r²` times that of the unit sphere. -/
theorem hausdorff_sphere_eq (c : Sp 3) {r : ℝ} (hr : 0 ≤ r) :
    μH[2] (sphere c r) = sphereArea r := by
  rcases eq_or_lt_of_le hr with h | hr'
  · subst_vars
    have hs : sphere c (0:ℝ) = ({c} : Set (Sp 3)) := by ext x; simp
    haveI := Measure.noAtoms_hausdorff (Sp 3) (d := (2:ℝ)) (by norm_num)
    rw [hs, measure_singleton]
    simp [sphereArea]
  · have hsm : r • sphere (0 : Sp 3) 1 = sphere 0 r := by
      rw [_root_.smul_sphere r (0 : Sp 3) (by norm_num : (0:ℝ) ≤ 1)]
      simp [Real.norm_eq_abs, abs_of_pos hr']
    have hvadd : c +ᵥ sphere (0 : Sp 3) r = sphere c r := by rw [vadd_sphere]; simp
    calc μH[2] (sphere c r) = μH[2] (c +ᵥ sphere (0 : Sp 3) r) := by rw [hvadd]
      _ = μH[2] (sphere (0 : Sp 3) r) := hausdorffMeasure_vadd c (Or.inl (by norm_num)) _
      _ = μH[2] (r • sphere (0 : Sp 3) 1) := by rw [hsm]
      _ = ‖r‖₊ ^ (2:ℝ) • μH[2] (sphere (0 : Sp 3) 1) :=
          Measure.hausdorffMeasure_smul₀ (by norm_num) hr'.ne' _
      _ = sphereArea r := by
          have h1 : ‖r‖₊ ^ (2:ℝ) = ‖r‖₊ ^ (2:ℕ) := by rw [← NNReal.rpow_natCast]; norm_num
          rw [h1, ENNReal.smul_def, smul_eq_mul, sphereArea, unitSphereArea]
          congr 1
          rw [ENNReal.coe_pow, ENNReal.ofReal_pow hr'.le]
          congr 1
          simp [ENNReal.ofReal, Real.nnnorm_of_nonneg hr'.le, Real.toNNReal_of_nonneg hr'.le]

/-- The unit sphere has positive area. -/
theorem unitSphereArea_pos : 0 < unitSphereArea := unit_sphere_area_pos

/-- The unit sphere has finite area. -/
theorem unitSphereArea_ne_top : unitSphereArea ≠ ⊤ := unit_sphere_area_ne_top

/-! ### Structural theorems for SASA -/

/-- SASA never exceeds the full sphere area. -/
theorem sasa_le_sphereArea (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N)
    (hr : 0 ≤ rad i) : sasa c rad i ≤ sphereArea (rad i) := by
  rw [← hausdorff_sphere_eq (c i) hr]
  exact measure_mono diff_subset

/-- **A buried atom has zero accessible area.** If some other atom's ball contains atom
`i`'s sphere, atom `i` contributes nothing. -/
theorem sasa_eq_zero_of_buried {c : Fin N → Sp 3} {rad : Fin N → ℝ} {i j : Fin N}
    (hij : j ≠ i) (h : dist (c i) (c j) + rad i < rad j) :
    sasa c rad i = 0 := by
  have hempty : sasaSet c rad i = ∅ := by
    ext x
    simp only [sasaSet, mem_diff, mem_empty_iff_false, iff_false, not_and, not_not]
    intro hx
    refine mem_iUnion₂.2 ⟨j, by simpa using hij, ?_⟩
    have h1 : dist x (c j) ≤ dist x (c i) + dist (c i) (c j) := dist_triangle _ _ _
    have h2 : dist x (c i) = rad i := hx
    simp only [mem_ball]
    linarith
  simp [sasa, hempty]

/-- **An isolated atom exposes its whole sphere.** -/
theorem sasa_eq_sphereArea_of_isolated {c : Fin N → Sp 3} {rad : Fin N → ℝ} {i : Fin N}
    (hr : 0 ≤ rad i)
    (h : ∀ j, j ≠ i → rad i + rad j ≤ dist (c i) (c j)) :
    sasa c rad i = sphereArea (rad i) := by
  have hset : sasaSet c rad i = sphere (c i) (rad i) := by
    refine Set.Subset.antisymm diff_subset (fun x hx => ⟨hx, ?_⟩)
    intro hmem
    obtain ⟨j, hj, hxj⟩ := mem_iUnion₂.1 hmem
    have hj' : j ≠ i := by simpa using hj
    have h1 : dist (c i) (c j) ≤ dist (c i) x + dist x (c j) := dist_triangle _ _ _
    have h2 : dist x (c i) = rad i := hx
    have h3 : dist (c i) x = rad i := by rw [dist_comm]; exact h2
    have h4 : dist x (c j) < rad j := hxj
    have h5 := h j hj'
    linarith
  rw [sasa, hset, hausdorff_sphere_eq (c i) hr]

/-- **Growing the neighbours can only bury more surface.** -/
theorem sasa_antitone_radii {c : Fin N → Sp 3} {rad rad' : Fin N → ℝ} {i : Fin N}
    (hi : rad' i = rad i) (h : ∀ j, j ≠ i → rad j ≤ rad' j) :
    sasa c rad' i ≤ sasa c rad i := by
  refine measure_mono ?_
  intro x hx
  refine ⟨by rw [← hi]; exact hx.1, ?_⟩
  intro hmem
  obtain ⟨j, hj, hxj⟩ := mem_iUnion₂.1 hmem
  have hj' : j ≠ i := by simpa using hj
  exact hx.2 (mem_iUnion₂.2 ⟨j, hj, lt_of_lt_of_le hxj (h j hj')⟩)

/-- The exposed set transforms covariantly under an isometry of space. -/
theorem sasaSet_isometry (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N) (e : Sp 3 ≃ᵢ Sp 3) :
    sasaSet (fun k => e (c k)) rad i = e '' sasaSet c rad i := by
  rw [sasaSet, sasaSet, Set.image_diff e.injective, e.image_sphere, Set.image_iUnion₂]
  simp only [e.image_ball]

/-- **Invariance under any isometry of space** (in particular under rotations). -/
theorem sasa_isometry (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N)
    (e : Sp 3 ≃ᵢ Sp 3) :
    sasa (fun k => e (c k)) rad i = sasa c rad i := by
  rw [sasa, sasaSet_isometry, e.hausdorffMeasure_image, sasa]

/-- **Rigid-motion invariance.** SASA is unchanged by translating the whole molecule. -/
theorem sasa_translation (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N) (t : Sp 3) :
    sasa (fun k => c k + t) rad i = sasa c rad i := by
  have h := sasa_isometry c rad i (IsometryEquiv.constVAdd t : Sp 3 ≃ᵢ Sp 3)
  simpa [IsometryEquiv.constVAdd, add_comm] using h

/-- Dilating a sphere by a positive factor. -/
theorem smul_sphere_of_pos {lam : ℝ} (hlam : 0 < lam) (x : Sp 3) (r : ℝ) :
    lam • sphere x r = sphere (lam • x) (lam * r) := by
  ext y
  simp only [Set.mem_smul_set, mem_sphere_iff_norm]
  constructor
  · rintro ⟨z, hz, rfl⟩
    rw [← smul_sub, norm_smul, hz, Real.norm_eq_abs, abs_of_pos hlam]
  · intro hy
    refine ⟨lam⁻¹ • y, ?_, by rw [smul_smul, mul_inv_cancel₀ hlam.ne', one_smul]⟩
    have hs : lam⁻¹ • y - x = lam⁻¹ • (y - lam • x) := by
      rw [smul_sub, smul_smul, inv_mul_cancel₀ hlam.ne', one_smul]
    rw [hs, norm_smul, hy, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hlam)]
    field_simp

/-- The exposed set transforms covariantly under a dilation of the whole molecule. -/
theorem sasaSet_dilation (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N) {lam : ℝ}
    (hlam : 0 < lam) :
    sasaSet (fun k => lam • c k) (fun k => lam * rad k) i = lam • sasaSet c rad i := by
  have hinj : Function.Injective (fun x : Sp 3 => lam • x) := smul_right_injective _ hlam.ne'
  rw [sasaSet, sasaSet, ← Set.image_smul, Set.image_diff hinj, Set.image_iUnion₂]
  simp only [Set.image_smul, smul_sphere_of_pos hlam, _root_.smul_ball hlam.ne',
    Real.norm_eq_abs, abs_of_pos hlam]

/-- **Degree-two homogeneity.** Dilating the molecule by `lam > 0` multiplies every
accessible area by `lam²`. -/
theorem sasa_dilation (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N) {lam : ℝ}
    (hlam : 0 < lam) :
    sasa (fun k => lam • c k) (fun k => lam * rad k) i
      = ENNReal.ofReal (lam ^ 2) * sasa c rad i := by
  rw [sasa, sasaSet_dilation c rad i hlam,
    Measure.hausdorffMeasure_smul₀ (by norm_num : (0:ℝ) ≤ 2) hlam.ne', sasa]
  have h1 : ‖lam‖₊ ^ (2:ℝ) = ‖lam‖₊ ^ (2:ℕ) := by rw [← NNReal.rpow_natCast]; norm_num
  rw [h1, ENNReal.smul_def, smul_eq_mul]
  congr 1
  rw [ENNReal.coe_pow, ENNReal.ofReal_pow hlam.le]
  congr 1
  simp [ENNReal.ofReal, Real.nnnorm_of_nonneg hlam.le, Real.toNNReal_of_nonneg hlam.le]

/-- **Occlusion (union) bound.**  The area lost by atom `i` is at most the total area of the
pieces of its sphere covered by the individual neighbours: this is the rigorous form of the
pairwise-overlap approximation used in fast SASA algorithms. -/
theorem sphereArea_le_sasa_add_occlusions (c : Fin N → Sp 3) (rad : Fin N → ℝ) (i : Fin N)
    (hr : 0 ≤ rad i) :
    sphereArea (rad i) ≤
      sasa c rad i + ∑ j ∈ Finset.univ.erase i,
        μH[2] (sphere (c i) (rad i) ∩ ball (c j) (rad j)) := by
  have hsub : sphere (c i) (rad i) ⊆
      sasaSet c rad i ∪ ⋃ j ∈ Finset.univ.erase i,
        (sphere (c i) (rad i) ∩ ball (c j) (rad j)) := by
    intro x hx
    by_cases hmem : x ∈ ⋃ j ∈ ({i}ᶜ : Set (Fin N)), ball (c j) (rad j)
    · obtain ⟨j, hj, hxj⟩ := mem_iUnion₂.1 hmem
      have hj' : j ≠ i := by simpa using hj
      exact Or.inr (mem_iUnion₂.2 ⟨j, by simp [hj'], ⟨hx, hxj⟩⟩)
    · exact Or.inl ⟨hx, hmem⟩
  calc sphereArea (rad i) = μH[2] (sphere (c i) (rad i)) := (hausdorff_sphere_eq (c i) hr).symm
    _ ≤ μH[2] (sasaSet c rad i ∪ ⋃ j ∈ Finset.univ.erase i,
          (sphere (c i) (rad i) ∩ ball (c j) (rad j))) := measure_mono hsub
    _ ≤ μH[2] (sasaSet c rad i) + μH[2] (⋃ j ∈ Finset.univ.erase i,
          (sphere (c i) (rad i) ∩ ball (c j) (rad j))) := measure_union_le _ _
    _ ≤ sasa c rad i + ∑ j ∈ Finset.univ.erase i,
          μH[2] (sphere (c i) (rad i) ∩ ball (c j) (rad j)) :=
        add_le_add le_rfl (measure_biUnion_finset_le _ _)

/-- The total accessible area of a molecule is at most the sum of the isolated atomic
areas. -/
theorem total_sasa_le (c : Fin N → Sp 3) (rad : Fin N → ℝ) (hr : ∀ i, 0 ≤ rad i) :
    ∑ i, sasa c rad i ≤ ∑ i, sphereArea (rad i) :=
  Finset.sum_le_sum fun i _ => sasa_le_sphereArea c rad i (hr i)

end RequestProject.Physics
