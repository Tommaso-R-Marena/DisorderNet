/-
# Part XXIV.4  The solvent, without hand-waving: Generalized Born and the accessible surface

Water cannot be carried around as `10^5` explicit molecules, so it is integrated out.  This
file formalises the two standard terms that the integration leaves behind, and proves what
they do and do not do.

**Polar part: Generalized Born.**  `fGB Ri Rj r = sqrt (r^2 + Ri Rj exp (-r^2/(4 Ri Rj)))`
is the Still interpolation function and

  `gbPair = -(1/2) (1/eps_in - 1/eps_out) k q_i q_j / fGB`.

* `fGB_self` -- at zero separation with equal radii `fGB = R`, so the diagonal term is
  exactly the Born self-energy `-(1/2)(1/eps_in - 1/eps_out) k q^2 / R`.
* `fGB_ge_sqrt`, `fGB_ge_dist`, `fGB_sub_dist_le` -- `fGB` is bounded below by
  `sqrt (Ri Rj)` (so the polar term is *bounded*, never singular), is at least the actual
  separation, and approaches it at large `r` at rate `Ri Rj / r`: the Generalized Born
  interaction becomes the screened Coulomb interaction far away.
* `born_self_neg`, `born_self_strictAnti_radius` -- solvation of a charge is stabilising,
  and *burying* it (increasing the effective Born radius) strictly costs energy.  This is
  the desolvation penalty that keeps charges of a disordered region in contact with water.
* `gbEnergy_bddBelow` -- the whole polar term is bounded, so adding it preserves the
  stability theorem of the previous file.

**Nonpolar part: the solvent-accessible surface.**  `accessibleSet` is the honest geometric
object -- the part of the solvent-accessible sphere of an atom that no other atom's sphere
occludes -- and `sasa` is its two-dimensional Hausdorff measure.

* `accessibleSet_subset_sphere`, `sasa_lt_top` -- the accessible set is part of a sphere.
* `accessible_of_far` -- an atom further away than the sum of the two accessible radii
  occludes nothing.
* `sasa_eq_zero_of_engulfed` -- an atom inside a larger one has zero accessible area.
* `sasa_mono` -- occlusion can only decrease area, hence `nonpolar_le_of_occlusion`: burying
  hydrophobic surface strictly lowers the nonpolar energy.  This is the term that makes a
  disordered region expand in good solvent and contract when its hydrophobic surface is
  exposed.
* `sasa_rigid_invariant`, `Hsolv_rigid_invariant` -- both solvation terms, and hence the
  full solvated Hamiltonian, are invariant under rigid motions of space.
-/
import Mathlib
import RequestProject.Hamiltonian

namespace IDR

namespace Solvation

open Real MeasureTheory Potentials MM
open scoped ENNReal NNReal

/-! ## The Generalized Born interpolation function -/

/-- Still's Generalized Born interpolation function. -/
noncomputable def fGB (Ri Rj r : ℝ) : ℝ :=
  Real.sqrt (r ^ 2 + Ri * Rj * Real.exp (-(r ^ 2 / (4 * Ri * Rj))))

/-- **The Born self-energy limit.**  At zero separation with equal radii `fGB = R`. -/
theorem fGB_self {R : ℝ} (hR : 0 < R) : fGB R R 0 = R := by
  unfold fGB
  simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, zero_div, neg_zero,
    Real.exp_zero, mul_one, zero_add]
  rw [show R * R = R ^ 2 by ring, Real.sqrt_sq hR.le]

/-- **The polar term is never singular.**  `fGB` is bounded below by `sqrt (Ri Rj)`. -/
theorem fGB_ge_sqrt {Ri Rj r : ℝ} (hi : 0 < Ri) (hj : 0 < Rj) :
    Real.sqrt (Ri * Rj) ≤ fGB Ri Rj r := by
  unfold fGB
  apply Real.sqrt_le_sqrt
  set s : ℝ := r ^ 2 / (4 * Ri * Rj) with hs
  have hspos : 0 ≤ s := by rw [hs]; positivity
  have hexp : 1 - s ≤ Real.exp (-s) := by
    have := Real.add_one_le_exp (-s)
    linarith
  have hr2 : r ^ 2 = 4 * Ri * Rj * s := by
    rw [hs]; field_simp
  have hRR : 0 < Ri * Rj := mul_pos hi hj
  nlinarith [hexp, hspos, hRR]

theorem fGB_pos {Ri Rj r : ℝ} (hi : 0 < Ri) (hj : 0 < Rj) : 0 < fGB Ri Rj r :=
  lt_of_lt_of_le (Real.sqrt_pos.mpr (mul_pos hi hj)) (fGB_ge_sqrt hi hj)

/-- `fGB` is at least the true separation: the polar interaction is weaker than the bare
Coulomb interaction at the same distance. -/
theorem fGB_ge_dist {Ri Rj r : ℝ} (hi : 0 < Ri) (hj : 0 < Rj) (hr : 0 ≤ r) :
    r ≤ fGB Ri Rj r := by
  unfold fGB
  have hnn : 0 ≤ Ri * Rj * Real.exp (-(r ^ 2 / (4 * Ri * Rj))) := by
    have := Real.exp_pos (-(r ^ 2 / (4 * Ri * Rj)))
    positivity
  calc r = Real.sqrt (r ^ 2) := (Real.sqrt_sq hr).symm
    _ ≤ Real.sqrt (r ^ 2 + Ri * Rj * Real.exp (-(r ^ 2 / (4 * Ri * Rj)))) := by
        apply Real.sqrt_le_sqrt; linarith

/-- **At large separation the Generalized Born interaction is the Coulomb interaction.**
The gap closes at rate `Ri Rj / r`. -/
theorem fGB_sub_dist_le {Ri Rj r : ℝ} (hi : 0 < Ri) (hj : 0 < Rj) (hr : 0 < r) :
    fGB Ri Rj r - r ≤ Ri * Rj / r := by
  have hge := fGB_ge_dist hi hj hr.le
  have hpos := fGB_pos (r := r) hi hj
  have hsq : (fGB Ri Rj r) ^ 2 = r ^ 2 + Ri * Rj * Real.exp (-(r ^ 2 / (4 * Ri * Rj))) := by
    unfold fGB
    rw [Real.sq_sqrt]
    have := Real.exp_pos (-(r ^ 2 / (4 * Ri * Rj)))
    positivity
  have hexp : Real.exp (-(r ^ 2 / (4 * Ri * Rj))) ≤ 1 := by
    apply Real.exp_le_one_iff.mpr
    have : 0 ≤ r ^ 2 / (4 * Ri * Rj) := by positivity
    linarith
  have hdiff : (fGB Ri Rj r - r) * (fGB Ri Rj r + r) ≤ Ri * Rj := by
    have hexpand : (fGB Ri Rj r - r) * (fGB Ri Rj r + r) = (fGB Ri Rj r) ^ 2 - r ^ 2 := by
      ring
    rw [hexpand, hsq]
    nlinarith [mul_pos hi hj]
  have hsum : r ≤ fGB Ri Rj r + r := by linarith
  rw [le_div_iff₀ hr]
  nlinarith [hdiff, hge]

/-! ## The polar solvation energy -/

/-- The Generalized Born pair term (joules) between charges `q1, q2` with effective Born
radii `R1, R2` at separation `r`, between a solute of relative permittivity `epsIn` and a
solvent of relative permittivity `epsOut`. -/
noncomputable def gbPair (epsIn epsOut q1 q2 R1 R2 r : ℝ) : ℝ :=
  -(1/2) * (1/epsIn - 1/epsOut) * coulombConst * q1 * q2 / fGB R1 R2 r

/-- **Solvation stabilises a charge.**  The Born self-energy of a nonzero charge in a
solvent more polarisable than the solute is strictly negative. -/
theorem born_self_neg {epsIn epsOut q R : ℝ} (hin : 0 < epsIn) (hout : epsIn < epsOut)
    (hq : q ≠ 0) (hR : 0 < R) :
    gbPair epsIn epsOut q q R R 0 < 0 := by
  unfold gbPair
  rw [fGB_self hR]
  have hgap : 0 < 1/epsIn - 1/epsOut := by
    have h1 : 0 < epsOut := lt_trans hin hout
    have : 1/epsOut < 1/epsIn := one_div_lt_one_div_of_lt hin hout
    linarith
  have hq2 : 0 < q * q := by
    rcases lt_or_gt_of_ne hq with h | h
    · exact mul_pos_of_neg_of_neg h h
    · exact mul_pos h h
  have hck := coulombConst_pos
  have hnum : 0 < 1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q) := by positivity
  have hrw : -(1/2) * (1/epsIn - 1/epsOut) * coulombConst * q * q
      = -(1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q)) := by ring
  rw [hrw]
  exact div_neg_of_neg_of_pos (by linarith) hR

/-- **The desolvation penalty.**  Burying a charge -- increasing its effective Born radius --
strictly raises the solvation energy: a charged residue of a disordered region pays to leave
the water. -/
theorem born_self_strictAnti_radius {epsIn epsOut q R R' : ℝ} (hin : 0 < epsIn)
    (hout : epsIn < epsOut) (hq : q ≠ 0) (hR : 0 < R) (hRR : R < R') :
    gbPair epsIn epsOut q q R R 0 < gbPair epsIn epsOut q q R' R' 0 := by
  have hR' : 0 < R' := lt_trans hR hRR
  unfold gbPair
  rw [fGB_self hR, fGB_self hR']
  have hgap : 0 < 1/epsIn - 1/epsOut := by
    have h1 : 0 < epsOut := lt_trans hin hout
    have : 1/epsOut < 1/epsIn := one_div_lt_one_div_of_lt hin hout
    linarith
  have hq2 : 0 < q * q := by
    rcases lt_or_gt_of_ne hq with h | h
    · exact mul_pos_of_neg_of_neg h h
    · exact mul_pos h h
  have hck := coulombConst_pos
  have hA : 0 < 1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q) := by positivity
  have hrwR : -(1/2) * (1/epsIn - 1/epsOut) * coulombConst * q * q / R
      = -((1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q)) / R) := by ring
  have hrwR' : -(1/2) * (1/epsIn - 1/epsOut) * coulombConst * q * q / R'
      = -((1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q)) / R') := by ring
  rw [hrwR, hrwR']
  have : (1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q)) / R'
      < (1/2 * (1/epsIn - 1/epsOut) * coulombConst * (q * q)) / R :=
    div_lt_div_of_pos_left hA hR hRR
  linarith

/-- The Generalized Born energy of a configuration: the double sum over all ordered pairs,
the diagonal contributing the Born self-energies. -/
noncomputable def gbEnergy {N : ℕ} (epsIn epsOut : ℝ) (q R : Fin N → ℝ) (x : Conf N) : ℝ :=
  ∑ i : Fin N, ∑ j : Fin N,
    gbPair epsIn epsOut (q i) (q j) (R i) (R j) (dist (x i) (x j))

/-- **The polar term is bounded**, hence adding it to a stable Hamiltonian leaves it
stable. -/
theorem gbEnergy_bddBelow {N : ℕ} {epsIn epsOut : ℝ} {q R : Fin N → ℝ}
    (hR : ∀ i, 0 < R i) :
    ∃ B : ℝ, ∀ x : Conf N, -B ≤ gbEnergy epsIn epsOut q R x := by
  classical
  refine ⟨∑ i : Fin N, ∑ j : Fin N,
    |(1/2) * (1/epsIn - 1/epsOut) * coulombConst * q i * q j| /
      Real.sqrt (R i * R j), fun x => ?_⟩
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_le_sum fun i _ => ?_
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_le_sum fun j _ => ?_
  set c : ℝ := (1/2) * (1/epsIn - 1/epsOut) * coulombConst * q i * q j with hc
  have hfpos : 0 < fGB (R i) (R j) (dist (x i) (x j)) := fGB_pos (hR i) (hR j)
  have hfge : Real.sqrt (R i * R j) ≤ fGB (R i) (R j) (dist (x i) (x j)) :=
    fGB_ge_sqrt (hR i) (hR j)
  have hspos : 0 < Real.sqrt (R i * R j) := Real.sqrt_pos.mpr (mul_pos (hR i) (hR j))
  have hval : gbPair epsIn epsOut (q i) (q j) (R i) (R j) (dist (x i) (x j))
      = -c / fGB (R i) (R j) (dist (x i) (x j)) := by
    unfold gbPair
    rw [hc]; ring_nf
  rw [hval]
  have habs : |c| / fGB (R i) (R j) (dist (x i) (x j)) ≤ |c| / Real.sqrt (R i * R j) :=
    div_le_div_of_nonneg_left (abs_nonneg c) hspos hfge
  have hle : -(|c| / fGB (R i) (R j) (dist (x i) (x j)))
      ≤ -c / fGB (R i) (R j) (dist (x i) (x j)) := by
    rw [neg_div, neg_le_neg_iff, div_le_div_iff_of_pos_right hfpos]
    exact le_abs_self c
  linarith

/-! ## The nonpolar term: the solvent-accessible surface -/

variable {N : ℕ}

/-- The solvent-accessible surface of atom `i`: the points at exactly the accessible radius
`rad i + probe` from atom `i` that lie strictly outside every other atom's accessible
sphere. -/
def accessibleSet (rad : Fin N → ℝ) (probe : ℝ) (x : Conf N) (i : Fin N) : Set Point :=
  {p | dist p (x i) = rad i + probe ∧ ∀ j, j ≠ i → rad j + probe < dist p (x j)}

/-- The solvent-accessible surface area of atom `i`: the 2-dimensional Hausdorff measure of
its accessible set. -/
noncomputable def sasa (rad : Fin N → ℝ) (probe : ℝ) (x : Conf N) (i : Fin N) : ℝ≥0∞ :=
  μH[2] (accessibleSet rad probe x i)

theorem accessibleSet_subset_sphere (rad : Fin N → ℝ) (probe : ℝ) (x : Conf N) (i : Fin N) :
    accessibleSet rad probe x i ⊆ Metric.sphere (x i) (rad i + probe) := by
  rintro p ⟨hp, -⟩
  simpa [Metric.mem_sphere] using hp

/-- **A distant atom occludes nothing.**  If atom `j` is further from `i` than the sum of
their accessible radii, every point of `i`'s sphere remains accessible as far as `j` is
concerned. -/
theorem accessible_of_far {rad : Fin N → ℝ} {probe : ℝ} {x : Conf N} {i j : Fin N}
    {p : Point} (hp : dist p (x i) = rad i + probe)
    (hfar : rad i + rad j + 2 * probe < dist (x i) (x j)) :
    rad j + probe < dist p (x j) := by
  have htri : dist (x i) (x j) ≤ dist (x i) p + dist p (x j) := dist_triangle _ _ _
  have hpi : dist (x i) p = rad i + probe := by rw [dist_comm]; exact hp
  linarith

/-- **Complete burial.**  An atom entirely inside another one's accessible sphere has zero
accessible surface. -/
theorem sasa_eq_zero_of_engulfed {rad : Fin N → ℝ} {probe : ℝ} {x : Conf N} {i j : Fin N}
    (hij : j ≠ i) (hin : dist (x i) (x j) + (rad i + probe) < rad j + probe) :
    sasa rad probe x i = 0 := by
  have hempty : accessibleSet rad probe x i = ∅ := by
    ext p
    simp only [Set.mem_empty_iff_false, iff_false]
    rintro ⟨hp, hocc⟩
    have hlt := hocc j hij
    have htri : dist p (x j) ≤ dist p (x i) + dist (x i) (x j) := dist_triangle _ _ _
    rw [hp] at htri
    linarith
  rw [sasa, hempty]
  simp

/-- Occlusion can only remove accessible surface. -/
theorem sasa_mono {rad : Fin N → ℝ} {probe : ℝ} {x y : Conf N} {i : Fin N}
    (h : accessibleSet rad probe x i ⊆ accessibleSet rad probe y i) :
    sasa rad probe x i ≤ sasa rad probe y i :=
  measure_mono h

/-- The nonpolar (hydrophobic) solvation energy: a surface tension per unit exposed area. -/
noncomputable def nonpolarEnergy (gamma rad : Fin N → ℝ) (probe : ℝ) (x : Conf N) : ℝ :=
  ∑ i : Fin N, gamma i * (sasa rad probe x i).toReal

/-- Exposing hydrophobic surface costs energy: with positive surface tensions the nonpolar
term is nonnegative, and zero exactly when every atom is buried. -/
theorem nonpolarEnergy_nonneg {gamma rad : Fin N → ℝ} {probe : ℝ} {x : Conf N}
    (hg : ∀ i, 0 ≤ gamma i) : 0 ≤ nonpolarEnergy gamma rad probe x :=
  Finset.sum_nonneg fun i _ => mul_nonneg (hg i) ENNReal.toReal_nonneg

/-- **Burial lowers the nonpolar energy.**  If every atom's accessible set shrinks, the
hydrophobic term falls.  This is the force that collapses a hydrophobic disordered region
in water and expands it in a good solvent. -/
theorem nonpolar_le_of_occlusion {gamma rad : Fin N → ℝ} {probe : ℝ} {x y : Conf N}
    (hg : ∀ i, 0 ≤ gamma i) (hfin : ∀ i, sasa rad probe y i ≠ ⊤)
    (h : ∀ i, accessibleSet rad probe x i ⊆ accessibleSet rad probe y i) :
    nonpolarEnergy gamma rad probe x ≤ nonpolarEnergy gamma rad probe y := by
  refine Finset.sum_le_sum fun i _ => ?_
  refine mul_le_mul_of_nonneg_left ?_ (hg i)
  exact ENNReal.toReal_mono (hfin i) (sasa_mono (h i))

/-! ## Rigid-motion invariance of the solvation terms -/

theorem gbEnergy_rigid_invariant {epsIn epsOut : ℝ} {q R : Fin N → ℝ} (g : RigidMotion)
    (x : Conf N) :
    gbEnergy epsIn epsOut q R (g.act x) = gbEnergy epsIn epsOut q R x := by
  unfold gbEnergy
  exact Finset.sum_congr rfl fun i _ =>
    Finset.sum_congr rfl fun j _ => by rw [g.dist_act x i j]

/-- The accessible set of a moved configuration is the moved accessible set. -/
theorem accessibleSet_act {rad : Fin N → ℝ} {probe : ℝ} (g : RigidMotion) (x : Conf N)
    (i : Fin N) :
    accessibleSet rad probe (g.act x) i = g.map '' (accessibleSet rad probe x i) := by
  ext p
  constructor
  · rintro ⟨hp, hocc⟩
    obtain ⟨p0, rfl⟩ := g.surjective_map p
    refine ⟨p0, ⟨?_, ?_⟩, rfl⟩
    · rw [g.act_apply x i, g.isometry_map.dist_eq] at hp
      exact hp
    · intro j hj
      have := hocc j hj
      rw [g.act_apply x j, g.isometry_map.dist_eq] at this
      exact this
  · rintro ⟨p0, ⟨hp, hocc⟩, rfl⟩
    refine ⟨?_, ?_⟩
    · rw [g.act_apply x i, g.isometry_map.dist_eq]
      exact hp
    · intro j hj
      rw [g.act_apply x j, g.isometry_map.dist_eq]
      exact hocc j hj

theorem sasa_rigid_invariant {rad : Fin N → ℝ} {probe : ℝ} (g : RigidMotion) (x : Conf N)
    (i : Fin N) : sasa rad probe (g.act x) i = sasa rad probe x i := by
  unfold sasa
  rw [accessibleSet_act]
  exact g.isometry_map.hausdorffMeasure_image (Or.inl (by norm_num)) _

theorem nonpolarEnergy_rigid_invariant {gamma rad : Fin N → ℝ} {probe : ℝ} (g : RigidMotion)
    (x : Conf N) :
    nonpolarEnergy gamma rad probe (g.act x) = nonpolarEnergy gamma rad probe x := by
  unfold nonpolarEnergy
  exact Finset.sum_congr rfl fun i _ => by rw [sasa_rigid_invariant]

/-! ## The full solvated Hamiltonian -/

/-- The complete potential energy of a solvated conformation: the molecular-mechanics
Hamiltonian plus the polar (Generalized Born) and nonpolar (surface-area) solvation
terms. -/
noncomputable def Hsolv (F : ForceField N) (epsIn epsOut : ℝ) (q R gamma rad : Fin N → ℝ)
    (probe : ℝ) (x : Conf N) : ℝ :=
  F.H x + gbEnergy epsIn epsOut q R x + nonpolarEnergy gamma rad probe x

/-- **The solvated Hamiltonian is rigid-motion invariant.** -/
theorem Hsolv_rigid_invariant (F : ForceField N) {epsIn epsOut : ℝ}
    {q R gamma rad : Fin N → ℝ} {probe : ℝ} (g : RigidMotion) (x : Conf N) :
    Hsolv F epsIn epsOut q R gamma rad probe (g.act x)
      = Hsolv F epsIn epsOut q R gamma rad probe x := by
  unfold Hsolv
  rw [F.H_rigid_invariant, gbEnergy_rigid_invariant, nonpolarEnergy_rigid_invariant]

/-- **The solvated Hamiltonian is still bounded below.**  Solvation does not destroy
stability: the polar term is bounded and the nonpolar term is nonnegative. -/
theorem Hsolv_bddBelow (F : ForceField N) {epsIn epsOut : ℝ} {q R gamma rad : Fin N → ℝ}
    {probe : ℝ} (hR : ∀ i, 0 < R i) (hg : ∀ i, 0 ≤ gamma i) :
    ∃ B : ℝ, ∀ x : Conf N, (∀ i j : Fin N, i ≠ j → x i ≠ x j) →
      -B ≤ Hsolv F epsIn epsOut q R gamma rad probe x := by
  obtain ⟨B1, hB1⟩ := F.H_bddBelow
  obtain ⟨B2, hB2⟩ := gbEnergy_bddBelow (epsIn := epsIn) (epsOut := epsOut) (q := q) hR
  refine ⟨B1 + B2, fun x hx => ?_⟩
  have h1 := hB1 x hx
  have h2 := hB2 x
  have h3 := nonpolarEnergy_nonneg (gamma := gamma) (rad := rad) (probe := probe) (x := x) hg
  unfold Hsolv
  linarith

end Solvation

end IDR
