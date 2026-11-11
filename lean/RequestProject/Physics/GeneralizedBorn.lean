import RequestProject.Physics.Laplacian

/-!
# Part CXXXIX — Continuum solvation: exact Born integrals and the Generalized Born model

Generalized Born (GB) solvation rests on two things that are usually only quoted: the exact
value of the Coulomb-field volume integral

  `(1/4π) ∫_{‖x‖ > a} ‖x‖⁻⁴ dx = 1/a`,

which is *the definition* of the Born radius in the Coulomb-field approximation, and the
exact electrostatic self-energy of a charged sphere in a dielectric continuum,

  `(ε/8π) ∫_{‖x‖ > a} |E|² dx = q²/(2 ε a)`.

Both are proved here as genuine Lebesgue integrals over three-dimensional space, using the
polar decomposition of Haar measure.  On top of them we build Still's GB interaction
function and prove its defining properties: it interpolates exactly between the Born
self-energy at zero separation and the Coulomb law at large separation, it is strictly
increasing in the separation, and it never falls below the true distance, so that GB
solvation never exceeds the vacuum Coulomb interaction in magnitude.
-/

noncomputable section

namespace RequestProject.Physics

open Real MeasureTheory Metric Filter Topology Set

/-- Volume of the unit ball of three-dimensional Euclidean space. -/
theorem volume_unitBall_three : volume.real (ball (0 : Sp 3) 1) = 4 * π / 3 := by
  have hg : Real.Gamma ((5 : ℝ) / 2) = 3 * Real.sqrt π / 4 := by
    have h1 : Real.Gamma ((5 : ℝ) / 2) = (3 / 2) * Real.Gamma (3 / 2) := by
      rw [show (5 : ℝ) / 2 = 3 / 2 + 1 by norm_num, Real.Gamma_add_one (by norm_num)]
    have h2 : Real.Gamma ((3 : ℝ) / 2) = (1 / 2) * Real.Gamma (1 / 2) := by
      rw [show (3 : ℝ) / 2 = 1 / 2 + 1 by norm_num, Real.Gamma_add_one (by norm_num)]
    rw [h1, h2, Real.Gamma_one_half_eq]
    ring
  rw [Measure.real, EuclideanSpace.volume_ball]
  have hpi : 0 < π := Real.pi_pos
  have hs : 0 < Real.sqrt π := Real.sqrt_pos.mpr hpi
  simp only [Fintype.card_fin]
  norm_num [hg]
  rw [ENNReal.toReal_ofReal (by positivity)]
  have h3 : Real.sqrt π ^ 3 = π * Real.sqrt π := by
    nlinarith [Real.sq_sqrt hpi.le]
  rw [h3]
  field_simp

/-- **The Coulomb-field (Born) integral.**  The volume integral of `‖x‖⁻⁴` over the exterior
of the ball of radius `a` is exactly `4π/a`. -/
theorem born_exterior_integral {a : ℝ} (ha : 0 < a) :
    ∫ x : Sp 3, (if a < ‖x‖ then ‖x‖ ^ (-4 : ℤ) else 0) = 4 * π / a := by
  have hdim : Module.finrank ℝ (Sp 3) = 3 := by simp
  have h := integral_fun_norm_addHaar (E := Sp 3) (F := ℝ) volume
    (fun y => if a < y then y ^ (-4 : ℤ) else 0)
  rw [hdim] at h
  rw [h]
  have hinner : ∫ y in Ioi (0 : ℝ), y ^ (3 - 1) • (if a < y then y ^ (-4 : ℤ) else 0) = 1 / a := by
    have hset : ∀ y ∈ Ioi (0 : ℝ), y ^ (3 - 1) • (if a < y then y ^ (-4 : ℤ) else 0)
        = Set.indicator (Ioi a) (fun y : ℝ => y ^ (-2 : ℝ)) y := by
      intro y hy
      by_cases hya : a < y
      · have hy0 : 0 < y := hy
        have hyne : y ≠ 0 := hy0.ne'
        have hrp : (y : ℝ) ^ (-2 : ℝ) = (y ^ (2 : ℕ))⁻¹ := by
          rw [show (-2 : ℝ) = -((2 : ℕ) : ℝ) by norm_num, Real.rpow_neg hy0.le, Real.rpow_natCast]
        simp [hya, Set.indicator_of_mem, zpow_neg, hrp]
        field_simp
      · simp [hya, Set.indicator_of_notMem]
    rw [setIntegral_congr_fun measurableSet_Ioi hset]
    rw [MeasureTheory.integral_indicator measurableSet_Ioi]
    rw [Measure.restrict_restrict measurableSet_Ioi]
    have hint : Ioi a ∩ Ioi (0 : ℝ) = Ioi a := by
      ext y; simp; intro h; linarith
    rw [hint, integral_Ioi_rpow_of_lt (by norm_num) ha]
    norm_num [Real.rpow_neg_one]
  rw [hinner, volume_unitBall_three]
  simp
  ring

/-- The Coulomb-field approximation to the inverse Born radius. -/
def cfaInvBornRadius (a : ℝ) : ℝ :=
  (1 / (4 * π)) * ∫ x : Sp 3, (if a < ‖x‖ then ‖x‖ ^ (-4 : ℤ) else 0)

/-- **For an isolated atom the Coulomb-field Born radius is exactly the van der Waals
radius.** -/
theorem cfaInvBornRadius_eq {a : ℝ} (ha : 0 < a) : cfaInvBornRadius a = 1 / a := by
  unfold cfaInvBornRadius
  rw [born_exterior_integral ha]
  have hpi : (0 : ℝ) < π := Real.pi_pos
  field_simp

/-- **Born self-energy.**  The field energy of a charge `q` on a sphere of radius `a`
immersed in a continuum of dielectric constant `ε` is `q²/(2εa)`. -/
theorem born_self_energy {a q eps : ℝ} (ha : 0 < a) (heps : 0 < eps) :
    (eps / (8 * π)) * ∫ x : Sp 3, (if a < ‖x‖ then (q / (eps * ‖x‖ ^ 2)) ^ 2 else 0)
      = q ^ 2 / (2 * eps * a) := by
  have hpi : (0 : ℝ) < π := Real.pi_pos
  have hpt : (fun x : Sp 3 => if a < ‖x‖ then (q / (eps * ‖x‖ ^ 2)) ^ 2 else 0)
      = fun x : Sp 3 => (q ^ 2 / eps ^ 2) * (if a < ‖x‖ then ‖x‖ ^ (-4 : ℤ) else 0) := by
    funext x
    by_cases hx : a < ‖x‖
    · have hxpos : 0 < ‖x‖ := lt_trans ha hx
      have hxne : ‖x‖ ≠ 0 := hxpos.ne'
      simp only [hx, if_true, zpow_neg]
      field_simp
    · simp [hx]
  rw [hpt, integral_const_mul, born_exterior_integral ha]
  field_simp
  ring

/-- **The Born solvation free energy**: the difference of the self-energies in the solvent
and in the solute interior. -/
theorem born_solvation_energy {a q epsIn epsOut : ℝ} (ha : 0 < a)
    (hin : 0 < epsIn) (hout : 0 < epsOut) :
    q ^ 2 / (2 * epsOut * a) - q ^ 2 / (2 * epsIn * a)
      = -(q ^ 2 / (2 * a)) * (1 / epsIn - 1 / epsOut) := by
  field_simp
  ring

/-! ### Still's Generalized Born interaction function -/

/-- Still's GB function `f_GB(r) = √(r² + a_i a_j e^{-r²/(4 a_i a_j)})`. -/
def gbF (ai aj r : ℝ) : ℝ := sqrt (r ^ 2 + ai * aj * exp (-r ^ 2 / (4 * ai * aj)))

/-- At zero separation the GB function is the geometric mean of the Born radii. -/
theorem gbF_zero {ai aj : ℝ} (hi : 0 < ai) (hj : 0 < aj) :
    gbF ai aj 0 = sqrt (ai * aj) := by
  unfold gbF
  norm_num

theorem gbF_self {a : ℝ} (ha : 0 < a) : gbF a a 0 = a := by
  rw [gbF_zero ha ha]
  rw [show a * a = a ^ 2 by ring, Real.sqrt_sq ha.le]

/-- The GB function never falls below the true interatomic distance: GB solvation screening
is never stronger than the bare Coulomb interaction. -/
theorem le_gbF {ai aj r : ℝ} (hi : 0 < ai) (hj : 0 < aj) (hr : 0 ≤ r) : r ≤ gbF ai aj r := by
  unfold gbF
  have hpos : 0 ≤ ai * aj * exp (-r ^ 2 / (4 * ai * aj)) := by positivity
  calc r = sqrt (r ^ 2) := (Real.sqrt_sq hr).symm
    _ ≤ sqrt (r ^ 2 + ai * aj * exp (-r ^ 2 / (4 * ai * aj))) := by
        apply Real.sqrt_le_sqrt; linarith

/-- The GB function is bounded above by the "no-overlap" value. -/
theorem gbF_le {ai aj r : ℝ} (hi : 0 < ai) (hj : 0 < aj) (hr : 0 ≤ r) :
    gbF ai aj r ≤ sqrt (r ^ 2 + ai * aj) := by
  unfold gbF
  apply Real.sqrt_le_sqrt
  have hexp : exp (-r ^ 2 / (4 * ai * aj)) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    have : 0 ≤ r ^ 2 := sq_nonneg r
    have h4 : 0 < 4 * ai * aj := by positivity
    apply div_nonpos_of_nonpos_of_nonneg <;> linarith
  nlinarith [mul_pos hi hj]

lemma gbF_pos {ai aj r : ℝ} (hi : 0 < ai) (hj : 0 < aj) : 0 < gbF ai aj r := by
  unfold gbF
  apply Real.sqrt_pos.mpr
  have : 0 < ai * aj * exp (-r ^ 2 / (4 * ai * aj)) := by positivity
  nlinarith [sq_nonneg r]

/-- One-sided Lipschitz estimate for the decaying exponential. -/
lemma exp_neg_sub_le {s t : ℝ} (hs : 0 ≤ s) (hst : s ≤ t) :
    exp (-s) - exp (-t) ≤ t - s := by
  have h1 : 1 - (t - s) ≤ exp (-(t - s)) := by
    have := Real.add_one_le_exp (-(t - s))
    linarith
  have hexp : exp (-t) = exp (-s) * exp (-(t - s)) := by
    rw [← Real.exp_add]; ring_nf
  have hs1 : exp (-s) ≤ 1 := by
    rw [Real.exp_le_one_iff]; linarith
  have hpos : 0 < exp (-s) := Real.exp_pos _
  rw [hexp]
  nlinarith

/-- The GB function is strictly increasing in the separation. -/
theorem gbF_strictMonoOn {ai aj : ℝ} (hi : 0 < ai) (hj : 0 < aj) :
    StrictMonoOn (gbF ai aj) (Ici 0) := by
  intro r1 hr1 r2 hr2 hlt
  have h1 : (0 : ℝ) ≤ r1 := hr1
  have h2 : (0 : ℝ) ≤ r2 := le_trans h1 hlt.le
  set A := ai * aj with hA
  have hApos : 0 < A := mul_pos hi hj
  have hkey : r1 ^ 2 + A * exp (-r1 ^ 2 / (4 * A)) < r2 ^ 2 + A * exp (-r2 ^ 2 / (4 * A)) := by
    have hsq : r1 ^ 2 < r2 ^ 2 := by nlinarith
    have hs : (0 : ℝ) ≤ r1 ^ 2 / (4 * A) := by positivity
    have hst : r1 ^ 2 / (4 * A) ≤ r2 ^ 2 / (4 * A) := by
      apply div_le_div_of_nonneg_right hsq.le
      positivity
    have hbound := exp_neg_sub_le hs hst
    have hrw1 : -r1 ^ 2 / (4 * A) = -(r1 ^ 2 / (4 * A)) := by ring
    have hrw2 : -r2 ^ 2 / (4 * A) = -(r2 ^ 2 / (4 * A)) := by ring
    rw [hrw1, hrw2]
    have hdiff : A * (exp (-(r1 ^ 2 / (4 * A))) - exp (-(r2 ^ 2 / (4 * A))))
        ≤ A * (r2 ^ 2 / (4 * A) - r1 ^ 2 / (4 * A)) := by
      apply mul_le_mul_of_nonneg_left hbound hApos.le
    have hsimp : A * (r2 ^ 2 / (4 * A) - r1 ^ 2 / (4 * A)) = (r2 ^ 2 - r1 ^ 2) / 4 := by
      field_simp
    rw [hsimp] at hdiff
    nlinarith
  have e1 : (4 : ℝ) * ai * aj = 4 * A := by rw [hA]; ring
  have hkey' : r1 ^ 2 + ai * aj * exp (-r1 ^ 2 / (4 * ai * aj))
      < r2 ^ 2 + ai * aj * exp (-r2 ^ 2 / (4 * ai * aj)) := by
    rw [e1, ← hA]
    exact hkey
  unfold gbF
  apply Real.sqrt_lt_sqrt
  · positivity
  · exact hkey'

/-- **GB reduces to Coulomb at large separation**: the GB kernel and the Coulomb kernel
agree in the limit. -/
theorem gbF_tendsto_coulomb {ai aj : ℝ} (hi : 0 < ai) (hj : 0 < aj) :
    Tendsto (fun r : ℝ => gbF ai aj r / r) atTop (𝓝 1) := by
  set A := ai * aj with hA
  have hApos : 0 < A := mul_pos hi hj
  have hlow : ∀ᶠ r : ℝ in atTop, (1 : ℝ) ≤ gbF ai aj r / r := by
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
    rw [le_div_iff₀ hr]
    simpa using le_gbF hi hj hr.le
  have hhigh : ∀ᶠ r : ℝ in atTop, gbF ai aj r / r ≤ sqrt (1 + A / r ^ 2) := by
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
    have hle := gbF_le hi hj hr.le
    rw [div_le_iff₀ hr]
    have hrw : sqrt (1 + A / r ^ 2) * r = sqrt ((1 + A / r ^ 2) * r ^ 2) := by
      rw [Real.sqrt_mul (by positivity), Real.sqrt_sq hr.le]
    rw [hrw]
    have hval : (1 + A / r ^ 2) * r ^ 2 = r ^ 2 + A := by
      field_simp
    rw [hval]
    exact hle
  have hsq : Tendsto (fun r : ℝ => sqrt (1 + A / r ^ 2)) atTop (𝓝 1) := by
    have h0 : Tendsto (fun r : ℝ => 1 + A / r ^ 2) atTop (𝓝 1) := by
      have : Tendsto (fun r : ℝ => A / r ^ 2) atTop (𝓝 0) := by
        apply Filter.Tendsto.const_div_atTop
        exact tendsto_pow_atTop (by norm_num)
      simpa using tendsto_const_nhds.add this
    have := (Real.continuous_sqrt.tendsto 1).comp h0
    simpa using this
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hsq hlow hhigh

/-- The Generalized Born solvation free energy of a set of charges. -/
def gbEnergy {N : ℕ} (epsIn epsOut : ℝ) (q a : Fin N → ℝ) (d : Fin N → Fin N → ℝ) : ℝ :=
  -(1 / 2) * (1 / epsIn - 1 / epsOut) *
    ∑ i, ∑ j, q i * q j / gbF (a i) (a j) (d i j)

/-- For a single atom the GB energy is exactly the Born solvation energy. -/
theorem gbEnergy_single {epsIn epsOut : ℝ} (q a : ℝ) (ha : 0 < a) :
    gbEnergy (N := 1) epsIn epsOut (fun _ => q) (fun _ => a) (fun _ _ => 0)
      = -(q ^ 2 / (2 * a)) * (1 / epsIn - 1 / epsOut) := by
  unfold gbEnergy
  rw [Finset.sum_const, Finset.sum_const]
  simp [gbF_self ha]
  ring

/-- Each GB cross term is bounded in magnitude by the corresponding Coulomb term: the
continuum solvation correction can never overshoot the vacuum interaction. -/
theorem gb_pair_le_coulomb {ai aj r qi qj : ℝ} (hi : 0 < ai) (hj : 0 < aj) (hr : 0 < r) :
    |qi * qj / gbF ai aj r| ≤ |qi * qj| / r := by
  have hgb : r ≤ gbF ai aj r := le_gbF hi hj hr.le
  have hgbpos : 0 < gbF ai aj r := gbF_pos hi hj
  rw [abs_div, abs_of_pos hgbpos]
  apply div_le_div_of_nonneg_left (abs_nonneg _) hr hgb

end RequestProject.Physics
