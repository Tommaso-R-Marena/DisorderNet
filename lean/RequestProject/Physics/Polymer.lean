import RequestProject.Physics.GeneralizedBorn

/-!
# Part CXLII — Exact polymer statistical mechanics

Disordered regions are polymers, and their thermodynamics is fixed by a handful of exact
statements which we prove here from the underlying integrals, in real three-dimensional
space:

* the Gaussian (ideal-chain) propagator normalisation and its exact second moment, giving
  `⟨R²⟩ = N b²` (`ideal_chain_mean_square`),
* the entropic spring: free energy, Hooke's law, and the `1/N` stiffness scaling,
* the exact Lagrange identity for the radius of gyration, `R_g² = (1/2N²) ΣΣ |rᵢ - rⱼ|²`,
* the freely jointed chain: the exact force–extension law is the Langevin function
  `coth x - 1/x`, with its low-force Hookean limit `x/3` and its saturation at full
  extension.
-/

noncomputable section

namespace RequestProject.Physics

open Real MeasureTheory Filter Topology Set intervalIntegral

/-! ### Radial Gaussian moments -/

/-- The exact radial moment `∫₀^∞ yᵐ e^{-a y²} dy = a^{-(m+1)/2} Γ((m+1)/2) / 2`. -/
theorem gauss_radial_moment {a : ℝ} (ha : 0 < a) (m : ℕ) :
    ∫ y in Ioi (0 : ℝ), y ^ m * exp (-a * y ^ 2)
      = a ^ (-((m : ℝ) + 1) / 2) * (1 / 2) * Real.Gamma (((m : ℝ) + 1) / 2) := by
  rw [← integral_rpow_mul_exp_neg_mul_rpow (p := 2) (q := (m : ℝ)) (b := a) two_pos
    (by have := Nat.cast_nonneg (α := ℝ) m; linarith) ha]
  refine setIntegral_congr_fun measurableSet_Ioi (fun y hy => ?_)
  rw [Real.rpow_natCast, show ((2 : ℝ)) = ((2 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]

theorem gamma_three_half : Real.Gamma ((3 : ℝ) / 2) = Real.sqrt π / 2 := by
  rw [show (3 : ℝ) / 2 = 1 / 2 + 1 by norm_num, Real.Gamma_add_one (by norm_num),
    Real.Gamma_one_half_eq]
  ring

theorem gamma_five_half : Real.Gamma ((5 : ℝ) / 2) = 3 * Real.sqrt π / 4 := by
  rw [show (5 : ℝ) / 2 = 3 / 2 + 1 by norm_num, Real.Gamma_add_one (by norm_num),
    gamma_three_half]
  ring

private theorem pi_div_rpow {a : ℝ} (ha : 0 < a) :
    (π / a) ^ ((3 : ℝ) / 2) = π * Real.sqrt π * a ^ (-(3 / 2) : ℝ) := by
  rw [Real.div_rpow Real.pi_pos.le ha.le, Real.rpow_neg ha.le,
    show (3 : ℝ) / 2 = 1 + 1 / 2 by norm_num, Real.rpow_add Real.pi_pos, Real.rpow_one,
    Real.sqrt_eq_rpow]
  ring

/-! ### The Gaussian chain -/

/-- The three-dimensional Gaussian integral. -/
theorem gaussian_integral_three {a : ℝ} (ha : 0 < a) :
    ∫ x : Sp 3, exp (-a * ‖x‖ ^ 2) = (π / a) ^ ((3 : ℝ) / 2) := by
  have hdim : Module.finrank ℝ (Sp 3) = 3 := by simp
  have h := integral_fun_norm_addHaar (E := Sp 3) (F := ℝ) volume (fun y => exp (-a * y ^ 2))
  rw [hdim] at h
  have h1 := gauss_radial_moment ha 2
  norm_num [gamma_three_half] at h1
  rw [h, show ((3 : ℕ) - 1) = 2 from rfl]
  simp only [smul_eq_mul, neg_mul, nsmul_eq_mul]
  rw [h1, volume_unitBall_three, pi_div_rpow ha]
  ring

/-- The exact second moment of the three-dimensional Gaussian. -/
theorem gaussian_second_moment_three {a : ℝ} (ha : 0 < a) :
    ∫ x : Sp 3, ‖x‖ ^ 2 * exp (-a * ‖x‖ ^ 2) = (3 / (2 * a)) * (π / a) ^ ((3 : ℝ) / 2) := by
  have hdim : Module.finrank ℝ (Sp 3) = 3 := by simp
  have h := integral_fun_norm_addHaar (E := Sp 3) (F := ℝ) volume
    (fun y => y ^ 2 * exp (-a * y ^ 2))
  rw [hdim] at h
  have h1 := gauss_radial_moment ha 4
  norm_num [gamma_five_half] at h1
  rw [h, show ((3 : ℕ) - 1) = 2 from rfl]
  have hcong : ∫ y in Ioi (0 : ℝ), y ^ 2 • (y ^ 2 * exp (-a * y ^ 2))
      = ∫ y in Ioi (0 : ℝ), y ^ 4 * exp (-(a * y ^ 2)) := by
    refine setIntegral_congr_fun measurableSet_Ioi (fun y hy => ?_)
    simp only [smul_eq_mul, neg_mul]
    ring
  rw [hcong, h1, volume_unitBall_three, pi_div_rpow ha]
  have key2 : a ^ (-(5 / 2) : ℝ) = a⁻¹ * a ^ (-(3 / 2) : ℝ) := by
    rw [show (-(5 / 2) : ℝ) = -1 + -(3 / 2) by norm_num, Real.rpow_add ha, Real.rpow_neg_one]
  rw [key2]
  simp only [nsmul_eq_mul, smul_eq_mul]
  have ha' : a ≠ 0 := ha.ne'
  field_simp
  ring

/-- **Ideal chain statistics.**  For the Gaussian chain propagator with `a = 3/(2Nb²)`, the
mean square end-to-end distance is exactly `N b²`. -/
theorem ideal_chain_mean_square {Nseg b : ℝ} (hN : 0 < Nseg) (hb : 0 < b) :
    (∫ x : Sp 3, ‖x‖ ^ 2 * exp (-(3 / (2 * Nseg * b ^ 2)) * ‖x‖ ^ 2)) /
      (∫ x : Sp 3, exp (-(3 / (2 * Nseg * b ^ 2)) * ‖x‖ ^ 2)) = Nseg * b ^ 2 := by
  have ha : 0 < 3 / (2 * Nseg * b ^ 2) := by positivity
  rw [gaussian_integral_three ha, gaussian_second_moment_three ha]
  have hpos : (0 : ℝ) < (π / (3 / (2 * Nseg * b ^ 2))) ^ ((3 : ℝ) / 2) :=
    Real.rpow_pos_of_pos (by positivity) _
  rw [mul_div_assoc, div_self hpos.ne', mul_one]
  field_simp

/-! ### The entropic spring -/

/-- Free energy of an ideal chain stretched to end-to-end distance `R`. -/
def chainFreeEnergy (kT Nseg b R : ℝ) : ℝ := 3 * kT * R ^ 2 / (2 * Nseg * b ^ 2)

/-- **Hooke's law for the entropic spring**: the restoring force is linear in the
extension, with spring constant `3kT/(Nb²)`. -/
theorem chain_force_hooke {kT Nseg b : ℝ} (hN : 0 < Nseg) (hb : 0 < b) (R : ℝ) :
    HasDerivAt (chainFreeEnergy kT Nseg b) (3 * kT / (Nseg * b ^ 2) * R) R := by
  have hd : HasDerivAt (fun R : ℝ => 3 * kT / (2 * Nseg * b ^ 2) * R ^ 2)
      (3 * kT / (2 * Nseg * b ^ 2) * (2 * R ^ 1)) R := by
    simpa using ((hasDerivAt_pow 2 R).const_mul (3 * kT / (2 * Nseg * b ^ 2)))
  have hfun : chainFreeEnergy kT Nseg b = fun R : ℝ => 3 * kT / (2 * Nseg * b ^ 2) * R ^ 2 := by
    funext R; simp [chainFreeEnergy]; ring
  rw [hfun]
  convert hd using 1
  have hN' : Nseg ≠ 0 := hN.ne'
  have hb' : b ≠ 0 := hb.ne'
  field_simp

/-- **Longer chains are softer**: the entropic spring constant scales as `1/N`. -/
theorem chain_stiffness_antitone {kT b : ℝ} (hkT : 0 < kT) (hb : 0 < b)
    {N1 N2 : ℝ} (h1 : 0 < N1) (h12 : N1 < N2) :
    3 * kT / (N2 * b ^ 2) < 3 * kT / (N1 * b ^ 2) := by
  have hb2 : 0 < b ^ 2 := by positivity
  apply div_lt_div_of_pos_left (by linarith) (by positivity)
  exact (mul_lt_mul_of_pos_right h12 hb2)

/-! ### Radius of gyration -/

variable {n : ℕ}

/-- Centre of mass of a configuration. -/
def centreOfMass (x : Fin n → Sp 3) : Sp 3 := (n : ℝ)⁻¹ • ∑ i, x i

/-- Squared radius of gyration. -/
def gyrationSq (x : Fin n → Sp 3) : ℝ := (n : ℝ)⁻¹ * ∑ i, ‖x i - centreOfMass x‖ ^ 2

/-- **The Lagrange identity for the radius of gyration**: it equals half the mean square
interparticle distance. -/
theorem gyrationSq_eq_pairs (hn : 0 < n) (x : Fin n → Sp 3) :
    gyrationSq x = (1 / (2 * (n : ℝ) ^ 2)) * ∑ i, ∑ j, ‖x i - x j‖ ^ 2 := by
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  set S : Sp 3 := ∑ i, x i with hSdef
  have hexp : ∀ a b : Sp 3, ‖a - b‖ ^ 2 = ‖a‖ ^ 2 - 2 * (inner ℝ a b) + ‖b‖ ^ 2 := by
    intro a b
    rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq]
    rw [inner_sub_sub_self]
    rw [real_inner_comm b a]
    ring
  -- the single sum
  have hone : ∑ i, ‖x i - centreOfMass x‖ ^ 2
      = (∑ i, ‖x i‖ ^ 2) - (n : ℝ)⁻¹ * ‖S‖ ^ 2 := by
    have : ∀ i, ‖x i - centreOfMass x‖ ^ 2
        = ‖x i‖ ^ 2 - 2 * ((n : ℝ)⁻¹ * (inner ℝ (x i) S)) + (n : ℝ)⁻¹ ^ 2 * ‖S‖ ^ 2 := by
      intro i
      rw [hexp, centreOfMass, ← hSdef, real_inner_smul_right, norm_smul]
      simp [mul_pow, abs_of_nonneg (by positivity : (0:ℝ) ≤ (n:ℝ)⁻¹)]
    rw [Finset.sum_congr rfl (fun i _ => this i)]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib]
    rw [← Finset.mul_sum, ← Finset.mul_sum, ← sum_inner]
    rw [← hSdef]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      real_inner_self_eq_norm_sq]
    field_simp
    ring
  -- the double sum
  have htwo : ∑ i, ∑ j, ‖x i - x j‖ ^ 2 = 2 * (n : ℝ) * (∑ i, ‖x i‖ ^ 2) - 2 * ‖S‖ ^ 2 := by
    have hinner : ∀ i, ∑ j, ‖x i - x j‖ ^ 2
        = (n : ℝ) * ‖x i‖ ^ 2 - 2 * (inner ℝ (x i) S) + ∑ j, ‖x j‖ ^ 2 := by
      intro i
      rw [Finset.sum_congr rfl (fun j _ => hexp (x i) (x j))]
      rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← inner_sum]
      rw [← hSdef]
      simp
    rw [Finset.sum_congr rfl (fun i _ => hinner i)]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
      ← sum_inner, ← hSdef]
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
      real_inner_self_eq_norm_sq]
    ring
  rw [gyrationSq, hone, htwo]
  field_simp

/-! ### The freely jointed chain: the exact Langevin law -/

/-- Partition function of one freely jointed bond at reduced force `x`, after reduction to
the projection variable `u = cos θ`. -/
def fjcPartition (x : ℝ) : ℝ := ∫ u in (-1 : ℝ)..1, exp (x * u)

/-- Mean projection of one bond at reduced force `x`. -/
def fjcMean (x : ℝ) : ℝ := (∫ u in (-1 : ℝ)..1, u * exp (x * u)) / fjcPartition x

theorem fjcPartition_eq {x : ℝ} (hx : x ≠ 0) : fjcPartition x = 2 * sinh x / x := by
  have hd : ∀ u ∈ uIcc (-1 : ℝ) 1, HasDerivAt (fun u : ℝ => exp (x * u) / x) (exp (x * u)) u := by
    intro u _
    have h1 : HasDerivAt (fun u : ℝ => x * u) x u := by
      simpa using (hasDerivAt_id u).const_mul x
    have h2 : HasDerivAt (fun u : ℝ => exp (x * u)) (exp (x * u) * x) u :=
      (Real.hasDerivAt_exp (x * u)).comp u h1
    have := h2.div_const x
    simpa [mul_div_assoc, div_self hx] using this
  have hint : IntervalIntegrable (fun u : ℝ => exp (x * u)) volume (-1) 1 := by
    apply Continuous.intervalIntegrable
    fun_prop
  rw [fjcPartition, integral_eq_sub_of_hasDerivAt hd hint]
  rw [Real.sinh_eq]
  field_simp

theorem fjc_numerator_eq (x : ℝ) :
    ∫ u in (-1 : ℝ)..1, u * exp (x * u) = 2 * (x * cosh x - sinh x) / x ^ 2 := by
  rcases eq_or_ne x 0 with rfl | hx
  · simp
  have hd : ∀ u ∈ uIcc (-1 : ℝ) 1,
      HasDerivAt (fun u : ℝ => (x * u - 1) * exp (x * u) / x ^ 2) (u * exp (x * u)) u := by
    intro u _
    have hlin : HasDerivAt (fun u : ℝ => x * u) x u := by
      simpa using (hasDerivAt_id u).const_mul x
    have h1 : HasDerivAt (fun u : ℝ => x * u - 1) x u := by simpa using hlin.sub_const 1
    have h2 : HasDerivAt (fun u : ℝ => exp (x * u)) (exp (x * u) * x) u :=
      (Real.hasDerivAt_exp (x * u)).comp u hlin
    have h3 := (h1.mul h2).div_const (x ^ 2)
    convert h3 using 1
    have hx2 : x ^ 2 ≠ 0 := pow_ne_zero 2 hx
    field_simp
    ring
  have hint : IntervalIntegrable (fun u : ℝ => u * exp (x * u)) volume (-1) 1 := by
    apply Continuous.intervalIntegrable
    fun_prop
  rw [integral_eq_sub_of_hasDerivAt hd hint, Real.sinh_eq, Real.cosh_eq]
  have hx2 : x ^ 2 ≠ 0 := pow_ne_zero 2 hx
  field_simp
  ring

/-- **The force–extension law of the freely jointed chain is exactly the Langevin
function.** -/
theorem fjcMean_eq_langevin {x : ℝ} (hx : x ≠ 0) : fjcMean x = cosh x / sinh x - 1 / x := by
  have hs : sinh x ≠ 0 := by
    simpa [Real.sinh_eq_zero] using hx
  rw [fjcMean, fjc_numerator_eq, fjcPartition_eq hx]
  field_simp

/-- The extension is always below full saturation. -/
theorem fjcMean_lt_one {x : ℝ} (hx : 0 < x) : fjcMean x < 1 := by
  have hx0 : x ≠ 0 := hx.ne'
  rw [fjcMean_eq_langevin hx0]
  have hs : 0 < sinh x := Mathlib.Meta.Positivity.sinh_pos_of_pos hx
  have hex : 0 < exp x := Real.exp_pos x
  have h2 : exp (2 * x) = exp x * exp x := by rw [← Real.exp_add]; ring_nf
  have hkey : 2 * x < exp x * exp x - 1 := by
    have h := Real.add_one_lt_exp (x := 2 * x) (by positivity)
    rw [h2] at h; linarith
  have hcs : x * cosh x - sinh x < x * sinh x := by
    have hE : Real.exp (-x) = 1 / Real.exp x := by rw [Real.exp_neg]; ring
    rw [Real.cosh_eq, Real.sinh_eq, hE]
    have hne : Real.exp x ≠ 0 := hex.ne'
    field_simp
    nlinarith [hkey, hex]
  rw [div_sub_div _ _ hs.ne' hx0, div_lt_one (by positivity)]
  nlinarith [hcs]

/-- **Saturation**: at large force the chain approaches full extension. -/
theorem fjcMean_tendsto_one : Tendsto fjcMean atTop (𝓝 1) := by
  have hu : Tendsto (fun x : ℝ => exp (-(2 * x))) atTop (𝓝 0) := by
    have h1 : Tendsto (fun x : ℝ => 2 * x) atTop atTop :=
      Filter.Tendsto.const_mul_atTop (by norm_num) tendsto_id
    exact Real.tendsto_exp_atBot.comp (tendsto_neg_atBot_iff.mpr h1)
  have hcoth : Tendsto (fun x : ℝ => (1 + exp (-(2 * x))) / (1 - exp (-(2 * x)))) atTop
      (𝓝 1) := by
    have h : Tendsto (fun x : ℝ => (1 + exp (-(2 * x))) / (1 - exp (-(2 * x)))) atTop
        (𝓝 ((1 + 0) / (1 - 0))) :=
      Tendsto.div (by simpa using tendsto_const_nhds.add hu)
        (by simpa using tendsto_const_nhds.sub hu) (by norm_num)
    simpa using h
  have hinv : Tendsto (fun x : ℝ => 1 / x) atTop (𝓝 0) := by
    simpa [one_div] using tendsto_inv_atTop_zero
  have hlim := hcoth.sub hinv
  rw [sub_zero] at hlim
  refine hlim.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with x hx
  rw [fjcMean_eq_langevin hx.ne']
  congr 1
  have hex : (0 : ℝ) < exp x := Real.exp_pos x
  have hE1 : (1 : ℝ) < exp x := by have h := Real.add_one_lt_exp hx.ne'; linarith
  have hE : Real.exp (-x) = 1 / Real.exp x := by rw [Real.exp_neg]; ring
  have hE2 : Real.exp (-(2 * x)) = 1 / (Real.exp x * Real.exp x) := by
    rw [Real.exp_neg, ← Real.exp_add]; ring_nf
  rw [Real.cosh_eq, Real.sinh_eq, hE, hE2]
  have hne : Real.exp x * Real.exp x - 1 ≠ 0 := by nlinarith
  have hne0 : Real.exp x ≠ 0 := hex.ne'
  field_simp

/-- The limit `sinh x / x → 1` as `x → 0`. -/
theorem sinh_div_tendsto_one : Tendsto (fun x : ℝ => sinh x / x) (𝓝[≠] 0) (𝓝 1) := by
  have h := Real.hasDerivAt_sinh 0
  rw [hasDerivAt_iff_tendsto_slope] at h
  simp only [Real.cosh_zero] at h
  refine h.congr ?_
  intro x
  simp [slope_def_field, div_eq_inv_mul]

/-- **Low-force (Hookean) limit**: the Langevin function has slope `1/3` at the origin,
recovering ideal-chain elasticity. -/
theorem fjcMean_tendsto_hooke : Tendsto (fun x : ℝ => fjcMean x / x) (𝓝[≠] 0) (𝓝 (1 / 3)) := by
  have hf : ∀ᶠ x : ℝ in 𝓝[≠] 0,
      HasDerivAt (fun x : ℝ => x * cosh x - sinh x) (x * sinh x) x := by
    filter_upwards with x
    have h1 : HasDerivAt (fun x : ℝ => x * cosh x) (1 * cosh x + x * sinh x) x :=
      (hasDerivAt_id x).mul (Real.hasDerivAt_cosh x)
    simpa using h1.sub (Real.hasDerivAt_sinh x)
  have hg : ∀ᶠ x : ℝ in 𝓝[≠] 0, HasDerivAt (fun x : ℝ => x ^ 3) (3 * x ^ 2) x := by
    filter_upwards with x
    simpa using hasDerivAt_pow 3 x
  have hg' : ∀ᶠ x : ℝ in 𝓝[≠] 0, (3 : ℝ) * x ^ 2 ≠ 0 := by
    filter_upwards [self_mem_nhdsWithin] with x hx
    have hx0 : x ≠ 0 := hx
    exact mul_ne_zero three_ne_zero (pow_ne_zero 2 hx0)
  have hcont : Continuous (fun x : ℝ => x * cosh x - sinh x) :=
    (continuous_id.mul Real.continuous_cosh).sub Real.continuous_sinh
  have hfa : Tendsto (fun x : ℝ => x * cosh x - sinh x) (𝓝[≠] 0) (𝓝 0) := by
    have h := (hcont.tendsto 0).mono_left (nhdsWithin_le_nhds (s := ({0}ᶜ : Set ℝ)))
    simpa using h
  have hga : Tendsto (fun x : ℝ => x ^ 3) (𝓝[≠] 0) (𝓝 0) := by
    have h := ((continuous_pow 3).tendsto (0 : ℝ)).mono_left
      (nhdsWithin_le_nhds (s := ({0}ᶜ : Set ℝ)))
    simpa using h
  have hdiv : Tendsto (fun x : ℝ => (x * sinh x) / (3 * x ^ 2)) (𝓝[≠] 0) (𝓝 (1 / 3)) := by
    have hc : (fun x : ℝ => sinh x / x / 3) =ᶠ[𝓝[≠] 0]
        (fun x : ℝ => (x * sinh x) / (3 * x ^ 2)) := by
      filter_upwards [self_mem_nhdsWithin] with x hx
      have hx0 : x ≠ 0 := hx
      field_simp
    refine Tendsto.congr' hc ?_
    simpa using sinh_div_tendsto_one.div_const 3
  have hL := HasDerivAt.lhopital_zero_nhdsNE hf hg hg' hfa hga hdiv
  have hfinal : Tendsto (fun x : ℝ =>
      ((x * cosh x - sinh x) / x ^ 3) / (sinh x / x)) (𝓝[≠] 0) (𝓝 ((1 / 3) / 1)) :=
    hL.div sinh_div_tendsto_one one_ne_zero
  rw [div_one] at hfinal
  refine hfinal.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with x hx
  have hx0 : x ≠ 0 := hx
  have hs : sinh x ≠ 0 := by simpa [Real.sinh_eq_zero] using hx0
  rw [fjcMean_eq_langevin hx0]
  field_simp

end RequestProject.Physics
