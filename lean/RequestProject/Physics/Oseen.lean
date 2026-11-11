import RequestProject.Physics.RadialMonomials

/-!
# Part CXLIV — Stokes flow and hydrodynamic interactions

Hydrodynamic coupling between the segments of a disordered chain is governed by the Stokes
equations `μ Δu = ∇p`, `∇·u = 0`.  Their fundamental solution — the Oseen tensor — is the
object that every Brownian/Stokesian dynamics scheme uses.  Here we verify it as an exact
solution of the partial differential equations (not as a modelling assumption):

* `oseen_divergence_free` — the flow is incompressible,
* `stokes_pressure_harmonic` — the pressure field is harmonic away from the point force,
* `oseen_solves_stokes` — the momentum equation `μ Δu = ∇p` holds componentwise,

together with the physical consequences: the Oseen tensor is symmetric (Lorentz
reciprocity), it decays only like `1/r`, and that decay is slow enough that the
hydrodynamic interaction is not integrable over space — the reason hydrodynamics cannot be
truncated the way short-ranged potentials can.
-/

noncomputable section

namespace RequestProject.Physics

open scoped RealInnerProductSpace
open Real MeasureTheory Metric Set

/-- The Oseen velocity field produced at `x` by a point force `F` at the origin in a fluid
of viscosity `mu`. -/
def oseenVel (mu : ℝ) (F : Sp 3) (x : Sp 3) : Sp 3 :=
  (1 / (8 * π * mu)) • ((‖x‖⁻¹) • F + (⟪F, x⟫ / ‖x‖ ^ 3) • x)

/-- The pressure field of a Stokeslet. -/
def stokesPressure (F : Sp 3) (x : Sp 3) : ℝ := ⟪F, x⟫ / (4 * π * ‖x‖ ^ 3)

/-! ### Coordinate expressions -/

lemma inner_eq_sum (F x : Sp 3) : ⟪F, x⟫ = ∑ j, F j * x j := by
  simp [PiLp.inner_apply, RCLike.inner_apply, mul_comm]

/-- The `a`-th component of the Oseen field, written as a combination of radial
monomials. -/
lemma oseenVel_apply_eq (mu : ℝ) (F : Sp 3) (a : Fin 3) :
    (fun y : Sp 3 => oseenVel mu F y a)
      = fun y : Sp 3 => (1 / (8 * π * mu)) *
          (F a * ‖y‖ ^ (-1 : ℤ) + ∑ j, F j * (y j * y a * ‖y‖ ^ (-3 : ℤ))) := by
  funext y
  have hsum : ∑ j, F j * (y j * y a * ‖y‖ ^ (-3 : ℤ))
      = (∑ j, F j * y j) * (y a * ‖y‖ ^ (-3 : ℤ)) := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun j _ => by ring
  simp only [oseenVel, PiLp.smul_apply, PiLp.add_apply, smul_eq_mul, hsum, inner_eq_sum]
  rw [zpow_neg, zpow_one, show ((3 : ℤ)) = ((3 : ℕ) : ℤ) from rfl, zpow_neg, zpow_natCast]
  field_simp

/-- The Stokeslet pressure, written as a combination of radial monomials. -/
lemma stokesPressure_eq (F : Sp 3) :
    stokesPressure F = fun y : Sp 3 => (1 / (4 * π)) * ∑ j, F j * (y j * ‖y‖ ^ (-3 : ℤ)) := by
  funext y
  have hsum : ∑ j, F j * (y j * ‖y‖ ^ (-3 : ℤ)) = (∑ j, F j * y j) * ‖y‖ ^ (-3 : ℤ) := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun j _ => by ring
  simp only [stokesPressure, hsum, inner_eq_sum]
  rw [show ((3 : ℤ)) = ((3 : ℕ) : ℤ) from rfl, zpow_neg, zpow_natCast]
  field_simp

/-! ### The Stokes equations -/

/-- The Stokeslet pressure is harmonic away from the point force. -/
theorem stokes_pressure_harmonic (F : Sp 3) {x : Sp 3} (hx : x ≠ 0) :
    laplacian (stokesPressure F) x = 0 := by
  have hlap : HasLaplacianAt (stokesPressure F) 0 x := by
    rw [stokesPressure_eq]
    have hterm : ∀ j ∈ (Finset.univ : Finset (Fin 3)),
        HasLaplacianAt (fun y : Sp 3 => F j * (y j * ‖y‖ ^ (-3 : ℤ))) 0 x := by
      intro j _
      have h := (hasLaplacianAt_mono1 (n := 3) j (-3) hx).const_mul (F j)
      convert h using 1
      push_cast
      ring
    have hs := HasLaplacianAt.sum hterm
    simpa using hs.const_mul (1 / (4 * π))
  simpa using hlap.laplacian_eq

/-- **The Oseen tensor solves the Stokes equations**: `μ Δu = ∇p` componentwise away from
the point force. -/
theorem oseen_solves_stokes {mu : ℝ} (hmu : 0 < mu) (F : Sp 3) {x : Sp 3} (hx : x ≠ 0)
    (a : Fin 3) :
    mu * laplacian (fun y => oseenVel mu F y a) x = pderiv a (stokesPressure F) x := by
  have hpi : (π : ℝ) ≠ 0 := Real.pi_pos.ne'
  have hmu' : mu ≠ 0 := hmu.ne'
  -- the Laplacian of the velocity component
  have hlap : HasLaplacianAt (fun y : Sp 3 => oseenVel mu F y a)
      ((1 / (8 * π * mu)) *
        (0 + ∑ j, F j * (2 * (if j = a then (1 : ℝ) else 0) * ‖x‖ ^ (-3 : ℤ)
          + (-6 : ℝ) * x j * x a * ‖x‖ ^ (-5 : ℤ)))) x := by
    rw [oseenVel_apply_eq]
    refine HasLaplacianAt.const_mul _ (HasLaplacianAt.add ?_ ?_)
    · have h := (hasLaplacianAt_norm_zpow (n := 3) (-1) hx).const_mul (F a)
      convert h using 1
      push_cast
      ring
    · refine HasLaplacianAt.sum ?_
      intro j _
      have h := (hasLaplacianAt_mono2 (n := 3) j a (-3) hx).const_mul (F j)
      convert h using 1
      push_cast
      ring
  rw [hlap.laplacian_eq]
  -- the pressure gradient
  have hpd : HasPDerivAt (stokesPressure F) a
      ((1 / (4 * π)) * ∑ j, F j * ((if j = a then (1 : ℝ) else 0) * ‖x‖ ^ (-3 : ℤ)
        + (-3 : ℝ) * x j * x a * ‖x‖ ^ (-5 : ℤ))) x := by
    rw [stokesPressure_eq]
    refine HasPDerivAt.const_mul _ (HasPDerivAt.sum ?_)
    intro j _
    have h := HasPDerivAt.const_mul (F j) (hasPDerivAt_mono1 (n := 3) j (-3) hx a)
    convert h using 1
    push_cast
    ring
  rw [hpd.pderiv_eq, zero_add]
  -- the two sums differ by a factor of two
  have hS : ∑ j, F j * (2 * (if j = a then (1 : ℝ) else 0) * ‖x‖ ^ (-3 : ℤ)
        + (-6 : ℝ) * x j * x a * ‖x‖ ^ (-5 : ℤ))
      = 2 * ∑ j, F j * ((if j = a then (1 : ℝ) else 0) * ‖x‖ ^ (-3 : ℤ)
        + (-3 : ℝ) * x j * x a * ‖x‖ ^ (-5 : ℤ)) := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  have hfin : ∀ T : ℝ, mu * (1 / (8 * π * mu) * (2 * T)) = 1 / (4 * π) * T := by
    intro T
    field_simp
    ring
  rw [hS, hfin]

/-! ### Incompressibility -/

/-- **Incompressibility**: the Oseen flow is divergence free away from the point force. -/
theorem oseen_divergence_free (mu : ℝ) (F : Sp 3) {x : Sp 3} (hx : x ≠ 0) :
    divergence (oseenVel mu F) x = 0 := by
  have hr : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hD : ∀ i : Fin 3, pderiv i (fun y => oseenVel mu F y i) x
      = (1 / (8 * π * mu)) * ((∑ j, F j * x j) * ‖x‖ ^ (-3 : ℤ)
          - 3 * (∑ j, F j * x j) * (x i) ^ 2 * ‖x‖ ^ (-5 : ℤ)) := by
    intro i
    have hpd : HasPDerivAt (fun y : Sp 3 => oseenVel mu F y i) i
        ((1 / (8 * π * mu)) * ((-1 : ℝ) * (F i * x i * ‖x‖ ^ (-3 : ℤ))
          + ∑ j, F j * ((if j = i then (1 : ℝ) else 0) * x i * ‖x‖ ^ (-3 : ℤ)
              + x j * ‖x‖ ^ (-3 : ℤ) + (-3 : ℝ) * x j * x i * x i * ‖x‖ ^ (-5 : ℤ)))) x := by
      rw [oseenVel_apply_eq]
      refine HasPDerivAt.const_mul _ (HasPDerivAt.add ?_ (HasPDerivAt.sum ?_))
      · have h := HasPDerivAt.const_mul (F i) (hasPDerivAt_mono0 (n := 3) (-1) hx i)
        convert h using 1
        push_cast
        ring
      · intro j _
        have h := HasPDerivAt.const_mul (F j) (hasPDerivAt_mono2 (n := 3) j i (-3) hx i)
        convert h using 1
        rw [if_pos (rfl : i = i)]
        push_cast
        ring
    rw [hpd.pderiv_eq]
    congr 1
    have hterm : ∀ j : Fin 3, F j * ((if j = i then (1 : ℝ) else 0) * x i * ‖x‖ ^ (-3 : ℤ)
          + x j * ‖x‖ ^ (-3 : ℤ) + (-3 : ℝ) * x j * x i * x i * ‖x‖ ^ (-5 : ℤ))
        = (if j = i then (1 : ℝ) else 0) * (F j * x i * ‖x‖ ^ (-3 : ℤ))
          + (F j * x j) * ‖x‖ ^ (-3 : ℤ)
          + (F j * x j) * ((-3 : ℝ) * x i * x i * ‖x‖ ^ (-5 : ℤ)) := fun j => by ring
    have e1 : ∑ j, (if j = i then (1 : ℝ) else 0) * (F j * x i * ‖x‖ ^ (-3 : ℤ))
        = F i * x i * ‖x‖ ^ (-3 : ℤ) := by simp
    have e2 : ∑ j, (F j * x j) * ‖x‖ ^ (-3 : ℤ) = (∑ j, F j * x j) * ‖x‖ ^ (-3 : ℤ) :=
      (Finset.sum_mul _ _ _).symm
    have e3 : ∑ j, (F j * x j) * ((-3 : ℝ) * x i * x i * ‖x‖ ^ (-5 : ℤ))
        = (∑ j, F j * x j) * ((-3 : ℝ) * x i * x i * ‖x‖ ^ (-5 : ℤ)) :=
      (Finset.sum_mul _ _ _).symm
    rw [Finset.sum_congr rfl (fun j _ => hterm j), Finset.sum_add_distrib, Finset.sum_add_distrib,
      e1, e2, e3]
    ring
  rw [divergence, Finset.sum_congr rfl (fun i _ => hD i), ← Finset.mul_sum]
  have hsum : ∑ i : Fin 3, ((∑ j, F j * x j) * ‖x‖ ^ (-3 : ℤ)
      - 3 * (∑ j, F j * x j) * (x i) ^ 2 * ‖x‖ ^ (-5 : ℤ)) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    have hc : ∀ i : Fin 3, 3 * (∑ j, F j * x j) * (x i) ^ 2 * ‖x‖ ^ (-5 : ℤ)
        = (3 * (∑ j, F j * x j) * ‖x‖ ^ (-5 : ℤ)) * (x i) ^ 2 := fun i => by ring
    rw [Finset.sum_congr rfl (fun i _ => hc i), ← Finset.mul_sum, sum_coord_sq]
    have hshift : ‖x‖ ^ 2 * ‖x‖ ^ (-5 : ℤ) = ‖x‖ ^ (-3 : ℤ) := by
      have h := norm_zpow_shift (x := x) hx (-1)
      norm_num at h
      exact h
    calc (3 : ℝ) * ((∑ j, F j * x j) * ‖x‖ ^ (-3 : ℤ))
          - 3 * (∑ j, F j * x j) * ‖x‖ ^ (-5 : ℤ) * ‖x‖ ^ 2
        = 3 * ((∑ j, F j * x j) * ‖x‖ ^ (-3 : ℤ))
          - 3 * (∑ j, F j * x j) * (‖x‖ ^ 2 * ‖x‖ ^ (-5 : ℤ)) := by ring
      _ = 0 := by rw [hshift]; ring
  rw [hsum, mul_zero]

/-! ### Physical consequences -/

/-- **Lorentz reciprocity**: the Oseen tensor is symmetric. -/
theorem oseen_symmetric (mu : ℝ) (F G : Sp 3) (x : Sp 3) :
    ⟪oseenVel mu F x, G⟫ = ⟪oseenVel mu G x, F⟫ := by
  simp only [oseenVel, inner_add_left, real_inner_smul_left]
  rw [real_inner_comm F G, real_inner_comm x G, real_inner_comm x F]
  ring

/-- **Long-ranged decay**: the Oseen flow decays like `1/r`. -/
theorem oseen_norm_le {mu : ℝ} (hmu : 0 < mu) (F : Sp 3) {x : Sp 3} (hx : x ≠ 0) :
    ‖oseenVel mu F x‖ ≤ ‖F‖ / (4 * π * mu * ‖x‖) := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hpi : (0 : ℝ) < π := Real.pi_pos
  have hc : (0 : ℝ) < 8 * π * mu := by positivity
  have hA : ‖(‖x‖⁻¹ : ℝ) • F‖ = ‖F‖ / ‖x‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hr), inv_mul_eq_div]
  have hB : ‖(⟪F, x⟫ / ‖x‖ ^ 3) • x‖ ≤ ‖F‖ / ‖x‖ := by
    rw [norm_smul, Real.norm_eq_abs, abs_div, abs_of_pos (by positivity : (0 : ℝ) < ‖x‖ ^ 3),
      div_mul_eq_mul_div]
    refine (div_le_div_iff₀ (by positivity) hr).mpr ?_
    have h := abs_real_inner_le_norm F x
    nlinarith [mul_le_mul_of_nonneg_right h (by positivity : (0 : ℝ) ≤ ‖x‖ * ‖x‖)]
  have hsum : ‖(‖x‖⁻¹ : ℝ) • F + (⟪F, x⟫ / ‖x‖ ^ 3) • x‖ ≤ 2 * (‖F‖ / ‖x‖) := by
    refine (norm_add_le _ _).trans ?_
    rw [hA]
    linarith
  rw [oseenVel, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : (0 : ℝ) < 1 / (8 * π * mu))]
  calc 1 / (8 * π * mu) * ‖(‖x‖⁻¹ : ℝ) • F + (⟪F, x⟫ / ‖x‖ ^ 3) • x‖
      ≤ 1 / (8 * π * mu) * (2 * (‖F‖ / ‖x‖)) := by
        exact mul_le_mul_of_nonneg_left hsum (by positivity)
    _ = ‖F‖ / (4 * π * mu * ‖x‖) := by field_simp; ring

/-- The hydrodynamic kernel `1/r` is *not* integrable over the exterior of any ball: unlike
short-ranged potentials, hydrodynamic interactions cannot be truncated without losing a
divergent amount of coupling. -/
theorem hydrodynamic_kernel_not_integrable {a : ℝ} (ha : 0 < a) :
    ¬ IntegrableOn (fun x : Sp 3 => ‖x‖⁻¹) {x : Sp 3 | a < ‖x‖} := by
  intro h
  have hms : MeasurableSet {x : Sp 3 | a < ‖x‖} :=
    measurableSet_lt measurable_const continuous_norm.measurable
  have hfun : (fun x : Sp 3 => if a < ‖x‖ then ‖x‖⁻¹ else 0)
      = Set.indicator {x : Sp 3 | a < ‖x‖} (fun x => ‖x‖⁻¹) := by
    funext x; simp [Set.indicator_apply]
  have hind : Integrable (fun x : Sp 3 => (fun y : ℝ => if a < y then y⁻¹ else 0) ‖x‖) := by
    simpa [hfun] using (integrable_indicator_iff hms).mpr h
  have hdim : Module.finrank ℝ (Sp 3) = 3 := by simp
  have h2 := (integrable_fun_norm_addHaar (μ := volume) (E := Sp 3)
    (f := fun y : ℝ => if a < y then y⁻¹ else 0)).mp hind
  rw [hdim] at h2
  have h3 : IntegrableOn (fun y : ℝ => y ^ (1 : ℝ)) (Ioi a) := by
    refine (h2.mono_set (Set.Ioi_subset_Ioi ha.le)).congr_fun ?_ measurableSet_Ioi
    intro y hy
    have hlt : a < y := hy
    have hy0 : (0 : ℝ) < y := ha.trans hlt
    simp only [smul_eq_mul, Real.rpow_one, show ((3 : ℕ) - 1) = 2 from rfl, if_pos hlt]
    field_simp
  rw [integrableOn_Ioi_rpow_iff ha] at h3
  norm_num at h3

end RequestProject.Physics
