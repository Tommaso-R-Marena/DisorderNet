import RequestProject.Physics.Laplacian
import RequestProject.Physics.ScreenedODE

/-!
# Part CXXXVIII — The screened Poisson (Debye–Hückel) equation in three dimensions

The electrostatics of a protein in an electrolyte is governed, in the linear-response
(Debye–Hückel) regime, by the screened Poisson equation `Δφ = κ² φ` away from the fixed
charges, `κ` the inverse Debye length.  Here we

* verify that the Yukawa kernel is an exact solution (`laplacian_yukawa`),
* solve the exterior Dirichlet problem around a spherical macro-ion exactly
  (`laplacian_debyeField`, `debyeField_boundary`) and prove that the bounded solution is
  unique in the radial class (`debye_exterior_unique`),
* record the physical corollaries: screening strictly lowers the potential
  (`yukawa_lt_coulomb`), it is monotone in the ionic strength (`yukawa_antitone_kappa`),
  the unscreened limit is Coulomb (`yukawa_tendsto_coulomb`) and the potential decays by a
  factor `e` per Debye length (`debye_length_decay`).
-/

noncomputable section

namespace RequestProject.Physics

open Real Filter Topology Set

/-- The Yukawa (screened Coulomb) kernel `e^{-κ r}/r`. -/
def yukawa (kappa : ℝ) (x : Sp 3) : ℝ := exp (-kappa * ‖x‖) / ‖x‖

/-- The radial Yukawa profile. -/
def yukawaProfile (kappa r : ℝ) : ℝ := exp (-kappa * r) / r

lemma hasDerivAt_yukawaProfile (kappa : ℝ) {r : ℝ} (hr : 0 < r) :
    HasDerivAt (yukawaProfile kappa)
      (-exp (-kappa * r) * (kappa * r + 1) / r ^ 2) r := by
  have h1 : HasDerivAt (fun s : ℝ => -kappa * s) (-kappa) r := by
    simpa using (hasDerivAt_id r).const_mul (-kappa)
  have hnum : HasDerivAt (fun s : ℝ => exp (-kappa * s)) (exp (-kappa * r) * -kappa) r := h1.exp
  have hden : HasDerivAt (fun s : ℝ => s) 1 r := hasDerivAt_id r
  have h : HasDerivAt (fun s : ℝ => exp (-kappa * s) / s)
      ((exp (-kappa * r) * -kappa * r - exp (-kappa * r) * 1) / r ^ 2) r := hnum.div hden hr.ne'
  have hrne : r ≠ 0 := hr.ne'
  convert h using 1
  field_simp
  ring

lemma hasDerivAt_yukawaProfile' (kappa : ℝ) {r : ℝ} (hr : 0 < r) :
    HasDerivAt (fun s => -exp (-kappa * s) * (kappa * s + 1) / s ^ 2)
      (exp (-kappa * r) * (kappa ^ 2 * r ^ 2 + 2 * kappa * r + 2) / r ^ 3) r := by
  have h1 : HasDerivAt (fun s : ℝ => -kappa * s) (-kappa) r := by
    simpa using (hasDerivAt_id r).const_mul (-kappa)
  have hexp : HasDerivAt (fun s : ℝ => exp (-kappa * s)) (exp (-kappa * r) * -kappa) r := h1.exp
  have hlin : HasDerivAt (fun s : ℝ => kappa * s + 1) kappa r := by
    simpa using ((hasDerivAt_id r).const_mul kappa).add_const 1
  have hnum : HasDerivAt (fun s : ℝ => -exp (-kappa * s) * (kappa * s + 1))
      (-(exp (-kappa * r) * -kappa) * (kappa * r + 1) + -exp (-kappa * r) * kappa) r :=
    (hexp.neg).mul hlin
  have hden : HasDerivAt (fun s : ℝ => s ^ 2) (2 * r) r := by
    simpa [mul_comm] using hasDerivAt_pow 2 r
  have hrne : r ≠ 0 := hr.ne'
  have h : HasDerivAt (fun s : ℝ => -exp (-kappa * s) * (kappa * s + 1) / s ^ 2)
      (((-(exp (-kappa * r) * -kappa) * (kappa * r + 1) + -exp (-kappa * r) * kappa) * r ^ 2
        - (-exp (-kappa * r) * (kappa * r + 1)) * (2 * r)) / (r ^ 2) ^ 2) r :=
    hnum.div hden (by positivity)
  convert h using 1
  field_simp
  ring

/-- **The Yukawa kernel solves the screened Poisson equation** `Δψ = κ²ψ` away from the
source. -/
theorem laplacian_yukawa (kappa : ℝ) {x : Sp 3} (hx : x ≠ 0) :
    laplacian (yukawa kappa) x = kappa ^ 2 * yukawa kappa x := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hfun : yukawa kappa = fun y : Sp 3 => yukawaProfile kappa ‖y‖ := rfl
  rw [hfun, laplacian_radial (f1 := fun s => -exp (-kappa * s) * (kappa * s + 1) / s ^ 2)
      (f2 := fun s => exp (-kappa * s) * (kappa ^ 2 * s ^ 2 + 2 * kappa * s + 2) / s ^ 3)
      (fun s hs => hasDerivAt_yukawaProfile kappa hs)
      (fun s hs => hasDerivAt_yukawaProfile' kappa hs) hx]
  have hrne : ‖x‖ ≠ 0 := hr.ne'
  simp only [yukawa, yukawaProfile]
  push_cast
  field_simp
  ring

/-- The Laplacian is linear under scalar multiples. -/
lemma laplacian_const_mul {n : ℕ} (c : ℝ) (u : Sp n → ℝ) (x : Sp n) :
    laplacian (fun y => c * u y) x = c * laplacian u x := by
  unfold laplacian pderiv2 dirDeriv2
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ => ?_
  have hg : (deriv fun t : ℝ => c * u (x + t • EuclideanSpace.single i (1 : ℝ)))
      = fun t : ℝ => c * deriv (fun s : ℝ => u (x + s • EuclideanSpace.single i (1 : ℝ))) t := by
    funext t
    exact deriv_const_mul_field _
  rw [hg, deriv_const_mul_field]

/-- The exterior Debye–Hückel field of a sphere of radius `a` held at potential `V`. -/
def debyeField (kappa a V : ℝ) (x : Sp 3) : ℝ :=
  V * a * exp (-kappa * (‖x‖ - a)) / ‖x‖

lemma debyeField_eq_smul_yukawa (kappa a V : ℝ) (x : Sp 3) :
    debyeField kappa a V x = (V * a * exp (kappa * a)) * yukawa kappa x := by
  unfold debyeField yukawa
  rw [show -kappa * (‖x‖ - a) = kappa * a + -kappa * ‖x‖ by ring, Real.exp_add]
  ring

/-- **Exact solution of the exterior Dirichlet problem**: `debyeField` solves `Δφ = κ²φ`
off the origin. -/
theorem laplacian_debyeField (kappa a V : ℝ) {x : Sp 3} (hx : x ≠ 0) :
    laplacian (debyeField kappa a V) x = kappa ^ 2 * debyeField kappa a V x := by
  have hfun : debyeField kappa a V
      = fun y : Sp 3 => (V * a * exp (kappa * a)) * yukawa kappa y := by
    funext y; exact debyeField_eq_smul_yukawa kappa a V y
  simp only [hfun]
  rw [laplacian_const_mul, laplacian_yukawa kappa hx]
  ring

/-- The exterior solution satisfies the boundary condition on the sphere of radius `a`. -/
theorem debyeField_boundary {kappa a V : ℝ} (ha : 0 < a) {x : Sp 3} (hx : ‖x‖ = a) :
    debyeField kappa a V x = V := by
  unfold debyeField
  rw [hx]
  simp [ha.ne']

/-- **Uniqueness of the physical exterior solution.**  In the spherically symmetric class,
a solution of the screened equation outside a sphere of radius `a` whose "reduced
potential" `r ↦ r φ(r)` is bounded is exactly the Debye–Hückel field. -/
theorem debye_exterior_unique {kappa a V : ℝ} (hk : 0 < kappa) (ha : 0 < a)
    {psi psip : ℝ → ℝ}
    (hsol : SolvesScreened kappa (fun r => r * psi r) (fun r => psi r + r * psip r) (Ici a))
    (hbdd : ∃ M : ℝ, ∀ r ∈ Ici a, |r * psi r| ≤ M)
    (hbc : psi a = V) :
    ∀ r ∈ Ici a, psi r = V * a * exp (-kappa * (r - a)) / r := by
  obtain ⟨M, hM⟩ := hbdd
  intro r hr
  have hdecay := screened_bounded_decays hk hsol hM r hr
  have hrpos : 0 < r := lt_of_lt_of_le ha hr
  have hkey : r * psi r = a * psi a * exp (-kappa * (r - a)) := hdecay
  rw [hbc] at hkey
  have hrne : r ≠ 0 := hrpos.ne'
  field_simp
  simp only [neg_mul] at hkey ⊢
  linear_combination hkey

/-! ### Physical corollaries -/

/-- Screening strictly lowers the Coulomb potential at every finite distance. -/
theorem yukawa_lt_coulomb {kappa : ℝ} (hk : 0 < kappa) {x : Sp 3} (hx : x ≠ 0) :
    yukawa kappa x < ‖x‖⁻¹ := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hexp : exp (-kappa * ‖x‖) < 1 := by
    rw [Real.exp_lt_one_iff]
    nlinarith
  have : exp (-kappa * ‖x‖) / ‖x‖ < 1 / ‖x‖ := by gcongr
  simpa [yukawa, one_div] using this

/-- The screened potential is decreasing in the inverse Debye length, i.e. in the ionic
strength of the buffer. -/
theorem yukawa_antitone_kappa {k1 k2 : ℝ} (h : k1 < k2) {x : Sp 3} (hx : x ≠ 0) :
    yukawa k2 x < yukawa k1 x := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hexp : exp (-k2 * ‖x‖) < exp (-k1 * ‖x‖) := by
    apply Real.exp_lt_exp.mpr
    nlinarith
  unfold yukawa
  gcongr

/-- In the salt-free limit the screened kernel becomes the Coulomb kernel. -/
theorem yukawa_tendsto_coulomb {x : Sp 3} (hx : x ≠ 0) :
    Tendsto (fun k : ℝ => yukawa k x) (𝓝 0) (𝓝 (‖x‖⁻¹)) := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hcont : Continuous fun k : ℝ => exp (-k * ‖x‖) / ‖x‖ := by fun_prop
  have h := hcont.tendsto 0
  simp only [neg_zero, zero_mul, Real.exp_zero] at h
  simpa [yukawa, one_div] using h

/-- Over one Debye length `1/κ` the reduced potential `r φ(r)` falls by exactly a factor
`e`. -/
theorem debye_length_decay {kappa a V : ℝ} (hk : 0 < kappa) {x y : Sp 3}
    (hx : 0 < ‖x‖) (hy : ‖y‖ = ‖x‖ + 1 / kappa) :
    ‖y‖ * debyeField kappa a V y = exp (-1) * (‖x‖ * debyeField kappa a V x) := by
  have hypos : 0 < ‖y‖ := by rw [hy]; positivity
  have hyne : ‖y‖ ≠ 0 := hypos.ne'
  have hxne : ‖x‖ ≠ 0 := hx.ne'
  have hkne : kappa ≠ 0 := hk.ne'
  unfold debyeField
  have e1 : ‖y‖ * (V * a * exp (-kappa * (‖y‖ - a)) / ‖y‖)
      = V * a * exp (-kappa * (‖y‖ - a)) := by field_simp
  have e2 : ‖x‖ * (V * a * exp (-kappa * (‖x‖ - a)) / ‖x‖)
      = V * a * exp (-kappa * (‖x‖ - a)) := by field_simp
  rw [e1, e2, hy]
  rw [show -kappa * (‖x‖ + 1 / kappa - a) = -1 + -kappa * (‖x‖ - a) by field_simp; ring,
    Real.exp_add]
  ring

end RequestProject.Physics
