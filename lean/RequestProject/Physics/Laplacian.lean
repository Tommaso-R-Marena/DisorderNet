import Mathlib

/-!
# Part CXXXV — Differential operators on continuum space

This file builds, from scratch, the differential-operator calculus that the continuum
parts of protein physics need: partial derivatives, the gradient, the divergence and the
Laplacian on `EuclideanSpace ℝ (Fin n)`, together with the *radial reduction* theorem

  `Δ (f ∘ ‖·‖) (x) = f'' r + (n-1) f' r / r`,  `r = ‖x‖ > 0`,

which is the engine behind every spherically symmetric field equation used later
(screened Poisson / Debye–Hückel, Coulomb, Born, Oseen).

Everything is stated for genuine `n`-dimensional Euclidean space; no discretisation and no
finite-dimensional surrogate is used.
-/

noncomputable section

namespace RequestProject.Physics

open scoped RealInnerProductSpace
open Real Filter Topology

/-- Physical space: `n`-dimensional Euclidean space. -/
abbrev Sp (n : ℕ) := EuclideanSpace ℝ (Fin n)

variable {n : ℕ}

/-- Directional derivative of a scalar field along `v`. -/
def dirDeriv (u : Sp n → ℝ) (v : Sp n) (x : Sp n) : ℝ :=
  deriv (fun t : ℝ => u (x + t • v)) 0

/-- Second directional derivative of a scalar field along `v`. -/
def dirDeriv2 (u : Sp n → ℝ) (v : Sp n) (x : Sp n) : ℝ :=
  deriv (deriv fun t : ℝ => u (x + t • v)) 0

/-- The `i`-th partial derivative. -/
def pderiv (i : Fin n) (u : Sp n → ℝ) (x : Sp n) : ℝ :=
  dirDeriv u (EuclideanSpace.single i (1 : ℝ)) x

/-- The `i`-th second partial derivative. -/
def pderiv2 (i : Fin n) (u : Sp n → ℝ) (x : Sp n) : ℝ :=
  dirDeriv2 u (EuclideanSpace.single i (1 : ℝ)) x

/-- The Laplacian `Δu = ∑ ∂²ᵢ u`. -/
def laplacian (u : Sp n → ℝ) (x : Sp n) : ℝ := ∑ i, pderiv2 i u x

/-- The gradient of a scalar field, as a vector of physical space. -/
def grad (u : Sp n → ℝ) (x : Sp n) : Sp n :=
  (WithLp.equiv 2 (Fin n → ℝ)).symm fun i => pderiv i u x

/-- The divergence of a vector field. -/
def divergence (V : Sp n → Sp n) (x : Sp n) : ℝ := ∑ i, pderiv i (fun y => V y i) x

@[simp] lemma grad_apply (u : Sp n → ℝ) (x : Sp n) (i : Fin n) :
    grad u x i = pderiv i u x := rfl

/-! ### The line through `x` in direction `v` -/

/-- The squared norm along a line is a quadratic polynomial. -/
lemma norm_sq_line (x v : Sp n) (t : ℝ) :
    ‖x + t • v‖ ^ 2 = ‖x‖ ^ 2 + 2 * t * ⟪x, v⟫ + t ^ 2 * ‖v‖ ^ 2 := by
  rw [norm_add_sq_real]
  simp [real_inner_smul_right, norm_smul, mul_pow, sq_abs]
  ring

lemma norm_line_eq_sqrt {x v : Sp n} (hv : ‖v‖ = 1) (t : ℝ) :
    ‖x + t • v‖ = Real.sqrt (‖x‖ ^ 2 + 2 * t * ⟪x, v⟫ + t ^ 2) := by
  have h := norm_sq_line x v t
  rw [hv] at h
  rw [← Real.sqrt_sq (norm_nonneg (x + t • v)), h]
  ring_nf

/-- Along a line, the norm stays positive as long as `|t| < ‖x‖`. -/
lemma norm_line_pos {x v : Sp n} (hv : ‖v‖ = 1) {t : ℝ} (ht : |t| < ‖x‖) :
    0 < ‖x + t • v‖ := by
  have h1 : ‖t • v‖ = |t| := by simp [norm_smul, hv]
  have hle : ‖x‖ - ‖t • v‖ ≤ ‖x + t • v‖ := by
    have := norm_sub_norm_le x (-(t • v))
    simpa [sub_neg_eq_add] using this
  have h2 : 0 < ‖x‖ - ‖t • v‖ := by rw [h1]; linarith
  linarith

/-- Derivative of `t ↦ ‖x + t v‖` for a unit vector `v`, away from the origin. -/
lemma hasDerivAt_norm_line {x v : Sp n} (hv : ‖v‖ = 1) {t : ℝ} (ht : x + t • v ≠ 0) :
    HasDerivAt (fun s : ℝ => ‖x + s • v‖) ((⟪x, v⟫ + t) / ‖x + t • v‖) t := by
  set q : ℝ → ℝ := fun s => ‖x‖ ^ 2 + 2 * s * ⟪x, v⟫ + s ^ 2 with hq
  have hqd : HasDerivAt q (2 * ⟪x, v⟫ + 2 * t) t := by
    have h1 : HasDerivAt (fun s : ℝ => 2 * s * ⟪x, v⟫) (2 * ⟪x, v⟫) t := by
      simpa using ((hasDerivAt_id t).const_mul (2 : ℝ)).mul_const (⟪x, v⟫)
    have h2 : HasDerivAt (fun s : ℝ => s ^ 2) (2 * t) t := by
      simpa [mul_comm] using (hasDerivAt_pow 2 t)
    simpa [hq] using ((hasDerivAt_const t (‖x‖ ^ 2)).add h1).add h2
  have hnormeq : ∀ s : ℝ, Real.sqrt (q s) = ‖x + s • v‖ := by
    intro s; rw [norm_line_eq_sqrt hv s]
  have hqt : q t ≠ 0 := by
    have hpos : 0 < ‖x + t • v‖ := norm_pos_iff.mpr ht
    have : Real.sqrt (q t) ≠ 0 := by rw [hnormeq t]; exact hpos.ne'
    intro h; rw [h] at this; simp at this
  have hs : HasDerivAt (fun s : ℝ => Real.sqrt (q s))
      ((2 * ⟪x, v⟫ + 2 * t) / (2 * Real.sqrt (q t))) t := hqd.sqrt hqt
  have hmain : HasDerivAt (fun s : ℝ => ‖x + s • v‖)
      ((2 * ⟪x, v⟫ + 2 * t) / (2 * ‖x + t • v‖)) t := by
    simpa [hnormeq] using hs
  have hpos : 0 < ‖x + t • v‖ := norm_pos_iff.mpr ht
  convert hmain using 1
  field_simp

/-- First derivative of a radial field along a line. -/
lemma hasDerivAt_radial_line {f f1 : ℝ → ℝ} (h1 : ∀ s, 0 < s → HasDerivAt f (f1 s) s)
    {x v : Sp n} (hv : ‖v‖ = 1) {t : ℝ} (ht : x + t • v ≠ 0) :
    HasDerivAt (fun s : ℝ => f ‖x + s • v‖)
      (f1 ‖x + t • v‖ * ((⟪x, v⟫ + t) / ‖x + t • v‖)) t := by
  have hpos : 0 < ‖x + t • v‖ := norm_pos_iff.mpr ht
  exact (h1 _ hpos).comp t (hasDerivAt_norm_line hv ht)

/-- The second directional derivative of a radial field, at a point off the origin,
along any unit vector `v`. -/
theorem dirDeriv2_radial {f f1 f2 : ℝ → ℝ}
    (h1 : ∀ s, 0 < s → HasDerivAt f (f1 s) s)
    (h2 : ∀ s, 0 < s → HasDerivAt f1 (f2 s) s)
    {x v : Sp n} (hv : ‖v‖ = 1) (hx : x ≠ 0) :
    dirDeriv2 (fun y => f ‖y‖) v x =
      f2 ‖x‖ * (⟪x, v⟫ / ‖x‖) ^ 2 + f1 ‖x‖ * (1 / ‖x‖ - ⟪x, v⟫ ^ 2 / ‖x‖ ^ 3) := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  set r := ‖x‖ with hrdef
  set c := ⟪x, v⟫ with hcdef
  set G : ℝ → ℝ := fun t => f1 ‖x + t • v‖ * ((c + t) / ‖x + t • v‖) with hG
  have hne : ∀ t : ℝ, |t| < r → x + t • v ≠ 0 := by
    intro t ht
    exact norm_pos_iff.mp (norm_line_pos hv ht)
  have hEq : deriv (fun s : ℝ => f ‖x + s • v‖) =ᶠ[𝓝 0] G := by
    have hball : Metric.ball (0 : ℝ) r ∈ 𝓝 (0 : ℝ) := Metric.ball_mem_nhds _ hr
    filter_upwards [hball] with t ht
    have ht' : |t| < r := by simpa [Real.dist_eq] using ht
    exact (hasDerivAt_radial_line h1 hv (hne t ht')).deriv
  have hx0 : x + (0 : ℝ) • v = x := by simp
  have hnormd : HasDerivAt (fun s : ℝ => ‖x + s • v‖) (c / r) 0 := by
    have := hasDerivAt_norm_line (x := x) (v := v) hv (t := 0) (by simpa using hx)
    simpa [hx0] using this
  have hf1 : HasDerivAt (fun s : ℝ => f1 ‖x + s • v‖) (f2 r * (c / r)) 0 := by
    have h2' : HasDerivAt f1 (f2 r) ‖x + (0 : ℝ) • v‖ := by rw [hx0]; exact h2 r hr
    exact h2'.comp 0 hnormd
  have hinv : HasDerivAt (fun s : ℝ => (‖x + s • v‖)⁻¹) (-(c / r) / r ^ 2) 0 := by
    have hne0 : ‖x + (0 : ℝ) • v‖ ≠ 0 := by rw [hx0]; exact hr.ne'
    have h := hnormd.inv hne0
    rw [hx0] at h
    exact h
  have hlin : HasDerivAt (fun s : ℝ => c + s) 1 0 := by
    simpa using (hasDerivAt_id (0 : ℝ)).const_add c
  have hGd : HasDerivAt G
      (f2 r * (c / r) * (c * r⁻¹) + f1 r * (1 * r⁻¹ + c * (-(c / r) / r ^ 2))) 0 := by
    have hprod : HasDerivAt (fun s : ℝ => (c + s) * (‖x + s • v‖)⁻¹)
        (1 * r⁻¹ + c * (-(c / r) / r ^ 2)) 0 := by
      have := hlin.mul hinv
      simpa [hx0] using this
    have := hf1.mul hprod
    simpa [hG, div_eq_mul_inv, hx0] using this
  have hrw : dirDeriv2 (fun y => f ‖y‖) v x = deriv G 0 := by
    unfold dirDeriv2
    exact Filter.EventuallyEq.deriv_eq hEq
  rw [hrw, hGd.deriv]
  field_simp
  ring

/-- **Radial reduction of the Laplacian.** For a radial field `u(x) = f(‖x‖)` on
`n`-dimensional space, `Δu(x) = f''(r) + (n-1) f'(r)/r` at every `x ≠ 0`. -/
theorem laplacian_radial {f f1 f2 : ℝ → ℝ}
    (h1 : ∀ s, 0 < s → HasDerivAt f (f1 s) s)
    (h2 : ∀ s, 0 < s → HasDerivAt f1 (f2 s) s)
    {x : Sp n} (hx : x ≠ 0) :
    laplacian (fun y => f ‖y‖) x = f2 ‖x‖ + (n - 1 : ℝ) * f1 ‖x‖ / ‖x‖ := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hsingle : ∀ i : Fin n, ‖EuclideanSpace.single i (1 : ℝ)‖ = 1 := by
    intro i; simp
  have hinner : ∀ i : Fin n, ⟪x, EuclideanSpace.single i (1 : ℝ)⟫ = x i := by
    intro i; simp [EuclideanSpace.inner_single_right]
  have hsum : ∑ i : Fin n, (x i) ^ 2 = ‖x‖ ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (Finset.sum_nonneg fun i _ => by positivity)]
    simp
  unfold laplacian pderiv2
  have hterm : ∀ i : Fin n, dirDeriv2 (fun y => f ‖y‖) (EuclideanSpace.single i (1 : ℝ)) x
      = f2 ‖x‖ * (x i / ‖x‖) ^ 2 + f1 ‖x‖ * (1 / ‖x‖ - (x i) ^ 2 / ‖x‖ ^ 3) := by
    intro i
    rw [dirDeriv2_radial h1 h2 (hsingle i) hx, hinner i]
  simp_rw [hterm]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  have e1 : ∑ i : Fin n, (x i / ‖x‖) ^ 2 = 1 := by
    have hd : ∑ i : Fin n, (x i / ‖x‖) ^ 2 = (∑ i : Fin n, (x i) ^ 2) / ‖x‖ ^ 2 := by
      rw [Finset.sum_div]; exact Finset.sum_congr rfl fun i _ => by rw [div_pow]
    rw [hd, hsum, div_self (by positivity)]
  have e2 : ∑ _i : Fin n, (1 / ‖x‖) = (n : ℝ) / ‖x‖ := by
    simp [Finset.sum_const, nsmul_eq_mul, div_eq_mul_inv]
  have e3 : ∑ i : Fin n, ((x i) ^ 2 / ‖x‖ ^ 3) = 1 / ‖x‖ := by
    rw [← Finset.sum_div, hsum]
    rw [div_eq_div_iff (by positivity) (by positivity)]
    ring
  rw [e1, Finset.sum_sub_distrib, e2, e3]
  field_simp

/-! ### First consequences: the Coulomb kernel is harmonic -/

lemma hasDerivAt_inv_pos {s : ℝ} (hs : 0 < s) :
    HasDerivAt (fun y : ℝ => y⁻¹) (-(1 / s ^ 2)) s := by
  have := hasDerivAt_inv hs.ne'
  simpa [one_div] using this

lemma hasDerivAt_neg_inv_sq {s : ℝ} (hs : 0 < s) :
    HasDerivAt (fun y : ℝ => -(1 / y ^ 2)) (2 / s ^ 3) s := by
  have hd : HasDerivAt (fun y : ℝ => y ^ (2 : ℕ)) (2 * s) s := by
    simpa [mul_comm] using hasDerivAt_pow 2 s
  have hinv : HasDerivAt (fun y : ℝ => (y ^ (2 : ℕ))⁻¹) (-(2 * s) / (s ^ 2) ^ 2) s :=
    hd.inv (by positivity)
  have h := hinv.neg
  convert h using 1
  · ext y; simp [one_div]
  · field_simp

/-- In three dimensions the Coulomb kernel `1/r` is harmonic away from the source. -/
theorem laplacian_coulomb {x : Sp 3} (hx : x ≠ 0) :
    laplacian (fun y => ‖y‖⁻¹) x = 0 := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  rw [laplacian_radial (f1 := fun s => -(1 / s ^ 2)) (f2 := fun s => 2 / s ^ 3)
    (fun s hs => hasDerivAt_inv_pos hs) (fun s hs => hasDerivAt_neg_inv_sq hs) hx]
  push_cast
  field_simp
  ring

end RequestProject.Physics
