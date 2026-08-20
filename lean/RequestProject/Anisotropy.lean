/-
# Part LVII.1  Anisotropic displacement and discrete conformers in a density map

Part XXXVIII models a smeared atom by a single occupancy and a single *isotropic* displacement
parameter, and says so: "anisotropic displacement, multiple discrete conformers, bulk-solvent
modelling and map sharpening are outside it".  For a disordered region those are not decorations.
The whole content of the claim "this region is disordered" is that the atom is in several places,
and the whole content of a refined `B` factor is that one number has been asked to stand for a
displacement that is in general a tensor.  This file removes both idealisations.

**Discrete conformers versus a large `B`.**  A two-conformer atom, sites at `±d`, each smeared by
a Gaussian of width `s`, has the real-space density `twoSite s d`.  The refinement program offers
instead the one-parameter family `gauss u` -- one site, one isotropic width.

* `twoSite_eq` -- the exact factorisation `twoSite s d x = gauss s x · e^{-d²/2s²} · cosh(xd/s²)`,
  which is what makes everything below computable.
* `twoSite_ne_gauss` -- **no inflated `B` reproduces a split site.**  For every `d ≠ 0` and every
  width `u > 0` whatsoever, the two densities differ somewhere.  This is not a statement about
  resolution: it says the single-site family does not contain the two-site density at all.
* `twoSite_center_lt_matched` -- and the deviation has a sign at the place one looks: matching
  the second moment (`u² = s² + d²`, the standard "inflate `B` by the spread" prescription)
  the fitted single-site density is strictly *too high* at the centre.
* `twoSite_bimodal` -- once the sites are resolved (`2s² ≤ d²`) the true density has a dip at the
  midpoint, `twoSite s d 0 < twoSite s d d`, which no Gaussian has anywhere.  The qualitative
  signature is a shape, not a width.

**The displacement is a tensor.**  `anisoDensity u` is the anisotropic Gaussian with principal
widths `u 0, u 1, u 2`; `equivB u` is the isotropic equivalent, the mean of the squares.

* `anisoDensity_inj` -- **the map determines the tensor.**  If two anisotropic densities agree at
  every point then their principal widths agree.  Nothing is lost by refining the tensor; the
  information is there.
* `equivB_not_determining` -- **the isotropic equivalent does not.**  Two tensors with the same
  `equivB` -- the same refined isotropic `B` factor -- have densities that differ at some point.
* `aniso_ratio_unbounded` -- **and the discarded information is unbounded.**  For every isotropic
  equivalent `B > 0` and every ratio `R` there is a displacement tensor with exactly that `B`
  whose principal widths differ by more than a factor `R`.  A quoted `B` factor constrains the
  mean square displacement and says nothing at all about its shape.
* `occupancy_anisotropy_degenerate` -- the Part XXXVIII occupancy/`B` trade-off survives in
  tensor form: any peak height is reproduced by any occupancy at a suitably scaled tensor, and
  the resulting densities are different functions.

The reading for a model of a disordered region: a deposited `(occupancy, B_iso)` pair is a
projection of the ensemble onto two numbers, and the projection is many-to-one in two independent
directions at once -- along conformer multiplicity and along the shape of the displacement.  A
model that predicts an ensemble should be compared with the map, not with the pair.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace Anisotropy

open Real Finset

/-! ## One dimension: discrete conformers versus an inflated `B` -/

/-- The normalised one-dimensional Gaussian of width `s`: the real-space density of a single
atom smeared by an isotropic displacement parameter. -/
noncomputable def gauss (s x : ℝ) : ℝ :=
  Real.exp (-(x ^ 2) / (2 * s ^ 2)) / (s * Real.sqrt (2 * Real.pi))

lemma sqrtTwoPi_pos : (0:ℝ) < Real.sqrt (2 * Real.pi) :=
  Real.sqrt_pos.2 (by positivity)

lemma gauss_pos {s : ℝ} (hs : 0 < s) (x : ℝ) : 0 < gauss s x := by
  have h2 := sqrtTwoPi_pos
  unfold gauss
  positivity

/-- The density of a two-conformer atom: sites at `±d`, equal occupancy, each smeared by a
Gaussian of width `s`. -/
noncomputable def twoSite (s d x : ℝ) : ℝ := (gauss s (x - d) + gauss s (x + d)) / 2

lemma gauss_shift {s : ℝ} (hs : s ≠ 0) (x d : ℝ) :
    gauss s (x - d) = gauss s x * Real.exp (x * d / s ^ 2 - d ^ 2 / (2 * s ^ 2)) := by
  have hs2 : s ^ 2 ≠ 0 := pow_ne_zero 2 hs
  have hkey : -((x - d) ^ 2) / (2 * s ^ 2)
      = -(x ^ 2) / (2 * s ^ 2) + (x * d / s ^ 2 - d ^ 2 / (2 * s ^ 2)) := by
    field_simp
    ring
  unfold gauss
  rw [hkey, Real.exp_add, div_mul_eq_mul_div]

/-- The exact factorisation of the two-site density: a Gaussian, an overall Debye-like factor,
and a hyperbolic cosine that carries all the dependence on the split. -/
lemma twoSite_eq {s : ℝ} (hs : s ≠ 0) (d x : ℝ) :
    twoSite s d x
      = gauss s x * Real.exp (-(d ^ 2) / (2 * s ^ 2)) * Real.cosh (x * (d / s ^ 2)) := by
  have h1 := gauss_shift hs x d
  have h2 : gauss s (x + d) = gauss s x * Real.exp (-(x * d / s ^ 2) - d ^ 2 / (2 * s ^ 2)) := by
    have := gauss_shift hs x (-d)
    simpa [sub_neg_eq_add, neg_div, mul_neg] using this
  unfold twoSite
  rw [h1, h2, Real.cosh_eq]
  have e1 : x * d / s ^ 2 - d ^ 2 / (2 * s ^ 2)
      = -(d ^ 2) / (2 * s ^ 2) + x * (d / s ^ 2) := by ring
  have e2 : -(x * d / s ^ 2) - d ^ 2 / (2 * s ^ 2)
      = -(d ^ 2) / (2 * s ^ 2) + -(x * (d / s ^ 2)) := by ring
  rw [e1, e2, Real.exp_add, Real.exp_add]
  ring

lemma gauss_ratio {s u : ℝ} (hs : s ≠ 0) (hu : u ≠ 0) (x : ℝ) :
    gauss u x = gauss s x * (s / u) * Real.exp ((1 / (2 * s ^ 2) - 1 / (2 * u ^ 2)) * x ^ 2) := by
  have hs2 : s ^ 2 ≠ 0 := pow_ne_zero 2 hs
  have hu2 : u ^ 2 ≠ 0 := pow_ne_zero 2 hu
  have hkey : -(x ^ 2) / (2 * u ^ 2)
      = -(x ^ 2) / (2 * s ^ 2) + (1 / (2 * s ^ 2) - 1 / (2 * u ^ 2)) * x ^ 2 := by
    field_simp
    ring
  unfold gauss
  rw [hkey, Real.exp_add]
  field_simp

/-- **No inflated isotropic displacement parameter reproduces a split site.**  For a genuine
two-conformer atom (`d ≠ 0`) and *any* single-site width `u > 0`, the two real-space densities
differ at some point: the one-parameter family does not contain the two-site density at all. -/
theorem twoSite_ne_gauss {s d u : ℝ} (hs : 0 < s) (hd : d ≠ 0) (hu : 0 < u) :
    ∃ x : ℝ, twoSite s d x ≠ gauss u x := by
  by_contra hcon
  push_neg at hcon
  set a : ℝ := d / s ^ 2 with ha
  have hane : a ≠ 0 := div_ne_zero hd (pow_ne_zero 2 hs.ne')
  set c0 : ℝ := 1 / (2 * s ^ 2) - 1 / (2 * u ^ 2) with hc0
  set K : ℝ := Real.exp (-(d ^ 2) / (2 * s ^ 2)) with hK
  have hKpos : 0 < K := Real.exp_pos _
  have key : ∀ x : ℝ, K * Real.cosh (x * a) = (s / u) * Real.exp (c0 * x ^ 2) := by
    intro x
    have h1 := hcon x
    rw [twoSite_eq hs.ne' d x, gauss_ratio hs.ne' hu.ne' x] at h1
    have hg := (gauss_pos hs x).ne'
    refine mul_left_cancel₀ hg ?_
    linear_combination h1
  have k0 := key 0
  simp only [zero_mul, Real.cosh_zero, mul_one, mul_zero, ne_eq, OfNat.ofNat_ne_zero,
    not_false_eq_true, zero_pow, Real.exp_zero] at k0
  have k1 := key 1
  have k2 := key 2
  rw [← k0] at k1 k2
  have hcosh1 : Real.cosh a = Real.exp c0 := by
    have h1 : K * Real.cosh (1 * a) = K * Real.exp (c0 * 1 ^ 2) := k1
    have h2 := mul_left_cancel₀ hKpos.ne' h1
    simpa using h2
  have hcosh2 : Real.cosh (2 * a) = Real.exp (c0 * 2 ^ 2) := by
    have h1 : K * Real.cosh (2 * a) = K * Real.exp (c0 * 2 ^ 2) := k2
    exact mul_left_cancel₀ hKpos.ne' h1
  have hexp4 : Real.exp (c0 * 2 ^ 2) = Real.cosh a ^ 4 := by
    rw [hcosh1, ← Real.exp_nat_mul]
    congr 1
    norm_num
    ring
  have hdouble : Real.cosh (2 * a) = 2 * Real.cosh a ^ 2 - 1 := by
    rw [Real.cosh_two_mul, Real.cosh_sq a]
    ring
  set y : ℝ := Real.cosh a with hy
  have hy1 : 1 ≤ y := Real.one_le_cosh a
  have heq : 2 * y ^ 2 - 1 = y ^ 4 := by
    rw [← hdouble, hcosh2, hexp4]
  have hyone : y = 1 := by
    have hz : (y ^ 2 - 1) ^ 2 = 0 := by linear_combination -heq
    have hz2 : y ^ 2 - 1 = 0 := (pow_eq_zero_iff (n := 2) (by norm_num)).1 hz
    have hfac : (y - 1) * (y + 1) = 0 := by linear_combination hz2
    rcases mul_eq_zero.1 hfac with h | h <;> linarith
  have hgt : 1 < y := (Real.one_lt_cosh (x := a)).2 hane
  rw [hyone] at hgt
  exact absurd hgt (lt_irrefl 1)

/-- Matching the second moment -- the standard "inflate `B` by the conformational spread"
prescription, `u² = s² + d²` -- the fitted single-site density is strictly *too high* at the
midpoint between the two conformers. -/
theorem twoSite_center_lt_matched {s d : ℝ} (hs : 0 < s) (hd : d ≠ 0) :
    twoSite s d 0 < gauss (Real.sqrt (s ^ 2 + d ^ 2)) 0 := by
  have hd2 : 0 < d ^ 2 := by positivity
  set u : ℝ := Real.sqrt (s ^ 2 + d ^ 2) with hu
  have hupos : 0 < u := Real.sqrt_pos.2 (by positivity)
  have husq : u ^ 2 = s ^ 2 + d ^ 2 := Real.sq_sqrt (by positivity)
  have hpi := sqrtTwoPi_pos
  have hL : twoSite s d 0 = Real.exp (-(d ^ 2) / (2 * s ^ 2)) / (s * Real.sqrt (2 * Real.pi)) := by
    unfold twoSite gauss
    have h1 : (0 - d) ^ 2 = d ^ 2 := by ring
    rw [h1, zero_add]
    ring
  have hR : gauss u 0 = 1 / (u * Real.sqrt (2 * Real.pi)) := by
    unfold gauss
    norm_num
  set t : ℝ := d ^ 2 / s ^ 2 with ht
  have htpos : 0 < t := by positivity
  have hexp : 1 + t < Real.exp t := by
    have := Real.add_one_lt_exp (x := t) htpos.ne'
    linarith
  have hE : Real.exp (-(d ^ 2) / (2 * s ^ 2)) = Real.exp (-(t / 2)) := by
    congr 1
    rw [ht]
    field_simp
  have hEsq : Real.exp (-(t / 2)) ^ 2 = Real.exp (-t) := by
    rw [sq, ← Real.exp_add]
    congr 1
    ring
  have h1 : u ^ 2 = s ^ 2 * (1 + t) := by
    rw [husq, ht]
    field_simp
  have hkey : Real.exp (-(t / 2)) * u < s := by
    have hsq : (Real.exp (-(t / 2)) * u) ^ 2 < s ^ 2 := by
      have h3 : (Real.exp (-(t / 2)) * u) ^ 2 = Real.exp (-t) * (s ^ 2 * (1 + t)) := by
        rw [mul_pow, hEsq, h1]
      rw [h3, Real.exp_neg, inv_mul_eq_div, div_lt_iff₀ (Real.exp_pos t)]
      nlinarith [Real.exp_pos t, sq_nonneg s, hs]
    exact lt_of_pow_lt_pow_left₀ 2 hs.le hsq
  rw [hL, hR, hE, div_lt_div_iff₀ (by positivity) (by positivity)]
  calc Real.exp (-(t / 2)) * (u * Real.sqrt (2 * Real.pi))
      = (Real.exp (-(t / 2)) * u) * Real.sqrt (2 * Real.pi) := by ring
    _ < s * Real.sqrt (2 * Real.pi) := mul_lt_mul_of_pos_right hkey hpi
    _ = 1 * (s * Real.sqrt (2 * Real.pi)) := by ring

/-- **Resolved conformers show as a dip.**  Once `2s² ≤ d²` the two-site density is strictly
lower at the midpoint than at a site -- a shape no Gaussian, of any width, has anywhere. -/
theorem twoSite_bimodal {s d : ℝ} (hs : 0 < s) (hd : 2 * s ^ 2 ≤ d ^ 2) :
    twoSite s d 0 < twoSite s d d := by
  have hpi := sqrtTwoPi_pos
  have hs2 : (0:ℝ) < s ^ 2 := by positivity
  have hL : twoSite s d 0 = Real.exp (-(d ^ 2) / (2 * s ^ 2)) / (s * Real.sqrt (2 * Real.pi)) := by
    unfold twoSite gauss
    have h1 : (0 - d) ^ 2 = d ^ 2 := by ring
    rw [h1, zero_add]
    ring
  have hR : twoSite s d d = (1 / (s * Real.sqrt (2 * Real.pi)) + gauss s (2 * d)) / 2 := by
    have h1 : gauss s (d - d) = 1 / (s * Real.sqrt (2 * Real.pi)) := by
      unfold gauss; norm_num
    have h2 : d + d = 2 * d := by ring
    unfold twoSite
    rw [h1, h2]
  have hgpos : 0 < gauss s (2 * d) := gauss_pos hs _
  have hbound : Real.exp (-(d ^ 2) / (2 * s ^ 2)) ≤ Real.exp (-1 : ℝ) := by
    apply Real.exp_le_exp.2
    rw [div_le_iff₀ (by positivity : (0:ℝ) < 2 * s ^ 2)]
    nlinarith
  have he : Real.exp (-1 : ℝ) < 1 / 2 := by
    have h2 : (2:ℝ) < Real.exp 1 := by
      have := Real.exp_one_gt_d9
      linarith
    rw [Real.exp_neg, inv_eq_one_div, div_lt_div_iff₀ (Real.exp_pos 1) (by norm_num)]
    linarith
  have hlt : Real.exp (-(d ^ 2) / (2 * s ^ 2)) < 1 / 2 := lt_of_le_of_lt hbound he
  have hden : (0:ℝ) < s * Real.sqrt (2 * Real.pi) := by positivity
  rw [hL, hR]
  have step : Real.exp (-(d ^ 2) / (2 * s ^ 2)) / (s * Real.sqrt (2 * Real.pi))
      < (1 / (s * Real.sqrt (2 * Real.pi))) / 2 := by
    rw [div_div, div_lt_div_iff₀ hden (by positivity)]
    nlinarith [mul_lt_mul_of_pos_right hlt (show (0:ℝ) < s * Real.sqrt (2 * Real.pi) * 2 by
      positivity)]
  have hmono : (1 / (s * Real.sqrt (2 * Real.pi))) / 2
      ≤ (1 / (s * Real.sqrt (2 * Real.pi)) + gauss s (2 * d)) / 2 := by
    have h1 : (1 : ℝ) / (s * Real.sqrt (2 * Real.pi))
        ≤ 1 / (s * Real.sqrt (2 * Real.pi)) + gauss s (2 * d) := by linarith
    linarith
  linarith

/-! ## Three dimensions: the displacement is a tensor -/

/-- The anisotropic Gaussian density with principal widths `u 0, u 1, u 2`, in its principal
frame. -/
noncomputable def anisoDensity (u : Fin 3 → ℝ) (x : Fin 3 → ℝ) : ℝ :=
  Real.exp (-∑ i, (x i) ^ 2 / (2 * (u i) ^ 2)) /
    ((u 0 * u 1 * u 2) * Real.sqrt (2 * Real.pi) ^ 3)

/-- The isotropic equivalent: the mean square principal displacement, which is what a refined
isotropic `B` factor reports. -/
noncomputable def equivB (u : Fin 3 → ℝ) : ℝ := ((u 0) ^ 2 + (u 1) ^ 2 + (u 2) ^ 2) / 3

/-- **The full map determines the displacement tensor.**  Two anisotropic densities that agree
at every point have the same principal widths: nothing is lost by refining the tensor. -/
theorem anisoDensity_inj {u v : Fin 3 → ℝ} (hu : ∀ i, 0 < u i) (hv : ∀ i, 0 < v i)
    (h : ∀ x : Fin 3 → ℝ, anisoDensity u x = anisoDensity v x) : u = v := by
  have hpi : (0:ℝ) < Real.sqrt (2 * Real.pi) ^ 3 := by
    have := sqrtTwoPi_pos
    positivity
  have hPu : 0 < u 0 * u 1 * u 2 := by
    have := hu 0; have := hu 1; have := hu 2; positivity
  have hPv : 0 < v 0 * v 1 * v 2 := by
    have := hv 0; have := hv 1; have := hv 2; positivity
  have hNu : 0 < (u 0 * u 1 * u 2) * Real.sqrt (2 * Real.pi) ^ 3 := mul_pos hPu hpi
  have hNv : 0 < (v 0 * v 1 * v 2) * Real.sqrt (2 * Real.pi) ^ 3 := mul_pos hPv hpi
  have hz : ∀ w : Fin 3 → ℝ, (∑ i : Fin 3, ((fun _ : Fin 3 => (0:ℝ)) i) ^ 2 / (2 * (w i) ^ 2))
      = 0 := by
    intro w; simp
  have h0 := h (fun _ => 0)
  have hprod : u 0 * u 1 * u 2 = v 0 * v 1 * v 2 := by
    unfold anisoDensity at h0
    rw [hz u, hz v] at h0
    simp only [neg_zero, Real.exp_zero] at h0
    rw [div_eq_div_iff hNu.ne' hNv.ne'] at h0
    have h1 : (u 0 * u 1 * u 2) * Real.sqrt (2 * Real.pi) ^ 3
        = (v 0 * v 1 * v 2) * Real.sqrt (2 * Real.pi) ^ 3 := by linarith
    exact mul_right_cancel₀ hpi.ne' h1
  have hexp : ∀ x : Fin 3 → ℝ,
      (∑ i, (x i) ^ 2 / (2 * (u i) ^ 2)) = ∑ i, (x i) ^ 2 / (2 * (v i) ^ 2) := by
    intro x
    have hx := h x
    unfold anisoDensity at hx
    rw [hprod, div_eq_div_iff hNv.ne' hNv.ne'] at hx
    have h2 := mul_right_cancel₀ hNv.ne' hx
    have h3 := Real.exp_eq_exp.1 h2
    linarith
  funext i
  have hui : 0 < u i := hu i
  have hvi : 0 < v i := hv i
  have hsingle : ∀ w : Fin 3 → ℝ, (∀ j, 0 < w j) →
      (∑ j, ((Pi.single i (1:ℝ) : Fin 3 → ℝ) j) ^ 2 / (2 * (w j) ^ 2)) = 1 / (2 * (w i) ^ 2) := by
    intro w _
    rw [Finset.sum_eq_single i]
    · simp
    · intro j _ hj
      simp [hj]
    · intro hmem
      exact absurd (Finset.mem_univ i) hmem
  have hi := hexp (Pi.single i (1:ℝ) : Fin 3 → ℝ)
  rw [hsingle u hu, hsingle v hv] at hi
  rw [div_eq_div_iff (by positivity) (by positivity)] at hi
  have hsq : (u i) ^ 2 = (v i) ^ 2 := by linarith
  have e1 : Real.sqrt ((u i) ^ 2) = u i := Real.sqrt_sq hui.le
  have e2 : Real.sqrt ((v i) ^ 2) = v i := Real.sqrt_sq hvi.le
  rw [← e1, ← e2, hsq]

/-- **The isotropic equivalent does not determine the density.**  Two displacement tensors with
the same isotropic `B` factor have densities that differ at some point. -/
theorem equivB_not_determining :
    ∃ u v : Fin 3 → ℝ, (∀ i, 0 < u i) ∧ (∀ i, 0 < v i) ∧ equivB u = equivB v ∧
      ∃ x : Fin 3 → ℝ, anisoDensity u x ≠ anisoDensity v x := by
  have hhalf : (0:ℝ) < Real.sqrt (1/2) := Real.sqrt_pos.2 (by norm_num)
  have htwo : (0:ℝ) < Real.sqrt 2 := Real.sqrt_pos.2 (by norm_num)
  have h1 : Real.sqrt (1/2) ^ 2 = 1/2 := Real.sq_sqrt (by norm_num)
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)
  have hu : ∀ i, (0:ℝ) < (![1, 1, 1] : Fin 3 → ℝ) i := by
    intro i; fin_cases i <;> norm_num
  have hv : ∀ i, (0:ℝ) < (![Real.sqrt (1/2), Real.sqrt (1/2), Real.sqrt 2] : Fin 3 → ℝ) i := by
    intro i; fin_cases i <;> simp
  refine ⟨![1, 1, 1], ![Real.sqrt (1/2), Real.sqrt (1/2), Real.sqrt 2], hu, hv, ?_, ?_⟩
  · unfold equivB
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    rw [h1, h2]
    norm_num
  · by_contra hcon
    push_neg at hcon
    have heq := anisoDensity_inj hu hv hcon
    have h0 := congrFun heq 0
    simp only [Matrix.cons_val_zero] at h0
    rw [← h0] at h1
    norm_num at h1

/-- **The discarded information is unbounded.**  For every isotropic equivalent `B > 0` and every
ratio `R` there is a displacement tensor with exactly that `B` whose principal widths differ by
more than a factor `R`.  A quoted `B` factor constrains the mean square displacement and says
nothing about its shape. -/
theorem aniso_ratio_unbounded {B : ℝ} (hB : 0 < B) (R : ℝ) :
    ∃ u : Fin 3 → ℝ, (∀ i, 0 < u i) ∧ equivB u = B ∧ R < u 2 / u 0 := by
  set Rp : ℝ := max R 0 + 1 with hRp
  have hmax : (0:ℝ) ≤ max R 0 := le_max_right _ _
  have hRp1 : 1 ≤ Rp := by rw [hRp]; linarith
  have hRppos : 0 < Rp := by linarith
  have hsB : 0 < Real.sqrt B := Real.sqrt_pos.2 hB
  set eps : ℝ := Real.sqrt B / Rp with heps
  have hepspos : 0 < eps := by positivity
  have hepssq : eps ^ 2 = B / Rp ^ 2 := by
    rw [heps, div_pow, Real.sq_sqrt hB.le]
  have hepsle : eps ^ 2 ≤ B := by
    rw [hepssq, div_le_iff₀ (by positivity)]
    nlinarith [mul_nonneg hB.le (show (0:ℝ) ≤ Rp ^ 2 - 1 by nlinarith [sq_nonneg (Rp - 1)])]
  set w : ℝ := Real.sqrt (3 * B - 2 * eps ^ 2) with hw
  have hwarg : 0 < 3 * B - 2 * eps ^ 2 := by linarith
  have hwpos : 0 < w := Real.sqrt_pos.2 hwarg
  have hwsq : w ^ 2 = 3 * B - 2 * eps ^ 2 := Real.sq_sqrt hwarg.le
  refine ⟨![eps, eps, w], ?_, ?_, ?_⟩
  · intro i; fin_cases i <;> simpa using ‹_›
  · unfold equivB
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
    rw [hwsq]
    ring
  · have hidx2 : (![eps, eps, w] : Fin 3 → ℝ) 2 = w := by simp
    have hidx0 : (![eps, eps, w] : Fin 3 → ℝ) 0 = eps := by simp
    rw [hidx2, hidx0]
    have hwB : Real.sqrt B ≤ w := by
      rw [hw]
      apply Real.sqrt_le_sqrt
      linarith
    have hquot : Real.sqrt B / eps = Rp := by
      rw [heps]
      field_simp
    have hle : Real.sqrt B / eps ≤ w / eps := by gcongr
    have hRlt : R < Rp := by
      have := le_max_left R 0
      rw [hRp]; linarith
    calc R < Rp := hRlt
      _ = Real.sqrt B / eps := hquot.symm
      _ ≤ w / eps := hle

/-- The Part XXXVIII occupancy/`B` trade-off, in tensor form: for any two occupancies there are
displacement tensors giving exactly the same peak height, with densities that are nonetheless
different functions. -/
theorem occupancy_anisotropy_degenerate {q q' : ℝ} (hq : 0 < q) (hq' : 0 < q') (hne : q ≠ q')
    {u : Fin 3 → ℝ} (hu : ∀ i, 0 < u i) :
    ∃ v : Fin 3 → ℝ, (∀ i, 0 < v i) ∧
      q' * anisoDensity v (fun _ => 0) = q * anisoDensity u (fun _ => 0) ∧
      ∃ x, anisoDensity v x ≠ anisoDensity u x := by
  have hpi : (0:ℝ) < Real.sqrt (2 * Real.pi) ^ 3 := by
    have := sqrtTwoPi_pos
    positivity
  have hP : 0 < u 0 * u 1 * u 2 := by
    have := hu 0; have := hu 1; have := hu 2; positivity
  set c : ℝ := (q' / q) ^ ((1:ℝ)/3) with hc
  have hcpos : 0 < c := Real.rpow_pos_of_pos (by positivity) _
  have hc3 : c ^ 3 = q' / q := by
    rw [hc, ← Real.rpow_natCast ((q'/q) ^ ((1:ℝ)/3)) 3, ← Real.rpow_mul (by positivity)]
    norm_num
  have hzc : (∑ i : Fin 3, ((fun _ : Fin 3 => (0:ℝ)) i) ^ 2 / (2 * (c * u i) ^ 2)) = 0 := by simp
  have hzu : (∑ i : Fin 3, ((fun _ : Fin 3 => (0:ℝ)) i) ^ 2 / (2 * (u i) ^ 2)) = 0 := by simp
  refine ⟨fun i => c * u i, fun i => mul_pos hcpos (hu i), ?_, ?_⟩
  · unfold anisoDensity
    simp only
    rw [hzc, hzu]
    simp only [neg_zero, Real.exp_zero]
    have hexpand : (c * u 0) * (c * u 1) * (c * u 2) = c ^ 3 * (u 0 * u 1 * u 2) := by ring
    rw [hexpand, hc3]
    field_simp
  · by_contra hcon
    push_neg at hcon
    have hv : ∀ i, (0:ℝ) < c * u i := fun i => mul_pos hcpos (hu i)
    have heq := anisoDensity_inj hv hu hcon
    have h0 := congrFun heq 0
    simp only at h0
    have hc1 : c = 1 :=
      mul_right_cancel₀ (hu 0).ne' (show c * u 0 = 1 * u 0 by rw [one_mul]; exact h0)
    rw [hc1] at hc3
    have : q' = q := by
      rw [one_pow] at hc3
      field_simp at hc3
      linarith
    exact hne this.symm

end Anisotropy
end IDR
