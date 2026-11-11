/-
# Part IX.5  Excluded volume: the swollen coil and the Flory exponent

The chain statistics of `RequestProject.Polymer` are those of a *phantom* chain: bonds
correlate but monomers may overlap.  A real polypeptide cannot, and self-avoidance is what
takes the measured size exponent of a disordered region away from the random-walk value
`1/2`.  This file proves the classical Flory argument exactly, with no approximations beyond
the form of the free energy itself.

The Flory free energy of a chain of `N` monomers occupying a region of size `R`,

  `F(R) = a·R²/N + v·N²/R³`   (`floryFreeEnergy`),

balances the entropic elasticity of the chain (the Gaussian term, growing with extension)
against the two-body repulsion between monomers (the excluded-volume term, growing with
compaction).

* `floryFreeEnergy_min` -- `F` has a *unique* minimiser on `(0,∞)`, and it is a strict global
  minimum.  The proof is exact algebra: the difference of free energies factors as
  `(R - R*)²·(3R³ + 6R*R² + 4R*²R + 2R*³)`, manifestly nonnegative and vanishing only at
  `R = R*`.
* `floryRadius_eq` -- the minimiser is `R* = (3v/2a)^{1/5}·N^{3/5}`: the **Flory exponent
  `ν = 3/5`**, in place of the ideal-chain `1/2`.
* `flory_swelling` -- consequently `R*/√N → ∞`: a self-avoiding chain is asymptotically more
  expanded than *any* ideal chain, by a factor growing as `N^{1/10}`.
* `flory_theta` -- at the theta point (`v = 0`, repulsion exactly cancelled by solvent) the
  free energy is minimised at `R = 0` and the swelling disappears; the exponent is a property
  of the *solvent*, not of the sequence alone.

The design consequence, with `RequestProject.Polymer`: the size of a disordered region is set
by a competition of two terms with different powers of `N`, so a model must be fitted to data
at more than one chain length before its scaling can be believed, and the fitted exponent is
a statement about the solvent conditions of the measurement.
-/
import Mathlib

namespace IDR

open Finset

namespace Flory

/-- The Flory free energy of a chain of `N` monomers at size `R`: entropic elasticity plus
two-body excluded-volume repulsion. -/
noncomputable def floryFreeEnergy (a v N R : ℝ) : ℝ := a * R ^ 2 / N + v * N ^ 2 / R ^ 3

/-- The Flory radius `R* = (3v/2a)^{1/5} N^{3/5}`. -/
noncomputable def floryRadius (a v N : ℝ) : ℝ :=
  (3 * v / (2 * a)) ^ ((1 : ℝ) / 5) * N ^ ((3 : ℝ) / 5)

lemma floryRadius_pos {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 0 < N) :
    0 < floryRadius a v N := by
  unfold floryRadius
  have h1 : 0 < 3 * v / (2 * a) := by positivity
  exact mul_pos (Real.rpow_pos_of_pos h1 _) (Real.rpow_pos_of_pos hN _)

/-- The defining property of the Flory radius: `R*⁵ = (3v/2a)·N³`. -/
lemma floryRadius_pow_five {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 0 < N) :
    (floryRadius a v N) ^ (5 : ℕ) = (3 * v / (2 * a)) * N ^ (3 : ℕ) := by
  have h1 : (0:ℝ) < 3 * v / (2 * a) := by positivity
  unfold floryRadius
  rw [mul_pow]
  rw [← Real.rpow_natCast ((3 * v / (2 * a)) ^ ((1 : ℝ) / 5)) 5,
    ← Real.rpow_natCast (N ^ ((3 : ℝ) / 5)) 5,
    ← Real.rpow_mul h1.le, ← Real.rpow_mul hN.le]
  norm_num

/-- **The Flory minimum.**  For every size `R > 0` other than the Flory radius, the free
energy is strictly larger.  Exact, with no expansion in any small parameter. -/
theorem floryFreeEnergy_min {a v N : ℝ} (ha : 0 < a) (hv : 0 < v) (hN : 0 < N)
    {R : ℝ} (hR : 0 < R) (hne : R ≠ floryRadius a v N) :
    floryFreeEnergy a v N (floryRadius a v N) < floryFreeEnergy a v N R := by
  set Rs := floryRadius a v N with hRs
  have hRspos : 0 < Rs := floryRadius_pos ha hv hN
  have h5 : Rs ^ (5 : ℕ) = (3 * v / (2 * a)) * N ^ (3 : ℕ) := floryRadius_pow_five ha hv hN
  -- express the excluded-volume coefficient through the Flory radius
  have hv' : v * N ^ 2 = (2 * a / (3 * N)) * Rs ^ 5 := by
    rw [h5]
    field_simp
  have hdiff : floryFreeEnergy a v N R - floryFreeEnergy a v N Rs
      = (a / (3 * N * R ^ 3))
        * ((R - Rs) ^ 2 * (3 * R ^ 3 + 6 * Rs * R ^ 2 + 4 * Rs ^ 2 * R + 2 * Rs ^ 3)) := by
    unfold floryFreeEnergy
    rw [hv']
    field_simp
    ring
  have hposcoef : 0 < a / (3 * N * R ^ 3) := by positivity
  have hsqpos : 0 < (R - Rs) ^ 2 := by
    have : R - Rs ≠ 0 := sub_ne_zero.mpr hne
    positivity
  have hcubic : 0 < 3 * R ^ 3 + 6 * Rs * R ^ 2 + 4 * Rs ^ 2 * R + 2 * Rs ^ 3 := by positivity
  have : 0 < floryFreeEnergy a v N R - floryFreeEnergy a v N Rs := by
    rw [hdiff]
    exact mul_pos hposcoef (mul_pos hsqpos hcubic)
  linarith

/-- The Flory radius grows as `N^{3/5}` -- the swollen-coil exponent `ν = 3/5`. -/
theorem floryRadius_eq (a v N : ℝ) :
    floryRadius a v N = (3 * v / (2 * a)) ^ ((1 : ℝ) / 5) * N ^ ((3 : ℝ) / 5) := rfl

/-- **Self-avoidance swells the chain.**  The ratio of the Flory radius to the ideal-chain
size `√N` diverges like `N^{1/10}`: no ideal chain, of any bond length, reproduces the size
of a long self-avoiding chain. -/
theorem flory_swelling {a v : ℝ} (ha : 0 < a) (hv : 0 < v) :
    Filter.Tendsto (fun N : ℝ => floryRadius a v N / Real.sqrt N) Filter.atTop
      Filter.atTop := by
  have hc : (0:ℝ) < (3 * v / (2 * a)) ^ ((1 : ℝ) / 5) :=
    Real.rpow_pos_of_pos (by positivity) _
  have heq : ∀ᶠ N : ℝ in Filter.atTop,
      floryRadius a v N / Real.sqrt N
        = (3 * v / (2 * a)) ^ ((1 : ℝ) / 5) * N ^ ((1 : ℝ) / 10) := by
    filter_upwards [Filter.eventually_gt_atTop 0] with N hN
    unfold floryRadius
    rw [Real.sqrt_eq_rpow, mul_div_assoc, ← Real.rpow_sub hN]
    norm_num
  refine Filter.Tendsto.congr' (Filter.EventuallyEq.symm heq) ?_
  exact Filter.Tendsto.const_mul_atTop hc (tendsto_rpow_atTop (by norm_num))

/-- **The theta point.**  With the excluded-volume parameter switched off the repulsive term
disappears and the free energy is strictly increasing in the size: the swelling is a property
of the solvent, and at the theta temperature the chain returns to ideal statistics. -/
theorem flory_theta {a N : ℝ} (ha : 0 < a) (hN : 0 < N) :
    StrictMonoOn (floryFreeEnergy a 0 N) (Set.Ici (0 : ℝ)) := by
  intro x hx y hy hxy
  have hx0 : (0:ℝ) ≤ x := hx
  unfold floryFreeEnergy
  simp only [zero_mul, zero_div, add_zero]
  have hsq : x ^ 2 < y ^ 2 := by nlinarith [hx0, hxy]
  have h2 : a * x ^ 2 < a * y ^ 2 := mul_lt_mul_of_pos_left hsq ha
  exact (div_lt_div_iff_of_pos_right hN).mpr h2

end Flory

end IDR
