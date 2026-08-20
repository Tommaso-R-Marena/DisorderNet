/-
# Part XXXVI  Circular dichroism: the secondary-structure content is a projection

`RequestProject.Dichroism` treats the experiment most often quoted for a disordered region, and
the numbers extracted from it: the far-UV CD spectrum and the helix / sheet / turn / PPII / coil
percentages returned by deconvolving it against a reference basis.

The forward model is the mixture `mix B f i = ∑ⱼ fⱼ Bⱼᵢ`.  A change of composition that leaves
the total and every measured wavelength unchanged is a null direction, and along one the fit is
literally blind.  Null directions exist whenever the classes outnumber the channels by two or
more, they occur in the interior of the simplex and not only at its boundary, and a concrete
four-class basis at two wavelengths admits two compositions differing by 13 percentage points of
β-sheet with identical spectra.  The case that matters most for a disordered region is the
near-coincidence of the PPII and statistical-coil reference spectra: were they equal the split
between them would be entirely free, and as it is the split is bounded only by the noise, every
transfer of weight `t` perturbing the data by at most `|t|·ε`.  What survives is exactly the
affine read-outs: a weighted content `∑ⱼ wⱼ fⱼ` is determined precisely when `w` is itself a
linear functional of the basis spectra.

`IDR.dichroism_laws` bundles the six statements.
-/
import Mathlib
import RequestProject.Dichroism

set_option autoImplicit false

namespace IDR

/-- **The design laws of a CD deconvolution.**

1. *Blind directions*: along a null direction both the total weight and the whole predicted
   spectrum are exactly constant.
2. *They exist by counting*: if the number of wavelengths plus one is less than the number of
   classes, a nonzero null direction exists whatever the reference spectra are.
3. *And they reach the interior*: around any strictly positive composition there is a different,
   still nonnegative and normalised composition with the same spectrum.
4. *A concrete witness*: a realistic four-class basis at two wavelengths gives two normalised,
   nonnegative compositions differing by more than 13 percentage points of β-sheet with identical
   spectra.
5. *Degenerate references are free parameters, near-degenerate ones are noise-limited*: if two
   reference spectra agree, all splits of their weight fit; if they differ by at most `ε`, a
   transfer of weight `t` moves the data by at most `|t|·ε`.
6. *What is determined*: any content whose weight vector is an affine read-out of the basis
   spectra takes the same value on every fitting composition. -/
theorem dichroism_laws :
    -- 1  along a null direction the data do not move
    (∀ (k m : ℕ) (B : Fin k → Fin m → ℝ) (d : Fin k → ℝ), CD.IsNull B d →
        ∀ (f : Fin k → ℝ) (t : ℝ),
          CD.mix B (f + t • d) = CD.mix B f ∧ ∑ j, (f + t • d) j = ∑ j, f j) ∧
    -- 2  null directions exist as soon as the classes outnumber the channels
    (∀ (k m : ℕ) (B : Fin k → Fin m → ℝ), m + 1 < k →
        ∃ d : Fin k → ℝ, d ≠ 0 ∧ CD.IsNull B d) ∧
    -- 3  and they produce a segment of fitting compositions in the interior of the simplex
    (∀ (k m : ℕ) (B : Fin k → Fin m → ℝ) (d f : Fin k → ℝ), d ≠ 0 → CD.IsNull B d →
        (∀ j, 0 < f j) →
        ∃ t : ℝ, 0 < t ∧ (∀ j, 0 ≤ (f + t • d) j) ∧ (∑ j, (f + t • d) j = ∑ j, f j) ∧
          CD.mix B (f + t • d) = CD.mix B f ∧ f + t • d ≠ f) ∧
    -- 4  a concrete two-wavelength witness
    ((∀ j, 0 ≤ CD.fw j) ∧ (∀ j, 0 ≤ CD.gw j) ∧ (∑ j, CD.fw j = 1) ∧ (∑ j, CD.gw j = 1) ∧
      CD.mix CD.Bw CD.fw = CD.mix CD.Bw CD.gw ∧ CD.gw 1 - CD.fw 1 > 13/100) ∧
    -- 5  equal reference spectra are a free parameter; near-equal ones are noise-limited
    (∀ (k m : ℕ) (B : Fin k → Fin m → ℝ) (j₁ j₂ : Fin k), j₁ ≠ j₂ → B j₁ = B j₂ →
        ∀ (f : Fin k → ℝ) (t : ℝ),
          CD.mix B (f + t • CD.swapDir j₁ j₂) = CD.mix B f ∧
            (∑ j, (f + t • CD.swapDir j₁ j₂) j = ∑ j, f j) ∧
            (f + t • CD.swapDir j₁ j₂) j₁ = f j₁ + t) ∧
    (∀ (k m : ℕ) (B : Fin k → Fin m → ℝ) (j₁ j₂ : Fin k) (eps : ℝ),
        (∀ i, |B j₁ i - B j₂ i| ≤ eps) → ∀ (f : Fin k → ℝ) (t : ℝ) (i : Fin m),
          |CD.mix B (f + t • CD.swapDir j₁ j₂) i - CD.mix B f i| ≤ |t| * eps) ∧
    -- 6  affine read-outs of the basis spectra are determined
    (∀ (k m : ℕ) (B : Fin k → Fin m → ℝ) (w : Fin k → ℝ) (c : Fin m → ℝ) (c0 : ℝ),
        (∀ j, w j = c0 + ∑ i, c i * B j i) → ∀ f g : Fin k → ℝ,
          (∑ j, f j = ∑ j, g j) → CD.mix B f = CD.mix B g →
          ∑ j, w j * f j = ∑ j, w j * g j) := by
  refine ⟨fun k m B d hd f t => ⟨CD.mix_perturb hd f t, CD.sum_perturb hd f t⟩,
    fun k m B h => CD.exists_null B h,
    fun k m B d f hd hnull hf => CD.segment_of_null hd hnull hf,
    CD.two_wavelength_witness,
    fun k m B j₁ j₂ hne heq f t => CD.equal_basis_split_free hne heq f t,
    fun k m B j₁ j₂ eps heps f t i => CD.near_degenerate_tolerance heps f t i,
    fun k m B w c c0 hw f g hsum hmix => CD.readout_determined hw hsum hmix⟩

end IDR
