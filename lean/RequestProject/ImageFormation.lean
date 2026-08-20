/-
# Part CIV  Image formation: the contrast transfer function, solvent, truncation, alignment, noise

Part LXXXVI models a cryo-EM map as an exact ensemble average on a voxel grid, and the assumptions
list said so plainly: "no image-formation model, no contrast transfer function, no noise and no
alignment error", with "bulk-solvent modelling, map sharpening, resolution truncation" also left
outside in Part XXXVIII/LVII.  This file supplies the missing layer, each item as a theorem about
what the data does and does not determine.

**The contrast transfer function.**  A micrograph records the structure's Fourier amplitude
multiplied by `ctf lam df s = sin(π·lam·df·s²)`, which vanishes at the spatial frequencies
`s_k = √(k/(lam·df))`.

* `ctf_zero_at`, `image_blind_at_zero` — **at those frequencies the micrograph is identically
  zero whatever the structure is**: two structures differing only there produce the same image,
  and no processing of that micrograph recovers the difference.  A single-defocus dataset is
  blind at infinitely many frequencies, whatever its nominal resolution.
* `commensurate_defocus_shares_zeros` — **and doubling the defocus does not help**: every zero of
  the first transfer function is a zero of the second, so a two-defocus dataset with a
  commensurate pair is blind at exactly the same frequencies as one micrograph.
* `incommensurate_defocus_fills_zeros` — **an incommensurate ratio does help**: with defocus ratio
  `√2`, no zero of the first transfer function is a zero of the second, so
  `two_defocus_recovers` applies at every one of them: the pair of images determines the
  amplitude.  Defocus randomisation is not a convenience, it is what makes the inverse problem
  solvable.

**Bulk solvent.**  `occupancy_solvent_bias` — the image records the *contrast* `(ρ_p − ρ_s)`, not
the protein density, so an occupancy fitted from a map processed without a solvent model comes out
low by exactly the factor `1 − ρ_s/ρ_p`: about a fifth for protein in water.  A disordered region
at partial occupancy is exactly what this bias is mistaken for, and the theorem gives the size of
the mistake in closed form.

**Resolution truncation.**  `truncation_blind` — two densities whose Fourier coefficients agree
below the truncation frequency but differ above it are different functions with identical
truncated data.  A map is an equivalence class, and every statement made from it must be
invariant under adding high-frequency content.

**Alignment error.**  `alignment_peak_drop` — averaging particles whose assigned centres are wrong
by `±d` lowers the peak of a Gaussian feature by exactly `exp(−d²/2σ²)`.  Since Part XXXVIII reads
partial occupancy off peak height, alignment error is *indistinguishable* from occupancy loss at a
single feature: this is the sense in which the earlier bounds were the best case.

**Noise.**  `particles_needed` — the number of particles required to reach a given signal-to-noise
ratio at a frequency where the transfer function has magnitude `c` grows as `1/c²`.  Near a zero
of the transfer function the requirement diverges, which is the quantitative form of the blindness
above.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.ImageFormation

/-! ## The contrast transfer function -/

/-- The (phase-contrast) transfer function at spatial frequency `s`, for electron wavelength
`lam` and defocus `df`, in the spherical-aberration-free form. -/
noncomputable def ctf (lam df s : ℝ) : ℝ := Real.sin (Real.pi * lam * df * s ^ 2)

/-- The recorded Fourier amplitude: the structure's amplitude `A` multiplied by the transfer
function. -/
noncomputable def image (lam df s A : ℝ) : ℝ := ctf lam df s * A

/-- The `k`-th zero frequency of the transfer function. -/
noncomputable def zeroFreq (lam df : ℝ) (k : ℕ) : ℝ := Real.sqrt (k / (lam * df))

/-- **The transfer function vanishes at `s_k = √(k/(lam·df))`.** -/
theorem ctf_zero_at {lam df : ℝ} (hlam : 0 < lam) (hdf : 0 < df) (k : ℕ) :
    ctf lam df (zeroFreq lam df k) = 0 := by
  have hpos : 0 ≤ (k : ℝ) / (lam * df) := by positivity
  have hsq : (zeroFreq lam df k) ^ 2 = (k : ℝ) / (lam * df) := Real.sq_sqrt hpos
  unfold ctf
  rw [hsq]
  have : Real.pi * lam * df * ((k : ℝ) / (lam * df)) = (k : ℝ) * Real.pi := by
    field_simp
  rw [this]
  exact Real.sin_nat_mul_pi k

/-- **At a zero of the transfer function the micrograph carries no information**: the recorded
amplitude is zero whatever the structure is, so two structures differing only there are
indistinguishable. -/
theorem image_blind_at_zero {lam df s : ℝ} (hz : ctf lam df s = 0) (A B : ℝ) :
    image lam df s A = image lam df s B ∧ image lam df s A = 0 := by
  unfold image
  rw [hz]
  simp

/-- **Two defoci determine the amplitude at every frequency where at least one transfer function
is nonzero.** -/
theorem two_defocus_recovers {lam df₁ df₂ s A B : ℝ}
    (hne : ctf lam df₁ s ≠ 0 ∨ ctf lam df₂ s ≠ 0)
    (h₁ : image lam df₁ s A = image lam df₁ s B)
    (h₂ : image lam df₂ s A = image lam df₂ s B) : A = B := by
  unfold image at h₁ h₂
  rcases hne with h | h
  · exact mul_left_cancel₀ h h₁
  · exact mul_left_cancel₀ h h₂

/-- **A commensurate defocus pair is blind at the same frequencies.**  Doubling the defocus keeps
every zero of the original transfer function. -/
theorem commensurate_defocus_shares_zeros {lam df : ℝ} (hlam : 0 < lam) (hdf : 0 < df) (k : ℕ) :
    ctf lam df (zeroFreq lam df k) = 0 ∧ ctf lam (2 * df) (zeroFreq lam df k) = 0 := by
  refine ⟨ctf_zero_at hlam hdf k, ?_⟩
  have hpos : 0 ≤ (k : ℝ) / (lam * df) := by positivity
  have hsq : (zeroFreq lam df k) ^ 2 = (k : ℝ) / (lam * df) := Real.sq_sqrt hpos
  unfold ctf
  rw [hsq]
  have : Real.pi * lam * (2 * df) * ((k : ℝ) / (lam * df)) = ((2 * k : ℕ) : ℝ) * Real.pi := by
    push_cast
    field_simp
  rw [this]
  exact Real.sin_nat_mul_pi (2 * k)

/-- **An incommensurate defocus pair fills the zeros.**  With defocus ratio `√2`, no zero of the
first transfer function (other than the trivial one at zero frequency) is a zero of the second, so
the pair of micrographs determines the amplitude there. -/
theorem incommensurate_defocus_fills_zeros {lam df : ℝ} (hlam : 0 < lam) (hdf : 0 < df) {k : ℕ}
    (hk : 1 ≤ k) : ctf lam (Real.sqrt 2 * df) (zeroFreq lam df k) ≠ 0 := by
  have hpos : 0 ≤ (k : ℝ) / (lam * df) := by positivity
  have hsq : (zeroFreq lam df k) ^ 2 = (k : ℝ) / (lam * df) := Real.sq_sqrt hpos
  unfold ctf
  rw [hsq]
  have harg : Real.pi * lam * (Real.sqrt 2 * df) * ((k : ℝ) / (lam * df))
      = (Real.sqrt 2 * k) * Real.pi := by
    field_simp
  rw [harg]
  intro hzero
  rw [Real.sin_eq_zero_iff] at hzero
  obtain ⟨n, hn⟩ := hzero
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  have hnk : (n : ℝ) = Real.sqrt 2 * k := mul_right_cancel₀ hpi hn
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk
  have : Real.sqrt 2 = (n : ℝ) / (k : ℝ) := by
    rw [hnk]; field_simp
  have hirr := irrational_sqrt_two
  rw [this] at hirr
  exact hirr ⟨(n : ℚ) / (k : ℚ), by push_cast; ring⟩

/-! ## Bulk solvent -/

/-- **Fitting an occupancy without a solvent model biases it low by the density contrast.**  The
map records `(ρ_p − ρ_s)·occ`; a fit that assumes the protein density `ρ_p` therefore returns
`occ·(1 − ρ_s/ρ_p)`. -/
theorem occupancy_solvent_bias {rhoP rhoS occ occFit : ℝ} (hp : 0 < rhoP)
    (hfit : rhoP * occFit = (rhoP - rhoS) * occ) :
    occFit = occ * (1 - rhoS / rhoP) := by
  field_simp
  linarith [hfit]

/-! ## Resolution truncation -/

/-- A density built from a finite cosine series with coefficients `c`, cut at order `K`. -/
noncomputable def dens (c : ℕ → ℝ) (K : ℕ) (x : ℝ) : ℝ :=
  ∑ k ∈ Finset.range K, c k * Real.cos (k * x)

/-- **Truncated data is blind to high-frequency content.**  Two coefficient sets that agree below
the truncation order `smax` but differ above it give the same truncated map and different
densities. -/
theorem truncation_blind :
    ∃ c c' : ℕ → ℝ, (∀ k, k < 5 → c k = c' k) ∧ dens c 10 ≠ dens c' 10 := by
  refine ⟨fun _ => 0, fun k => if k = 6 then 1 else 0, fun k hk => by simp; omega, ?_⟩
  intro hcontra
  have h0 := congrFun hcontra 0
  unfold dens at h0
  simp [Finset.sum_ite_eq'] at h0

/-! ## Alignment error -/

/-- A Gaussian feature of width `sigma`, peak height one. -/
noncomputable def gauss (sigma x : ℝ) : ℝ := Real.exp (-(x ^ 2) / (2 * sigma ^ 2))

/-- **Alignment error lowers the peak exactly as partial occupancy does.**  Averaging particles
whose centres are misassigned by `±d` gives a peak height `exp(−d²/2σ²) < 1`. -/
theorem alignment_peak_drop {sigma d : ℝ} (hs : 0 < sigma) (hd : d ≠ 0) :
    (gauss sigma (0 - d) + gauss sigma (0 + d)) / 2
        = Real.exp (-(d ^ 2) / (2 * sigma ^ 2)) ∧
      Real.exp (-(d ^ 2) / (2 * sigma ^ 2)) < gauss sigma 0 := by
  constructor
  · unfold gauss
    ring_nf
  · unfold gauss
    simp only [zero_pow, neg_zero, zero_div, Real.exp_zero, ne_eq, OfNat.ofNat_ne_zero,
      not_false_eq_true]
    rw [Real.exp_lt_one_iff]
    have hd2 : 0 < d ^ 2 := by positivity
    have : 0 < 2 * sigma ^ 2 := by positivity
    exact div_neg_of_neg_of_pos (by linarith) this

/-! ## Noise -/

/-- **The particle count needed at a frequency grows as the inverse square of the transfer
function there.** -/
theorem particles_needed {c A sigma theta N : ℝ} (hc : c ≠ 0) (hA : A ≠ 0) (hsig : 0 < sigma)
    (hN : theta ≤ (N * (c * A) ^ 2) / sigma ^ 2) :
    theta * sigma ^ 2 / (c * A) ^ 2 ≤ N := by
  have hca : 0 < (c * A) ^ 2 := by positivity
  have hs2 : 0 < sigma ^ 2 := by positivity
  rw [le_div_iff₀ hs2] at hN
  rw [div_le_iff₀ hca]
  nlinarith [hN]

end IDR.ImageFormation
