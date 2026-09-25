/-
# Part XXXVI  Circular dichroism: the secondary-structure content is a projection, not a
measurement

A far-UV circular dichroism spectrum is the single most frequently reported experiment on a
disordered region, and it is almost always reported after deconvolution: the spectrum is written
as a nonnegative, sum-to-one mixture of reference spectra for helix, sheet, turn, polyproline II
and coil, and the fitted fractions are quoted as the secondary-structure content of the region.
This file asks what those fractions are a statement about.

The forward model is exactly the one used by every deconvolution program: `mix B f i = ∑ⱼ fⱼ Bⱼᵢ`,
the mixture of `k` basis spectra `B` sampled at `m` wavelengths (`mix`).  A *null direction*
(`IsNull`) is a change of composition that leaves both the total and the spectrum unchanged.

* `mix_perturb`, `sum_perturb` -- along a null direction the data are literally constant, so the
  fit cannot distinguish `f` from `f + t·d`.
* `exists_null` -- and a null direction always exists when `m + 1 < k`: with fewer independent
  wavelength channels than classes minus one, deconvolution is underdetermined as a matter of
  linear algebra, whatever the numbers are.  (The real basis sets are worse than this bound
  suggests, because the sampled basis spectra are strongly correlated and the *effective* number
  of channels is far below the number of wavelengths recorded.)
* `segment_of_null` -- underdetermination is not confined to the boundary of the simplex: around
  any strictly positive composition there is a whole segment of nonnegative, normalised
  compositions with identical spectra.
* `two_wavelength_witness` -- concretely, at two wavelengths with a four-class basis of realistic
  ellipticities, two compositions differing by 13 percentage points of β-sheet give the *same*
  spectrum.
* `swapDir_isNull`, `equal_basis_split_free` -- the sharpest case for a disordered region: the
  polyproline II and statistical-coil reference spectra are nearly identical, and if they were
  exactly identical the PPII/coil split would be completely free — any division of their combined
  weight fits equally well.
* `near_degenerate_tolerance` -- since they are only nearly identical, the split is bounded by the
  noise and not by the physics: if two basis spectra differ by at most `ε` at every wavelength,
  moving weight `t` between them perturbs the data by at most `|t|·ε`, so a data tolerance `η`
  admits every split with `|t| ≤ η/ε`.
* `readout_determined` -- what *is* determined: a weighted composition `∑ⱼ wⱼ fⱼ` is the same for
  every fitting composition as soon as the weight vector `w` is itself an affine read-out of the
  basis spectra (`wⱼ = c₀ + ∑ᵢ cᵢ Bⱼᵢ`).  Secondary-structure content is identifiable exactly to
  the extent that it is a linear functional of the measured spectrum, and no further.

Design consequence, in the same shape as Parts XXXI and XXXV: a model of a disordered region
should predict the spectrum and be compared with the spectrum.  Fitted helix/sheet/PPII/coil
percentages are a projection of the data onto a chosen basis, and the components of that
projection that lie in the null space of the measurement are chosen by the regulariser, not by
the protein.
-/
import Mathlib

set_option autoImplicit false

namespace CD

variable {k m : ℕ}

/-- The forward model of a circular dichroism deconvolution: the spectrum produced at the `m`
sampled wavelengths by the composition `f` over `k` reference classes with basis spectra `B`. -/
def mix (B : Fin k → Fin m → ℝ) (f : Fin k → ℝ) : Fin m → ℝ := fun i => ∑ j, f j * B j i

/-- A *null direction* of the basis `B`: a change of composition that alters neither the total
weight nor the predicted spectrum. -/
def IsNull (B : Fin k → Fin m → ℝ) (d : Fin k → ℝ) : Prop :=
  (∑ j, d j = 0) ∧ ∀ i, ∑ j, d j * B j i = 0

lemma mix_add (B : Fin k → Fin m → ℝ) (f g : Fin k → ℝ) :
    mix B (f + g) = mix B f + mix B g := by
  funext i
  simp [mix, add_mul, Finset.sum_add_distrib]

lemma mix_smul (B : Fin k → Fin m → ℝ) (t : ℝ) (f : Fin k → ℝ) :
    mix B (t • f) = t • mix B f := by
  funext i
  simp [mix, Finset.mul_sum, mul_assoc]

/-- Along a null direction the predicted spectrum is exactly constant. -/
lemma mix_perturb {B : Fin k → Fin m → ℝ} {d : Fin k → ℝ} (hd : IsNull B d)
    (f : Fin k → ℝ) (t : ℝ) : mix B (f + t • d) = mix B f := by
  funext i
  have : ∑ j, (t * d j) * B j i = 0 := by
    have := hd.2 i
    calc ∑ j, (t * d j) * B j i = t * ∑ j, d j * B j i := by
            rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by ring
      _ = 0 := by rw [this]; ring
  simp only [mix, Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_mul, Finset.sum_add_distrib, this,
    add_zero]

/-- Along a null direction the total weight is exactly constant. -/
lemma sum_perturb {B : Fin k → Fin m → ℝ} {d : Fin k → ℝ} (hd : IsNull B d)
    (f : Fin k → ℝ) (t : ℝ) : ∑ j, (f + t • d) j = ∑ j, f j := by
  have : ∑ j, (t * d j) = 0 := by rw [← Finset.mul_sum, hd.1, mul_zero]
  simp [Finset.sum_add_distrib, this]

/-! ### Counting: fewer channels than classes -/

/-- **Deconvolution is underdetermined when the classes outnumber the channels.**  If the number
of wavelengths `m` satisfies `m + 1 < k`, there is a nonzero null direction: a change of
composition invisible both to the normalisation and to every measured wavelength. -/
theorem exists_null (B : Fin k → Fin m → ℝ) (h : m + 1 < k) :
    ∃ d : Fin k → ℝ, d ≠ 0 ∧ IsNull B d := by
  let L : (Fin k → ℝ) →ₗ[ℝ] (ℝ × (Fin m → ℝ)) :=
    { toFun := fun d => (∑ j, d j, mix B d)
      map_add' := fun x y => by
        refine Prod.ext ?_ ?_
        · simp [Finset.sum_add_distrib]
        · simpa using mix_add B x y
      map_smul' := fun c x => by
        refine Prod.ext ?_ ?_
        · simp [Finset.mul_sum]
        · simpa using mix_smul B c x }
  have hdim : Module.finrank ℝ (ℝ × (Fin m → ℝ)) < Module.finrank ℝ (Fin k → ℝ) := by
    simpa [Module.finrank_prod, Module.finrank_pi, add_comm] using h
  have hker : LinearMap.ker L ≠ ⊥ := by
    intro hbot
    have hinj : Function.Injective L := LinearMap.ker_eq_bot.mp hbot
    have := LinearMap.finrank_le_finrank_of_injective (f := L) hinj
    omega
  obtain ⟨d, hdmem, hdne⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  refine ⟨d, hdne, ?_, ?_⟩
  · have : L d = 0 := hdmem
    have := congrArg Prod.fst this
    simpa [L] using this
  · intro i
    have : L d = 0 := hdmem
    have h2 := congrArg Prod.snd this
    have : mix B d = 0 := by simpa [L] using h2
    have := congrFun this i
    simpa [mix] using this

/-! ### A segment of fitting compositions -/

/-- **Underdetermination in the interior.**  Around any strictly positive composition, a nonzero
null direction produces a genuinely different composition that is still nonnegative, has the same
total weight, and predicts exactly the same spectrum. -/
theorem segment_of_null {B : Fin k → Fin m → ℝ} {d f : Fin k → ℝ} (hd : d ≠ 0)
    (hnull : IsNull B d) (hf : ∀ j, 0 < f j) :
    ∃ t : ℝ, 0 < t ∧ (∀ j, 0 ≤ (f + t • d) j) ∧ (∑ j, (f + t • d) j = ∑ j, f j) ∧
      mix B (f + t • d) = mix B f ∧ f + t • d ≠ f := by
  obtain ⟨j₀, hj₀⟩ : ∃ j, d j ≠ 0 := by
    by_contra hc
    push_neg at hc
    exact hd (funext fun j => hc j)
  have hne : (Finset.univ : Finset (Fin k)).Nonempty := ⟨j₀, Finset.mem_univ _⟩
  set M : ℝ := 1 + ∑ j, |d j| with hM
  have hMpos : 0 < M := by
    have : (0:ℝ) ≤ ∑ j, |d j| := Finset.sum_nonneg fun j _ => abs_nonneg _
    linarith
  set c : ℝ := Finset.univ.inf' hne f with hc
  have hcpos : 0 < c := (Finset.lt_inf'_iff hne).mpr fun j _ => hf j
  refine ⟨c / M, div_pos hcpos hMpos, ?_, sum_perturb hnull f _, mix_perturb hnull f _, ?_⟩
  · intro j
    have hcj : c ≤ f j := Finset.inf'_le _ (Finset.mem_univ j)
    have hdj : |d j| ≤ ∑ j, |d j| :=
      Finset.single_le_sum (f := fun j => |d j|) (fun _ _ => abs_nonneg _) (Finset.mem_univ j)
    have hkey : (c / M) * |d j| ≤ c := by
      have h1 : (c / M) * |d j| ≤ (c / M) * (∑ j, |d j|) := by gcongr
      have h2 : (c / M) * (∑ j, |d j|) ≤ c := by
        rw [div_mul_eq_mul_div, div_le_iff₀ hMpos]
        nlinarith [hcpos.le]
      linarith
    have hlb : -((c / M) * |d j|) ≤ (c / M) * d j := by
      have := neg_abs_le (d j)
      nlinarith [div_pos hcpos hMpos]
    have : 0 ≤ f j + (c / M) * d j := by
      have := hcj
      linarith
    simpa [Pi.add_apply] using this
  · intro hcontra
    have := congrFun hcontra j₀
    simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_eq_left, mul_eq_zero] at this
    rcases this with h | h
    · exact absurd h (ne_of_gt (div_pos hcpos hMpos))
    · exact hj₀ h

/-! ### A concrete two-wavelength witness -/

/-- A four-class reference basis (helix, sheet, turn, coil) sampled at 222 nm and 208 nm, in units
of 10³ deg cm² dmol⁻¹. -/
noncomputable def Bw : Fin 4 → Fin 2 → ℝ := ![![-33, -33], ![-12, -4], ![-5, -2], ![-2, -8]]

/-- One composition: 30 % helix, 20 % sheet, 20 % turn, 30 % coil. -/
noncomputable def fw : Fin 4 → ℝ := ![3/10, 1/5, 1/5, 3/10]

/-- Another composition: 27.6 % helix, 33.05 % sheet, 1.3 % turn, 38.05 % coil. -/
noncomputable def gw : Fin 4 → ℝ := ![552/2000, 661/2000, 26/2000, 761/2000]

/-- **Two secondary-structure contents, one spectrum.**  With a realistic four-class basis
sampled at two wavelengths, the compositions `fw` and `gw` are both nonnegative and normalised,
differ by more than 13 percentage points of β-sheet, and produce identical spectra. -/
theorem two_wavelength_witness :
    (∀ j, 0 ≤ fw j) ∧ (∀ j, 0 ≤ gw j) ∧ (∑ j, fw j = 1) ∧ (∑ j, gw j = 1) ∧
      mix Bw fw = mix Bw gw ∧ gw 1 - fw 1 > 13/100 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro j; fin_cases j <;> norm_num [fw]
  · intro j; fin_cases j <;> norm_num [gw]
  · norm_num [fw, Fin.sum_univ_succ]
  · norm_num [gw, Fin.sum_univ_succ]
  · funext i
    fin_cases i <;> norm_num [mix, Bw, fw, gw, Fin.sum_univ_succ]
  · norm_num [fw, gw]

/-! ### Two indistinguishable reference spectra -/

/-- The direction that moves unit weight from class `j₁` to class `j₂`. -/
def swapDir (j₁ j₂ : Fin k) : Fin k → ℝ :=
  fun j => (if j = j₁ then (1:ℝ) else 0) - (if j = j₂ then (1:ℝ) else 0)

lemma mix_swapDir (B : Fin k → Fin m → ℝ) (j₁ j₂ : Fin k) (i : Fin m) :
    ∑ j, swapDir j₁ j₂ j * B j i = B j₁ i - B j₂ i := by
  simp [swapDir, sub_mul, Finset.sum_sub_distrib, Finset.sum_ite_eq' Finset.univ]

lemma sum_swapDir (j₁ j₂ : Fin k) : ∑ j, swapDir j₁ j₂ j = 0 := by
  simp [swapDir, Finset.sum_sub_distrib]

/-- If two reference spectra coincide, moving weight between the two classes is a null
direction. -/
theorem swapDir_isNull {B : Fin k → Fin m → ℝ} {j₁ j₂ : Fin k} (heq : B j₁ = B j₂) :
    IsNull B (swapDir j₁ j₂) := by
  refine ⟨sum_swapDir j₁ j₂, fun i => ?_⟩
  rw [mix_swapDir, heq, sub_self]

/-- **An exactly degenerate pair is a free parameter.**  If the PPII and coil reference spectra
coincide, every division of their combined weight fits the data equally well: the whole family
`f + t·(e_{j₁} − e_{j₂})` has the same total and the same spectrum, for every `t`. -/
theorem equal_basis_split_free {B : Fin k → Fin m → ℝ} {j₁ j₂ : Fin k} (hne : j₁ ≠ j₂)
    (heq : B j₁ = B j₂) (f : Fin k → ℝ) (t : ℝ) :
    mix B (f + t • swapDir j₁ j₂) = mix B f ∧
      (∑ j, (f + t • swapDir j₁ j₂) j = ∑ j, f j) ∧
      (f + t • swapDir j₁ j₂) j₁ = f j₁ + t := by
  refine ⟨mix_perturb (swapDir_isNull heq) f t, sum_perturb (swapDir_isNull heq) f t, ?_⟩
  simp [swapDir, hne]

/-- **A nearly degenerate pair is limited by the noise, not by the physics.**  If two reference
spectra differ by at most `ε` at every wavelength, then moving weight `t` between the two classes
changes the predicted spectrum by at most `|t|·ε`; so every split with `|t| ≤ η/ε` is admitted by
a data tolerance `η`. -/
theorem near_degenerate_tolerance {B : Fin k → Fin m → ℝ} {j₁ j₂ : Fin k} {eps : ℝ}
    (heps : ∀ i, |B j₁ i - B j₂ i| ≤ eps) (f : Fin k → ℝ) (t : ℝ) (i : Fin m) :
    |mix B (f + t • swapDir j₁ j₂) i - mix B f i| ≤ |t| * eps := by
  have hsplit : mix B (f + t • swapDir j₁ j₂) i - mix B f i = t * (B j₁ i - B j₂ i) := by
    simp only [mix, Pi.add_apply, Pi.smul_apply, smul_eq_mul, add_mul, Finset.sum_add_distrib]
    have : ∑ j, t * swapDir j₁ j₂ j * B j i = t * ∑ j, swapDir j₁ j₂ j * B j i := by
      rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun j _ => by ring
    rw [this, mix_swapDir]
    ring
  rw [hsplit, abs_mul]
  exact mul_le_mul_of_nonneg_left (heps i) (abs_nonneg t)

/-! ### What the spectrum does determine -/

/-- **The identifiable part of a composition.**  If a weight vector `w` is an affine read-out of
the basis spectra, `wⱼ = c₀ + ∑ᵢ cᵢ Bⱼᵢ`, then the weighted content `∑ⱼ wⱼ fⱼ` is the same for
any two compositions with the same total weight and the same predicted spectrum.  Secondary
structure content is determined exactly insofar as it is a linear functional of the data. -/
theorem readout_determined {B : Fin k → Fin m → ℝ} {w : Fin k → ℝ} {c : Fin m → ℝ} {c0 : ℝ}
    (hw : ∀ j, w j = c0 + ∑ i, c i * B j i) {f g : Fin k → ℝ}
    (hsum : ∑ j, f j = ∑ j, g j) (hmix : mix B f = mix B g) :
    ∑ j, w j * f j = ∑ j, w j * g j := by
  have key : ∀ h : Fin k → ℝ,
      ∑ j, w j * h j = c0 * (∑ j, h j) + ∑ i, c i * mix B h i := by
    intro h
    have h1 : ∑ j, w j * h j = ∑ j, (c0 * h j + ∑ i, c i * B j i * h j) := by
      refine Finset.sum_congr rfl fun j _ => ?_
      rw [hw j, add_mul, Finset.sum_mul]
    rw [h1, Finset.sum_add_distrib, ← Finset.mul_sum]
    congr 1
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [mix, Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [key f, key g, hsum, hmix]

end CD
