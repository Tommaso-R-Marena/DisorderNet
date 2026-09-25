/-
# Part CXLIX  Why FRET calls a disordered region more compact than scattering does

Part XLVII.1 established the only *general* relation between the two experiments that dominate the
characterisation of disordered regions: the second and sixth moments of the distance distribution
are ordered, `⟨r²⟩³ ≤ ⟨r⁶⟩`, and each experiment is blind to what the other sees.  The persistent
empirical observation is stronger and directional: for the same chain, single-molecule FRET
typically reports a *smaller* size than small-angle scattering.  This part proves that this is a
theorem about the ensemble, not an artefact — under an explicit and physically meaningful
hypothesis, and with an explicit counterexample showing the hypothesis cannot be dropped.

The mechanism is the curvature of the transfer efficiency as a function of the *squared* distance
`u = r²`, which is the variable scattering averages:

`E = A/(A + u³)`,  `A = R₀⁶`,

whose second derivative changes sign at `2u³ = A`, i.e. at `r = R₀·2^{−1/6} ≈ 0.89 R₀`.  Above that
distance `E` is a convex function of `u`, so Jensen's inequality runs one way and one way only.

* `tangent_line_ineq` — the supporting-line inequality for `u ↦ A/(A + u³)` at any point of the
  convex region, proved algebraically: the difference factors as
  `(u − m)²·(m²(3u² + 2mu + m²) − A(u + 2m))` over a positive denominator, and the bracket is
  non-negative once `A ≤ 2m³` and `A ≤ 2u³`.  No differentiation, no limits.

* `mean_in_convex_region` — an ensemble all of whose conformations are in the convex region has its
  mean square distance there too, so the supporting line may be taken at the measured value.

* `jensen_expanded` — hence Jensen: `A/(A + ⟨u⟩³) ≤ ⟨A/(A + u³)⟩`.

* `saxs_fret_ordering` — **the theorem.**  For an ensemble whose donor–acceptor distances all
  exceed `R₀·2^{−1/6}`, the measured mean transfer efficiency is at least the efficiency of a
  single conformation at the scattering-derived root-mean-square distance.  Equivalently
  (`apparent_le_rms_sixth`) the FRET-apparent distance never exceeds the root-mean-square distance:
  **FRET reports the more compact size, always, in the expanded regime.**

* `apparent_le_rms_le_sixth` — and this is strictly more informative than what Part XLVII.1 gave:
  the apparent sixth power is squeezed below `⟨r²⟩³`, which is itself below `⟨r⁶⟩`.

* `ordering_needs_expanded_regime` — the hypothesis is necessary, not technical.  An explicit
  two-state ensemble with a collapsed member (`R₀ = 1`, distances `0` and `1`, equal weights) has
  mean efficiency `3/4`, strictly below the efficiency `8/9` of its own root-mean-square distance.
  A region with a populated collapsed state can therefore look *larger* by FRET than by scattering,
  which is exactly the regime where the two experiments are reported to disagree in sign.

Design consequence: the direction of the FRET/SAXS discrepancy is diagnostic.  Under the expanded
hypothesis the ordering is forced, so an observed FRET size above the scattering size is not a
calibration problem — it is evidence for populated compact conformations, and a model of the region
must contain them.
-/
import Mathlib
import RequestProject.SaxsFret

set_option autoImplicit false

namespace IDR
namespace FretSaxsOrdering

open Finset IDR.SaxsFret

variable {n : ℕ}

/-- The bracket in the supporting-line inequality is non-negative throughout the convex region. -/
theorem convex_region_bracket {A u m : ℝ} (hA : 0 < A) (hu : 0 ≤ u) (hm : 0 ≤ m)
    (h1 : A ≤ 2 * m ^ 3) (h2 : A ≤ 2 * u ^ 3) :
    A * (u + 2 * m) ≤ m ^ 2 * (3 * u ^ 2 + 2 * m * u + m ^ 2) := by
  rcases le_total u m with h | h
  · nlinarith [sq_nonneg (u - m), sq_nonneg (u + m), mul_nonneg hu hm]
  · nlinarith [sq_nonneg (u - m), sq_nonneg (u + m), mul_nonneg hu hm]

/-- **The supporting-line inequality.**  On the convex region `2u³ ≥ A` the graph of
`u ↦ A/(A + u³)` lies above its tangent line at any point `m` of the region. -/
theorem tangent_line_ineq {A u m : ℝ} (hA : 0 < A) (hu : 0 ≤ u) (hm : 0 ≤ m)
    (h1 : A ≤ 2 * m ^ 3) (h2 : A ≤ 2 * u ^ 3) :
    A / (A + m ^ 3) - 3 * A * m ^ 2 / (A + m ^ 3) ^ 2 * (u - m) ≤ A / (A + u ^ 3) := by
  have hdu : 0 < A + u ^ 3 := by positivity
  have hdm : 0 < A + m ^ 3 := by positivity
  have hbr := convex_region_bracket hA hu hm h1 h2
  rw [← sub_nonneg]
  have key : A / (A + u ^ 3) - (A / (A + m ^ 3) - 3 * A * m ^ 2 / (A + m ^ 3) ^ 2 * (u - m))
      = A * ((u - m) ^ 2 * (m ^ 2 * (3 * u ^ 2 + 2 * m * u + m ^ 2) - A * (u + 2 * m)))
        / ((A + u ^ 3) * (A + m ^ 3) ^ 2) := by
    field_simp
    ring
  rw [key]
  apply div_nonneg
  · exact mul_nonneg hA.le (mul_nonneg (sq_nonneg _) (by linarith))
  · positivity

/-- An ensemble living entirely in the convex region has its mean there too. -/
theorem mean_in_convex_region {A : ℝ} {w u : Fin n → ℝ} (hA : 0 < A) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hu : ∀ j, 0 ≤ u j) (hconv : ∀ j, A ≤ 2 * u j ^ 3) :
    A ≤ 2 * (∑ j, w j * u j) ^ 3 := by
  set c : ℝ := (A / 2) ^ ((1:ℝ)/3) with hc
  have hA2 : 0 ≤ A / 2 := by linarith
  have hc0 : 0 ≤ c := Real.rpow_nonneg hA2 _
  have hc3 : c ^ 3 = A / 2 := by
    rw [hc, ← Real.rpow_natCast ((A / 2) ^ ((1:ℝ)/3)) 3, ← Real.rpow_mul hA2]
    norm_num
  have hcu : ∀ j, c ≤ u j := by
    intro j
    by_contra hlt
    push_neg at hlt
    have : u j ^ 3 < c ^ 3 := by
      have := pow_lt_pow_left₀ hlt (hu j) (n := 3) (by norm_num)
      simpa using this
    rw [hc3] at this
    linarith [hconv j]
  have hmc : c ≤ ∑ j, w j * u j := by
    calc c = ∑ j, w j * c := by rw [← Finset.sum_mul, hsum, one_mul]
      _ ≤ ∑ j, w j * u j := Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hcu j) (hw j)
  have hpow : c ^ 3 ≤ (∑ j, w j * u j) ^ 3 := pow_le_pow_left₀ hc0 hmc 3
  rw [hc3] at hpow
  linarith

/-- **Jensen's inequality in the convex region.**  The ensemble average of `A/(A + u³)` is at least
its value at the ensemble average of `u`. -/
theorem jensen_expanded {A : ℝ} {w u : Fin n → ℝ} (hA : 0 < A) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hu : ∀ j, 0 ≤ u j) (hconv : ∀ j, A ≤ 2 * u j ^ 3) :
    A / (A + (∑ j, w j * u j) ^ 3) ≤ ∑ j, w j * (A / (A + u j ^ 3)) := by
  set m : ℝ := ∑ j, w j * u j with hm
  have hm0 : 0 ≤ m := Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (hu j)
  have hmc : A ≤ 2 * m ^ 3 := mean_in_convex_region hA hw hsum hu hconv
  have hstep : ∀ j ∈ (Finset.univ : Finset (Fin n)),
      w j * (A / (A + m ^ 3) - 3 * A * m ^ 2 / (A + m ^ 3) ^ 2 * (u j - m))
        ≤ w j * (A / (A + u j ^ 3)) := by
    intro j _
    exact mul_le_mul_of_nonneg_left (tangent_line_ineq hA (hu j) hm0 hmc (hconv j)) (hw j)
  have hsum' := Finset.sum_le_sum hstep
  have hlhs : ∑ j, w j * (A / (A + m ^ 3) - 3 * A * m ^ 2 / (A + m ^ 3) ^ 2 * (u j - m))
      = A / (A + m ^ 3) := by
    have hexp : ∀ j ∈ (Finset.univ : Finset (Fin n)),
        w j * (A / (A + m ^ 3) - 3 * A * m ^ 2 / (A + m ^ 3) ^ 2 * (u j - m))
          = (A / (A + m ^ 3) + 3 * A * m ^ 2 / (A + m ^ 3) ^ 2 * m) * w j
            - (3 * A * m ^ 2 / (A + m ^ 3) ^ 2) * (w j * u j) := fun j _ => by ring
    rw [Finset.sum_congr rfl hexp, Finset.sum_sub_distrib]
    simp only [← Finset.mul_sum]
    rw [hsum, ← hm]
    ring
  rwa [hlhs] at hsum'

/-- **The FRET/SAXS ordering theorem.**  If every conformation of the ensemble has a
donor–acceptor distance in the expanded regime `R₀⁶ ≤ 2r⁶` (that is, `r ≥ R₀·2^{−1/6}`), then the
measured mean transfer efficiency is at least the efficiency of a single conformation sitting at
the root-mean-square distance reported by scattering. -/
theorem saxs_fret_ordering {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hexp : ∀ j, R0 ^ 6 ≤ 2 * r j ^ 6) :
    R0 ^ 6 / (R0 ^ 6 + meanSq w r ^ 3) ≤ meanEff R0 w r := by
  have hA : (0:ℝ) < R0 ^ 6 := by positivity
  have hu : ∀ j, 0 ≤ r j ^ 2 := fun j => sq_nonneg _
  have hconv : ∀ j, R0 ^ 6 ≤ 2 * (r j ^ 2) ^ 3 := by
    intro j
    have : (r j ^ 2) ^ 3 = r j ^ 6 := by ring
    rw [this]
    exact hexp j
  have hj := jensen_expanded (u := fun j => r j ^ 2) hA hw hsum hu hconv
  have h1 : (∑ j, w j * r j ^ 2) = meanSq w r := rfl
  have h2 : (∑ j, w j * (R0 ^ 6 / (R0 ^ 6 + (r j ^ 2) ^ 3))) = meanEff R0 w r := by
    rw [meanEff]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [eff, show (r j ^ 2) ^ 3 = r j ^ 6 from by ring]
  rwa [h1, h2] at hj

/-- **The FRET-apparent distance never exceeds the scattering size.**  Equivalent form of the
ordering theorem, in the sixth powers that avoid roots: the apparent sixth power extracted from the
measured efficiency is at most the cube of the mean square distance. -/
theorem apparent_le_rms_sixth {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hexp : ∀ j, R0 ^ 6 ≤ 2 * r j ^ 6) :
    apparentSixth R0 (meanEff R0 w r) ≤ meanSq w r ^ 3 := by
  have hA : (0:ℝ) < R0 ^ 6 := by positivity
  have hM : 0 ≤ meanSq w r := Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (sq_nonneg _)
  have hden : (0:ℝ) < R0 ^ 6 + meanSq w r ^ 3 := by positivity
  have hEpos : 0 < R0 ^ 6 / (R0 ^ 6 + meanSq w r ^ 3) := div_pos hA hden
  have hle := saxs_fret_ordering hR hw hsum hexp
  have hval : apparentSixth R0 (R0 ^ 6 / (R0 ^ 6 + meanSq w r ^ 3)) = meanSq w r ^ 3 := by
    rw [apparentSixth]
    field_simp
    ring
  have := apparentSixth_antitone (R0 := R0) hR hEpos hle
  rwa [hval] at this

/-- The new bound sits strictly inside the general moment ordering of Part XLVII.1: the apparent
sixth power is below `⟨r²⟩³`, which is below `⟨r⁶⟩`. -/
theorem apparent_le_rms_le_sixth {R0 : ℝ} {w r : Fin n → ℝ} (hR : 0 < R0) (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hexp : ∀ j, R0 ^ 6 ≤ 2 * r j ^ 6) :
    apparentSixth R0 (meanEff R0 w r) ≤ meanSq w r ^ 3 ∧ meanSq w r ^ 3 ≤ meanSixth w r :=
  ⟨apparent_le_rms_sixth hR hw hsum hexp, meanSq_cube_le_meanSixth hw hsum⟩

/-- **The expanded hypothesis cannot be dropped.**  A two-state ensemble with a collapsed member
has a mean transfer efficiency *below* the efficiency of its own root-mean-square distance, so the
ordering reverses: with `R₀ = 1` and distances `0` and `1` at equal weight the mean efficiency is
`3/4` while the root-mean-square efficiency is `8/9`. -/
theorem ordering_needs_expanded_regime :
    ∃ w r : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, 0 ≤ r j) ∧
      meanEff 1 w r < (1:ℝ) ^ 6 / ((1:ℝ) ^ 6 + meanSq w r ^ 3) := by
  refine ⟨![1/2, 1/2], ![0, 1], ?_, ?_, ?_, ?_⟩
  · intro j
    fin_cases j <;> norm_num
  · rw [Fin.sum_univ_two]
    norm_num
  · intro j
    fin_cases j <;> norm_num
  · rw [meanEff, meanSq, Fin.sum_univ_two, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [eff, eff]
    norm_num

end FretSaxsOrdering
end IDR
