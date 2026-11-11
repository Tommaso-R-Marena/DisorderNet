/-
# Part XLVII  Which moment does the experiment weigh?

`RequestProject.SaxsFret` compares the two experiments that dominate the structural
characterisation of disordered regions.  Scattering reports a second moment of the distance
distribution; single-molecule FRET reports the average of `R₀⁶/(R₀⁶ + r⁶)`, which is dominated by
the short distances.  The file derives exactly what each measures and how far apart they can be.

`IDR.saxs_fret_laws` bundles five statements:

1. the forward model is monotone -- the transfer efficiency decreases with distance and lies in
   `(0, 1]`;
2. **the inversion is biased**: the distance obtained by solving `eff R₀ d = E` for the measured
   `E` never exceeds the sixth-moment mean distance, so a heterogeneous ensemble is always
   reported as more compact than it is;
3. and the bias is exactly a width effect: it vanishes for a homogeneous ensemble, and in
   general the inferred distance lies between the smallest and largest distances present;
4. the second and sixth moments are ordered (`⟨r²⟩³ ≤ ⟨r⁶⟩`), which is the only general relation
   between the two experiments;
5. and it is only an inequality: two two-state ensembles with the same mean square distance have
   efficiencies `1/2` and `5/9`, and two with the same efficiency `1/2` have different mean
   square distances.  Neither experiment determines the other.
-/
import Mathlib
import RequestProject.SaxsFret

set_option autoImplicit false

namespace IDR

open IDR.SaxsFret

/-- **The scattering/FRET laws for a labelled disordered region.**

1. *Forward model*: `eff` is positive, at most `1`, and decreasing in the distance.
2. *Biased inversion*: `apparentSixth R₀ (meanEff R₀ w r) ≤ meanSixth w r`.
3. *A width effect*: exact on a homogeneous ensemble, and bracketed by the extreme distances in
   general.
4. *Power-mean ordering*: `meanSq³ ≤ meanSixth`.
5. *Mutual blindness*: explicit two-state ensembles with equal second moment and different
   efficiencies, and with equal efficiency and different second moments. -/
theorem saxs_fret_laws :
    (∀ (R0 r s : ℝ), 0 < R0 → 0 ≤ r → r ≤ s →
        0 < eff R0 r ∧ eff R0 r ≤ 1 ∧ eff R0 s ≤ eff R0 r) ∧
    (∀ (n : ℕ) (R0 : ℝ) (w r : Fin n → ℝ), 0 < R0 → (∀ i, 0 < w i) → ∑ i, w i = 1 →
        apparentSixth R0 (meanEff R0 w r) ≤ meanSixth w r) ∧
    ((∀ (n : ℕ) (R0 s : ℝ) (w r : Fin n → ℝ), 0 < R0 → ∑ i, w i = 1 → (∀ i, r i = s) →
        apparentSixth R0 (meanEff R0 w r) = s ^ 6) ∧
      (∀ (n : ℕ) (R0 a b : ℝ) (w r : Fin n → ℝ), 0 < R0 → (∀ i, 0 < w i) → ∑ i, w i = 1 →
        0 < n → 0 ≤ a → (∀ i, a ≤ r i ∧ r i ≤ b) →
          a ^ 6 ≤ apparentSixth R0 (meanEff R0 w r) ∧
            apparentSixth R0 (meanEff R0 w r) ≤ b ^ 6)) ∧
    (∀ (n : ℕ) (w r : Fin n → ℝ), (∀ i, 0 ≤ w i) → ∑ i, w i = 1 →
        meanSq w r ^ 3 ≤ meanSixth w r) ∧
    ((meanSq half ![1, 1] = meanSq half ![0, Real.sqrt 2] ∧
        meanEff 1 half ![1, 1] = 1/2 ∧ meanEff 1 half ![0, Real.sqrt 2] = 5/9) ∧
      (meanEff 1 half ![1, 1] = meanEff 1 half ![Real.sqrt tcube, Real.sqrt tcube⁻¹] ∧
        meanSq half ![1, 1] < meanSq half ![Real.sqrt tcube, Real.sqrt tcube⁻¹])) := by
  exact ⟨fun R0 r s hR hr hrs => ⟨eff_pos hR, eff_le_one hR, eff_antitone hR hr hrs⟩,
    fun n R0 w r hR hw hsum => apparentSixth_le_meanSixth hR hw hsum,
    ⟨fun n R0 s w r hR hsum hconst => apparentSixth_eq_of_homogeneous hR hsum hconst,
      fun n R0 a b w r hR hw hsum hn ha hab => apparentSixth_mem_Icc hR hw hsum hn ha hab⟩,
    fun n w r hw hsum => meanSq_cube_le_meanSixth hw hsum,
    ⟨saxs_blind_to_fret, fret_blind_to_saxs⟩⟩

end IDR
