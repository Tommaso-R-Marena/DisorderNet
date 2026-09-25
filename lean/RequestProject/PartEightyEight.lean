/-
# Part LXXXVIII  What a regularised ensemble fit actually reports

Capstone for `RequestProject.Conditioning`.

Every practical ensemble fit is regularised — by a maximum-entropy prior, a Tikhonov penalty, an
early stop, or a conformational pool fixed in advance.  Mode by mode the regularised reconstruction
is `recon = s·d/(s² + lam)` for a mode of sensitivity `s` at penalty `lam`, and this part computes
exactly what that returns: a shrinkage of the truth plus a bounded multiple of the noise, with an
exchange rate between the two that cannot be improved, and a resolution boundary at `s² = lam` known
before any data are seen.
-/
import RequestProject.Conditioning

set_option autoImplicit false

namespace IDR

open Conditioning Finset

/-- **The conditioning laws.**  For a mode of sensitivity `s ≥ 0` and true amplitude `c` measured
with noise `n`, reconstructed at penalty `lam > 0`:

1. *what the fit returns*: exactly `filt·c + gain·n`, a shrinkage of the truth plus amplified noise;
2. *shrinkage is unavoidable*: the filter factor lies in `[0, 1)`, so every mode is pulled towards
   the prior, however sensitive the experiment;
3. *the penalty caps the noise*: the gain is at most `1/(2√lam)` whatever the sensitivity, and the
   cap is attained at `s = √lam`, so the constant is right;
4. *and the cap is needed*: without a penalty the gain is `1/s` and exceeds every bound;
5. *the exchange rate*: `s·(1 − filt) = lam·gain` exactly — stability and bias are one quantity,
   and buying more of one sells the other;
6. *the resolution boundary*: the datum outweighs the prior exactly when `s² ≥ lam`;
7. *the total error*: bias plus noise, `lam·|c|/(s² + lam) + |n|/(2√lam)`. -/
theorem conditioning_laws {s c n lam : ℝ} (hs : 0 ≤ s) (hlam : 0 < lam) :
    recon s lam (s * c + n) = filt s lam * c + gain s lam * n ∧
    (0 ≤ filt s lam ∧ filt s lam < 1) ∧
    (gain s lam ≤ 1 / (2 * Real.sqrt lam) ∧
      gain (Real.sqrt lam) lam = 1 / (2 * Real.sqrt lam)) ∧
    (∀ M : ℝ, ∃ t : ℝ, 0 < t ∧ M < gain t 0) ∧
    s * (1 - filt s lam) = lam * gain s lam ∧
    (1 / 2 ≤ filt s lam ↔ lam ≤ s ^ 2) ∧
    |recon s lam (s * c + n) - c| ≤ lam * |c| / (s ^ 2 + lam) + |n| / (2 * Real.sqrt lam) := by
  refine ⟨recon_eq_filt_add_noise s c n lam hlam,
    ⟨filt_nonneg s lam hlam, filt_lt_one s lam hlam⟩,
    ⟨gain_le_inv_two_sqrt hlam, gain_eq_at_sqrt hlam⟩,
    gain_unbounded_of_no_penalty,
    stability_bias_identity s lam hlam,
    filt_ge_half_iff hlam,
    error_le hs hlam⟩

/-- **What a regularised fit has actually measured.**  Across a spectrum of `K` modes with
sensitivities `sens`, the modes with `s² ≥ lam` are those the data control; on every other mode the
reported amplitude is dominated by the prior, whatever the data say, and a mode with `s² ≤ eps·lam`
survives only to the fraction `eps`.  Raising the penalty can only shrink the controlled set, whose
size is at most the number of modes. -/
theorem resolved_modes_laws {K : ℕ} (sens : Fin K → ℝ) {lam : ℝ} (hlam : 0 < lam) :
    ((resolvedModes sens lam).card ≤ K) ∧
    (∀ mu : ℝ, lam ≤ mu → resolvedModes sens mu ⊆ resolvedModes sens lam) ∧
    (∀ i : Fin K, i ∉ resolvedModes sens lam → filt (sens i) lam < 1 / 2) ∧
    (∀ (eps c : ℝ), 0 ≤ eps → ∀ i : Fin K, sens i ^ 2 ≤ eps * lam →
      filt (sens i) lam * |c| ≤ eps * |c|) := by
  refine ⟨card_resolvedModes_le sens lam,
    fun mu h => resolvedModes_anti sens h,
    fun i hi => unresolved_report_prior sens hlam hi,
    fun eps c heps i h => recon_le_of_insensitive hlam heps h⟩

end IDR
