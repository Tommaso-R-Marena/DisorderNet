/-
# Part LXXXVIII  Conditioning: what a regularised ensemble fit actually reports

Fitting an ensemble to data is an inverse problem, and every practical fit is regularised — by a
maximum-entropy prior, by a Tikhonov penalty, by an early stop, by a pool of conformations chosen in
advance.  Regularisation is usually described as a technical necessity.  It is in fact an exact
trade, and this file computes it.

Diagonalise the measurement: a conformational mode with true amplitude `c` is seen by the
experiment with sensitivity `s ≥ 0`, so the datum is `d = s·c + n` with noise `n`, and the
regularised reconstruction at penalty `lam > 0` is `recon = s·d/(s² + lam)` — the Tikhonov solution
of `min ‖s·x − d‖² + lam·‖x‖²`, written mode by mode.

* `recon_eq_filt_add_noise` — the reconstruction is exactly `filt·c + gain·n`, with *filter factor*
  `filt = s²/(s² + lam)` and *noise gain* `gain = s/(s² + lam)`.
* `filt_nonneg`, `filt_lt_one`, `filt_mono_sens`, `filt_anti_pen` — the filter factor is in `[0,1)`,
  increasing in the sensitivity and decreasing in the penalty: **regularisation is a systematic
  shrinkage of every mode towards the prior**, never a neutral technicality.
* `gain_le_inv_two_sqrt`, `gain_eq_at_sqrt` — the noise gain is at most `1/(2√lam)`, attained at
  `s = √lam`: the penalty buys a hard, `s`-independent stability bound.
* `gain_unbounded_of_no_penalty` — and it is needed: without a penalty the gain is `1/s`, which is
  unbounded as the sensitivity falls, so the unregularised fit amplifies noise without limit.
* `stability_bias_identity` — the exact exchange rate: `s·(1 − filt) = lam·gain`.  Stability and
  bias are the same quantity read in two directions; one cannot be improved without paying the
  other.
* `filt_ge_half_iff`, `prior_dominates_of_insensitive` — the resolution boundary is `s² = lam`,
  known before any data are seen.  Above it the datum contributes more than the prior; below it the
  reported amplitude is the prior's, and `recon_le_of_insensitive` bounds by `eps·|c|` how much of a
  mode with `s² ≤ eps·lam` survives at all.
* `error_le` — the total error of the reported amplitude is at most
  `lam·|c|/(s² + lam) + |n|/(2√lam)`, bias plus noise, both terms explicit.
* `resolvedModes`, `card_resolvedModes_le`, `unresolved_report_prior` — across a spectrum of modes,
  the ones with `s² ≥ lam` are the ones the fit reports from the data; on the rest it reports the
  prior, whatever the data say.  That count, not the number of fitted parameters, is the number of
  numbers a regularised ensemble fit has actually measured.

Design consequence: the penalty is part of the claim.  A regularised ensemble model should quote
`lam`, the sensitivities of the modes it reports, and hence which of its reported features are
data-driven and which are the prior seen through the fit.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset

namespace Conditioning

/-- Tikhonov filter factor of a mode with sensitivity `s` at penalty `lam`. -/
noncomputable def filt (s lam : ℝ) : ℝ := s ^ 2 / (s ^ 2 + lam)

/-- Noise gain of a mode with sensitivity `s` at penalty `lam`. -/
noncomputable def gain (s lam : ℝ) : ℝ := s / (s ^ 2 + lam)

/-- The regularised reconstruction of a mode from its datum. -/
noncomputable def recon (s lam d : ℝ) : ℝ := s * d / (s ^ 2 + lam)

theorem denom_pos {s lam : ℝ} (hlam : 0 < lam) : 0 < s ^ 2 + lam := by positivity

/-! ## The reconstruction is a shrinkage plus amplified noise -/

/-- **What the fit returns.**  With datum `d = s·c + n`, the reconstruction is exactly
`filt·c + gain·n`. -/
theorem recon_eq_filt_add_noise (s c n lam : ℝ) (hlam : 0 < lam) :
    recon s lam (s * c + n) = filt s lam * c + gain s lam * n := by
  have h := (denom_pos (s := s) hlam).ne'
  simp only [recon, filt, gain]
  field_simp

theorem filt_nonneg (s lam : ℝ) (hlam : 0 < lam) : 0 ≤ filt s lam :=
  div_nonneg (sq_nonneg s) (denom_pos (s := s) hlam).le

/-- **Regularisation shrinks every mode.** -/
theorem filt_lt_one (s lam : ℝ) (hlam : 0 < lam) : filt s lam < 1 := by
  have h := denom_pos (s := s) hlam
  rw [filt, div_lt_one h]
  linarith

theorem filt_mono_sens {s t lam : ℝ} (hlam : 0 < lam) (hs : 0 ≤ s) (hst : s ≤ t) :
    filt s lam ≤ filt t lam := by
  have hs2 := denom_pos (s := s) hlam
  have ht2 := denom_pos (s := t) hlam
  have hst2 : s ^ 2 ≤ t ^ 2 := by nlinarith
  simp only [filt]
  rw [div_le_div_iff₀ hs2 ht2]
  nlinarith [mul_nonneg hlam.le (sub_nonneg.mpr hst2)]

theorem filt_anti_pen {s lam mu : ℝ} (hlam : 0 < lam) (hmu : lam ≤ mu) :
    filt s mu ≤ filt s lam := by
  have hs2 := denom_pos (s := s) hlam
  have ht2 : (0 : ℝ) < s ^ 2 + mu := by linarith
  simp only [filt]
  rw [div_le_div_iff₀ ht2 hs2]
  nlinarith [sq_nonneg s]

/-! ## Stability bought, and its price -/

/-- **The penalty caps the noise gain**, uniformly in the sensitivity. -/
theorem gain_le_inv_two_sqrt {s lam : ℝ} (hlam : 0 < lam) :
    gain s lam ≤ 1 / (2 * Real.sqrt lam) := by
  have hsq : Real.sqrt lam > 0 := Real.sqrt_pos.mpr hlam
  have hsq2 : Real.sqrt lam ^ 2 = lam := Real.sq_sqrt hlam.le
  have hd := denom_pos (s := s) hlam
  simp only [gain]
  rw [div_le_div_iff₀ hd (by positivity)]
  nlinarith [sq_nonneg (s - Real.sqrt lam)]

/-- …and the cap is attained, so it is the right constant. -/
theorem gain_eq_at_sqrt {lam : ℝ} (hlam : 0 < lam) :
    gain (Real.sqrt lam) lam = 1 / (2 * Real.sqrt lam) := by
  have hsq : Real.sqrt lam > 0 := Real.sqrt_pos.mpr hlam
  have hsq2 : Real.sqrt lam ^ 2 = lam := Real.sq_sqrt hlam.le
  simp only [gain, hsq2]
  field_simp
  rw [hsq2]
  ring

/-- **Without a penalty there is no cap.**  The unregularised gain `1/s` exceeds any bound at small
enough sensitivity. -/
theorem gain_unbounded_of_no_penalty (M : ℝ) : ∃ s : ℝ, 0 < s ∧ M < gain s 0 := by
  refine ⟨1 / (max M 1 + 1), by positivity, ?_⟩
  have h1 : (0 : ℝ) < max M 1 + 1 := by positivity
  have hs : gain (1 / (max M 1 + 1)) 0 = max M 1 + 1 := by
    simp only [gain, add_zero]
    rw [div_pow, one_pow, one_div, one_div, inv_div_inv]
    field_simp
  rw [hs]
  have : M ≤ max M 1 := le_max_left _ _
  linarith

/-- **The exact exchange rate between stability and bias.** -/
theorem stability_bias_identity (s lam : ℝ) (hlam : 0 < lam) :
    s * (1 - filt s lam) = lam * gain s lam := by
  have h := (denom_pos (s := s) hlam).ne'
  simp only [filt, gain]
  field_simp
  ring

/-! ## The resolution boundary -/

/-- **The boundary is at `s² = lam`**: the datum outweighs the prior exactly above it. -/
theorem filt_ge_half_iff {s lam : ℝ} (hlam : 0 < lam) : 1 / 2 ≤ filt s lam ↔ lam ≤ s ^ 2 := by
  have h := denom_pos (s := s) hlam
  simp only [filt]
  rw [le_div_iff₀ h]
  constructor <;> intro hx <;> linarith

/-- Below the boundary the reported amplitude is dominated by the prior. -/
theorem prior_dominates_of_insensitive {s lam : ℝ} (hlam : 0 < lam) (h : s ^ 2 < lam) :
    filt s lam < 1 / 2 := by
  by_contra hcon
  push_neg at hcon
  exact absurd ((filt_ge_half_iff hlam).mp hcon) (not_le.mpr h)

/-- **A mode the experiment barely sees is returned at the prior**, whatever the data say: its
data-driven part is at most `eps·|c|`. -/
theorem recon_le_of_insensitive {s lam eps c : ℝ} (hlam : 0 < lam) (heps : 0 ≤ eps)
    (h : s ^ 2 ≤ eps * lam) : filt s lam * |c| ≤ eps * |c| := by
  have hd := denom_pos (s := s) hlam
  have hfilt : filt s lam ≤ eps := by
    simp only [filt]
    rw [div_le_iff₀ hd]
    nlinarith [sq_nonneg s]
  exact mul_le_mul_of_nonneg_right hfilt (abs_nonneg c)

/-- **The total error of a reported amplitude**: bias plus amplified noise, both explicit. -/
theorem error_le {s c n lam : ℝ} (hs : 0 ≤ s) (hlam : 0 < lam) :
    |recon s lam (s * c + n) - c| ≤ lam * |c| / (s ^ 2 + lam) + |n| / (2 * Real.sqrt lam) := by
  have hd := denom_pos (s := s) hlam
  have hrec := recon_eq_filt_add_noise s c n lam hlam
  have hbias : filt s lam * c - c = -(lam / (s ^ 2 + lam)) * c := by
    simp only [filt]
    field_simp
    ring
  have hsplit : recon s lam (s * c + n) - c = -(lam / (s ^ 2 + lam)) * c + gain s lam * n := by
    rw [hrec, ← hbias]; ring
  rw [hsplit]
  refine (abs_add_le _ _).trans ?_
  have h1 : |-(lam / (s ^ 2 + lam)) * c| = lam * |c| / (s ^ 2 + lam) := by
    rw [abs_mul, abs_neg, abs_of_nonneg (by positivity : (0:ℝ) ≤ lam / (s ^ 2 + lam))]
    field_simp
  have h2 : |gain s lam * n| ≤ |n| / (2 * Real.sqrt lam) := by
    have hgn : 0 ≤ gain s lam := div_nonneg hs hd.le
    rw [abs_mul, abs_of_nonneg hgn]
    have hg := gain_le_inv_two_sqrt (s := s) hlam
    calc gain s lam * |n| ≤ (1 / (2 * Real.sqrt lam)) * |n| :=
          mul_le_mul_of_nonneg_right hg (abs_nonneg n)
      _ = |n| / (2 * Real.sqrt lam) := by ring
  linarith [h1 ▸ le_refl (lam * |c| / (s ^ 2 + lam))]

/-! ## Across a spectrum of modes -/

variable {K : ℕ}

/-- The modes a fit at penalty `lam` reports from the data rather than from the prior. -/
noncomputable def resolvedModes (sens : Fin K → ℝ) (lam : ℝ) : Finset (Fin K) :=
  univ.filter fun i => lam ≤ sens i ^ 2

/-- Raising the penalty can only shrink the set of modes the data control. -/
theorem resolvedModes_anti (sens : Fin K → ℝ) {lam mu : ℝ} (h : lam ≤ mu) :
    resolvedModes sens mu ⊆ resolvedModes sens lam := by
  intro i hi
  simp only [resolvedModes, mem_filter, mem_univ, true_and] at hi ⊢
  linarith

theorem card_resolvedModes_le (sens : Fin K → ℝ) (lam : ℝ) :
    (resolvedModes sens lam).card ≤ K := by
  simpa using Finset.card_filter_le (univ : Finset (Fin K)) fun i => lam ≤ sens i ^ 2

/-- **On every unresolved mode the fit reports the prior**, in the precise sense that the datum
contributes less than half of the reported amplitude. -/
theorem unresolved_report_prior (sens : Fin K → ℝ) {lam : ℝ} (hlam : 0 < lam) {i : Fin K}
    (hi : i ∉ resolvedModes sens lam) : filt (sens i) lam < 1 / 2 := by
  have h : ¬ lam ≤ sens i ^ 2 := by
    simpa [resolvedModes, mem_filter] using hi
  exact prior_dominates_of_insensitive hlam (lt_of_not_ge h)

end Conditioning

end IDR
