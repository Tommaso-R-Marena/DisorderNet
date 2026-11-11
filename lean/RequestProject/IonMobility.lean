/-
# Part LXXXIII  Ion mobility: an arrival-time distribution is a width, not a shape

An ion-mobility experiment on a disordered region pushes the ion through a drift cell and records
the *arrival-time distribution* (ATD).  Each conformer family `j` contributes a peak centred at its
own drift time `t j` — proportional to its collision cross section — with an instrumental width
`sig j` set by diffusion and by the resolving power of the cell, and the recorded trace is the
`w`-weighted mixture of those peaks.

The file works with the first two moments of that mixture, which is what a fitted ATD reports.

* `meanTime`, `spread`, `width2` — the mixture mean, the *conformational* variance of the drift
  times, and the total measured variance (second raw moment minus the square of the mean).
* `width2_eq_instr_add_spread` — **the law of total variance, in the instrument's units.**  The
  measured squared width is the mean instrumental variance plus the conformational spread, exactly.
* `spread_eq_width2_sub_instr`, `width2_ge_instr` — therefore the conformational spread is
  identified, but *only* as the excess of the measured width over the instrument function; and the
  measured width can never fall below the instrument function.
* `spread_nonneg`, `spread_eq_zero_iff`, `width2_single`, `excess_width_refutes_single` — a model
  committed to one conformation predicts the instrumental width and nothing more, so any excess
  width, however small, refutes every single-structure model.  This is the ion-mobility analogue of
  the sub-stoichiometric crosslink yield.
* `meanTime_le_max`, `min_le_meanTime`, `mixture_hits_intermediate` — the centroid is bracketed by
  the conformer drift times and, conversely, *every* intermediate arrival time is reproduced by a
  two-state mixture of a compact and an extended conformer with weights in `(0,1)`.  A centroid
  alone therefore never selects a structure.
* `moment_ambiguity` — worse: two explicitly different ensembles have the *same* mean and the
  *same* spread.  Mean and width together still do not determine the ensemble; the distribution is
  the datum.
* `resolved_iff`, `resolved_two_percent` — the quantitative price of resolving conformers.  With
  peak widths `t/R` at resolving power `R` and a separation criterion `k·(sig₁+sig₂) ≤ |t₁−t₂|`,
  two families are resolved iff `k·(t₁+t₂) ≤ R·|t₁−t₂|`; two conformers whose cross sections differ
  by two percent are resolved (at `k = 1`) exactly when `R ≥ 101`.
* `card_ge_of_distinct_peaks` — and a resolved ATD is a lower bound on model complexity: `k`
  distinct peaks force at least `k` conformers in any model reproducing them.

Design consequence: an ion-mobility measurement of a disordered region should be reported as a
mean *and* an excess width — a distribution — and a model of the region must predict both.  The
single "measured CCS" that a single structure would predict is the one quantity the experiment does
not deliver.
-/
import Mathlib

namespace IDR

open Finset

namespace IonMob

variable {m : ℕ}

/-! ## Moments of the arrival-time distribution -/

/-- Centroid of the arrival-time distribution: the weight average of the conformer drift times. -/
def meanTime (w t : Fin m → ℝ) : ℝ := ∑ j, w j * t j

/-- Conformational spread: the variance of the drift times across the ensemble. -/
def spread (w t : Fin m → ℝ) : ℝ := ∑ j, w j * (t j - meanTime w t) ^ 2

/-- Mean instrumental variance: the width the trace would have if every conformer had the same
drift time. -/
def instr (w sig : Fin m → ℝ) : ℝ := ∑ j, w j * sig j ^ 2

/-- Total measured squared width of the trace: the second raw moment of the mixture minus the
square of its mean.  Component `j` is a peak of mean `t j` and variance `sig j ^ 2`, so its second
raw moment is `sig j ^ 2 + t j ^ 2`. -/
def width2 (w t sig : Fin m → ℝ) : ℝ :=
  (∑ j, w j * (sig j ^ 2 + t j ^ 2)) - (meanTime w t) ^ 2

/-- The conformational spread is the second raw moment minus the squared mean. -/
theorem spread_eq (w t : Fin m → ℝ) (hw : ∑ j, w j = 1) :
    spread w t = (∑ j, w j * t j ^ 2) - (meanTime w t) ^ 2 := by
  simp only [spread, meanTime]
  set M := ∑ j, w j * t j with hMdef
  have h : ∀ j : Fin m, w j * (t j - M) ^ 2
      = w j * t j ^ 2 - 2 * M * (w j * t j) + M ^ 2 * w j := fun j => by ring
  rw [Finset.sum_congr rfl (fun j _ => h j), Finset.sum_add_distrib, Finset.sum_sub_distrib,
    ← Finset.mul_sum, ← Finset.mul_sum, hw, ← hMdef]
  ring

/-- **Law of total variance.**  The measured squared width of an arrival-time distribution is the
mean instrumental variance plus the conformational spread of the drift times. -/
theorem width2_eq_instr_add_spread (w t sig : Fin m → ℝ) (hw : ∑ j, w j = 1) :
    width2 w t sig = instr w sig + spread w t := by
  have hsplit : ∑ j, w j * (sig j ^ 2 + t j ^ 2)
      = (∑ j, w j * sig j ^ 2) + ∑ j, w j * t j ^ 2 := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  rw [width2, hsplit, spread_eq w t hw, instr]
  ring

/-- **Deconvolution.**  The conformational spread is exactly the excess of the measured width over
the instrument function — no more and no less can be read off an ATD's second moment. -/
theorem spread_eq_width2_sub_instr (w t sig : Fin m → ℝ) (hw : ∑ j, w j = 1) :
    spread w t = width2 w t sig - instr w sig := by
  rw [width2_eq_instr_add_spread w t sig hw]; ring

/-- The spread is nonnegative. -/
theorem spread_nonneg (w t : Fin m → ℝ) (hwpos : ∀ j, 0 ≤ w j) : 0 ≤ spread w t :=
  Finset.sum_nonneg fun j _ => mul_nonneg (hwpos j) (sq_nonneg _)

/-- The measured width never falls below the instrument function. -/
theorem width2_ge_instr (w t sig : Fin m → ℝ) (hwpos : ∀ j, 0 ≤ w j) (hw : ∑ j, w j = 1) :
    instr w sig ≤ width2 w t sig := by
  rw [width2_eq_instr_add_spread w t sig hw]
  linarith [spread_nonneg w t hwpos]

/-- The spread vanishes exactly when every populated conformer has the centroid drift time. -/
theorem spread_eq_zero_iff (w t : Fin m → ℝ) (hwpos : ∀ j, 0 ≤ w j) :
    spread w t = 0 ↔ ∀ j, 0 < w j → t j = meanTime w t := by
  constructor
  · intro h j hj
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin m)), 0 ≤ w i * (t i - meanTime w t) ^ 2 :=
      fun i _ => mul_nonneg (hwpos i) (sq_nonneg _)
    have := (Finset.sum_eq_zero_iff_of_nonneg hterm).1 h j (Finset.mem_univ j)
    rcases mul_eq_zero.1 this with h0 | h0
    · exact absurd h0 (ne_of_gt hj)
    · have : t j - meanTime w t = 0 := by
        exact pow_eq_zero_iff (n := 2) (by norm_num) |>.1 h0
      linarith
  · intro h
    refine Finset.sum_eq_zero fun j _ => ?_
    rcases lt_or_eq_of_le (hwpos j) with hj | hj
    · rw [h j hj]; ring
    · rw [← hj]; ring

/-- A model committed to a single conformation predicts the instrumental width, exactly. -/
theorem width2_single (t sig : Fin 1 → ℝ) :
    width2 (fun _ => (1 : ℝ)) t sig = sig 0 ^ 2 := by
  simp [width2, meanTime]

/-- **Any excess width refutes every single-structure model.**  If the measured squared width
exceeds the instrumental variance of the conformer a one-structure model proposes, that model is
wrong, whatever the structure. -/
theorem excess_width_refutes_single (t sig : Fin 1 → ℝ) (measured : ℝ)
    (h : sig 0 ^ 2 < measured) : width2 (fun _ => (1 : ℝ)) t sig ≠ measured := by
  rw [width2_single]; exact ne_of_lt h

/-! ## The centroid selects nothing -/

/-- The centroid is at most the largest conformer drift time. -/
theorem meanTime_le_max (w t : Fin m → ℝ) (hwpos : ∀ j, 0 ≤ w j) (hw : ∑ j, w j = 1)
    {c : ℝ} (hc : ∀ j, t j ≤ c) : meanTime w t ≤ c := by
  have : ∑ j, w j * t j ≤ ∑ j, w j * c :=
    Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hc j) (hwpos j)
  rw [meanTime]
  calc ∑ j, w j * t j ≤ ∑ j, w j * c := this
    _ = c := by rw [← Finset.sum_mul, hw, one_mul]

/-- The centroid is at least the smallest conformer drift time. -/
theorem min_le_meanTime (w t : Fin m → ℝ) (hwpos : ∀ j, 0 ≤ w j) (hw : ∑ j, w j = 1)
    {c : ℝ} (hc : ∀ j, c ≤ t j) : c ≤ meanTime w t := by
  have : ∑ j, w j * c ≤ ∑ j, w j * t j :=
    Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hc j) (hwpos j)
  rw [meanTime]
  calc c = ∑ j, w j * c := by rw [← Finset.sum_mul, hw, one_mul]
    _ ≤ ∑ j, w j * t j := this

/-- **Every intermediate arrival time is a mixture.**  Given a compact conformer of drift time `a`
and an extended one of drift time `b > a`, any measured centroid strictly between them is
reproduced exactly by a two-state ensemble with both weights strictly positive. -/
theorem mixture_hits_intermediate {a b x : ℝ} (hxa : a < x) (hxb : x < b) :
    ∃ p : ℝ, 0 < p ∧ p < 1 ∧ meanTime ![p, 1 - p] ![a, b] = x := by
  refine ⟨(b - x) / (b - a), ?_, ?_, ?_⟩
  · exact div_pos (by linarith) (by linarith)
  · rw [div_lt_one (by linarith)]; linarith
  · have hba : b - a ≠ 0 := by linarith
    simp only [meanTime, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring

/-- **Mean and width still do not determine the ensemble.**  Two explicitly different ensembles on
the same five drift times share both their centroid and their conformational spread: a symmetric
two-state mixture at `1` and `3`, and a three-state mixture at `0`, `2`, `4`. -/
theorem moment_ambiguity :
    let t : Fin 5 → ℝ := ![0, 1, 2, 3, 4]
    let wA : Fin 5 → ℝ := ![0, 1/2, 0, 1/2, 0]
    let wB : Fin 5 → ℝ := ![1/8, 0, 3/4, 0, 1/8]
    (∑ j, wA j = 1) ∧ (∑ j, wB j = 1) ∧ wA ≠ wB ∧
      meanTime wA t = meanTime wB t ∧ spread wA t = spread wB t := by
  intro t wA wB
  have hA : ∑ j, wA j = 1 := by simp [wA, Fin.sum_univ_five]; norm_num
  have hB : ∑ j, wB j = 1 := by simp [wB, Fin.sum_univ_five]; norm_num
  have hmA : meanTime wA t = 2 := by
    simp [meanTime, wA, t, Fin.sum_univ_five]; norm_num
  have hmB : meanTime wB t = 2 := by
    simp [meanTime, wB, t, Fin.sum_univ_five]; norm_num
  refine ⟨hA, hB, ?_, by rw [hmA, hmB], ?_⟩
  · intro h
    have : wA 0 = wB 0 := by rw [h]
    simp [wA, wB] at this
  · rw [spread, spread, hmA, hmB]
    simp [wA, wB, t, Fin.sum_univ_five]
    norm_num

/-! ## Resolving power -/

/-- Two peaks of drift times `t₁`, `t₂` and widths `t/R` are separated at criterion `k` when the
sum of their widths, scaled by `k`, does not exceed their separation. -/
def resolvedAt (R k t₁ t₂ : ℝ) : Prop := k * (t₁ / R + t₂ / R) ≤ |t₁ - t₂|

/-- **The resolving power required to see two conformers.**  With widths set by the resolving power
`R`, two families are separated at criterion `k` exactly when `k·(t₁+t₂) ≤ R·|t₁−t₂|`: the cost of
resolving conformers grows like the inverse of their *fractional* cross-section difference. -/
theorem resolved_iff {R k t₁ t₂ : ℝ} (hR : 0 < R) :
    resolvedAt R k t₁ t₂ ↔ k * (t₁ + t₂) ≤ R * |t₁ - t₂| := by
  have h : k * (t₁ / R + t₂ / R) = k * (t₁ + t₂) / R := by field_simp
  rw [resolvedAt, h, div_le_iff₀ hR, mul_comm |t₁ - t₂| R]

/-- Two conformers whose cross sections differ by two percent are separated (at criterion `k = 1`)
exactly when the resolving power reaches `101`. -/
theorem resolved_two_percent {R : ℝ} (hR : 0 < R) :
    resolvedAt R 1 1 (51 / 50) ↔ 101 ≤ R := by
  rw [resolved_iff hR]
  have habs : |(1 : ℝ) - 51 / 50| = 1 / 50 := by
    rw [abs_of_nonpos (by norm_num)]; norm_num
  rw [habs]
  constructor <;> intro h <;> linarith

/-! ## Peak counting bounds model complexity -/

/-- **Resolved peaks are a lower bound on the number of conformers.**  If a model with `m`
conformers reproduces `k` distinct resolved arrival times, then `k ≤ m`. -/
theorem card_ge_of_distinct_peaks {k : ℕ} (t : Fin m → ℝ) (peaks : Fin k → ℝ)
    (hinj : Function.Injective peaks) (hmem : ∀ i, ∃ j, t j = peaks i) : k ≤ m := by
  choose f hf using hmem
  have hfinj : Function.Injective f := by
    intro a b hab
    apply hinj
    rw [← hf a, ← hf b, hab]
  simpa using Fintype.card_le_of_injective f hfinj

end IonMob

end IDR
