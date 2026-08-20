/-
# Part LXXIX  Salt: what screening keeps, what it discards, and why condensation can be reentrant

Salt is the cheapest experimental knob on a disordered region, and the least well modelled.  In the
pairwise charge model of Part LXXIII the effect of added salt is to replace the kernel `w d` by a
screened one; writing the Debye factor in the fugacity variable `x = exp(-kappa b)` puts the whole
salt dependence of the patterning energy into a single real parameter, `screen x d = x^d / d`, with
`x = 1` at zero salt and `x -> 0` at infinite salt.  The resulting function
`energy N q x = sum_{d=1}^{N-1} (x^d / d) C(d)` is then an exactly analysable object: a power series
whose coefficients are the charge autocorrelations of Part LXXIII.

* `energy_eq_sum` -- the salt dependence in autocorrelation coordinates.  Salt does not change what
  a pairwise model can see; it reweights the same `N - 1` numbers.
* `energy_high_salt` -- **what survives screening.**  For `0 <= x <= 1` the energy differs from
  `C(1) x` by at most `x^2 * sum_{d>=2} |C(d)|/d`.  At high salt the *only* surviving sequence
  information is the nearest-neighbour charge correlation, with an explicit rate; and
  `energy_neg_of_small` turns that into a sign statement -- a sequence with `C(1) < 0` cannot be
  condensation-favouring at sufficiently high salt, whatever it does at low salt.
* `energy_reent_poly`, `energy_reent_half`, `energy_reent_four_fifths`, `energy_reent_one` --
  **an explicit reentrant sequence.**  For the twelve-residue charge sequence `reent` the
  patterning energy is *negative* at zero salt, *positive* at intermediate salt, and *negative*
  again at high salt.
* `reentrant_window` -- **hence two transitions, not one.**  By the intermediate value theorem the
  energy has a zero on either side of the favourable window: the condensation-favouring region of
  salt space is an interval bounded away from both limits.  This is reentrance derived, rather than
  fitted -- it follows from the sign pattern of the charge autocorrelation alone, with no
  ion-specific or hydration physics added.
* `reentrant_demixing` -- and the phase statement: with the threshold of Part LXXV the same
  sequence is predicted to condense at intermediate salt and to be stable at both zero and high
  salt.

The relationship this part adds is between the *lag structure* of the charge sequence and the
*shape* of the salt dependence.  A monotone salt response is not a property of electrostatics; it
is a property of sequences whose autocorrelation does not change sign in the relevant window.  The
same theory that says a pairwise model is blind to the homometric pair says that its salt
dependence is a Dirichlet-type transform of the autocorrelation vector -- and that transform can
be non-monotone.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.SequencePhase

set_option autoImplicit false

namespace IDR

namespace Screen

open Finset IDR.Pattern IDR.SeqPhase

/-- The screened chain kernel in the fugacity variable `x = exp(-kappa b)`:
`screen x d = x^d / d`.  `x = 1` is zero salt, `x -> 0` is infinite salt. -/
noncomputable def screen (x : ℝ) : ℕ → ℝ := fun d => x ^ d / d

/-- The patterning energy of the sequence `q` at screening parameter `x`. -/
noncomputable def energy (N : ℕ) (q : ℕ → ℝ) (x : ℝ) : ℝ := pairEnergy N (screen x) q

/-- The salt dependence, in autocorrelation coordinates. -/
theorem energy_eq_sum (N : ℕ) (q : ℕ → ℝ) (x : ℝ) :
    energy N q x = ∑ d ∈ Ico 1 N, (x ^ d / d) * autocorr N q d := by
  rw [energy, pairEnergy_eq_sum_autocorr]
  rfl

/-- The energy is a continuous function of the screening parameter. -/
theorem energy_continuous (N : ℕ) (q : ℕ → ℝ) : Continuous (energy N q) := by
  have h : energy N q
      = fun x : ℝ => ∑ j ∈ range N, ∑ i ∈ range j,
          (x ^ (j - i) / ((j - i : ℕ) : ℝ)) * (q i * q j) := by
    funext x
    simp [energy, pairEnergy, screen]
  rw [h]
  fun_prop

/-- At infinite salt the patterning energy vanishes. -/
theorem energy_zero (N : ℕ) (q : ℕ → ℝ) : energy N q 0 = 0 := by
  rw [energy_eq_sum]
  refine Finset.sum_eq_zero fun d hd => ?_
  simp only [Finset.mem_Ico] at hd
  rw [zero_pow (by omega)]
  simp

/-- **What survives screening.**  At high salt the patterning energy is the nearest-neighbour
charge correlation times the screening parameter, up to a second-order remainder. -/
theorem energy_high_salt (N : ℕ) (q : ℕ → ℝ) (hN : 1 < N) (x : ℝ) (hx0 : 0 ≤ x) (hx1 : x ≤ 1) :
    |energy N q x - autocorr N q 1 * x|
      ≤ x ^ 2 * ∑ d ∈ Ico 2 N, |autocorr N q d| / d := by
  rw [energy_eq_sum, Finset.sum_eq_sum_Ico_succ_bot hN]
  have hone : (x ^ 1 / (1 : ℕ)) * autocorr N q 1 = autocorr N q 1 * x := by
    push_cast; ring
  rw [hone, add_sub_cancel_left]
  calc |∑ d ∈ Ico 2 N, (x ^ d / d) * autocorr N q d|
      ≤ ∑ d ∈ Ico 2 N, |(x ^ d / d) * autocorr N q d| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ d ∈ Ico 2 N, x ^ 2 * (|autocorr N q d| / d) := by
        refine Finset.sum_le_sum fun d hd => ?_
        simp only [Finset.mem_Ico] at hd
        have hxd : x ^ d ≤ x ^ 2 := pow_le_pow_of_le_one hx0 hx1 hd.1
        have hd0 : (0 : ℝ) < d := by
          have : 0 < d := by omega
          exact_mod_cast this
        have hnn : 0 ≤ |autocorr N q d| / d := by positivity
        calc |(x ^ d / d) * autocorr N q d| = x ^ d * (|autocorr N q d| / d) := by
              rw [abs_mul, abs_div, abs_of_nonneg (pow_nonneg hx0 d), abs_of_nonneg hd0.le]
              ring
          _ ≤ x ^ 2 * (|autocorr N q d| / d) := mul_le_mul_of_nonneg_right hxd hnn
    _ = x ^ 2 * ∑ d ∈ Ico 2 N, |autocorr N q d| / d := by rw [Finset.mul_sum]

/-- **A sequence with anticorrelated neighbours cannot condense at high salt.** -/
theorem energy_neg_of_small (N : ℕ) (q : ℕ → ℝ) (hN : 1 < N) (x : ℝ) (hx0 : 0 < x) (hx1 : x ≤ 1)
    (hsmall : x * ∑ d ∈ Ico 2 N, |autocorr N q d| / d < -autocorr N q 1) :
    energy N q x < 0 := by
  have hb := energy_high_salt N q hN x hx0.le hx1
  have h1 : energy N q x - autocorr N q 1 * x ≤ x ^ 2 * ∑ d ∈ Ico 2 N, |autocorr N q d| / d :=
    (le_abs_self _).trans hb
  nlinarith [hb, h1, mul_pos hx0 hx0]

/-! ## An explicit reentrant sequence -/

/-- A twelve-residue charge sequence whose patterning energy changes sign twice as salt is
increased. -/
def reent : ℕ → ℝ
  | 0 => 1 | 1 => 1 | 2 => -1 | 3 => 1 | 4 => -1 | 5 => 1
  | 6 => -1 | 7 => -1 | 8 => -1 | 9 => -1 | 10 => -1 | 11 => 1
  | _ => 0

/-- Its salt dependence in closed form: the autocorrelation vector is
`(-1, 4, -1, 0, -1, 0, -3, 0, -3, 0, 1)`. -/
theorem energy_reent_poly (x : ℝ) :
    energy 12 reent x
      = -x + 2 * x ^ 2 - x ^ 3 / 3 - x ^ 5 / 5 - (3 / 7) * x ^ 7 - x ^ 9 / 3 + x ^ 11 / 11 := by
  simp only [energy, pairEnergy, screen, reent, Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num
  ring

/-- At zero salt the sequence is not condensation-favouring. -/
theorem energy_reent_one : energy 12 reent 1 < 0 := by
  rw [energy_reent_poly]; norm_num

/-- At intermediate salt it is. -/
theorem energy_reent_four_fifths : 1 / 10 < energy 12 reent (4 / 5) := by
  rw [energy_reent_poly]; norm_num

/-- At high salt it is not. -/
theorem energy_reent_half : energy 12 reent (1 / 2) < 0 := by
  rw [energy_reent_poly]; norm_num

/-- **Reentrance.**  The condensation-favouring region of salt space is an interval bounded away
from both the zero-salt and the high-salt limit: the patterning energy vanishes once below the
favourable window and once above it. -/
theorem reentrant_window :
    ∃ x₁ ∈ Set.Ioo (1 / 2 : ℝ) (4 / 5), ∃ x₂ ∈ Set.Ioo (4 / 5 : ℝ) 1,
      energy 12 reent x₁ = 0 ∧ energy 12 reent x₂ = 0 := by
  have hcont : Continuous (energy 12 reent) := energy_continuous 12 reent
  have hlow : energy 12 reent (1 / 2) < 0 := energy_reent_half
  have hmid : 0 < energy 12 reent (4 / 5) := lt_trans (by norm_num) energy_reent_four_fifths
  have hhigh : energy 12 reent 1 < 0 := energy_reent_one
  obtain ⟨x₁, hx₁mem, hx₁⟩ :=
    intermediate_value_Ioo (by norm_num : (1 / 2 : ℝ) ≤ 4 / 5) hcont.continuousOn
      (Set.mem_Ioo.mpr ⟨hlow, hmid⟩)
  obtain ⟨x₂, hx₂mem, hx₂⟩ :=
    intermediate_value_Ioo' (by norm_num : (4 / 5 : ℝ) ≤ 1) hcont.continuousOn
      (Set.mem_Ioo.mpr ⟨hhigh, hmid⟩)
  exact ⟨x₁, hx₁mem, x₂, hx₂mem, hx₁, hx₂⟩

/-- **Reentrant condensation.**  With the critical coupling of Part LXXV at chain length one, an
offset of `19/10` and unit slope, the model predicts that this sequence condenses at intermediate
salt and is stable at every composition both at zero salt and at high salt. -/
theorem reentrant_demixing :
    Demixes 1 (chiEff (19 / 10) 1 (energy 12 reent (4 / 5))) ∧
      (∀ c, ¬ IDR.Phase.PhaseSeparates (Set.Icc (0 : ℝ) 1)
        (IDR.FH.fh 1 (chiEff (19 / 10) 1 (energy 12 reent 1))) c) ∧
      (∀ c, ¬ IDR.Phase.PhaseSeparates (Set.Icc (0 : ℝ) 1)
        (IDR.FH.fh 1 (chiEff (19 / 10) 1 (energy 12 reent (1 / 2)))) c) := by
  have hthr : threshold 1 (19 / 10) 1 = 1 / 10 := by
    rw [threshold, IDR.FH.chiC_one]; norm_num
  have hmain := demixes_iff_gt_threshold (N := 1) (chi0 := 19 / 10) (lam := 1) one_pos one_pos
  refine ⟨(hmain _).1 ?_, (hmain _).2 ?_, (hmain _).2 ?_⟩
  · rw [hthr]; exact energy_reent_four_fifths
  · rw [hthr]; linarith [energy_reent_one]
  · rw [hthr]; linarith [energy_reent_half]

end Screen

end IDR
