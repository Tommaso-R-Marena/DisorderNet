/-
# Part LXXIX  Salt: the high-salt limit, and reentrant condensation

`RequestProject.Screening` puts the whole salt dependence of the pairwise charge model of
Part LXXIII into one parameter, the screening fugacity `x = exp(-kappa b)`, with kernel
`screen x d = x^d / d`.  The patterning energy becomes
`energy N q x = sum_{d=1}^{N-1} (x^d / d) C(d)`, a power series in the screening parameter whose
coefficients are the charge autocorrelations.

`IDR.screening_laws` bundles four statements.

1. *Salt reweights, it does not enlarge.*  The salt-dependent energy is a linear functional of the
   same `N - 1` autocorrelation coordinates, for every salt concentration.
2. *What survives screening.*  For `0 <= x <= 1` the energy is `C(1) x` up to a remainder bounded
   by `x^2 sum_{d>=2} |C(d)|/d`: at high salt the only surviving sequence information is the
   nearest-neighbour charge correlation, with an explicit rate.
3. *Reentrance.*  An explicit twelve-residue sequence has patterning energy negative at zero salt,
   positive at intermediate salt and negative again at high salt, so by the intermediate value
   theorem the favourable region is an interval bounded away from both limits.  Non-monotone salt
   dependence therefore needs no ion-specific or hydration physics: a sign pattern in the charge
   autocorrelation suffices.
4. *And the phase statement.*  With the critical coupling of Part LXXV the same sequence is
   predicted to condense at intermediate salt and to be stable at every composition at both zero
   and high salt -- two salt transitions, from a model with one parameter.

Together with Parts LXXIII and LXXVIII this completes the account of what a pairwise charge model
of a disordered region can be asked: it is blind along the homometric direction, its kernel is
determined by a modification scan, and its salt dependence is a transform of the autocorrelation
vector which can change sign twice.
-/
import Mathlib
import RequestProject.Screening

set_option autoImplicit false

namespace IDR

open Finset IDR.Pattern IDR.SeqPhase IDR.Screen

/-- **The screening laws.**

1. the salt dependence in autocorrelation coordinates;
2. the high-salt expansion, with an explicit second-order remainder;
3. an explicit reentrant sequence, with a zero of the patterning energy on each side of the
   favourable window;
4. the corresponding prediction of condensation at intermediate salt only. -/
theorem screening_laws :
    (∀ (N : ℕ) (q : ℕ → ℝ) (x : ℝ),
        energy N q x = ∑ d ∈ Ico 1 N, (x ^ d / d) * autocorr N q d) ∧
    (∀ (N : ℕ) (q : ℕ → ℝ), 1 < N → ∀ x : ℝ, 0 ≤ x → x ≤ 1 →
        |energy N q x - autocorr N q 1 * x| ≤ x ^ 2 * ∑ d ∈ Ico 2 N, |autocorr N q d| / d) ∧
    (energy 12 reent 1 < 0 ∧ 1 / 10 < energy 12 reent (4 / 5) ∧ energy 12 reent (1 / 2) < 0 ∧
      ∃ x₁ ∈ Set.Ioo (1 / 2 : ℝ) (4 / 5), ∃ x₂ ∈ Set.Ioo (4 / 5 : ℝ) 1,
        energy 12 reent x₁ = 0 ∧ energy 12 reent x₂ = 0) ∧
    (Demixes 1 (chiEff (19 / 10) 1 (energy 12 reent (4 / 5))) ∧
      (∀ c, ¬ IDR.Phase.PhaseSeparates (Set.Icc (0 : ℝ) 1)
        (IDR.FH.fh 1 (chiEff (19 / 10) 1 (energy 12 reent 1))) c) ∧
      (∀ c, ¬ IDR.Phase.PhaseSeparates (Set.Icc (0 : ℝ) 1)
        (IDR.FH.fh 1 (chiEff (19 / 10) 1 (energy 12 reent (1 / 2)))) c)) :=
  ⟨fun N q x => energy_eq_sum N q x,
    fun N q hN x hx0 hx1 => energy_high_salt N q hN x hx0 hx1,
    ⟨energy_reent_one, energy_reent_four_fifths, energy_reent_half, reentrant_window⟩,
    reentrant_demixing⟩

end IDR
