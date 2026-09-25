/-
# Part LXXVI  From sequence to phase diagram, and what that link can carry

Part LXXIII prices the sequence side of a model of a disordered region: a pairwise,
separation-dependent charge model reads the sequence only through its charge autocorrelation.
Part LXXV prices the collective side: a chain of `N` segments demixes exactly above the coupling
`chiC N = (1 + sqrt N)^2 / (2N)`.  This file joins them, which is what a predictive model of
condensation actually claims to do: sequence in, phase diagram out.

The link is the *effective coupling* `chiEff chi0 lam P = chi0 + lam * P`, an affine increasing
function of a patterning parameter `P` -- the standard shape of every published
sequence-to-phase-behaviour correlation.

* `demixes_iff_gt_threshold` -- **an exact sequence threshold.**  With `lam > 0` the solution
  demixes precisely when the patterning parameter exceeds
  `threshold N chi0 lam = (chiC N - chi0)/lam`, and is stable at every composition otherwise.
  The threshold is a number the model must commit to, and it is testable.
* `threshold_strictAnti` -- **a length--sequence trade-off.**  The threshold strictly decreases
  with chain length: a longer disordered region condenses at a lower patterning parameter than a
  short one with the same composition and the same chemistry.  Sequence blockiness and chain
  length buy the same thing.
* `blocky_demixes_alternating_stable` -- **the link is not vacuous.**  Two length-four charge
  sequences with the *same composition* (two positive, two negative) sit on opposite sides of the
  threshold at one and the same chain length and chemistry: the blocked sequence condenses and
  the alternating one does not.  Composition alone cannot predict condensation.
* `homometric_same_phase_diagram` -- **but the link cannot be finer than the autocorrelation.**
  For *every* kernel `w`, *every* function `Phi` of the resulting pairwise energy and *every*
  chain length, the two sequences of Part LXXIII have literally the same free-energy density and
  hence the same phase diagram.  So a pairwise-electrostatics theory of condensation is unable,
  in principle, to explain any measured difference between them -- and if such a difference is
  measured, the whole model class is falsified, not merely its fitted parameters.

The joint statement is the useful one.  A sequence-to-phase-diagram model can be exactly right
about a threshold and exactly blind to a pair of sequences on either side of it; the way to
report it honestly is to state the patterning coordinate it uses, the chain length it was fitted
at, and the sequences it cannot distinguish.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.FloryHuggins

set_option autoImplicit false

namespace IDR

namespace SeqPhase

open Set Finset IDR.Phase IDR.FH IDR.Pattern

/-- The effective Flory--Huggins coupling predicted from a patterning parameter `P` by an affine
sequence-to-coupling law. -/
noncomputable def chiEff (chi0 lam P : ℝ) : ℝ := chi0 + lam * P

/-- The patterning parameter above which the model predicts a condensate. -/
noncomputable def threshold (N chi0 lam : ℝ) : ℝ := (chiC N - chi0) / lam

/-- A chain of length `N` with coupling `chi` demixes at some composition. -/
def Demixes (N chi : ℝ) : Prop := ∃ c, PhaseSeparates (Icc (0 : ℝ) 1) (fh N chi) c

/-- **The exact sequence threshold.**  Under an affine increasing sequence-to-coupling law the
model predicts a condensate exactly above `threshold N chi0 lam`, and stability at every
composition at or below it. -/
theorem demixes_iff_gt_threshold {N chi0 lam : ℝ} (hN : 0 < N) (hlam : 0 < lam) (P : ℝ) :
    (threshold N chi0 lam < P → Demixes N (chiEff chi0 lam P)) ∧
      (P ≤ threshold N chi0 lam →
        ∀ c, ¬ PhaseSeparates (Icc (0 : ℝ) 1) (fh N (chiEff chi0 lam P)) c) := by
  constructor
  · intro hP
    have hchi : chiC N < chiEff chi0 lam P := by
      rw [threshold, div_lt_iff₀ hlam] at hP
      rw [chiEff]
      linarith
    exact fh_demixes_above_chiC hN hchi
  · intro hP c
    have hchi : chiEff chi0 lam P ≤ chiC N := by
      rw [threshold, le_div_iff₀ hlam] at hP
      rw [chiEff]
      linarith
    exact no_demixing_below_chiC hN hchi c

/-- **Length and blockiness buy the same thing.**  The patterning parameter needed to condense
strictly decreases with chain length. -/
theorem threshold_strictAnti {N M chi0 lam : ℝ} (hN : 0 < N) (hNM : N < M) (hlam : 0 < lam) :
    threshold M chi0 lam < threshold N chi0 lam := by
  have h := chiC_strictAnti hN hNM
  rw [threshold, threshold, div_lt_div_iff_of_pos_right hlam]
  linarith

/-! ## Two sequences of equal composition on opposite sides of the threshold -/

/-- A blocked length-four charge sequence `+ + - -`. -/
def blocky4 : ℕ → ℝ
  | 0 => 1 | 1 => 1 | 2 => -1 | 3 => -1 | _ => 0

/-- The alternating length-four charge sequence `+ - + -`, of the same composition. -/
def alt4 : ℕ → ℝ
  | 0 => 1 | 1 => -1 | 2 => 1 | 3 => -1 | _ => 0

/-- The nearest-neighbour patterning parameter: the like-charge contact score
`sum_i q i * q (i+1)`, the pairwise energy of the Kronecker kernel at separation one. -/
noncomputable def blockiness (N : ℕ) (q : ℕ → ℝ) : ℝ :=
  pairEnergy N (fun d => if d = 1 then (1 : ℝ) else 0) q

theorem blockiness_blocky4 : blockiness 4 blocky4 = 1 := by
  norm_num [blockiness, pairEnergy, Finset.sum_range_succ, blocky4]

theorem blockiness_alt4 : blockiness 4 alt4 = -3 := by
  norm_num [blockiness, pairEnergy, Finset.sum_range_succ, alt4]

/-- The two sequences have the same composition: every function of the individual charges sums
to the same value. -/
theorem blocky4_alt4_same_composition (f : ℝ → ℝ) :
    ∑ i ∈ range 4, f (blocky4 i) = ∑ i ∈ range 4, f (alt4 i) := by
  simp [Finset.sum_range_succ, blocky4, alt4]
  ring

/-- **Composition does not decide condensation.**  At chain length `1`, offset `3/2` and slope
`1`, the blocked sequence is predicted to condense and the alternating sequence of the same
composition is predicted to be stable at every composition. -/
theorem blocky_demixes_alternating_stable :
    Demixes 1 (chiEff (3/2) 1 (blockiness 4 blocky4)) ∧
      ∀ c, ¬ PhaseSeparates (Icc (0 : ℝ) 1) (fh 1 (chiEff (3/2) 1 (blockiness 4 alt4))) c := by
  have hthr : threshold 1 (3/2) 1 = 1/2 := by
    rw [threshold, chiC_one]
    norm_num
  have hlaws := demixes_iff_gt_threshold (N := 1) (chi0 := 3/2) (lam := 1)
    (by norm_num) (by norm_num)
  refine ⟨(hlaws (blockiness 4 blocky4)).1 ?_, (hlaws (blockiness 4 alt4)).2 ?_⟩
  · rw [hthr, blockiness_blocky4]; norm_num
  · rw [hthr, blockiness_alt4]; norm_num

/-! ## The homometric pair has the same phase diagram in every pairwise model -/

/-- **A pairwise charge model predicts identical phase behaviour for the homometric pair.**  For
every kernel `w`, every sequence-to-coupling law `Phi`, and every chain length, the two sequences
of Part LXXIII give literally the same free-energy density -- hence the same spinodal, the same
binodal, and the same predicted condensate. -/
theorem homometric_same_phase_diagram (w : ℕ → ℝ) (Phi : ℝ → ℝ) (N : ℝ) :
    fh N (Phi (pairEnergy 9 w chargeA)) = fh N (Phi (pairEnergy 9 w chargeB)) := by
  rw [homometric_blind w]

/-- In particular the predicted demixing verdicts agree, for every chain length and every
sequence-to-coupling law built on a pairwise kernel. -/
theorem homometric_same_verdict (w : ℕ → ℝ) (Phi : ℝ → ℝ) (N : ℝ) :
    Demixes N (Phi (pairEnergy 9 w chargeA)) ↔ Demixes N (Phi (pairEnergy 9 w chargeB)) := by
  rw [Demixes, Demixes, homometric_blind w]

end SeqPhase

end IDR
