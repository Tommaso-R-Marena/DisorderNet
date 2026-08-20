/-
# Part LXXVI  Sequence to phase diagram: the threshold, the trade-off, and the blind spot

`RequestProject.SequencePhase` joins the sequence side (Part LXXIII) to the collective side
(Part LXXV) through the affine sequence-to-coupling law `chiEff chi0 lam P = chi0 + lam P`, the
shape of every published correlation between a patterning parameter and condensation.

`IDR.sequence_phase_laws` bundles four statements.

1. *An exact threshold.*  For `lam > 0` the model predicts a condensate exactly when the
   patterning parameter exceeds `(chiC N - chi0)/lam`, and stability at every composition
   otherwise.
2. *A length--sequence trade-off.*  That threshold strictly decreases with chain length: a longer
   region needs less blockiness, and the two are interchangeable within the model.
3. *The link has content.*  Two length-four sequences of identical composition -- blocked and
   alternating -- fall on opposite sides of the threshold at one chain length and one chemistry:
   the blocked one condenses, the alternating one is stable at every composition.  Composition
   does not decide condensation; patterning does.
4. *And the link has a hard blind spot.*  For every kernel, every sequence-to-coupling law and
   every chain length, the homometric pair of Part LXXIII has literally the same free-energy
   density, hence the same phase diagram and the same verdict.  A pairwise charge model cannot
   explain any measured difference between those two sequences; observing one falsifies the
   entire model class rather than its fitted parameters.

Taken with Parts LXIX-LXXII, this completes the picture the original question asked for.  A model
of an intrinsically disordered region can be made fully verifiable, and can carry real predictive
content from sequence to phase behaviour -- provided every claim is matched to what the data and
the model class can actually determine: the reported functionals of Part LXXII, the
autocorrelation coordinates of Part LXXIII, the fluctuation bounds of Part LXXIV, and the
critical point of Part LXXV.
-/
import Mathlib
import RequestProject.SequencePhase

set_option autoImplicit false

namespace IDR

open Set Finset IDR.Phase IDR.FH IDR.Pattern IDR.SeqPhase

/-- **The sequence-to-phase-diagram laws.**

1. an exact patterning threshold for condensation;
2. the threshold strictly decreases with chain length;
3. two sequences of equal composition on opposite sides of it;
4. the homometric pair of Part LXXIII has the same predicted phase diagram under every pairwise
   kernel and every sequence-to-coupling law. -/
theorem sequence_phase_laws :
    (∀ (N chi0 lam : ℝ), 0 < N → 0 < lam → ∀ P : ℝ,
        (threshold N chi0 lam < P → Demixes N (chiEff chi0 lam P)) ∧
        (P ≤ threshold N chi0 lam →
          ∀ c, ¬ PhaseSeparates (Icc (0 : ℝ) 1) (fh N (chiEff chi0 lam P)) c)) ∧
    (∀ (N M chi0 lam : ℝ), 0 < N → N < M → 0 < lam →
        threshold M chi0 lam < threshold N chi0 lam) ∧
    ((∀ f : ℝ → ℝ, ∑ i ∈ range 4, f (blocky4 i) = ∑ i ∈ range 4, f (alt4 i)) ∧
      blockiness 4 blocky4 = 1 ∧ blockiness 4 alt4 = -3 ∧
      Demixes 1 (chiEff (3/2) 1 (blockiness 4 blocky4)) ∧
      ∀ c, ¬ PhaseSeparates (Icc (0 : ℝ) 1) (fh 1 (chiEff (3/2) 1 (blockiness 4 alt4))) c) ∧
    (∀ (w : ℕ → ℝ) (Phi : ℝ → ℝ) (N : ℝ),
        fh N (Phi (pairEnergy 9 w chargeA)) = fh N (Phi (pairEnergy 9 w chargeB)) ∧
        (Demixes N (Phi (pairEnergy 9 w chargeA)) ↔
          Demixes N (Phi (pairEnergy 9 w chargeB)))) :=
  ⟨fun _ _ _ hN hlam P => demixes_iff_gt_threshold hN hlam P,
    fun _ _ _ _ hN hNM hlam => threshold_strictAnti hN hNM hlam,
    ⟨blocky4_alt4_same_composition, blockiness_blocky4, blockiness_alt4,
      blocky_demixes_alternating_stable.1, blocky_demixes_alternating_stable.2⟩,
    fun w Phi N => ⟨homometric_same_phase_diagram w Phi N, homometric_same_verdict w Phi N⟩⟩

end IDR
