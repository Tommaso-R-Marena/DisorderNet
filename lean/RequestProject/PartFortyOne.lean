/-
# Part XLI  Neutral evolution: what has to be conserved, and what a model must be blind to

`RequestProject.Evolution` treats the fact that makes disordered regions hard to transfer
between species: their sequences diverge far faster than their function, so a model of a
disordered region cannot be a lookup on sequence identity, and the quantity it is trained to
reproduce has to be one whose symmetries the model shares.

`IDR.neutral_evolution_laws` bundles the five statements, all proved for the mean-field
Debye--Hückel descriptor of `RequestProject.Electrostatics`:

1. *the conserved quantity has a symmetry group* -- the screened electrostatic energy (at every
   salt concentration), the charge decoration and the net charge are unchanged by reading the
   chain backwards;
2. *the group is a genuine restriction* -- a directional descriptor, the net charge of the
   N-terminal half, is not reversal invariant;
3. *conservation without identity* -- the block polyampholyte and its reversal agree at no
   position at all and yet have identical values of all three descriptors;
4. *the neutral set is exponentially large* -- more than `4^m/m` sequences share one
   zero-net-charge composition, so a training set of size `T` sees at most a fraction
   `T·m/4^m` of it;
5. *the model must carry the symmetry* -- a model that does not is wrong by at least half its
   own asymmetry on one of the two sequences, and symmetrising it never costs anything.
-/
import Mathlib
import RequestProject.Evolution

set_option autoImplicit false

namespace IDR

open IDR.Electro IDR.Evolution

/-- **The neutral-evolution laws for a model of a disordered region.**

1. *Symmetry of the target.*  For every chain, bond length and salt concentration, the screened
   electrostatic energy, the charge decoration and the net charge are invariant under reversal
   of the chain.
2. *The symmetry is a restriction.*  A directional descriptor -- the net charge of the
   N-terminal half -- is not reversal invariant, so clause 1 is a statement about the pairwise
   level of description, not a triviality.
3. *Conserved physics in a diverged sequence.*  The block polyampholyte `+^m −^m` and its
   reversal have sequence identity exactly `0` and identical values of all three descriptors,
   at every salt concentration.
4. *The neutral set is exponentially large.*  More than `4^m/m` distinct `±1` sequences of
   length `2m` carry the same zero-net-charge composition, and a training set drawn from that
   class covers a fraction at most `T·m/4^m` of it.
5. *A model must carry the symmetry.*  Against a reversal-invariant target, a model has error
   at least half its own asymmetry on one of `x`, `rev x`; and its symmetrisation has squared
   error at most the mean of the two squared errors, strictly less when the model is
   asymmetric. -/
theorem neutral_evolution_laws :
    (∀ (N : ℕ) (b kappa : ℝ) (q : Fin N → ℝ),
        screenedEnergy b kappa (q ∘ Fin.rev) = screenedEnergy b kappa q ∧
        scd (q ∘ Fin.rev) = scd q ∧ netCharge (q ∘ Fin.rev) = netCharge q) ∧
    headCharge (![(1 : ℝ), -1] ∘ Fin.rev) ≠ headCharge (![(1 : ℝ), -1]) ∧
    (∀ (m : ℕ) (b kappa : ℝ),
        identity (blockSeq m ∘ Fin.rev) (blockSeq m) = 0 ∧
        screenedEnergy b kappa (blockSeq m ∘ Fin.rev) = screenedEnergy b kappa (blockSeq m) ∧
        scd (blockSeq m ∘ Fin.rev) = scd (blockSeq m) ∧
        netCharge (blockSeq m ∘ Fin.rev) = netCharge (blockSeq m)) ∧
    (∀ m : ℕ, 4 ≤ m →
        4 ^ m < m * (((Finset.univ : Finset (Fin (2 * m))).powersetCard m).image
          (chargeOf (n := 2 * m))).card) ∧
    (∀ {α : Type} (f M : α → ℝ) (g : α → α), (∀ x, f (g x) = f x) → ∀ x : α,
        |M x - M (g x)| / 2 ≤ max |M x - f x| (|M (g x) - f (g x)|) ∧
        (symmetrise M g x - f x) ^ 2
          ≤ ((M x - f x) ^ 2 + (M (g x) - f (g x)) ^ 2) / 2 ∧
        (M x ≠ M (g x) →
          (symmetrise M g x - f x) ^ 2
            < ((M x - f x) ^ 2 + (M (g x) - f (g x)) ^ 2) / 2)) := by
  refine ⟨fun N b kappa q => ⟨screenedEnergy_rev b kappa q, scd_rev q, netCharge_rev q⟩,
    headCharge_not_rev_invariant,
    fun m b kappa => diverged_sequences_same_physics m b kappa,
    fun m hm => composition_class_exponential m hm,
    fun f M g hf x => ⟨asymmetry_error_half f M g hf x, symmetrised_error_le f M g hf x,
      fun hne => symmetrised_error_lt f M g hf x hne⟩⟩

end IDR
