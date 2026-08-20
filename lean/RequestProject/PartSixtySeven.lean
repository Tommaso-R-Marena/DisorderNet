/-
# Part LXVII  How much sequence a pairwise theory can carry

A model of a disordered region is worth having only if it is sequence-resolved.  The analytic
sequence-to-ensemble theories in use -- sequence charge decoration, random-phase-approximation
free energies, preaveraged Debye-Hückel treatments -- share one structural feature: the charge
sequence enters through pairwise terms whose strength depends on the separation `j - i` along the
chain.  `RequestProject.SequenceDegeneracy` measures exactly how much of the sequence survives
that structure.

`IDR.sequence_resolution_laws` bundles four statements:

1. *Such a model sees the sequence only through its autocorrelation.*  The pairwise energy is
   identically `Σ_d k d · shell d` with `shell d = Σ_i q i q (i+d)`.  The identity holds for
   every kernel, hence at every screening length, salt concentration and temperature.
2. *So two sequences with equal autocorrelation are conflated absolutely.*  They give the same
   conformational energy function, and therefore the same value of every functional of the
   model: the same partition function, the same Boltzmann ensemble at every temperature, the
   same radius of gyration, the same value of every observable.  Sequence charge decoration is
   one such functional and is blind for the same reason.
3. *And the autocorrelation does not determine the sequence.*  Two explicit 12-residue charge
   patterns, `+ + + + − + − − + + − −` and `+ + − + + − + + + − − −`, have the same net charge,
   the same composition and the same autocorrelation at every separation, while being distinct,
   not each other's reverse, and not each other's charge inversion.
4. *Third order lifts the degeneracy.*  Their nearest-neighbour three-body correlations are `2`
   and `-6`.

The consequence for design is sharp, and it is the sequence-space counterpart of Part LXIV's
result about structure-based coarse-graining: a model can be sequence-resolved only in so far as
its sequence dependence is not pairwise-in-separation.  Two real charge patterns -- not
pathological ones, plain `±1` patterns of twelve residues -- are provably indistinguishable to
every theory of that class, at every state point; and a term of third order in the charge
sequence is enough to tell them apart.  Any claim that a fitted pairwise sequence parameter
"captures the patterning" is therefore false as stated: it captures the autocorrelation, which
is strictly less.
-/
import Mathlib
import RequestProject.SequenceDegeneracy

set_option autoImplicit false

namespace IDR

open IDR.SeqDeg

/-- **The sequence-resolution laws.**

1. a pairwise separation-dependent model sees the charge sequence only through its
   autocorrelation;
2. equal autocorrelation forces the same energy function and hence the same value of every
   functional of the model, and the same sequence charge decoration;
3. two explicit, genuinely distinct 12-residue charge patterns have equal net charge and equal
   autocorrelation at every separation;
4. and their three-body correlations differ. -/
theorem sequence_resolution_laws :
    (∀ (N : ℕ) (k q : ℕ → ℝ),
        pairSum N k q = ∑ d ∈ Finset.range N, if 0 < d then k d * shell N q d else 0) ∧
    (∀ (N : ℕ) (q q' : ℕ → ℝ), (∀ d, shell N q d = shell N q' d) →
        (∀ (X : Type) (U0 : X → ℝ) (u : X → ℕ → ℝ) (F : (X → ℝ) → ℝ),
            F (confEnergy N U0 u q) = F (confEnergy N U0 u q')) ∧
          scd N q = scd N q') ∧
    (netCharge 12 seqA = netCharge 12 seqB ∧
      (∀ d, shell 12 seqA d = shell 12 seqB d) ∧
      (seqA 2 ≠ seqB 2 ∧ seqA 4 ≠ seqB (11 - 4) ∧ seqA 0 ≠ -seqB 0 ∧ seqA 3 ≠ -seqB (11 - 3))) ∧
    (triple 12 seqA = 2 ∧ triple 12 seqB = -6) := by
  refine ⟨fun N k q => pairSum_eq_shell N k q,
    fun N q q' h => ⟨fun X U0 u F => sequence_blind N U0 u h F, scd_congr_of_shell_eq h⟩,
    ⟨by rw [netCharge_seqA, netCharge_seqB], shell_seqA_eq_seqB, seqA_ne_seqB⟩,
    ⟨triple_seqA, triple_seqB⟩⟩

end IDR
