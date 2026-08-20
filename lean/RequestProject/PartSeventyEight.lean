/-
# Part LXXVIII  Modification scanning: the experiment that breaks the sequence blind spot

`RequestProject.Phosphorylation` treats post-translational modification as a controlled
perturbation of the pairwise charge model of Part LXXIII, and solves it exactly.

`IDR.phosphorylation_laws` bundles six statements.

1. *The single-site law.*  `E(phos z k q) = E(q) - z h_k`: phosphorylation shifts the patterning
   energy by the charge removed times the local field at that site.
2. *Order-independence.*  Modifications of distinct sites commute, so the pairwise model forbids
   any dependence of the outcome on the order in which multisite marks are written.
3. *Epistasis is the kernel.*  The non-additivity of a double phosphorylation is exactly
   `z^2 w(d)`, `d` the spacing: independent of the sequence, the composition and the position of
   the pair along the chain.
4. *No three-body term.*  The third-order inclusion-exclusion interaction of three phosphosites
   vanishes identically; pairwise electrostatics allows pairwise epistasis and nothing beyond it.
5. *Identifiability.*  A phosphorylation scan over spacings determines the kernel, and hence the
   predicted patterning energy of every sequence.  Part LXXIII showed that sequence comparison
   cannot separate two homometric sequences under any kernel; modification scanning determines the
   kernel itself.  The two experiments are not interchangeable, and the informative one is the
   perturbation.
6. *The phenotype.*  A polycationic patch under a contact kernel sits above the demixing threshold
   of Part LXXV at chain length one, and a single phosphorylation of physiological charge `z = 2`
   takes it below: the model predicts a condensate that one kinase event dissolves.

The relationship added here is between chemistry and identifiability.  The blindness of Part LXXIII
comes from the model reading the sequence through its autocorrelation; a modification acts at a
single site, and its second-order response is the kernel evaluated at a single spacing.  A model of
a disordered region should therefore be fitted to modification data if its kernel is to be
determined at all -- and it should be reported with the three-body prediction that falsifies it.
-/
import Mathlib
import RequestProject.Phosphorylation

set_option autoImplicit false

namespace IDR

open Finset IDR.Pattern IDR.SeqPhase IDR.Phospho

/-- **The phosphorylation laws.**

1. the single-site shift is the local field;
2. modifications of distinct sites commute;
3. the double-phosphorylation epistasis is exactly `z^2 w(spacing)`, for every sequence;
4. the three-site interaction vanishes;
5. a spacing scan determines the kernel, hence every prediction of the model;
6. one phosphorylation event can carry a sequence across the demixing threshold. -/
theorem phosphorylation_laws :
    (∀ (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k : ℕ), k < N →
        pairEnergy N w (phos z k q) = pairEnergy N w q - z * locField N w q k) ∧
    (∀ (y z : ℝ) (k l : ℕ), k ≠ l → ∀ q : ℕ → ℝ,
        phos z k (phos y l q) = phos y l (phos z k q)) ∧
    (∀ (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l : ℕ), k < N → l < N → k ≠ l →
        shift2 N w q z k l - shift1 N w q z k - shift1 N w q z l = z ^ 2 * w (Nat.dist k l)) ∧
    (∀ (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l m : ℕ), k < N → l < N → m < N →
        k ≠ l → k ≠ m → l ≠ m →
        shift3 N w q z k l m
            - (shift2 N w q z k l + shift2 N w q z k m + shift2 N w q z l m)
            + (shift1 N w q z k + shift1 N w q z l + shift1 N w q z m) = 0) ∧
    (∀ (N : ℕ) (w w' q0 : ℕ → ℝ) (z : ℝ), z ≠ 0 →
        (∀ d, 1 ≤ d → d < N →
          shift2 N w q0 z 0 d - shift1 N w q0 z 0 - shift1 N w q0 z d
            = shift2 N w' q0 z 0 d - shift1 N w' q0 z 0 - shift1 N w' q0 z d) →
        ∀ q : ℕ → ℝ, pairEnergy N w q = pairEnergy N w' q) ∧
    (Demixes 1 (chiEff 0 1 (pairEnergy 4 contact patch4)) ∧
      ∀ c, ¬ IDR.Phase.PhaseSeparates (Set.Icc (0 : ℝ) 1)
        (IDR.FH.fh 1 (chiEff 0 1 (pairEnergy 4 contact (phos 2 0 patch4)))) c) :=
  ⟨fun N w q z k hk => pairEnergy_phos N w q z k hk,
    fun y z k l hkl q => phos_comm y z k l hkl q,
    fun N w q z k l hk hl hkl => phospho_epistasis N w q z k l hk hl hkl,
    fun N w q z k l m hk hl hm hkl hkm hlm =>
      phospho_no_three_body N w q z k l m hk hl hm hkl hkm hlm,
    fun N w w' q0 z hz h => pairEnergy_eq_of_epistasis_eq N w w' q0 z hz h,
    phospho_dissolves_condensate⟩

end IDR
