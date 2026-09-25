/-
# Part LXXIII  The reach of pairwise charge patterning

Parts LXIX-LXXII price the *ensemble* side of a model of a disordered region.  This part prices
the *sequence* side, for the one family of sequence models that current theory uses
quantitatively: pairwise, separation-dependent charge energies
`E(q) = sum_{i<j} w (j-i) * q i * q j`, the common form of `SCD`, of `kappa`, and of every
Debye--Hückel chain energy.

`IDR.charge_patterning_laws` bundles four statements.

1. *Every pairwise model is a linear readout of the charge autocorrelation.*  Regrouping by
   separation, `E(q) = sum_{d=1}^{N-1} w d * C(d)` with `C(d) = sum_i q i * q (i+d)`.  A sequence
   of `N` charges enters through `N - 1` numbers.
2. *And the autocorrelation is exactly the invariant of the class.*  Two sequences agree under
   every kernel iff they have equal autocorrelation at every lag; Kronecker kernels recover the
   individual lags, so nothing coarser will do.
3. *A distance-independent kernel sees composition only.*  Sequence sensitivity of a pairwise
   model is exactly the variation of its kernel with separation.
4. *The class is incomplete, and provably so.*  Two explicit charge sequences of length nine have
   the same composition, are not related by reversal or charge conjugation, and have identical
   autocorrelation -- so every pairwise model, at every screening length, assigns them the same
   energy -- yet an explicit three-body correlator takes the values `2` and `-2` on them.

The moral for the design question is the mirror image of Part LXXII.  On the ensemble side, what
can be verified is a low-dimensional linear readout of the data; on the sequence side, what a
pairwise patterning model can express is a low-dimensional linear readout of the sequence.  A
model intended to capture a disordered region fully must therefore carry sequence information
beyond the pair-separation channel -- a many-body sequence term, or an explicit ensemble -- and
any claim that a fitted patterning parameter determines the behaviour of a sequence is refuted by
the homometric pair.
-/
import Mathlib
import RequestProject.ChargePatterning

set_option autoImplicit false

namespace IDR

open Finset IDR.Pattern

/-- **The charge-patterning laws.**

1. every pairwise separation-dependent charge energy is `sum_d w d * C d`;
2. equality of autocorrelations at all lags is equivalent to equality of energies under all
   kernels;
3. a constant kernel depends on the charge sequence only through its composition;
4. an explicit homometric pair of length-nine sequences -- same composition, unrelated by
   reversal or conjugation -- is invisible to every pairwise model yet separated by a three-body
   correlator. -/
theorem charge_patterning_laws :
    (∀ (N : ℕ) (w q : ℕ → ℝ), pairEnergy N w q = ∑ d ∈ Ico 1 N, w d * autocorr N q d) ∧
    (∀ (N : ℕ) (q q' : ℕ → ℝ),
        (∀ d, 1 ≤ d → d < N → autocorr N q d = autocorr N q' d) ↔
          ∀ w : ℕ → ℝ, pairEnergy N w q = pairEnergy N w q') ∧
    (∀ (N : ℕ) (c : ℝ) (q q' : ℕ → ℝ), (∑ i ∈ range N, q i = ∑ i ∈ range N, q' i) →
        (∑ i ∈ range N, (q i) ^ 2 = ∑ i ∈ range N, (q' i) ^ 2) →
          pairEnergy N (fun _ => c) q = pairEnergy N (fun _ => c) q') ∧
    ((∀ f : ℝ → ℝ, ∑ i ∈ range 9, f (chargeA i) = ∑ i ∈ range 9, f (chargeB i)) ∧
      chargeB ≠ chargeA ∧ chargeB ≠ rev9 chargeA ∧ chargeB ≠ (fun i => -chargeA i) ∧
      chargeB ≠ (fun i => -(rev9 chargeA i)) ∧
      (∀ w : ℕ → ℝ, pairEnergy 9 w chargeA = pairEnergy 9 w chargeB) ∧
      (∀ b kappa : ℝ, debye 9 b kappa chargeA = debye 9 b kappa chargeB) ∧
      triple 9 chargeA ≠ triple 9 chargeB) :=
  ⟨pairEnergy_eq_sum_autocorr, autocorr_eq_iff_pairEnergy_eq,
    fun _ c _ _ h1 h2 => pairEnergy_blind_of_same_composition h1 h2 c,
    homometric_same_composition, chargeB_ne_chargeA, chargeB_ne_rev, chargeB_ne_neg,
    chargeB_ne_neg_rev, homometric_blind, homometric_debye, homometric_triple_ne⟩

end IDR
