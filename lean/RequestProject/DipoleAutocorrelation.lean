/-
# Part CXXIV  What the exact dipole can and cannot see

Parts CXX--CXXIII computed the dipole moment, the susceptibility and the extension of a
disordered polyampholyte exactly, and showed them to be extremal functionals of the charge
pattern.  It is tempting to read that as "the observable determines the sequence".  It does
not, and this file proves the precise limitation by connecting the exact theory to the
limitative result of Part LXXIII (`RequestProject/ChargePatterning.lean`).

* `mean_dipole_sq_eq_pairEnergy` -- the exactly computed mean squared dipole is, up to the
  factor `-b²`, a **pairwise separation-dependent charge energy** in the sense of Part LXXIII,
  with the linear kernel `w d = d`.
* `mean_dipole_sq_of_autocorr_eq` -- consequently it is a linear readout of the charge
  autocorrelation alone: two neutral sequences whose autocorrelations agree at every lag have
  *exactly* the same mean squared dipole, however differently their residues are ordered.

So the exact solution and the incompleteness theorem are two sides of one statement.  The
susceptibility of a disordered region is computable in closed form from its sequence, it is
maximised by the diblock and minimised by the perfectly mixed pattern -- and it still cannot
be inverted to recover the sequence, because it lives inside the pairwise class that Part
LXXIII proves is blind to genuinely different sequences.  A model of a disordered region that
reports only such an observable is provably incomplete, no matter how exactly it is computed.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.ChargePatterningExact

namespace IDR
namespace Charge

open Finset
open scoped Classical

variable {N : ℕ}

/-- **The exact susceptibility is a pairwise separation model.**  The mean squared dipole
moment of a neutral disordered polyampholyte is `-b²` times the pairwise charge energy with
the linear kernel `w d = d`. -/
theorem mean_dipole_sq_eq_pairEnergy (q : ℕ → ℝ) (b : ℝ) (hQ : pre q (N + 1) = 0) :
    (chainEns N).expect (fun s => (dipole q b s) ^ 2)
      = -b ^ 2 * Pattern.pairEnergy (N + 1) (fun d => (d : ℝ)) q := by
  have hpe : Pattern.pairEnergy (N + 1) (fun d => (d : ℝ)) q
      = ∑ j ∈ range (N + 1), ∑ i ∈ range j, q i * q j * ((j : ℝ) - i) := by
    rw [Pattern.pairEnergy]
    refine Finset.sum_congr rfl fun j _ => ?_
    refine Finset.sum_congr rfl fun i hi => ?_
    have hij : i ≤ j := le_of_lt (Finset.mem_range.1 hi)
    have hcast : ((j - i : ℕ) : ℝ) = (j : ℝ) - i := by
      have h := Nat.cast_sub (R := ℝ) hij
      simpa using h
    rw [hcast]
    ring
  rw [mean_dipole_sq q b hQ, hpe, neutral_lin_kernel q (N + 1) hQ]
  ring

/-- **The exact susceptibility is blind beyond the autocorrelation.**  Two neutral sequences
with the same charge autocorrelation at every lag have exactly the same mean squared dipole
moment -- and Part LXXIII exhibits sequences that share an autocorrelation without being the
same sequence. -/
theorem mean_dipole_sq_of_autocorr_eq (q q' : ℕ → ℝ) (b : ℝ) (hQ : pre q (N + 1) = 0)
    (hQ' : pre q' (N + 1) = 0)
    (hac : ∀ d, 1 ≤ d → d < N + 1 →
      Pattern.autocorr (N + 1) q d = Pattern.autocorr (N + 1) q' d) :
    (chainEns N).expect (fun s => (dipole q b s) ^ 2)
      = (chainEns N).expect (fun s => (dipole q' b s) ^ 2) := by
  have hpe := (Pattern.autocorr_eq_iff_pairEnergy_eq (N + 1) q q').1 hac (fun d => (d : ℝ))
  rw [mean_dipole_sq_eq_pairEnergy q b hQ, mean_dipole_sq_eq_pairEnergy q' b hQ', hpe]

end Charge
end IDR
