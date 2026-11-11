/-
# Part CXXVIII  What a model of a charged disordered region must carry, and what still escapes it

Parts CXXVI and CXXVII proved a negative and a quantitative fact: the electrostatic patterning of
a disordered region is a low-salt phenomenon, so a predictor that reads the sequence and nothing
else must be wrong by an amount growing like the cube of the length of the region.  This part
draws the design rule that follows, and states its limit in the same breath.

* `autoModel` — the representation a pairwise electrostatic model needs: the charge
  autocorrelation `C d` of the region, together with the kernel supplied by the solution
  condition.
* `autoModel_exact` — **sufficiency.**  For every kernel `w`, in particular for the
  Debye–Hückel kernel at any ionic strength and bond length, the model that stores the
  autocorrelation and is handed the salt condition reproduces the energy *exactly*
  (`autoModel_debye_exact` states the Debye case).  Nothing else about the sequence is needed.
* `salt_aware_design` — the design rule as one statement:
  1. *(context is indispensable)* a sequence-only predictor errs by at least `(t³/3 − t)/8` on
     the four sequence/condition pairs of Part CXXVI;
  2. *(what to store)* the autocorrelation plus the solution condition is enough for every
     pairwise electrostatic model;
  3. *(what is still missing)* and it is nevertheless not enough in general: the homometric pair
     of Part LXXIII has identical autocorrelation at every lag — so every pairwise model at every
     ionic strength gives the two sequences the same energy — while a three-body correlator
     separates them.

The conclusion is the one this development has argued from the beginning, now sharpened to the
electrostatics of a real disordered region: a model must be *conditional* (it must take the
solution condition as an input) and it must be *many-body* (a pairwise summary of the sequence is
provably incomplete).  Neither requirement can be traded for more data or more parameters of the
wrong kind.
-/
import Mathlib
import RequestProject.SaltCrossover
import RequestProject.DebyeScreening

set_option autoImplicit false

namespace IDR
namespace Salt

open Finset

/-- The pairwise electrostatic model that stores the charge autocorrelation of the region and
receives the interaction kernel from the solution condition. -/
def autoModel (N : ℕ) (w C : ℕ → ℝ) : ℝ := ∑ d ∈ Ico 1 N, w d * C d

/-- **Sufficiency.**  The autocorrelation together with the kernel reproduces every pairwise
electrostatic energy exactly. -/
theorem autoModel_exact (N : ℕ) (w q : ℕ → ℝ) :
    autoModel N w (Pattern.autocorr N q) = Pattern.pairEnergy N w q :=
  (Pattern.pairEnergy_eq_sum_autocorr N w q).symm

/-- The Debye–Hückel case: at every ionic strength and bond length, the stored autocorrelation
plus the solution condition gives the energy exactly. -/
theorem autoModel_debye_exact (N : ℕ) (b kappa : ℝ) (q : ℕ → ℝ) :
    autoModel N (fun d => Real.exp (-(kappa * (b * Real.sqrt d))) / (b * Real.sqrt d))
        (Pattern.autocorr N q)
      = Pattern.debye N b kappa q :=
  autoModel_exact N _ q

/-- **The design rule for a model of a charged disordered region.**  Context is indispensable,
the charge autocorrelation with the solution condition is what a pairwise model must carry, and
even that is provably incomplete. -/
theorem salt_aware_design {t : ℕ} {kappa : ℝ} (hk : 0 < kappa) (ht : 4 ≤ t)
    (hcross : 12 < kappa * t) :
    (∀ (f : (ℕ → ℝ) → ℝ) (eps : ℝ),
        |f altR - energy (2 * t) 0 altR| ≤ eps →
        |f (blkR t) - energy (2 * t) 0 (blkR t)| ≤ eps →
        |f altR - energy (2 * t) kappa altR| ≤ eps →
        |f (blkR t) - energy (2 * t) kappa (blkR t)| ≤ eps →
        ((t : ℝ) ^ 3 / 3 - t) / 8 ≤ eps)
      ∧ (∀ (N : ℕ) (b kappa' : ℝ) (q : ℕ → ℝ),
          autoModel N (fun d => Real.exp (-(kappa' * (b * Real.sqrt d))) / (b * Real.sqrt d))
              (Pattern.autocorr N q)
            = Pattern.debye N b kappa' q)
      ∧ ((∀ d : ℕ, Pattern.autocorr 9 Pattern.chargeA d = Pattern.autocorr 9 Pattern.chargeB d)
          ∧ (∀ b kappa' : ℝ,
              Pattern.debye 9 b kappa' Pattern.chargeA = Pattern.debye 9 b kappa' Pattern.chargeB)
          ∧ Pattern.triple 9 Pattern.chargeA ≠ Pattern.triple 9 Pattern.chargeB) := by
  refine ⟨fun f eps h1 h2 h3 h4 => no_salt_blind_prediction hk ht hcross f h1 h2 h3 h4,
    fun N b kappa' q => autoModel_debye_exact N b kappa' q,
    ⟨Pattern.homometric_autocorr, fun b kappa' => Pattern.homometric_debye b kappa',
      Pattern.homometric_triple_ne⟩⟩

end Salt
end IDR
