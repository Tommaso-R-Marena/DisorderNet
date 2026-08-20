/-
# Part XLIV  Fitting an ensemble to data: how much agreement is evidence

`RequestProject.Fitting` prices the central move of ensemble modelling -- reweighting a pool of
conformations until the predicted averages match `n` measurements -- and finds that the
agreement so obtained is, on its own, almost free.

`IDR.ensemble_fitting_laws` bundles four statements:

1. *Carathéodory.*  If the data can be fitted at all, they can be fitted exactly by an ensemble
   supported on at most `n + 1` conformations, whatever the size of the pool.  Exact agreement
   with `n` measurements is arithmetic, not evidence, beyond the fact that the data are
   feasible.
2. *Underdetermination.*  If the pool has more than `n + 1` members and some fit uses all of
   them, there is a nonzero direction in weight space that sums to zero and is invisible in
   every measured observable, and a whole interval of steps along it consists of exact fits.
3. *An explicit case.*  Three conformations, one measured observable and one unmeasured one:
   two exact fits of the same datum predict the two extreme values of the unmeasured
   observable.
4. *The positive counterpart.*  Uniqueness of the fit is exactly the triviality of that
   kernel -- the condition an experimental design has to establish, and one that a pool larger
   than `n + 1` never satisfies.
-/
import Mathlib
import RequestProject.Fitting

set_option autoImplicit false

namespace IDR

open IDR.Fitting

/-- **The ensemble-fitting laws.**

1. *Few structures suffice.*  Any fittable data set is fitted exactly by an ensemble supported
   on at most `n + 1` conformations.
2. *Beyond that the fit is a continuum.*  With more than `n + 1` conformations and a strictly
   positive fit, an interval of exact fits exists along a direction invisible to the data.
3. *Explicit blindness.*  Two exact fits of one datum give the unmeasured observable the values
   `1` and `0`.
4. *Uniqueness is a rank condition* on the fit map, and nothing else. -/
theorem ensemble_fitting_laws :
    (∀ (n k : ℕ) (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) (w : Fin k → ℝ), IsFit A b w →
        ∃ w' : Fin k → ℝ, IsFit A b w' ∧
          (Finset.univ.filter (fun j => w' j ≠ 0)).card ≤ n + 1) ∧
    (∀ (n k : ℕ) (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) (w : Fin k → ℝ), (∀ j, 0 < w j) →
        IsFit A b w → n + 1 < k →
        ∃ u : Fin k → ℝ, u ≠ 0 ∧ (∑ j, u j = 0) ∧ (∀ i, ∑ j, u j * A i j = 0) ∧
          ∃ eps > (0 : ℝ), ∀ t : ℝ, |t| ≤ eps → IsFit A b (fun j => w j + t * u j)) ∧
    (IsFit poolMeasured ![1] ![0, 1, 0] ∧
      IsFit poolMeasured ![1] ![1 / 2, 0, 1 / 2] ∧
      (∑ j, (![0, 1, 0] : Fin 3 → ℝ) j * poolUnmeasured j) = 1 ∧
      (∑ j, (![1 / 2, 0, 1 / 2] : Fin 3 → ℝ) j * poolUnmeasured j) = 0) ∧
    (∀ (n k : ℕ) (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ), LinearMap.ker (fitMap A) = ⊥ →
        ∀ w w' : Fin k → ℝ, IsFit A b w → IsFit A b w' → w = w') := by
  refine ⟨fun n k A b w hfit => fit_with_few_structures A b hfit,
    fun n k A b w hpos hfit hk => fit_not_unique A b hpos hfit hk,
    fit_blind_to_unmeasured,
    fun n k A b hker w w' h h' => fit_unique_of_ker_trivial A b hker h h'⟩

end IDR
