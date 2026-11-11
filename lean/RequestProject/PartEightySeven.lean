/-
# Part LXXXVII  How much of the ensemble has the sampling seen?

Capstone for `RequestProject.Coverage`.

Every ensemble model of a disordered region is built from a finite sample of conformations.  The
population the sample never visited is invisible to every diagnostic computed from the sample, so
the modeller needs a bound on it.  This part gives the exact expectation, the cost of coverage, and
— the positive half — an estimator of the unseen population that uses nothing but the sample.
-/
import RequestProject.Coverage

set_option autoImplicit false

namespace IDR

open Coverage Finset

/-- **The coverage laws.**  For `N` independent draws from an ensemble with `m` conformational
states of populations `w`:

1. *the sample weights are a probability distribution*, so the expectations below are honest;
2. *the probability a given state is never drawn* is exactly `(1 - w x)^N`;
3. *the expected unseen population* is exactly the missing mass `Σ_x w x (1 - w x)^N`;
4. *it is never zero for a broad ensemble*: if some state carries population strictly between `0`
   and `1`, the expected unseen population is strictly positive at every sample size, so no finite
   sample ever certifies that it has seen the whole ensemble;
5. *the cost of coverage*: leaving at most `eps` of a uniform `m`-state ensemble unseen requires at
   least `(1 - eps)·m` draws — linear in the number of populated states, which for a disordered
   region grows exponentially with chain length;
6. *Good–Turing*: the expected number of states drawn exactly once in a sample of size `N + 1` is
   `(N + 1)` times the missing mass at size `N`.  The singleton fraction of the sample is therefore
   an unbiased estimate of the population the sample is missing, computable without knowing `m` or
   `w` — the number an ensemble model should report next to its fit. -/
theorem coverage_laws {m : ℕ} (w : Fin m → ℝ) (hw : ∀ j, 0 ≤ w j) (hle : ∀ j, w j ≤ 1)
    (hsum : ∑ j, w j = 1) :
    (∀ N : ℕ, ∑ s : Fin N → Fin m, sw w s = 1) ∧
    (∀ (x : Fin m) (N : ℕ),
      ∑ s ∈ univ.filter fun s : Fin N → Fin m => occ s x = 0, sw w s = (1 - w x) ^ N) ∧
    (∀ N : ℕ, ∑ s : Fin N → Fin m, sw w s * unseenMass w s = missingMass w N) ∧
    (∀ x : Fin m, 0 < w x → w x < 1 → ∀ N : ℕ, 0 < missingMass w N) ∧
    (∀ m' : ℕ, 0 < m' → ∀ (N : ℕ) (eps : ℝ),
      missingMass (fun _ : Fin m' => (1 : ℝ) / m') N ≤ eps → (1 - eps) * m' ≤ N) ∧
    (∀ N : ℕ, ∑ s : Fin (N + 1) → Fin m, sw w s * (singletonCount s : ℝ)
      = (N + 1) * missingMass w N) := by
  refine ⟨fun N => sum_sw_eq_one w hsum N, fun x N => prob_unseen w hsum x N,
    fun N => expected_unseenMass w hsum N,
    fun x hx0 hx1 N => missingMass_pos w hw hle hx0 hx1 N,
    fun m' hm' N eps h => sample_size_needed m' hm' N h,
    fun N => good_turing w hsum N⟩

end IDR
