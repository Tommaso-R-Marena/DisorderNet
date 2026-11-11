/-
# Part CXXXIII  The titration, designed: how many salt conditions, and how far up in salt

Part CXXIX proves two limitations of a salt series in the abstract — a series with fewer
conditions than lags leaves a blind direction, and conditions above `log(2N³/eps)` are
uninformative at energy resolution `eps`.  This part turns them into the two numbers an
experimenter actually needs, and states them as theorems rather than as estimates.

* `identification_requires_at_least` — **the count.**  If a set of `k` ionic strengths identifies
  every correlation profile (that is, no non-trivial profile is invisible to all of them), then
  `N − 1 ≤ k`: one condition per lag, with no discount for a clever choice of conditions.
* `demo_conditions_needed` — the worked case `N = 20`: eighteen ionic strengths, however chosen,
  leave a correlation profile that is exactly invisible.
* `demo_log_bound`, `demo_usable_range` — **the range.**  For a twenty-residue region measured to
  a resolution of `10⁻³ kT`, every inverse screening length above `17` (in inverse residue units)
  is useless: *no* two sequences of unit charges differ there by as much as the resolution.  Since
  the bound is logarithmic, buying another decade of instrument precision extends the usable range
  by only `log 10 ≈ 2.3`.
* `demo_report` — the two numbers together.

The design that follows: at least `N − 1` well-separated ionic strengths, all of them within the
usable window, and no expectation that a better calorimeter will substitute for more conditions.
-/
import Mathlib
import RequestProject.SaltTitration

set_option autoImplicit false

namespace IDR
namespace TitrationDesign

open Finset

/-! ## 1. The count: one condition per lag -/

/-- **Identification requires at least one salt condition per lag.**  If no non-trivial
correlation profile is invisible to the `k` conditions `kap`, then `N − 1 ≤ k`. -/
theorem identification_requires_at_least {N k : ℕ} (kap : Fin k → ℝ)
    (hid : ∀ c : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < N ∧ c d ≠ 0) →
      ∃ j, Titration.curve N c (kap j) ≠ 0) :
    N - 1 ≤ k := by
  by_contra hcon
  push_neg at hcon
  have hk : k + 1 < N := by omega
  obtain ⟨c, hne, hzero⟩ := Titration.finite_titration_underdetermined hk kap
  obtain ⟨j, hj⟩ := hid c hne
  exact hj (hzero j)

/-- The worked case: for a twenty-residue region, eighteen ionic strengths — however chosen —
leave a correlation profile that is exactly invisible to all of them. -/
theorem demo_conditions_needed (kap : Fin 18 → ℝ) :
    ∃ c : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < 20 ∧ c d ≠ 0) ∧
      ∀ j, Titration.curve 20 c (kap j) = 0 :=
  Titration.finite_titration_underdetermined (by norm_num) kap

/-! ## 2. The range: where the titration stops carrying information -/

/-- The design threshold for `N = 20` residues at resolution `10⁻³`: `log(2N³/eps) < 17`. -/
lemma demo_log_bound : Real.log (2 * (20 : ℝ) ^ 3 / (1 / 1000)) < 17 := by
  have h1 : (2 * (20 : ℝ) ^ 3 / (1 / 1000)) = 16000000 := by norm_num
  rw [h1]
  have he : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
  have h2 : (16000000 : ℝ) < Real.exp 17 := by
    have hx : Real.exp 17 = (Real.exp 1) ^ (17 : ℕ) := by
      rw [← Real.exp_nat_mul]; norm_num
    rw [hx]
    calc (16000000 : ℝ) < (2.7182818283 : ℝ) ^ (17 : ℕ) := by norm_num
      _ ≤ (Real.exp 1) ^ (17 : ℕ) := pow_le_pow_left₀ (by norm_num) he.le 17
  calc Real.log 16000000 < Real.log (Real.exp 17) := Real.log_lt_log (by norm_num) h2
    _ = 17 := Real.log_exp 17

/-- **The usable window, for a worked case.**  For a twenty-residue region and an energy
resolution of `10⁻³` (in units of `kT`), no two sequences of unit charges differ by as much as the
resolution at any inverse screening length `κ ≥ 17`: those conditions carry no information. -/
theorem demo_usable_range {kappa : ℝ} (hkap : 17 ≤ kappa) {q q' : ℕ → ℝ}
    (hq : ∀ i, |q i| ≤ 1) (hq' : ∀ i, |q' i| ≤ 1) :
    |Salt.energy 20 kappa q - Salt.energy 20 kappa q'| < 1 / 1000 := by
  refine Titration.high_salt_uninformative (N := 20) (eps := 1 / 1000) (by norm_num) (by norm_num)
    (by linarith) ?_ hq hq'
  calc Real.log (2 * ((20 : ℕ) : ℝ) ^ 3 / (1 / 1000))
      = Real.log (2 * (20 : ℝ) ^ 3 / (1 / 1000)) := by norm_num
    _ < 17 := demo_log_bound
    _ ≤ kappa := hkap

/-- **The worked design record.**  For a twenty-residue charged region read out to `10⁻³ kT`:
at least nineteen ionic strengths are needed (eighteen always leave a blind direction), and every
condition must lie below `κ = 17`. -/
theorem demo_report :
    (∀ kap : Fin 18 → ℝ, ∃ c : ℕ → ℝ, (∃ d, 1 ≤ d ∧ d < 20 ∧ c d ≠ 0) ∧
        ∀ j, Titration.curve 20 c (kap j) = 0) ∧
    (∀ kappa : ℝ, 17 ≤ kappa → ∀ q q' : ℕ → ℝ, (∀ i, |q i| ≤ 1) → (∀ i, |q' i| ≤ 1) →
        |Salt.energy 20 kappa q - Salt.energy 20 kappa q'| < 1 / 1000) :=
  ⟨demo_conditions_needed, fun _ hkap _ _ hq hq' => demo_usable_range hkap hq hq'⟩

end TitrationDesign
end IDR
