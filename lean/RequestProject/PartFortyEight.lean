/-
# Part XLVIII  The landscape does not determine the rate

`RequestProject.FirstPassage` solves the hopping dynamics on a one-dimensional free-energy
profile exactly, and reads off what the profile does and does not fix.

`IDR.first_passage_laws` bundles six statements:

1. *Solvability*: the closed form `Σ_{i<n} (Σ_{j≤i} p j)/(p i · kp i)` satisfies the
   first-step-analysis system, so the statements below are about a real object;
2. *the discrete Kramers formula*: and it is the only solution -- every mean first-passage time
   from the reflecting end to the absorbing end has exactly this value, and it is positive;
3. *the landscape does not determine the rate*: scaling every rate by `c > 0` preserves detailed
   balance, hence the equilibrium profile and every equilibrium observable, and divides every
   first-passage time by `c`;
4. *and not merely by an overall factor*: two rate profiles on the same flat landscape, differing
   only in the rate of one step, give crossing times `3` and `5`;
5. *what the profile does determine is a bound, exponential in the barrier*: for Boltzmann
   weights `p i = exp (-β F i)`, the crossing time is at least `exp (β (F b - F 0)) / kp b` for
   every intermediate state `b`;
6. *and a matching upper bound* `Σ_{i<n} 1/(p i · kp i)` for a normalised profile.

Read as a design constraint: a model that reports the equilibrium ensemble of a disordered
region -- even exactly, even resolved along a well-chosen coordinate -- underdetermines every
kinetic observable by an arbitrary factor.  Kinetics requires position-dependent rates (a
diffusion profile) as data of their own; the landscape supplies only the Arrhenius bracket.
-/
import Mathlib
import RequestProject.FirstPassage

set_option autoImplicit false

namespace IDR

open IDR.FirstPassage

/-- **The first-passage laws of a one-dimensional free-energy profile.**

1. *Solvability*: `mfptFun` solves the first-passage system.
2. *Kramers formula and positivity*: every solution equals it, and it is positive.
3. *Rate freedom*: rescaling the rates preserves detailed balance and divides the time.
4. *Step-by-step freedom*: an explicit pair of profiles with times `3` and `5`.
5. *Arrhenius lower bound*.
6. *Matching upper bound*. -/
theorem first_passage_laws :
    (∀ (n : ℕ) (p kp km : ℕ → ℝ), DetailedBalance n p kp km → (∀ i, i ≤ n → 0 < p i) →
        (∀ i, i < n → 0 < kp i) → km 0 = 0 → IsMFPT n kp km (mfptFun n p kp)) ∧
    (∀ (n : ℕ) (p kp km T : ℕ → ℝ), DetailedBalance n p kp km → (∀ i, i ≤ n → 0 < p i) →
        (∀ i, i < n → 0 < kp i) → km 0 = 0 → IsMFPT n kp km T →
          T 0 = mfptFormula n p kp ∧ (0 < n → 0 < T 0)) ∧
    (∀ (n : ℕ) (p kp km : ℕ → ℝ) (c : ℝ), 0 < n → 0 < c → c ≠ 1 → DetailedBalance n p kp km →
        (∀ i, i ≤ n → 0 < p i) → (∀ i, i < n → 0 < kp i) →
          DetailedBalance n p (fun i => c * kp i) (fun i => c * km i) ∧
            mfptFormula n p (fun i => c * kp i) = mfptFormula n p kp / c ∧
            mfptFormula n p (fun i => c * kp i) ≠ mfptFormula n p kp) ∧
    (DetailedBalance 2 flatP fastKp fastKm ∧ DetailedBalance 2 flatP slowKp slowKm ∧
      mfptFormula 2 flatP fastKp = 3 ∧ mfptFormula 2 flatP slowKp = 5) ∧
    (∀ (n : ℕ) (kp F : ℕ → ℝ) (b : ℕ) (beta : ℝ), b < n → (∀ i, i < n → 0 < kp i) →
        Real.exp (beta * (F b - F 0)) / kp b
          ≤ mfptFormula n (fun i => Real.exp (-(beta * F i))) kp) ∧
    (∀ (n : ℕ) (p kp : ℕ → ℝ), (∀ i, i ≤ n → 0 < p i) → (∀ i, i < n → 0 < kp i) →
        (∀ i, i < n → cum p i ≤ 1) →
          mfptFormula n p kp ≤ ∑ i ∈ Finset.range n, 1 / (p i * kp i)) := by
  obtain ⟨-, -, -, hfast, hslow, -, -, h3, h5⟩ := kinetics_not_determined_by_profile
  refine ⟨fun n p kp km hdb hp hkp hkm0 => mfptFun_isMFPT hdb hp hkp hkm0,
    fun n p kp km T hdb hp hkp hkm0 hT => ⟨mfpt_eq hdb hp hkp hkm0 hT, fun hn => ?_⟩,
    fun n p kp km c hn hc hc1 hdb hp hkp =>
      landscape_does_not_determine_rate hn hc hc1 hdb hp hkp,
    ⟨hfast, hslow, h3, h5⟩,
    fun n kp F b beta hb hkp => mfpt_ge_exp_barrier hb hkp,
    fun n p kp hp hkp hnorm => mfptFormula_le hp hkp hnorm⟩
  rw [mfpt_eq hdb hp hkp hkm0 hT]
  exact mfptFormula_pos hn hp hkp

end IDR
