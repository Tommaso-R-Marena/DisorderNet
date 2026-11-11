/-
# Part LIV  Transition-path times: the crossing is not the waiting

The kinetic caveat of Parts XLVIII--L named transition-path times as outside what was proved.
`RequestProject.TransitionPath` brings them inside, without any new dynamical assumption: the
committor-conditioned (reactive) dynamics is constructed explicitly, shown to be a
detailed-balanced hopping chain in its own right, and the machinery already proved for
first-passage times is then applied to it verbatim.

`IDR.transition_path_laws` bundles four statements:

1. *The `h`-transform*: conditioning on reaching the product before returning to the reactant
   gives a detailed-balanced chain with profile `p q²` and forward rates `kp · q(i+1)/q(i)`,
   reflecting at its lower end because a reactive trajectory cannot return.
2. *Existence and uniqueness*: the mean transition-path time is therefore the unique solution
   of the reactive first-step system, and is given in closed form by the discrete Kramers
   formula for the reweighted profile.
3. *Positivity*: the crossing takes strictly positive time.
4. *Slow reaction, fast paths*: for every `M` there is a detailed-balanced three-state model
   with mean first-passage time at least `M` and mean transition-path time at most `1/M`.

Statement 4 is the one that matters for a disordered region.  A single-molecule experiment that
resolves transition paths is not measuring a rate, and a model calibrated to a relaxation time
is not thereby calibrated to the crossing: the two quantities can be pushed apart arbitrarily
far by the same barrier, in opposite directions.  Reporting both, as this development now can,
is the only honest option.
-/
import Mathlib
import RequestProject.TransitionPath

set_option autoImplicit false

namespace IDR

open IDR.FirstPassage
open IDR.Committor
open IDR.TransitionPath

/-- **Transition-path times, exactly.**

1. the reactive chain is detailed-balanced with profile `p q²`, and reflecting at its lower end;
2. the mean transition-path time is the unique solution of its first-step system;
3. it is strictly positive;
4. and it can be arbitrarily small while the mean first-passage time is arbitrarily large. -/
theorem transition_path_laws :
    (∀ (n : ℕ) (p kp km q : ℕ → ℝ), DetailedBalance n p kp km →
        (∀ i, 1 ≤ i → i ≤ n → 0 < q i) →
          DetailedBalance (n - 1) (reactP p q) (reactKp kp q) (reactKm km q) ∧
            reactKm km q 0 = 0) ∧
    (∀ (n : ℕ) (p kp km q T : ℕ → ℝ), 1 ≤ n → DetailedBalance n p kp km →
        (∀ i, i ≤ n → 0 < p i) → (∀ i, i < n → 0 < kp i) →
        (∀ i, 1 ≤ i → i ≤ n → 0 < q i) →
        IsMFPT (n - 1) (reactKp kp q) (reactKm km q) T → T 0 = tptFormula n p kp q) ∧
    (∀ (n : ℕ) (p kp q : ℕ → ℝ), 1 < n → (∀ i, i ≤ n → 0 < p i) → (∀ i, i < n → 0 < kp i) →
        (∀ i, 1 ≤ i → i ≤ n → 0 < q i) → 0 < tptFormula n p kp q) ∧
    (∀ M : ℝ, 0 < M → ∃ e : ℝ, 0 < e ∧ DetailedBalance 2 (wP e) wKp (wKm e) ∧
        M ≤ mfptFormula 2 (wP e) wKp ∧
        tptFormula 2 (wP e) wKp (committorFun 2 (wP e) wKp) ≤ 1 / M) :=
  ⟨fun _ _ _ km q hdb hq => ⟨reactive_detailedBalance hdb hq, reactKm_zero km q⟩,
    fun _ _ _ _ _ _ hn hdb hp hkp hq hT => tpt_eq hn hdb hp hkp hq hT,
    fun _ _ _ _ hn hp hkp hq => tptFormula_pos hn hp hkp hq,
    fun M hM => slow_reaction_fast_paths M hM⟩

end IDR
