/-
# Part XLIX  The committor: the mechanism is not in the landscape either

`RequestProject.Committor` solves the harmonic (committor) system of a hopping chain on a
one-dimensional profile with both ends absorbing, and reads off what "mechanism" is and is not
determined by the free-energy profile.

`IDR.committor_laws` bundles five statements:

1. *Existence*: the closed form `q i = (Σ_{k<i} 1/(p k · kp k)) / (Σ_{k<n} 1/(p k · kp k))`
   solves the committor system, with the right boundary values;
2. *uniqueness*: and it is the only solution -- so "the committor" is well defined and is a ratio
   of resistances, detailed balance playing the role of Kirchhoff's law;
3. *monotonicity and range*: it increases strictly along the coordinate and lies in `[0,1]`, so
   there is exactly one transition region;
4. *when the landscape suffices*: at constant diffusion coefficient the committor is a functional
   of the equilibrium profile alone, dominated by the states of least weight -- the sense in
   which "the transition state is the top of the barrier" is a theorem;
5. *and when it does not*: on one flat landscape, two rate profiles differing in a single step
   place the transition state at `q 1 = 1/2` and at `q 1 = 1/3`.

With `IDR.first_passage_laws` (Part XLVIII) this completes the point: neither the rate nor the
mechanism of a conformational transition in a disordered region is a functional of its
free-energy landscape.  A model that reports populations, however accurately, has to report a
kinetic profile as well before any of the usual mechanistic language is licensed.
-/
import Mathlib
import RequestProject.Committor

set_option autoImplicit false

namespace IDR

open IDR.FirstPassage
open IDR.Committor

/-- **The committor laws of a one-dimensional free-energy profile.**

1. *Existence* of the closed-form committor.
2. *Uniqueness* of the solution of the harmonic system.
3. *Strict monotonicity and range* `[0,1]`.
4. *Constant diffusion coefficient*: the committor is then a functional of the profile alone.
5. *In general it is not*: two rate profiles on one flat landscape give `1/2` and `1/3`. -/
theorem committor_laws :
    (∀ (n : ℕ) (p kp km : ℕ → ℝ), 0 < n → DetailedBalance n p kp km → (∀ i, i ≤ n → 0 < p i) →
        (∀ i, i < n → 0 < kp i) → IsCommittor n kp km (committorFun n p kp)) ∧
    (∀ (n : ℕ) (p kp km q : ℕ → ℝ), 0 < n → DetailedBalance n p kp km → (∀ i, i ≤ n → 0 < p i) →
        (∀ i, i < n → 0 < kp i) → IsCommittor n kp km q →
          ∀ i, i ≤ n → q i = committorFun n p kp i) ∧
    (∀ (n : ℕ) (p kp : ℕ → ℝ), 0 < n → (∀ i, i ≤ n → 0 < p i) → (∀ i, i < n → 0 < kp i) →
        (∀ i j, i < j → j ≤ n → committorFun n p kp i < committorFun n p kp j) ∧
          ∀ i, i ≤ n → 0 ≤ committorFun n p kp i ∧ committorFun n p kp i ≤ 1) ∧
    (∀ (n : ℕ) (p : ℕ → ℝ) (D : ℝ) (i : ℕ), D ≠ 0 →
        committorFun n p (fun _ => D) i
          = (∑ k ∈ Finset.range i, 1 / p k) / (∑ k ∈ Finset.range n, 1 / p k)) ∧
    (committorFun 2 flatP fastKp 1 = 1/2 ∧ committorFun 2 flatP slowKp 1 = 1/3) :=
  ⟨fun _ _ _ _ hn hdb hp hkp => committorFun_isCommittor hn hdb hp hkp,
    fun _ _ _ _ _ hn hdb hp hkp hq => committor_unique hn hdb hp hkp hq,
    fun _ _ _ hn hp hkp =>
      ⟨fun _ _ hij hj => committor_strictMono hn hp hkp hij hj,
        fun _ hi => committor_mem_Icc hn hp hkp hi⟩,
    fun _ _ _ i hD => committor_uniform_rate hD i,
    mechanism_not_determined_by_landscape⟩

end IDR
