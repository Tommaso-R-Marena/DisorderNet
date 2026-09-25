/-
# Part LXXXV  The capacity threshold as a *testable* design rule

Capstone for `RequestProject.CapacityExact` and `RequestProject.Falsification`.

Earlier parts prove capacity *necessary*: too few components, and the model is wrong.  That
is a filter.  This part upgrades the capacity statement to a two-sided, numerical law -- the
exact best error at each component count, the exact improvement each added component buys,
the exact count a target accuracy demands -- and then packages it as a pre-registered
experimental test whose success is *not* a theorem, so that running it can teach something.

`Capacity.minErr_uniform` and `Capacity.fixed_capacity_degrades` in `CapacityExact.lean` add the
scaling: on a uniform target the attainable error at capacity `k` is exactly `2(m-k)/m`, so any
fixed component count degrades towards the maximal error as the ensemble broadens.
-/
import RequestProject.CapacityExact
import RequestProject.Falsification

set_option autoImplicit false

namespace IDR

open Capacity

/-- **The capacity-threshold laws.**  Against a target populating `m` conformational states
with measured populations `w₀ ≥ ⋯ ≥ w_{m-1}`, writing `minErr P k = 2(w_k + ⋯ + w_{m-1})`:

1. *the floor*: every model with at most `k` components is at population-space `ℓ¹` distance
   at least `minErr P k` from the target, whatever its parameters and however trained;
2. *the floor is attained*: for every `1 ≤ k ≤ m` some `k`-component model achieves exactly
   `minErr P k`, so the law is an equality and not merely a bound;
3. *the step*: the `(k+1)`-st component improves the attainable error by exactly `2 w_k`;
4. *the kink*: the attainable error is strictly positive below `k = m` and exactly zero from
   `k = m` on -- improvement stops at the threshold, and not before;
5. *the design rule*: for a target accuracy `eps ≥ 0` the least sufficient component count
   `optimalK P eps` attains `eps`, no smaller count does, and it never exceeds `m`;
6. *the observable failure*: an under-capacity model assigns population zero to a set of
   conformations that really carries at least `tail P k` of the population;
7. *what capacity does not buy*: a model with exactly `m` components can still be maximally
   wrong, so "at or above threshold" is never by itself a guarantee of fit;
8. *robustness*: profiles differing by `eta` in `ℓ¹` have floors differing by at most
   `2 eta`, so experimental error bars propagate to the prediction without amplification. -/
theorem capacity_threshold_laws {X : Type*} [Fintype X] [DecidableEq X]
    {m : ℕ} (P : Profile m) {g : Fin m → X} (hg : Function.Injective g) :
    (∀ (k : ℕ) (M : Ens X), M.card ≤ k → minErr P k ≤ Ens.ell1 M (target P g)) ∧
    (∀ k : ℕ, 0 < k → k ≤ m →
        ∃ M : Ens X, M.card = k ∧ Ens.ell1 M (target P g) = minErr P k) ∧
    (∀ (k : ℕ) (hk : k < m), minErr P k - minErr P (k + 1) = 2 * P.w ⟨k, hk⟩) ∧
    ((∀ k : ℕ, k < m → 0 < minErr P k) ∧ ∀ k : ℕ, minErr P k = 0 ↔ m ≤ k) ∧
    (∀ eps : ℝ, 0 ≤ eps →
        minErr P (optimalK P eps) ≤ eps ∧
        (∀ k : ℕ, k < optimalK P eps → eps < minErr P k) ∧
        optimalK P eps ≤ m) ∧
    (∀ (k : ℕ) (M : Ens X), M.card ≤ k →
        ∃ A : Finset X, (∀ x ∈ A, M.prob x = 0) ∧ P.tail k ≤ ∑ x ∈ A, (target P g).prob x) ∧
    (∃ (g' : Fin m → (Fin m ⊕ Fin m)) (M : Ens (Fin m ⊕ Fin m)),
        Function.Injective g' ∧ M.card = m ∧ Ens.ell1 M (target P g') = 2) ∧
    (∀ (Q : Profile m) (k : ℕ) (eta : ℝ), (∑ i, |P.w i - Q.w i|) ≤ eta →
        |minErr P k - minErr Q k| ≤ 2 * eta) := by
  refine ⟨fun k M hM => ell1_ge_two_tail P hg hM, ?_, fun k hk => minErr_step P hk,
    ⟨fun k hk => minErr_pos P hk, fun k => minErr_eq_zero_iff P⟩,
    fun eps heps => ⟨optimalK_spec P heps, fun k hk => optimalK_min P hk,
      optimalK_le P heps⟩,
    fun k M hM => missed_states_of_under_capacity P hg hM,
    at_capacity_not_sufficient P,
    fun Q k eta h => minErr_perturb P Q h⟩
  intro k hk hkm
  exact ⟨truncModel P hk hkm g, truncModel_card P hk hkm g, minErr_eq P hk hkm hg⟩

/-- **The pre-registered test, and the line between what is proved and what is at risk.**

1. On a system whose pre-registered tolerance lies below the baseline's floor, *no* model
   with the baseline's component count can meet the tolerance: the predicted failure of an
   under-capacity baseline is a theorem, not a hypothesis.
2. At capacity `m` an explicit model has error exactly zero, so the design rule asks for
   something achievable.
3. Nonetheless the pre-registered success criterion can fail: there are outcomes consistent
   with everything proved above on which the criterion is false.  The empirical content of
   the test sits entirely in whether the threshold-respecting model actually fits. -/
theorem prereg_test_laws (S : Prereg.SystemSpec) (heps : 0 ≤ S.eps)
    (hB : S.BaselineExcluded) {X : Type*} [Fintype X] [DecidableEq X]
    {g : Fin S.m → X} (hg : Function.Injective g) :
    (∀ M : Ens X, M.card ≤ S.baselineK →
        ((S.eps : ℚ) : ℝ) < Ens.ell1 M (target S.profile g)) ∧
    (∀ hm : 0 < S.m,
        Ens.ell1 (truncModel S.profile hm le_rfl g) (target S.profile g) = 0) ∧
    (∃ o : Prereg.Outcome, Prereg.Consistent S o ∧ ¬ Prereg.Confirms S o) :=
  ⟨fun _ hM => Prereg.baseline_must_fail S hB hg hM,
    fun hm => Prereg.threshold_model_attains S hm hg,
    Prereg.confirms_refutable S heps⟩

end IDR
