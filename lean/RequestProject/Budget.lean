/-
# Part LXXXIX.5  Where to spend: components, or data?

The capacity law prices *architecture*; `RequestProject.SampleComplexity` prices *data*.  Put
on the same target they answer a question a group actually faces when planning a project on a
disordered region: given a fixed budget, does accuracy come from fitting more components or
from sampling more conformations?

On the maximally disordered target — `m` equally populated states — the two prices are exact
and they are very different.

* `optimalK_uniform` — the accuracy `eps` needs exactly `⌈m(1 - eps/2)⌉` components, and that
  many suffice.  This is a statement about the model, and it is achievable with no data at all
  once the populated states are known.
* `support_honest_needs_coverage` (Part V.3) — a *support-honest* learner, i.e. any
  reweighting or "weighted frames" model that can only put weight on conformations it has
  seen, needs `n ≥ m(1 - eps)` samples for the same accuracy.  This is a statement about the
  route, and no cleverness inside the class removes it.
* `budget_window` — both at once, plus the constructive half: the truncated model with
  `⌈m(1 - eps/2)⌉` components attains `eps` outright.  So on broad ensembles the accuracy is
  reachable by capacity and knowledge of which states are populated, and *not* reachable by
  sampling within the support-honest class unless the sample covers a constant fraction of a
  state space that is exponential in the length of the region (`RequestProject.Chain`).

The design reading is prescriptive and it cuts against the reflex of buying more simulation:
for a broad IDR ensemble, compute spent on enlarging a conformational pool buys accuracy only
linearly in coverage of an exponentially large space, while compute spent on identifying and
representing the populated states buys it exactly.
-/
import Mathlib
import RequestProject.CapacityExact
import RequestProject.SampleComplexity

set_option autoImplicit false

namespace IDR
namespace Budget

open Finset Capacity

/-- **The component count the uniform target needs, exactly.**  For accuracy `eps` on `m`
equally populated states the design rule returns `⌈m(1 - eps/2)⌉`. -/
theorem optimalK_uniform {m : ℕ} (hm : 0 < m) {eps : ℝ} (heps : 0 ≤ eps) :
    optimalK (uniformProfile hm) eps = ⌈(m : ℝ) * (1 - eps / 2)⌉₊ := by
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
  set x : ℝ := (m : ℝ) * (1 - eps / 2) with hx
  have hxm : x ≤ (m : ℝ) := by
    rw [hx]; nlinarith
  have hceil_le : ⌈x⌉₊ ≤ m := by
    have : ⌈x⌉₊ ≤ ⌈(m : ℝ)⌉₊ := Nat.ceil_le_ceil hxm
    simpa using this
  -- the value at any `k ≤ m`
  have hval : ∀ k : ℕ, k ≤ m → minErr (uniformProfile hm) k = 2 * ((m : ℝ) - k) / m :=
    fun k hk => minErr_uniform hm hk
  refine le_antisymm ?_ ?_
  · -- the ceiling achieves the accuracy, so the least such count is no larger
    refine Nat.sInf_le ?_
    have hk : (x : ℝ) ≤ (⌈x⌉₊ : ℝ) := Nat.le_ceil x
    have : minErr (uniformProfile hm) ⌈x⌉₊ = 2 * ((m : ℝ) - (⌈x⌉₊ : ℝ)) / m :=
      hval _ hceil_le
    simp only [Set.mem_setOf_eq, this]
    rw [div_le_iff₀ hmR]
    nlinarith [hk]
  · -- every count achieving the accuracy is at least the ceiling
    have hmem : minErr (uniformProfile hm) (optimalK (uniformProfile hm) eps) ≤ eps :=
      optimalK_spec (uniformProfile hm) heps
    set k : ℕ := optimalK (uniformProfile hm) eps with hk
    refine Nat.ceil_le.2 ?_
    by_cases hkm : k ≤ m
    · have hvk : minErr (uniformProfile hm) k = 2 * ((m : ℝ) - k) / m := hval k hkm
      rw [hvk, div_le_iff₀ hmR] at hmem
      rw [hx]
      nlinarith [hmem]
    · push_neg at hkm
      have : (m : ℝ) ≤ (k : ℝ) := by exact_mod_cast hkm.le
      linarith [hxm]

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- **The budget window on the maximally disordered target.**  Fix `m` equally populated
states and a target accuracy `eps`.

1. *Architecture is priced exactly*: no model with fewer than `⌈m(1 - eps/2)⌉` components
   reaches `eps`, however it is trained;
2. *and the price is payable*: the explicit truncated model with that many components reaches
   it, with no data at all once the populated states are known;
3. *the sampling route is priced differently*: a support-honest learner — anything that can
   only put weight on conformations it has actually seen — needs `n ≥ m(1 - eps)` samples for
   the same accuracy.

Since `m` is exponential in the length of a disordered region, clause 3 is the expensive one.
The design consequence is to spend on identifying and representing populated states rather
than on enlarging a pool. -/
theorem budget_window {m : ℕ} (hm : 0 < m) {eps : ℝ} (heps : 0 ≤ eps)
    {g : Fin m → X} (hg : Function.Injective g)
    (hpos : 0 < ⌈(m : ℝ) * (1 - eps / 2)⌉₊) :
    (∀ (k : ℕ), k < ⌈(m : ℝ) * (1 - eps / 2)⌉₊ → ∀ M : Ens X, M.card ≤ k →
        eps < Ens.ell1 M (target (uniformProfile hm) g)) ∧
    (∃ hkm : ⌈(m : ℝ) * (1 - eps / 2)⌉₊ ≤ m,
        Ens.ell1 (truncModel (uniformProfile hm) hpos hkm g)
          (target (uniformProfile hm) g) ≤ eps) ∧
    (∀ (n : ℕ) (T : (Fin n → Fin m) → (Fin m → ℝ)), Learn.SupportHonest T →
        Learn.risk (Learn.unifW m) T ≤ eps → (m : ℝ) * (1 - eps) ≤ n) := by
  have hmR : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
  have hceil_le : ⌈(m : ℝ) * (1 - eps / 2)⌉₊ ≤ m := by
    have hxm : (m : ℝ) * (1 - eps / 2) ≤ (m : ℝ) := by nlinarith
    have : ⌈(m : ℝ) * (1 - eps / 2)⌉₊ ≤ ⌈(m : ℝ)⌉₊ := Nat.ceil_le_ceil hxm
    simpa using this
  have hopt := optimalK_uniform hm heps
  refine ⟨fun k hk M hM => ?_, ⟨hceil_le, ?_⟩, fun n T hT hrisk =>
    Learn.support_honest_needs_coverage hT hm hrisk⟩
  · have hlt : k < optimalK (uniformProfile hm) eps := by rw [hopt]; exact hk
    have hfloor : minErr (uniformProfile hm) k ≤ Ens.ell1 M (target (uniformProfile hm) g) :=
      ell1_ge_two_tail (uniformProfile hm) hg hM
    exact lt_of_lt_of_le (optimalK_min (uniformProfile hm) hlt) hfloor
  · rw [minErr_eq (uniformProfile hm) hpos hceil_le hg, ← hopt]
    exact optimalK_spec (uniformProfile hm) heps

end Budget
end IDR
