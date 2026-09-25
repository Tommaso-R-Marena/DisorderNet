/-
# Part LXXXIX.2  Making the predicted failure visible: what to measure, and how sharply

The capacity law of `RequestProject.CapacityExact` predicts a population-space discrepancy of
at least `2·tail P k` for any model with `k` components.  An experiment does not see
population space; it sees a handful of ensemble-averaged observables — a SAXS profile, a set
of chemical shifts, a FRET efficiency, an R₂ rate.  Whether the predicted failure is *visible*
in those numbers is a separate question from whether it exists, and it is the question that
decides whether a proposed test can succeed.  This file settles it, in both directions.

* `expect_gap_le_range` — **the ceiling.**  An observable taking values in `[a, b]` reports a
  discrepancy of at most `(b - a)/2` times the population-space `ℓ¹` error.  A model sitting
  exactly on the capacity floor therefore moves such an observable by at most `(b - a)·tail`
  (`gap_le_of_at_floor`) — so a low-contrast probe can be fully consistent with a badly wrong
  ensemble.  This is the mechanism behind the familiar and otherwise puzzling situation of an
  under-capacity model with an excellent χ².
* `contrast_requirement` — **the design rule that follows.**  If the measurement's precision
  is `sigma`, an observable can expose the predicted failure only if its dynamic range across
  the conformations satisfies `b - a ≥ sigma / tail`.  This is a number the experimenter can
  compute before choosing a probe.
* `indicator_gap` and `optimal_reporter` — **the floor is attained by a specific probe.**  The
  indicator of the missed subensemble has range `1` and reports the full discrepancy `tail`.
  The instruction is therefore concrete: build a reporter that fires on the states the
  under-capacity model omits (a contact, a distance window, a labelled pair), rather than a
  global average.
* `observables_needed` — **how many independent observables the test needs.**  If every state
  population is to be a consequence of the data rather than of the prior, the number of
  independent observables must be at least `m - 1`.  Below that, the fit cannot certify the
  populations the capacity test is scored against, whatever its quality.

Together these turn the capacity threshold from a statement about distributions into a
specification for an experiment: how many components to fit, how many independent observables
to measure, which probe to build, and how precise it has to be.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.CapacityExact
import RequestProject.Identifiability

set_option autoImplicit false

namespace IDR
namespace Visible

open Finset
open scoped Classical

variable {X : Type*} [Fintype X] [DecidableEq X]

/-! ## The ceiling: what a bounded observable can reveal -/

omit [DecidableEq X] in
/-- **A bounded observable sees at most its range times the `ℓ¹` error.**  For any observable
`f` with values in `[a, b]`, the discrepancy between the model's and the target's prediction
is at most `(b - a)/2` times the population-space `ℓ¹` distance. -/
theorem expect_gap_le_range (M E : Ens X) {f : X → ℝ} {a b : ℝ}
    (hlb : ∀ x, a ≤ f x) (hub : ∀ x, f x ≤ b) :
    |M.expect f - E.expect f| ≤ (b - a) / 2 * Ens.ell1 M E := by
  classical
  set c : ℝ := (a + b) / 2 with hc
  have hshift : M.expect f - E.expect f
      = ∑ x, (M.prob x - E.prob x) * (f x - c) := by
    have hM := M.expect_eq_sum_prob f
    have hE := E.expect_eq_sum_prob f
    have hzero : ∑ x, (M.prob x - E.prob x) = 0 := by
      rw [Finset.sum_sub_distrib, M.sum_prob, E.sum_prob]; ring
    have : ∑ x, (M.prob x - E.prob x) * (f x - c)
        = ∑ x, (M.prob x - E.prob x) * f x - c * ∑ x, (M.prob x - E.prob x) := by
      rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun x _ => by ring
    rw [this, hzero, hM, hE, ← Finset.sum_sub_distrib]
    simp only [mul_zero, sub_zero]
    exact Finset.sum_congr rfl fun x _ => by ring
  have hbound : ∀ x : X, |f x - c| ≤ (b - a) / 2 := by
    intro x
    rw [abs_le]
    constructor
    · have := hlb x; rw [hc]; linarith
    · have := hub x; rw [hc]; linarith
  calc |M.expect f - E.expect f| = |∑ x, (M.prob x - E.prob x) * (f x - c)| := by rw [hshift]
    _ ≤ ∑ x, |(M.prob x - E.prob x) * (f x - c)| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ x, |M.prob x - E.prob x| * |f x - c| := by
        exact Finset.sum_congr rfl fun x _ => abs_mul _ _
    _ ≤ ∑ x, |M.prob x - E.prob x| * ((b - a) / 2) := by
        refine Finset.sum_le_sum fun x _ => ?_
        exact mul_le_mul_of_nonneg_left (hbound x) (abs_nonneg _)
    _ = (b - a) / 2 * Ens.ell1 M E := by
        rw [← Finset.sum_mul, Ens.ell1]; ring

/-- **A model on the capacity floor barely moves a low-contrast observable.**  If the model's
population error is no more than the floor `2·tail P k` allows, then every observable with
values in `[a, b]` differs from the truth by at most `(b - a)·tail P k`.  An excellent fit to
such an observable is therefore no evidence that the model has enough components. -/
theorem gap_le_of_at_floor {m k : ℕ} (P : Capacity.Profile m) (M E : Ens X)
    (hfloor : Ens.ell1 M E ≤ 2 * P.tail k) {f : X → ℝ} {a b : ℝ}
    (hlb : ∀ x, a ≤ f x) (hub : ∀ x, f x ≤ b) (hab : a ≤ b) :
    |M.expect f - E.expect f| ≤ (b - a) * P.tail k := by
  have h := expect_gap_le_range M E hlb hub
  have hrange : 0 ≤ (b - a) / 2 := by linarith
  nlinarith [h, hfloor, hrange]

/-- **The contrast a probe must have.**  Suppose the measurement resolves a discrepancy only
when it exceeds `sigma`, and suppose the model's population error is at the floor
`2·tail P k`.  Then a probe with values in `[a, b]` that actually exposes the failure must
have dynamic range at least `sigma / tail P k`.  Choose the probe accordingly, before running
the experiment. -/
theorem contrast_requirement {m k : ℕ} (P : Capacity.Profile m) (M E : Ens X)
    (hfloor : Ens.ell1 M E ≤ 2 * P.tail k) {f : X → ℝ} {a b sigma : ℝ}
    (hlb : ∀ x, a ≤ f x) (hub : ∀ x, f x ≤ b) (hab : a ≤ b) (hk : k < m)
    (hsee : sigma ≤ |M.expect f - E.expect f|) :
    sigma / P.tail k ≤ b - a := by
  have htau : 0 < P.tail k := P.tail_pos hk
  have h := gap_le_of_at_floor P M E hfloor hlb hub hab
  rw [div_le_iff₀ htau]
  nlinarith [hsee, h]

/-! ## The floor is attained: the missed subensemble is the optimal reporter -/

/-- The average of the indicator of a set of conformations is the population of that set. -/
lemma expect_indicator (E : Ens X) (A : Finset X) :
    E.expect (fun x => if x ∈ A then (1 : ℝ) else 0) = ∑ x ∈ A, E.prob x := by
  classical
  rw [E.expect_eq_sum_prob]
  rw [Finset.sum_congr rfl (fun x _ => by
    by_cases hx : x ∈ A <;> simp [hx] : ∀ x ∈ (Finset.univ : Finset X),
      E.prob x * (if x ∈ A then (1 : ℝ) else 0) = if x ∈ A then E.prob x else 0)]
  rw [Finset.sum_ite_mem]
  simp

/-- **The indicator of the missed subensemble reports the whole discrepancy.**  If the model
gives population zero to every conformation of `A` and the truth gives `A` population at least
`tau`, then the indicator of `A` — an observable with range `1` — differs between model and
truth by at least `tau`.  Compare `expect_gap_le_range`: with range `1` no observable can do
better than `ℓ¹/2`, so on a model at the capacity floor this probe is optimal. -/
theorem indicator_gap {M E : Ens X} {A : Finset X} {tau : ℝ}
    (hA : ∀ x ∈ A, M.prob x = 0) (hmass : tau ≤ ∑ x ∈ A, E.prob x) :
    tau ≤ |M.expect (fun x => if x ∈ A then (1 : ℝ) else 0)
            - E.expect (fun x => if x ∈ A then (1 : ℝ) else 0)| := by
  classical
  have hM : M.expect (fun x => if x ∈ A then (1 : ℝ) else 0) = 0 := by
    rw [expect_indicator]
    exact Finset.sum_eq_zero fun x hx => hA x hx
  rw [hM, expect_indicator, zero_sub, abs_neg]
  exact le_trans hmass (le_abs_self _)

/-- **The reporter to build.**  For any model with at most `k` components there is an
observable — the indicator of a specific set of conformations, values in `{0,1}` — on which
the model predicts exactly `0` while the true ensemble predicts at least `tail P k`.  So the
capacity failure is not merely a distance in an unobservable space: it is a prediction about
a probe that can be built, and the required precision is `tail P k`. -/
theorem optimal_reporter {m k : ℕ} (P : Capacity.Profile m) {g : Fin m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ k) :
    ∃ A : Finset X,
      M.expect (fun x => if x ∈ A then (1 : ℝ) else 0) = 0 ∧
      P.tail k ≤ (Capacity.target P g).expect (fun x => if x ∈ A then (1 : ℝ) else 0) ∧
      P.tail k ≤ |M.expect (fun x => if x ∈ A then (1 : ℝ) else 0)
                  - (Capacity.target P g).expect (fun x => if x ∈ A then (1 : ℝ) else 0)| := by
  classical
  obtain ⟨A, hA0, hAmass⟩ := Capacity.missed_states_of_under_capacity P hg hM
  refine ⟨A, ?_, ?_, indicator_gap hA0 hAmass⟩
  · rw [expect_indicator]
    exact Finset.sum_eq_zero fun x hx => hA0 x hx
  · rw [expect_indicator]; exact hAmass

/-! ## How many independent observables the test needs -/

/-- **The measurement suite must carry at least `m - 1` independent observables.**  If every
state population is determined by the data — which is what it means for the populations the
capacity test is scored against to be a property of the experiment rather than of the prior —
then the number of measured observables satisfies `m ≤ k + 1`.  Contrapositive of
`IDR.Identify.exists_population_not_determined`, stated as the design requirement. -/
theorem observables_needed {m k : ℕ} (g : Fin k → Fin m → ℝ) {p : Fin m → ℝ} {d : ℝ}
    (hd : 0 < d) (hp : ∀ i, d ≤ p i) (hp1 : ∑ i, p i = 1)
    (hall : ∀ i : Fin m, Identify.Determined g p (Pi.single i (1 : ℝ))) :
    m ≤ k + 1 := by
  by_contra hcon
  push_neg at hcon
  obtain ⟨i, hi⟩ := Identify.exists_population_not_determined (by omega) g hd hp hp1
  exact hi (hall i)

end Visible
end IDR
