/-
# Part LXXXV.2  The pre-registered falsification test, as a formal object

`RequestProject.CapacityExact` proves an *exact* attainable-error law: against a target with
measured populations `w₀ ≥ ⋯ ≥ w_{m-1}`, the best possible `ℓ¹` error of a `k`-component
model is exactly `2(w_k + ⋯ + w_{m-1})`.  That is a number, computable before any model is
fitted, from populations measured independently of the model.  This file turns it into a
*test*: a pre-registered, machine-checkable criterion that a real experiment can fail.

The point of the file is to keep the two halves of such a test rigorously apart.

* `baseline_must_fail` -- **the entailed half.**  If the pre-registered tolerance `eps` is
  below the floor of the practitioner's fixed-`k` baseline (a fixed three-mode mixture, say),
  then *no* such model can reach the tolerance.  This is a theorem, so an experiment which
  reports a below-tolerance error for an under-capacity model has an error somewhere else --
  in the state count, the populations, or the fit metric -- and that is exactly the value of
  proving it: it protects the test from failing for a dumb reason.
* `threshold_model_attains` -- at capacity `m` the floor is zero and an explicit model
  attains it, so the design rule is not vacuous.
* `Confirms` -- the pre-registered success criterion on the outcome of the experiment: the
  under-capacity baseline misses the tolerance, the threshold-respecting model meets it.
  Decidable, so the verdict is a computation, not a judgement call.
* `confirms_refutable` -- **the empirical half is genuinely open.**  There are outcomes
  consistent with everything proved here on which `Confirms` is false.  The test can fail;
  no theorem in this development forces it to pass.  `at_capacity_not_sufficient` in
  `CapacityExact` is the structural reason: capacity is necessary, never sufficient.
* `panelA`, `panelB`, `panelC` and the theorems about them -- worked instantiations.  The
  populations there are *stipulated illustrations*, not measurements; they show the shape of
  the arithmetic a real record fills in, and every numerical claim about them is decided by
  computation rather than asserted.

`PREREGISTRATION.md` in the project root is the protocol these definitions encode: which
systems, which independent source of the state count, which baseline, which analysis, and
which outcomes would refute the prediction.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Metric
import RequestProject.CapacityExact

namespace IDR
namespace Prereg

open Capacity

/-! ## A pre-registered record for one system -/

/-- Everything about one system that must be fixed **before** any model is fitted: the
number `m` of populated states and their populations `q` (from an independent measurement),
the tolerance `eps` at which the comparison will be scored, and the component count
`baselineK` of the practitioner's baseline model. -/
structure SystemSpec where
  /-- number of populated conformational states, from independent measurement -/
  m : ℕ
  /-- their populations, in decreasing order -/
  q : Fin m → ℚ
  qpos : ∀ i, 0 < q i
  qanti : ∀ i j : Fin m, i ≤ j → q j ≤ q i
  qsum : ∑ i, q i = 1
  /-- the pre-registered `ℓ¹` tolerance -/
  eps : ℚ
  /-- the component count of the baseline model (e.g. `3` for a fixed three-mode mixture) -/
  baselineK : ℕ

namespace SystemSpec

variable (S : SystemSpec)

/-- The population profile of the system. -/
def profile : Profile S.m := Profile.ofRat S.q S.qpos S.qanti S.qsum

/-- The attainable-error floor at capacity `k`, in exact rational arithmetic. -/
def floorAt (k : ℕ) : ℚ := minErrRat S.q k

/-- The pre-registered admissibility condition on the baseline: its floor is above the
tolerance, so the theory says it *cannot* pass.  Decidable, hence checkable before the
experiment. -/
def BaselineExcluded : Prop := S.eps < S.floorAt S.baselineK

instance : Decidable S.BaselineExcluded := by
  unfold BaselineExcluded; infer_instance

end SystemSpec

/-! ## The entailed half -/

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- **The entailed half of the test.**  If the tolerance is below the baseline's floor, then
every model with at most `baselineK` components -- whatever its parameters, however long it
was trained, on however much data -- misses the tolerance on this system.  Nothing empirical
can be learned from observing this; it is what makes the *other* half of the test
interpretable. -/
theorem baseline_must_fail (S : SystemSpec) (hB : S.BaselineExcluded) {g : Fin S.m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ S.baselineK) :
    ((S.eps : ℚ) : ℝ) < Ens.ell1 M (target S.profile g) := by
  have hfloor : minErr S.profile S.baselineK ≤ Ens.ell1 M (target S.profile g) :=
    ell1_ge_two_tail S.profile hg hM
  have hval : minErr S.profile S.baselineK = ((S.floorAt S.baselineK : ℚ) : ℝ) :=
    minErr_ofRat S.q S.qpos S.qanti S.qsum S.baselineK
  have hcast : ((S.eps : ℚ) : ℝ) < ((S.floorAt S.baselineK : ℚ) : ℝ) := by
    exact_mod_cast hB
  linarith [hval ▸ hfloor]

/-- At the threshold the floor is zero, and the explicit truncated model attains it: the
design rule asks for something achievable. -/
theorem threshold_model_attains (S : SystemSpec) (hm : 0 < S.m) {g : Fin S.m → X}
    (hg : Function.Injective g) :
    Ens.ell1 (truncModel S.profile hm le_rfl g) (target S.profile g) = 0 := by
  rw [minErr_eq S.profile hm le_rfl hg]
  exact (minErr_eq_zero_iff S.profile).2 le_rfl

/-! ## Measurement error in the reported populations

Reported populations carry error bars.  The admissibility condition is therefore
strengthened: the tolerance must sit below the floor by more than twice the `ℓ¹` uncertainty
of the reported populations.  Then the predicted failure of the baseline holds against the
*true* populations, whatever they are within the error bars. -/

/-- Admissibility with an error budget `eta` on the reported populations. -/
def SystemSpec.BaselineExcludedRobust (S : SystemSpec) (eta : ℚ) : Prop :=
  S.eps + 2 * eta < S.floorAt S.baselineK

/-- **The predicted baseline failure survives the error bars.**  If the tolerance lies below
the reported floor by more than `2 eta`, then for *every* true population profile within
`eta` in `ℓ¹` of the reported one, no model with the baseline's component count reaches the
tolerance. -/
theorem baseline_must_fail_robust (S : SystemSpec) {eta : ℚ}
    (hB : S.BaselineExcludedRobust eta) (Q : Profile S.m)
    (hQ : ∑ i, |S.profile.w i - Q.w i| ≤ ((eta : ℚ) : ℝ)) {g : Fin S.m → X}
    (hg : Function.Injective g) {M : Ens X} (hM : M.card ≤ S.baselineK) :
    ((S.eps : ℚ) : ℝ) < Ens.ell1 M (target Q g) := by
  have hfloor : minErr Q S.baselineK ≤ Ens.ell1 M (target Q g) :=
    ell1_ge_two_tail Q hg hM
  have hpert : |minErr S.profile S.baselineK - minErr Q S.baselineK| ≤ 2 * ((eta : ℚ) : ℝ) :=
    minErr_perturb S.profile Q hQ
  have hval : minErr S.profile S.baselineK = ((S.floorAt S.baselineK : ℚ) : ℝ) :=
    minErr_ofRat S.q S.qpos S.qanti S.qsum S.baselineK
  have hcast : ((S.eps : ℚ) : ℝ) + 2 * ((eta : ℚ) : ℝ) < ((S.floorAt S.baselineK : ℚ) : ℝ) := by
    exact_mod_cast hB
  have habs := abs_le.1 hpert
  rw [hval] at habs
  linarith [habs.1, habs.2]

/-! ## The pre-registered criterion, and its refutability -/

/-- The outcome of the experiment: the achieved `ℓ¹` error of the under-capacity baseline and
of the threshold-respecting model, on the same data with the same training budget. -/
structure Outcome where
  /-- error achieved by the practitioner's fixed-`baselineK` model -/
  errBaseline : ℚ
  /-- error achieved by the model built at or above the threshold -/
  errThreshold : ℚ

/-- Outcomes that are consistent with the proved theory: the baseline's error is at least its
floor, and errors are nonnegative.  Anything violating this indicates an error in the
measurement or the pipeline, not a refutation of the theory. -/
def Consistent (S : SystemSpec) (o : Outcome) : Prop :=
  S.floorAt S.baselineK ≤ o.errBaseline ∧ 0 ≤ o.errThreshold

/-- **The pre-registered success criterion.**  The under-capacity baseline misses the
tolerance and the threshold-respecting model meets it. -/
def Confirms (S : SystemSpec) (o : Outcome) : Prop :=
  S.eps < o.errBaseline ∧ o.errThreshold ≤ S.eps

instance (S : SystemSpec) (o : Outcome) : Decidable (Confirms S o) := by
  unfold Confirms; infer_instance

instance (S : SystemSpec) (o : Outcome) : Decidable (Consistent S o) := by
  unfold Consistent; infer_instance

/-- **The test is genuinely falsifiable.**  For every admissible system there are outcomes
fully consistent with everything proved here on which the pre-registered criterion fails: the
threshold-respecting model may simply not fit.  No theorem in this development entails that
the experiment succeeds. -/
theorem confirms_refutable (S : SystemSpec) (heps : 0 ≤ S.eps) :
    ∃ o : Outcome, Consistent S o ∧ ¬ Confirms S o := by
  refine ⟨⟨S.floorAt S.baselineK, S.eps + 1⟩, ⟨le_rfl, by linarith⟩, ?_⟩
  rintro ⟨-, h2⟩
  simp only at h2
  linarith

/-- **The other half is entailed, and only that half.**  Given admissibility, the first
conjunct of the criterion holds automatically for a genuine `ℓ¹` error of an under-capacity
model, so all the empirical content of the test sits in the second conjunct. -/
theorem confirms_iff_threshold_fits (S : SystemSpec) (o : Outcome)
    (hcons : Consistent S o) (hB : S.BaselineExcluded) :
    Confirms S o ↔ o.errThreshold ≤ S.eps := by
  constructor
  · rintro ⟨-, h⟩; exact h
  · intro h
    exact ⟨lt_of_lt_of_le hB hcons.1, h⟩

/-! ## Worked instantiations

The populations below are **stipulated illustrations**, not measurements: they show what a
filled-in record looks like and what the arithmetic then decides.  A real record replaces
them with populations from an independent experiment, as specified in `PREREGISTRATION.md`. -/

/-- Populations of illustration A: five states, `0.40, 0.25, 0.15, 0.12, 0.08`. -/
def qA : Fin 5 → ℚ := ![2/5, 1/4, 3/20, 3/25, 2/25]

/-- Populations of illustration B: seven states with a long tail. -/
def qB : Fin 7 → ℚ := ![3/10, 1/5, 3/20, 1/10, 1/10, 1/10, 1/20]

/-- Populations of illustration C: four states, one dominant. -/
def qC : Fin 4 → ℚ := ![9/10, 3/50, 1/50, 1/50]

/-- Illustration A: five populated states, tolerance `0.10`, baseline a fixed three-mode
mixture. -/
def panelA : SystemSpec where
  m := 5
  q := qA
  qpos := by intro i; fin_cases i <;> norm_num [qA]
  qanti := by intro i j h; fin_cases i <;> fin_cases j <;> revert h <;> norm_num [qA, Fin.le_def]
  qsum := by simp [qA, Fin.sum_univ_five]; norm_num
  eps := 1/10
  baselineK := 3

/-- Illustration B: seven populated states, same tolerance and same baseline. -/
def panelB : SystemSpec where
  m := 7
  q := qB
  qpos := by intro i; fin_cases i <;> norm_num [qB]
  qanti := by intro i j h; fin_cases i <;> fin_cases j <;> revert h <;> norm_num [qB, Fin.le_def]
  qsum := by simp [qB, Fin.sum_univ_seven]; norm_num
  eps := 1/10
  baselineK := 3

/-- Illustration C: four populated states, one dominant.  Here the same baseline is *not*
excluded -- the theory declines to predict a failure, which is what a filter that does real
work looks like. -/
def panelC : SystemSpec where
  m := 4
  q := qC
  qpos := by intro i; fin_cases i <;> norm_num [qC]
  qanti := by intro i j h; fin_cases i <;> fin_cases j <;> revert h <;> norm_num [qC, Fin.le_def]
  qsum := by simp [qC, Fin.sum_univ_four]; norm_num
  eps := 1/10
  baselineK := 3

lemma panelA_floorAt (k : ℕ) : panelA.floorAt k = minErrRat qA k := rfl
lemma panelB_floorAt (k : ℕ) : panelB.floorAt k = minErrRat qB k := rfl
lemma panelC_floorAt (k : ℕ) : panelC.floorAt k = minErrRat qC k := rfl

/-- The floor of a three-component model on illustration A: `0.4` in `ℓ¹`, four times the
pre-registered tolerance. -/
theorem panelA_floor_three : panelA.floorAt 3 = 2/5 := by
  rw [panelA_floorAt]; simp [minErrRat, tailRat, qA, Fin.sum_univ_five]; norm_num

theorem panelA_baseline_excluded : panelA.BaselineExcluded := by
  show (1/10 : ℚ) < panelA.floorAt 3
  rw [panelA_floor_three]; norm_num

theorem panelB_baseline_excluded : panelB.BaselineExcluded := by
  show (1/10 : ℚ) < panelB.floorAt 3
  rw [panelB_floorAt]
  simp [minErrRat, tailRat, qB, Fin.sum_univ_seven]
  norm_num

/-- The rule does not fire everywhere: on illustration C a three-component model is not
excluded by the floor, so no failure is predicted there. -/
theorem panelC_baseline_not_excluded : ¬ panelC.BaselineExcluded := by
  show ¬ (1/10 : ℚ) < panelC.floorAt 3
  rw [panelC_floorAt]
  simp [minErrRat, tailRat, qC, Fin.sum_univ_four]
  norm_num

/-- The full predicted curve for illustration A: the attainable error at `k = 0,…,5`.  Every
entry is fixed before the experiment, from the populations alone. -/
theorem panelA_curve :
    (panelA.floorAt 0, panelA.floorAt 1, panelA.floorAt 2, panelA.floorAt 3,
      panelA.floorAt 4, panelA.floorAt 5)
      = (2, 6/5, 7/10, 2/5, 4/25, 0) := by
  simp [panelA_floorAt, minErrRat, tailRat, qA, Fin.sum_univ_five, Prod.mk.injEq]
  norm_num

/-! ## An end-to-end dry run

Before the protocol is executed on real data it is worth checking that the pipeline does what
it claims on a target whose populations are exactly the record's -- a synthetic case where the
answer is known.  On illustration A the three-component baseline provably misses the
tolerance and the model built at the threshold provably meets it, and the pre-registered
criterion returns "confirm" on the resulting outcome.  This is a validation of the scoring
machinery, not evidence about any real system. -/

/-- **Dry run on illustration A.**  Against a target with exactly the record's populations,
no three-component model reaches the tolerance `0.1`, while the model built at the threshold
has error exactly `0`. -/
theorem panelA_dry_run {g : Fin panelA.m → X} (hg : Function.Injective g) :
    (∀ M : Ens X, M.card ≤ 3 → (1/10 : ℝ) < Ens.ell1 M (target panelA.profile g)) ∧
      Ens.ell1 (truncModel panelA.profile (show 0 < panelA.m by decide) le_rfl g)
        (target panelA.profile g) = 0 := by
  refine ⟨fun M hM => ?_, threshold_model_attains panelA (show 0 < panelA.m by decide) hg⟩
  have := baseline_must_fail panelA panelA_baseline_excluded hg hM
  norm_num [panelA] at this
  exact this

/-- The pre-registered criterion is satisfiable: on the outcome the dry run produces -- the
baseline at its floor `0.4`, the threshold model at `0` -- it returns "confirm".  Together
with `confirms_refutable`, the criterion is neither vacuous nor unfalsifiable. -/
theorem panelA_dry_run_confirms : Confirms panelA ⟨2/5, 0⟩ := by
  constructor
  · show (1/10 : ℚ) < 2/5
    norm_num
  · show (0 : ℚ) ≤ 1/10
    norm_num

/-! ## Panel level: the same rule across several systems

One system, one threshold, one confirmation is a design rule.  What raises the stakes is a
*panel* of systems with different independently known state counts, scored under one
pre-registered criterion: the rule must fire where the theory says it fires and stay silent
where it does not.  The definitions below fix that scoring before any data is seen. -/

/-- A pre-registered panel of systems. -/
structure Panel where
  /-- number of systems in the panel -/
  n : ℕ
  /-- the pre-registered record of each system -/
  spec : Fin n → SystemSpec

namespace Panel

variable (Pa : Panel)

/-- The panel is admissible for the test if the baseline is excluded on every system: the
theory commits to a predicted failure everywhere before any data is seen. -/
def Admissible : Prop := ∀ i, (Pa.spec i).BaselineExcluded

/-- The panel-level success criterion: every system confirms. -/
def Confirms (o : Fin Pa.n → Outcome) : Prop := ∀ i, Prereg.Confirms (Pa.spec i) (o i)

/-- Outcomes consistent with the proved theory on every system. -/
def Consistent (o : Fin Pa.n → Outcome) : Prop := ∀ i, Prereg.Consistent (Pa.spec i) (o i)

/-- **All the empirical content of the panel test is in the threshold-respecting models.**
Given admissibility and consistency, the panel confirms exactly when every
threshold-respecting model meets its tolerance; the baseline failures are theorems. -/
theorem confirms_iff (o : Fin Pa.n → Outcome) (hcons : Pa.Consistent o)
    (hadm : Pa.Admissible) :
    Pa.Confirms o ↔ ∀ i, (o i).errThreshold ≤ (Pa.spec i).eps := by
  constructor
  · intro h i; exact (h i).2
  · intro h i
    exact (confirms_iff_threshold_fits (Pa.spec i) (o i) (hcons i) (hadm i)).2 (h i)

/-- **A panel test can fail on any single system.**  For every admissible panel with at least
one system and nonnegative tolerances there are consistent outcomes that refute the
pre-registered criterion. -/
theorem refutable (hn : 0 < Pa.n) (heps : ∀ i, 0 ≤ (Pa.spec i).eps) :
    ∃ o : Fin Pa.n → Outcome, Pa.Consistent o ∧ ¬ Pa.Confirms o := by
  classical
  refine ⟨fun i => ⟨(Pa.spec i).floorAt (Pa.spec i).baselineK, (Pa.spec i).eps + 1⟩,
    fun i => ⟨le_rfl, by linarith [heps i]⟩, ?_⟩
  intro hcon
  have h := (hcon ⟨0, hn⟩).2
  simp only at h
  linarith

end Panel

/-- The worked three-system panel. -/
def illustrativePanel : Panel := ⟨3, ![panelA, panelB, panelC]⟩

/-- **The rule is selective on the worked panel**: it predicts failure of the three-component
baseline on the first two systems and predicts nothing on the third.  A rule that fired
everywhere would carry no information; this one does not. -/
theorem illustrativePanel_selective :
    (illustrativePanel.spec ⟨0, by norm_num [illustrativePanel]⟩).BaselineExcluded ∧
      (illustrativePanel.spec ⟨1, by norm_num [illustrativePanel]⟩).BaselineExcluded ∧
      ¬ (illustrativePanel.spec ⟨2, by norm_num [illustrativePanel]⟩).BaselineExcluded :=
  ⟨panelA_baseline_excluded, panelB_baseline_excluded, panelC_baseline_not_excluded⟩

end Prereg
end IDR
