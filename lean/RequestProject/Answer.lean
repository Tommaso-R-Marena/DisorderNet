/-
# The verdict: what a model of intrinsic disorder must be, and what can never work

This file assembles the previous ones into a direct answer to the question

> will structure-aware models ever be able to fully predict intrinsically disordered
> regions and their binding?

A *prediction problem* is a family of target ensembles `T : I → Ens X` indexed by the
inputs `I` the model is given (sequence, and possibly context); a *predictor* is a map
`A : I → Ens X`, and `Solves A T` says that `A` is right on every input, on every
observable.

**Negative side.**  Each of the following restrictions, *on its own*, makes some prediction
problem unsolvable:

* `not_solves_of_deterministic_output` -- the model outputs one structure per input.
  Together with `Ens.deterministic_iff_var_zero` this is sharp: a single-structure output
  is exactly right precisely on the rigidly ordered targets, and wrong on every disordered
  one, no matter how the structure is chosen.
* `not_solves_of_bounded_capacity` -- the model outputs at most `k` structures (templates,
  decoys, sampled models) per input, for a fixed `k`.
* `not_solves_of_context_blind` -- the model reads the sequence but not the thermodynamic
  context.
* `exists_failure_of_not_separating` -- the model is fitted to, or evaluated on, a family
  of statistics that does not determine the ensemble (coordinates, distograms,
  per-residue marginals, means and variances: `not_separates_linearObs`,
  `not_separates_distanceObs`, `not_separates_marginalObs`,
  `not_separates_quadraticObs`).

**Positive side.**  `solvable_by_conditional_ensembles` shows that the obstructions above
are exactly the features that have to be dropped: a predictor whose *output* is a
probability distribution over conformations, whose *capacity* is allowed to grow with the
target, and whose *input* includes the context, can be exactly right on every observable.
`exists_latentModel_captures` gives the same statement in the latent-variable form actually
used by generative models, and `crossEntropy_min_iff` shows that fitting the weights of
such a model by maximum likelihood is consistent.

So the answer is: a structure-aware model in the strict sense -- sequence in, one
structure (with or without a confidence) out -- can *never* fully predict a disordered
region or its binding, and the reason is not a shortage of data or parameters but the
`Solves`-level obstructions above.  A model that keeps the structural machinery but
predicts a *context-conditional distribution* of unbounded capacity can, in principle, be
exactly right.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.Binding
import RequestProject.FiniteData

namespace IDR

open Finset
open scoped Classical

variable {I X : Type*}

/-- A predictor `A` solves the prediction problem `T` when, on every input, its output is
observationally identical to the target ensemble. -/
def Solves (A : I → Ens X) (T : I → Ens X) : Prop := ∀ i, (A i).Same (T i)

/-- **Ensemble models are universal.**  Every prediction problem is solved exactly by some
context-conditional ensemble predictor. -/
theorem solvable_by_conditional_ensembles (T : I → Ens X) : ∃ A : I → Ens X, Solves A T :=
  ⟨T, fun _ => Ens.Same.refl _⟩

lemma deterministic_of_same {E F : Ens X} (h : E.Same F) (hE : E.Deterministic) :
    F.Deterministic := by
  obtain ⟨x, hx⟩ := hE
  exact ⟨x, h.symm.trans hx⟩

/-- **The design specification, positively.**  For every family of target ensembles
indexed by the model's inputs (sequence *and* thermodynamic context):

1. a predictor that outputs a conditional distribution over conformations is exactly right
   on every input and every observable;
2. any such correct predictor must use at least as many mixture components as the target
   has populated conformations, so capacity has to scale with the breadth of the disorder
   rather than being fixed by the architecture;
3. the mixture weights can be fitted consistently by maximum likelihood: the true weights
   are the unique minimiser of the expected negative log-likelihood.

Together these say what a model of an intrinsically disordered region has to be, and how
it is fitted. -/
theorem disorder_model_design {C : Type*} (T : C → Ens X) :
    (∃ A : C → Ens X, Solves A T) ∧
    (∀ A : C → Ens X, Solves A T → ∀ (c : C) (L : ℕ) (g : Fin L → X),
      Function.Injective g → (∀ l, 0 < (T c).prob (g l)) → L ≤ (A c).card) ∧
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∀ j, 0 ≤ p j) → ∑ j, p j = 1 → (∀ j, 0 < q j) →
      ∑ j, q j = 1 → crossEntropy p p ≤ crossEntropy p q ∧
        (crossEntropy p q = crossEntropy p p ↔ p = q)) :=
  ⟨solvable_by_conditional_ensembles T,
   fun _ hS c _ g hg hpos => Ens.card_le_of_same (hS c) g hg hpos,
   fun _ _ _ hp hps hq hqs => crossEntropy_min_iff hp hps hq hqs⟩

/-! ## The three structural obstructions -/

/-- **Single-structure output.**  A predictor that returns one conformation per input
cannot solve a problem whose target populates two different conformations. -/
theorem not_solves_of_deterministic_output {T A : I → Ens X} (hA : ∀ i, (A i).Deterministic)
    {i : I} {x y : X} (hx : 0 < (T i).prob x) (hy : 0 < (T i).prob y) (hxy : x ≠ y) :
    ¬ Solves A T := by
  intro hS
  exact Ens.not_deterministic_of_two_points hx hy hxy
    (deterministic_of_same (hS i) (hA i))

/-- **Bounded capacity.**  A predictor limited to `k` structures per input cannot solve a
problem whose target populates `k + 1` distinct conformations. -/
theorem not_solves_of_bounded_capacity {k : ℕ} {T A : I → Ens X} (hA : ∀ i, (A i).card ≤ k)
    {i : I} (g : Fin (k + 1) → X) (hg : Function.Injective g)
    (hpos : ∀ l, 0 < (T i).prob (g l)) : ¬ Solves A T := by
  intro hS
  have := Ens.card_le_of_same (hS i) g hg hpos
  have := hA i
  omega

/-- **Context blindness.**  A predictor that ignores the thermodynamic context cannot
solve a problem in which one sequence has two different context-dependent ensembles. -/
theorem not_solves_of_context_blind {S C : Type*} {T A : S × C → Ens X}
    (hA : ∀ (s : S) (c c' : C), A (s, c) = A (s, c')) {s : S} {c₁ c₂ : C}
    (hT : ¬ (T (s, c₁)).Same (T (s, c₂))) : ¬ Solves A T := by
  intro hS
  have h1 : (A (s, c₁)).Same (T (s, c₁)) := hS _
  have h2 : (A (s, c₂)).Same (T (s, c₂)) := hS _
  rw [hA s c₁ c₂] at h1
  exact hT (h1.symm.trans h2)

/-- **A confidence score does not help.**  A predictor that outputs a structure together
with a per-residue confidence (a pLDDT-like number, a predicted error bar) still commits
to one structure; the induced ensemble is a point mass, so the previous theorem applies
verbatim.  A confidence score can *flag* disorder; it cannot *represent* it. -/
theorem confidence_score_does_not_help {T : I → Ens X} (A : I → X × ℝ)
    {i : I} {x y : X} (hx : 0 < (T i).prob x) (hy : 0 < (T i).prob y) (hxy : x ≠ y) :
    ¬ Solves (fun i => Ens.dirac (A i).1) T :=
  not_solves_of_deterministic_output (fun i => Ens.dirac_deterministic (A i).1) hx hy hxy

/-! ## Non-vacuity: the hypotheses are met by real disorder -/

lemma pmOne_not_deterministic : ¬ pmOne.Deterministic := by
  rw [Ens.deterministic_iff_var_zero]
  intro h
  have := h 0
  rw [pmOne_var] at this
  norm_num at this

lemma pmOne_prob_pos_neg_one : 0 < pmOne.prob ![(-1 : ℝ)] :=
  unif_prob_pos _ _ 0

lemma pmOne_prob_pos_one : 0 < pmOne.prob ![(1 : ℝ)] :=
  unif_prob_pos _ _ 1

lemma neg_one_ne_one : ![(-1 : ℝ)] ≠ ![(1 : ℝ)] := by
  intro h
  have := congrFun h 0
  norm_num at this

/-! ## The verdict -/

/-- **Structure-aware models and intrinsic disorder: the verdict.**

1. No predictor that outputs a single structure per input solves the disordered target
   `pmOne` -- and by `Ens.deterministic_iff_var_zero` this is sharp: single-structure
   output is right exactly on the rigidly ordered ensembles.
2. For every fixed capacity `k` there is a target that no `k`-structure predictor solves,
   so no fixed architecture size suffices either.
3. There is a prediction problem -- one sequence, two thermodynamic contexts, e.g. free
   and partner-bound -- that no context-blind predictor solves.
4. Matching a mean structure and its fluctuations (or coordinates, distograms, per-residue
   marginals) does not determine the ensemble, so no model fitted only to those statistics
   is right; and more generally *no* finite family of statistics determines it.
5. By contrast, predictors that output a context-conditional probability distribution over
   conformations solve *every* prediction problem exactly.

Points 1-4 are the reasons a structure-aware model can never fully predict a disordered
region or its binding; point 5 says what has to replace it. -/
theorem structure_aware_verdict :
    (∀ A : Unit → Ens (Conf 1), (∀ i, (A i).Deterministic) →
        ¬ Solves A (fun _ => pmOne)) ∧
    (∀ k : ℕ, ∃ T : Unit → Ens (Conf 1), ∀ A : Unit → Ens (Conf 1),
        (∀ i, (A i).card ≤ k) → ¬ Solves A T) ∧
    (∃ T : Unit × Bool → Ens (Conf 1), ∀ A : Unit × Bool → Ens (Conf 1),
        (∀ (s : Unit) (c c' : Bool), A (s, c) = A (s, c')) → ¬ Solves A T) ∧
    (¬ Separates (LinearObs 1) ∧ ¬ Separates (QuadraticObs 1) ∧
      ¬ Separates (MarginalObs 2) ∧ ¬ Separates (DistanceObs 2)) ∧
    (∀ (m : ℕ) (f : Fin m → (Conf 1 → ℝ)),
      ¬ Separates {g : Conf 1 → ℝ | ∃ i, g = f i}) ∧
    (∀ (I : Type) (T : I → Ens (Conf 1)), ∃ A : I → Ens (Conf 1), Solves A T) := by
  refine ⟨?_, ?_, ?_, ⟨not_separates_linearObs, not_separates_quadraticObs,
      not_separates_marginalObs, not_separates_distanceObs⟩,
    fun m f => not_separates_of_finite f,
    fun I T => solvable_by_conditional_ensembles T⟩
  · intro A hA
    exact not_solves_of_deterministic_output (i := ()) hA pmOne_prob_pos_neg_one
      pmOne_prob_pos_one neg_one_ne_one
  · intro k
    refine ⟨fun _ => unif (Nat.succ_pos k) (fun l : Fin (k + 1) => (fun _ => (l : ℝ))), ?_⟩
    intro A hA hS
    have hinj : Function.Injective
        (fun l : Fin (k + 1) => (fun _ => (l : ℝ)) : Fin (k + 1) → Conf 1) := by
      intro a b hab
      have h1 : ((a : ℕ) : ℝ) = ((b : ℕ) : ℝ) := congrFun hab 0
      have h2 : (a : ℕ) = (b : ℕ) := by exact_mod_cast h1
      exact Fin.ext h2
    exact not_solves_of_bounded_capacity (i := ()) hA _ hinj
      (fun l => unif_prob_pos _ _ l) hS
  · refine ⟨fun p => if p.2 then pmOne else Ens.dirac ![(1 : ℝ)], ?_⟩
    intro A hA
    refine not_solves_of_context_blind (s := ()) (c₁ := true) (c₂ := false) hA ?_
    simpa using contexts_can_differ

end IDR
