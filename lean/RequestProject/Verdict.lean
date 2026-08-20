/-
# The general verdict: the shape of any correct model of intrinsic disorder

`RequestProject.Answer` answered the original question for *structure-aware* models.  This
file states the architecture-free version of that answer, in two capstone theorems.

* `model_must_be` collects the positive constraints.  They fix the model's *type*: it
  predicts, from sequence **and** thermodynamic context, a probability distribution over
  conformations, drawn from a class that realises every ensemble, whose output space is
  convex, whose capacity grows with the target, whose parameters may equivalently be an
  energy function (identifiable only up to an additive constant), and whose weights are
  fitted consistently by maximum likelihood.
* `model_cannot_be` collects the impossibilities, each holding against *every* model of
  the stated shape, no matter how it is built or trained -- and, where the notion makes
  sense, quantitatively, so that "approximately right" is ruled out too.

Together they say: the design question has essentially one answer, and it is a statement
about what the model *outputs*, not about how it computes.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.Binding
import RequestProject.FiniteData
import RequestProject.ModelNature
import RequestProject.EnergyModels
import RequestProject.CoarseGraining
import RequestProject.Dynamics
import RequestProject.Answer

namespace IDR

open Finset
open scoped Classical

/-! ## Concrete targets used to witness the impossibilities -/

/-- The `k + 1` distinct one-dimensional conformations `0, 1, ..., k`. -/
noncomputable def ladderConf (k : ℕ) : Fin (k + 1) → Conf 1 := fun l _ => (l : ℝ)

lemma ladderConf_injective (k : ℕ) : Function.Injective (ladderConf k) := by
  intro a b hab
  have h1 : ((a : ℕ) : ℝ) = ((b : ℕ) : ℝ) := congrFun hab 0
  exact Fin.ext (by exact_mod_cast h1)

/-- A family of observationally distinct targets, one per natural number. -/
noncomputable def ladderTarget (n : ℕ) : Ens (Conf 1) := Ens.dirac (fun _ => (n : ℝ))

lemma ladderTarget_not_same {n m : ℕ} (h : n ≠ m) :
    ¬ (ladderTarget n).Same (ladderTarget m) := by
  intro hs
  have hnm := hs (fun c => c 0)
  simp only [ladderTarget, Ens.expect_dirac] at hnm
  exact h (by exact_mod_cast hnm)

lemma pmOne_prob_neg_one : pmOne.prob ![(-1 : ℝ)] = 1 / 2 := by
  rw [Ens.prob, pmOne_expect, if_pos rfl, if_neg (Ne.symm neg_one_ne_one)]
  norm_num

lemma pmOne_prob_one : pmOne.prob ![(1 : ℝ)] = 1 / 2 := by
  rw [Ens.prob, pmOne_expect, if_pos rfl, if_neg neg_one_ne_one]
  norm_num

/-! ## What the model must be -/

/-- **The positive specification, architecture-free.**

1. *The output object.*  Every ensemble is a convex combination of single structures, so
   the quantity to be predicted lives in the simplex over conformation space: the model's
   output type must be a distribution, and single structures are exactly its extreme
   points.
2. *Sufficiency.*  A predictor that takes the full input (sequence together with
   thermodynamic context) and returns such a distribution can be exactly right on every
   input and every observable.  Nothing else has to change.
3. *The exact criterion.*  A class of outputs solves every prediction problem if and only
   if it realises every ensemble up to observational equality.  Expressivity of the output
   class is the whole content of the design problem; the computational architecture is
   unconstrained.
4. *Convexity.*  Such a class must be able to represent every population ratio between any
   two of its outputs -- "choose one of finitely many templates" is not enough.
5. *Capacity.*  A correct output has at least as many components as the target has
   populated conformations, so capacity must scale with the breadth of the disorder.
6. *Energy parametrisation.*  The weights may equivalently be produced by an energy
   function at any fixed temperature; nothing is lost, and nothing is gained, by that
   reparametrisation.
7. *Identifiability.*  Two energies give the same predictions exactly when they differ by
   an additive constant, so only energy differences are learnable or testable.
8. *Fitting.*  The weights are fitted consistently by maximum likelihood: the true weights
   are the unique minimiser of the expected negative log-likelihood. -/
theorem model_must_be :
    (∀ (X : Type) (E : Ens X) (f : X → ℝ),
        E.expect f = ∑ j, E.w j * (Ens.dirac (E.pt j)).expect f) ∧
    (∀ (I X : Type) (T : I → Ens X), ∃ A : I → Ens X, Solves A T) ∧
    (∀ (X : Type) (M : Set (Ens X)), Expressive M ↔
        ∀ (I : Type) (T : I → Ens X), ∃ A : I → Ens X,
          (∀ i, A i ∈ M) ∧ ∀ i, (A i).Same (T i)) ∧
    (∀ (X : Type) (M : Set (Ens X)), Expressive M → ∀ (E F : Ens X) (t : ℝ),
        0 ≤ t → t ≤ 1 → ∃ m ∈ M, ∀ f : X → ℝ,
          m.expect f = t * E.expect f + (1 - t) * F.expect f) ∧
    (∀ (X : Type) (A T : Ens X), A.Same T → ∀ (L : ℕ) (g : Fin L → X),
        Function.Injective g → (∀ l, 0 < T.prob (g l)) → L ≤ A.card) ∧
    (∀ (X : Type) (E : Ens X) (hpos : 0 < E.card), Function.Injective E.pt →
        (∀ j, 0 < E.w j) → ∀ beta : ℝ, beta ≠ 0 →
          ∃ U : X → ℝ, (energyEns hpos E.pt beta U).Same E) ∧
    (∀ (X : Type) (m : ℕ) (hm : 0 < m) (lib : Fin m → X), Function.Injective lib →
        ∀ beta : ℝ, beta ≠ 0 → ∀ U V : X → ℝ,
          (energyEns hm lib beta U).Same (energyEns hm lib beta V) →
            ∃ c : ℝ, ∀ j, U (lib j) = V (lib j) + c) ∧
    (∀ (m : ℕ) (p q : Fin m → ℝ), (∀ j, 0 ≤ p j) → ∑ j, p j = 1 → (∀ j, 0 < q j) →
        ∑ j, q j = 1 → crossEntropy p p ≤ crossEntropy p q ∧
          (crossEntropy p q = crossEntropy p p ↔ p = q)) := by
  refine ⟨fun _ E f => Ens.expect_eq_sum_dirac E f,
    fun _ _ T => solvable_by_conditional_ensembles T,
    fun _ M => expressive_iff_solves_all M,
    fun _ _ hM E F t ht0 ht1 => expressive_closed_under_mix hM E F ht0 ht1,
    fun _ _ _ h L g hg hpos => Ens.card_le_of_same h g hg hpos,
    fun _ E hpos hinj hw beta hbeta => exists_energy_representation E hpos hinj hw hbeta,
    fun _ _ hm lib hlib beta hbeta U V h =>
      energy_unique_up_to_const hm lib hlib hbeta U V h,
    fun _ _ _ hp hps hq hqs => crossEntropy_min_iff hp hps hq hqs⟩

/-! ## What the model cannot be -/

/-- **The impossibilities, architecture-free.**  Each clause rules out a whole class of
models, against a target that a disordered region actually realises.

1. *Not a single structure -- and not even approximately one.*  Against a two-state target
   every single-structure output is off by at least the population it misses, uniformly
   over observables bounded by one.  Confidences and error bars do not change this.
2. *Not of bounded capacity -- and not even approximately.*  A model emitting at most `k`
   structures is off by at least `1/(k+1)` on the uniform ensemble over `k+1`
   conformations; drawing more samples from it does not converge.
3. *Not routed through a finite internal code.*  If the target-dependence of the model
   passes through a finite set of internal states -- a disorder class, a chosen template,
   a discrete vocabulary -- it fails on one of any `card + 1` distinct targets, whatever
   the decoder.
4. *Not context-blind.*  One sequence with two thermodynamic contexts already defeats any
   predictor of the sequence alone.
5. *Not fitted to finitely many statistics.*  No finite family of observables determines
   the ensemble, so a model whose target-dependence is through finitely many measured or
   predicted numbers is wrong on some target.
6. *Not factorised over parts of the chain, and not mean-field.*  A model that predicts
   segments independently, or that carries no correlations, has identically zero coupling
   and so cannot reproduce a correlated target.
7. *Not a reweighting of a fixed reference ensemble.*  Boltzmann reweighting only
   redistributes population among already-populated conformations, so it cannot produce
   induced fit.
8. *Not able to impose hard constraints through energies.*  At any finite temperature every
   library conformation retains positive population, so what the model must never produce
   has to be excluded from its support, not penalised.
9. *Not exactly equal to a continuous truth.*  A finitely supported model always differs
   maximally from an atomless conformational distribution on some bounded observable --
   which is why correctness has to be measured approximately, and why clauses 1 and 2,
   which are quantitative, are the relevant ones. -/
theorem model_cannot_be :
    (∀ (eps : ℝ), eps < 1 / 2 → ∀ a : Conf 1, ¬ ApproxSame eps pmOne (Ens.dirac a)) ∧
    (∀ (k : ℕ) (M : Ens (Conf 1)), M.card ≤ k → ∀ eps : ℝ, eps < 1 / (k + 1) →
        ¬ ApproxSame eps M (unif (Nat.succ_pos k) (ladderConf k))) ∧
    (∀ (Z : Type) [Fintype Z] (R : ℕ → Z) (D : Z → Ens (Conf 1)) (A : ℕ → Ens (Conf 1)),
        (∀ i, A i = D (R i)) → ∃ i, ¬ (A i).Same (ladderTarget i)) ∧
    (∃ T : Unit × Bool → Ens (Conf 1), ∀ A : Unit × Bool → Ens (Conf 1),
        (∀ (s : Unit) (c c' : Bool), A (s, c) = A (s, c')) → ¬ Solves A T) ∧
    (∀ (m : ℕ) (f : Fin m → (Conf 1 → ℝ)) (A : Ens (Conf 1) → Ens (Conf 1)),
        (∀ E F : Ens (Conf 1), AgreeOn {g : Conf 1 → ℝ | ∃ i, g = f i} E F → A E = A F) →
        ∃ E : Ens (Conf 1), ¬ (A E).Same E) ∧
    ((∀ M : Ens (ℝ × ℝ), Factorised M → ¬ M.Same corrPair) ∧
      (∀ M : Ens (Conf 2), MeanField M → ¬ M.Same corrEns)) ∧
    (∀ (X : Type) (E B : Ens X) (x : X), 0 < B.prob x → E.prob x = 0 →
        ∀ (beta : ℝ) (U : X → ℝ), ¬ (E.boltzmann beta U).Same B) ∧
    (∀ (X : Type) (m : ℕ) (hm : 0 < m) (lib : Fin m → X) (beta : ℝ) (U : X → ℝ)
        (T : Ens X) (j : Fin m), T.prob (lib j) = 0 →
        ¬ (energyEns hm lib beta U).Same T) ∧
    (∀ (mu : MeasureTheory.Measure ℝ), (∀ x : ℝ, mu {x} = 0) → ∀ E : Ens ℝ,
        ∃ f : ℝ → ℝ, (∀ x, 0 ≤ f x ∧ f x ≤ 1) ∧ Measurable f ∧
          ∫ x, f x ∂mu = 0 ∧ E.expect f = 1) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ⟨fun M hM => no_factorised_captures_corrPair M hM,
      fun M hM => no_meanField_captures_corrEns M hM⟩,
    fun _ E B x hB hE beta U => no_boltzmann_of_unpopulated E B hB hE beta U,
    fun _ _ hm lib beta U T j hT => no_energy_model_excludes hm lib beta U hT,
    fun mu hmu E => no_finite_ensemble_eq_atomless mu hmu E⟩
  · intro eps heps a
    refine not_approx_dirac (x := ![(-1 : ℝ)]) (y := ![(1 : ℝ)])
      (by rw [pmOne_prob_neg_one]; norm_num) (by rw [pmOne_prob_one]; norm_num)
      neg_one_ne_one ?_ a
    rw [pmOne_prob_neg_one, pmOne_prob_one]
    simpa using heps
  · intro k M hM eps heps
    exact not_approx_of_bounded_capacity hM (ladderConf k) (ladderConf_injective k)
      (by linarith)
  · intro Z _ R D A hA
    refine no_finite_representation (T := ladderTarget) R D hA
      (L := Fintype.card Z + 1) (by omega) (fun l => (l : ℕ)) ?_
    intro l l' hll'
    refine ladderTarget_not_same ?_
    intro hc
    exact hll' (Fin.ext hc)
  · obtain ⟨-, -, h, -⟩ := structure_aware_verdict
    exact h
  · intro m f A hA
    exact finite_statistics_predictor_fails f A hA

/-! ## The scope of the model: resolution, time, and where the correlations live -/

/-- **What the model must be about, not only what it must output.**

1. *Resolution, necessary side.*  Correctness at full resolution implies correctness for
   every coarse descriptor, so any descriptor mismatch refutes a model.
2. *Resolution, insufficient side.*  Two observationally different ensembles can share a
   coarse description, so agreement at reduced resolution is never evidence of
   correctness.
3. *Supervision.*  If the descriptor merges even one pair of conformations, a model whose
   dependence on the target is only through it is wrong on some target -- a lossy
   descriptor cannot serve as the training signal for a claim made at full resolution.
4. *Time.*  Equilibrium populations do not determine kinetics: any predictor of dynamics
   that is a function of the equilibrium distribution alone is wrong about some system
   whose populations it gets exactly right.
5. *The repair for time.*  Because the framework is stated over an arbitrary conformation
   space, taking that space to be trajectory space gives the same universality: a model
   that outputs a distribution over trajectories can be exactly right, and every negative
   result above applies to it verbatim.
6. *Where the correlations live.*  Every two-part ensemble is a mixture of factorised
   ones.  Independent per-part prediction is fatal, but per-part prediction *conditioned on
   a latent state*, mixed over that state, is fully general: the latent variable is exactly
   the carrier of the correlations. -/
theorem model_scope :
    (∀ (X Y : Type) (h : X → Y) (E F : Ens X), E.Same F → (E.map h).Same (F.map h)) ∧
    (∃ (h : Conf 1 → ℝ) (E F : Ens (Conf 1)), (E.map h).Same (F.map h) ∧ ¬ E.Same F) ∧
    (∀ (X Y : Type) (h : X → Y) (x y : X), x ≠ y → h x = h y →
        ∀ A : Ens X → Ens X,
          (∀ E F : Ens X, AgreeOn (PullbackObs h) E F → A E = A F) →
            ∃ E : Ens X, ¬ (A E).Same E) ∧
    (∀ A : (Fin 2 → ℝ) → Kernel 2, ∃ K : Kernel 2, Stationary halfHalf K ∧
        ∃ f g : Fin 2 → ℝ,
          twoTime halfHalf (A halfHalf) f g ≠ twoTime halfHalf K f g) ∧
    (∀ (I X : Type) (T : ℕ) (Tgt : I → Ens (Fin T → X)),
        ∃ A : I → Ens (Fin T → X), Solves A Tgt) ∧
    (∀ (X Y : Type) (M : Ens (X × Y)) (f : X × Y → ℝ),
        M.expect f
          = ∑ j, M.w j * ((Ens.dirac (M.pt j).1).prod (Ens.dirac (M.pt j).2)).expect f) :=
  ⟨fun _ _ h _ _ hEF => map_same_of_same h hEF,
   exists_same_coarse_ne_fine,
   fun _ _ _ _ _ hxy hh A hA => coarse_trained_model_fails hxy hh A hA,
   fun A => no_static_model_predicts_kinetics A,
   fun _ _ T Tgt => trajectory_ensembles_universal T Tgt,
   fun _ _ M f => mixture_of_products_universal M f⟩

end IDR
