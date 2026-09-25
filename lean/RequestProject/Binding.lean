/-
# Binding: context dependence, reweighting, and why it is harder still

A disordered region does not have *an* ensemble; it has one ensemble per thermodynamic
context (partner present or absent, post-translational modification, pH, crowding).  This
file formalises three obstructions specific to binding.

* `exists_context_failure` -- **context dependence**.  Any predictor that is a function of
  the sequence alone assigns one ensemble to one sequence; if two contexts give different
  ensembles, the predictor is wrong in at least one of them.  The cure is structural, not
  quantitative: the predictor must take the context as an input
  (`conditional_predictor_exact`).
* `reweight` and `boltzmann` -- binding by thermodynamic reweighting of the free-state
  ensemble.  `reweight_prob_pos_iff` shows that reweighting can only redistribute weight
  among conformations that are *already populated*: this is conformational selection, and
  `no_reweight_of_unpopulated` shows that a reweighting model can never describe induced
  fit, where the complex populates geometry absent from the free state.
* `binding_not_determined_by_two_moments` -- the equilibrium binding constant of a
  disordered region is not a function of its mean structure and fluctuations: two
  ensembles with identical mean and variance have different binding constants.  Predicting
  a mean structure with error bars therefore cannot predict binding.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-! ## Context dependence -/

/-- **A sequence-only predictor must be wrong somewhere.**  If the target ensembles in two
contexts differ, no single ensemble -- no output that ignores the context -- is correct in
both.  This is the formal content of "the same intrinsically disordered sequence folds
differently on different partners, and not at all on its own". -/
theorem exists_context_failure {C : Type*} (E : C → Ens X) {c₁ c₂ : C}
    (h : ¬ (E c₁).Same (E c₂)) (M : Ens X) : ∃ c : C, ¬ M.Same (E c) := by
  by_contra hcon
  push_neg at hcon
  exact h ((hcon c₁).symm.trans (hcon c₂))

/-- The same statement for a predictor that maps sequences to models: if some sequence has
context-dependent behaviour, every sequence-only predictor errs. -/
theorem exists_sequence_context_failure {S C : Type*} (E : S → C → Ens X)
    {s : S} {c₁ c₂ : C} (h : ¬ (E s c₁).Same (E s c₂)) (A : S → Ens X) :
    ∃ (s : S) (c : C), ¬ (A s).Same (E s c) := by
  obtain ⟨c, hc⟩ := exists_context_failure (E s) h (A s)
  exact ⟨s, c, hc⟩

/-- **The repair is to condition on the context.**  A predictor that takes the context as
an input can be exactly right everywhere. -/
theorem conditional_predictor_exact {S C : Type*} (E : S → C → Ens X) :
    ∃ A : S → C → Ens X, ∀ s c, (A s c).Same (E s c) :=
  ⟨E, fun _ _ => Ens.Same.refl _⟩

/-- A concrete pair of contexts with different ensembles: free (disordered) versus bound
(ordered).  So the hypothesis of `exists_context_failure` is satisfiable. -/
theorem contexts_can_differ :
    ¬ (pmOne.Same (Ens.dirac ![(1 : ℝ)])) := by
  intro h
  have := h (fun c => c 0)
  rw [pmOne_expect, Ens.expect_dirac] at this
  norm_num at this

/-! ## Thermodynamic reweighting -/

namespace Ens

lemma exists_w_pos (E : Ens X) : ∃ j, 0 < E.w j := by
  obtain ⟨j, -, hj⟩ : ∃ j ∈ (Finset.univ : Finset (Fin E.card)), E.w j ≠ 0 := by
    refine Finset.exists_ne_zero_of_sum_ne_zero ?_
    rw [E.w_sum]; norm_num
  exact ⟨j, lt_of_le_of_ne (E.w_nonneg j) (Ne.symm hj)⟩

lemma expect_pos (E : Ens X) {g : X → ℝ} (hg : ∀ x, 0 < g x) : 0 < E.expect g := by
  obtain ⟨j, hj⟩ := E.exists_w_pos
  refine Finset.sum_pos' (fun i _ => mul_nonneg (E.w_nonneg i) (le_of_lt (hg _)))
    ⟨j, Finset.mem_univ j, mul_pos hj (hg _)⟩

/-- Reweighting an ensemble by a strictly positive weight function `g` (for binding,
`g x = exp (-β ΔU x)`, the Boltzmann factor of the interaction energy), renormalised. -/
noncomputable def reweight (E : Ens X) (g : X → ℝ) (hg : ∀ x, 0 < g x) : Ens X where
  card := E.card
  pt := E.pt
  w := fun j => E.w j * g (E.pt j) / E.expect g
  w_nonneg := fun j =>
    div_nonneg (mul_nonneg (E.w_nonneg j) (le_of_lt (hg _))) (le_of_lt (E.expect_pos hg))
  w_sum := by
    rw [← Finset.sum_div]
    exact div_self (ne_of_gt (E.expect_pos hg))

/-- Boltzmann reweighting of the free-state ensemble by an interaction energy `U` at
inverse temperature `β`: the standard statistical-mechanical model of binding. -/
noncomputable def boltzmann (E : Ens X) (beta : ℝ) (U : X → ℝ) : Ens X :=
  E.reweight (fun x => Real.exp (-(beta * U x))) (fun _ => Real.exp_pos _)

/-- **Conformational selection.**  Reweighting redistributes population among the
conformations that the free ensemble already visits, and creates nothing new. -/
theorem reweight_prob_pos_iff (E : Ens X) {g : X → ℝ} (hg : ∀ x, 0 < g x) (x : X) :
    0 < (E.reweight g hg).prob x ↔ 0 < E.prob x := by
  rw [prob_pos_iff, prob_pos_iff]
  constructor
  · rintro ⟨j, hj, hpj⟩
    refine ⟨j, ?_, hpj⟩
    by_contra hle
    push_neg at hle
    have hw : E.w j = 0 := le_antisymm hle (E.w_nonneg j)
    simp only [reweight, hw] at hj
    simp at hj
  · rintro ⟨j, hj, hpj⟩
    refine ⟨j, ?_, hpj⟩
    have : 0 < E.w j * g (E.pt j) := mul_pos hj (hg _)
    exact div_pos this (E.expect_pos hg)

theorem boltzmann_prob_pos_iff (E : Ens X) (beta : ℝ) (U : X → ℝ) (x : X) :
    0 < (E.boltzmann beta U).prob x ↔ 0 < E.prob x :=
  E.reweight_prob_pos_iff _ x

end Ens

/-- **Reweighting models cannot describe induced fit.**  If the complex populates a
conformation that the free-state ensemble never visits, then no reweighting whatsoever of
the free-state ensemble -- no choice of interaction energy, no temperature -- reproduces
the bound ensemble. -/
theorem no_reweight_of_unpopulated (E B : Ens X) {x : X} (hB : 0 < B.prob x)
    (hE : E.prob x = 0) (g : X → ℝ) (hg : ∀ y, 0 < g y) :
    ¬ (E.reweight g hg).Same B := by
  intro h
  have h1 : 0 < (E.reweight g hg).prob x := by
    rw [Ens.prob_eq_of_same h]; exact hB
  have h2 : 0 < E.prob x := (E.reweight_prob_pos_iff hg x).1 h1
  exact absurd hE (ne_of_gt h2)

/-- The same for the Boltzmann model of binding. -/
theorem no_boltzmann_of_unpopulated (E B : Ens X) {x : X} (hB : 0 < B.prob x)
    (hE : E.prob x = 0) (beta : ℝ) (U : X → ℝ) : ¬ (E.boltzmann beta U).Same B :=
  no_reweight_of_unpopulated E B hB hE _ _

/-- Induced fit really can happen: a bound state populating geometry absent from the free
ensemble. -/
theorem induced_fit_example :
    0 < (Ens.dirac ![(5 : ℝ)]).prob ![(5 : ℝ)] ∧ pmOne.prob ![(5 : ℝ)] = 0 := by
  have hne1 : ![(-1 : ℝ)] ≠ ![(5 : ℝ)] := by
    intro h
    have := congrFun h 0
    norm_num at this
  have hne2 : ![(1 : ℝ)] ≠ ![(5 : ℝ)] := by
    intro h
    have := congrFun h 0
    norm_num at this
  constructor
  · have h : (Ens.dirac ![(5 : ℝ)]).prob ![(5 : ℝ)] = 1 := by
      simp [Ens.prob]
    rw [h]; norm_num
  · rw [Ens.prob, pmOne_expect]
    simp [hne1, hne2]

/-! ## Binding constants are not functions of the mean and the fluctuations -/

lemma pmOne_mean : pmOne.mean 0 = 0 := by
  rw [Ens.mean, pmOne_expect]; norm_num

lemma pmTwo_mean : pmTwo.mean 0 = 0 := by
  rw [Ens.mean, pmTwo_expect]; norm_num

lemma pmOne_var : pmOne.var 0 = 1 := by
  rw [Ens.var, pmOne_mean, pmOne_expect]; norm_num

lemma pmTwo_var : pmTwo.var 0 = 1 := by
  rw [Ens.var, pmTwo_mean, pmTwo_expect]; norm_num

/-- **Binding is not predictable from a mean structure with error bars.**  There are two
conformational ensembles with the same mean conformation and the same fluctuation about
it, but different equilibrium binding constants `Z = ⟨exp (-β ΔU)⟩` -- here realised by a
strictly positive Boltzmann weight `g`.  A model that outputs a structure and a confidence
therefore cannot, even in principle, predict binding. -/
theorem binding_not_determined_by_two_moments :
    pmOne.mean 0 = pmTwo.mean 0 ∧ pmOne.var 0 = pmTwo.var 0 ∧
      ∃ g : Conf 1 → ℝ, (∀ x, 0 < g x) ∧ pmOne.expect g ≠ pmTwo.expect g := by
  refine ⟨by rw [pmOne_mean, pmTwo_mean], by rw [pmOne_var, pmTwo_var],
    fun c => 1 + (c 0) ^ 4, fun x => by positivity, ?_⟩
  rw [pmOne_expect, pmTwo_expect]
  norm_num

/-! ## Fuzzy complexes: even the bound state need not be a structure -/

/-- **Thermodynamic binding never fully orders a disordered region.**  Reweighting by a
strictly positive Boltzmann factor keeps every populated conformation populated, so a
degree of freedom that is disordered in the free state is still disordered in the complex.
Complete disorder-to-order transitions are therefore an idealisation: they need an
interaction that excludes conformations outright, not one of finite energy. -/
theorem reweight_var_pos {n : ℕ} (E : Ens (Conf n)) {g : Conf n → ℝ} (hg : ∀ x, 0 < g x)
    {x y : Conf n} {i : Fin n} (hx : 0 < E.prob x) (hy : 0 < E.prob y) (hxy : x i ≠ y i) :
    0 < (E.reweight g hg).var i :=
  Ens.var_pos_of_two_points _ ((E.reweight_prob_pos_iff hg x).2 hx)
    ((E.reweight_prob_pos_iff hg y).2 hy) hxy

/-- The Boltzmann form of the same statement: at any finite interaction strength and any
finite temperature the complex retains the disorder of the free state. -/
theorem boltzmann_var_pos {n : ℕ} (E : Ens (Conf n)) (beta : ℝ) (U : Conf n → ℝ)
    {x y : Conf n} {i : Fin n} (hx : 0 < E.prob x) (hy : 0 < E.prob y) (hxy : x i ≠ y i) :
    0 < (E.boltzmann beta U).var i :=
  reweight_var_pos E _ hx hy hxy


/-- If the bound-state ensemble itself has two populated conformations -- a *fuzzy
complex*, which is what most disordered regions form -- then the bound state is not a
structure either, and "predict the structure of the complex" has no correct answer. -/
theorem fuzzy_complex_not_deterministic {B : Ens X} {x y : X} (hx : 0 < B.prob x)
    (hy : 0 < B.prob y) (hxy : x ≠ y) : ¬ B.Deterministic :=
  Ens.not_deterministic_of_two_points hx hy hxy

end IDR
