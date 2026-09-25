/-
# The model cannot be static either: populations do not determine kinetics

Everything so far has treated the target as an equilibrium *distribution*.  A disordered
region, however, is characterised as much by its motion -- exchange rates between states,
the lifetime of an encounter complex, the timescale on which a motif becomes available --
as by its populations.  This file shows that the equilibrium ensemble, the object the
previous files argued the model must output, is still not enough if kinetics is in scope,
and identifies what replaces it.

* `stationary_does_not_determine_dynamics` -- two Markov dynamics with *identical*
  equilibrium populations differ in a two-time observable.  Populations do not determine
  rates.
* `no_static_model_predicts_kinetics` -- consequently any predictor whose output is a
  function of the equilibrium distribution alone is wrong about the dynamics of some
  system, however accurate its populations.
* `trajectory_ensembles_universal` -- and the repair needs no new theory: because the
  framework of `RequestProject.EnsembleCore` is stated over an arbitrary conformation
  space, taking that space to be the space of *trajectories* `Fin T → X` turns every
  earlier theorem into a statement about kinetic models.  The model must output a
  distribution over trajectories; then, exactly as before, it can be right on everything.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.Answer

namespace IDR

open Finset
open scoped Classical

/-! ## Markov dynamics on a finite set of states -/

/-- A transition kernel on `m` conformational states. -/
structure Kernel (m : ℕ) where
  /-- probability of moving from state `i` to state `j` in one step -/
  P : Fin m → Fin m → ℝ
  nonneg : ∀ i j, 0 ≤ P i j
  row_sum : ∀ i, ∑ j, P i j = 1

/-- The distribution `pi` is stationary for the kernel `K`: the equilibrium populations. -/
def Stationary {m : ℕ} (pi : Fin m → ℝ) (K : Kernel m) : Prop :=
  ∀ j, ∑ i, pi i * K.P i j = pi j

/-- The two-time correlation `⟨f (x 0) * g (x 1)⟩`: the simplest observable that sees the
dynamics rather than only the populations.  Relaxation rates, exchange rates and lifetimes
are all read off from observables of this kind. -/
def twoTime {m : ℕ} (pi : Fin m → ℝ) (K : Kernel m) (f g : Fin m → ℝ) : ℝ :=
  ∑ i, pi i * (f i * ∑ j, K.P i j * g j)

/-- The frozen dynamics: every state is absorbing. -/
def stayKernel : Kernel 2 where
  P := fun i j => if i = j then 1 else 0
  nonneg := by intro i j; by_cases h : i = j <;> simp [h]
  row_sum := by intro i; fin_cases i <;> norm_num [Fin.sum_univ_succ]

/-- The maximally fast dynamics: the two states exchange at every step. -/
def swapKernel : Kernel 2 where
  P := fun i j => if i = j then 0 else 1
  nonneg := by intro i j; by_cases h : i = j <;> simp [h]
  row_sum := by intro i; fin_cases i <;> norm_num [Fin.sum_univ_succ]

/-- The uniform populations. -/
noncomputable def halfHalf : Fin 2 → ℝ := fun _ => 1 / 2

lemma stationary_stay : Stationary halfHalf stayKernel := by
  intro j
  fin_cases j <;> norm_num [halfHalf, stayKernel, Fin.sum_univ_succ]

lemma stationary_swap : Stationary halfHalf swapKernel := by
  intro j
  fin_cases j <;> norm_num [halfHalf, swapKernel, Fin.sum_univ_succ]

/-- **Equilibrium populations do not determine the kinetics.**  Two dynamics -- one frozen,
one exchanging at every step -- have exactly the same equilibrium ensemble and differ in a
two-time observable.  Any two ensembles-of-conformations statement is therefore blind to
the difference between a rigid mixture of two states and a region interconverting between
them; yet biologically these are entirely different objects. -/
theorem stationary_does_not_determine_dynamics :
    Stationary halfHalf stayKernel ∧ Stationary halfHalf swapKernel ∧
      ∃ f g : Fin 2 → ℝ, twoTime halfHalf stayKernel f g ≠ twoTime halfHalf swapKernel f g := by
  refine ⟨stationary_stay, stationary_swap,
    (fun i => if i = 0 then 1 else 0), (fun i => if i = 0 then 1 else 0), ?_⟩
  norm_num [twoTime, halfHalf, stayKernel, swapKernel, Fin.sum_univ_succ]

lemma twoTime_stay :
    twoTime halfHalf stayKernel (fun i => if i = 0 then 1 else 0)
      (fun i => if i = 0 then 1 else 0) = 1 / 2 := by
  norm_num [twoTime, halfHalf, stayKernel, Fin.sum_univ_succ]

lemma twoTime_swap :
    twoTime halfHalf swapKernel (fun i => if i = 0 then 1 else 0)
      (fun i => if i = 0 then 1 else 0) = 0 := by
  norm_num [twoTime, halfHalf, swapKernel, Fin.sum_univ_succ]

/-- **A model of populations alone cannot predict kinetics.**  Let `A` be any map from an
equilibrium distribution to a predicted dynamics -- this is exactly what a model that
outputs an ensemble and nothing else provides, however it is post-processed.  Then there is
a true dynamics with those very populations whose two-time observables `A` gets wrong.  So
a correct ensemble is *necessary but not sufficient*: the output type has to be enlarged
again, from distributions over conformations to distributions over trajectories. -/
theorem no_static_model_predicts_kinetics (A : (Fin 2 → ℝ) → Kernel 2) :
    ∃ K : Kernel 2, Stationary halfHalf K ∧
      ∃ f g : Fin 2 → ℝ, twoTime halfHalf (A halfHalf) f g ≠ twoTime halfHalf K f g := by
  by_cases h : twoTime halfHalf (A halfHalf) (fun i => if i = 0 then 1 else 0)
      (fun i => if i = 0 then 1 else 0) = 1 / 2
  · refine ⟨swapKernel, stationary_swap, (fun i => if i = 0 then 1 else 0),
      (fun i => if i = 0 then 1 else 0), ?_⟩
    rw [h, twoTime_swap]
    norm_num
  · refine ⟨stayKernel, stationary_stay, (fun i => if i = 0 then 1 else 0),
      (fun i => if i = 0 then 1 else 0), ?_⟩
    rw [twoTime_stay]
    exact h

/-! ## The repair: distributions over trajectories -/

/-- **The theory applies verbatim to kinetics.**  Nothing in the framework depends on the
conformation space being a space of *structures*: taking it to be the space of trajectories
`Fin T → X`, a model that outputs a context-conditional distribution over trajectories
solves every kinetic prediction problem exactly -- and, by the same theorems, none of the
restricted model classes does.  Generality of the conformation space is what makes the
answer robust: "predict a distribution over the objects you are asked about" is the whole
specification, whether those objects are structures, trajectories, or complexes. -/
theorem trajectory_ensembles_universal {I X : Type*} (T : ℕ) (Tgt : I → Ens (Fin T → X)) :
    ∃ A : I → Ens (Fin T → X), Solves A Tgt :=
  solvable_by_conditional_ensembles Tgt

/-- A single trajectory is to kinetics what a single structure is to thermodynamics: right
only in the degenerate case.  If two distinct trajectories are populated, no
single-trajectory prediction is correct. -/
theorem no_single_trajectory {X : Type*} {T : ℕ} {E : Ens (Fin T → X)} {x y : Fin T → X}
    (hx : 0 < E.prob x) (hy : 0 < E.prob y) (hxy : x ≠ y) : ¬ E.Deterministic :=
  Ens.not_deterministic_of_two_points hx hy hxy

end IDR
