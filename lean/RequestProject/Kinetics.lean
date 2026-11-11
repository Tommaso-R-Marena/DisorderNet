/-
# Part IV.6  Coarse-grained kinetics: when is a reduced model still a model?

`RequestProject.Dynamics` shows that equilibrium populations do not determine kinetics, and
that the repair is to take conformation space to be trajectory space.  This file treats the
reduction that every practical kinetic model of a disordered region performs: **lumping**
microstates into a handful of macrostates (secondary-structure classes, FRET-resolvable
states, the clusters of a Markov state model) and running a Markov chain on those.

* `IsStochastic`, `evolve`, `Stationary`, `DetailedBalance`, `stationary_of_detailedBalance`
  -- the standard setting: a transition kernel on a conformational library, and the fact
  that a force field obeying detailed balance leaves the Boltzmann populations invariant.
* `lump`, `Lumpable` -- the coarse-graining of populations by a partition `phi`, and
  **Dynkin's lumpability condition**: microstates in the same macrostate must have the same
  total transition probability into each macrostate.
* `lump_evolve` -- under that condition, and only then, coarse-graining commutes with
  dynamics: the reduced chain `Q` predicts the reduced populations exactly, for *every*
  initial condition.  `lump_stationary` -- and it inherits the right equilibrium.
* `no_markov_coarse_graining` -- **the generic situation is failure.**  There is a
  three-state chain and a two-state partition for which *no* reduced transition matrix
  whatsoever reproduces the coarse dynamics: two microstates in the same macrostate have
  different fates, so the coarse variable is not Markov and its evolution depends on
  history the reduced model has discarded.

The design consequence: a kinetic model of a disordered region built on a coarse observable
is not merely inaccurate, it is not a Markov model of that observable at all unless the
lumpability condition is checked.  Either verify `Lumpable`, or keep the memory -- which,
in the language of `RequestProject.Dynamics`, means modelling trajectories rather than
states.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.Dynamics

namespace IDR

open Finset
open scoped Classical

namespace Kinetics

variable {n p : ℕ}

/-- A transition kernel on a conformational library. -/
def IsStochastic (P : Fin n → Fin n → ℝ) : Prop :=
  (∀ i j, 0 ≤ P i j) ∧ ∀ i, ∑ j, P i j = 1

/-- One step of the dynamics on populations. -/
def evolve (P : Fin n → Fin n → ℝ) (w : Fin n → ℝ) : Fin n → ℝ := fun j => ∑ i, w i * P i j

/-- An equilibrium population vector. -/
def Stationary (P : Fin n → Fin n → ℝ) (pi : Fin n → ℝ) : Prop := evolve P pi = pi

/-- Detailed balance: the microscopic reversibility a force field-derived kernel obeys. -/
def DetailedBalance (P : Fin n → Fin n → ℝ) (pi : Fin n → ℝ) : Prop :=
  ∀ i j, pi i * P i j = pi j * P j i

/-- Detailed balance makes the Boltzmann populations stationary. -/
theorem stationary_of_detailedBalance {P : Fin n → Fin n → ℝ} {pi : Fin n → ℝ}
    (hP : IsStochastic P) (h : DetailedBalance P pi) : Stationary P pi := by
  funext j
  simp only [evolve]
  rw [Finset.sum_congr rfl (fun i _ => h i j), ← Finset.mul_sum, hP.2 j, mul_one]

/-! ## Lumping microstates into macrostates -/

/-- Coarse-grained populations: the population of a macrostate is the total population of
its microstates. -/
def lump (phi : Fin n → Fin p) (w : Fin n → ℝ) : Fin p → ℝ :=
  fun b => ∑ i ∈ Finset.univ.filter (fun i => phi i = b), w i

/-- **Dynkin's lumpability condition** relative to a reduced kernel `Q`: microstates lying
in the same macrostate must have the same total transition probability into each
macrostate. -/
def Lumpable (P : Fin n → Fin n → ℝ) (phi : Fin n → Fin p) (Q : Fin p → Fin p → ℝ) : Prop :=
  ∀ (i : Fin n) (b : Fin p), ∑ j ∈ Finset.univ.filter (fun j => phi j = b), P i j = Q (phi i) b

/-- **Coarse-graining commutes with the dynamics exactly under lumpability.**  The reduced
chain then predicts the reduced populations for *every* initial condition -- which is what
it means for the macrostate to be a Markov variable. -/
theorem lump_evolve {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {Q : Fin p → Fin p → ℝ}
    (h : Lumpable P phi Q) (w : Fin n → ℝ) :
    lump phi (evolve P w) = evolve Q (lump phi w) := by
  classical
  funext b
  have hlhs : lump phi (evolve P w) b = ∑ i, w i * Q (phi i) b := by
    simp only [lump, evolve]
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.mul_sum, h i b]
  have hrhs : evolve Q (lump phi w) b = ∑ i, w i * Q (phi i) b := by
    simp only [evolve, lump, Finset.sum_mul]
    have hfib : ∀ x : Fin p, ∑ i ∈ Finset.univ.filter (fun i => phi i = x), w i * Q x b
        = ∑ i ∈ Finset.univ.filter (fun i => phi i = x), w i * Q (phi i) b := by
      intro x
      refine Finset.sum_congr rfl fun i hi => ?_
      rw [(Finset.mem_filter.1 hi).2]
    rw [Finset.sum_congr rfl (fun x _ => hfib x)]
    exact Finset.sum_fiberwise Finset.univ phi (fun i => w i * Q (phi i) b)
  rw [hlhs, hrhs]

/-- A lumpable reduction inherits the right equilibrium. -/
theorem lump_stationary {P : Fin n → Fin n → ℝ} {phi : Fin n → Fin p} {Q : Fin p → Fin p → ℝ}
    {pi : Fin n → ℝ} (h : Lumpable P phi Q) (hpi : Stationary P pi) :
    Stationary Q (lump phi pi) := by
  have := lump_evolve h pi
  rw [hpi] at this
  exact this.symm

/-! ## Generically, the coarse variable is not Markov -/

/-- Three microstates: `0` and `1` form one macrostate, `2` the other.  From `0` the chain
leaves for the second macrostate at once; from `1` it stays forever.  The two microstates
are indistinguishable to the coarse observable but have opposite fates. -/
noncomputable def badP : Fin 3 → Fin 3 → ℝ :=
  fun i j => if i = 0 then (if j = 2 then 1 else 0) else (if j = i then 1 else 0)

/-- The coarse observable: `{0,1} ↦ 0`, `{2} ↦ 1`. -/
def badPhi : Fin 3 → Fin 2 := fun i => if i = 2 then 1 else 0

theorem badP_stochastic : IsStochastic badP := by
  constructor
  · intro i j
    simp only [badP]
    split <;> split <;> norm_num
  · intro i
    rw [Fin.sum_univ_three]
    fin_cases i <;> simp [badP]

lemma lump_badPhi_one (w : Fin 3 → ℝ) : lump badPhi w 1 = w 2 := by
  have h : (Finset.univ.filter (fun i : Fin 3 => badPhi i = 1)) = {2} := by decide
  rw [lump, h, Finset.sum_singleton]

lemma lump_badPhi_zero (w : Fin 3 → ℝ) : lump badPhi w 0 = w 0 + w 1 := by
  have h : (Finset.univ.filter (fun i : Fin 3 => badPhi i = 0)) = {0, 1} := by decide
  rw [lump, h]
  norm_num

/-- **No Markov model of the coarse observable exists.**  For this chain and this
partition, *no* reduced transition matrix reproduces the coarse dynamics: two initial
conditions with identical coarse populations evolve into different coarse populations.  A
coarse-grained kinetic model of a disordered region therefore has memory unless Dynkin's
condition is verified -- accuracy of the reduced rates is not the issue; Markovianity
itself fails. -/
theorem no_markov_coarse_graining :
    ¬ ∃ Q : Fin 2 → Fin 2 → ℝ, ∀ w : Fin 3 → ℝ,
        lump badPhi (evolve badP w) = evolve Q (lump badPhi w) := by
  rintro ⟨Q, hQ⟩
  -- start at microstate `0`: the population leaves the first macrostate at once
  have h0 := congrFun (hQ (fun i => if i = 0 then (1 : ℝ) else 0)) 1
  -- start at microstate `1`: the population stays, although the coarse state is the same
  have h1 := congrFun (hQ (fun i => if i = 1 then (1 : ℝ) else 0)) 1
  simp only [evolve, lump_badPhi_one, lump_badPhi_zero, Fin.sum_univ_two, Fin.sum_univ_three,
    badP] at h0 h1
  norm_num at h0 h1
  have e20 : ¬ ((2 : Fin 3) = 0) := by decide
  have e21 : ¬ ((2 : Fin 3) = 1) := by decide
  simp only [if_neg e20] at h0
  simp only [if_neg e21] at h1
  linarith

end Kinetics

end IDR
