/-
# Part XV.3  Single molecules in time: what a trajectory reports about the ensemble

The most direct experiments on disordered regions -- single-molecule FRET traces, force
clamps, in-cell tracking -- observe *one* copy of the chain for a finite time and average
along the trajectory.  The tacit assumption is ergodicity: that a time average is an
ensemble average.  This file proves the exact form of that assumption and exhibits the
regime in which it fails, which is precisely the regime disordered regions inhabit
(exchange between structural basins slower than the observation window, or absent).

* `distAt`, `timeAvg` -- the law of the chain after `t` steps and the expected time average
  of an observable over a window of `T` frames.
* `timeAvg_stationary` -- **the positive result**: started from the equilibrium ensemble,
  the expected time average is exactly the ensemble average, for every observable and every
  window length.  Single-molecule averaging is unbiased when, and only in the sense that,
  the starting law is the target.
* `supportedOn_distAt` -- a trajectory started inside a kinetically closed set of
  conformations never leaves it.
* `distAt_eq_of_agree` / `timeAvg_eq_of_agree` -- **trajectories are blind outside their
  basin**: two kinetic models that agree on the rows of a closed set produce *identical*
  statistics for every observable and every window length, however much they differ
  elsewhere.
* `single_molecule_cannot_identify_the_ensemble` -- the explicit witness: two three-state
  kinetic models with the same intra-basin dynamics, identical single-molecule statistics
  from any start in the basin, and different equilibrium ensembles (a conformation that is
  the entire equilibrium of one model is transient in the other).  No length of trace and
  no observable separates them.
* `reducible_stationary_not_unique` -- and the underlying structural fact: when exchange is
  absent the equilibrium ensemble is not even determined by the kinetics, so "the ensemble"
  a model is asked to predict must be specified together with the preparation, or the
  model must carry the basin decomposition explicitly.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Kinetics

namespace IDR

open Finset
open scoped Classical

namespace Trajectory

open Kinetics

variable {n : ℕ}

/-- The law of the chain after `t` steps, started from the law `w`. -/
def distAt (P : Fin n → Fin n → ℝ) (w : Fin n → ℝ) : ℕ → (Fin n → ℝ)
  | 0 => w
  | (t + 1) => evolve P (distAt P w t)

@[simp] lemma distAt_zero (P : Fin n → Fin n → ℝ) (w : Fin n → ℝ) : distAt P w 0 = w := rfl

@[simp] lemma distAt_succ (P : Fin n → Fin n → ℝ) (w : Fin n → ℝ) (t : ℕ) :
    distAt P w (t + 1) = evolve P (distAt P w t) := rfl

/-- The expected time average of the observable `f` over a window of `T` frames. -/
noncomputable def timeAvg (P : Fin n → Fin n → ℝ) (w f : Fin n → ℝ) (T : ℕ) : ℝ :=
  (1 / (T : ℝ)) * ∑ t ∈ Finset.range T, ∑ j, distAt P w t j * f j

/-! ## The positive result: from equilibrium, time averaging is unbiased -/

lemma distAt_stationary {P : Fin n → Fin n → ℝ} {pi : Fin n → ℝ} (h : Kinetics.Stationary P pi) :
    ∀ t, distAt P pi t = pi := by
  intro t
  induction t with
  | zero => rfl
  | succ t ih => rw [distAt_succ, ih]; exact h

/-- **Ergodic averaging is unbiased at equilibrium.**  A trajectory started from the
equilibrium ensemble reports, in expectation, exactly the ensemble average -- for every
observable and every window length. -/
theorem timeAvg_stationary {P : Fin n → Fin n → ℝ} {pi : Fin n → ℝ} (h : Kinetics.Stationary P pi)
    (f : Fin n → ℝ) {T : ℕ} (hT : 0 < T) : timeAvg P pi f T = ∑ j, pi j * f j := by
  have hTne : (T : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hT.ne'
  simp only [timeAvg, distAt_stationary h]
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  field_simp

/-! ## Kinetically closed sets, and blindness outside them -/

/-- A set of conformations from which the chain cannot escape in one step. -/
def Closed (P : Fin n → Fin n → ℝ) (S : Finset (Fin n)) : Prop :=
  ∀ i ∈ S, ∀ j ∉ S, P i j = 0

/-- A law supported inside a set of conformations. -/
def SupportedOn (w : Fin n → ℝ) (S : Finset (Fin n)) : Prop := ∀ j ∉ S, w j = 0

lemma supportedOn_evolve {P : Fin n → Fin n → ℝ} {S : Finset (Fin n)} {w : Fin n → ℝ}
    (hC : Closed P S) (hw : SupportedOn w S) : SupportedOn (evolve P w) S := by
  intro j hj
  refine Finset.sum_eq_zero fun i _ => ?_
  by_cases hi : i ∈ S
  · rw [hC i hi j hj, mul_zero]
  · rw [hw i hi, zero_mul]

/-- A trajectory started inside a kinetically closed set stays inside it forever. -/
lemma supportedOn_distAt {P : Fin n → Fin n → ℝ} {S : Finset (Fin n)} {w : Fin n → ℝ}
    (hC : Closed P S) (hw : SupportedOn w S) : ∀ t, SupportedOn (distAt P w t) S := by
  intro t
  induction t with
  | zero => exact hw
  | succ t ih => exact supportedOn_evolve hC ih

/-- **A trajectory is blind outside its basin.**  Two kinetic models whose transition rows
agree on a closed set of conformations generate exactly the same law at every time, for
every start inside that set -- no matter how differently they behave elsewhere. -/
theorem distAt_eq_of_agree {P P' : Fin n → Fin n → ℝ} {S : Finset (Fin n)} {w : Fin n → ℝ}
    (hC : Closed P S) (hagree : ∀ i ∈ S, ∀ j, P i j = P' i j) (hw : SupportedOn w S) :
    ∀ t, distAt P w t = distAt P' w t := by
  intro t
  induction t with
  | zero => rfl
  | succ t ih =>
      have hsupp := supportedOn_distAt hC hw t
      rw [distAt_succ, distAt_succ, ← ih]
      funext j
      refine Finset.sum_congr rfl fun i _ => ?_
      by_cases hi : i ∈ S
      · rw [hagree i hi j]
      · rw [hsupp i hi, zero_mul, zero_mul]

/-- The same statement for what is actually measured: the expected time average of any
observable over any window. -/
theorem timeAvg_eq_of_agree {P P' : Fin n → Fin n → ℝ} {S : Finset (Fin n)} {w : Fin n → ℝ}
    (hC : Closed P S) (hagree : ∀ i ∈ S, ∀ j, P i j = P' i j) (hw : SupportedOn w S)
    (f : Fin n → ℝ) (T : ℕ) : timeAvg P w f T = timeAvg P' w f T := by
  simp only [timeAvg]
  congr 1
  exact Finset.sum_congr rfl fun t _ => by rw [distAt_eq_of_agree hC hagree hw t]

/-! ## The witness: identical traces, different ensembles

Three conformations.  `0` and `1` exchange rapidly with each other -- the basin the
experiment watches.  Conformation `2` is a second basin.  In the first model `2` is a trap
(it is its own equilibrium); in the second model `2` decays into the watched basin.  The
two models have the same intra-basin rows, hence identical single-molecule statistics from
any start in the basin, and different equilibrium ensembles. -/

/-- The watched basin. -/
def basin : Finset (Fin 3) := {0, 1}

/-- Model A: conformations `0` and `1` exchange; conformation `2` is a trap. -/
noncomputable def trapP : Fin 3 → Fin 3 → ℝ :=
  fun i j => if i = 2 then (if j = 2 then 1 else 0) else (if j = 2 then 0 else 1 / 2)

/-- Model B: identical inside the basin, but conformation `2` decays into it. -/
noncomputable def decayP : Fin 3 → Fin 3 → ℝ :=
  fun i j => if i = 2 then (if j = 0 then 1 else 0) else (if j = 2 then 0 else 1 / 2)

/-- The conformation `2` on its own: the equilibrium ensemble of model A. -/
def deltaTwo : Fin 3 → ℝ := fun j => if j = 2 then 1 else 0

theorem trapP_stochastic : IsStochastic trapP := by
  constructor
  · intro i j
    simp only [trapP]
    split <;> split <;> norm_num
  · intro i
    rw [Fin.sum_univ_three]
    fin_cases i <;> norm_num [trapP, Fin.ext_iff]

theorem decayP_stochastic : IsStochastic decayP := by
  constructor
  · intro i j
    simp only [decayP]
    split <;> split <;> norm_num
  · intro i
    rw [Fin.sum_univ_three]
    fin_cases i <;> norm_num [decayP, Fin.ext_iff]

theorem basin_closed_trap : Closed trapP basin := by
  intro i hi j hj
  have hj2 : j = 2 := by
    fin_cases j <;> simp_all [basin]
  have hi2 : i ≠ 2 := by
    intro h; rw [h] at hi; simp [basin] at hi
  simp [trapP, hi2, hj2]

theorem rows_agree : ∀ i ∈ basin, ∀ j, trapP i j = decayP i j := by
  intro i hi j
  have hi2 : i ≠ 2 := by
    intro h; rw [h] at hi; simp [basin] at hi
  simp [trapP, decayP, hi2]

/-- Conformation `2` is the entire equilibrium of model A. -/
theorem deltaTwo_stationary_trap : Kinetics.Stationary trapP deltaTwo := by
  funext j
  simp only [evolve, deltaTwo, Fin.sum_univ_three]
  fin_cases j <;> norm_num [trapP, Fin.ext_iff]

/-- It is *not* an equilibrium of model B. -/
theorem deltaTwo_not_stationary_decay : ¬ Kinetics.Stationary decayP deltaTwo := by
  intro h
  have h0 := congrFun h 0
  simp only [evolve, deltaTwo, decayP, Fin.sum_univ_three] at h0
  norm_num [Fin.ext_iff] at h0

/-- **A single molecule cannot identify the ensemble.**  Model A and model B differ in
their equilibrium ensembles -- a conformation that is the whole equilibrium of A is not
even stationary under B -- yet every trajectory started in the watched basin has exactly
the same expected time average, for every observable and every window length.  A
single-molecule measurement therefore constrains a model only within the basin it happens
to visit; the ensemble weights *between* basins are not identifiable from it, and a model
of a disordered region must obtain them elsewhere (or report them as unidentified). -/
theorem single_molecule_cannot_identify_the_ensemble :
    (∀ w : Fin 3 → ℝ, SupportedOn w basin → ∀ (f : Fin 3 → ℝ) (T : ℕ),
        timeAvg trapP w f T = timeAvg decayP w f T)
      ∧ Kinetics.Stationary trapP deltaTwo ∧ ¬ Kinetics.Stationary decayP deltaTwo := by
  refine ⟨fun w hw f T => timeAvg_eq_of_agree basin_closed_trap rows_agree hw f T,
    deltaTwo_stationary_trap, deltaTwo_not_stationary_decay⟩

/-- The basin ensemble: `0` and `1` equally populated. -/
noncomputable def basinEq : Fin 3 → ℝ := fun j => if j = 2 then 0 else 1 / 2

theorem basinEq_stationary_trap : Kinetics.Stationary trapP basinEq := by
  funext j
  simp only [evolve, basinEq, Fin.sum_univ_three]
  fin_cases j <;> norm_num [trapP, Fin.ext_iff]

/-- **Without exchange the equilibrium ensemble is not determined by the kinetics.**  Model
A has (at least) two distinct stationary ensembles, so "the ensemble of the region" is not
a function of its energy landscape alone once ergodicity is broken: it depends on the
preparation.  A model must therefore either carry the basin decomposition and its weights
explicitly, or declare the between-basin populations unidentified. -/
theorem reducible_stationary_not_unique :
    Kinetics.Stationary trapP deltaTwo ∧ Kinetics.Stationary trapP basinEq ∧ deltaTwo ≠ basinEq := by
  refine ⟨deltaTwo_stationary_trap, basinEq_stationary_trap, fun h => ?_⟩
  have := congrFun h 2
  norm_num [deltaTwo, basinEq] at this

end Trajectory

end IDR
