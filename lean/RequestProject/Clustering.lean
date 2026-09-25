/-
# Part CVI  Whatever the clustering, the timescales come out short

Part LXII proves what a Markov state model built on a *given* clustering says, and the assumptions
list recorded that "the choice of clustering" was outside it.  That choice is the one free hand a
practitioner has, and the usual defence of it — that the implied timescales are checked for
convergence as the number of clusters grows — is a *one-sided* diagnostic.  This file proves why.

The setting is a finite microscopic chain `P` with positive weights `pi`, a clustering map
`cl : X → M`, and the stationary-weighted lumping `lumpP` that Part LXII takes as the estimate.

* `lump_inner_eq`, `lump_mean_eq`, `lump_form_eq` — **the lumped chain is the microscopic chain
  restricted to functions constant on clusters.**  The weighted inner product, the mean, and the
  Dirichlet-type form of a macrostate observable all coincide exactly with those of its lift.  No
  approximation is made in this step: the lumped model is an exact restriction, and everything it
  can say is something the microscopic chain says about a piecewise-constant observable.
* `lump_gap_transfer` — **hence every relaxation bound of the microscopic chain is inherited by the
  model.**  If the true chain relaxes at rate `gap` on mean-zero observables, so does the lumped
  one, so the lumped chain's own gap is at least the true gap, and the implied timescale it reports
  is at most the true timescale.  Clustering can only make relaxation look faster.
* `lump_comp` — **and coarsening cannot help.**  Lumping twice is lumping once by the composite
  map, so a coarser model is a lumping of a finer one, and `lump_refine_transfer` applies the
  previous result to it: refining the clustering can only increase the implied timescale.  A
  timescale that has stopped changing as clusters are added is a *lower bound* that has stopped
  improving; it is not evidence that it has reached the truth.

The design consequence for a model of a disordered region: an implied-timescale plateau is not a
validation.  A disordered region's slow motions are precisely the ones a state decomposition built
from structural coordinates is least likely to resolve, and every such failure biases the reported
timescale in the same direction — short.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset

namespace IDR.Clustering

variable {X M K : Type*} [Fintype X] [Fintype M] [Fintype K] [DecidableEq M] [DecidableEq K]

/-- The microstates assigned to a macrostate. -/
def cluster (cl : X → M) (A : M) : Finset X := Finset.univ.filter (fun x => cl x = A)

/-- The stationary weight of a macrostate. -/
def lumpPi (pi : X → ℝ) (cl : X → M) (A : M) : ℝ := ∑ x ∈ cluster cl A, pi x

/-- The stationary-weighted lumping of a transition matrix: the Markov state model estimate in
the infinite-sampling limit. -/
noncomputable def lumpP (pi : X → ℝ) (P : X → X → ℝ) (cl : X → M) (A B : M) : ℝ :=
  (∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster cl B, P x y) / lumpPi pi cl A

/-- The lift of a macrostate observable to the microstates. -/
def lift (cl : X → M) (f : M → ℝ) : X → ℝ := fun x => f (cl x)

/-- The weighted inner product. -/
def wInner (pi : X → ℝ) (u v : X → ℝ) : ℝ := ∑ x, pi x * (u x * v x)

/-- The bilinear form of the chain. -/
def wForm (pi : X → ℝ) (P : X → X → ℝ) (u v : X → ℝ) : ℝ :=
  ∑ x, ∑ y, pi x * u x * P x y * v y

/-- The weighted mean. -/
def wMean (pi : X → ℝ) (u : X → ℝ) : ℝ := ∑ x, pi x * u x

/-- **The weighted inner product of a macrostate observable is that of its lift.** -/
theorem lump_inner_eq (pi : X → ℝ) (cl : X → M) (f g : M → ℝ) :
    (∑ A, lumpPi pi cl A * (f A * g A)) = wInner pi (lift cl f) (lift cl g) := by
  unfold wInner lumpPi lift
  rw [← Finset.sum_fiberwise (Finset.univ : Finset X) cl (fun x => pi x * (f (cl x) * g (cl x)))]
  refine Finset.sum_congr rfl fun A _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun x hx => ?_
  have : cl x = A := (Finset.mem_filter.mp hx).2
  rw [this]

/-- **The weighted mean of a macrostate observable is that of its lift.** -/
theorem lump_mean_eq (pi : X → ℝ) (cl : X → M) (f : M → ℝ) :
    (∑ A, lumpPi pi cl A * f A) = wMean pi (lift cl f) := by
  unfold wMean lumpPi lift
  rw [← Finset.sum_fiberwise (Finset.univ : Finset X) cl (fun x => pi x * f (cl x))]
  refine Finset.sum_congr rfl fun A _ => ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun x hx => ?_
  have : cl x = A := (Finset.mem_filter.mp hx).2
  rw [this]

/-- **The bilinear form of the lumped chain is that of the microscopic chain on lifts.** -/
theorem lump_form_eq {pi : X → ℝ} {P : X → X → ℝ} {cl : X → M}
    (hpi : ∀ A, lumpPi pi cl A ≠ 0) (f g : M → ℝ) :
    (∑ A, ∑ B, lumpPi pi cl A * f A * lumpP pi P cl A B * g B)
      = wForm pi P (lift cl f) (lift cl g) := by
  have key : ∀ A B, lumpPi pi cl A * f A * lumpP pi P cl A B * g B
      = ∑ x ∈ cluster cl A, ∑ y ∈ cluster cl B,
          pi x * (lift cl f) x * P x y * (lift cl g) y := by
    intro A B
    have hS : lumpPi pi cl A * lumpP pi P cl A B
        = ∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster cl B, P x y := by
      unfold lumpP
      rw [mul_comm]
      exact div_mul_cancel₀ _ (hpi A)
    have hL : lumpPi pi cl A * f A * lumpP pi P cl A B * g B
        = (f A * g B) * (lumpPi pi cl A * lumpP pi P cl A B) := by ring
    rw [hL, hS, Finset.mul_sum]
    refine Finset.sum_congr rfl fun x hx => ?_
    have hxA : cl x = A := (Finset.mem_filter.mp hx).2
    unfold lift
    simp only [Finset.mul_sum]
    refine Finset.sum_congr rfl fun y hy => ?_
    have hyB : cl y = B := (Finset.mem_filter.mp hy).2
    rw [hxA, hyB]
    ring
  calc (∑ A, ∑ B, lumpPi pi cl A * f A * lumpP pi P cl A B * g B)
      = ∑ A, ∑ B, ∑ x ∈ cluster cl A, ∑ y ∈ cluster cl B,
          pi x * (lift cl f) x * P x y * (lift cl g) y := by
        exact Finset.sum_congr rfl fun A _ => Finset.sum_congr rfl fun B _ => key A B
    _ = ∑ A, ∑ x ∈ cluster cl A, ∑ B, ∑ y ∈ cluster cl B,
          pi x * (lift cl f) x * P x y * (lift cl g) y := by
        refine Finset.sum_congr rfl fun A _ => ?_
        rw [Finset.sum_comm]
    _ = ∑ A, ∑ x ∈ cluster cl A, ∑ y, pi x * (lift cl f) x * P x y * (lift cl g) y := by
        refine Finset.sum_congr rfl fun A _ => Finset.sum_congr rfl fun x _ => ?_
        exact Finset.sum_fiberwise (Finset.univ : Finset X) cl _
    _ = wForm pi P (lift cl f) (lift cl g) := by
        unfold wForm
        exact Finset.sum_fiberwise (Finset.univ : Finset X) cl _

/-- **Every relaxation bound of the microscopic chain is inherited by the Markov state model.**
If the true chain contracts mean-zero observables by `1 − gap`, so does the lumped chain — so the
model's own gap is at least the true one, and the timescale it reports is at most the truth. -/
theorem lump_gap_transfer {pi : X → ℝ} {P : X → X → ℝ} {cl : X → M} {gap : ℝ}
    (hpi : ∀ A, lumpPi pi cl A ≠ 0)
    (hmicro : ∀ u : X → ℝ, wMean pi u = 0 → wForm pi P u u ≤ (1 - gap) * wInner pi u u)
    (f : M → ℝ) (hf : (∑ A, lumpPi pi cl A * f A) = 0) :
    (∑ A, ∑ B, lumpPi pi cl A * f A * lumpP pi P cl A B * f B)
      ≤ (1 - gap) * ∑ A, lumpPi pi cl A * (f A * f A) := by
  have hmean : wMean pi (lift cl f) = 0 := by rw [← lump_mean_eq pi cl f]; exact hf
  rw [lump_form_eq hpi f f, lump_inner_eq pi cl f f]
  exact hmicro _ hmean

/-! ## Coarsening is lumping again -/

omit [Fintype K] in
/-- Summing over the microstates of a macrostate, then over the macrostates of a supercluster, is
summing over the microstates of the supercluster. -/
theorem sum_cluster_comp (cl : X → M) (cm : M → K) (C : K) (F : X → ℝ) :
    (∑ A ∈ cluster cm C, ∑ x ∈ cluster cl A, F x) = ∑ x ∈ cluster (cm ∘ cl) C, F x := by
  have h1 : ∀ A : M, (if cm A = C then ∑ x ∈ cluster cl A, F x else 0)
      = ∑ x ∈ cluster cl A, (if cm (cl x) = C then F x else 0) := by
    intro A
    by_cases hA : cm A = C
    · rw [if_pos hA]
      refine Finset.sum_congr rfl fun x hx => ?_
      have hxA : cl x = A := (Finset.mem_filter.mp hx).2
      rw [hxA, if_pos hA]
    · rw [if_neg hA]
      refine (Finset.sum_eq_zero fun x hx => ?_).symm
      have hxA : cl x = A := (Finset.mem_filter.mp hx).2
      rw [hxA, if_neg hA]
  calc (∑ A ∈ cluster cm C, ∑ x ∈ cluster cl A, F x)
      = ∑ A, (if cm A = C then ∑ x ∈ cluster cl A, F x else 0) := by
        rw [cluster, Finset.sum_filter]
    _ = ∑ A, ∑ x ∈ cluster cl A, (if cm (cl x) = C then F x else 0) :=
        Finset.sum_congr rfl fun A _ => h1 A
    _ = ∑ x, (if cm (cl x) = C then F x else 0) :=
        Finset.sum_fiberwise (Finset.univ : Finset X) cl _
    _ = ∑ x ∈ cluster (cm ∘ cl) C, F x := by
        rw [cluster, Finset.sum_filter]
        rfl

omit [Fintype K] in
/-- **Lumping twice is lumping once by the composite map** — for the weights. -/
theorem lumpPi_comp (pi : X → ℝ) (cl : X → M) (cm : M → K) (C : K) :
    lumpPi (lumpPi pi cl) cm C = lumpPi pi (cm ∘ cl) C := by
  unfold lumpPi
  exact sum_cluster_comp cl cm C pi

omit [Fintype K] in
/-- **Lumping twice is lumping once by the composite map** — for the transition matrix.  A coarser
Markov state model is a lumping of a finer one. -/
theorem lump_comp {pi : X → ℝ} {P : X → X → ℝ} (cl : X → M) (cm : M → K)
    (hpi : ∀ A, lumpPi pi cl A ≠ 0) (C D : K) :
    lumpP (lumpPi pi cl) (lumpP pi P cl) cm C D = lumpP pi P (cm ∘ cl) C D := by
  have hstep : ∀ A : M, lumpPi pi cl A * ∑ B ∈ cluster cm D, lumpP pi P cl A B
      = ∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster (cm ∘ cl) D, P x y := by
    intro A
    have hB : ∀ B : M, lumpPi pi cl A * lumpP pi P cl A B
        = ∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster cl B, P x y := by
      intro B
      unfold lumpP
      rw [mul_comm]
      exact div_mul_cancel₀ _ (hpi A)
    calc lumpPi pi cl A * ∑ B ∈ cluster cm D, lumpP pi P cl A B
        = ∑ B ∈ cluster cm D, lumpPi pi cl A * lumpP pi P cl A B := by rw [Finset.mul_sum]
      _ = ∑ B ∈ cluster cm D, ∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster cl B, P x y :=
          Finset.sum_congr rfl fun B _ => hB B
      _ = ∑ x ∈ cluster cl A, ∑ B ∈ cluster cm D, pi x * ∑ y ∈ cluster cl B, P x y :=
          Finset.sum_comm
      _ = ∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster (cm ∘ cl) D, P x y := by
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [← Finset.mul_sum, sum_cluster_comp cl cm D (fun y => P x y)]
  unfold lumpP
  rw [lumpPi_comp pi cl cm C]
  congr 1
  calc (∑ A ∈ cluster cm C, lumpPi pi cl A * ∑ B ∈ cluster cm D, lumpP pi P cl A B)
      = ∑ A ∈ cluster cm C, ∑ x ∈ cluster cl A, pi x * ∑ y ∈ cluster (cm ∘ cl) D, P x y :=
        Finset.sum_congr rfl fun A _ => hstep A
    _ = ∑ x ∈ cluster (cm ∘ cl) C, pi x * ∑ y ∈ cluster (cm ∘ cl) D, P x y :=
        sum_cluster_comp cl cm C _

/-- **Refining a clustering can only increase the implied timescale.**  A relaxation bound
satisfied by a fine Markov state model is inherited by every coarsening of it, so the coarser
model's gap is at least the finer model's. -/
theorem lump_refine_transfer {pi : X → ℝ} {P : X → X → ℝ} {cl : X → M} {cm : M → K} {gap : ℝ}
    (hpi' : ∀ C, lumpPi (lumpPi pi cl) cm C ≠ 0)
    (hfine : ∀ u : M → ℝ, wMean (lumpPi pi cl) u = 0 →
      wForm (lumpPi pi cl) (lumpP pi P cl) u u ≤ (1 - gap) * wInner (lumpPi pi cl) u u)
    (f : K → ℝ) (hf : (∑ C, lumpPi (lumpPi pi cl) cm C * f C) = 0) :
    (∑ C, ∑ D, lumpPi (lumpPi pi cl) cm C * f C
        * lumpP (lumpPi pi cl) (lumpP pi P cl) cm C D * f D)
      ≤ (1 - gap) * ∑ C, lumpPi (lumpPi pi cl) cm C * (f C * f C) :=
  lump_gap_transfer hpi' hfine f hf

end IDR.Clustering
