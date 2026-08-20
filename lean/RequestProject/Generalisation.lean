/-
# Part XI.5  Learnability: when does a model of disorder generalise across sequences?

Parts I--X are about *one* region: what its ensemble is, how it is measured, how many
structures a model of it needs, how much data it costs.  A model, however, is trained on a
finite set of proteins and then applied to a new one, and nothing so far says when that is
possible.  This file supplies the missing half, and it is a sharp dichotomy in the modulus of
continuity of the map

  `T : sequence-and-context  ↦  conformational ensemble`.

* `nearest_neighbour_generalisation` -- **the positive law.**  If `T` is `K`-Lipschitz on the
  observables of interest and the training set is a `delta`-net of the input space, then the
  nearest-neighbour predictor -- the crudest model there is -- is already accurate to `K·delta`
  on every one of those observables, at every input, including inputs never seen.  Sequence
  coverage, not architecture, is what buys generalisation.
* `switch_forces_capacity` -- **the negative law.**  Disordered regions do the opposite of
  varying smoothly: a single phosphorylation, a single charge substitution, the arrival of a
  partner, can move the ensemble a long way.  If two inputs at distance `d` have ensembles
  that differ by `G` on some observable, then *every* `L`-Lipschitz predictor has error at
  least `(G − L·d)/2` on one of them.  Smoothness in sequence space is not a free
  regularisation: at a switch it is an error floor, and the model must have modulus of
  continuity at least `G/d` to be right.
* `no_smooth_model_of_switch` -- the immediate corollary: for any target accuracy `eps` below
  half the switch gap, a predictor with Lipschitz constant `L < (G − 2·eps)/d` cannot reach
  it.

Together these say what "design a model that captures disorder" means at the level of the
learning problem: cover the input space, and give the model enough capacity for *variation*
in sequence, not only for spread in conformation.
-/
import Mathlib
import RequestProject.EnsembleCore

namespace IDR

open scoped Classical

variable {S X : Type*} [MetricSpace S]

/-- The target map is `K`-Lipschitz on a class `Obs` of observables: nearby inputs -- one
substitution, a small change of ionic strength -- have ensembles whose averages of every
observable in the class are close. -/
def LipschitzTarget (K : ℝ) (T : S → Ens X) (Obs : Set (X → ℝ)) : Prop :=
  ∀ f ∈ Obs, ∀ s s' : S, |(T s).expect f - (T s').expect f| ≤ K * dist s s'

/-- **Generalisation from a covering training set.**  If the target map is `K`-Lipschitz and
every input is within `delta` of some training input, the nearest-neighbour predictor is
accurate to `K·delta` everywhere, on every observable of the class. -/
theorem nearest_neighbour_generalisation {K delta : ℝ} {T : S → Ens X} {Obs : Set (X → ℝ)}
    (hK : 0 ≤ K) (hT : LipschitzTarget K T Obs) {train : Set S}
    (hnet : ∀ s : S, ∃ t ∈ train, dist s t ≤ delta) :
    ∃ A : S → Ens X,
      (∀ s : S, ∃ t ∈ train, A s = T t) ∧
      (∀ s : S, ∀ f ∈ Obs, |(A s).expect f - (T s).expect f| ≤ K * delta) := by
  classical
  choose t ht hdist using hnet
  refine ⟨fun s => T (t s), fun s => ⟨t s, ht s, rfl⟩, fun s f hf => ?_⟩
  refine (hT f hf (t s) s).trans ?_
  have h0 : dist (t s) s ≤ delta := by rw [dist_comm]; exact hdist s
  exact mul_le_mul_of_nonneg_left h0 hK

/-- **A switch is an error floor for every smooth model.**  If two inputs at distance `d`
have ensembles differing by `G` on an observable, no predictor with Lipschitz constant `L`
can be more accurate than `(G − L·d)/2` on both of them. -/
theorem switch_forces_capacity {L : ℝ} {T A : S → Ens X} {f : X → ℝ} {s s' : S}
    (hA : |(A s).expect f - (A s').expect f| ≤ L * dist s s') :
    |(T s).expect f - (T s').expect f| - L * dist s s'
      ≤ |(A s).expect f - (T s).expect f| + |(A s').expect f - (T s').expect f| := by
  have htri : |(T s).expect f - (T s').expect f|
      ≤ |(T s).expect f - (A s).expect f| + |(A s).expect f - (A s').expect f|
        + |(A s').expect f - (T s').expect f| := by
    calc |(T s).expect f - (T s').expect f|
        ≤ |(T s).expect f - (A s).expect f| + |(A s).expect f - (T s').expect f| := by
          simpa using abs_sub_le ((T s).expect f) ((A s).expect f) ((T s').expect f)
      _ ≤ |(T s).expect f - (A s).expect f|
            + (|(A s).expect f - (A s').expect f| + |(A s').expect f - (T s').expect f|) := by
          gcongr
          simpa using abs_sub_le ((A s).expect f) ((A s').expect f) ((T s').expect f)
      _ = |(T s).expect f - (A s).expect f| + |(A s).expect f - (A s').expect f|
            + |(A s').expect f - (T s').expect f| := by ring
  have h1 : |(T s).expect f - (A s).expect f| = |(A s).expect f - (T s).expect f| :=
    abs_sub_comm _ _
  linarith

/-- **No smooth model of a switch.**  If the ensembles of two inputs at distance `d` differ
by `G` on some observable, then a predictor whose output varies with Lipschitz constant `L`
cannot be uniformly accurate to better than `(G − L·d)/2`. -/
theorem no_smooth_model_of_switch {L G d eps : ℝ} {T A : S → Ens X} {f : X → ℝ} {s s' : S}
    (hd : dist s s' = d) (hgap : G ≤ |(T s).expect f - (T s').expect f|)
    (hA : |(A s).expect f - (A s').expect f| ≤ L * d)
    (hacc : ∀ u : S, |(A u).expect f - (T u).expect f| ≤ eps) :
    G - L * d ≤ 2 * eps := by
  have h := switch_forces_capacity (L := L) (T := T) (A := A) (f := f) (s := s) (s' := s')
    (by rw [hd]; exact hA)
  rw [hd] at h
  have h1 := hacc s
  have h2 := hacc s'
  linarith

end IDR
