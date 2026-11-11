/-
# Resolution: coarse descriptors constrain a model but never determine it

Models of disorder are usually judged, and often trained, at a reduced resolution: a radius
of gyration, a secondary-structure propensity, a per-residue disorder score, a contact
frequency.  Formally each of these is a *coarse-graining map* `h : X → Y` from conformation
space to a descriptor space, and the model is compared with the truth only through the
push-forward `Ens.map h`.

* `map_same_of_same` -- correctness at full resolution implies correctness at every coarse
  resolution: matching the descriptors is a genuine *necessary* condition, and a model that
  fails it is refuted.
* `exists_same_coarse_ne_fine` -- but the converse fails: two observationally different
  ensembles can have identical coarse descriptions.  A model validated at reduced
  resolution is not validated.
* `not_separates_pullback_of_not_injective` and `coarse_trained_model_fails` -- worse, if
  the descriptor merges any two conformations then the family of coarse observables does
  not separate ensembles, so by the master theorem *every* model trained only on that
  descriptor is wrong on some target.  A lossy descriptor cannot be the model's training
  signal, however much data is collected in it.
* `exists_lift_of_surjective` -- and in the other direction, every coarse target is the
  shadow of some fine ensemble, so coarse data never singles out a fine model: the fibre
  of a coarse description is always non-empty, and (by the above) generally large.

The conclusion is a statement about *what the model must be*, not merely about how it is
tested: it must be defined -- and supervised -- at the resolution at which its predictions
are claimed to hold.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics

namespace IDR

open Finset
open scoped Classical

variable {X Y : Type*}

namespace Ens

/-- The coarse-grained (push-forward) ensemble: the distribution of the descriptor
`h`. -/
def map (h : X → Y) (E : Ens X) : Ens Y where
  card := E.card
  pt := fun j => h (E.pt j)
  w := E.w
  w_nonneg := E.w_nonneg
  w_sum := E.w_sum

@[simp] lemma expect_map (h : X → Y) (E : Ens X) (f : Y → ℝ) :
    (E.map h).expect f = E.expect (fun x => f (h x)) := rfl

end Ens

/-- **Coarse agreement is necessary.**  A model that is right at full resolution is right
about every descriptor. -/
theorem map_same_of_same (h : X → Y) {E F : Ens X} (hEF : E.Same F) :
    (E.map h).Same (F.map h) := fun f => hEF (fun x => f (h x))

/-- **Coarse agreement is not sufficient.**  Two ensembles can have the same distribution
of a descriptor -- here the magnitude of a coordinate, standing for any symmetric or
lossy reduced description -- and still differ observationally. -/
theorem exists_same_coarse_ne_fine :
    ∃ (h : Conf 1 → ℝ) (E F : Ens (Conf 1)), (E.map h).Same (F.map h) ∧ ¬ E.Same F := by
  refine ⟨fun c => |c 0|, pmOne, Ens.dirac ![(1 : ℝ)], ?_, ?_⟩
  · intro f
    rw [Ens.expect_map, Ens.expect_map, pmOne_expect, Ens.expect_dirac]
    norm_num
  · intro hs
    have := hs (fun c => c 0)
    rw [pmOne_expect, Ens.expect_dirac] at this
    norm_num at this

/-- The observables visible through the descriptor `h`: the pullbacks `f ∘ h`. -/
def PullbackObs (h : X → Y) : Set (X → ℝ) := {g | ∃ f : Y → ℝ, ∀ x, g x = f (h x)}

/-- **A descriptor that merges two conformations cannot determine the ensemble.**  As soon
as the reduced description is lossy anywhere, the family of observables it exposes fails to
separate ensembles. -/
theorem not_separates_pullback_of_not_injective {h : X → Y} {x y : X} (hxy : x ≠ y)
    (hh : h x = h y) : ¬ Separates (PullbackObs h) := by
  refine not_separates_iff.2 ⟨Ens.dirac x, Ens.dirac y, ?_, ?_⟩
  · rintro g ⟨f, hf⟩
    rw [Ens.expect_dirac, Ens.expect_dirac, hf, hf, hh]
  · intro hs
    have := hs (fun z => if z = x then (1 : ℝ) else 0)
    rw [Ens.expect_dirac, Ens.expect_dirac, if_pos rfl, if_neg (Ne.symm hxy)] at this
    norm_num at this

/-- **Every model supervised only by a lossy descriptor is wrong somewhere.**  Combining
the previous statement with the master theorem of `RequestProject.Statistics`: if the
model's dependence on the target is only through the descriptor `h`, and `h` merges even
one pair of conformations, then some target is not reproduced. -/
theorem coarse_trained_model_fails {h : X → Y} {x y : X} (hxy : x ≠ y) (hh : h x = h y)
    (A : Ens X → Ens X) (hA : ∀ E F : Ens X, AgreeOn (PullbackObs h) E F → A E = A F) :
    ∃ E : Ens X, ¬ (A E).Same E :=
  exists_failure_of_not_separating A hA (not_separates_pullback_of_not_injective hxy hh)

/-- **Coarse data never singles out a fine model.**  If the descriptor is onto, every
coarse-grained target is the shadow of some full-resolution ensemble; so a coarse
observation, however precise, leaves the fine model unconstrained beyond its own fibre. -/
theorem exists_lift_of_surjective {h : X → Y} (hsurj : Function.Surjective h) (F : Ens Y) :
    ∃ E : Ens X, (E.map h).Same F := by
  classical
  choose s hs using hsurj
  refine ⟨{ card := F.card
            pt := fun j => s (F.pt j)
            w := F.w
            w_nonneg := F.w_nonneg
            w_sum := F.w_sum }, fun f => ?_⟩
  show ∑ j : Fin F.card, F.w j * f (h (s (F.pt j))) = ∑ j : Fin F.card, F.w j * f (F.pt j)
  exact Finset.sum_congr rfl fun j _ => by rw [hs (F.pt j)]

end IDR
