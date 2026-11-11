/-
# Relabelling residues does not change the grouped AUC

The hardness construction of `AUCHardness.lean` builds its residues out of a structured index
type; a decision problem needs them numbered `0, …, N-1`.  These lemmas transport the Mann–Whitney
statistic and the three pair sets along any bijection of the residue set, which is what makes the
two presentations interchangeable.
-/
import RequestProject.AUCCeiling

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Relabel

variable {I J G : Type*} [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J] [DecidableEq G]

/-- The image of a pair set under a relabelling. -/
def mapPairs (e : I ≃ J) (X : Finset (I × I)) : Finset (J × J) :=
  X.image (fun q => (e q.1, e q.2))

omit [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J] in
/-- The pair map underlying `mapPairs` is injective. -/
theorem pairMap_injective (e : I ≃ J) :
    Function.Injective (fun q : I × I => (e q.1, e q.2)) := by
  rintro ⟨a, b⟩ ⟨c, d⟩ h
  simp only [Prod.mk.injEq, EmbeddingLike.apply_eq_iff_eq] at h
  simp [h.1, h.2]

omit [Fintype I] [Fintype J] [DecidableEq I] in
theorem mem_mapPairs (e : I ≃ J) (X : Finset (I × I)) (q : J × J) :
    q ∈ mapPairs e X ↔ (e.symm q.1, e.symm q.2) ∈ X := by
  simp only [mapPairs, Finset.mem_image, Prod.ext_iff]
  constructor
  · rintro ⟨a, hab, h1, h2⟩
    simpa [← h1, ← h2] using hab
  · intro h
    exact ⟨(e.symm q.1, e.symm q.2), h, by simp, by simp⟩

omit [Fintype I] [Fintype J] [DecidableEq I] in
theorem U_mapPairs (e : I ≃ J) (X : Finset (I × I)) (s : I → ℝ) :
    U (mapPairs e X) (fun j => s (e.symm j)) = U X s := by
  rw [U, mapPairs, Finset.sum_image (fun a _ b _ h => pairMap_injective e h)]
  exact Finset.sum_congr rfl fun q _ => by simp

omit [Fintype I] [Fintype J] [DecidableEq I] in
theorem card_mapPairs (e : I ≃ J) (X : Finset (I × I)) : (mapPairs e X).card = X.card :=
  Finset.card_image_of_injective _ (pairMap_injective e)

omit [DecidableEq I] in
theorem allPairs_relabel (e : I ≃ J) (lab : I → Bool) :
    allPairs (fun j => lab (e.symm j)) = mapPairs e (allPairs lab) := by
  ext q
  rw [mem_mapPairs]
  simp [allPairs, posSet, negSet, Finset.mem_product]

omit [DecidableEq I] in
theorem withinPairs_relabel (e : I ≃ J) (lab : I → Bool) (grp : I → G) :
    withinPairs (fun j => lab (e.symm j)) (fun j => grp (e.symm j))
      = mapPairs e (withinPairs lab grp) := by
  ext q
  rw [mem_mapPairs]
  simp [withinPairs, allPairs, posSet, negSet, Finset.mem_product]

omit [DecidableEq I] in
theorem betweenPairs_relabel (e : I ≃ J) (lab : I → Bool) (grp : I → G) :
    betweenPairs (fun j => lab (e.symm j)) (fun j => grp (e.symm j))
      = mapPairs e (betweenPairs lab grp) := by
  ext q
  rw [mem_mapPairs]
  simp [betweenPairs, allPairs, posSet, negSet, Finset.mem_product]

omit [Fintype I] [Fintype J] [DecidableEq I] [DecidableEq J] [DecidableEq G] in
theorem shift_relabel (e : I ≃ J) (grp : I → G) (b : G → ℝ) (s : I → ℝ) :
    shift (fun j => grp (e.symm j)) b (fun j => s (e.symm j))
      = fun j => (shift grp b s) (e.symm j) := rfl

omit [DecidableEq I] in
/-- **The pooled statistic is unchanged by relabelling.** -/
theorem U_allPairs_relabel (e : I ≃ J) (lab : I → Bool) (s : I → ℝ) :
    U (allPairs (fun j => lab (e.symm j))) (fun j => s (e.symm j)) = U (allPairs lab) s := by
  rw [allPairs_relabel e lab, U_mapPairs]

omit [DecidableEq I] in
/-- **The within-protein statistic is unchanged by relabelling.** -/
theorem U_withinPairs_relabel (e : I ≃ J) (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    U (withinPairs (fun j => lab (e.symm j)) (fun j => grp (e.symm j))) (fun j => s (e.symm j))
      = U (withinPairs lab grp) s := by
  rw [withinPairs_relabel e lab grp, U_mapPairs]

omit [DecidableEq I] in
/-- **The number of cross-protein comparisons is unchanged by relabelling.** -/
theorem card_betweenPairs_relabel (e : I ≃ J) (lab : I → Bool) (grp : I → G) :
    (betweenPairs (fun j => lab (e.symm j)) (fun j => grp (e.symm j))).card
      = (betweenPairs lab grp).card := by
  rw [betweenPairs_relabel e lab grp, card_mapPairs]

end Relabel

end IDR.GroupedAUC
