/-
# Every architectural invariance has a price

An architecture is a set of invariances: a window-limited network cannot see outside its
receptive field, a composition-based predictor cannot see sequence patterning, a
context-blind predictor cannot see the binding partner, a coarse-grained model cannot see
the atoms.  This file proves, in one blow and quantitatively, that **any invariance the
model has and the target does not costs at least half the distance between the targets it
conflates**.

* `invariance_cost` / `invariance_cost_pair` -- the master bound.  If a predictor returns
  observationally identical answers on two inputs, its `ℓ¹` error on one of them is at
  least half the `ℓ¹` distance between the two targets.  Every obstruction below is an
  instance, obtained by exhibiting one conflated pair.
* `receptive_field_error` and `distal_switch_error` -- **locality**.  A predictor whose
  output depends only on the residues in a window `S` is off by at least `1` in `ℓ¹`
  (a 50% population error) on a region whose conformational state is switched by a single
  residue outside `S`.  Long-range coupling in a disordered region is not a matter of
  needing more parameters: a bounded receptive field is *provably* wrong.
* `composition_blind_error` -- **patterning beats composition**.  Two sequences of
  identical amino-acid composition can have different ensembles, so any predictor that
  reads only composition (net charge, fraction of charged residues, hydrophobicity) is off
  by at least `1` in `ℓ¹` on one of them.  Charge *patterning*, not composition, must enter
  the model.
* `context_blind_error` -- **context**.  The quantitative form of context blindness: a
  predictor that ignores the partner or the post-translational modification is off by at
  least half the distance between the two ensembles it must tell apart.
* `no_two_tower_model` -- **no independent-domain model**.  A model that predicts two
  segments by two independent heads, however each head is computed, produces a factorised
  distribution and therefore cannot represent inter-segment coupling.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Geometry
import RequestProject.Statistics
import RequestProject.ModelNature
import RequestProject.EnergyModels
import RequestProject.Metric

namespace IDR

open Finset
open scoped Classical

/-! ## The master bound -/

section Master

variable {X I : Type*} [Fintype X]

/-- Two conflated inputs: the errors on them cannot both be small, because together they
must bridge the whole distance between the two targets. -/
theorem invariance_cost {A T : I → Ens X} {i i' : I} (h : (A i).Same (A i')) :
    Ens.ell1 (T i) (T i') ≤ Ens.ell1 (A i) (T i) + Ens.ell1 (A i') (T i') := by
  have h0 : Ens.ell1 (A i) (A i') = 0 := (Ens.ell1_eq_zero_iff _ _).2 h
  calc Ens.ell1 (T i) (T i')
      ≤ Ens.ell1 (T i) (A i') + Ens.ell1 (A i') (T i') := Ens.ell1_triangle _ _ _
    _ ≤ (Ens.ell1 (T i) (A i) + Ens.ell1 (A i) (A i')) + Ens.ell1 (A i') (T i') := by
        have := Ens.ell1_triangle (T i) (A i) (A i')
        linarith
    _ = Ens.ell1 (A i) (T i) + Ens.ell1 (A i') (T i') := by
        rw [h0, Ens.ell1_comm (T i) (A i)]; ring

/-- **The price of an invariance.**  A predictor that cannot distinguish two inputs is, on
one of them, off by at least half the distance between their targets. -/
theorem invariance_cost_pair {A T : I → Ens X} {i i' : I} (h : (A i).Same (A i')) :
    Ens.ell1 (T i) (T i') / 2 ≤ max (Ens.ell1 (A i) (T i)) (Ens.ell1 (A i') (T i')) := by
  have hb := invariance_cost (A := A) (T := T) h
  rcases le_total (Ens.ell1 (A i) (T i)) (Ens.ell1 (A i') (T i')) with hle | hle
  · rw [max_eq_right hle]; linarith
  · rw [max_eq_left hle]; linarith

/-- In `ApproxSame` form: an invariant predictor uniformly accurate to `eps` forces the
conflated targets to be within `2 eps` of each other. -/
theorem approx_of_invariant {A T : I → Ens X} {i i' : I} (h : (A i).Same (A i'))
    {eps : ℝ} (h1 : ApproxSame eps (A i) (T i)) (h2 : ApproxSame eps (A i') (T i')) :
    Ens.ell1 (T i) (T i') ≤ 2 * eps := by
  have e1 := (Ens.approxSame_iff_ell1_le eps (A i) (T i)).1 h1
  have e2 := (Ens.approxSame_iff_ell1_le eps (A i') (T i')).1 h2
  have := invariance_cost (A := A) (T := T) h
  linarith

end Master

/-! ## Two point masses are maximally far apart -/

section Dirac

variable {X : Type*} [Fintype X] [DecidableEq X]

/-- Distinct single structures are at the maximal `ℓ¹` distance `2`. -/
lemma ell1_dirac_dirac {a b : X} (hab : a ≠ b) :
    Ens.ell1 (Ens.dirac a) (Ens.dirac b) = 2 := by
  classical
  have hzero : ∀ x ∈ (Finset.univ : Finset X) \ ({a, b} : Finset X),
      |(Ens.dirac a).prob x - (Ens.dirac b).prob x| = 0 := by
    intro x hx
    have hxa : a ≠ x := by
      intro h; exact (Finset.mem_sdiff.1 hx).2 (by simp [← h])
    have hxb : b ≠ x := by
      intro h; exact (Finset.mem_sdiff.1 hx).2 (by simp [← h])
    simp [prob_dirac, hxa, hxb]
  rw [Ens.ell1, ← Finset.sum_subset (Finset.subset_univ ({a, b} : Finset X))
    (fun x hx hxS => hzero x (Finset.mem_sdiff.2 ⟨hx, hxS⟩)), Finset.sum_pair hab]
  simp only [prob_dirac, if_neg hab, if_neg (Ne.symm hab)]
  norm_num

end Dirac

/-! ## Locality: a bounded receptive field -/

section Locality

variable {L : ℕ} {Alph X : Type*} [Fintype X]

/-- Two sequences agree on the set of positions `S`. -/
def SeqAgreeOn (S : Finset (Fin L)) (s s' : Fin L → Alph) : Prop := ∀ i ∈ S, s i = s' i

/-- A predictor with receptive field `S`: its output depends only on the residues in
`S`. -/
def ReceptiveField (S : Finset (Fin L)) (A : (Fin L → Alph) → Ens X) : Prop :=
  ∀ s s', SeqAgreeOn S s s' → (A s).Same (A s')

/-- **The cost of a bounded receptive field.**  On two sequences that agree inside the
window, a window-limited predictor is off by at least half the distance between their
ensembles. -/
theorem receptive_field_error {S : Finset (Fin L)} {A T : (Fin L → Alph) → Ens X}
    (hA : ReceptiveField S A) {s s' : Fin L → Alph} (hss : SeqAgreeOn S s s') :
    Ens.ell1 (T s) (T s') / 2 ≤ max (Ens.ell1 (A s) (T s)) (Ens.ell1 (A s') (T s')) :=
  invariance_cost_pair (A := A) (T := T) (hA s s' hss)

variable [DecidableEq X]

/-- **A single distal residue defeats any window.**  Let the conformational state of the
region be switched by the residue at position `j` -- the target is one structure for one
identity of that residue and a different structure for another.  Then *every* predictor
whose receptive field `S` omits `j` has `ℓ¹` error at least `1`, i.e. it misplaces at least
half of the population, on one of the two sequences.  Enlarging the network does not help;
only enlarging the receptive field does. -/
theorem distal_switch_error {S : Finset (Fin L)} {A : (Fin L → Alph) → Ens X}
    (hA : ReceptiveField S A) {j : Fin L} (hj : j ∉ S) [DecidableEq Alph]
    (u v : X) (huv : u ≠ v) (a b : Alph) (hab : a ≠ b) (s : Fin L → Alph)
    (T : (Fin L → Alph) → Ens X)
    (hT : ∀ t : Fin L → Alph, T t = if t j = a then Ens.dirac u else Ens.dirac v) :
    1 ≤ max (Ens.ell1 (A (Function.update s j a)) (T (Function.update s j a)))
        (Ens.ell1 (A (Function.update s j b)) (T (Function.update s j b))) := by
  classical
  set s₁ := Function.update s j a with hs₁
  set s₂ := Function.update s j b with hs₂
  have hagree : SeqAgreeOn S s₁ s₂ := by
    intro i hi
    have hij : i ≠ j := fun h => hj (h ▸ hi)
    simp [hs₁, hs₂, Function.update_of_ne hij]
  have h1 : T s₁ = Ens.dirac u := by
    rw [hT s₁, if_pos]
    simp [hs₁]
  have h2 : T s₂ = Ens.dirac v := by
    rw [hT s₂, if_neg]
    simp [hs₂, Ne.symm hab]
  have hdist : Ens.ell1 (T s₁) (T s₂) = 2 := by
    rw [h1, h2]; exact ell1_dirac_dirac huv
  have := receptive_field_error (T := T) hA hagree
  rw [hdist] at this
  linarith

end Locality

/-! ## Composition versus patterning -/

section Patterning

/-- The composition of a two-letter sequence: how many residues of the first kind it
carries (net charge, fraction of charged residues, ...). -/
def compo {L : ℕ} (s : Fin L → Bool) : ℕ := (Finset.univ.filter (fun i => s i = true)).card

/-- Two four-residue sequences of identical composition but different patterning. -/
def patA : Fin 4 → Bool := ![true, true, false, false]

/-- The second, permuted, sequence: same composition, alternating pattern. -/
def patB : Fin 4 → Bool := ![true, false, true, false]

lemma compo_patA_eq_patB : compo patA = compo patB := by decide

lemma patA_one_ne_patB_one : patA 1 ≠ patB 1 := by decide

/-- **Composition is not enough: patterning must enter the model.**  Take a region whose
conformational state is set by the identity of one residue -- so that the two sequences
above, of identical composition, have different ensembles.  Any predictor that reads only
the composition of the sequence is then off by at least `1` in `ℓ¹` on one of them: it
misplaces at least half of the conformational population.  Net charge and residue fractions
cannot be the model's only sequence features. -/
theorem composition_blind_error (A : (Fin 4 → Bool) → Ens Bool)
    (hA : ∀ s s', compo s = compo s' → (A s).Same (A s'))
    (T : (Fin 4 → Bool) → Ens Bool) (hT : ∀ s, T s = Ens.dirac (s 1)) :
    1 ≤ max (Ens.ell1 (A patA) (T patA)) (Ens.ell1 (A patB) (T patB)) := by
  have hinv : (A patA).Same (A patB) := hA _ _ compo_patA_eq_patB
  have hdist : Ens.ell1 (T patA) (T patB) = 2 := by
    rw [hT patA, hT patB]
    exact ell1_dirac_dirac patA_one_ne_patB_one
  have := invariance_cost_pair (A := A) (T := T) hinv
  rw [hdist] at this
  linarith

end Patterning

/-! ## Context, quantitatively -/

section Context

variable {X S C : Type*} [Fintype X]

/-- **The quantitative cost of context blindness.**  A predictor that returns the same
ensemble for a chain free and bound (or unmodified and phosphorylated) is off, on one of
the two, by at least half the `ℓ¹` distance between the two true ensembles.  Since binding
and modification can reorganise a disordered region completely, that distance is of order
one. -/
theorem context_blind_error {A T : S × C → Ens X} {s : S} {c₁ c₂ : C}
    (h : A (s, c₁) = A (s, c₂)) :
    Ens.ell1 (T (s, c₁)) (T (s, c₂)) / 2
      ≤ max (Ens.ell1 (A (s, c₁)) (T (s, c₁))) (Ens.ell1 (A (s, c₂)) (T (s, c₂))) :=
  invariance_cost_pair (A := A) (T := T) (by rw [h]; exact Ens.Same.refl _)

end Context

/-! ## Independent heads cannot couple -/

/-- **No two-tower model.**  If a model predicts the conformation of two segments with two
independent heads -- whatever each head computes from the sequence and the context -- its
output is a product distribution, and it therefore fails on a target with correlated
segments.  Coupling has to be carried by a shared latent variable, exactly as in
`IDR.mixture_of_products_universal`. -/
theorem no_two_tower_model {I : Type*} (P Q : I → Ens ℝ) (i : I) :
    ¬ ((P i).prod (Q i)).Same corrPair :=
  no_prod_captures_corrPair (P i) (Q i)

end IDR
