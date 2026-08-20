/-
# A general framework for conformational ensembles and the models that predict them

This file generalises the setting of `RequestProject.DisorderedRegions` from conformations
in `Fin n → ℝ` to an arbitrary conformation space `X`.

* An `Ens X` is a finitely supported probability distribution on `X`: a thermodynamic
  ensemble.  The *same* structure also serves as the definition of a *model*: a finite
  mixture / latent-variable generative model with `card` latent states, latent
  distribution `w` and decoder `pt`.  A model is therefore correct exactly when it is
  observationally equal (`Same`) to the target ensemble.
* `Deterministic` singles out the models that always return one structure -- the
  idealisation of a classical structure predictor.
* `prob` is the probability of an individual conformation, and `card_le_of_same` is the
  basic **capacity theorem**: a model must have at least as many mixture components as the
  target ensemble has distinct populated conformations.
-/
import Mathlib

namespace IDR

open Finset
open scoped Classical

variable {X Y : Type*}

/-- A conformational ensemble on the conformation space `X`: finitely many conformations
carrying probability weights.  Read as a *model*, this is a finite latent-variable
generative model: `card` latent states, latent distribution `w`, decoder `pt`. -/
structure Ens (X : Type*) where
  /-- number of populated conformations (equivalently, of mixture components) -/
  card : ℕ
  /-- the conformations (equivalently, the decoder) -/
  pt : Fin card → X
  /-- the statistical weights (equivalently, the latent distribution) -/
  w : Fin card → ℝ
  w_nonneg : ∀ j, 0 ≤ w j
  w_sum : ∑ j, w j = 1

namespace Ens

variable (E : Ens X)

/-- The ensemble average of an observable `f`. -/
def expect (f : X → ℝ) : ℝ := ∑ j, E.w j * f (E.pt j)

@[simp] lemma expect_const (a : ℝ) : E.expect (fun _ => a) = a := by
  simp [expect, ← Finset.sum_mul, E.w_sum]

lemma expect_add (f g : X → ℝ) :
    E.expect (fun c => f c + g c) = E.expect f + E.expect g := by
  simp [expect, mul_add, Finset.sum_add_distrib]

lemma expect_smul (a : ℝ) (f : X → ℝ) :
    E.expect (fun c => a * f c) = a * E.expect f := by
  simp only [expect, Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

lemma expect_sub (f g : X → ℝ) :
    E.expect (fun c => f c - g c) = E.expect f - E.expect g := by
  simp [expect, mul_sub, Finset.sum_sub_distrib]

lemma expect_sum {ι : Type*} (s : Finset ι) (g : ι → X → ℝ) :
    E.expect (fun c => ∑ i ∈ s, g i c) = ∑ i ∈ s, E.expect (g i) := by
  simp only [expect, Finset.mul_sum]
  exact Finset.sum_comm

lemma expect_nonneg {f : X → ℝ} (hf : ∀ c, 0 ≤ f c) : 0 ≤ E.expect f :=
  Finset.sum_nonneg fun j _ => mul_nonneg (E.w_nonneg j) (hf _)

lemma expect_mono {f g : X → ℝ} (h : ∀ c, f c ≤ g c) : E.expect f ≤ E.expect g :=
  Finset.sum_le_sum fun j _ => by
    exact mul_le_mul_of_nonneg_left (h _) (E.w_nonneg j)

/-- Observational equality of two ensembles: every observable has the same average.  This
is the strongest possible notion of a model being *right*: no experiment, i.e. no
measurable quantity, can tell the two apart. -/
def Same (E F : Ens X) : Prop := ∀ f : X → ℝ, E.expect f = F.expect f

lemma Same.refl (E : Ens X) : E.Same E := fun _ => rfl

lemma Same.symm {E F : Ens X} (h : E.Same F) : F.Same E := fun f => (h f).symm

lemma Same.trans {E F G : Ens X} (h₁ : E.Same F) (h₂ : F.Same G) : E.Same G :=
  fun f => (h₁ f).trans (h₂ f)

/-- The probability that the ensemble assigns to the individual conformation `x`. -/
noncomputable def prob (x : X) : ℝ := E.expect (fun y => if y = x then (1 : ℝ) else 0)

lemma prob_nonneg (x : X) : 0 ≤ E.prob x :=
  E.expect_nonneg fun y => by by_cases h : y = x <;> simp [h]

lemma prob_pos_iff (x : X) : 0 < E.prob x ↔ ∃ j, 0 < E.w j ∧ E.pt j = x := by
  constructor
  · intro h
    by_contra hcon
    push_neg at hcon
    have : E.prob x = 0 := by
      simp only [prob, expect]
      refine Finset.sum_eq_zero fun j _ => ?_
      by_cases hj : E.pt j = x
      · have := hcon j
        have hw : E.w j = 0 := le_antisymm (by
          rcases lt_or_ge 0 (E.w j) with h' | h'
          · exact absurd hj (this h')
          · exact h') (E.w_nonneg j)
        simp [hw]
      · simp [hj]
    exact absurd this (ne_of_gt h)
  · rintro ⟨j, hj, hx⟩
    simp only [prob, expect]
    refine Finset.sum_pos' (fun i _ => ?_) ⟨j, Finset.mem_univ j, ?_⟩
    · by_cases h : E.pt i = x <;> simp [h, E.w_nonneg i]
    · simp [hx, hj]

/-- If the listed conformations are distinct, the probability of one of them is its
weight. -/
lemma prob_pt_of_injective (E : Ens X) (hinj : Function.Injective E.pt) (j : Fin E.card) :
    E.prob (E.pt j) = E.w j := by
  simp only [prob, expect]
  rw [Finset.sum_eq_single j]
  · simp
  · intro k _ hk
    have : E.pt k ≠ E.pt j := fun h => hk (hinj h)
    simp [this]
  · intro h
    exact absurd (Finset.mem_univ j) h

/-- The point mass at `x`: the ensemble (equivalently, the model) that always returns the
single conformation `x`. -/
def dirac (x : X) : Ens X where
  card := 1
  pt := fun _ => x
  w := fun _ => 1
  w_nonneg := by intro j; norm_num
  w_sum := by simp

@[simp] lemma expect_dirac (x : X) (f : X → ℝ) : (dirac x).expect f = f x := by
  simp [expect, dirac]

/-- A *deterministic* (single-structure) model: observationally a point mass.  This is the
idealisation of a classical structure predictor, which returns one set of coordinates. -/
def Deterministic (E : Ens X) : Prop := ∃ x : X, E.Same (dirac x)

lemma deterministic_iff (E : Ens X) :
    E.Deterministic ↔ ∃ x : X, ∀ f : X → ℝ, E.expect f = f x := by
  constructor
  · rintro ⟨x, hx⟩
    exact ⟨x, fun f => by simpa using hx f⟩
  · rintro ⟨x, hx⟩
    exact ⟨x, fun f => by simpa using hx f⟩

lemma dirac_deterministic (x : X) : (dirac x : Ens X).Deterministic := ⟨x, Same.refl _⟩

lemma prob_eq_of_same {E F : Ens X} (h : E.Same F) (x : X) : E.prob x = F.prob x :=
  h _

/-- **Capacity theorem.**  If a model `M` reproduces the ensemble `E`, then `M` has at
least as many mixture components (latent states) as `E` has distinct populated
conformations.  Model complexity must therefore scale with the size of the conformational
ensemble: a fixed, finite architecture cannot cover arbitrarily broad disorder. -/
theorem card_le_of_same {L : ℕ} {E M : Ens X} (h : M.Same E) (g : Fin L → X)
    (hg : Function.Injective g) (hpos : ∀ l, 0 < E.prob (g l)) : L ≤ M.card := by
  have hM : ∀ l, ∃ j : Fin M.card, M.pt j = g l := by
    intro l
    have : 0 < M.prob (g l) := by rw [prob_eq_of_same h]; exact hpos l
    obtain ⟨j, -, hj⟩ := (M.prob_pos_iff (g l)).1 this
    exact ⟨j, hj⟩
  choose z hz using hM
  have hinj : Function.Injective z := by
    intro a b hab
    apply hg
    rw [← hz a, ← hz b, hab]
  simpa using Fintype.card_le_of_injective z hinj

/-- Two conformations that are both populated force at least two mixture components; in
particular the ensemble is not deterministic. -/
theorem not_deterministic_of_two_points {E : Ens X} {x y : X} (hx : 0 < E.prob x)
    (hy : 0 < E.prob y) (hxy : x ≠ y) : ¬ E.Deterministic := by
  rintro ⟨a, ha⟩
  -- the point mass at `a` gives probability `0` to anything different from `a`
  have key : ∀ b : X, b ≠ a → E.prob b = 0 := by
    intro b hb
    have := ha (fun y => if y = b then (1 : ℝ) else 0)
    simpa [prob, Ne.symm hb] using this
  rcases eq_or_ne x a with rfl | hxa
  · exact absurd (key y (Ne.symm hxy)) (ne_of_gt hy)
  · exact absurd (key x hxa) (ne_of_gt hx)

end Ens

end IDR
