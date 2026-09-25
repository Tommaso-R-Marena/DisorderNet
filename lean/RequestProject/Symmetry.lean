/-
# Part XI.2  Equivariance: a symmetric target has no single-structure answer

Every serious structural architecture is built to be *equivariant*: rotate the input and the
output rotates with it; permute equivalent copies and the answer permutes.  Equivariance is
the design principle of the entire field, and it is right.  This file proves that for a
disordered region it has a consequence which is usually overlooked, and which is fatal to
single-structure output.

If the target ensemble is itself invariant under a symmetry `s` -- three rotamers of a
side-chain torsion equally populated, a homo-repeat, a symmetric dimer interface -- then any
*equivariant* predictor that returns one structure must return a **fixed point of `s`**
(`equivariant_point_is_fixed`).  Two consequences:

* if `s` has no fixed points, no equivariant single-structure predictor exists at all
  (`no_equivariant_point_predictor`);
* if the fixed points of `s` are unpopulated -- the generic case: the centroid of a
  symmetric set of conformations is not itself a conformation -- then the structure the
  predictor returns has probability zero in the truth
  (`equivariant_point_prediction_unpopulated`).  It is not a slightly wrong structure; it is
  a conformation the region never adopts.

`rotamer_prediction_is_origin` and `rotamer_prediction_unpopulated` make this concrete on the
three-fold symmetric rotamer triangle -- the cube roots of unity in the plane, the standard
caricature of an sp3 torsion with gauche+, trans and gauche- equally populated.  The forced
answer is the centroid `0`, which is at distance `1` from every conformation in the ensemble.

The positive counterpart is `equivariant_ensemble_predictor_exists`: an *ensemble*-valued
predictor has no such obstruction, because the set of ensembles, unlike the conformation
space, contains the symmetric average.  Equivariance is not the problem; equivariance
*plus* a point-valued output is.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.CoarseGraining

namespace IDR

open Finset

variable {X : Type*}

/-! ## The obstruction -/

/-- **An equivariant single-structure predictor must return a fixed point of any symmetry of
its input.**  `hsame` is the modelling requirement that the prediction depends on the target
only through its observables; `hequiv` is equivariance; `hinv` says the target ensemble is
symmetric. -/
theorem equivariant_point_is_fixed {s : X → X} {A : Ens X → X}
    (hsame : ∀ E F : Ens X, E.Same F → A E = A F)
    (hequiv : ∀ E : Ens X, A (E.map s) = s (A E))
    {E : Ens X} (hinv : (E.map s).Same E) : s (A E) = A E := by
  rw [← hequiv E, hsame _ _ hinv]

/-- If the symmetry has no fixed point, an equivariant single-structure predictor cannot
exist. -/
theorem no_equivariant_point_predictor {s : X → X} (hfree : ∀ x : X, s x ≠ x)
    {E : Ens X} (hinv : (E.map s).Same E) :
    ¬ ∃ A : Ens X → X, (∀ E F : Ens X, E.Same F → A E = A F) ∧
      (∀ E : Ens X, A (E.map s) = s (A E)) := by
  rintro ⟨A, hsame, hequiv⟩
  exact hfree (A E) (equivariant_point_is_fixed hsame hequiv hinv)

/-- **The generic case.**  If the fixed points of the symmetry are unpopulated, then the
structure an equivariant predictor returns is one the region never adopts: its probability in
the true ensemble is zero. -/
theorem equivariant_point_prediction_unpopulated {s : X → X} {A : Ens X → X}
    (hsame : ∀ E F : Ens X, E.Same F → A E = A F)
    (hequiv : ∀ E : Ens X, A (E.map s) = s (A E))
    {E : Ens X} (hinv : (E.map s).Same E) (hfix : ∀ x : X, s x = x → E.prob x = 0) :
    E.prob (A E) = 0 :=
  hfix _ (equivariant_point_is_fixed hsame hequiv hinv)

/-- **The positive counterpart.**  An ensemble-valued predictor can be equivariant *and*
exactly right: the identity is one.  The obstruction of `no_equivariant_point_predictor` is
not caused by equivariance but by the point-valued output type -- conformation space has no
room for the symmetric average, the space of ensembles does. -/
theorem equivariant_ensemble_predictor_exists (s : X → X) :
    ∃ A : Ens X → Ens X, (∀ E : Ens X, (A E).Same E) ∧
      (∀ E : Ens X, (A (E.map s)).Same ((A E).map s)) :=
  ⟨id, fun E => Ens.Same.refl E, fun _ => Ens.Same.refl _⟩

/-! ## The three-fold rotamer triangle -/

namespace Rotamer

open Complex

/-- A primitive cube root of unity: the `120°` rotation of the plane. -/
noncomputable def om : ℂ := Complex.exp ((2 * Real.pi / 3 : ℝ) * Complex.I)

lemma om_pow_three : om ^ 3 = 1 := by
  rw [om, ← Complex.exp_nat_mul,
    show ((3 : ℕ) : ℂ) * (((2 * Real.pi / 3 : ℝ) : ℂ) * Complex.I)
      = ((2 * Real.pi : ℝ) : ℂ) * Complex.I by push_cast; ring]
  simp

lemma om_ne_one : om ≠ 1 := by
  intro h
  have h2 := congrArg Complex.re h
  rw [om, Complex.exp_ofReal_mul_I_re,
    show (2 * Real.pi / 3 : ℝ) = Real.pi - Real.pi / 3 by ring,
    Real.cos_pi_sub, Real.cos_pi_div_three] at h2
  norm_num at h2

lemma abs_om : ‖om‖ = 1 := by
  rw [om, Complex.norm_exp_ofReal_mul_I]

lemma om_ne_zero : om ≠ 0 := by
  intro h
  have hone := abs_om
  rw [h] at hone
  simp at hone

/-- The three rotamers: the cube roots of unity, equally populated.  This is the standard
caricature of an sp3 torsion whose gauche+, trans and gauche- states are degenerate. -/
noncomputable def ens : Ens ℂ where
  card := 3
  pt := fun j => om ^ (j : ℕ)
  w := fun _ => 1 / 3
  w_nonneg := by norm_num
  w_sum := by norm_num

/-- The rotation by `120°`. -/
noncomputable def rot : ℂ → ℂ := fun z => om * z

/-- The rotamer ensemble is invariant under the three-fold rotation. -/
lemma ens_invariant : (ens.map rot).Same ens := by
  intro f
  rw [Ens.expect_map]
  simp only [Ens.expect, ens, rot, Fin.sum_univ_three, Fin.val_zero, Fin.val_one, Fin.val_two]
  have h0 : om * om ^ (0 : ℕ) = om ^ (1 : ℕ) := by ring
  have h1 : om * om ^ (1 : ℕ) = om ^ (2 : ℕ) := by ring
  have h2 : om * om ^ (2 : ℕ) = om ^ (0 : ℕ) := by
    rw [show om * om ^ (2 : ℕ) = om ^ 3 by ring, om_pow_three]; norm_num
  rw [h0, h1, h2]
  ring

/-- Only the centroid is fixed by the rotation. -/
lemma rot_fixed_iff {z : ℂ} : rot z = z ↔ z = 0 := by
  constructor
  · intro h
    have : (om - 1) * z = 0 := by rw [sub_mul, one_mul, rot] at *; linear_combination h
    rcases mul_eq_zero.1 this with h' | h'
    · exact absurd (sub_eq_zero.1 h') om_ne_one
    · exact h'
  · rintro rfl; simp [rot]

/-- The centroid is not one of the rotamers: it carries no population. -/
lemma prob_zero_of_fixed {z : ℂ} (hz : rot z = z) : ens.prob z = 0 := by
  rw [rot_fixed_iff] at hz
  subst hz
  by_contra h
  obtain ⟨j, -, hj⟩ := (ens.prob_pos_iff (0 : ℂ)).1
    (lt_of_le_of_ne (ens.prob_nonneg 0) (Ne.symm h))
  exact pow_ne_zero _ om_ne_zero hj

/-- **The forced answer is the centroid.**  On the three-fold symmetric rotamer ensemble,
every equivariant single-structure predictor returns `0`. -/
theorem rotamer_prediction_is_origin {A : Ens ℂ → ℂ}
    (hsame : ∀ E F : Ens ℂ, E.Same F → A E = A F)
    (hequiv : ∀ E : Ens ℂ, A (E.map rot) = rot (A E)) : A ens = 0 :=
  rot_fixed_iff.1 (equivariant_point_is_fixed hsame hequiv ens_invariant)

/-- **And the centroid is not a conformation.**  The structure returned has probability zero
in the target: the prediction is not an inaccurate structure, it is a structure the region
never adopts. -/
theorem rotamer_prediction_unpopulated {A : Ens ℂ → ℂ}
    (hsame : ∀ E F : Ens ℂ, E.Same F → A E = A F)
    (hequiv : ∀ E : Ens ℂ, A (E.map rot) = rot (A E)) : ens.prob (A ens) = 0 :=
  equivariant_point_prediction_unpopulated hsame hequiv ens_invariant
    (fun _ hz => prob_zero_of_fixed hz)

/-- Quantitatively: the forced answer is at distance `1` from every conformation in the
ensemble, i.e. a full rotamer radius away. -/
theorem rotamer_prediction_distance {A : Ens ℂ → ℂ}
    (hsame : ∀ E F : Ens ℂ, E.Same F → A E = A F)
    (hequiv : ∀ E : Ens ℂ, A (E.map rot) = rot (A E)) (j : Fin ens.card) :
    dist (A ens) (ens.pt j) = 1 := by
  rw [rotamer_prediction_is_origin hsame hequiv]
  simp only [ens, dist_eq_norm, zero_sub, norm_neg, norm_pow, abs_om, one_pow]

end Rotamer

end IDR
