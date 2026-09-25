/-
# Modular ensembles III: how large the price can be, and when it is small

Two complementary quantitative statements about building a full-length disordered ensemble
from fragment panels.

**When it is safe.**  `l1_le_of_conditional_defect`: if, conditionally on the seam state,
the two segments are decoupled to within `delta` — the natural way to say "the linker is
long and the ends do not talk" — then the glued model is within `delta·|X|·|Z|` of the truth
in the population `ℓ¹` metric.  No entropy, no Pinsker: a direct bound in the units the
model will be judged in.

**When it is not.**  A minimal but physically standard counterexample: the two termini of a
disordered region are held together by a long-range contact (electrostatic or hydrophobic),
while the intervening seam sits in a single state.  Then

* `longRange_margXY`, `longRange_margYZ` — each fragment panel is exactly what an
  uncorrelated ensemble would give, so **no measurement on either fragment can detect the
  coupling**;
* `longRange_glue` — the modular model is the uniform product;
* `longRange_l1` — its population error is `1`, the maximum possible for two distributions;
* `longRange_contact` — it reports the end-to-end contact probability as `1/2` when the
  truth is `1`;
* `longRange_cmi` — the seam information is `log 2`, and therefore
  `longRange_no_modular_model_is_close`: *every* modular model, however fitted, is at
  relative entropy at least `log 2` from the truth.
-/
import Mathlib
import RequestProject.ModularGluing
import RequestProject.ModularOptimality

namespace RequestProject.Modular

open Finset IDR.Pinsker

section Defect

variable {X Y Z : Type*} [Fintype X] [Fintype Y] [Fintype Z]
variable {p : X → Y → Z → ℝ}

/-- **Weak coupling across the seam is enough.**  If the joint law of the two segments given
the seam differs from the product of their conditional laws by at most `delta` — stated in
the cleared-denominator form `|p·p(y) − p(x,y)·p(y,z)| ≤ delta·p(y)²` — then the modular
model is accurate to `delta·|X|·|Z|` in the population metric. -/
theorem l1_le_of_conditional_defect (hp : ∀ x y z, 0 ≤ p x y z)
    (hs : ∑ x, ∑ y, ∑ z, p x y z = 1) {delta : ℝ}
    (hdef : ∀ x y z, |p x y z * margY p y - margXY p x y * margYZ p y z|
      ≤ delta * (margY p y) ^ 2) :
    ∑ x, ∑ y, ∑ z, |p x y z - glue p x y z|
      ≤ delta * (Fintype.card X : ℝ) * (Fintype.card Z : ℝ) := by
  have hpt : ∀ x y z, |p x y z - glue p x y z| ≤ delta * margY p y := by
    intro x y z
    rcases eq_or_lt_of_le (margY_nonneg hp y) with hy | hy
    · rw [eq_zero_of_margY_eq_zero hp hy.symm x z, glue,
        margXY_eq_zero_of_margY_eq_zero hp hy.symm x, ← hy]
      simp
    · have hrw : p x y z - glue p x y z
          = (p x y z * margY p y - margXY p x y * margYZ p y z) / margY p y := by
        rw [glue]; field_simp
      rw [hrw, abs_div, abs_of_pos hy, div_le_iff₀ hy]
      calc |p x y z * margY p y - margXY p x y * margYZ p y z|
          ≤ delta * (margY p y) ^ 2 := hdef x y z
        _ = delta * margY p y * margY p y := by ring
  have hstep : ∀ y : Y, ∑ x, ∑ z : Z, |p x y z - glue p x y z|
      ≤ (Fintype.card X : ℝ) * (Fintype.card Z : ℝ) * (delta * margY p y) := by
    intro y
    calc ∑ x, ∑ z : Z, |p x y z - glue p x y z|
        ≤ ∑ _x : X, ∑ _z : Z, delta * margY p y :=
          Finset.sum_le_sum fun x _ => Finset.sum_le_sum fun z _ => hpt x y z
      _ = (Fintype.card X : ℝ) * (Fintype.card Z : ℝ) * (delta * margY p y) := by
          simp [Finset.sum_const, Finset.card_univ, mul_assoc]
  have hswap : ∑ x, ∑ y, ∑ z, |p x y z - glue p x y z|
      = ∑ y, ∑ x, ∑ z, |p x y z - glue p x y z| := Finset.sum_comm
  have hsumY : ∑ y, margY p y = 1 := by
    rw [← hs, Finset.sum_comm]
    exact Finset.sum_congr rfl fun y _ => rfl
  calc ∑ x, ∑ y, ∑ z, |p x y z - glue p x y z|
      = ∑ y, ∑ x, ∑ z, |p x y z - glue p x y z| := hswap
    _ ≤ ∑ y, (Fintype.card X : ℝ) * (Fintype.card Z : ℝ) * (delta * margY p y) :=
        Finset.sum_le_sum fun y _ => hstep y
    _ = (Fintype.card X : ℝ) * (Fintype.card Z : ℝ) * delta * ∑ y, margY p y := by
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun y _ => by ring
    _ = delta * (Fintype.card X : ℝ) * (Fintype.card Z : ℝ) := by rw [hsumY]; ring

end Defect

/-! ## A long-range contact defeats every modular model -/

/-- The two termini of a disordered region are held together by a long-range contact: each
terminus is in one of two states (`false` = released, `true` = engaged) and the two are
always in the same state, while the intervening seam has a single state. -/
noncomputable def longRange : Bool → Unit → Bool → ℝ :=
  fun x _ z => if x = z then 1/2 else 0

lemma longRange_nonneg : ∀ x y z, 0 ≤ longRange x y z := by
  intro x y z
  rw [longRange]
  split <;> norm_num

lemma longRange_sum : ∑ x, ∑ y, ∑ z, longRange x y z = 1 := by
  simp [longRange]

/-- The first fragment's panel is exactly that of an uncorrelated ensemble. -/
lemma longRange_margXY (x : Bool) (y : Unit) : margXY longRange x y = 1/2 := by
  cases x <;> simp [margXY, longRange]

/-- The second fragment's panel is exactly that of an uncorrelated ensemble. -/
lemma longRange_margYZ (y : Unit) (z : Bool) : margYZ longRange y z = 1/2 := by
  cases z <;> simp [margYZ, longRange]

lemma longRange_margY (y : Unit) : margY longRange y = 1 := by
  simp [margY, longRange]

/-- The modular model is the uniform product: the coupling is gone. -/
lemma longRange_glue (x : Bool) (y : Unit) (z : Bool) : glue longRange x y z = 1/4 := by
  rw [glue, longRange_margXY, longRange_margYZ, longRange_margY]
  norm_num

/-- The modular model is at the maximal population distance from the truth. -/
theorem longRange_l1 : ∑ x, ∑ y, ∑ z, |longRange x y z - glue longRange x y z| = 1 := by
  simp [longRange, longRange_glue]
  norm_num

/-- The end-to-end contact probability: the weight of the conformations in which the two
termini are in the same state. -/
noncomputable def contactProb (p : Bool → Unit → Bool → ℝ) : ℝ :=
  ∑ x, ∑ y, ∑ z, if x = z then p x y z else 0

/-- The truth has the contact always formed; the modular model reports it half the time.
A modular construction can be exact on every fragment observable and still be wrong by a
factor of two on the observable the experiment is about. -/
theorem longRange_contact : contactProb longRange = 1 ∧ contactProb (glue longRange) = 1/2 := by
  constructor
  · simp [contactProb, longRange]
  · simp [contactProb, longRange_glue]
    norm_num

/-- The seam information of the long-range contact is one bit. -/
theorem longRange_cmi : cmi longRange = Real.log 2 := by
  rw [cmi]
  simp only [longRange_margXY, longRange_margYZ, longRange_margY, longRange, Fintype.sum_bool]
  norm_num
  ring

/-- **Every modular model of the long-range contact is at least one bit from the truth.**
The floor holds for any strictly positive ensemble that is conditionally independent across
the seam — that is, for every ensemble any modular construction can produce. -/
theorem longRange_no_modular_model_is_close {q : Bool → Unit → Bool → ℝ}
    (hq : ∀ x y z, 0 < q x y z) (hqs : ∑ x, ∑ y, ∑ z, q x y z = 1) (hci : CondIndep q) :
    Real.log 2 ≤ ∑ x, ∑ y, ∑ z, longRange x y z * Real.log (longRange x y z / q x y z) := by
  have := no_modular_model_beats_seam_information (p := longRange) (q := q)
    longRange_nonneg hq longRange_sum hqs hci
  rwa [longRange_cmi] at this

end RequestProject.Modular
