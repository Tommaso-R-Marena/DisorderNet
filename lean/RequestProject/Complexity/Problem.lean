/-
# Decision problems and size-bounded many-one reductions

This is the framework in which the NP-hardness of the recalibration optimum is made fully
machine-checked.  Lean has no cost model for its own functions, so "polynomial time" cannot be
formalised by looking at a Lean program.  What *can* be formalised — and is what a Karp reduction
actually supplies — is:

* an **explicit total function** from instances of one problem to instances of another,
* a **machine-checked correctness proof**: yes-instances map to yes-instances and back,
* a **machine-checked polynomial bound on the size of the produced instance**.

`Reduction A B` bundles exactly these three, and `Reduction.comp` shows they compose.  Every
reduction in this development is a concrete, computable Lean function, so the missing ingredient
relative to the textbook notion is only the running-time bound, which no Lean statement can talk
about without a machine model.  Nothing else in the chain is left informal.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.Complexity

/-- A decision problem: a type of instances, a size measure, and the yes-predicate. -/
structure Problem where
  /-- The type of instances. -/
  Inst : Type
  /-- The size of an instance (the length of its encoding, measured combinatorially). -/
  size : Inst → ℕ
  /-- The yes-instances. -/
  Yes : Inst → Prop

/-- A **size-bounded many-one reduction**: an explicit map on instances that preserves and
reflects the answer, together with a polynomial bound on the size of the output. -/
structure Reduction (A B : Problem) where
  /-- The instance map. -/
  map : A.Inst → B.Inst
  /-- Correctness of the reduction. -/
  correct : ∀ x, A.Yes x ↔ B.Yes (map x)
  /-- Degree of the size bound. -/
  deg : ℕ
  /-- Constant of the size bound. -/
  const : ℕ
  /-- The produced instance is polynomially larger than the input. -/
  size_le : ∀ x, B.size (map x) ≤ const * (A.size x + 1) ^ deg

namespace Reduction

/-- The identity reduction. -/
def refl (A : Problem) : Reduction A A where
  map := id
  correct := fun _ => Iff.rfl
  deg := 1
  const := 1
  size_le := fun x => by simp

/-- Reductions compose: this is what makes a chain of reductions a proof of hardness. -/
def comp {A B C : Problem} (f : Reduction A B) (g : Reduction B C) : Reduction A C where
  map := g.map ∘ f.map
  correct := fun x => (f.correct x).trans (g.correct _)
  deg := f.deg * g.deg
  const := g.const * (f.const + 1) ^ g.deg
  size_le := by
    intro x
    have h1 : C.size (g.map (f.map x)) ≤ g.const * (B.size (f.map x) + 1) ^ g.deg :=
      g.size_le _
    have h2 : B.size (f.map x) + 1 ≤ (f.const + 1) * (A.size x + 1) ^ f.deg := by
      have hfx := f.size_le x
      have hp : 1 ≤ (A.size x + 1) ^ f.deg := Nat.one_le_pow _ _ (Nat.succ_pos _)
      calc B.size (f.map x) + 1 ≤ f.const * (A.size x + 1) ^ f.deg + 1 := by omega
        _ ≤ f.const * (A.size x + 1) ^ f.deg + (A.size x + 1) ^ f.deg := by omega
        _ = (f.const + 1) * (A.size x + 1) ^ f.deg := by ring
    calc C.size (g.map (f.map x)) ≤ g.const * (B.size (f.map x) + 1) ^ g.deg := h1
      _ ≤ g.const * ((f.const + 1) * (A.size x + 1) ^ f.deg) ^ g.deg :=
          Nat.mul_le_mul_left _ (Nat.pow_le_pow_left h2 _)
      _ = g.const * (f.const + 1) ^ g.deg * (A.size x + 1) ^ (f.deg * g.deg) := by
          rw [Nat.mul_pow, ← pow_mul]; ring

end Reduction

/-- Hardness of `B` for a class `C` of problems: every problem in the class reduces to `B`. -/
def Hard (C : Problem → Prop) (B : Problem) : Prop :=
  ∀ A, C A → Nonempty (Reduction A B)

/-- Hardness transfers along a reduction. -/
theorem Hard.trans {C : Problem → Prop} {B B' : Problem} (h : Hard C B)
    (f : Reduction B B') : Hard C B' :=
  fun A hA => ⟨((h A hA).some).comp f⟩

end IDR.Complexity
