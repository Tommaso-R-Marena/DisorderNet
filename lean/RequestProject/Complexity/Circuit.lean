/-
# Boolean circuits as straight-line programs

A circuit is a list of instructions, each of which is a constant, an input, a negation, a
conjunction or a disjunction of *earlier* gates, plus the index of the output gate.  Evaluation is
total: a reference to a gate that has not been computed yet reads `false`, so no dependent types
are needed; well-formed circuits (`Circuit.WF`) are the ones all of whose references point
backwards, and those are the only ones the development builds.

The one operation needed later is `hardwire`, which fixes the first `x.length` inputs to the bits
of `x` and renumbers the rest: `(hardwire C x).eval w = C.eval (x ++ w)`.  This is what turns an
NP verifier into a satisfiability question about a single circuit.
-/
import RequestProject.Complexity.Problem

set_option autoImplicit false

namespace IDR.Complexity

/-- A straight-line instruction: constant, input bit, negation, conjunction, disjunction.
The arguments of `not`, `and`, `or` are indices of earlier gates. -/
inductive Instr where
  | cnst (b : Bool) : Instr
  | inp (j : ℕ) : Instr
  | neg (a : ℕ) : Instr
  | conj (a b : ℕ) : Instr
  | disj (a b : ℕ) : Instr
deriving DecidableEq, Repr

/-- A Boolean circuit: a straight-line program together with the index of its output gate. -/
structure Circuit where
  /-- The gates, in evaluation order. -/
  instrs : List Instr
  /-- The index of the output gate. -/
  out : ℕ
deriving DecidableEq, Repr

namespace Circuit

/-- Number of gates. -/
def size (C : Circuit) : ℕ := C.instrs.length

/-- The value of one instruction, given the input string and the values of the earlier gates. -/
def evalInstr (x : List Bool) (vals : List Bool) : Instr → Bool
  | .cnst b => b
  | .inp j => x.getD j false
  | .neg a => !(vals.getD a false)
  | .conj a b => (vals.getD a false) && (vals.getD b false)
  | .disj a b => (vals.getD a false) || (vals.getD b false)

/-- The list of gate values of a straight-line program. -/
def gateVals (x : List Bool) (l : List Instr) : List Bool :=
  l.foldl (fun vals ins => vals ++ [evalInstr x vals ins]) []

@[simp] theorem gateVals_nil (x : List Bool) : gateVals x [] = [] := rfl

theorem gateVals_append_one (x : List Bool) (l : List Instr) (ins : Instr) :
    gateVals x (l ++ [ins]) = gateVals x l ++ [evalInstr x (gateVals x l) ins] := by
  simp [gateVals]

@[simp] theorem length_gateVals (x : List Bool) (l : List Instr) :
    (gateVals x l).length = l.length := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l ins ih => simp [gateVals_append_one, ih]

/-- Evaluation of a circuit on an input string. -/
def eval (C : Circuit) (x : List Bool) : Bool := (gateVals x C.instrs).getD C.out false

/-- Well-formedness of an instruction sitting at position `i`: all gate references point to
strictly earlier gates. -/
def _root_.IDR.Complexity.Instr.WF (i : ℕ) : Instr → Prop
  | .cnst _ => True
  | .inp _ => True
  | .neg a => a < i
  | .conj a b => a < i ∧ b < i
  | .disj a b => a < i ∧ b < i

/-- A circuit is well formed when every reference points backwards and the output gate exists. -/
structure WF (C : Circuit) : Prop where
  /-- Every instruction only refers to earlier gates. -/
  backwards : ∀ i ins, C.instrs[i]? = some ins → Instr.WF i ins
  /-- The output index is a gate of the circuit. -/
  out_lt : C.out < C.instrs.length

/-! ### Hardwiring a prefix of the inputs -/

/-- Replace input `j` by the constant `x[j]` when `j < x.length`, and renumber the remaining
inputs. -/
def hardwireInstr (x : List Bool) : Instr → Instr
  | .inp j => if j < x.length then .cnst (x.getD j false) else .inp (j - x.length)
  | ins => ins

/-- The circuit obtained by fixing the first `x.length` inputs to the bits of `x`. -/
def hardwire (C : Circuit) (x : List Bool) : Circuit :=
  ⟨C.instrs.map (hardwireInstr x), C.out⟩

@[simp] theorem hardwire_size (C : Circuit) (x : List Bool) : (C.hardwire x).size = C.size := by
  simp [hardwire, size]

theorem gateVals_hardwire (x w : List Bool) (l : List Instr) :
    gateVals w (l.map (hardwireInstr x)) = gateVals (x ++ w) l := by
  induction l using List.reverseRecOn with
  | nil => simp
  | append_singleton l ins ih =>
      rw [List.map_append]
      simp only [List.map_cons, List.map_nil]
      rw [gateVals_append_one, gateVals_append_one, ih]
      congr 1
      cases ins with
      | cnst b => rfl
      | inp j =>
          simp only [hardwireInstr, evalInstr]
          by_cases hj : j < x.length
          · simp only [hj, if_true]
            congr 1
            exact (List.getD_append x w false j hj).symm
          · simp only [hj, if_false]
            congr 1
            exact (List.getD_append_right x w false j (by omega)).symm
      | neg a => rfl
      | conj a b => rfl
      | disj a b => rfl

/-- **Hardwiring is correct.** -/
theorem eval_hardwire (C : Circuit) (x w : List Bool) :
    (C.hardwire x).eval w = C.eval (x ++ w) := by
  simp [eval, hardwire, gateVals_hardwire]

theorem wf_hardwire {C : Circuit} (hC : C.WF) (x : List Bool) : (C.hardwire x).WF := by
  refine ⟨?_, ?_⟩
  · intro i ins hins
    simp only [hardwire, List.getElem?_map, Option.map_eq_some_iff] at hins
    obtain ⟨ins', hins', rfl⟩ := hins
    have := hC.backwards i ins' hins'
    cases ins' with
    | cnst b => trivial
    | inp j => simp only [hardwireInstr]; split <;> trivial
    | neg a => exact this
    | conj a b => exact this
    | disj a b => exact this
  · simpa [hardwire] using hC.out_lt

end Circuit

end IDR.Complexity
