/-
# CNF formulas and the Tseitin transformation

`Cnf` is a list of clauses, a clause a list of literals, a literal a variable index with a
polarity.  `cnfSat` is the satisfiability problem for these.

`tseitin` turns a circuit-satisfiability instance into an equisatisfiable CNF of linear size, by
giving every gate its own variable and writing down the defining equivalence of each gate as
clauses.  Input variable `j` becomes variable `2 * j`, gate `i` becomes variable `2 * i + 1`.
Inputs beyond the declared arity read `false` in the circuit semantics, and are translated to the
constant `false`, which is what keeps the two satisfiability questions equivalent.
-/
import RequestProject.Complexity.NP

set_option autoImplicit false

namespace IDR.Complexity

/-- A literal: a variable index together with a polarity (`true` = the variable itself). -/
structure Lit where
  /-- The variable index. -/
  var : ℕ
  /-- The polarity. -/
  pol : Bool
deriving DecidableEq, Repr

/-- A clause is a disjunction of literals. -/
abbrev Clause : Type := List Lit

/-- A CNF formula is a conjunction of clauses. -/
abbrev Cnf : Type := List Clause

/-- Whether a literal is satisfied by an assignment. -/
def Lit.holds (σ : ℕ → Bool) (l : Lit) : Prop := σ l.var = l.pol

/-- Whether a clause is satisfied by an assignment. -/
def Clause.holds (σ : ℕ → Bool) (c : Clause) : Prop := ∃ l ∈ c, Lit.holds σ l

/-- Whether a CNF formula is satisfied by an assignment. -/
def Cnf.holds (σ : ℕ → Bool) (F : Cnf) : Prop := ∀ c ∈ F, Clause.holds σ c

/-- The size of a CNF formula: number of clauses plus number of literal occurrences. -/
def Cnf.size (F : Cnf) : ℕ := F.length + (F.map List.length).sum

@[simp] theorem Cnf.size_append (F G : Cnf) : (F ++ G).size = F.size + G.size := by
  simp [Cnf.size]; omega

@[simp] theorem Cnf.holds_append {σ : ℕ → Bool} {F G : Cnf} :
    (F ++ G).holds σ ↔ F.holds σ ∧ G.holds σ := by
  constructor
  · intro h; exact ⟨fun c hc => h c (by simp [hc]), fun c hc => h c (by simp [hc])⟩
  · rintro ⟨h1, h2⟩ c hc
    rcases List.mem_append.mp hc with h | h
    · exact h1 c h
    · exact h2 c h

/-- **CNF satisfiability.** -/
def cnfSat : Problem where
  Inst := Cnf
  size := Cnf.size
  Yes := fun F => ∃ σ : ℕ → Bool, F.holds σ

/-! ## The Tseitin transformation -/

/-- The variable of input `j`. -/
def inpVar (j : ℕ) : ℕ := 2 * j

/-- The variable of gate `i`. -/
def gateVar (i : ℕ) : ℕ := 2 * i + 1

theorem inpVar_ne_gateVar (j i : ℕ) : inpVar j ≠ gateVar i := by
  simp [inpVar, gateVar]; omega

theorem inpVar_injective : Function.Injective inpVar := by
  intro a b h; simpa [inpVar] using h

theorem gateVar_injective : Function.Injective gateVar := by
  intro a b h; simpa [gateVar] using h

/-- The clauses defining gate `i`, whose instruction is `ins`, for a circuit of arity `arity`. -/
def tseitinClauses (arity i : ℕ) : Instr → Cnf
  | .cnst b => [[⟨gateVar i, b⟩]]
  | .inp j =>
      if j < arity then
        [[⟨gateVar i, false⟩, ⟨inpVar j, true⟩], [⟨gateVar i, true⟩, ⟨inpVar j, false⟩]]
      else [[⟨gateVar i, false⟩]]
  | .neg a => [[⟨gateVar i, false⟩, ⟨gateVar a, false⟩], [⟨gateVar i, true⟩, ⟨gateVar a, true⟩]]
  | .conj a b =>
      [[⟨gateVar i, false⟩, ⟨gateVar a, true⟩], [⟨gateVar i, false⟩, ⟨gateVar b, true⟩],
        [⟨gateVar i, true⟩, ⟨gateVar a, false⟩, ⟨gateVar b, false⟩]]
  | .disj a b =>
      [[⟨gateVar i, true⟩, ⟨gateVar a, false⟩], [⟨gateVar i, true⟩, ⟨gateVar b, false⟩],
        [⟨gateVar i, false⟩, ⟨gateVar a, true⟩, ⟨gateVar b, true⟩]]

/-- The Tseitin encoding of a circuit satisfiability instance. -/
def tseitin (I : SatInst) : Cnf :=
  (I.C.instrs.zipIdx.flatMap (fun p => tseitinClauses I.arity p.2 p.1))
    ++ [[⟨gateVar I.C.out, true⟩]]

end IDR.Complexity
