/-
# NP, spelled out, and the hardness of circuit satisfiability

`Verifier L` is the certificate definition of NP made concrete: a witness length `wlen n`, a
verifier circuit `V n` for each input length, both of polynomial size, such that `x ∈ L` iff some
witness of the prescribed length makes the circuit accept `x ++ w`.  This is the *non-uniform*
version of NP: the verifier is a circuit family rather than a machine, because Lean has no cost
model with which to say "polynomial-time machine".  Every language in the usual class NP has such
a family (a polynomial-time verifier is simulated by polynomial-size circuits), so hardness for
the class defined here is at least as strong a statement as NP-hardness — and here the whole chain
below is machine-checked rather than cited.

`np_reduces_to_circuitSat` is the first link: hardwiring the input into the verifier circuit turns
membership into satisfiability of a single circuit.
-/
import RequestProject.Complexity.Circuit

set_option autoImplicit false

namespace IDR.Complexity

/-- A language: a set of bit strings. -/
def Lang : Type := List Bool → Prop

/-- The decision problem attached to a language; the size of an instance is its length. -/
def langProblem (L : Lang) : Problem where
  Inst := List Bool
  size := List.length
  Yes := L

/-- A **polynomial-size nondeterministic verifier** for `L`: the certificate definition of NP,
with the verifier given by a circuit family. -/
structure Verifier (L : Lang) where
  /-- The length of the certificate for inputs of length `n`. -/
  wlen : ℕ → ℕ
  /-- The verifier circuit for inputs of length `n`. -/
  V : ℕ → Circuit
  /-- Verifier circuits are well formed. -/
  V_wf : ∀ n, (V n).WF
  /-- Constant in the polynomial bound on the certificate length. -/
  wc : ℕ
  /-- Degree in the polynomial bound on the certificate length. -/
  wd : ℕ
  /-- Certificates are polynomially long. -/
  wlen_le : ∀ n, wlen n ≤ wc * (n + 1) ^ wd
  /-- Constant in the polynomial bound on the verifier size. -/
  vc : ℕ
  /-- Degree in the polynomial bound on the verifier size. -/
  vd : ℕ
  /-- Verifiers are polynomially large. -/
  V_size_le : ∀ n, (V n).size ≤ vc * (n + 1) ^ vd
  /-- Correctness of the verifier. -/
  correct : ∀ x : List Bool,
    L x ↔ ∃ w : List Bool, w.length = wlen x.length ∧ (V x.length).eval (x ++ w) = true

/-- `L` is in NP: it has a polynomial-size nondeterministic verifier. -/
def InNP (L : Lang) : Prop := Nonempty (Verifier L)

/-- The class of decision problems coming from languages in NP. -/
def NPProblem (A : Problem) : Prop := ∃ L : Lang, InNP L ∧ A = langProblem L

/-! ## Circuit satisfiability -/

/-- An instance of circuit satisfiability: a well-formed circuit and the number of inputs. -/
structure SatInst where
  /-- The circuit. -/
  C : Circuit
  /-- The number of input bits. -/
  arity : ℕ
  /-- Well-formedness of the circuit. -/
  wf : C.WF

/-- Size of a circuit satisfiability instance. -/
def SatInst.size (I : SatInst) : ℕ := I.C.size + I.arity

/-- **Circuit satisfiability.** -/
def circuitSat : Problem where
  Inst := SatInst
  size := SatInst.size
  Yes := fun I => ∃ w : List Bool, w.length = I.arity ∧ I.C.eval w = true

/-- **Every language in NP reduces to circuit satisfiability**, by hardwiring the input into the
verifier circuit. -/
def npReduction {L : Lang} (v : Verifier L) : Reduction (langProblem L) circuitSat where
  map := fun x => ⟨(v.V x.length).hardwire x, v.wlen x.length, Circuit.wf_hardwire (v.V_wf _) x⟩
  correct := by
    intro x
    show L x ↔ ∃ w : List Bool,
      w.length = v.wlen x.length ∧ ((v.V x.length).hardwire x).eval w = true
    rw [v.correct x]
    constructor
    · rintro ⟨w, hw, hacc⟩
      exact ⟨w, hw, by rw [Circuit.eval_hardwire]; exact hacc⟩
    · rintro ⟨w, hw, hacc⟩
      refine ⟨w, hw, ?_⟩
      rw [Circuit.eval_hardwire] at hacc
      exact hacc
  deg := max v.vd v.wd
  const := v.vc + v.wc
  size_le := by
    intro x
    have h1 : (v.V x.length).size ≤ v.vc * (x.length + 1) ^ max v.vd v.wd := by
      refine le_trans (v.V_size_le _) (Nat.mul_le_mul_left _ ?_)
      exact Nat.pow_le_pow_right (Nat.succ_pos _) (le_max_left _ _)
    have h2 : v.wlen x.length ≤ v.wc * (x.length + 1) ^ max v.vd v.wd := by
      refine le_trans (v.wlen_le _) (Nat.mul_le_mul_left _ ?_)
      exact Nat.pow_le_pow_right (Nat.succ_pos _) (le_max_right _ _)
    show (Circuit.hardwire (v.V x.length) x).size + v.wlen x.length ≤ _
    rw [Circuit.hardwire_size]
    calc (v.V x.length).size + v.wlen x.length
        ≤ v.vc * (x.length + 1) ^ max v.vd v.wd + v.wc * (x.length + 1) ^ max v.vd v.wd :=
          Nat.add_le_add h1 h2
      _ = (v.vc + v.wc) * (x.length + 1) ^ max v.vd v.wd := by ring

/-- **Circuit satisfiability is NP-hard.** -/
theorem circuitSat_hard : Hard NPProblem circuitSat := by
  rintro A ⟨L, ⟨v⟩, rfl⟩
  exact ⟨npReduction v⟩

end IDR.Complexity
