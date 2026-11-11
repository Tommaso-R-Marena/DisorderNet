/-
# Correctness of the Tseitin transformation

`tseitin_correct`: a circuit-satisfiability instance is a yes-instance exactly when its Tseitin
CNF is satisfiable; `tseitin_size_le`: the CNF is linear in the circuit.  Together they give the
second link of the chain, `tseitinReduction`, and hence NP-hardness of CNF satisfiability.
-/
import RequestProject.Complexity.CNF

set_option autoImplicit false

namespace IDR.Complexity

namespace Circuit

/-- Evaluating a longer program extends the list of gate values. -/
theorem gateVals_append_exists (x : List Bool) (l₁ l₂ : List Instr) :
    ∃ r, gateVals x (l₁ ++ l₂) = gateVals x l₁ ++ r := by
  induction l₂ using List.reverseRecOn with
  | nil => exact ⟨[], by simp⟩
  | append_singleton l ins ih =>
      obtain ⟨r, hr⟩ := ih
      refine ⟨r ++ [evalInstr x (gateVals x (l₁ ++ l)) ins], ?_⟩
      rw [← List.append_assoc, gateVals_append_one, hr, List.append_assoc]

/-- Gate values already computed do not change when the program is extended. -/
theorem getD_gateVals_take (x : List Bool) (l : List Instr) {a i : ℕ} (h : a < i) :
    (gateVals x l).getD a false = (gateVals x (l.take i)).getD a false := by
  obtain ⟨r, hr⟩ := gateVals_append_exists x (l.take i) (l.drop i)
  rw [List.take_append_drop] at hr
  rw [hr]
  by_cases hlen : a < (gateVals x (l.take i)).length
  · exact List.getD_append _ _ _ _ hlen
  · have h1 : (gateVals x (l.take i)).length = (l.take i).length := length_gateVals _ _
    have h2 : (l.take i).length ≤ a := by omega
    have h3 : l.length ≤ a := by
      have := List.length_take (l := l) (i := i)
      omega
    rw [List.getD_append_right _ _ _ _ (by omega)]
    have h4 : (gateVals x l).getD a false = false := by
      apply List.getD_eq_default
      rw [length_gateVals]; omega
    have h5 : r.getD (a - (gateVals x (l.take i)).length) false = false := by
      have hrlen : (gateVals x (l.take i)).length + r.length = l.length := by
        have := congrArg List.length hr
        simp [length_gateVals] at this ⊢
        omega
      apply List.getD_eq_default
      omega
    rw [h5]
    have h6 : (gateVals x (l.take i)).getD a false = false := by
      apply List.getD_eq_default
      rw [h1]; omega
    exact h6.symm

/-- The value of gate `i` is the value of its instruction on the gates before it. -/
theorem getD_gateVals_eq (x : List Bool) (l : List Instr) {i : ℕ} {ins : Instr}
    (h : l[i]? = some ins) :
    (gateVals x l).getD i false = evalInstr x (gateVals x (l.take i)) ins := by
  have hi : i < l.length := by
    by_contra hcon
    rw [List.getElem?_eq_none (by omega)] at h
    exact absurd h.symm (by simp)
  have hsplit : l.take (i + 1) = l.take i ++ [ins] := by
    have : l[i] = ins := by
      have := List.getElem?_eq_getElem hi
      rw [h] at this
      exact (Option.some_inj.mp this).symm
    rw [List.take_add_one, List.getElem?_eq_getElem hi, this]
    rfl
  obtain ⟨r, hr⟩ := gateVals_append_exists x (l.take (i + 1)) (l.drop (i + 1))
  rw [List.take_append_drop] at hr
  rw [hr, hsplit, gateVals_append_one]
  have hlen : (gateVals x (l.take i)).length = i := by
    rw [length_gateVals, List.length_take]; omega
  rw [List.append_assoc]
  rw [List.getD_append_right _ _ _ _ (by omega), hlen]
  simp

end Circuit

namespace Tseitin

open Circuit

/-- The assignment read off a satisfying input string: input variables get the input bits, gate
variables get the computed gate values. -/
def satOf (I : SatInst) (w : List Bool) : ℕ → Bool := fun v =>
  if v % 2 = 0 then w.getD (v / 2) false
  else (gateVals w I.C.instrs).getD (v / 2) false

@[simp] theorem satOf_inpVar (I : SatInst) (w : List Bool) (j : ℕ) :
    satOf I w (inpVar j) = w.getD j false := by
  simp [satOf, inpVar]

@[simp] theorem satOf_gateVar (I : SatInst) (w : List Bool) (i : ℕ) :
    satOf I w (gateVar i) = (gateVals w I.C.instrs).getD i false := by
  have h : (2 * i + 1) % 2 = 1 := by omega
  simp [satOf, gateVar, h, Nat.mul_add_div]

/-- Membership in the Tseitin CNF. -/
theorem mem_tseitin {I : SatInst} {c : Clause} :
    c ∈ tseitin I ↔
      c = [⟨gateVar I.C.out, true⟩] ∨
        ∃ i ins, I.C.instrs[i]? = some ins ∧ c ∈ tseitinClauses I.arity i ins := by
  simp only [tseitin, List.mem_append, List.mem_flatMap, List.mem_singleton]
  constructor
  · rintro (⟨p, hp, hc⟩ | rfl)
    · exact Or.inr ⟨p.2, p.1, List.mem_zipIdx_iff_getElem?.mp hp, hc⟩
    · exact Or.inl rfl
  · rintro (rfl | ⟨i, ins, hi, hc⟩)
    · exact Or.inr rfl
    · exact Or.inl ⟨(ins, i), List.mem_zipIdx_iff_getElem?.mpr hi, hc⟩

/-- The value an assignment gives to the instruction of a gate. -/
def instrVal (arity : ℕ) (σ : ℕ → Bool) : Instr → Bool
  | .cnst b => b
  | .inp j => if j < arity then σ (inpVar j) else false
  | .neg a => !(σ (gateVar a))
  | .conj a b => σ (gateVar a) && σ (gateVar b)
  | .disj a b => σ (gateVar a) || σ (gateVar b)

/-- **The defining clauses of a gate say exactly that the gate variable carries its value.** -/
theorem tseitinClauses_holds_iff (arity i : ℕ) (ins : Instr) (σ : ℕ → Bool) :
    (tseitinClauses arity i ins).holds σ ↔ σ (gateVar i) = instrVal arity σ ins := by
  cases ins with
  | cnst b => simp [tseitinClauses, Cnf.holds, Clause.holds, Lit.holds, instrVal]
  | inp j =>
      by_cases hj : j < arity
      · simp only [tseitinClauses, Cnf.holds, Clause.holds, Lit.holds, instrVal, hj,
          if_true, List.mem_cons, List.not_mem_nil, or_false, exists_eq_or_imp,
          exists_eq_left, forall_eq_or_imp, forall_eq]
        cases σ (gateVar i) <;> cases σ (inpVar j) <;> simp
      · simp [tseitinClauses, Cnf.holds, Clause.holds, Lit.holds, instrVal, hj]
  | neg a =>
      simp only [tseitinClauses, Cnf.holds, Clause.holds, Lit.holds, instrVal,
        List.mem_cons, List.not_mem_nil, or_false, exists_eq_or_imp, exists_eq_left,
        forall_eq_or_imp, forall_eq]
      cases σ (gateVar i) <;> cases σ (gateVar a) <;> simp
  | conj a b =>
      simp only [tseitinClauses, Cnf.holds, Clause.holds, Lit.holds, instrVal,
        List.mem_cons, List.not_mem_nil, or_false, exists_eq_or_imp, exists_eq_left,
        forall_eq_or_imp, forall_eq]
      cases σ (gateVar i) <;> cases σ (gateVar a) <;> cases σ (gateVar b) <;> simp
  | disj a b =>
      simp only [tseitinClauses, Cnf.holds, Clause.holds, Lit.holds, instrVal,
        List.mem_cons, List.not_mem_nil, or_false, exists_eq_or_imp, exists_eq_left,
        forall_eq_or_imp, forall_eq]
      cases σ (gateVar i) <;> cases σ (gateVar a) <;> cases σ (gateVar b) <;> simp

/-- In a well-formed circuit the value of a gate is its instruction evaluated at the gate values
of the whole circuit. -/
theorem gateVal_spec (I : SatInst) (w : List Bool) {i : ℕ} {ins : Instr}
    (h : I.C.instrs[i]? = some ins) :
    (gateVals w I.C.instrs).getD i false = evalInstr w (gateVals w I.C.instrs) ins := by
  have hwf := I.wf.backwards i ins h
  rw [getD_gateVals_eq w I.C.instrs h]
  cases ins with
  | cnst b => rfl
  | inp j => rfl
  | neg a =>
      simp only [evalInstr]
      rw [getD_gateVals_take w I.C.instrs hwf]
  | conj a b =>
      simp only [evalInstr]
      rw [getD_gateVals_take w I.C.instrs hwf.1, getD_gateVals_take w I.C.instrs hwf.2]
  | disj a b =>
      simp only [evalInstr]
      rw [getD_gateVals_take w I.C.instrs hwf.1, getD_gateVals_take w I.C.instrs hwf.2]

/-- **Soundness**: a satisfying input yields a satisfying assignment of the CNF. -/
theorem holds_satOf {I : SatInst} {w : List Bool} (hw : w.length = I.arity)
    (h : I.C.eval w = true) : (tseitin I).holds (satOf I w) := by
  intro c hc
  rcases mem_tseitin.mp hc with rfl | ⟨i, ins, hi, hcm⟩
  · exact ⟨⟨gateVar I.C.out, true⟩, by simp, by simpa [Lit.holds, Circuit.eval] using h⟩
  · refine (tseitinClauses_holds_iff I.arity i ins (satOf I w)).mpr ?_ c hcm
    rw [satOf_gateVar, gateVal_spec I w hi]
    cases ins with
    | cnst b => rfl
    | inp j =>
        simp only [evalInstr, instrVal]
        by_cases hj : j < I.arity
        · rw [if_pos hj, satOf_inpVar]
        · rw [if_neg hj]
          apply List.getD_eq_default
          omega
    | neg a => simp only [evalInstr, instrVal, satOf_gateVar]
    | conj a b => simp only [evalInstr, instrVal, satOf_gateVar]
    | disj a b => simp only [evalInstr, instrVal, satOf_gateVar]

/-- **Completeness**: a satisfying assignment of the CNF yields a satisfying input. -/
theorem exists_input_of_holds {I : SatInst} {σ : ℕ → Bool} (h : (tseitin I).holds σ) :
    ∃ w : List Bool, w.length = I.arity ∧ I.C.eval w = true := by
  classical
  set w : List Bool := (List.range I.arity).map (fun j => σ (inpVar j)) with hwdef
  have hwlen : w.length = I.arity := by simp [hwdef]
  have hwget : ∀ j, w.getD j false = if j < I.arity then σ (inpVar j) else false := by
    intro j
    by_cases hj : j < I.arity
    · rw [if_pos hj]
      have hlen : j < w.length := by omega
      rw [List.getD_eq_getElem _ _ hlen]
      simp [hwdef]
    · rw [if_neg hj]
      exact List.getD_eq_default _ _ (by omega)
  have key : ∀ i, i < I.C.instrs.length → (gateVals w I.C.instrs).getD i false = σ (gateVar i) := by
    intro i
    induction i using Nat.strong_induction_on with
    | _ i ih =>
      intro hi
      obtain ⟨ins, hins⟩ : ∃ ins, I.C.instrs[i]? = some ins := ⟨_, List.getElem?_eq_getElem hi⟩
      have hwf := I.wf.backwards i ins hins
      have hcl : (tseitinClauses I.arity i ins).holds σ := fun c hc =>
        h c (mem_tseitin.mpr (Or.inr ⟨i, ins, hins, hc⟩))
      have hgi : σ (gateVar i) = instrVal I.arity σ ins :=
        (tseitinClauses_holds_iff _ _ _ _).mp hcl
      rw [gateVal_spec I w hins, hgi]
      cases ins with
      | cnst b => rfl
      | inp j => simpa only [evalInstr, instrVal] using hwget j
      | neg a =>
          have ha : a < i := hwf
          simp only [evalInstr, instrVal, ih a ha (by omega)]
      | conj a b =>
          have ha : a < i := hwf.1
          have hb : b < i := hwf.2
          simp only [evalInstr, instrVal, ih a ha (by omega), ih b hb (by omega)]
      | disj a b =>
          have ha : a < i := hwf.1
          have hb : b < i := hwf.2
          simp only [evalInstr, instrVal, ih a ha (by omega), ih b hb (by omega)]
  refine ⟨w, hwlen, ?_⟩
  have hout : Clause.holds σ [(⟨gateVar I.C.out, true⟩ : Lit)] :=
    h _ (mem_tseitin.mpr (Or.inl rfl))
  obtain ⟨l, hl, hlh⟩ := hout
  simp only [List.mem_singleton] at hl
  subst hl
  have : σ (gateVar I.C.out) = true := hlh
  rw [Circuit.eval, key I.C.out I.wf.out_lt, this]

/-- **The Tseitin CNF is equisatisfiable with the circuit.** -/
theorem tseitin_correct (I : SatInst) :
    (∃ w : List Bool, w.length = I.arity ∧ I.C.eval w = true) ↔
      ∃ σ : ℕ → Bool, (tseitin I).holds σ := by
  constructor
  · rintro ⟨w, hw, h⟩
    exact ⟨satOf I w, holds_satOf hw h⟩
  · rintro ⟨σ, h⟩
    exact exists_input_of_holds h

/-- Each gate contributes at most ten to the size of the CNF. -/
theorem tseitinClauses_size_le (arity i : ℕ) (ins : Instr) :
    (tseitinClauses arity i ins).size ≤ 10 := by
  cases ins with
  | cnst b => simp [tseitinClauses, Cnf.size]
  | inp j => by_cases hj : j < arity <;> simp [tseitinClauses, Cnf.size, hj]
  | neg a => simp [tseitinClauses, Cnf.size]
  | conj a b => simp [tseitinClauses, Cnf.size]
  | disj a b => simp [tseitinClauses, Cnf.size]

theorem flatMap_size_le (arity : ℕ) (l : List (Instr × ℕ)) :
    Cnf.size (l.flatMap (fun p => tseitinClauses arity p.2 p.1)) ≤ 10 * l.length := by
  induction l with
  | nil => simp [Cnf.size]
  | cons a l ih =>
      rw [List.flatMap_cons, Cnf.size_append]
      have := tseitinClauses_size_le arity a.2 a.1
      simp only [List.length_cons]
      omega

/-- **The Tseitin CNF is linear in the circuit.** -/
theorem tseitin_size_le (I : SatInst) : (tseitin I).size ≤ 12 * (I.size + 1) := by
  rw [tseitin, Cnf.size_append]
  have h1 := flatMap_size_le I.arity I.C.instrs.zipIdx
  have h2 : Cnf.size [[(⟨gateVar I.C.out, true⟩ : Lit)]] = 2 := by simp [Cnf.size]
  have h3 : I.C.instrs.zipIdx.length = I.C.size := by simp [Circuit.size]
  have h4 : I.size = I.C.size + I.arity := rfl
  omega

end Tseitin

/-- **Circuit satisfiability reduces to CNF satisfiability.** -/
def tseitinReduction : Reduction circuitSat cnfSat where
  map := tseitin
  correct := Tseitin.tseitin_correct
  deg := 1
  const := 12
  size_le := by
    intro I
    have := Tseitin.tseitin_size_le I
    simpa [pow_one] using this

/-- **CNF satisfiability is NP-hard.** -/
theorem cnfSat_hard : Hard NPProblem cnfSat :=
  circuitSat_hard.trans tseitinReduction

end IDR.Complexity
