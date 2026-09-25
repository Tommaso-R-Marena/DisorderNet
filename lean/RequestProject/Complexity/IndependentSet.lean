/-
# Independent set, and the reduction from CNF satisfiability

The graph has one vertex per *literal occurrence*; two occurrences are joined when they lie in the
same clause or when they are a variable and its negation.  An independent set of size `#clauses`
picks one literal from each clause with no variable chosen both positively and negatively — which
is exactly a satisfying assignment.  This is the classical Karp reduction, here for CNF formulas
of arbitrary clause width, so no separate 3-CNF step is needed.
-/
import RequestProject.Complexity.Tseitin

set_option autoImplicit false

namespace IDR.Complexity

/-- An instance of the independent set problem: `n` vertices `0, …, n-1`, a list of edges (given
as ordered pairs; endpoints outside the range and self-loops are allowed and simply forbid the
vertex), and a target size. -/
structure GraphInst where
  /-- The number of vertices. -/
  n : ℕ
  /-- The edges. -/
  edges : List (ℕ × ℕ)
  /-- The target size of the independent set. -/
  k : ℕ

/-- A set of vertices is independent when it is inside the vertex range and contains no edge. -/
def GraphInst.Indep (I : GraphInst) (S : Finset ℕ) : Prop :=
  S ⊆ Finset.range I.n ∧ ∀ e ∈ I.edges, ¬ (e.1 ∈ S ∧ e.2 ∈ S)

/-- The size of a graph instance. -/
def GraphInst.size (I : GraphInst) : ℕ := I.n + I.edges.length + I.k

/-- **The independent set problem.** -/
def independentSet : Problem where
  Inst := GraphInst
  size := GraphInst.size
  Yes := fun I => ∃ S : Finset ℕ, I.Indep S ∧ I.k ≤ S.card

/-! ## The reduction from CNF satisfiability -/

namespace SatIndep

/-- The literal occurrences of a formula: `(clause index, literal)`, in order. -/
def lits (F : Cnf) : List (ℕ × Lit) :=
  F.zipIdx.flatMap (fun p => p.1.map (fun l => (p.2, l)))

/-- Two occurrences conflict when they are in the same clause, or are opposite literals of the
same variable. -/
def conflict (F : Cnf) (a b : ℕ) : Bool :=
  match (lits F)[a]?, (lits F)[b]? with
  | some (i, l), some (j, l') => (i == j) || (l.var == l'.var && l.pol != l'.pol)
  | _, _ => false

/-- The edge list of the conflict graph. -/
def edgeList (F : Cnf) : List (ℕ × ℕ) :=
  (List.range (lits F).length).flatMap fun a =>
    ((List.range (lits F).length).filter (fun b => a ≠ b && conflict F a b)).map (fun b => (a, b))

/-- The graph instance built from a CNF formula. -/
def graphOf (F : Cnf) : GraphInst where
  n := (lits F).length
  edges := edgeList F
  k := F.length

theorem mem_edgeList {F : Cnf} {a b : ℕ} :
    (a, b) ∈ edgeList F ↔
      a < (lits F).length ∧ b < (lits F).length ∧ a ≠ b ∧ conflict F a b = true := by
  simp only [edgeList, List.mem_flatMap, List.mem_map, List.mem_filter, List.mem_range,
    Prod.mk.injEq, Bool.and_eq_true, decide_eq_true_eq, ne_eq]
  constructor
  · rintro ⟨x, hx, y, ⟨hy, hne, hcf⟩, rfl, rfl⟩
    exact ⟨hx, hy, hne, hcf⟩
  · rintro ⟨ha, hb, hne, hcf⟩
    exact ⟨a, ha, b, ⟨hb, hne, hcf⟩, rfl, rfl⟩

/-- The occurrences of a formula are exactly the literals of its clauses, tagged with the clause
index. -/
theorem mem_lits {F : Cnf} {i : ℕ} {l : Lit} :
    (i, l) ∈ lits F ↔ ∃ c, F[i]? = some c ∧ l ∈ c := by
  simp only [lits, List.mem_flatMap, List.mem_map, List.mem_zipIdx_iff_getElem?, Prod.mk.injEq]
  constructor
  · rintro ⟨p, hp, l', hl', h1, h2⟩
    exact ⟨p.1, by rw [← h1]; exact hp, h2 ▸ hl'⟩
  · rintro ⟨c, hc, hl⟩
    exact ⟨(c, i), hc, l, hl, rfl, rfl⟩

/-- Each occurrence index points at a literal of the clause it records. -/
theorem lits_getElem {F : Cnf} {v i : ℕ} {l : Lit} (h : (lits F)[v]? = some (i, l)) :
    ∃ c, F[i]? = some c ∧ l ∈ c :=
  mem_lits.mp (List.mem_iff_getElem?.mpr ⟨v, h⟩)

/-- Every literal of every clause occurs. -/
theorem exists_index_of_mem {F : Cnf} {i : ℕ} {c : Clause} (hc : F[i]? = some c) {l : Lit}
    (hl : l ∈ c) : ∃ v : ℕ, (lits F)[v]? = some (i, l) :=
  List.mem_iff_getElem?.mp (mem_lits.mpr ⟨c, hc, hl⟩)

/-- Occurrence indices record clauses of the formula. -/
theorem lits_fst_lt {F : Cnf} {v i : ℕ} {l : Lit} (h : (lits F)[v]? = some (i, l)) :
    i < F.length := by
  obtain ⟨c, hc, -⟩ := lits_getElem h
  by_contra hcon
  rw [List.getElem?_eq_none_iff.mpr (by omega)] at hc
  exact absurd hc (by simp)

/-! ### The occurrence at a vertex -/

/-- The occurrence recorded at a vertex (junk outside the range). -/
def occ (F : Cnf) (v : ℕ) : ℕ × Lit := ((lits F)[v]?).getD (0, ⟨0, false⟩)

theorem occ_spec {F : Cnf} {v : ℕ} (h : v < (lits F).length) :
    (lits F)[v]? = some (occ F v) := by
  rw [occ, List.getElem?_eq_getElem h]
  rfl

theorem occ_fst_lt {F : Cnf} {v : ℕ} (h : v < (lits F).length) : (occ F v).1 < F.length :=
  lits_fst_lt (i := (occ F v).1) (l := (occ F v).2) (by rw [occ_spec h])

theorem occ_mem {F : Cnf} {v : ℕ} (h : v < (lits F).length) :
    ∃ c, F[(occ F v).1]? = some c ∧ (occ F v).2 ∈ c :=
  lits_getElem (i := (occ F v).1) (l := (occ F v).2) (by rw [occ_spec h])

theorem conflict_spec {F : Cnf} {a b : ℕ} (ha : a < (lits F).length)
    (hb : b < (lits F).length) :
    conflict F a b = true ↔
      (occ F a).1 = (occ F b).1 ∨
        ((occ F a).2.var = (occ F b).2.var ∧ (occ F a).2.pol ≠ (occ F b).2.pol) := by
  rw [conflict, occ_spec ha, occ_spec hb]
  simp [Bool.or_eq_true, Bool.and_eq_true]

/-! ### Correctness -/

/-- **Correctness of the reduction.** -/
theorem graphOf_correct (F : Cnf) :
    (∃ σ : ℕ → Bool, F.holds σ) ↔
      ∃ S : Finset ℕ, (graphOf F).Indep S ∧ (graphOf F).k ≤ S.card := by
  classical
  constructor
  · rintro ⟨σ, hσ⟩
    have hpick : ∀ i, i < F.length →
        ∃ v, v < (lits F).length ∧ (occ F v).1 = i ∧ Lit.holds σ (occ F v).2 := by
      intro i hi
      have hc : F[i]? = some F[i] := List.getElem?_eq_getElem hi
      obtain ⟨l, hl, hlh⟩ := hσ F[i] (List.mem_iff_getElem?.mpr ⟨i, hc⟩)
      obtain ⟨v, hv⟩ := exists_index_of_mem hc hl
      have hvlt : v < (lits F).length := by
        by_contra hcon
        rw [List.getElem?_eq_none_iff.mpr (by omega)] at hv
        exact absurd hv (by simp)
      have hocc : occ F v = (i, l) := by
        have := occ_spec hvlt
        rw [hv] at this
        exact (Option.some_inj.mp this).symm
      exact ⟨v, hvlt, by rw [hocc], by rw [hocc]; exact hlh⟩
    choose! g hg1 hg2 hg3 using hpick
    refine ⟨(Finset.range F.length).image g, ⟨?_, ?_⟩, ?_⟩
    · intro x hx
      simp only [Finset.mem_image, Finset.mem_range] at hx
      obtain ⟨i, hi, rfl⟩ := hx
      simpa [graphOf] using hg1 i hi
    · rintro ⟨x, y⟩ he ⟨hxS, hyS⟩
      simp only [Finset.mem_image, Finset.mem_range] at hxS hyS
      obtain ⟨i, hi, rfl⟩ := hxS
      obtain ⟨j, hj, rfl⟩ := hyS
      have he' := mem_edgeList.mp he
      obtain ⟨hxl, hyl, hne, hcf⟩ := he'
      rcases (conflict_spec hxl hyl).mp hcf with h | ⟨hvar, hpol⟩
      · rw [hg2 i hi, hg2 j hj] at h
        exact hne (by rw [h])
      · have h1 : σ (occ F (g i)).2.var = (occ F (g i)).2.pol := hg3 i hi
        have h2 : σ (occ F (g j)).2.var = (occ F (g j)).2.pol := hg3 j hj
        rw [hvar, h2] at h1
        exact hpol h1.symm
    · have hinj : Set.InjOn g (Finset.range F.length) := by
        intro i hi j hj hij
        simp only [Finset.coe_range, Set.mem_Iio] at hi hj
        have := hg2 i hi
        rw [hij, hg2 j hj] at this
        exact this.symm
      rw [Finset.card_image_of_injOn hinj, Finset.card_range]
      exact Nat.le_refl _
  · rintro ⟨S, ⟨hSsub, hSedge⟩, hcard⟩
    have hSlt : ∀ v ∈ S, v < (lits F).length := by
      intro v hv
      simpa [graphOf] using hSsub hv
    have hnc : ∀ x ∈ S, ∀ y ∈ S, x ≠ y → conflict F x y = false := by
      intro x hx y hy hxy
      by_contra hcon
      have hcf : conflict F x y = true := by
        cases h : conflict F x y with
        | false => exact absurd h hcon
        | true => rfl
      exact hSedge (x, y) (mem_edgeList.mpr ⟨hSlt x hx, hSlt y hy, hxy, hcf⟩) ⟨hx, hy⟩
    set σ : ℕ → Bool :=
      fun x => decide (∃ v ∈ S, (occ F v).2.var = x ∧ (occ F v).2.pol = true) with hσdef
    have hlit : ∀ v ∈ S, Lit.holds σ (occ F v).2 := by
      intro v hv
      show σ (occ F v).2.var = (occ F v).2.pol
      cases hpol : (occ F v).2.pol with
      | true =>
          simp only [hσdef, decide_eq_true_eq]
          exact ⟨v, hv, rfl, hpol⟩
      | false =>
          simp only [hσdef, decide_eq_false_iff_not, not_exists]
          rintro w ⟨hw, hvar, hwpol⟩
          have hne : w ≠ v := by
            rintro rfl
            rw [hpol] at hwpol
            exact Bool.noConfusion hwpol
          have : conflict F v w = true :=
            (conflict_spec (hSlt v hv) (hSlt w hw)).mpr
              (Or.inr ⟨hvar.symm, by rw [hpol, hwpol]; exact Bool.noConfusion⟩)
          rw [hnc v hv w hw (Ne.symm hne)] at this
          exact Bool.noConfusion this
    -- the clause indices of `S` cover all clauses
    have himg : S.image (fun v => (occ F v).1) = Finset.range F.length := by
      apply Finset.eq_of_subset_of_card_le
      · intro i hi
        simp only [Finset.mem_image] at hi
        obtain ⟨v, hv, rfl⟩ := hi
        exact Finset.mem_range.mpr (occ_fst_lt (hSlt v hv))
      · have hinj : Set.InjOn (fun v => (occ F v).1) S := by
          intro x hx y hy hxy
          by_contra hne
          have : conflict F x y = true :=
            (conflict_spec (hSlt x hx) (hSlt y hy)).mpr (Or.inl hxy)
          rw [hnc x hx y hy hne] at this
          exact Bool.noConfusion this
        rw [Finset.card_image_of_injOn hinj, Finset.card_range]
        simpa [graphOf] using hcard
    refine ⟨σ, ?_⟩
    intro c hc
    obtain ⟨i, hi⟩ := List.mem_iff_getElem?.mp hc
    have hilt : i < F.length := by
      by_contra hcon
      rw [List.getElem?_eq_none_iff.mpr (by omega)] at hi
      exact absurd hi (by simp)
    have : i ∈ S.image (fun v => (occ F v).1) := by rw [himg]; exact Finset.mem_range.mpr hilt
    simp only [Finset.mem_image] at this
    obtain ⟨v, hv, hvi⟩ := this
    obtain ⟨c', hc', hmem⟩ := occ_mem (hSlt v hv)
    rw [hvi, hi] at hc'
    have : c' = c := (Option.some_inj.mp hc').symm
    subst this
    exact ⟨(occ F v).2, hmem, hlit v hv⟩

/-! ### Size -/

theorem lits_length_aux (F : Cnf) (k : ℕ) :
    ((F.zipIdx k).flatMap (fun p => p.1.map (fun l => (p.2, l)))).length
      = (F.map List.length).sum := by
  induction F generalizing k with
  | nil => simp
  | cons c F ih => simp [List.zipIdx_cons, ih]

/-- The number of occurrences is bounded by the size of the formula. -/
theorem lits_length_le (F : Cnf) : (lits F).length ≤ F.size := by
  rw [lits, lits_length_aux F 0]
  simp [Cnf.size]

theorem length_flatMap_le {α β : Type} (l : List α) (f : α → List β) (m : ℕ)
    (h : ∀ a ∈ l, (f a).length ≤ m) : (l.flatMap f).length ≤ l.length * m := by
  induction l with
  | nil => simp
  | cons a l ih =>
      rw [List.flatMap_cons, List.length_append, List.length_cons]
      have h1 := h a (by simp)
      have h2 := ih (fun x hx => h x (by simp [hx]))
      calc (f a).length + (l.flatMap f).length ≤ m + l.length * m := Nat.add_le_add h1 h2
        _ = (l.length + 1) * m := by ring

theorem edgeList_length_le (F : Cnf) :
    (edgeList F).length ≤ (lits F).length * (lits F).length := by
  rw [edgeList]
  refine le_trans (length_flatMap_le _ _ (lits F).length ?_) ?_
  · intro a _
    rw [List.length_map]
    exact le_trans (List.length_filter_le _ _) (by simp)
  · simp

/-- **The graph is polynomially large.** -/
theorem graphOf_size_le (F : Cnf) : (graphOf F).size ≤ 3 * (F.size + 1) ^ 2 := by
  have h1 := lits_length_le F
  have h2 := edgeList_length_le F
  have h3 : F.length ≤ F.size := by simp [Cnf.size]
  have h4 : (lits F).length * (lits F).length ≤ F.size * F.size := Nat.mul_le_mul h1 h1
  have : (graphOf F).size = (lits F).length + (edgeList F).length + F.length := rfl
  rw [this]
  have hsq : 3 * (F.size + 1) ^ 2 = 3 * (F.size * F.size) + 6 * F.size + 3 := by ring
  omega

end SatIndep

/-- **CNF satisfiability reduces to independent set.** -/
def satIndepReduction : Reduction cnfSat independentSet where
  map := SatIndep.graphOf
  correct := SatIndep.graphOf_correct
  deg := 2
  const := 3
  size_le := SatIndep.graphOf_size_le

/-- **Independent set is NP-hard.** -/
theorem independentSet_hard : Hard NPProblem independentSet :=
  cnfSat_hard.trans satIndepReduction

end IDR.Complexity
