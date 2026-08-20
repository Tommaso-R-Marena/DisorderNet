/-
# Independent set reduces to weighted linear ordering

Each vertex `v` of the graph becomes two items, `2v` and `2v+1`, joined by a *cheap* arc of weight
`1` from `2v` to `2v+1`; each edge `{u,v}` contributes two *heavy* arcs of weight `H = n+1`, from
`2v+1` to `2u` and from `2u+1` to `2v`.  A linear order collects the weight of every arc whose
tail it places above its head.

Placing all heavy arcs correctly forces the set `S = {v : 2v is above 2v+1}` to be independent:
for an edge `{u,v}` with both endpoints in `S` the four constraints close a cycle
`2u+1 < 2u < 2v+1 < 2v < 2u+1`.  Since a single lost heavy arc costs `H > n` and all the cheap
arcs together are worth only `n`, the optimum is `H·(#heavy arcs) + α(G)`, with `α(G)` the
independence number.  Hence the linear ordering optimum answers the independent set question.
-/
import RequestProject.Complexity.LopProblem

set_option autoImplicit false

namespace IDR.Complexity

open Finset IDR.GroupedAUC.Hardness

namespace IndepLop

variable (I : GraphInst)

/-- Two items per vertex. -/
def K : ℕ := 2 * I.n

theorem K_eq : K I = 2 * I.n := rfl

/-- The weight of a heavy arc: more than all the cheap arcs together. -/
def H : ℕ := I.n + 1

/-- Adjacency of the graph, symmetrised. -/
def adj (u v : ℕ) : Bool := decide ((u, v) ∈ I.edges ∨ (v, u) ∈ I.edges)

theorem adj_symm (u v : ℕ) : adj I u v = adj I v u := by
  simp only [adj, decide_eq_decide]
  exact or_comm

/-- The cheap arcs: from `2v` to `2v+1`. -/
def cheapW (x y : Fin (K I)) : ℕ := if x.val % 2 = 0 ∧ y.val = x.val + 1 then 1 else 0

/-- The heavy arcs: from `2v+1` to `2u` whenever `u` and `v` are adjacent. -/
def heavyW (x y : Fin (K I)) : ℕ :=
  if x.val % 2 = 1 ∧ y.val % 2 = 0 ∧ adj I (x.val / 2) (y.val / 2) = true then 1 else 0

theorem heavyW_le_one (x y : Fin (K I)) : heavyW I x y ≤ 1 := by
  rw [heavyW]; split <;> simp

/-- The weight matrix of the linear ordering instance. -/
def Wof (x y : Fin (K I)) : ℕ := cheapW I x y + H I * heavyW I x y

/-- The number of heavy arcs. -/
def heavyTotal : ℕ := ∑ x : Fin (K I), ∑ y : Fin (K I), heavyW I x y

/-- The linear ordering instance built from a graph. -/
def lopOf : LopInst where
  K := K I
  W := Wof I
  target := H I * heavyTotal I + I.k

/-- The rank of an item index, extended by junk outside the range. -/
def rk (r : Fin (K I) → ℕ) (m : ℕ) : ℕ := if h : m < K I then r ⟨m, h⟩ else 0

@[simp] theorem rk_coe (r : Fin (K I) → ℕ) (x : Fin (K I)) : rk I r x.val = r x := by
  rw [rk, dif_pos x.isLt]

/-- The set of vertices placed "in above out" by a ranking. -/
def chosen (r : Fin (K I) → ℕ) : Finset ℕ :=
  (Finset.range I.n).filter (fun v => rk I r (2 * v + 1) < rk I r (2 * v))

theorem chosen_subset (r : Fin (K I) → ℕ) : chosen I r ⊆ Finset.range I.n :=
  Finset.filter_subset _ _

theorem card_chosen_le (r : Fin (K I) → ℕ) : (chosen I r).card ≤ I.n := by
  simpa using Finset.card_le_card (chosen_subset I r)

/-! ### Splitting the objective -/

/-- Splitting a `range (2 * n)` sum into the even and odd halves. -/
theorem sum_range_two_mul (G : ℕ → ℕ) (n : ℕ) :
    ∑ m ∈ Finset.range (2 * n), G m = ∑ v ∈ Finset.range n, (G (2 * v) + G (2 * v + 1)) := by
  induction n with
  | zero => simp
  | succ n ih =>
      have h : 2 * (n + 1) = (2 * n + 1) + 1 := by ring
      rw [h, Finset.sum_range_succ, Finset.sum_range_succ, ih, Finset.sum_range_succ]
      ring

/-- The cheap part of the objective counts exactly the chosen vertices. -/
theorem cheap_sum_eq_card (r : Fin (K I) → ℕ) :
    (∑ x : Fin (K I), ∑ y : Fin (K I), if r y < r x then cheapW I x y else 0)
      = (chosen I r).card := by
  classical
  -- the inner sum keeps only the successor item
  have hinner : ∀ x : Fin (K I),
      (∑ y : Fin (K I), if r y < r x then cheapW I x y else 0)
        = if x.val % 2 = 0 ∧ rk I r (x.val + 1) < rk I r x.val then 1 else 0 := by
    intro x
    by_cases hx : x.val % 2 = 0
    · have hlt : x.val + 1 < K I := by
        have h2 : x.val < K I := x.isLt
        have hK : K I = 2 * I.n := K_eq I
        omega
      have hsingle : ∀ y : Fin (K I), y ≠ ⟨x.val + 1, hlt⟩ →
          (if r y < r x then cheapW I x y else 0) = 0 := by
        intro y hy
        have : y.val ≠ x.val + 1 := by
          intro hcon
          exact hy (Fin.ext hcon)
        simp [cheapW, this]
      rw [Finset.sum_eq_single (⟨x.val + 1, hlt⟩ : Fin (K I)) (fun y _ hy => hsingle y hy)
        (fun hcon => absurd (Finset.mem_univ _) hcon)]
      have hc : cheapW I x ⟨x.val + 1, hlt⟩ = 1 := by simp [cheapW, hx]
      rw [hc, rk, dif_pos hlt, rk_coe]
      by_cases hlt2 : r (⟨x.val + 1, hlt⟩ : Fin (K I)) < r x
      · simp [hlt2, hx]
      · simp [hlt2, hx]
    · have : ∀ y : Fin (K I), (if r y < r x then cheapW I x y else 0) = 0 := by
        intro y; simp [cheapW, hx]
      simp [this, hx]
  rw [Finset.sum_congr rfl (fun x _ => hinner x)]
  -- turn the sum over `Fin (K I)` into a sum over `range (2 * n)`
  set G : ℕ → ℕ := fun m => if m % 2 = 0 ∧ rk I r (m + 1) < rk I r m then 1 else 0 with hG
  have hsum : (∑ x : Fin (K I), G x.val) = ∑ m ∈ Finset.range (K I), G m :=
    Fin.sum_univ_eq_sum_range G (K I)
  rw [show (∑ x : Fin (K I), if x.val % 2 = 0 ∧ rk I r (x.val + 1) < rk I r x.val then 1 else 0)
      = ∑ x : Fin (K I), G x.val from rfl, hsum, K_eq, sum_range_two_mul]
  rw [chosen, Finset.card_filter]
  refine Finset.sum_congr rfl fun v _ => ?_
  have hodd : G (2 * v + 1) = 0 := by simp [hG]
  have heven : G (2 * v) = if rk I r (2 * v + 1) < rk I r (2 * v) then 1 else 0 := by
    simp [hG]
  omega

/-- The heavy part of the objective never exceeds the total heavy weight. -/
theorem heavy_sum_le (r : Fin (K I) → ℕ) :
    (∑ x : Fin (K I), ∑ y : Fin (K I), if r y < r x then heavyW I x y else 0) ≤ heavyTotal I := by
  refine Finset.sum_le_sum fun x _ => Finset.sum_le_sum fun y _ => ?_
  split <;> simp

/-- If the heavy part is maximal then every heavy arc is placed correctly. -/
theorem heavy_arcs_of_sum_eq {r : Fin (K I) → ℕ}
    (h : (∑ x : Fin (K I), ∑ y : Fin (K I), if r y < r x then heavyW I x y else 0)
        = heavyTotal I) :
    ∀ x y : Fin (K I), heavyW I x y = 1 → r y < r x := by
  have hle : ∀ x ∈ (univ : Finset (Fin (K I))),
      (∑ y : Fin (K I), if r y < r x then heavyW I x y else 0)
        ≤ ∑ y : Fin (K I), heavyW I x y := by
    intro x _
    exact Finset.sum_le_sum fun y _ => by split <;> simp
  have houter := (Finset.sum_eq_sum_iff_of_le hle).mp h
  intro x y hxy
  have hx := houter x (Finset.mem_univ x)
  have hin : ∀ y ∈ (univ : Finset (Fin (K I))),
      (if r y < r x then heavyW I x y else 0) ≤ heavyW I x y := by
    intro y _; split <;> simp
  have := (Finset.sum_eq_sum_iff_of_le hin).mp hx y (Finset.mem_univ y)
  by_contra hcon
  rw [if_neg hcon, hxy] at this
  exact absurd this.symm (by simp)

/-- A ranking that places every heavy arc correctly collects all the heavy weight. -/
theorem heavy_sum_eq_of_arcs {r : Fin (K I) → ℕ}
    (hheavy : ∀ x y : Fin (K I), heavyW I x y = 1 → r y < r x) :
    (∑ x : Fin (K I), ∑ y : Fin (K I), if r y < r x then heavyW I x y else 0)
      = heavyTotal I := by
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  by_cases hw : heavyW I x y = 1
  · rw [if_pos (hheavy x y hw)]
  · have : heavyW I x y = 0 := by
      have := heavyW_le_one I x y
      omega
    rw [this]
    split <;> rfl

/-- The objective splits into its cheap and heavy parts. -/
theorem rankValue_split (r : Fin (K I) → ℕ) :
    rankValue (Wof I) r
      = (chosen I r).card
        + H I * (∑ x : Fin (K I), ∑ y : Fin (K I), if r y < r x then heavyW I x y else 0) := by
  rw [← cheap_sum_eq_card, Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [Wof]
  split <;> simp

/-! ### From rankings to independent sets and back -/

/-- The chosen set of a ranking that places all heavy arcs correctly is independent. -/
theorem indep_chosen {r : Fin (K I) → ℕ}
    (hheavy : ∀ x y : Fin (K I), heavyW I x y = 1 → r y < r x) : I.Indep (chosen I r) := by
  refine ⟨chosen_subset I r, ?_⟩
  rintro ⟨u, v⟩ he ⟨hu, hv⟩
  rw [chosen, Finset.mem_filter, Finset.mem_range] at hu hv
  obtain ⟨hun, hu'⟩ := hu
  obtain ⟨hvn, hv'⟩ := hv
  have h2u : 2 * u < K I := by rw [K_eq]; omega
  have h2u1 : 2 * u + 1 < K I := by rw [K_eq]; omega
  have h2v : 2 * v < K I := by rw [K_eq]; omega
  have h2v1 : 2 * v + 1 < K I := by rw [K_eq]; omega
  set xu : Fin (K I) := ⟨2 * u, h2u⟩
  set xu' : Fin (K I) := ⟨2 * u + 1, h2u1⟩
  set xv : Fin (K I) := ⟨2 * v, h2v⟩
  set xv' : Fin (K I) := ⟨2 * v + 1, h2v1⟩
  have hadj1 : adj I u v = true := by simp [adj]; exact Or.inl he
  have hadj2 : adj I v u = true := by rw [adj_symm]; exact hadj1
  have hw1 : heavyW I xu' xv = 1 := by
    have h1 : (2 * u + 1) % 2 = 1 := by omega
    have h2 : (2 * v) % 2 = 0 := by omega
    have h3 : (2 * u + 1) / 2 = u := by omega
    have h4 : (2 * v) / 2 = v := by omega
    simp only [heavyW, xu', xv, h1, h2, h3, h4, hadj1, and_self, if_true]
  have hw2 : heavyW I xv' xu = 1 := by
    have h1 : (2 * v + 1) % 2 = 1 := by omega
    have h2 : (2 * u) % 2 = 0 := by omega
    have h3 : (2 * v + 1) / 2 = v := by omega
    have h4 : (2 * u) / 2 = u := by omega
    simp only [heavyW, xv', xu, h1, h2, h3, h4, hadj2, and_self, if_true]
  have e1 : r xv < r xu' := hheavy _ _ hw1
  have e2 : r xu < r xv' := hheavy _ _ hw2
  rw [rk, dif_pos h2u1, rk, dif_pos h2u] at hu'
  rw [rk, dif_pos h2v1, rk, dif_pos h2v] at hv'
  have hu2 : r xu' < r xu := hu'
  have hv2 : r xv' < r xv := hv'
  omega

/-- The band of an item: the four blocks of the linear order built from an independent set. -/
def band (S : Finset ℕ) [DecidablePred (· ∈ S)] (x : Fin (K I)) : ℕ :=
  if x.val % 2 = 0 then (if x.val / 2 ∈ S then 2 else 0) else (if x.val / 2 ∈ S then 1 else 3)

/-- The ranking built from an independent set: blocks in the order
"in of non-chosen", "out of chosen", "in of chosen", "out of non-chosen". -/
def rankOf (S : Finset ℕ) [DecidablePred (· ∈ S)] (x : Fin (K I)) : ℕ :=
  band I S x * K I + x.val

theorem rankOf_lt_of_band_lt (S : Finset ℕ) [DecidablePred (· ∈ S)] {x y : Fin (K I)}
    (h : band I S x < band I S y) : rankOf I S x < rankOf I S y := by
  calc rankOf I S x = band I S x * K I + x.val := rfl
    _ < band I S x * K I + K I := by have := x.isLt; omega
    _ = (band I S x + 1) * K I := by ring
    _ ≤ band I S y * K I := Nat.mul_le_mul_right _ h
    _ ≤ rankOf I S y := Nat.le_add_right _ _

theorem rankOf_injective (S : Finset ℕ) [DecidablePred (· ∈ S)] :
    Function.Injective (rankOf I S) := by
  intro x y hxy
  rcases lt_trichotomy (band I S x) (band I S y) with h | h | h
  · exact absurd hxy (Nat.ne_of_lt (rankOf_lt_of_band_lt I S h))
  · have : band I S x * K I + x.val = band I S y * K I + y.val := hxy
    rw [h] at this
    exact Fin.ext (by omega)
  · exact absurd hxy.symm (Nat.ne_of_lt (rankOf_lt_of_band_lt I S h))

/-- Conversely, every independent set gives a ranking that places all heavy arcs correctly and
chooses exactly that set. -/
theorem exists_rank_of_indep {S : Finset ℕ} (hS : I.Indep S) :
    ∃ r : Fin (K I) → ℕ, Function.Injective r ∧ chosen I r = S ∧
      (∀ x y : Fin (K I), heavyW I x y = 1 → r y < r x) := by
  classical
  refine ⟨rankOf I S, rankOf_injective I S, ?_, ?_⟩
  · ext v
    rw [chosen, Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨hvn, hlt⟩
      by_contra hvS
      have h2v : 2 * v < K I := by rw [K_eq]; omega
      have h2v1 : 2 * v + 1 < K I := by rw [K_eq]; omega
      have hb1 : band I S ⟨2 * v, h2v⟩ = 0 := by
        have h1 : (2 * v) % 2 = 0 := by omega
        have h2 : (2 * v) / 2 = v := by omega
        simp [band, h1, h2, hvS]
      have hb2 : band I S ⟨2 * v + 1, h2v1⟩ = 3 := by
        have h1 : (2 * v + 1) % 2 ≠ 0 := by omega
        have h2 : (2 * v + 1) / 2 = v := by omega
        simp [band, h2, hvS]
      have := rankOf_lt_of_band_lt I S (x := ⟨2 * v, h2v⟩) (y := ⟨2 * v + 1, h2v1⟩)
        (by rw [hb1, hb2]; norm_num)
      rw [rk, dif_pos h2v1, rk, dif_pos h2v] at hlt
      omega
    · intro hvS
      have hvn : v < I.n := Finset.mem_range.mp (hS.1 hvS)
      refine ⟨hvn, ?_⟩
      have h2v : 2 * v < K I := by rw [K_eq]; omega
      have h2v1 : 2 * v + 1 < K I := by rw [K_eq]; omega
      have hb1 : band I S ⟨2 * v, h2v⟩ = 2 := by
        have h1 : (2 * v) % 2 = 0 := by omega
        have h2 : (2 * v) / 2 = v := by omega
        simp [band, h1, h2, hvS]
      have hb2 : band I S ⟨2 * v + 1, h2v1⟩ = 1 := by
        have h1 : (2 * v + 1) % 2 ≠ 0 := by omega
        have h2 : (2 * v + 1) / 2 = v := by omega
        simp [band, h2, hvS]
      have := rankOf_lt_of_band_lt I S (x := ⟨2 * v + 1, h2v1⟩) (y := ⟨2 * v, h2v⟩)
        (by rw [hb1, hb2]; norm_num)
      rw [rk, dif_pos h2v1, rk, dif_pos h2v]
      exact this
  · intro x y hxy
    rw [heavyW] at hxy
    have hcond : x.val % 2 = 1 ∧ y.val % 2 = 0 ∧ adj I (x.val / 2) (y.val / 2) = true := by
      by_contra hcon
      rw [if_neg hcon] at hxy
      exact absurd hxy (by simp)
    obtain ⟨hxodd, hyeven, hadj⟩ := hcond
    set u := x.val / 2
    set v := y.val / 2
    apply rankOf_lt_of_band_lt
    have hbx : band I S x = if u ∈ S then 1 else 3 := by
      rw [band, if_neg (by omega)]
    have hby : band I S y = if v ∈ S then 2 else 0 := by
      rw [band, if_pos hyeven]
    by_cases hu : u ∈ S <;> by_cases hv : v ∈ S
    · exfalso
      have : (u, v) ∈ I.edges ∨ (v, u) ∈ I.edges := by
        simpa [adj] using hadj
      rcases this with h | h
      · exact hS.2 (u, v) h ⟨hu, hv⟩
      · exact hS.2 (v, u) h ⟨hv, hu⟩
    · rw [hbx, hby, if_pos hu, if_neg hv]; norm_num
    · rw [hbx, hby, if_neg hu, if_pos hv]; norm_num
    · rw [hbx, hby, if_neg hu, if_neg hv]; norm_num

/-- **Correctness of the reduction.** -/
theorem lopOf_correct :
    (∃ S : Finset ℕ, I.Indep S ∧ I.k ≤ S.card) ↔ (lopOf I).target ≤ lopOpt (lopOf I).W := by
  classical
  show (∃ S : Finset ℕ, I.Indep S ∧ I.k ≤ S.card) ↔
    H I * heavyTotal I + I.k ≤ lopOpt (Wof I)
  constructor
  · rintro ⟨S, hS, hcard⟩
    obtain ⟨r, hinj, hch, hheavy⟩ := exists_rank_of_indep I hS
    have h1 : rankValue (Wof I) r = S.card + H I * heavyTotal I := by
      rw [rankValue_split, hch, heavy_sum_eq_of_arcs I hheavy]
    have h2 := rankValue_le_lopOpt (Wof I) r hinj
    calc H I * heavyTotal I + I.k ≤ H I * heavyTotal I + S.card := Nat.add_le_add_left hcard _
      _ = S.card + H I * heavyTotal I := Nat.add_comm _ _
      _ = rankValue (Wof I) r := h1.symm
      _ ≤ lopOpt (Wof I) := h2
  · intro h
    obtain ⟨r, hinj, hr⟩ := exists_rank_eq_lopOpt (Wof I)
    have h' : H I * heavyTotal I + I.k ≤ rankValue (Wof I) r := by rw [hr]; exact h
    rw [rankValue_split] at h'
    set hs := ∑ x : Fin (K I), ∑ y : Fin (K I), if r y < r x then heavyW I x y else 0 with hsdef
    have hle : hs ≤ heavyTotal I := heavy_sum_le I r
    have hcard := card_chosen_le I r
    have hH : H I = I.n + 1 := rfl
    set A := H I * heavyTotal I with hA
    set B := H I * hs with hB
    have heq : hs = heavyTotal I := by
      by_contra hne
      have hstep : hs + 1 ≤ heavyTotal I := by omega
      have : B + H I ≤ A := by
        rw [hA, hB]
        calc H I * hs + H I = H I * (hs + 1) := by ring
          _ ≤ H I * heavyTotal I := Nat.mul_le_mul_left _ hstep
      omega
    have harcs := heavy_arcs_of_sum_eq I (r := r) (by rw [← hsdef]; exact heq)
    refine ⟨chosen I r, indep_chosen I harcs, ?_⟩
    have : B = A := by rw [hA, hB, heq]
    omega

/-! ### Size of the instance -/

theorem KK_eq : K I * K I = 4 * (I.n * I.n) := by rw [K_eq]; ring

theorem heavyTotal_le : heavyTotal I ≤ 4 * (I.n * I.n) := by
  rw [heavyTotal, ← KK_eq]
  calc ∑ x : Fin (K I), ∑ y : Fin (K I), heavyW I x y
      ≤ ∑ _x : Fin (K I), ∑ _y : Fin (K I), 1 :=
        Finset.sum_le_sum fun x _ => Finset.sum_le_sum fun y _ => heavyW_le_one I x y
    _ = K I * K I := by simp [mul_comm]

theorem cheapTotal_le : (∑ x : Fin (K I), ∑ y : Fin (K I), cheapW I x y) ≤ 4 * (I.n * I.n) := by
  rw [← KK_eq]
  calc ∑ x : Fin (K I), ∑ y : Fin (K I), cheapW I x y
      ≤ ∑ _x : Fin (K I), ∑ _y : Fin (K I), 1 := by
        refine Finset.sum_le_sum fun x _ => Finset.sum_le_sum fun y _ => ?_
        rw [cheapW]; split <;> simp
    _ = K I * K I := by simp [mul_comm]

/-- The arithmetic behind the size bound. -/
theorem size_arith (n s k : ℕ) (hn : n ≤ s) (hk : k ≤ s) :
    2 * n + (4 * (n * n) + (n + 1) * (4 * (n * n))) + ((n + 1) * (4 * (n * n)) + k)
      ≤ 16 * (s + 1) ^ 3 := by
  nlinarith [Nat.mul_le_mul (Nat.mul_le_mul hn hn) hn, Nat.mul_le_mul hn hn, hn, hk]

/-- **The linear ordering instance is polynomially large.** -/
theorem lopOf_size_le : (lopOf I).size ≤ 16 * (I.size + 1) ^ 3 := by
  have hW : (∑ x : Fin (K I), ∑ y : Fin (K I), (lopOf I).W x y)
      = (∑ x : Fin (K I), ∑ y : Fin (K I), cheapW I x y) + H I * heavyTotal I := by
    show (∑ x : Fin (K I), ∑ y : Fin (K I), Wof I x y) = _
    rw [heavyTotal, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    rfl
  have hsize : (lopOf I).size
      = K I + ((∑ x : Fin (K I), ∑ y : Fin (K I), cheapW I x y) + H I * heavyTotal I)
        + (H I * heavyTotal I + I.k) := by
    rw [LopInst.size, ← hW]
    rfl
  have hn : I.n ≤ I.size := by rw [GraphInst.size]; omega
  have hk : I.k ≤ I.size := by rw [GraphInst.size]; omega
  have hKn : K I = 2 * I.n := rfl
  have hHn : H I = I.n + 1 := rfl
  have h1 := cheapTotal_le I
  have h2 : H I * heavyTotal I ≤ (I.n + 1) * (4 * (I.n * I.n)) := by
    rw [hHn]
    exact Nat.mul_le_mul_left _ (heavyTotal_le I)
  rw [hsize]
  refine le_trans ?_ (size_arith I.n I.size I.k hn hk)
  exact Nat.add_le_add (Nat.add_le_add (le_of_eq hKn) (Nat.add_le_add h1 h2))
    (Nat.add_le_add h2 (le_refl I.k))

end IndepLop

/-- **Independent set reduces to weighted linear ordering.** -/
def indepLopReduction : Reduction independentSet lopProblem where
  map := IndepLop.lopOf
  correct := IndepLop.lopOf_correct
  deg := 3
  const := 16
  size_le := IndepLop.lopOf_size_le

/-- **Weighted linear ordering is NP-hard** — machine-checked, not cited. -/
theorem lopProblem_hard : Hard NPProblem lopProblem :=
  independentSet_hard.trans indepLopReduction

end IDR.Complexity
