/-
# Part CXV  A rigorous lower bound on the connective constant, by bridges

Part CXIV gave the exact connective constant of the directed lattice (`log 2`) and an improved
rigorous *upper* bound `log 780 / 6` for the genuine square lattice.  The matching lower bound
available so far was the directed-walk bound `log 2`, which is far from the truth.

This file improves the lower bound by the classical *bridge* construction, made completely
formal.  A bridge is a self-avoiding walk that starts at the origin, leaves the line `x = 0`
immediately and never goes to the right of its own endpoint:

* `IsBridge` : `∀ p ∈ sites l, (p = 0 ∨ 0 < p.1) ∧ p.1 ≤ (endpoint l).1`.

The point of the definition is `bridge_append`: the concatenation of two bridges is again a
bridge **and is again self-avoiding** — the second bridge lives strictly to the right of the
line through the junction, so it cannot collide with the first.  Hence the bridge count is
*super*multiplicative (`bcnt_supermultiplicative`), the opposite of the submultiplicativity of
the full count, and every single bridge count gives a lower bound on the connective constant
(`log_bcnt_div_le_connectiveConstant`), exactly dual to `connectiveConstant_le_div`.

With the exact counts `bcnt 6 = 101` and `bcnt 7 = 251` this gives

  `log 251 / 7 ≤ μ ≤ log 780 / 6`,  i.e.  `0.789… ≤ μ ≤ 1.109…`,

strictly better on both sides than the previously available `log 2 ≤ μ ≤ log 100 / 4`.
The exact value of `μ` remains open mathematics; what is delivered here is the machinery that
turns any finite computation into a rigorous two-sided bracket.
-/
import Mathlib
import RequestProject.LatticeWalk
import RequestProject.SelfAvoiding
import RequestProject.ConnectiveExact

namespace IDR.SAW

open scoped BigOperators

/-! ## Endpoints -/

section Endpoint

variable {V : Type*} [AddCommGroup V]

lemma foldl_add_left (x : V) (l : List V) :
    List.foldl (· + ·) x l = x + endpoint l := by
  induction l generalizing x with
  | nil => simp [endpoint]
  | cons d t ih =>
      have h1 : endpoint (d :: t) = d + endpoint t := by
        show List.foldl (· + ·) 0 (d :: t) = d + endpoint t
        rw [List.foldl_cons, ih (0 + d), zero_add]
      rw [List.foldl_cons, ih (x + d), h1]
      abel

lemma endpoint_append (l₁ l₂ : List V) :
    endpoint (l₁ ++ l₂) = endpoint l₁ + endpoint l₂ := by
  rw [endpoint, List.foldl_append, foldl_add_left, ← endpoint]

lemma zero_mem_sites (l : List V) : (0 : V) ∈ sites l := by
  obtain ⟨t, ht⟩ := sitesFrom_head 0 l
  rw [sites, ht]
  exact List.mem_cons_self

lemma sites_append (l₁ l₂ : List V) :
    sites (l₁ ++ l₂) = sites l₁ ++ ((sites l₂).map fun y => endpoint l₁ + y).tail := by
  rw [← sitesFrom_eq_map]
  simp [sites, sitesFrom, List.scanl_append, endpoint]

/-- In a self-avoiding chain the origin is visited only at the start. -/
lemma ne_zero_of_mem_tail_sites {l : List V} (h : IsSAW l) {q : V}
    (hq : q ∈ (sites l).tail) : q ≠ 0 := by
  obtain ⟨t, ht⟩ := sitesFrom_head 0 l
  rw [IsSAW, sites, ht] at h
  rw [sites, ht] at hq
  simp only [List.tail_cons] at hq
  rintro rfl
  exact (List.nodup_cons.1 h).1 hq

end Endpoint

/-! ## Bridges -/

/-- A **bridge**: a chain that leaves the line `x = 0` at once and never passes to the right
of its own endpoint.  (Self-avoidance is imposed separately.) -/
def IsBridge (l : List Site) : Prop :=
  (∀ p ∈ sites l, p = 0 ∨ 0 < p.1) ∧ (∀ p ∈ sites l, p.1 ≤ (endpoint l).1)

instance (l : List Site) : Decidable (IsBridge l) := by unfold IsBridge; infer_instance

lemma IsBridge.endpoint_nonneg {l : List Site} (h : IsBridge l) : 0 ≤ (endpoint l).1 := by
  simpa using h.2 0 (zero_mem_sites l)

/-- **Two bridges concatenate to a bridge, and the result is still self-avoiding.**  This is
the geometric heart of the lower bound: the second bridge lies strictly to the right of the
vertical line through the junction, while the first lies weakly to its left. -/
theorem bridge_append {l₁ l₂ : List Site} (h1 : IsBridge l₁) (s1 : IsSAW l₁)
    (h2 : IsBridge l₂) (s2 : IsSAW l₂) :
    IsBridge (l₁ ++ l₂) ∧ IsSAW (l₁ ++ l₂) := by
  set X := endpoint l₁ with hXdef
  set Y := endpoint l₂ with hYdef
  have hX0 : 0 ≤ X.1 := h1.endpoint_nonneg
  have hY0 : 0 ≤ Y.1 := h2.endpoint_nonneg
  have hsplit : sites (l₁ ++ l₂) = sites l₁ ++ ((sites l₂).map fun y => X + y).tail :=
    sites_append l₁ l₂
  have htailmap : ((sites l₂).map fun y => X + y).tail
      = ((sites l₂).tail).map fun y => X + y := List.map_tail.symm
  -- description of the right-hand block
  have hright : ∀ p ∈ ((sites l₂).map fun y => X + y).tail,
      X.1 < p.1 ∧ p.1 ≤ X.1 + Y.1 := by
    intro p hp
    rw [htailmap] at hp
    obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hp
    have hq0 : q ≠ 0 := ne_zero_of_mem_tail_sites s2 hq
    have hqmem : q ∈ sites l₂ := List.mem_of_mem_tail hq
    have hqpos : 0 < q.1 := (h2.1 q hqmem).resolve_left hq0
    have hqle : q.1 ≤ Y.1 := h2.2 q hqmem
    constructor
    · simp only [Prod.fst_add]; linarith
    · simp only [Prod.fst_add]; linarith
  have hleft : ∀ p ∈ sites l₁, p.1 ≤ X.1 := h1.2
  have hend : endpoint (l₁ ++ l₂) = X + Y := endpoint_append l₁ l₂
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro p hp
    rw [hsplit, List.mem_append] at hp
    rcases hp with hp | hp
    · exact h1.1 p hp
    · exact Or.inr (lt_of_le_of_lt hX0 (hright p hp).1)
  · intro p hp
    rw [hend]
    rw [hsplit, List.mem_append] at hp
    rcases hp with hp | hp
    · have := hleft p hp
      simp only [Prod.fst_add]; linarith
    · have := (hright p hp).2
      simp only [Prod.fst_add]; linarith
  · rw [IsSAW, hsplit, List.nodup_append]
    refine ⟨s1, ?_, ?_⟩
    · rw [htailmap]
      refine List.Nodup.map (fun a b hab => by simpa using hab) ?_
      exact List.Nodup.sublist (List.tail_sublist _) s2
    · intro a ha b hb hab
      subst hab
      have h1' := hleft a ha
      have h2' := (hright a hb).1
      linarith

/-! ## Counting bridges -/

/-- The self-avoiding bridges of a chain of `n` bonds. -/
def bridgeFinset (n : ℕ) : Finset (Fin n → Fin 4) :=
  Finset.univ.filter fun w => IsBridge (stepsOf w) ∧ IsSAW (stepsOf w)

/-- The number of self-avoiding bridges of a chain of `n` bonds. -/
def bcnt (n : ℕ) : ℕ := (bridgeFinset n).card

@[simp] lemma mem_bridgeFinset {n : ℕ} (w : Fin n → Fin 4) :
    w ∈ bridgeFinset n ↔ IsBridge (stepsOf w) ∧ IsSAW (stepsOf w) := by
  simp [bridgeFinset]

lemma bcnt_le_cnt (n : ℕ) : bcnt n ≤ cnt n := by
  refine Finset.card_le_card ?_
  intro w hw
  rw [mem_bridgeFinset] at hw
  exact (mem_sawFinsetOf dir w).2 hw.2

lemma stepsOf_addCases {m n : ℕ} (u : Fin m → Fin 4) (v : Fin n → Fin 4) :
    stepsOf (Fin.addCases u v : Fin (m + n) → Fin 4) = stepsOf u ++ stepsOf v := by
  rw [stepsOf, stepsOfDir_add]
  congr 1
  · congr 1
    funext i
    exact Fin.addCases_left i
  · congr 1
    funext i
    exact Fin.addCases_right i

/-- **The bridge count is supermultiplicative** — the mirror image of the submultiplicativity
of the full self-avoiding count. -/
theorem bcnt_supermultiplicative (m n : ℕ) : bcnt m * bcnt n ≤ bcnt (m + n) := by
  classical
  have hcard : (bridgeFinset m ×ˢ bridgeFinset n).card = bcnt m * bcnt n := by
    simp [bcnt, Finset.card_product]
  rw [← hcard, bcnt]
  refine Finset.card_le_card_of_injOn
    (fun uv => (Fin.addCases uv.1 uv.2 : Fin (m + n) → Fin 4)) ?_ ?_
  · rintro ⟨u, v⟩ huv
    rw [Finset.mem_coe, Finset.mem_product, mem_bridgeFinset, mem_bridgeFinset] at huv
    simp only [Finset.mem_coe, mem_bridgeFinset, stepsOf_addCases]
    exact bridge_append huv.1.1 huv.1.2 huv.2.1 huv.2.2
  · rintro ⟨u, v⟩ _ ⟨u', v'⟩ _ h
    have hu : u = u' := by
      funext i
      have := congrFun h (Fin.castAdd n i)
      simpa [Fin.addCases_left] using this
    have hv : v = v' := by
      funext i
      have := congrFun h (Fin.natAdd m i)
      simpa [Fin.addCases_right] using this
    simp [hu, hv]

lemma bcnt_pow_le (m k : ℕ) : bcnt m ^ k ≤ bcnt (m * k) := by
  induction k with
  | zero =>
      simp only [pow_zero, Nat.mul_zero]
      decide
  | succ j ih =>
      have hstep : bcnt (m * j) * bcnt m ≤ bcnt (m * (j + 1)) := by
        have := bcnt_supermultiplicative (m * j) m
        calc bcnt (m * j) * bcnt m ≤ bcnt (m * j + m) := this
          _ = bcnt (m * (j + 1)) := by ring_nf
      calc bcnt m ^ (j + 1) = bcnt m ^ j * bcnt m := by ring
        _ ≤ bcnt (m * j) * bcnt m := Nat.mul_le_mul_right _ ih
        _ ≤ bcnt (m * (j + 1)) := hstep

/-- **Every bridge count gives a lower bound on the connective constant**, dual to
`connectiveConstant_le_div`. -/
theorem log_bcnt_div_le_connectiveConstant {m : ℕ} (hm : 0 < m) :
    Real.log (bcnt m) / m ≤ connectiveConstant := by
  rcases Nat.eq_zero_or_pos (bcnt m) with hb | hb
  · rw [hb]
    simp only [Nat.cast_zero, Real.log_zero, zero_div]
    exact le_trans (Real.log_nonneg (by norm_num)) log_two_le_connectiveConstant
  have hsub : Filter.Tendsto (fun k : ℕ => m * k) Filter.atTop Filter.atTop := by
    refine Filter.tendsto_atTop_atTop.2 fun b => ⟨b, fun a ha => le_trans ha ?_⟩
    exact Nat.le_mul_of_pos_left a hm
  have htend := (tendsto_connectiveConstant).comp hsub
  refine ge_of_tendsto htend ?_
  filter_upwards [Filter.eventually_gt_atTop 0] with k hk
  have hmk : 0 < m * k := Nat.mul_pos hm hk
  have hle : ((bcnt m : ℝ)) ^ k ≤ (cnt (m * k) : ℝ) := by
    have h1 : bcnt m ^ k ≤ cnt (m * k) := le_trans (bcnt_pow_le m k) (bcnt_le_cnt _)
    exact_mod_cast h1
  have hlog : (k : ℝ) * Real.log (bcnt m) ≤ logCnt (m * k) := by
    have hbpos : (0 : ℝ) < (bcnt m : ℝ) := by exact_mod_cast hb
    have := Real.log_le_log (by positivity) hle
    rwa [Real.log_pow] at this
  have hmk' : (0 : ℝ) < ((m * k : ℕ) : ℝ) := by exact_mod_cast hmk
  have hm' : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
  have hk' : (0 : ℝ) < (k : ℝ) := by exact_mod_cast hk
  simp only [Function.comp_apply]
  rw [div_le_div_iff₀ hm' hmk']
  push_cast
  push_cast at hlog
  nlinarith [hlog]

/-! ## The exact six-bond bridge count, and the resulting bracket -/

set_option maxRecDepth 100000 in
set_option maxHeartbeats 4000000 in
/-- There are exactly `101` self-avoiding bridges of six bonds on the square lattice. -/
theorem bcnt_six : bcnt 6 = 101 := by decide

set_option maxRecDepth 200000 in
set_option maxHeartbeats 20000000 in
/-- There are exactly `251` self-avoiding bridges of seven bonds on the square lattice. -/
theorem bcnt_seven : bcnt 7 = 251 := by decide

/-- The six-bond bridge bound on the connective constant of `ℤ²`. -/
theorem log101_div6_le_connectiveConstant : Real.log 101 / 6 ≤ connectiveConstant := by
  have h := log_bcnt_div_le_connectiveConstant (m := 6) (by norm_num)
  rwa [bcnt_six] at h

/-- **Improved lower bound on the connective constant of `ℤ²`**, from the seven-bond bridge
count. -/
theorem log251_div7_le_connectiveConstant : Real.log 251 / 7 ≤ connectiveConstant := by
  have h := log_bcnt_div_le_connectiveConstant (m := 7) (by norm_num)
  rwa [bcnt_seven] at h

/-- The bridge bound strictly improves the directed-walk bound `log 2`. -/
theorem improves_on_log_two : Real.log 2 < Real.log 251 / 7 := by
  have h1 : (2 : ℝ) ^ 7 < (251 : ℝ) := by norm_num
  have := Real.log_lt_log (by positivity) h1
  rw [Real.log_pow] at this
  push_cast at this
  linarith

/-- **The two-sided rigorous bracket for the connective constant of the square lattice**:
`log 251 / 7 ≤ μ ≤ log 780 / 6`, i.e. `0.789… ≤ μ ≤ 1.110…`.  Both bounds strictly improve the
previously available pair `log 2 ≤ μ ≤ log 100 / 4`, and the two constructions
(`bcnt_supermultiplicative`, `cnt_submultiplicative`) turn any further finite computation into
a sharper bracket.  The exact value of `μ` is not determined by any finite computation and
remains an open problem. -/
theorem connectiveConstant_bracket :
    Real.log 251 / 7 ≤ connectiveConstant ∧ connectiveConstant ≤ Real.log 780 / 6 :=
  ⟨log251_div7_le_connectiveConstant, connectiveConstant_le_log780_div6⟩

end IDR.SAW
