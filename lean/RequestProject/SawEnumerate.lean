/-
# Part CXVIII  A verified depth-first enumerator, and a sharper upper bound on `μ`

The upper bounds on the connective constant proved so far come from exact conformation counts
`cnt n` obtained by *exhaustive* enumeration of all `4 ^ n` bond sequences, which stops being
feasible at `n = 7`.  Since `μ ≤ (log cnt n) / n` for every `n`, the bound improves with every
further term — but only if the terms can be computed.

This file supplies the missing machinery: a depth-first enumerator with pruning,

  `contCount visited cur k` — the number of `k`-bond self-avoiding continuations of a chain
  whose end is at `cur` and which already occupies the sites `cur :: visited`,

together with a proof that it counts exactly what it should (`contCount_eq`), and hence that
`cnt n = contCount [] 0 n` (`cnt_eq_contCount`).  The enumerator visits one node per
self-avoiding *prefix* instead of one per bond sequence, which is what makes the further terms
computable inside the kernel.

Results:

* `contCount_eq`, `cnt_eq_contCount` — correctness of the enumerator.
* `cnt_eight`, `cnt_ten` — the exact conformation counts `5916` and `44100`, neither of which
  was previously available.
* `connectiveConstant_le_log44100_div10` — `μ ≤ log 44100 / 10 = 1.0694…`, improving every
  earlier upper bound, and `connectiveConstant_bracket_sharp` — the bracket

    `log (1 + √2) ≤ μ ≤ log 44100 / 10`,   i.e.   `2.4142… ≤ e^μ ≤ 2.9137…`,

  with the lower end from the exactly solved partially directed chain of Part CXVI.

The exact value of `μ` remains open; what this part adds is that each further finite computation
now genuinely narrows the machine-checked bracket.
-/
import Mathlib
import RequestProject.LatticeWalk
import RequestProject.SelfAvoiding
import RequestProject.PartiallyDirected

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

namespace IDR.SAW

open scoped BigOperators

/-- The depth-first count of `k`-bond self-avoiding continuations of a chain ending at `cur`
whose earlier sites are `visited` (`cur` itself excluded from that list). -/
def contCount (visited : List Site) (cur : Site) : ℕ → ℕ
  | 0 => 1
  | (k + 1) =>
      (if cur + (1, 0) ∈ cur :: visited then 0
        else contCount (cur :: visited) (cur + (1, 0)) k) +
      (if cur + (-1, 0) ∈ cur :: visited then 0
        else contCount (cur :: visited) (cur + (-1, 0)) k) +
      (if cur + (0, 1) ∈ cur :: visited then 0
        else contCount (cur :: visited) (cur + (0, 1)) k) +
      (if cur + (0, -1) ∈ cur :: visited then 0
        else contCount (cur :: visited) (cur + (0, -1)) k)

/-- The enumerator, as a sum over the four bond directions. -/
lemma contCount_succ (visited : List Site) (cur : Site) (k : ℕ) :
    contCount visited cur (k + 1)
      = ∑ i : Fin 4, if (cur + dir i) ∈ cur :: visited then 0
          else contCount (cur :: visited) (cur + dir i) k := by
  rw [Fin.sum_univ_four]
  rfl

/-- The conformations counted by `contCount`: those whose sites are distinct and avoid the
sites already occupied. -/
def GoodCont (visited : List Site) (cur : Site) {k : ℕ} (w : Fin k → Fin 4) : Prop :=
  (sitesFrom cur (stepsOfDir dir w)).Nodup ∧
    ∀ p ∈ (sitesFrom cur (stepsOfDir dir w)).tail, p ∉ visited

instance (visited : List Site) (cur : Site) (k : ℕ) :
    DecidablePred (GoodCont visited cur (k := k)) := fun w => by
  unfold GoodCont; infer_instance

/-- Splitting a count over the first bond. -/
lemma card_filter_succ {k : ℕ} (P : (Fin (k + 1) → Fin 4) → Prop) [DecidablePred P] :
    (Finset.univ.filter P).card
      = ∑ i : Fin 4, (Finset.univ.filter (fun v : Fin k → Fin 4 => P (Fin.cons i v))).card := by
  classical
  have h1 : (Finset.univ.filter P).card = ∑ w : Fin (k + 1) → Fin 4, if P w then 1 else 0 :=
    Finset.card_filter _ _
  have h2 : ∑ w : Fin (k + 1) → Fin 4, (if P w then 1 else 0)
      = ∑ x : Fin 4 × (Fin k → Fin 4), if P (Fin.cons x.1 x.2) then 1 else 0 := by
    refine (Fintype.sum_equiv (Fin.consEquiv (fun _ : Fin (k + 1) => Fin 4)) _ _ ?_).symm
    intro x
    simp [Fin.consEquiv]
  rw [h1, h2, Fintype.sum_prod_type]
  exact Finset.sum_congr rfl fun i _ => (Finset.card_filter _ _).symm

lemma stepsOfDir_cons {k : ℕ} (i : Fin 4) (v : Fin k → Fin 4) :
    stepsOfDir dir (Fin.cons i v : Fin (k + 1) → Fin 4) = dir i :: stepsOfDir dir v := by
  simp [stepsOfDir, List.ofFn_succ]

/-- The recursive characterisation of an admissible continuation. -/
lemma goodCont_cons {k : ℕ} (visited : List Site) (cur : Site) (i : Fin 4)
    (v : Fin k → Fin 4) :
    GoodCont visited cur (Fin.cons i v) ↔
      (cur + dir i) ∉ cur :: visited ∧ GoodCont (cur :: visited) (cur + dir i) v := by
  classical
  set p := cur + dir i with hp
  set S := sitesFrom p (stepsOfDir dir v) with hS
  have hsplit : sitesFrom cur (stepsOfDir dir (Fin.cons i v : Fin (k + 1) → Fin 4))
      = cur :: S := by
    rw [stepsOfDir_cons, sitesFrom_cons, hS, hp]
  obtain ⟨t, ht⟩ := sitesFrom_head p (stepsOfDir dir v)
  have hStail : S.tail = t := by rw [hS, ht]; rfl
  have hShead : S = p :: t := by rw [hS, ht]
  constructor
  · rintro ⟨hnodup, havoid⟩
    rw [hsplit] at hnodup havoid
    rw [List.nodup_cons] at hnodup
    have hcurS : cur ∉ S := hnodup.1
    have hSnodup : S.Nodup := hnodup.2
    have hSavoid : ∀ q ∈ S, q ∉ visited := by
      intro q hq
      exact havoid q (by simpa using hq)
    refine ⟨?_, hSnodup, ?_⟩
    · intro hmem
      rcases List.mem_cons.1 hmem with h | h
      · have hpS : p ∈ S := by rw [hShead]; simp
        exact hcurS (h ▸ hpS)
      · exact hSavoid p (by rw [hShead]; simp) h
    · intro q hq hq2
      have hqS : q ∈ S := by rw [hShead]; exact List.mem_cons_of_mem _ (by rwa [hStail] at hq)
      rcases List.mem_cons.1 hq2 with h | h
      · exact hcurS (h ▸ hqS)
      · exact hSavoid q hqS h
  · rintro ⟨hpnot, hSnodup, hSavoid⟩
    have hpvis : p ∉ visited := fun h => hpnot (List.mem_cons_of_mem _ h)
    have hpcur : p ≠ cur := fun h => hpnot (by rw [h]; simp)
    have hcurS : cur ∉ S := by
      rw [hShead]
      intro hmem
      rcases List.mem_cons.1 hmem with h | h
      · exact hpcur h.symm
      · exact (hSavoid cur (by rwa [hStail])) (by simp)
    refine ⟨?_, ?_⟩
    · rw [hsplit, List.nodup_cons]
      exact ⟨hcurS, hSnodup⟩
    · intro q hq
      rw [hsplit] at hq
      simp only [List.tail_cons] at hq
      rw [hShead] at hq
      rcases List.mem_cons.1 hq with h | h
      · rw [h]; exact hpvis
      · have := hSavoid q (by rwa [hStail])
        exact fun hv => this (List.mem_cons_of_mem _ hv)

/-- **The enumerator is correct**: `contCount` counts exactly the self-avoiding continuations. -/
theorem contCount_eq : ∀ (k : ℕ) (visited : List Site) (cur : Site),
    contCount visited cur k
      = (Finset.univ.filter (GoodCont visited cur (k := k))).card := by
  intro k
  induction k with
  | zero =>
      intro visited cur
      have h : ∀ w : Fin 0 → Fin 4, GoodCont visited cur w := by
        intro w
        refine ⟨?_, ?_⟩ <;> simp [stepsOfDir, sitesFrom]
      simp [contCount, Finset.filter_true_of_mem (fun w _ => h w)]
  | succ k ih =>
      intro visited cur
      rw [contCount_succ, card_filter_succ]
      refine Finset.sum_congr rfl fun i _ => ?_
      by_cases hmem : (cur + dir i) ∈ cur :: visited
      · rw [if_pos hmem]
        symm
        rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
        intro v _
        rw [goodCont_cons]
        exact fun h => h.1 hmem
      · rw [if_neg hmem, ih (cur :: visited) (cur + dir i)]
        congr 1
        apply Finset.filter_congr
        intro v _
        rw [goodCont_cons]
        simp [hmem]

/-- The conformation count is the depth-first count from the origin. -/
theorem cnt_eq_contCount (n : ℕ) : cnt n = contCount [] 0 n := by
  classical
  rw [contCount_eq]
  have hfilter : (Finset.univ.filter (fun w : Fin n → Fin 4 => IsSAW (stepsOfDir dir w)))
      = (Finset.univ.filter (GoodCont [] (0 : Site) (k := n))) := by
    apply Finset.filter_congr
    intro w _
    simp [GoodCont, IsSAW, sites]
  rw [cnt, cntOf, sawFinsetOf]
  exact congrArg Finset.card hfilter

/-! ## Further exact conformation counts -/

/-- The exact count of eight-bond conformations. -/
theorem cnt_eight : cnt 8 = 5916 := by rw [cnt_eq_contCount]; decide +kernel

/-- The exact count of ten-bond conformations. -/
theorem cnt_ten : cnt 10 = 44100 := by rw [cnt_eq_contCount]; decide +kernel

/-! ## The sharper bound -/

/-- **A sharper rigorous upper bound on the connective constant**, from the exact count of
ten-bond conformations. -/
theorem connectiveConstant_le_log44100_div10 : connectiveConstant ≤ Real.log 44100 / 10 := by
  have h := connectiveConstant_le_div (n := 10) (by norm_num)
  have hval : logCnt 10 = Real.log 44100 := by
    rw [logCnt, logCntOf, show cntOf dir 10 = 44100 from cnt_ten]
    norm_num
  rwa [hval] at h

/-- The new upper bound improves the seven-bond bound of Part XXXIX. -/
theorem improves_on_cnt_seven : Real.log 44100 / 10 < Real.log 2172 / 7 := by
  have h1 : (0:ℝ) < 44100 := by norm_num
  have hlt : Real.log (44100 ^ (7:ℕ)) < Real.log (2172 ^ (10:ℕ)) := by
    refine Real.log_lt_log (by positivity) ?_
    norm_num
  rw [Real.log_pow, Real.log_pow] at hlt
  rw [div_lt_div_iff₀ (by norm_num) (by norm_num)]
  push_cast at hlt
  linarith

/-- **The sharpened bracket** for the connective constant of the square lattice:
`log (1 + √2) ≤ μ ≤ log 44100 / 10`, i.e. `2.4142… ≤ e^μ ≤ 2.9137…`. -/
theorem connectiveConstant_bracket_sharp :
    Real.log (1 + Real.sqrt 2) ≤ connectiveConstant ∧
      connectiveConstant ≤ Real.log 44100 / 10 :=
  ⟨PD.log_one_add_sqrt_two_le_connectiveConstant, connectiveConstant_le_log44100_div10⟩

end IDR.SAW
