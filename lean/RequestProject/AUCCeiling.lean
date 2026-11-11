/-
# Theorem 2 — the ceiling of the per-protein bias

What is `max_{b ∈ ℝ^K} AUC_pooled(s + b)`?  This file answers the four questions asked of it:
attainment, structure, a closed-form upper bound, and when that bound is reached.

**Attainment.**  `exists_max_bias`, `exists_max_bias_auc`: the supremum is always *attained*, for
every dataset and every score field.  The reason is not compactness — `ℝ^K` is not compact and the
objective is not continuous — but finiteness: `s + b` enters the statistic only through the finite
comparison pattern it induces on the pairs, so the objective takes finitely many values.

**Structure.**  `U_shift_eq`: the within-protein part is frozen (Theorem 1), so the optimisation is
purely over the cross-protein pairs.

**The ceiling.**  `U_shift_le_ceiling`, `auc_shift_le_ceiling`:
`AUC_pooled(s+b) ≤ w_within·AUC_within(s) + w_between` for every `b`, a closed form computable
from the unbiased predictor alone.  This is the *only* closed form available in general: the
optimum itself is the optimum of a linear ordering problem (see the encoding lemmas
`exists_order_ge`, `exists_bias_of_order` at the end of the file).

**When the ceiling is reached.**  `U_shift_eq_ceiling_iff`, `ceiling_attainable_iff`: exactly when
the system of difference constraints
`b k − b l > s n − s p` for every cross-protein pair `(p, n)` with `p` in protein `k`, `n` in
protein `l`
is feasible — a system of difference constraints, decidable in `O(K³)` by negative-cycle
detection, *not* an exponential search.  `cycle_obstruction` is the exact obstruction: every
cyclic chain of cross-protein pairs must have strictly negative total gap.  Its shortest instance
is a *crossed pair* (`crossed_pair_le_one`, `ceiling_gap_of_crossed`): two proteins `k`, `l` and
comparisons `p_k` vs `n_l`, `p_l` vs `n_k` with `(s n_l − s p_k) + (s n_k − s p_l) ≥ 0` can never
both be got right, whatever the bias, so each such disjoint configuration lowers the reachable
pooled AUC by a full pair — a bound computable in `O(n²)` from the score table.

**Complexity.**  `exists_order_ge` and `exists_bias_of_order` identify the well-separated regime of
the problem with the *linear ordering problem*: maximising `Σ_{k≠l} w_{k l}·1[b k − b l > 1]` over
`b ∈ ℝ^K` has the same optimum as maximising `Σ_{π l < π k} w_{k l}` over permutations `π`: a
combinatorial optimisation over orderings of the proteins, not a smooth optimisation over `ℝ^K`.
The converse — that every weighted linear ordering instance is realised by an explicit score table,
so that the recalibration optimum is at least as hard as linear ordering — is proved in
`AUCHardness.lean` (`bias_optimum_eq_lop`, `lop_reduces_to_bias_optimum`).
-/
import RequestProject.AUCInvariance

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section Ceiling

variable {I G : Type*} [Fintype I] [DecidableEq G]

/-- After a per-protein bias, only the cross-protein part of the statistic has moved. -/
theorem U_shift_eq (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ) :
    U (allPairs lab) (shift grp b s)
      = U (withinPairs lab grp) s + U (betweenPairs lab grp) (shift grp b s) := by
  rw [U_split lab grp, U_within_shift]

/-! ### Attainment -/

omit [DecidableEq G] in
/-- **The supremum over per-protein biases is attained.**  The objective takes only finitely many
values, because the bias enters only through the comparison pattern it induces. -/
theorem exists_max_bias (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    ∃ b : G → ℝ, ∀ b' : G → ℝ,
      U (allPairs lab) (shift grp b' s) ≤ U (allPairs lab) (shift grp b s) := by
  classical
  set F : (G → ℝ) → ℝ := fun b => U (allPairs lab) (shift grp b s) with hFdef
  set P : (G → ℝ) → (I × I → Bool × Bool) := fun b q =>
    (decide (s q.2 + b (grp q.2) < s q.1 + b (grp q.1)),
      decide (s q.1 + b (grp q.1) = s q.2 + b (grp q.2))) with hPdef
  set Phi : (I × I → Bool × Bool) → ℝ := fun f =>
    ∑ q ∈ allPairs lab, (if (f q).1 = true then (1 : ℝ) else if (f q).2 = true then 1 / 2 else 0)
    with hPhidef
  have hFPhi : ∀ b, F b = Phi (P b) := by
    intro b
    refine Finset.sum_congr rfl fun q _ => ?_
    simp only [hPdef, kern, shift, decide_eq_true_eq]
  have hsub : Set.range F ⊆ Set.range Phi := by
    rintro v ⟨b, rfl⟩
    exact ⟨P b, (hFPhi b).symm⟩
  have hfin : (Set.range F).Finite := Set.Finite.subset (Set.finite_range Phi) hsub
  have hne : hfin.toFinset.Nonempty := by
    refine ⟨F (fun _ => 0), ?_⟩
    simp [Set.Finite.mem_toFinset]
  obtain ⟨v, hv, hmax⟩ := hfin.toFinset.exists_max_image id hne
  rw [Set.Finite.mem_toFinset] at hv
  obtain ⟨b, hb⟩ := hv
  refine ⟨b, fun b' => ?_⟩
  have : F b' ∈ hfin.toFinset := by
    rw [Set.Finite.mem_toFinset]; exact Set.mem_range_self b'
  have hle := hmax _ this
  calc U (allPairs lab) (shift grp b' s) = F b' := rfl
    _ ≤ v := hle
    _ = U (allPairs lab) (shift grp b s) := hb.symm

omit [DecidableEq G] in
/-- The same statement for the AUC itself. -/
theorem exists_max_bias_auc (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    ∃ b : G → ℝ, ∀ b' : G → ℝ,
      auc (allPairs lab) (shift grp b' s) ≤ auc (allPairs lab) (shift grp b s) := by
  obtain ⟨b, hb⟩ := exists_max_bias lab grp s
  refine ⟨b, fun b' => ?_⟩
  unfold auc
  have hc : (0:ℝ) ≤ ((allPairs lab).card : ℝ) := by positivity
  exact div_le_div_of_nonneg_right (hb b') hc

/-! ### The ceiling -/

/-- **The ceiling.**  However the per-protein biases are chosen, the pooled statistic cannot exceed
the frozen within-protein statistic plus one for every cross-protein pair. -/
theorem U_shift_le_ceiling (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ) :
    U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card := by
  rw [U_shift_eq]
  linarith [U_le_card (betweenPairs lab grp) (shift grp b s)]

/-- The ceiling in AUC form: `w_within · AUC_within(s) + w_between`. -/
theorem auc_shift_le_ceiling (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (hw : (withinPairs lab grp).card ≠ 0) (hb : (betweenPairs lab grp).card ≠ 0) :
    auc (allPairs lab) (shift grp b s)
      ≤ ((withinPairs lab grp).card / (allPairs lab).card) * auc (withinPairs lab grp) s
        + ((betweenPairs lab grp).card / (allPairs lab).card) := by
  have hac : (0 : ℝ) < ((allPairs lab).card : ℝ) := by
    have : 0 < (allPairs lab).card :=
      lt_of_lt_of_le (Nat.pos_of_ne_zero hb) (Finset.card_filter_le _ _)
    exact_mod_cast this
  have h := U_shift_le_ceiling lab grp s b
  unfold auc
  rw [div_le_iff₀ hac]
  have hwc : ((withinPairs lab grp).card : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hw
  calc U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card := h
    _ = (((withinPairs lab grp).card : ℝ) / (allPairs lab).card
          * (U (withinPairs lab grp) s / (withinPairs lab grp).card)
        + ((betweenPairs lab grp).card : ℝ) / (allPairs lab).card) * (allPairs lab).card := by
        field_simp

/-- The bias reaches the ceiling exactly when every cross-protein comparison comes out right. -/
theorem U_shift_eq_ceiling_iff (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ) :
    U (allPairs lab) (shift grp b s) = U (withinPairs lab grp) s + (betweenPairs lab grp).card
      ↔ ∀ q ∈ betweenPairs lab grp, s q.2 - s q.1 < b (grp q.1) - b (grp q.2) := by
  rw [U_shift_eq]
  constructor
  · intro h q hq
    have hsum : U (betweenPairs lab grp) (shift grp b s) = (betweenPairs lab grp).card := by
      linarith
    have hall : ∀ r ∈ betweenPairs lab grp, kern (shift grp b s r.1) (shift grp b s r.2) = 1 := by
      by_contra hcon
      push_neg at hcon
      obtain ⟨r, hr, hrne⟩ := hcon
      have hlt : kern (shift grp b s r.1) (shift grp b s r.2) < 1 :=
        lt_of_le_of_ne (kern_le_one _ _) hrne
      have : U (betweenPairs lab grp) (shift grp b s) < (betweenPairs lab grp).card := by
        unfold U
        calc ∑ x ∈ betweenPairs lab grp, kern (shift grp b s x.1) (shift grp b s x.2)
            < ∑ _x ∈ betweenPairs lab grp, (1 : ℝ) :=
              Finset.sum_lt_sum (fun x _ => kern_le_one _ _) ⟨r, hr, hlt⟩
          _ = (betweenPairs lab grp).card := by simp
      linarith
    have := kern_eq_one_iff.mp (hall q hq)
    simp only [shift] at this
    linarith
  · intro h
    have hall : ∀ r ∈ betweenPairs lab grp, kern (shift grp b s r.1) (shift grp b s r.2) = 1 := by
      intro r hr
      apply kern_eq_one_of_lt
      have := h r hr
      simp only [shift]
      linarith
    have : U (betweenPairs lab grp) (shift grp b s) = (betweenPairs lab grp).card := by
      unfold U
      rw [Finset.sum_congr rfl hall]
      simp
    rw [this]

/-- **The ceiling is attainable exactly when a system of difference constraints is feasible.**
Feasibility of such a system is decided in `O(K³)` by negative-cycle detection. -/
theorem ceiling_attainable_iff (lab : I → Bool) (grp : I → G) (s : I → ℝ) :
    (∃ b : G → ℝ, U (allPairs lab) (shift grp b s)
        = U (withinPairs lab grp) s + (betweenPairs lab grp).card)
      ↔ ∃ b : G → ℝ, ∀ q ∈ betweenPairs lab grp,
          s q.2 - s q.1 < b (grp q.1) - b (grp q.2) := by
  constructor
  · rintro ⟨b, hb⟩
    exact ⟨b, (U_shift_eq_ceiling_iff lab grp s b).mp hb⟩
  · rintro ⟨b, hb⟩
    exact ⟨b, (U_shift_eq_ceiling_iff lab grp s b).mpr hb⟩

/-! ### The obstruction: cyclic chains of cross-protein pairs -/

/-- **The exact obstruction.**  If the ceiling is attainable then every cyclic chain of
cross-protein comparisons — pair `i` compares a positive of protein `k i` with a negative of
protein `k (i+1)` — has strictly negative total gap.  A single violated cycle certifies, in
`O(K³)` time, that the ceiling is out of reach. -/
theorem cycle_obstruction (lab : I → Bool) (grp : I → G) (s : I → ℝ) (m : ℕ)
    (q : Fin (m + 1) → I × I) (hq : ∀ i, q i ∈ betweenPairs lab grp)
    (hchain : ∀ i : Fin (m + 1), grp (q (i + 1)).1 = grp (q i).2)
    (hfeas : ∃ b : G → ℝ, ∀ r ∈ betweenPairs lab grp,
      s r.2 - s r.1 < b (grp r.1) - b (grp r.2)) :
    ∑ i, (s (q i).2 - s (q i).1) < 0 := by
  obtain ⟨b, hb⟩ := hfeas
  have hstep : ∀ i, s (q i).2 - s (q i).1 < b (grp (q i).1) - b (grp (q i).2) :=
    fun i => hb _ (hq i)
  have hlt : ∑ i, (s (q i).2 - s (q i).1)
      < ∑ i, (b (grp (q i).1) - b (grp (q i).2)) :=
    Finset.sum_lt_sum_of_nonempty ⟨0, Finset.mem_univ _⟩ (fun i _ => hstep i)
  have htel : ∑ i : Fin (m + 1), (b (grp (q i).1) - b (grp (q i).2)) = 0 := by
    have h1 : ∑ i : Fin (m + 1), b (grp (q i).2) = ∑ i : Fin (m + 1), b (grp (q i).1) := by
      have : ∀ i : Fin (m + 1), b (grp (q i).2) = b (grp (q (i + 1)).1) := by
        intro i; rw [hchain i]
      rw [Finset.sum_congr rfl (fun i _ => this i)]
      exact Fintype.sum_equiv (Equiv.addRight (1 : Fin (m + 1))) _ _ (fun i => rfl)
    rw [Finset.sum_sub_distrib, h1, sub_self]
  linarith [hlt, htel]

/-- A *crossed pair*: two cross-protein comparisons in opposite directions whose gaps do not leave
room for any bias.  At most one of them can be got right, whatever the bias — the two comparisons
are worth at most one pair together. -/
theorem crossed_pair_le_one (bk bl x y x' y' : ℝ) (hcross : 0 ≤ (y - x) + (y' - x')) :
    kern (x + bk) (y + bl) + kern (x' + bl) (y' + bk) ≤ 1 := by
  unfold kern
  split_ifs with h1 h2 h3 h4 h3 h4 <;> norm_num <;> linarith

end Ceiling

section Crossed

variable {I G : Type*} [Fintype I] [DecidableEq I] [DecidableEq G]

/-- **Each crossed configuration costs a full pair.**  If two distinct cross-protein comparisons
`q`, `q'` run between the same two proteins in opposite directions and their gaps sum to a
non-negative number, then no bias can bring the pooled statistic within one pair of the ceiling. -/
theorem ceiling_gap_of_crossed (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (q q' : I × I) (hq : q ∈ betweenPairs lab grp) (hq' : q' ∈ betweenPairs lab grp)
    (hne : q ≠ q')
    (hgrp1 : grp q.1 = grp q'.2) (hgrp2 : grp q.2 = grp q'.1)
    (hcross : 0 ≤ (s q.2 - s q.1) + (s q'.2 - s q'.1)) :
    U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card - 1 := by
  classical
  rw [U_shift_eq]
  have hpair : ({q, q'} : Finset (I × I)) ⊆ betweenPairs lab grp := by
    intro r hr
    rcases Finset.mem_insert.mp hr with h | h
    · exact h ▸ hq
    · exact (Finset.mem_singleton.mp h) ▸ hq'
  have hcard2 : ({q, q'} : Finset (I × I)).card = 2 := by
    rw [Finset.card_insert_of_notMem (by simpa using hne), Finset.card_singleton]
  have hsplit : U (betweenPairs lab grp) (shift grp b s)
      = U ({q, q'} : Finset (I × I)) (shift grp b s)
        + U (betweenPairs lab grp \ {q, q'}) (shift grp b s) := by
    unfold U
    rw [add_comm]
    exact (Finset.sum_sdiff hpair).symm
  have h2 : U ({q, q'} : Finset (I × I)) (shift grp b s) ≤ 1 := by
    unfold U
    rw [Finset.sum_insert (by simpa using hne), Finset.sum_singleton]
    have := crossed_pair_le_one (b (grp q.1)) (b (grp q.2)) (s q.1) (s q.2) (s q'.1) (s q'.2)
      hcross
    simpa [shift, hgrp1, hgrp2] using this
  have hrest : U (betweenPairs lab grp \ {q, q'}) (shift grp b s)
      ≤ ((betweenPairs lab grp).card : ℝ) - 2 := by
    have := U_le_card (betweenPairs lab grp \ {q, q'}) (shift grp b s)
    have hc : ((betweenPairs lab grp \ {q, q'}).card : ℝ) = (betweenPairs lab grp).card - 2 := by
      have hadd := Finset.card_sdiff_add_card_eq_card hpair
      rw [hcard2] at hadd
      have : ((betweenPairs lab grp \ {q, q'}).card : ℝ) + 2 = (betweenPairs lab grp).card := by
        exact_mod_cast congrArg (fun n : ℕ => (n : ℝ)) hadd
      linarith
    linarith [this, hc.le, hc.ge]
  linarith [hsplit, h2, hrest]

omit [Fintype I] [DecidableEq I] in
/-- A set `A` of comparisons carrying a fixed-point-free involution `σ`, each orbit of which is
worth at most one, contributes at most `|A|/2`. -/
theorem U_le_half_of_involution (A : Finset (I × I)) (s' : I → ℝ) (sig : I × I → I × I)
    (hmap : ∀ q ∈ A, sig q ∈ A) (hinv : ∀ q ∈ A, sig (sig q) = q)
    (horbit : ∀ q ∈ A, kern (s' q.1) (s' q.2) + kern (s' (sig q).1) (s' (sig q).2) ≤ 1) :
    U A s' ≤ (A.card : ℝ) / 2 := by
  classical
  have hreindex : ∑ q ∈ A, kern (s' (sig q).1) (s' (sig q).2) = U A s' := by
    unfold U
    refine Finset.sum_nbij' (fun q => sig q) (fun q => sig q) hmap hmap ?_ ?_ ?_
    · intro q hq; exact hinv q hq
    · intro q hq; exact hinv q hq
    · intro q _; rfl
  have h2 : U A s' + U A s' ≤ (A.card : ℝ) := by
    calc U A s' + U A s'
        = ∑ q ∈ A, (kern (s' q.1) (s' q.2) + kern (s' (sig q).1) (s' (sig q).2)) := by
          rw [Finset.sum_add_distrib, hreindex]
          rfl
      _ ≤ ∑ _q ∈ A, (1 : ℝ) := Finset.sum_le_sum horbit
      _ = (A.card : ℝ) := by simp
  linarith

/-- **A matching of crossed configurations lowers the ceiling by half its size.**  If `A` is a set
of cross-protein comparisons paired up by `sig` into crossed configurations, then no bias can bring
the pooled statistic to within `|A|/2` of the ceiling.  Since `A` may be taken to be any matching
in the graph of crossed comparisons, this bound is computable in polynomial time. -/
theorem ceiling_gap_of_crossed_matching (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (A : Finset (I × I)) (hA : A ⊆ betweenPairs lab grp) (sig : I × I → I × I)
    (hmap : ∀ q ∈ A, sig q ∈ A) (hinv : ∀ q ∈ A, sig (sig q) = q)
    (hgrp1 : ∀ q ∈ A, grp q.1 = grp (sig q).2) (hgrp2 : ∀ q ∈ A, grp q.2 = grp (sig q).1)
    (hcross : ∀ q ∈ A, 0 ≤ (s q.2 - s q.1) + (s (sig q).2 - s (sig q).1)) :
    U (allPairs lab) (shift grp b s)
      ≤ U (withinPairs lab grp) s + (betweenPairs lab grp).card - (A.card : ℝ) / 2 := by
  classical
  rw [U_shift_eq]
  have hsplit : U (betweenPairs lab grp) (shift grp b s)
      = U A (shift grp b s) + U (betweenPairs lab grp \ A) (shift grp b s) := by
    unfold U
    rw [add_comm]
    exact (Finset.sum_sdiff hA).symm
  have hhalf : U A (shift grp b s) ≤ (A.card : ℝ) / 2 := by
    refine U_le_half_of_involution A (shift grp b s) sig hmap hinv (fun q hq => ?_)
    have := crossed_pair_le_one (b (grp q.1)) (b (grp q.2)) (s q.1) (s q.2)
      (s (sig q).1) (s (sig q).2) (hcross q hq)
    simpa [shift, hgrp1 q hq, hgrp2 q hq] using this
  have hrest : U (betweenPairs lab grp \ A) (shift grp b s)
      ≤ ((betweenPairs lab grp).card : ℝ) - A.card := by
    have h1 := U_le_card (betweenPairs lab grp \ A) (shift grp b s)
    have hadd := Finset.card_sdiff_add_card_eq_card hA
    have hc : ((betweenPairs lab grp \ A).card : ℝ) + A.card = (betweenPairs lab grp).card := by
      exact_mod_cast congrArg (fun n : ℕ => (n : ℝ)) hadd
    linarith
  linarith

end Crossed

/-! ### The separated regime is a linear ordering problem -/

section Ordering

variable {G : Type*} [Fintype G] [DecidableEq G] [LinearOrder G]

/-- The value of a bias vector in the well-separated regime: a pair `(k, l)` pays `w k l` when
protein `k` is placed more than one unit above protein `l`. -/
noncomputable def sepValue (w : G → G → ℝ) (b : G → ℝ) : ℝ :=
  ∑ k : G, ∑ l : G, if 1 < b k - b l then w k l else 0

/-- The value of a linear order: a pair `(k, l)` pays `w k l` when `k` is ranked above `l`. -/
noncomputable def orderValue (w : G → G → ℝ) (r : G → ℝ) : ℝ :=
  ∑ k : G, ∑ l : G, if r l < r k then w k l else 0

omit [DecidableEq G] [LinearOrder G] in
/-- Every bias vector is dominated by the linear order it induces: the separated objective never
exceeds the best linear ordering value. -/
theorem exists_order_ge (w : G → G → ℝ) (hw : ∀ k l, 0 ≤ w k l) (b : G → ℝ) :
    sepValue w b ≤ orderValue w b := by
  unfold sepValue orderValue
  refine Finset.sum_le_sum (fun k _ => Finset.sum_le_sum (fun l _ => ?_))
  by_cases h : 1 < b k - b l
  · have hlt : b l < b k := by linarith
    simp [h, hlt]
  · simp only [h, if_false]
    split_ifs with h2
    · exact hw k l
    · exact le_rfl

omit [DecidableEq G] [LinearOrder G] in
/-- Conversely every linear order is realised by a bias vector, with the same value: rank `k`
at `2·(rank of k)`. -/
theorem exists_bias_of_order (w : G → G → ℝ) (r : G → ℝ)
    (hinj : ∀ k l : G, r k ≠ r l → 1 ≤ |r k - r l|) :
    sepValue w (fun k => 2 * r k) = orderValue w r := by
  unfold sepValue orderValue
  refine Finset.sum_congr rfl (fun k _ => Finset.sum_congr rfl (fun l _ => ?_))
  by_cases h : r l < r k
  · have hne : r k ≠ r l := ne_of_gt h
    have := hinj k l hne
    rw [abs_of_pos (by linarith)] at this
    have : (1 : ℝ) < 2 * r k - 2 * r l := by linarith
    simp [h, this]
  · have : ¬ (1 < 2 * r k - 2 * r l) := by
      push_neg at h ⊢
      linarith
    simp [h, this]

end Ordering

end IDR.GroupedAUC
