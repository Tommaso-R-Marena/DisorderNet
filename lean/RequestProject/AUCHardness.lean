/-
# The recalibration optimum *is* a linear ordering problem

`AUCCeiling.lean` identifies, in the well-separated regime, the optimisation
`max_b AUC_pooled(s + b)` with the *linear ordering problem* (`sepValue`, `orderValue`,
`exists_order_ge`, `exists_bias_of_order`) — but only in one direction: every score table gives
rise to an ordering problem.  The converse — that *every* linear ordering instance arises from a
score table — is what a hardness claim needs, and it is what this file supplies.

Given a weight matrix `W : Fin K → Fin K → ℕ` (with `W k k = 0`) we build an explicit residue
table: a finite item set, a label, a protein map and a score, such that

  `max_b U_pooled(s + b) = U_within(s) + C + lopOpt W`,

with `C` an explicit constant read off the table (`baseCount`) and
`lopOpt W = max_π Σ_{π l < π k} W k l` the optimum of the linear ordering instance `W`.
Since maximum acyclic subgraph / weighted linear ordering is NP-hard (Karp; Garey–Johnson GT44),
this exhibits the per-protein recalibration optimum as a problem *at least as hard* as linear
ordering: no polynomial algorithm computes `max_b AUC_pooled(s+b)` for all score tables unless
P = NP.  (The complexity statement itself is classical and informal; what is machine-checked here
is the reduction identity, which is the mathematical content.)

## The construction

Each protein `k` carries `M` *background* positives at score `0` and `M` background negatives at
score `-N`, with `N = 2K+2`.  For every ordered pair `(k, l)` and every unit of weight
`j < W k l` there is a *slot*: one extra positive in protein `k` at a large score `S`, and one
extra negative in protein `l` at score `S + 1`.

* the slot's own comparison (positive of `k` versus negative of `l`) has gap `1`: it is won
  exactly when `b k - b l > 1`;
* every other cross-protein comparison has gap `≤ -N` (always won) or `≥ N` (never won) as long
  as all biases lie within `N` of one another;
* the `M²` background comparisons in each direction of each protein pair pin the optimum inside
  that box: leaving it costs `M²/2` comparisons, more than everything else in the table put
  together (`slotCount_bound`).

So the optimisation over biases collapses to: choose a linear order of the proteins, collect
`W k l` for every pair placed in the order `k` above `l`.
-/
import RequestProject.AUCCeiling

set_option autoImplicit false
set_option synthInstance.maxSize 1000

namespace IDR.GroupedAUC

open Finset

namespace Hardness

/-! ## The linear ordering problem -/

section LOP

variable {K : ℕ}

/-- The value of a permutation as a linear order of the proteins: pay `W k l` whenever `k` is
placed above `l`. -/
def lopValue (W : Fin K → Fin K → ℕ) (π : Equiv.Perm (Fin K)) : ℕ :=
  ∑ k : Fin K, ∑ l : Fin K, if π l < π k then W k l else 0

/-- **The linear ordering optimum** of the weight matrix `W`. -/
def lopOpt (W : Fin K → Fin K → ℕ) : ℕ := (univ : Finset (Equiv.Perm (Fin K))).sup (lopValue W)

theorem lopValue_le_lopOpt (W : Fin K → Fin K → ℕ) (π : Equiv.Perm (Fin K)) :
    lopValue W π ≤ lopOpt W :=
  Finset.le_sup (f := lopValue W) (mem_univ π)

theorem exists_lopValue_eq (W : Fin K → Fin K → ℕ) :
    ∃ π : Equiv.Perm (Fin K), lopValue W π = lopOpt W := by
  classical
  obtain ⟨π, -, hπ⟩ :=
    Finset.exists_mem_eq_sup (univ : Finset (Equiv.Perm (Fin K))) univ_nonempty (lopValue W)
  exact ⟨π, hπ.symm⟩

end LOP

/-! ## The score table built from a weight matrix -/

section Table

variable (K M : ℕ) (W : Fin K → Fin K → ℕ)

/-- A *slot*: one unit of weight `W k l`, for an ordered pair of proteins `(k, l)`. -/
abbrev Slot : Type := Σ p : Fin K × Fin K, Fin (W p.1 p.2)

/-- The items of the constructed table: `M` background positives and `M` background negatives per
protein (the `Bool` is the label), plus, for each slot, one positive in the first protein and one
negative in the second (again recorded by the `Bool`). -/
abbrev Itm : Type := ((Fin K × Fin M) × Bool) ⊕ (Slot K W × Bool)

/-- The half-width of the box the background comparisons confine the bias to. -/
def boxN : ℤ := 2 * K + 2

/-- The score of the slot's positive; slots are placed far apart and far above the background. -/
noncomputable def slotScore (σ : Slot K W) : ℤ :=
  (2 * boxN K + 2) * ((Fintype.equivFin (Slot K W) σ : ℕ) + 1)

/-- The protein an item belongs to. -/
def grp : Itm K M W → Fin K
  | Sum.inl ((k, _), _) => k
  | Sum.inr (⟨(k, l), _⟩, side) => if side then k else l

/-- The label: `true` = positive. -/
def lab : Itm K M W → Bool
  | Sum.inl (_, c) => c
  | Sum.inr (_, side) => side

/-- The integer score of an item. -/
noncomputable def sc : Itm K M W → ℤ
  | Sum.inl (_, true) => 0
  | Sum.inl (_, false) => -boxN K
  | Sum.inr (σ, true) => slotScore K W σ
  | Sum.inr (σ, false) => slotScore K W σ + 1

/-- The real-valued score field of the constructed table. -/
noncomputable def scr : Itm K M W → ℝ := fun i => (sc K M W i : ℝ)

/-- The number of slots: the total weight of the linear ordering instance. -/
def slotCount : ℕ := Fintype.card (Slot K W)

/-- The gap of a comparison: the negative's score minus the positive's score.  The comparison is
won exactly when the bias difference exceeds the gap. -/
noncomputable def gapOf (q : Itm K M W × Itm K M W) : ℤ := sc K M W q.2 - sc K M W q.1

/-- The cross-protein comparisons that are won for every bias inside the box. -/
noncomputable def basePairs : Finset (Itm K M W × Itm K M W) :=
  (betweenPairs (lab K M W) (grp K M W)).filter (fun q => gapOf K M W q ≤ -boxN K)

end Table

/-! ## Elementary facts about the table -/

section Facts

variable {K M : ℕ} {W : Fin K → Fin K → ℕ}

theorem boxN_pos : 0 < boxN K := by unfold boxN; positivity

theorem boxN_le_slotScore (σ : Slot K W) : boxN K ≤ slotScore K W σ := by
  unfold slotScore
  have h := boxN_pos (K := K)
  have h2 : (1:ℤ) ≤ ((Fintype.equivFin (Slot K W) σ : ℕ) : ℤ) + 1 := by
    have : (0:ℤ) ≤ ((Fintype.equivFin (Slot K W) σ : ℕ) : ℤ) := Int.natCast_nonneg _
    omega
  nlinarith

theorem slotScore_apart {σ τ : Slot K W} (h : σ ≠ τ) :
    2 * boxN K + 2 ≤ |slotScore K W σ - slotScore K W τ| := by
  have hpos : (0:ℤ) < 2 * boxN K + 2 := by have := boxN_pos (K := K); omega
  have hne : ((Fintype.equivFin (Slot K W) σ : ℕ) : ℤ)
      ≠ ((Fintype.equivFin (Slot K W) τ : ℕ) : ℤ) := by
    intro hc
    apply h
    have : (Fintype.equivFin (Slot K W) σ : ℕ) = (Fintype.equivFin (Slot K W) τ : ℕ) := by
      exact_mod_cast hc
    exact (Fintype.equivFin (Slot K W)).injective (Fin.ext this)
  set a : ℤ := ((Fintype.equivFin (Slot K W) σ : ℕ) : ℤ) + 1 with ha
  set c : ℤ := ((Fintype.equivFin (Slot K W) τ : ℕ) : ℤ) + 1 with hcc
  have hac : a ≠ c := by omega
  have hrw : slotScore K W σ - slotScore K W τ = (2 * boxN K + 2) * (a - c) := by
    unfold slotScore; rw [ha, hcc]; ring
  rw [hrw, abs_mul, abs_of_pos hpos]
  have h1 : (1:ℤ) ≤ |a - c| := by
    rcases lt_or_gt_of_ne hac with h' | h'
    · rw [abs_of_neg (by omega : a - c < 0)]; omega
    · rw [abs_of_pos (by omega : 0 < a - c)]; omega
  nlinarith

/-- Slots run between two *different* proteins, because the diagonal weights vanish. -/
theorem slot_grp_ne (hW : ∀ k, W k k = 0) (σ : Slot K W) : σ.1.1 ≠ σ.1.2 := by
  intro h
  have hz : W σ.1.1 σ.1.2 = 0 := by rw [← h]; exact hW _
  exact absurd σ.2.isLt (by simp [hz])

/-- The comparison kernel after a per-protein bias only sees the bias difference against the
gap of the comparison. -/
theorem kern_shift_eq {I G : Type*} (grp : I → G) (b : G → ℝ) (s : I → ℝ) (p n : I) :
    kern (shift grp b s p) (shift grp b s n) = kern (b (grp p) - b (grp n)) (s n - s p) := by
  have e1 : (s n + b (grp n) < s p + b (grp p)) ↔ (s n - s p < b (grp p) - b (grp n)) := by
    constructor <;> intro h <;> linarith
  have e2 : (s p + b (grp p) = s n + b (grp n)) ↔ (b (grp p) - b (grp n) = s n - s p) := by
    constructor <;> intro h <;> linarith
  simp only [kern, shift, e1, e2]

theorem mem_betweenPairs_iff {I G : Type*} [Fintype I] [DecidableEq G] (lab : I → Bool)
    (grp : I → G) (q : I × I) :
    q ∈ betweenPairs lab grp ↔ lab q.1 = true ∧ lab q.2 = false ∧ grp q.1 ≠ grp q.2 := by
  classical
  simp [betweenPairs, allPairs, posSet, negSet, Finset.mem_filter, Finset.mem_product,
    and_assoc]

/-- **The gap trichotomy.**  Every cross-protein comparison of the constructed table is either a
slot comparison (gap `1`, decided by the sign of the bias difference), or has gap at most `-N`
(won inside the box) or at least `N` (lost inside the box). -/
theorem gap_trichotomy {q : Itm K M W × Itm K M W}
    (hq : q ∈ betweenPairs (lab K M W) (grp K M W)) :
    gapOf K M W q ≤ -boxN K ∨ (∃ σ : Slot K W, q = (Sum.inr (σ, true), Sum.inr (σ, false)))
      ∨ boxN K ≤ gapOf K M W q := by
  classical
  rw [mem_betweenPairs_iff] at hq
  obtain ⟨h1, h2, -⟩ := hq
  obtain ⟨x, y⟩ := q
  match x, h1 with
  | Sum.inl ((k, i), true), _ =>
      match y, h2 with
      | Sum.inl ((k', i'), false), _ =>
          left; simp [gapOf, sc]
      | Sum.inr (τ, false), _ =>
          right; right
          have := boxN_le_slotScore (K := K) (W := W) τ
          simp only [gapOf, sc]
          omega
  | Sum.inr (σ, true), _ =>
      match y, h2 with
      | Sum.inl ((k', i'), false), _ =>
          left
          have h0 : 0 < slotScore K W σ :=
            lt_of_lt_of_le (boxN_pos (K := K)) (boxN_le_slotScore σ)
          simp only [gapOf, sc]
          omega
      | Sum.inr (τ, false), _ =>
          by_cases h : σ = τ
          · subst h; right; left; exact ⟨σ, rfl⟩
          · have hap := slotScore_apart (K := K) (W := W) h
            have hb := boxN_pos (K := K)
            rcases abs_cases (slotScore K W σ - slotScore K W τ) with ⟨he, -⟩ | ⟨he, -⟩
            · left; simp only [gapOf, sc]; omega
            · right; right; simp only [gapOf, sc]; omega

/-- A slot comparison has gap `1`: it is won exactly when the bias of the first protein exceeds
that of the second by more than one. -/
theorem gapOf_slot (σ : Slot K W) :
    gapOf K M W (Sum.inr (σ, true), Sum.inr (σ, false)) = 1 := by
  simp [gapOf, sc]

/-- Slot comparisons are cross-protein comparisons of the table. -/
theorem slot_mem_betweenPairs (hW : ∀ k, W k k = 0) (σ : Slot K W) :
    ((Sum.inr (σ, true), Sum.inr (σ, false)) : Itm K M W × Itm K M W)
      ∈ betweenPairs (lab K M W) (grp K M W) := by
  classical
  rw [mem_betweenPairs_iff]
  refine ⟨rfl, rfl, ?_⟩
  simpa [grp] using slot_grp_ne hW σ

/-! ## The two halves of the optimisation -/

/-- The slot comparisons, as a finite set of pairs. -/
noncomputable def desPairs (K M : ℕ) (W : Fin K → Fin K → ℕ) : Finset (Itm K M W × Itm K M W) :=
  Finset.image (fun σ : Slot K W =>
    ((Sum.inr (σ, true) : Itm K M W), (Sum.inr (σ, false) : Itm K M W))) univ

/-- The `M²` background comparisons running from protein `k` to protein `l`. -/
noncomputable def bgPairs (K M : ℕ) (W : Fin K → Fin K → ℕ) (k l : Fin K) :
    Finset (Itm K M W × Itm K M W) :=
  Finset.image (fun ii : Fin M × Fin M =>
    ((Sum.inl ((k, ii.1), true) : Itm K M W), (Sum.inl ((l, ii.2), false) : Itm K M W))) univ

theorem card_bgPairs (k l : Fin K) : (bgPairs K M W k l).card = M * M := by
  classical
  rw [bgPairs, Finset.card_image_of_injective _ ?_, Finset.card_univ]
  · simp
  · rintro ⟨i, j⟩ ⟨i', j'⟩ h
    simp only [Prod.mk.injEq, Sum.inl.injEq] at h
    obtain ⟨h1, h2⟩ := h
    simp_all

theorem bgPairs_subset_basePairs {k l : Fin K} (hkl : k ≠ l) :
    bgPairs K M W k l ⊆ basePairs K M W := by
  classical
  intro q hq
  simp only [bgPairs, Finset.mem_image, Finset.mem_univ, true_and] at hq
  obtain ⟨⟨i, j⟩, rfl⟩ := hq
  rw [basePairs, Finset.mem_filter]
  refine ⟨?_, by simp [gapOf, sc]⟩
  rw [mem_betweenPairs_iff]
  exact ⟨rfl, rfl, by simpa [grp] using hkl⟩

/-- Every cross-protein comparison that is not won inside the box has a slot item as its
negative. -/
theorem sdiff_basePairs_subset :
    betweenPairs (lab K M W) (grp K M W) \ basePairs K M W
      ⊆ (univ : Finset (Itm K M W)) ×ˢ
          (Finset.image (fun σ : Slot K W => (Sum.inr (σ, false) : Itm K M W)) univ) := by
  classical
  intro q hq
  rw [Finset.mem_sdiff] at hq
  obtain ⟨hq, hnot⟩ := hq
  have hgap : ¬ (gapOf K M W q ≤ -boxN K) := by
    intro h; exact hnot (by rw [basePairs, Finset.mem_filter]; exact ⟨hq, h⟩)
  rw [Finset.mem_product]
  refine ⟨Finset.mem_univ _, ?_⟩
  rw [mem_betweenPairs_iff] at hq
  obtain ⟨h1, h2, -⟩ := hq
  obtain ⟨x, y⟩ := q
  match y, h2 with
  | Sum.inl ((k', i'), false), _ =>
      exfalso
      apply hgap
      match x, h1 with
      | Sum.inl ((k, i), true), _ => simp [gapOf, sc]
      | Sum.inr (σ, true), _ =>
          have h0 : 0 < slotScore K W σ :=
            lt_of_lt_of_le (boxN_pos (K := K)) (boxN_le_slotScore σ)
          simp only [gapOf, sc]
          omega
  | Sum.inr (τ, false), _ =>
      simp only [Finset.mem_image, Finset.mem_univ, true_and]
      exact ⟨τ, rfl⟩

end Facts

/-! ## Evaluating the statistic -/

section Evaluate

variable {K M : ℕ} {W : Fin K → Fin K → ℕ}

/-- The value of a single comparison of the table under a per-protein bias. -/
theorem kern_eval (b : Fin K → ℝ) (q : Itm K M W × Itm K M W) :
    kern (shift (grp K M W) b (scr K M W) q.1) (shift (grp K M W) b (scr K M W) q.2)
      = kern (b (grp K M W q.1) - b (grp K M W q.2)) ((gapOf K M W q : ℤ) : ℝ) := by
  rw [kern_shift_eq]
  congr 1
  simp [scr, gapOf]

/-- Sums over slots are sums over ordered protein pairs weighted by `W`. -/
theorem slot_sum_eq (g : Fin K → Fin K → ℝ) :
    ∑ σ : Slot K W, g σ.1.1 σ.1.2 = ∑ k : Fin K, ∑ l : Fin K, (W k l : ℝ) * g k l := by
  classical
  rw [← Finset.univ_sigma_univ, Finset.sum_sigma]
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl (fun k _ => Finset.sum_congr rfl (fun l _ => ?_))
  simp [mul_comm]

/-- **Inside the box** the cross-protein statistic is the base count plus the slot term. -/
theorem U_between_in_box (hW : ∀ k, W k k = 0) (b : Fin K → ℝ)
    (hbox : ∀ k l, |b k - b l| < (boxN K : ℝ)) :
    U (betweenPairs (lab K M W) (grp K M W)) (shift (grp K M W) b (scr K M W))
      = (basePairs K M W).card + ∑ σ : Slot K W, kern (b σ.1.1 - b σ.1.2) 1 := by
  classical
  set X := betweenPairs (lab K M W) (grp K M W) with hX
  set A := basePairs K M W with hAdef
  set f : Itm K M W × Itm K M W → ℝ :=
    fun q => kern (shift (grp K M W) b (scr K M W) q.1) (shift (grp K M W) b (scr K M W) q.2)
    with hf
  have hAsub : A ⊆ X := Finset.filter_subset _ _
  have hlow : ∀ k l : Fin K, -(boxN K : ℝ) < b k - b l := by
    intro k l
    have := hbox k l
    rw [abs_lt] at this
    exact this.1
  have hhigh : ∀ k l : Fin K, b k - b l < (boxN K : ℝ) := by
    intro k l
    have := hbox k l
    rw [abs_lt] at this
    exact this.2
  -- the base comparisons are all won
  have hbase : ∑ q ∈ A, f q = (A.card : ℝ) := by
    rw [Finset.sum_congr rfl (fun q hq => ?_), Finset.sum_const, nsmul_eq_mul, mul_one]
    have hgap : gapOf K M W q ≤ -boxN K := (Finset.mem_filter.mp hq).2
    have hgapR : ((gapOf K M W q : ℤ) : ℝ) ≤ -(boxN K : ℝ) := by exact_mod_cast hgap
    have : ((gapOf K M W q : ℤ) : ℝ) < b (grp K M W q.1) - b (grp K M W q.2) :=
      lt_of_le_of_lt hgapR (hlow _ _)
    rw [hf]
    simpa [kern_eval] using kern_eq_one_of_lt this
  -- the remaining comparisons are the slot comparisons
  have hdes : desPairs K M W ⊆ X \ A := by
    intro q hq
    simp only [desPairs, Finset.mem_image, Finset.mem_univ, true_and] at hq
    obtain ⟨σ, rfl⟩ := hq
    rw [Finset.mem_sdiff]
    refine ⟨slot_mem_betweenPairs hW σ, ?_⟩
    intro hmem
    have hgap : gapOf K M W
        ((Sum.inr (σ, true) : Itm K M W), (Sum.inr (σ, false) : Itm K M W)) ≤ -boxN K :=
      (Finset.mem_filter.mp hmem).2
    rw [gapOf_slot] at hgap
    have := boxN_pos (K := K)
    omega
  have hrest : ∑ q ∈ desPairs K M W, f q = ∑ q ∈ X \ A, f q := by
    refine Finset.sum_subset hdes (fun q hq hqn => ?_)
    rw [Finset.mem_sdiff] at hq
    have hgapnot : ¬ (gapOf K M W q ≤ -boxN K) := by
      intro h; exact hq.2 (Finset.mem_filter.mpr ⟨hq.1, h⟩)
    rcases gap_trichotomy hq.1 with h | ⟨σ, rfl⟩ | h
    · exact absurd h hgapnot
    · exact absurd (by simp [desPairs]) hqn
    · have hgapR : ((boxN K : ℤ) : ℝ) ≤ ((gapOf K M W q : ℤ) : ℝ) := by exact_mod_cast h
      have hlt : b (grp K M W q.1) - b (grp K M W q.2) < ((gapOf K M W q : ℤ) : ℝ) :=
        lt_of_lt_of_le (by simpa using hhigh (grp K M W q.1) (grp K M W q.2)) hgapR
      rw [hf]
      simp only [kern_eval]
      unfold kern
      have h1 : ¬ ((gapOf K M W q : ℤ) : ℝ) < b (grp K M W q.1) - b (grp K M W q.2) :=
        not_lt.mpr hlt.le
      have h2 : b (grp K M W q.1) - b (grp K M W q.2) ≠ ((gapOf K M W q : ℤ) : ℝ) := ne_of_lt hlt
      simp [h1, h2]
  have himg : ∑ q ∈ desPairs K M W, f q = ∑ σ : Slot K W, kern (b σ.1.1 - b σ.1.2) 1 := by
    rw [desPairs, Finset.sum_image]
    · refine Finset.sum_congr rfl (fun σ _ => ?_)
      rw [hf]
      simp only
      rw [kern_shift_eq]
      norm_num [grp, scr, sc]
    · intro σ _ τ _ h
      simp only [Prod.mk.injEq, Sum.inr.injEq, Prod.mk.injEq] at h
      exact h.1.1
  have hsum : ∑ q ∈ X \ A, f q + ∑ q ∈ A, f q = ∑ q ∈ X, f q := Finset.sum_sdiff hAsub
  have : U X (shift (grp K M W) b (scr K M W)) = ∑ q ∈ X, f q := rfl
  rw [this, ← hsum, hbase, ← hrest, himg]
  ring

/-- The slot term never exceeds the linear ordering optimum: the comparisons a bias wins are the
pairs of a linear order, so their total weight is an ordering value. -/
theorem slot_sum_le_lopOpt (b : Fin K → ℝ) :
    ∑ σ : Slot K W, kern (b σ.1.1 - b σ.1.2) 1 ≤ (lopOpt W : ℝ) := by
  classical
  obtain ⟨π, hπ⟩ : ∃ π : Equiv.Perm (Fin K), ∀ k l, b l < b k → π l < π k := by
    refine ⟨(Tuple.sort b)⁻¹, fun k l h => ?_⟩
    by_contra hc
    push_neg at hc
    have hm := Tuple.monotone_sort b hc
    simp only [Function.comp_apply, Equiv.Perm.inv_def, Equiv.apply_symm_apply] at hm
    linarith
  have hstep : ∑ σ : Slot K W, kern (b σ.1.1 - b σ.1.2) 1
      ≤ ∑ k : Fin K, ∑ l : Fin K, (W k l : ℝ) * (if π l < π k then 1 else 0) := by
    rw [slot_sum_eq (W := W) (fun k l => kern (b k - b l) 1)]
    refine Finset.sum_le_sum (fun k _ => Finset.sum_le_sum (fun l _ => ?_))
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    by_cases hlt : π l < π k
    · simpa [hlt] using kern_le_one (b k - b l) 1
    · have hble : b k ≤ b l := by
        by_contra hc
        push_neg at hc
        exact hlt (hπ k l hc)
      have h0 : kern (b k - b l) 1 = 0 := by
        unfold kern
        have h1 : ¬ (1 : ℝ) < b k - b l := by linarith
        have h2 : b k - b l ≠ 1 := by intro hc; linarith
        simp [h1, h2]
      simp [hlt, h0]
  have hcast : ∑ k : Fin K, ∑ l : Fin K, (W k l : ℝ) * (if π l < π k then 1 else 0)
      = ((lopValue W π : ℕ) : ℝ) := by
    unfold lopValue
    push_cast
    refine Finset.sum_congr rfl (fun k _ => Finset.sum_congr rfl (fun l _ => ?_))
    by_cases hlt : π l < π k <;> simp [hlt]
  rw [hcast] at hstep
  exact hstep.trans (by exact_mod_cast Nat.cast_le.mpr (lopValue_le_lopOpt W π))

/-- **Outside the box** the lost background comparisons already outweigh everything else. -/
theorem U_between_le_of_bg_loss
    (hM : 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M)
    (b : Fin K → ℝ) {k l : Fin K} (hne : k ≠ l) (hle : b k - b l ≤ -(boxN K : ℝ)) :
    U (betweenPairs (lab K M W) (grp K M W)) (shift (grp K M W) b (scr K M W))
      ≤ (basePairs K M W).card := by
  classical
  set X := betweenPairs (lab K M W) (grp K M W) with hX
  set A := basePairs K M W with hAdef
  set BG := bgPairs K M W k l with hBG
  set f : Itm K M W × Itm K M W → ℝ :=
    fun q => kern (shift (grp K M W) b (scr K M W) q.1) (shift (grp K M W) b (scr K M W) q.2)
    with hf
  have hAsub : A ⊆ X := Finset.filter_subset _ _
  have hBGA : BG ⊆ A := bgPairs_subset_basePairs hne
  have hBGcard : BG.card = M * M := card_bgPairs k l
  -- the lost background comparisons
  have hbg : ∑ q ∈ BG, f q ≤ (M * M : ℝ) / 2 := by
    have hterm : ∀ q ∈ BG, f q ≤ 1 / 2 := by
      intro q hq
      simp only [hBG, bgPairs, Finset.mem_image, Finset.mem_univ, true_and] at hq
      obtain ⟨⟨i, j⟩, rfl⟩ := hq
      rw [hf]
      simp only
      rw [kern_shift_eq]
      have hg1 : grp K M W (Sum.inl ((k, i), true) : Itm K M W) = k := rfl
      have hg2 : grp K M W (Sum.inl ((l, j), false) : Itm K M W) = l := rfl
      rw [hg1, hg2]
      have hsc : scr K M W (Sum.inl ((l, j), false) : Itm K M W)
          - scr K M W (Sum.inl ((k, i), true) : Itm K M W) = -(boxN K : ℝ) := by
        simp [scr, sc]
      rw [hsc]
      unfold kern
      have h1 : ¬ (-(boxN K : ℝ) < b k - b l) := not_lt.mpr hle
      simp only [h1, if_false]
      split_ifs <;> norm_num
    calc ∑ q ∈ BG, f q ≤ ∑ _q ∈ BG, (1 / 2 : ℝ) := Finset.sum_le_sum hterm
      _ = (BG.card : ℝ) / 2 := by simp; ring
      _ = (M * M : ℝ) / 2 := by rw [hBGcard]; push_cast; ring
  -- the remaining base comparisons
  have hrest : ∑ q ∈ A \ BG, f q ≤ (A.card : ℝ) - (M * M : ℝ) := by
    have h1 : ∑ q ∈ A \ BG, f q ≤ ((A \ BG).card : ℝ) := U_le_card _ _
    have h2 : (A \ BG).card + BG.card = A.card := Finset.card_sdiff_add_card_eq_card hBGA
    have h3 : ((A \ BG).card : ℝ) = (A.card : ℝ) - (M * M : ℝ) := by
      have : ((A \ BG).card : ℝ) + (BG.card : ℝ) = (A.card : ℝ) := by exact_mod_cast h2
      rw [hBGcard] at this
      push_cast at this ⊢
      linarith
    linarith [h1, h3.le, h3.ge]
  -- everything that is not a base comparison has a slot item as its negative
  have houter : ∑ q ∈ X \ A, f q ≤ (Fintype.card (Itm K M W) * slotCount K W : ℝ) := by
    have h1 : ∑ q ∈ X \ A, f q ≤ ((X \ A).card : ℝ) := U_le_card _ _
    have h2 : (X \ A).card ≤ Fintype.card (Itm K M W) * slotCount K W := by
      refine le_trans (Finset.card_le_card sdiff_basePairs_subset) ?_
      rw [Finset.card_product, Finset.card_univ]
      exact Nat.mul_le_mul_left _ (le_trans Finset.card_image_le (by simp [slotCount]))
    have : ((X \ A).card : ℝ) ≤ (Fintype.card (Itm K M W) * slotCount K W : ℝ) := by
      exact_mod_cast h2
    linarith
  have hMR : 2 * (Fintype.card (Itm K M W) * slotCount K W : ℝ) ≤ (M * M : ℝ) := by
    exact_mod_cast hM
  have hsplitA : ∑ q ∈ A \ BG, f q + ∑ q ∈ BG, f q = ∑ q ∈ A, f q := Finset.sum_sdiff hBGA
  have hsplitX : ∑ q ∈ X \ A, f q + ∑ q ∈ A, f q = ∑ q ∈ X, f q := Finset.sum_sdiff hAsub
  have hU : U X (shift (grp K M W) b (scr K M W)) = ∑ q ∈ X, f q := rfl
  rw [hU, ← hsplitX, ← hsplitA]
  linarith

end Evaluate

/-! ## The reduction -/

section Reduction

variable {K M : ℕ} {W : Fin K → Fin K → ℕ}

theorem card_Itm : Fintype.card (Itm K M W) = 2 * (K * M) + 2 * slotCount K W := by
  simp [slotCount, Fintype.card_sum, Fintype.card_prod]
  ring

/-- A background multiplicity large enough to pin the optimum inside the box always exists. -/
theorem exists_good_multiplicity (K : ℕ) (W : Fin K → Fin K → ℕ) :
    ∃ M : ℕ, 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M := by
  classical
  set T := slotCount K W with hT
  refine ⟨4 * K * T + 4 * T + 4, ?_⟩
  rw [card_Itm]
  set M := 4 * K * T + 4 * T + 4 with hM
  have h : 2 * ((2 * (K * M) + 2 * T) * T) = 4 * K * M * T + 4 * T * T := by ring
  rw [h, hM]
  nlinarith [Nat.zero_le K, Nat.zero_le T, Nat.zero_le (K * T), Nat.zero_le (T * T),
    Nat.zero_le (K * T * T)]

/-- **The recalibration optimum of the constructed table is the linear ordering optimum.**

For the score table built from the weight matrix `W`, the largest pooled Mann–Whitney statistic
reachable by a per-protein bias equals the frozen within-protein statistic, plus an explicit
constant read off the table, plus the optimum `lopOpt W` of the linear ordering instance `W`.
Every weighted linear ordering instance therefore *is* a per-protein recalibration problem. -/
theorem bias_optimum_eq_lop (hW : ∀ k, W k k = 0)
    (hM : 2 * (Fintype.card (Itm K M W) * slotCount K W) ≤ M * M) :
    (∀ b : Fin K → ℝ,
        U (allPairs (lab K M W)) (shift (grp K M W) b (scr K M W))
          ≤ U (withinPairs (lab K M W) (grp K M W)) (scr K M W)
            + (basePairs K M W).card + (lopOpt W : ℝ))
      ∧ ∃ b : Fin K → ℝ,
        U (allPairs (lab K M W)) (shift (grp K M W) b (scr K M W))
          = U (withinPairs (lab K M W) (grp K M W)) (scr K M W)
            + (basePairs K M W).card + (lopOpt W : ℝ) := by
  classical
  constructor
  · intro b
    rw [U_shift_eq]
    by_cases hbox : ∀ k l : Fin K, |b k - b l| < (boxN K : ℝ)
    · have h := U_between_in_box (M := M) hW b hbox
      have hle := slot_sum_le_lopOpt (W := W) b
      rw [h]
      linarith
    · push_neg at hbox
      obtain ⟨k, l, hkl⟩ := hbox
      have hne : k ≠ l := by
        intro h
        rw [h] at hkl
        simp only [sub_self, abs_zero] at hkl
        have := boxN_pos (K := K)
        have : (0:ℝ) < (boxN K : ℝ) := by exact_mod_cast this
        linarith
      have hloss : U (betweenPairs (lab K M W) (grp K M W)) (shift (grp K M W) b (scr K M W))
          ≤ (basePairs K M W).card := by
        rcases abs_cases (b k - b l) with ⟨he, -⟩ | ⟨he, -⟩
        · refine U_between_le_of_bg_loss hM b (Ne.symm hne) ?_
          rw [he] at hkl
          linarith
        · refine U_between_le_of_bg_loss hM b hne ?_
          rw [he] at hkl
          linarith
      have : (0:ℝ) ≤ (lopOpt W : ℝ) := by positivity
      linarith
  · obtain ⟨pi, hpi⟩ := exists_lopValue_eq W
    refine ⟨fun k => 2 * ((pi k : ℕ) : ℝ), ?_⟩
    have hbox : ∀ k l : Fin K, |2 * ((pi k : ℕ) : ℝ) - 2 * ((pi l : ℕ) : ℝ)| < (boxN K : ℝ) := by
      intro k l
      have h1 : ((pi k : ℕ) : ℝ) < (K : ℝ) := by exact_mod_cast (pi k).isLt
      have h2 : ((pi l : ℕ) : ℝ) < (K : ℝ) := by exact_mod_cast (pi l).isLt
      have h3 : (0:ℝ) ≤ ((pi k : ℕ) : ℝ) := by positivity
      have h4 : (0:ℝ) ≤ ((pi l : ℕ) : ℝ) := by positivity
      have hb : ((boxN K : ℤ) : ℝ) = 2 * (K : ℝ) + 2 := by simp [boxN]
      rw [abs_lt, hb]
      constructor <;> linarith
    rw [U_shift_eq, U_between_in_box hW _ hbox]
    have hslot : ∑ σ : Slot K W,
        kern (2 * ((pi σ.1.1 : ℕ) : ℝ) - 2 * ((pi σ.1.2 : ℕ) : ℝ)) 1 = (lopOpt W : ℝ) := by
      rw [slot_sum_eq (W := W)
        (fun k l => kern (2 * ((pi k : ℕ) : ℝ) - 2 * ((pi l : ℕ) : ℝ)) 1)]
      rw [← hpi]
      unfold lopValue
      push_cast
      refine Finset.sum_congr rfl (fun k _ => Finset.sum_congr rfl (fun l _ => ?_))
      by_cases hlt : pi l < pi k
      · have h1 : ((pi l : ℕ) : ℝ) + 1 ≤ ((pi k : ℕ) : ℝ) := by
          have : (pi l : ℕ) + 1 ≤ (pi k : ℕ) := hlt
          exact_mod_cast this
        have : kern (2 * ((pi k : ℕ) : ℝ) - 2 * ((pi l : ℕ) : ℝ)) 1 = 1 :=
          kern_eq_one_of_lt (by linarith)
        simp [hlt, this]
      · have h1 : ((pi k : ℕ) : ℝ) ≤ ((pi l : ℕ) : ℝ) := by
          have : (pi k : ℕ) ≤ (pi l : ℕ) := not_lt.mp hlt
          exact_mod_cast this
        have h0 : kern (2 * ((pi k : ℕ) : ℝ) - 2 * ((pi l : ℕ) : ℝ)) 1 = 0 := by
          unfold kern
          have e1 : ¬ (1:ℝ) < 2 * ((pi k : ℕ) : ℝ) - 2 * ((pi l : ℕ) : ℝ) := by
            push_neg; linarith
          have e2 : 2 * ((pi k : ℕ) : ℝ) - 2 * ((pi l : ℕ) : ℝ) ≠ 1 := by intro hc; linarith
          simp [e1, e2]
        simp [hlt, h0]
    rw [hslot]
    ring

/-- **Every weighted linear ordering instance is a per-protein recalibration problem.**

Given any weight matrix `W` on `K` proteins with vanishing diagonal there is a residue table
— a finite item set with labels, protein memberships and scores — whose reachable pooled
Mann–Whitney optimum over per-protein biases is `U_within + C + lopOpt W`, with `C` a constant of
the table.  Reading the optimum therefore solves the linear ordering instance. -/
theorem lop_reduces_to_bias_optimum (K : ℕ) (W : Fin K → Fin K → ℕ) (hW : ∀ k, W k k = 0) :
    ∃ (I : Type) (_ : Fintype I) (_ : DecidableEq I) (lab : I → Bool) (grp : I → Fin K)
      (s : I → ℝ) (C : ℝ),
      (∀ b : Fin K → ℝ, U (allPairs lab) (shift grp b s)
          ≤ U (withinPairs lab grp) s + C + (lopOpt W : ℝ))
        ∧ (∃ b : Fin K → ℝ, U (allPairs lab) (shift grp b s)
          = U (withinPairs lab grp) s + C + (lopOpt W : ℝ)) := by
  classical
  obtain ⟨M, hM⟩ := exists_good_multiplicity K W
  obtain ⟨hup, hex⟩ := bias_optimum_eq_lop (M := M) hW hM
  exact ⟨Itm K M W, inferInstance, inferInstance, lab K M W, grp K M W, scr K M W,
    ((basePairs K M W).card : ℝ), hup, hex⟩

end Reduction

end Hardness

end IDR.GroupedAUC
