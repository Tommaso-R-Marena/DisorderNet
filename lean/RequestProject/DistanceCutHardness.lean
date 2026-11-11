/-
# Realisability of a mean-distance panel is a max-cut question

`DistanceCutEnsembles.lean` shows that ensembles of chain conformations realise the whole cut cone
truncated at the contour scale (`cutEnsemble_meanDist`).  This file turns that inclusion into a
statement about *computation*: for the two-state ensembles that realise those panels — each
residue in one of two positions, the compact/extended exchange of a disordered region — the
question "is this measured panel realisable?" is decided by a family of linear tests whose
right-hand side is, in each instance, the **maximum cut** of a weighted graph on the residues.

* `maxCut_eq_greatest_twoState_panel_value` — **the reduction identity.**  For a weight matrix `c`
  on the residues and a scale `t ≥ 0`, the largest value of the weighted mean-distance functional
  `∑_{i,j} c i j · D i j` over all two-state ensembles at scale `t` is exactly `t · maxCut c`, and
  it is attained by a single conformation.  Fitting the best two-state ensemble to a weighted
  panel target is therefore *exactly* max cut, which is NP-hard (Karp).  This is the analogue, for
  ensemble data, of the linear ordering reduction in `AUCHardness.lean`.

* `cutRealisable_iff_forall_weight` — **the realisability test.**  A panel `p` is realisable by a
  two-state ensemble at scale `t` iff `∑_{i,j} c i j · p i j ≤ t · maxCut c` for *every* weight
  matrix `c`.  The forward direction is a computation; the converse is separation (the realisable
  panels form a compact convex set, `isCompact_cutRealisableSet`).  So a realisability certificate
  is exactly a max-cut bound, one per weighting: checking a panel against a single weighting
  already requires the max cut of that weighting.

* `cutRealisable_iff_twoStateEnsemble` — the bridge to the ensembles themselves: the panels of the
  previous two theorems are precisely the mean-distance panels of ensembles of chain conformations
  in which every residue sits at one of two points a distance `t` apart.

What is *not* claimed: ensemble panels in general are convex combinations of Euclidean metrics, a
cone strictly larger than the cut cone, so the hardness proved here is hardness of the two-state
(cut) realisability problem, not of general ensemble realisability.  The NP-hardness of max cut
itself is the classical input and is quoted, not machine-checked; what is machine-checked is that
the realisability question *is* that problem.
-/
import Mathlib
import RequestProject.DistanceCutEnsembles

set_option autoImplicit false

namespace RequestProject.DistanceCutHardness

open Finset RequestProject.DistanceRealizability RequestProject.DistanceCutEnsembles

variable {N : ℕ}

/-! ## Cuts, panels and the max-cut value -/

/-- The cut pseudometric of a two-block split of the residues: `1` for residues on opposite sides,
`0` for residues on the same side. -/
def cutPanel (S : Fin N → Bool) : Fin N → Fin N → ℝ := fun i j => if S i = S j then 0 else 1

/-- The value a weight matrix assigns to a panel. -/
def panelValue (c p : Fin N → Fin N → ℝ) : ℝ := ∑ i, ∑ j, c i j * p i j

/-- The weight of the cut defined by a split. -/
def cutValue (c : Fin N → Fin N → ℝ) (S : Fin N → Bool) : ℝ := panelValue c (cutPanel S)

/-- **The maximum cut** of a weight matrix on the residues. -/
noncomputable def maxCut (c : Fin N → Fin N → ℝ) : ℝ :=
  (univ : Finset (Fin N → Bool)).sup' univ_nonempty (cutValue c)

theorem cutValue_le_maxCut (c : Fin N → Fin N → ℝ) (S : Fin N → Bool) :
    cutValue c S ≤ maxCut c :=
  Finset.le_sup' (cutValue c) (mem_univ S)

theorem exists_cutValue_eq_maxCut (c : Fin N → Fin N → ℝ) :
    ∃ S : Fin N → Bool, cutValue c S = maxCut c := by
  classical
  obtain ⟨S, -, hS⟩ := Finset.exists_mem_eq_sup' (univ_nonempty) (cutValue c)
  exact ⟨S, hS.symm⟩

/-- The empty cut has value zero, so the maximum cut is never negative. -/
theorem maxCut_nonneg (c : Fin N → Fin N → ℝ) : 0 ≤ maxCut c := by
  have h : cutValue c (fun _ => false) = 0 := by
    simp [cutValue, panelValue, cutPanel]
  rw [← h]
  exact cutValue_le_maxCut c _

/-! ## Two-state ensembles -/

section Ensembles

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- A *two-state* conformation at scale `t` in the direction `u`: every residue sits either at the
origin or at `t • u`.  This is the compact/extended exchange of a disordered region in its
simplest form, and it is the family used in `cutEnsemble_meanDist`. -/
def IsTwoState (t : ℝ) (u : E) (x : ℕ → E) : Prop := ∀ i, x i = 0 ∨ x i = t • u

open Classical in
/-- The split a two-state conformation induces on the residues. -/
noncomputable def stateSplit (x : ℕ → E) : Fin N → Bool :=
  fun i => if x (i : ℕ) = 0 then false else true

omit [NormedSpace ℝ E] in
theorem stateSplit_eq_false {x : ℕ → E} {i : Fin N} (h : x (i : ℕ) = 0) :
    stateSplit (N := N) x i = false := by
  classical
  simp [stateSplit, h]

omit [NormedSpace ℝ E] in
theorem stateSplit_eq_true {x : ℕ → E} {i : Fin N} (h : x (i : ℕ) ≠ 0) :
    stateSplit (N := N) x i = true := by
  classical
  simp [stateSplit, h]

theorem dist_eq_of_twoState {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t) {x : ℕ → E}
    (hx : IsTwoState t u x) (i j : Fin N) :
    dist (x (i : ℕ)) (x (j : ℕ)) = t * cutPanel (stateSplit (N := N) x) i j := by
  classical
  have hnorm : ‖t • u‖ = t := by
    rw [norm_smul, hu, mul_one, Real.norm_eq_abs, abs_of_nonneg ht]
  simp only [cutPanel]
  rcases hx (i : ℕ) with hi | hi <;> rcases hx (j : ℕ) with hj | hj
  · rw [stateSplit_eq_false hi, stateSplit_eq_false hj]
    simp [hi, hj]
  · by_cases h0 : (t • u : E) = 0
    · rw [stateSplit_eq_false hi, stateSplit_eq_false (by rw [hj, h0])]
      simp [hi, hj, h0]
    · rw [stateSplit_eq_false hi, stateSplit_eq_true (by rw [hj]; exact h0)]
      rw [hi, hj]
      simp [dist_eq_norm, hnorm]
  · by_cases h0 : (t • u : E) = 0
    · rw [stateSplit_eq_false (by rw [hi, h0]), stateSplit_eq_false hj]
      simp [hi, hj, h0]
    · rw [stateSplit_eq_true (by rw [hi]; exact h0), stateSplit_eq_false hj]
      rw [hi, hj]
      simp [dist_eq_norm, hnorm]
  · by_cases h0 : (t • u : E) = 0
    · rw [stateSplit_eq_false (by rw [hi, h0]), stateSplit_eq_false (by rw [hj, h0])]
      simp [hi, hj]
    · rw [stateSplit_eq_true (by rw [hi]; exact h0), stateSplit_eq_true (by rw [hj]; exact h0)]
      simp [hi, hj]

/-- The mean-distance panel of an ensemble, read on the residues `Fin N`. -/
noncomputable def ensPanel {M : ℕ} (w : Fin M → ℝ) (X : Fin M → ℕ → E) : Fin N → Fin N → ℝ :=
  fun i j => meanDist w X (i : ℕ) (j : ℕ)

/-- **No two-state ensemble beats the maximum cut.**  For every ensemble of two-state
conformations at scale `t`, the weighted mean-distance functional is at most `t · maxCut c`. -/
theorem twoState_panelValue_le {M : ℕ} {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t)
    (c : Fin N → Fin N → ℝ) (w : Fin M → ℝ) (X : Fin M → ℕ → E)
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a ≤ 1) (hX : ∀ a, IsTwoState t u (X a)) :
    panelValue c (ensPanel w X) ≤ t * maxCut c := by
  classical
  have h1 : ∀ i j : Fin N, c i j * ensPanel (N := N) w X i j
      = ∑ a, c i j * (w a * (t * cutPanel (stateSplit (N := N) (X a)) i j)) := by
    intro i j
    unfold ensPanel meanDist
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun a _ => by rw [dist_eq_of_twoState hu ht (hX a) i j]
  have hswap : ∀ F : Fin N → Fin M → ℝ, ∑ i, ∑ a, F i a = ∑ a, ∑ i, F i a :=
    fun F => Finset.sum_comm
  have hexp : panelValue c (ensPanel (N := N) w X)
      = ∑ a, w a * (t * cutValue c (stateSplit (N := N) (X a))) := by
    calc panelValue c (ensPanel (N := N) w X)
        = ∑ i : Fin N, ∑ j : Fin N, ∑ a,
            c i j * (w a * (t * cutPanel (stateSplit (N := N) (X a)) i j)) := by
          unfold panelValue
          exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => h1 i j
      _ = ∑ i : Fin N, ∑ a, ∑ j : Fin N,
            c i j * (w a * (t * cutPanel (stateSplit (N := N) (X a)) i j)) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ a, ∑ i : Fin N, ∑ j : Fin N,
            c i j * (w a * (t * cutPanel (stateSplit (N := N) (X a)) i j)) :=
          hswap (fun i a => ∑ j : Fin N,
            c i j * (w a * (t * cutPanel (stateSplit (N := N) (X a)) i j)))
      _ = ∑ a, w a * (t * cutValue c (stateSplit (N := N) (X a))) := by
          refine Finset.sum_congr rfl fun a _ => ?_
          simp only [cutValue, panelValue, Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [hexp]
  have hbound : ∀ a, w a * (t * cutValue c (stateSplit (N := N) (X a)))
      ≤ w a * (t * maxCut c) := by
    intro a
    have := cutValue_le_maxCut c (stateSplit (N := N) (X a))
    have ht' : t * cutValue c (stateSplit (N := N) (X a)) ≤ t * maxCut c :=
      mul_le_mul_of_nonneg_left this ht
    exact mul_le_mul_of_nonneg_left ht' (hw a)
  calc ∑ a, w a * (t * cutValue c (stateSplit (N := N) (X a)))
      ≤ ∑ a, w a * (t * maxCut c) := Finset.sum_le_sum fun a _ => hbound a
    _ = (∑ a, w a) * (t * maxCut c) := by rw [Finset.sum_mul]
    _ ≤ 1 * (t * maxCut c) :=
        mul_le_mul_of_nonneg_right hsum (mul_nonneg ht (maxCut_nonneg c))
    _ = t * maxCut c := one_mul _

/-- A single two-state conformation realising a prescribed split, as an ensemble. -/
theorem exists_twoState_ensemble_of_split {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t)
    (S : Fin N → Bool) :
    ∃ (M : ℕ) (w : Fin M → ℝ) (X : Fin M → ℕ → E),
      (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧ (∀ a, IsTwoState t u (X a)) ∧
      (∀ a k, dist (X a k) (X a (k + 1)) ≤ t) ∧
      (∀ i j : Fin N, ensPanel (N := N) w X i j = t * cutPanel S i j) := by
  classical
  set S' : ℕ → Bool := fun k => if h : k < N then S ⟨k, h⟩ else false with hS'
  refine ⟨1, fun _ => 1, fun _ => splitConfig S' t u, fun _ => zero_le_one, by simp, ?_, ?_, ?_⟩
  · intro a i
    simp only [splitConfig]
    by_cases h : S' i = true
    · exact Or.inr (by simp [h])
    · exact Or.inl (by simp [h])
  · intro a k
    rw [dist_splitConfig hu ht]
    split <;> simp [ht]
  · intro i j
    have hSi : S' (i : ℕ) = S i := by simp [hS', i.isLt]
    have hSj : S' (j : ℕ) = S j := by simp [hS', j.isLt]
    unfold ensPanel meanDist
    rw [Finset.sum_const]
    simp only [Finset.card_univ, Fintype.card_fin, one_smul, one_mul]
    rw [dist_splitConfig hu ht, hSi, hSj, cutPanel]
    split <;> simp

/-- **The best two-state ensemble is the maximum cut.**  For a weight matrix `c` on the residues
and a contour scale `t ≥ 0`, the greatest value of the weighted mean-distance functional over all
two-state ensembles at scale `t` is exactly `t · maxCut c`, and it is attained by a single
conformation.  Optimising an ensemble against a weighted panel target is therefore max cut, which
is NP-hard. -/
theorem maxCut_eq_greatest_twoState_panelValue {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t)
    (c : Fin N → Fin N → ℝ) :
    IsGreatest {v : ℝ | ∃ (M : ℕ) (w : Fin M → ℝ) (X : Fin M → ℕ → E),
        (∀ a, 0 ≤ w a) ∧ (∑ a, w a ≤ 1) ∧ (∀ a, IsTwoState t u (X a)) ∧
        v = panelValue c (ensPanel (N := N) w X)} (t * maxCut c) := by
  classical
  constructor
  · obtain ⟨S, hS⟩ := exists_cutValue_eq_maxCut c
    obtain ⟨M, w, X, hw, hsum, hX, -, hpanel⟩ := exists_twoState_ensemble_of_split (N := N) hu ht S
    refine ⟨M, w, X, hw, le_of_eq hsum, hX, ?_⟩
    have hval : panelValue c (ensPanel (N := N) w X) = t * cutValue c S := by
      have hstep : (∑ i : Fin N, ∑ j : Fin N, c i j * ensPanel (N := N) w X i j)
          = ∑ i : Fin N, ∑ j : Fin N, c i j * (t * cutPanel S i j) :=
        Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by rw [hpanel i j]
      unfold panelValue
      rw [hstep]
      simp only [cutValue, panelValue, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
    rw [hval, hS]
  · rintro v ⟨M, w, X, hw, hsum, hX, rfl⟩
    exact twoState_panelValue_le hu ht c w X hw hsum hX

end Ensembles

/-! ## Which panels are realisable -/

section Realisability

/-- A panel is *cut-realisable at scale `t`* when it is a convex combination (of total weight at
most one) of cut pseudometrics scaled by `t`: exactly the panels of two-state ensembles. -/
def CutRealisable (t : ℝ) (p : Fin N → Fin N → ℝ) : Prop :=
  ∃ lam : (Fin N → Bool) → ℝ, (∀ S, 0 ≤ lam S) ∧ (∑ S, lam S ≤ 1) ∧
    ∀ i j, p i j = ∑ S, lam S * (t * cutPanel S i j)

/-- Every cut-realisable panel passes every max-cut test. -/
theorem panelValue_le_of_cutRealisable {t : ℝ} (ht : 0 ≤ t) {p : Fin N → Fin N → ℝ}
    (hp : CutRealisable t p) (c : Fin N → Fin N → ℝ) : panelValue c p ≤ t * maxCut c := by
  classical
  obtain ⟨lam, hlam, hsum, hrep⟩ := hp
  have hswap : ∀ F : Fin N → (Fin N → Bool) → ℝ, ∑ i, ∑ S, F i S = ∑ S, ∑ i, F i S :=
    fun F => Finset.sum_comm
  have hexp : panelValue c p = ∑ S, lam S * (t * cutValue c S) := by
    calc panelValue c p
        = ∑ i : Fin N, ∑ j : Fin N, ∑ S, c i j * (lam S * (t * cutPanel S i j)) := by
          unfold panelValue
          refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
          rw [hrep i j, Finset.mul_sum]
      _ = ∑ i : Fin N, ∑ S, ∑ j : Fin N, c i j * (lam S * (t * cutPanel S i j)) :=
          Finset.sum_congr rfl fun i _ => Finset.sum_comm
      _ = ∑ S, ∑ i : Fin N, ∑ j : Fin N, c i j * (lam S * (t * cutPanel S i j)) :=
          hswap (fun i S => ∑ j : Fin N, c i j * (lam S * (t * cutPanel S i j)))
      _ = ∑ S, lam S * (t * cutValue c S) := by
          refine Finset.sum_congr rfl fun S _ => ?_
          simp only [cutValue, panelValue, Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring
  rw [hexp]
  calc ∑ S, lam S * (t * cutValue c S)
      ≤ ∑ _S : Fin N → Bool, lam _S * (t * maxCut c) :=
        Finset.sum_le_sum fun S _ =>
          mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (cutValue_le_maxCut c S) ht)
            (hlam S)
    _ = (∑ S, lam S) * (t * maxCut c) := by rw [Finset.sum_mul]
    _ ≤ 1 * (t * maxCut c) := mul_le_mul_of_nonneg_right hsum (mul_nonneg ht (maxCut_nonneg c))
    _ = t * maxCut c := one_mul _

/-! ### The realisable panels form a compact convex set -/

/-- The set of admissible mixing weights over the splits. -/
def simplexSet (N : ℕ) : Set ((Fin N → Bool) → ℝ) :=
  {lam | (∀ S, 0 ≤ lam S) ∧ ∑ S, lam S ≤ 1}

/-- Mixing the cut pseudometrics, as a linear map. -/
noncomputable def cutMap (t : ℝ) : ((Fin N → Bool) → ℝ) →ₗ[ℝ] (Fin N → Fin N → ℝ) where
  toFun lam := fun i j => ∑ S, lam S * (t * cutPanel S i j)
  map_add' x y := by
    funext i j
    simp [add_mul, Finset.sum_add_distrib]
  map_smul' a x := by
    funext i j
    simp only [RingHom.id_apply, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
    exact Finset.sum_congr rfl fun S _ => by ring

/-- The set of cut-realisable panels at scale `t`. -/
def cutRealisableSet (t : ℝ) : Set (Fin N → Fin N → ℝ) := {p | CutRealisable t p}

theorem cutRealisableSet_eq_image (t : ℝ) :
    cutRealisableSet (N := N) t = cutMap t '' simplexSet N := by
  ext p
  constructor
  · rintro ⟨lam, h1, h2, h3⟩
    exact ⟨lam, ⟨h1, h2⟩, by funext i j; exact (h3 i j).symm⟩
  · rintro ⟨lam, ⟨h1, h2⟩, rfl⟩
    exact ⟨lam, h1, h2, fun i j => rfl⟩

theorem convex_simplexSet : Convex ℝ (simplexSet N) := by
  rintro x ⟨hx1, hx2⟩ y ⟨hy1, hy2⟩ a b ha hb hab
  constructor
  · intro S
    have h : 0 ≤ a * x S + b * y S := by
      have := hx1 S; have := hy1 S; positivity
    simpa using h
  · have hsum : ∑ S, (a • x + b • y) S = a * (∑ S, x S) + b * (∑ S, y S) := by
      simp [Finset.sum_add_distrib, Finset.mul_sum]
    rw [hsum]
    nlinarith [hx2, hy2, ha, hb]

theorem isCompact_simplexSet : IsCompact (simplexSet N) := by
  apply Metric.isCompact_of_isClosed_isBounded
  · have h1 : IsClosed {lam : (Fin N → Bool) → ℝ | ∀ S, 0 ≤ lam S} := by
      have hEq : {lam : (Fin N → Bool) → ℝ | ∀ S, 0 ≤ lam S}
          = ⋂ S, {lam : (Fin N → Bool) → ℝ | 0 ≤ lam S} := by
        ext lam; simp [Set.mem_iInter]
      rw [hEq]
      exact isClosed_iInter (fun S => isClosed_le continuous_const (continuous_apply S))
    have h2 : IsClosed {lam : (Fin N → Bool) → ℝ | ∑ S, lam S ≤ 1} :=
      isClosed_le (continuous_finset_sum _ (fun S _ => continuous_apply S)) continuous_const
    exact h1.inter h2
  · refine Bornology.IsBounded.subset
      (Metric.isBounded_closedBall (x := (0 : (Fin N → Bool) → ℝ)) (r := 1)) ?_
    rintro lam ⟨hnn, hsum⟩
    simp only [Metric.mem_closedBall, dist_zero_right]
    refine (pi_norm_le_iff_of_nonneg (by norm_num)).mpr (fun S => ?_)
    have h1 : lam S ≤ ∑ S', lam S' := Finset.single_le_sum (fun S' _ => hnn S') (mem_univ S)
    rw [Real.norm_eq_abs, abs_of_nonneg (hnn S)]
    linarith

theorem isCompact_cutRealisableSet (t : ℝ) : IsCompact (cutRealisableSet (N := N) t) := by
  rw [cutRealisableSet_eq_image]
  exact isCompact_simplexSet.image (LinearMap.continuous_of_finiteDimensional _)

theorem isClosed_cutRealisableSet (t : ℝ) : IsClosed (cutRealisableSet (N := N) t) :=
  (isCompact_cutRealisableSet t).isClosed

theorem convex_cutRealisableSet (t : ℝ) : Convex ℝ (cutRealisableSet (N := N) t) := by
  rw [cutRealisableSet_eq_image]
  exact convex_simplexSet.linear_image (cutMap t)

/-! ### Separation: realisability is exactly a family of max-cut bounds -/

/-- The panel with a single one in position `(i, j)`. -/
def unitPanel (i j : Fin N) : Fin N → Fin N → ℝ :=
  fun a b => if a = i then (if b = j then 1 else 0) else 0

/-- Every continuous linear functional on panels is a weighted panel test. -/
theorem exists_weights_of_functional (f : (Fin N → Fin N → ℝ) →L[ℝ] ℝ) :
    ∃ c : Fin N → Fin N → ℝ, ∀ p, f p = panelValue c p := by
  refine ⟨fun i j => f (unitPanel i j), fun p => ?_⟩
  have hdec : p = ∑ i, ∑ j, p i j • unitPanel i j := by
    funext a b
    simp [unitPanel, Finset.sum_apply, Finset.sum_ite_eq]
  calc f p = f (∑ i, ∑ j, p i j • unitPanel i j) := by rw [← hdec]
    _ = ∑ i, ∑ j, p i j * f (unitPanel i j) := by
        rw [map_sum]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [map_sum]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [map_smul]
        simp
    _ = panelValue (fun i j => f (unitPanel i j)) p :=
        Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => mul_comm _ _

/-- A single cut, at scale `t`, is realisable. -/
theorem cutRealisable_cutPanel (t : ℝ) (S : Fin N → Bool) :
    CutRealisable t (fun i j => t * cutPanel S i j) := by
  classical
  refine ⟨fun S' => if S' = S then 1 else 0, fun S' => by positivity, by simp, fun i j => ?_⟩
  rw [Finset.sum_eq_single S]
  · simp
  · intro S' _ hne
    simp [hne]
  · intro h
    exact absurd (mem_univ S) h

/-- The value a weight matrix assigns to a single scaled cut. -/
theorem panelValue_cutPanel (t : ℝ) (c : Fin N → Fin N → ℝ) (S : Fin N → Bool) :
    panelValue c (fun i j => t * cutPanel S i j) = t * cutValue c S := by
  unfold panelValue cutValue
  simp only [panelValue, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => by ring

/-- **Realisability is exactly a family of max-cut bounds.**

A measured panel `p` is realisable by a two-state ensemble at contour scale `t` if and only if it
passes the max-cut test `∑_{i,j} c i j · p i j ≤ t · maxCut c` for every weight matrix `c`.  Each
such test is an instance of max cut, so certifying (or refuting) realisability means solving
max-cut instances on the residues. -/
theorem cutRealisable_iff_forall_weight {t : ℝ} (ht : 0 ≤ t) (p : Fin N → Fin N → ℝ) :
    CutRealisable t p ↔ ∀ c : Fin N → Fin N → ℝ, panelValue c p ≤ t * maxCut c := by
  constructor
  · intro hp c
    exact panelValue_le_of_cutRealisable ht hp c
  · intro hall
    by_contra hp
    have hnot : p ∉ cutRealisableSet (N := N) t := hp
    obtain ⟨f, u, hlt, hu⟩ :=
      geometric_hahn_banach_closed_point (convex_cutRealisableSet t) (isClosed_cutRealisableSet t)
        hnot
    obtain ⟨c, hc⟩ := exists_weights_of_functional f
    obtain ⟨S, hS⟩ := exists_cutValue_eq_maxCut c
    have hmem : (fun i j => t * cutPanel S i j) ∈ cutRealisableSet (N := N) t :=
      cutRealisable_cutPanel t S
    have h1 : f (fun i j => t * cutPanel S i j) < u := hlt _ hmem
    have h2 : f (fun i j => t * cutPanel S i j) = t * maxCut c := by
      rw [hc, panelValue_cutPanel, hS]
    have h3 : u < panelValue c p := by rw [← hc]; exact hu
    have h4 := hall c
    rw [h2] at h1
    linarith

/-- **The support function of the realisable panels is the maximum cut.**  Optimising a weighted
panel functional over realisable panels returns `t · maxCut c`, so an oracle for the optimum is an
oracle for max cut. -/
theorem maxCut_eq_greatest_panelValue_over_realisable {t : ℝ} (ht : 0 ≤ t)
    (c : Fin N → Fin N → ℝ) :
    IsGreatest {v : ℝ | ∃ p : Fin N → Fin N → ℝ, CutRealisable t p ∧ v = panelValue c p}
      (t * maxCut c) := by
  obtain ⟨S, hS⟩ := exists_cutValue_eq_maxCut c
  refine ⟨⟨fun i j => t * cutPanel S i j, cutRealisable_cutPanel t S, ?_⟩, ?_⟩
  · rw [panelValue_cutPanel, hS]
  · rintro v ⟨p, hp, rfl⟩
    exact panelValue_le_of_cutRealisable ht hp c

/-- **A refutation of realisability is a max-cut violation.**  A panel fails to be realisable
exactly when some weighting beats the maximum cut of that weighting: every certificate of
non-realisability is a max-cut instance together with a weighting that violates its bound. -/
theorem not_cutRealisable_iff_exists_weight {t : ℝ} (ht : 0 ≤ t) (p : Fin N → Fin N → ℝ) :
    ¬ CutRealisable t p ↔ ∃ c : Fin N → Fin N → ℝ, t * maxCut c < panelValue c p := by
  rw [cutRealisable_iff_forall_weight ht]
  push_neg
  rfl

end Realisability

/-! ## The realisable panels are exactly the two-state ensemble panels -/

section Bridge

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The panel of a two-state ensemble is cut-realisable. -/
theorem cutRealisable_ensPanel {M : ℕ} {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t)
    (w : Fin M → ℝ) (X : Fin M → ℕ → E) (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a ≤ 1)
    (hX : ∀ a, IsTwoState t u (X a)) :
    CutRealisable t (ensPanel (N := N) w X) := by
  classical
  refine ⟨fun S => ∑ a ∈ univ.filter (fun a => stateSplit (N := N) (X a) = S), w a, ?_, ?_, ?_⟩
  · intro S
    exact Finset.sum_nonneg fun a _ => hw a
  · rw [Finset.sum_fiberwise (univ : Finset (Fin M)) (fun a => stateSplit (N := N) (X a)) w]
    exact hsum
  · intro i j
    have hkey : ∀ S : Fin N → Bool,
        (∑ a ∈ univ.filter (fun a => stateSplit (N := N) (X a) = S), w a) * (t * cutPanel S i j)
          = ∑ a ∈ univ.filter (fun a => stateSplit (N := N) (X a) = S),
              w a * dist (X a (i : ℕ)) (X a (j : ℕ)) := by
      intro S
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun a ha => ?_
      have hS : stateSplit (N := N) (X a) = S := (Finset.mem_filter.mp ha).2
      rw [dist_eq_of_twoState hu ht (hX a) i j, hS]
    unfold ensPanel meanDist
    rw [← Finset.sum_fiberwise (univ : Finset (Fin M)) (fun a => stateSplit (N := N) (X a))
      (fun a => w a * dist (X a (i : ℕ)) (X a (j : ℕ)))]
    exact Finset.sum_congr rfl fun S _ => (hkey S).symm

/-- Every cut-realisable panel is the mean-distance panel of an ensemble of two-state chain
conformations with all bonds at most `t`. -/
theorem exists_twoState_ensemble_of_cutRealisable {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t)
    {p : Fin N → Fin N → ℝ} (hp : CutRealisable t p) :
    ∃ (M : ℕ) (w : Fin M → ℝ) (X : Fin M → ℕ → E),
      (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧ (∀ a, IsTwoState t u (X a)) ∧
      (∀ a k, dist (X a k) (X a (k + 1)) ≤ t) ∧
      (∀ i j : Fin N, ensPanel (N := N) w X i j = p i j) := by
  classical
  obtain ⟨lam, hlam, hsum, hrep⟩ := hp
  set r := Fintype.card (Fin N → Bool) with hr
  set e : (Fin N → Bool) ≃ Fin r := Fintype.equivFin _ with he
  set S' : (Fin N → Bool) → ℕ → Bool :=
    fun S k => if h : k < N then S ⟨k, h⟩ else false with hS'
  refine ⟨r + 1, Fin.cons (1 - ∑ S, lam S) (fun a => lam (e.symm a)),
    Fin.cons (fun _ => (0 : E)) (fun a => splitConfig (S' (e.symm a)) t u), ?_, ?_, ?_, ?_, ?_⟩
  · refine Fin.cases ?_ ?_
    · simpa using hsum
    · intro a; simpa using hlam _
  · rw [Fin.sum_cons]
    have : ∑ a : Fin r, lam (e.symm a) = ∑ S, lam S := Equiv.sum_comp e.symm lam
    rw [this]
    ring
  · refine Fin.cases ?_ ?_
    · intro k; exact Or.inl rfl
    · intro a k
      simp only [Fin.cons_succ, splitConfig]
      by_cases h : S' (e.symm a) k = true
      · exact Or.inr (by simp [h])
      · exact Or.inl (by simp [h])
  · refine Fin.cases ?_ ?_
    · intro k; simpa using ht
    · intro a k
      rw [Fin.cons_succ, dist_splitConfig hu ht]
      split <;> simp [ht]
  · intro i j
    unfold ensPanel meanDist
    rw [Fin.sum_univ_succ]
    simp only [Fin.cons_succ, Fin.cons_zero, dist_self, mul_zero, zero_add]
    have hterm : ∀ a : Fin r,
        lam (e.symm a) * dist (splitConfig (S' (e.symm a)) t u (i : ℕ))
            (splitConfig (S' (e.symm a)) t u (j : ℕ))
          = lam (e.symm a) * (t * cutPanel (e.symm a) i j) := by
      intro a
      rw [dist_splitConfig hu ht]
      have hSi : S' (e.symm a) (i : ℕ) = (e.symm a) i := by simp [hS', i.isLt]
      have hSj : S' (e.symm a) (j : ℕ) = (e.symm a) j := by simp [hS', j.isLt]
      rw [hSi, hSj, cutPanel]
      split <;> simp
    rw [Finset.sum_congr rfl fun a _ => hterm a]
    rw [Equiv.sum_comp e.symm (fun S => lam S * (t * cutPanel S i j))]
    exact (hrep i j).symm

/-- **The two descriptions agree**: a panel is cut-realisable at scale `t` exactly when it is the
mean-distance panel of an ensemble of two-state conformations at that scale. -/
theorem cutRealisable_iff_twoStateEnsemble {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t)
    (p : Fin N → Fin N → ℝ) :
    CutRealisable t p ↔
      ∃ (M : ℕ) (w : Fin M → ℝ) (X : Fin M → ℕ → E),
        (∀ a, 0 ≤ w a) ∧ (∑ a, w a ≤ 1) ∧ (∀ a, IsTwoState t u (X a)) ∧
        (∀ i j : Fin N, ensPanel (N := N) w X i j = p i j) := by
  constructor
  · intro hp
    obtain ⟨M, w, X, hw, hsum, hX, -, hpanel⟩ :=
      exists_twoState_ensemble_of_cutRealisable (N := N) hu ht hp
    exact ⟨M, w, X, hw, le_of_eq hsum, hX, hpanel⟩
  · rintro ⟨M, w, X, hw, hsum, hX, hpanel⟩
    have h := cutRealisable_ensPanel (N := N) hu ht w X hw hsum hX
    obtain ⟨lam, h1, h2, h3⟩ := h
    exact ⟨lam, h1, h2, fun i j => by rw [← hpanel i j]; exact h3 i j⟩

end Bridge

/-! ## Why the hardness is stated for two-state ensembles only

General ensembles realise strictly more panels than two-state ensembles do: an equilateral
triangle of three residues, a single Euclidean conformation with every bond at most `1`, has the
panel `p i j = 1` for `i ≠ j`, and that panel violates the max-cut test at the weighting `c ≡ 1`
(`6 > 4`).  So the cut cone is a proper subset of the ensemble panels at a fixed scale, and the
hardness proved above does not transfer to general ensemble realisability by inclusion alone. -/

section Strict

/-- Three points of the plane at mutual distance one. -/
noncomputable def triConfig : ℕ → EuclideanSpace ℝ (Fin 2) :=
  fun k => if k = 0 then !₂[0, 0] else if k = 1 then !₂[1, 0] else !₂[1 / 2, Real.sqrt 3 / 2]

/-- The mean-distance panel of the equilateral triangle. -/
def triPanel : Fin 3 → Fin 3 → ℝ := fun i j => if i = j then 0 else 1

theorem triConfig_zero : triConfig 0 = !₂[0, 0] := by norm_num [triConfig]

theorem triConfig_one : triConfig 1 = !₂[1, 0] := by norm_num [triConfig]

theorem triConfig_two : triConfig 2 = !₂[1 / 2, Real.sqrt 3 / 2] := by norm_num [triConfig]

theorem triConfig_add_two (n : ℕ) : triConfig (n + 2) = !₂[1 / 2, Real.sqrt 3 / 2] := by
  simp only [triConfig, if_neg (by omega : ¬ n + 2 = 0), if_neg (by omega : ¬ n + 2 = 1)]

theorem dist_triConfig_zero_one : dist (triConfig 0) (triConfig 1) = 1 := by
  rw [triConfig_zero, triConfig_one, EuclideanSpace.dist_eq]
  simp [Fin.sum_univ_two, Real.dist_eq]

theorem dist_triConfig_zero_two : dist (triConfig 0) (triConfig 2) = 1 := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  rw [triConfig_zero, triConfig_two, EuclideanSpace.dist_eq]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Real.dist_eq]
  rw [show |(0:ℝ) - 1 / 2| ^ 2 + |(0:ℝ) - Real.sqrt 3 / 2| ^ 2 = 1 by
    rw [sq_abs, sq_abs]; nlinarith [h3]]
  simp

theorem dist_triConfig_one_two : dist (triConfig 1) (triConfig 2) = 1 := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  rw [triConfig_one, triConfig_two, EuclideanSpace.dist_eq]
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Real.dist_eq]
  rw [show |(1:ℝ) - 1 / 2| ^ 2 + |(0:ℝ) - Real.sqrt 3 / 2| ^ 2 = 1 by
    rw [sq_abs, sq_abs]; nlinarith [h3]]
  simp

/-- Every bond of the triangular conformation has length at most one. -/
theorem dist_triConfig_succ_le (k : ℕ) : dist (triConfig k) (triConfig (k + 1)) ≤ 1 := by
  match k with
  | 0 => exact le_of_eq dist_triConfig_zero_one
  | 1 => exact le_of_eq dist_triConfig_one_two
  | (n + 2) =>
    rw [triConfig_add_two, show n + 2 + 1 = (n + 1) + 2 by ring, triConfig_add_two, dist_self]
    norm_num

/-- No split of three residues separates more than four ordered pairs. -/
theorem cutValue_ones_le_four (S : Fin 3 → Bool) : cutValue (fun _ _ => (1:ℝ)) S ≤ 4 := by
  simp only [cutValue, panelValue, cutPanel, one_mul, Fin.sum_univ_three]
  rcases Bool.eq_false_or_eq_true (S 0) with h0 | h0 <;>
    rcases Bool.eq_false_or_eq_true (S 1) with h1 | h1 <;>
      rcases Bool.eq_false_or_eq_true (S 2) with h2 | h2 <;>
        simp [h0, h1, h2] <;> norm_num

theorem maxCut_ones_le_four : maxCut (N := 3) (fun _ _ => (1:ℝ)) ≤ 4 :=
  Finset.sup'_le _ _ fun S _ => cutValue_ones_le_four S

theorem panelValue_ones_triPanel : panelValue (fun _ _ => (1:ℝ)) triPanel = 6 := by
  simp only [panelValue, triPanel, one_mul, Fin.sum_univ_three]
  norm_num [Fin.ext_iff]

/-- The equilateral panel is **not** cut-realisable: the all-ones weighting scores `6` on it while
no cut of three residues scores more than `4`. -/
theorem triPanel_not_cutRealisable : ¬ CutRealisable (N := 3) 1 triPanel := by
  rw [not_cutRealisable_iff_exists_weight (by norm_num)]
  refine ⟨fun _ _ => (1:ℝ), ?_⟩
  rw [panelValue_ones_triPanel, one_mul]
  exact lt_of_le_of_lt maxCut_ones_le_four (by norm_num)

/-- **The two-state family is strictly smaller than the ensemble family.**  A single Euclidean
conformation -- three residues at the corners of a unit equilateral triangle, every bond at most
`1` -- has a mean-distance panel that no two-state ensemble at scale `1` can produce.  Hardness of
the cut realisability problem therefore does not transfer to general ensemble realisability by
inclusion alone. -/
theorem exists_ensemble_panel_not_cutRealisable :
    ∃ (M : ℕ) (w : Fin M → ℝ) (X : Fin M → ℕ → EuclideanSpace ℝ (Fin 2)),
      (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧ (∀ a k, dist (X a k) (X a (k + 1)) ≤ 1) ∧
      (∀ i j : Fin 3, ensPanel (N := 3) w X i j = triPanel i j) ∧
      ¬ CutRealisable (N := 3) 1 triPanel := by
  refine ⟨1, fun _ => 1, fun _ => triConfig, fun _ => zero_le_one, by simp,
    fun _ k => dist_triConfig_succ_le k, ?_, triPanel_not_cutRealisable⟩
  intro i j
  have hval : ensPanel (N := 3) (fun _ : Fin 1 => (1:ℝ)) (fun _ => triConfig) i j
      = dist (triConfig (i : ℕ)) (triConfig (j : ℕ)) := by
    simp [ensPanel, meanDist]
  rw [hval]
  match i, j with
  | 0, 0 => simp [triPanel]
  | 0, 1 => simpa [triPanel] using dist_triConfig_zero_one
  | 0, 2 => simpa [triPanel] using dist_triConfig_zero_two
  | 1, 0 => rw [dist_comm]; simpa [triPanel] using dist_triConfig_zero_one
  | 1, 1 => simp [triPanel]
  | 1, 2 => simpa [triPanel] using dist_triConfig_one_two
  | 2, 0 => rw [dist_comm]; simpa [triPanel] using dist_triConfig_zero_two
  | 2, 1 => rw [dist_comm]; simpa [triPanel] using dist_triConfig_one_two
  | 2, 2 => simp [triPanel]

end Strict

end RequestProject.DistanceCutHardness
