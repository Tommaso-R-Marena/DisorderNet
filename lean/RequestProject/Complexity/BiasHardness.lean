/-
# The recalibration decision problem, and the end of the chain

`biasThreshold` is the decision problem a benchmark actually faces: given a residue table (labels,
proteins, integer scores) and an allowance `d`, is there a per-protein bias bringing the pooled
Mann–Whitney statistic within `d/2` pairs of the ceiling `U_within + |between|`?

`lopBiasReduction` turns a weighted linear ordering instance into such a table, using the
construction of `AUCHardness.lean` renumbered to `Fin N`; `biasThreshold_hard` chains it with the
reductions from independent set, CNF satisfiability, circuit satisfiability and the NP verifier
definition.  The result is a machine-checked NP-hardness proof of the recalibration question with
no cited classical input.
-/
import RequestProject.Complexity.LOPHardness
import RequestProject.Complexity.Relabel

set_option autoImplicit false
set_option synthInstance.maxSize 1000

namespace IDR.Complexity

open Finset IDR.GroupedAUC IDR.GroupedAUC.Hardness

/-- An instance of the recalibration decision problem: a residue table with integer scores and an
allowed deficit `d`, measured in half-pairs. -/
structure TableInst where
  /-- Number of residues. -/
  N : ℕ
  /-- Number of proteins. -/
  K : ℕ
  /-- Labels. -/
  lab : Fin N → Bool
  /-- Protein of each residue. -/
  grp : Fin N → Fin K
  /-- Integer scores. -/
  sc : Fin N → ℤ
  /-- The allowed deficit from the ceiling, in half-pairs. -/
  d : ℤ

/-- The real-valued score function of a table. -/
def TableInst.scr (T : TableInst) : Fin T.N → ℝ := fun i => (T.sc i : ℝ)

/-- Size of a table instance: residues, proteins, scores and target, all in unary. -/
def TableInst.size (T : TableInst) : ℕ :=
  T.N + T.K + (∑ i : Fin T.N, (T.sc i).natAbs) + T.d.natAbs

/-- **The recalibration decision problem**: can some per-protein bias come within `d/2` pairs of
the ceiling? -/
def biasThreshold : Problem where
  Inst := TableInst
  size := TableInst.size
  Yes := fun T => ∃ b : Fin T.K → ℝ,
    U (withinPairs T.lab T.grp) T.scr + ((betweenPairs T.lab T.grp).card : ℝ) - (T.d : ℝ) / 2
      ≤ U (allPairs T.lab) (shift T.grp b T.scr)

namespace LopBias

variable (I : LopInst)

/-- The weight matrix with its diagonal cleared; this changes nothing about the optimum. -/
def Wz : Fin I.K → Fin I.K → ℕ := fun k l => if k = l then 0 else I.W k l

theorem Wz_diag (k : Fin I.K) : Wz I k k = 0 := by simp [Wz]

theorem lopValue_Wz (pi : Equiv.Perm (Fin I.K)) : lopValue (Wz I) pi = lopValue I.W pi := by
  unfold lopValue
  refine Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun l _ => ?_
  by_cases h : k = l
  · subst h; simp
  · simp [Wz, h]

theorem lopOpt_Wz : lopOpt (Wz I) = lopOpt I.W := by
  unfold lopOpt
  exact Finset.sup_congr rfl fun pi _ => lopValue_Wz I pi

/-- The total weight of the instance: the number of slots of the constructed table. -/
def slots : ℕ := slotCount I.K (Wz I)

/-- The background multiplicity that pins the optimum inside the box while keeping the table
polynomially large. -/
noncomputable def mult : ℕ := (exists_good_multiplicity_card_le I.K (Wz I)).choose

/-- The residues of the constructed table. -/
abbrev Itm' : Type := Itm I.K (mult I) (Wz I)

/-- The number of residues of the constructed table. -/
noncomputable def size' : ℕ := Fintype.card (Itm' I)

/-- The renumbering of the residues. -/
noncomputable def enum : Itm' I ≃ Fin (size' I) := Fintype.equivFin _

theorem mult_spec :
    2 * (Fintype.card (Itm' I) * slotCount I.K (Wz I)) ≤ mult I * mult I ∧
      Fintype.card (Itm' I) ≤ 26 * (I.K + 1) ^ 2 * (slotCount I.K (Wz I) + 1) :=
  (exists_good_multiplicity_card_le I.K (Wz I)).choose_spec

/-- The table built from a linear ordering instance. -/
noncomputable def tableOf : TableInst where
  N := size' I
  K := I.K
  lab := fun j => lab I.K (mult I) (Wz I) ((enum I).symm j)
  grp := fun j => grp I.K (mult I) (Wz I) ((enum I).symm j)
  sc := fun j => sc I.K (mult I) (Wz I) ((enum I).symm j)
  d := 2 * (((betweenPairs (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I))).card : ℤ)
        - ((basePairs I.K (mult I) (Wz I)).card : ℤ) - (I.target : ℤ))

/-- **Correctness**: the table reaches the target deficit exactly when the linear ordering
instance meets its target. -/
theorem tableOf_correct : lopProblem.Yes I ↔ biasThreshold.Yes (tableOf I) := by
  classical
  have hd : ((tableOf I).d : ℝ) / 2
      = ((betweenPairs (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I))).card : ℝ)
        - ((basePairs I.K (mult I) (Wz I)).card : ℝ) - (I.target : ℝ) := by
    simp only [tableOf]
    push_cast
    ring
  have hbet : ((betweenPairs (tableOf I).lab (tableOf I).grp).card : ℝ)
      = ((betweenPairs (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I))).card : ℝ) := by
    exact_mod_cast congrArg (Nat.cast : ℕ → ℝ)
      (card_betweenPairs_relabel (enum I) (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I)))
  have hwit : U (withinPairs (tableOf I).lab (tableOf I).grp) (tableOf I).scr
      = U (withinPairs (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I)))
          (scr I.K (mult I) (Wz I)) :=
    U_withinPairs_relabel (enum I) (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I))
      (scr I.K (mult I) (Wz I))
  have hall : ∀ b : Fin I.K → ℝ,
      U (allPairs (tableOf I).lab) (shift (tableOf I).grp b (tableOf I).scr)
        = U (allPairs (lab I.K (mult I) (Wz I)))
            (shift (grp I.K (mult I) (Wz I)) b (scr I.K (mult I) (Wz I))) := fun b =>
    U_allPairs_relabel (enum I) (lab I.K (mult I) (Wz I))
      (shift (grp I.K (mult I) (Wz I)) b (scr I.K (mult I) (Wz I)))
  have hiff : biasThreshold.Yes (tableOf I) ↔
      ∃ b : Fin I.K → ℝ,
        U (withinPairs (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I)))
            (scr I.K (mult I) (Wz I))
          + ((basePairs I.K (mult I) (Wz I)).card : ℝ) + (I.target : ℝ)
        ≤ U (allPairs (lab I.K (mult I) (Wz I)))
            (shift (grp I.K (mult I) (Wz I)) b (scr I.K (mult I) (Wz I))) := by
    show (∃ b : Fin (tableOf I).K → ℝ, _) ↔ _
    refine exists_congr fun b => ?_
    rw [hwit, hbet, hd, hall b]
    constructor <;> intro h <;> linarith
  rw [hiff, bias_threshold_iff_lop (Wz_diag I) (mult_spec I).1 I.target, lopOpt_Wz]
  rfl

/-! ### Size of the constructed table -/

/-- The number of slots is the total weight of the instance. -/
theorem slotCount_eq (K : ℕ) (W : Fin K → Fin K → ℕ) :
    slotCount K W = ∑ k : Fin K, ∑ l : Fin K, W k l := by
  rw [slotCount, Fintype.card_sigma]
  simp [Fintype.sum_prod_type]

theorem slotScore_eq (K : ℕ) (W : Fin K → Fin K → ℕ) (σ : Slot K W) :
    slotScore K W σ = (((4 * K + 6) * ((Fintype.equivFin (Slot K W) σ : ℕ) + 1) : ℕ) : ℤ) := by
  rw [slotScore, boxN]
  push_cast
  ring

theorem sc_natAbs_le (K M : ℕ) (W : Fin K → Fin K → ℕ) (i : Itm K M W) :
    (sc K M W i).natAbs ≤ (4 * K + 6) * slotCount K W + 2 * K + 3 := by
  set P := (4 * K + 6) * slotCount K W with hP
  rcases i with ⟨⟨k, m⟩, c⟩ | ⟨σ, side⟩
  · cases c
    · have : sc K M W (Sum.inl ((k, m), false)) = -((2 * K + 2 : ℕ) : ℤ) := by
        simp [sc, boxN]
      rw [this, Int.natAbs_neg, Int.natAbs_natCast]
      omega
    · have : sc K M W (Sum.inl ((k, m), true)) = 0 := rfl
      rw [this]; simp
  · have hidx : ((Fintype.equivFin (Slot K W) σ : ℕ) + 1) ≤ slotCount K W := by
      have := (Fintype.equivFin (Slot K W) σ).isLt
      rw [slotCount]
      omega
    have hmul : (4 * K + 6) * ((Fintype.equivFin (Slot K W) σ : ℕ) + 1) ≤ P :=
      Nat.mul_le_mul_left _ hidx
    cases side
    · have : sc K M W (Sum.inr (σ, false)) = slotScore K W σ + 1 := rfl
      rw [this, slotScore_eq]
      rw [show ((((4 * K + 6) * ((Fintype.equivFin (Slot K W) σ : ℕ) + 1) : ℕ) : ℤ) + 1)
            = (((4 * K + 6) * ((Fintype.equivFin (Slot K W) σ : ℕ) + 1) + 1 : ℕ) : ℤ) by push_cast; ring]
      rw [Int.natAbs_natCast]
      omega
    · have : sc K M W (Sum.inr (σ, true)) = slotScore K W σ := rfl
      rw [this, slotScore_eq, Int.natAbs_natCast]
      omega

/-- **The table is polynomially large.** -/
theorem tableOf_size_le : (tableOf I).size ≤ 10000 * (I.size + 1) ^ 6 := by
  classical
  set s := I.size with hs
  set T := slotCount I.K (Wz I) with hT
  set N := Fintype.card (Itm' I) with hN
  set Bc := (betweenPairs (lab I.K (mult I) (Wz I)) (grp I.K (mult I) (Wz I))).card with hBc
  set Cc := (basePairs I.K (mult I) (Wz I)).card with hCc
  have hK : I.K ≤ s := by rw [hs, LopInst.size]; omega
  have htg : I.target ≤ s := by rw [hs, LopInst.size]; omega
  have hTs : T ≤ s := by
    have h1 : T = ∑ k : Fin I.K, ∑ l : Fin I.K, Wz I k l := slotCount_eq _ _
    have h2 : ∑ k : Fin I.K, ∑ l : Fin I.K, Wz I k l ≤ ∑ k : Fin I.K, ∑ l : Fin I.K, I.W k l :=
      Finset.sum_le_sum fun k _ => Finset.sum_le_sum fun l _ => by rw [Wz]; split <;> simp
    rw [hs, LopInst.size]
    omega
  have hNs : N ≤ 26 * (s + 1) ^ 3 := by
    have hc := (mult_spec I).2
    calc N ≤ 26 * (I.K + 1) ^ 2 * (T + 1) := hc
      _ ≤ 26 * (s + 1) ^ 2 * (s + 1) :=
          Nat.mul_le_mul (Nat.mul_le_mul_left _ (Nat.pow_le_pow_left (by omega) 2)) (by omega)
      _ = 26 * (s + 1) ^ 3 := by ring
  have hscb : ∀ i, (sc I.K (mult I) (Wz I) i).natAbs ≤ 10 * (s + 1) ^ 2 := by
    intro i
    refine le_trans (sc_natAbs_le _ _ _ i) ?_
    have h1 : (4 * I.K + 6) * T ≤ (4 * s + 6) * s := Nat.mul_le_mul (by omega) hTs
    nlinarith [h1, hK]
  have hsum : (∑ j : Fin (tableOf I).N, ((tableOf I).sc j).natAbs) ≤ N * (10 * (s + 1) ^ 2) := by
    calc (∑ j : Fin (tableOf I).N, ((tableOf I).sc j).natAbs)
        ≤ ∑ _j : Fin (tableOf I).N, 10 * (s + 1) ^ 2 := Finset.sum_le_sum fun j _ => hscb _
      _ = N * (10 * (s + 1) ^ 2) := by
            simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
            rfl
  have hB : Bc ≤ N * N := by
    have h : Bc ≤ Fintype.card (Itm' I × Itm' I) := by
      rw [hBc, ← Finset.card_univ]
      exact Finset.card_le_univ _
    rw [Fintype.card_prod] at h
    exact h
  have hC : Cc ≤ N * N := by
    have h : Cc ≤ Fintype.card (Itm' I × Itm' I) := by
      rw [hCc, ← Finset.card_univ]
      exact Finset.card_le_univ _
    rw [Fintype.card_prod] at h
    exact h
  have hdeq : (tableOf I).d = 2 * ((Bc : ℤ) - (Cc : ℤ) - (I.target : ℤ)) := rfl
  have hdd : ((tableOf I).d).natAbs ≤ 2 * (Bc + Cc + I.target) := by rw [hdeq]; omega
  have hsize : (tableOf I).size
      = N + I.K + (∑ j : Fin (tableOf I).N, ((tableOf I).sc j).natAbs)
        + ((tableOf I).d).natAbs := rfl
  rw [hsize]
  set A := (s + 1) ^ 6 with hA6
  have e3 : (s + 1) ^ 3 ≤ A := Nat.pow_le_pow_right (by omega) (by norm_num)
  have e5 : (s + 1) ^ 5 ≤ A := Nat.pow_le_pow_right (by omega) (by norm_num)
  have e1 : s ≤ A := by
    calc s ≤ (s + 1) ^ 1 := by simp
      _ ≤ A := Nat.pow_le_pow_right (by omega) (by norm_num)
  have t1 : N ≤ 26 * A := le_trans hNs (Nat.mul_le_mul_left _ e3)
  have t3 : (∑ j : Fin (tableOf I).N, ((tableOf I).sc j).natAbs) ≤ 260 * A := by
    refine le_trans hsum ?_
    calc N * (10 * (s + 1) ^ 2) ≤ (26 * (s + 1) ^ 3) * (10 * (s + 1) ^ 2) :=
          Nat.mul_le_mul_right _ hNs
      _ = 260 * (s + 1) ^ 5 := by ring
      _ ≤ 260 * A := Nat.mul_le_mul_left _ e5
  have hNN : N * N ≤ 676 * A := by
    calc N * N ≤ (26 * (s + 1) ^ 3) * (26 * (s + 1) ^ 3) := Nat.mul_le_mul hNs hNs
      _ = 676 * A := by rw [hA6]; ring
  omega

end LopBias

/-- **Weighted linear ordering reduces to the recalibration decision problem.** -/
noncomputable def lopBiasReduction : Reduction lopProblem biasThreshold where
  map := LopBias.tableOf
  correct := LopBias.tableOf_correct
  deg := 6
  const := 10000
  size_le := LopBias.tableOf_size_le

/-- **The recalibration decision problem is NP-hard**, with every link of the chain — from the
verifier definition of NP through circuit satisfiability, CNF satisfiability, independent set and
weighted linear ordering — machine-checked. -/
theorem biasThreshold_hard : Hard NPProblem biasThreshold :=
  lopProblem_hard.trans lopBiasReduction

end IDR.Complexity
