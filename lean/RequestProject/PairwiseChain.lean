/-
# The chain instance: on a chain compared in adjacent pairs, block noise costs `4/b`

`RequestProject.PairwiseProtocol` proves the general law: under block-correlated label noise the
pair-level error rate of a within-group pairwise protocol obeys

    ν_pair ≤ (boundary mass / total scored pairs) · ν_label.

This file computes the boundary fraction in the case the law was designed for: `n` residues on a
chain, compared in **adjacent pairs** (the groups are `{0,1}, {2,3}, …`), with the annotation
erring by flipping whole blocks of `b` consecutive residues.  A block carries at most four
boundary-crossing scored pairs (only the two pairs cut by its two ends), there are `n / b`
blocks and at least `n` scored pairs, so

* `chain_boundaryDeg_le_four` -- each block carries at most `4` boundary-crossing pairs;
* `chain_pairwise_noise_bound` -- `ν_pair ≤ (4 / b) · ν_label`;
* `chain_capacity_ceiling_gain` -- equivalently, the capacity ceiling of the pairwise
  leaderboard, `1 / (2 ν_pair)`, is at least `b / 4` times the residue-level ceiling
  `1 / (2 ν_label)`.

So the protocol's noise -- and with it the number of methods a challenge can honestly order --
improves *linearly in the length of the correlated blocks*.  The gain comes from the structure
of the errors, not from better annotation.
-/
import Mathlib
import RequestProject.PairwiseProtocol

set_option autoImplicit false

namespace IDR
namespace Pairwise

open Finset

/-! ## 1.  Arithmetic of blocks on a chain -/

private lemma nat_div_eq_iff {v b K : ℕ} (hb : 0 < b) :
    v / b = K ↔ K * b ≤ v ∧ v < (K + 1) * b := by
  rw [show (K * b ≤ v ↔ K ≤ v / b) from (Nat.le_div_iff_mul_le hb).symm,
    show (v < (K + 1) * b ↔ v / b < K + 1) from (Nat.div_lt_iff_lt_mul hb).symm]
  omega

/-- The two residues at a cut are the last of one block and the first of the next. -/
private lemma boundary_pair_form {b K u : ℕ} (hne : u / b ≠ (u + 1) / b)
    (hK : u / b = K ∨ (u + 1) / b = K) :
    (u + 1 = K * b) ∨ (u + 1 = (K + 1) * b) := by
  have hsucc : (u + 1) / b = u / b + if b ∣ u + 1 then 1 else 0 := Nat.succ_div
  by_cases hdvd : b ∣ u + 1
  · rw [if_pos hdvd] at hsucc
    have hq : u + 1 = ((u + 1) / b) * b := (Nat.div_mul_cancel hdvd).symm
    rcases hK with h | h
    · right; rw [hq, hsucc, h]
    · left; rw [hq, h]
  · rw [if_neg hdvd, Nat.add_zero] at hsucc
    exact absurd hsucc.symm hne

/-- The comparison groups of the chain protocol: adjacent residues `{2k, 2k+1}`. -/
def chainGrp (n : ℕ) : Fin n → ℕ := fun x => (x : ℕ) / 2

/-- The noise blocks of the chain: `b` consecutive residues. -/
def chainBlk (n b : ℕ) : Fin n → ℕ := fun x => (x : ℕ) / b

variable {n b : ℕ}

private lemma block_lt (hbn : b ∣ n) (x : Fin n) : (x : ℕ) / b < n / b :=
  Nat.div_lt_div_of_lt_of_dvd hbn x.isLt

/-- The blocks that occur are exactly `0, …, n/b − 1`. -/
lemma chain_blockSet (hb : 0 < b) (hbn : b ∣ n) :
    blockSet (chainBlk n b) = Finset.range (n / b) := by
  ext K
  simp only [blockSet, Finset.mem_image, Finset.mem_univ, true_and, Finset.mem_range]
  constructor
  · rintro ⟨x, hx⟩
    exact hx ▸ block_lt hbn x
  · intro hK
    have hle : (K + 1) * b ≤ n := by
      calc (K + 1) * b ≤ (n / b) * b := Nat.mul_le_mul_right b hK
        _ = n := Nat.div_mul_cancel hbn
    have hlt : K * b < n := by nlinarith [hb, hle]
    exact ⟨⟨K * b, hlt⟩, by simp [chainBlk, Nat.mul_div_cancel _ hb]⟩

/-- Every block holds exactly `b` residues. -/
lemma chain_blockFiber_card (hb : 0 < b) (hbn : b ∣ n) {K : ℕ}
    (hK : K ∈ blockSet (chainBlk n b)) : (blockFiber (chainBlk n b) K).card = b := by
  rw [chain_blockSet hb hbn, Finset.mem_range] at hK
  have hle : (K + 1) * b ≤ n := by
    calc (K + 1) * b ≤ (n / b) * b := Nat.mul_le_mul_right b hK
      _ = n := Nat.div_mul_cancel hbn
  have himg : (blockFiber (chainBlk n b) K).image (Fin.val) = Finset.Ico (K * b) ((K + 1) * b) := by
    ext v
    simp only [Finset.mem_image, blockFiber, Finset.mem_filter, Finset.mem_univ, true_and,
      Finset.mem_Ico, chainBlk]
    constructor
    · rintro ⟨x, hx, rfl⟩
      exact (nat_div_eq_iff hb).mp hx
    · intro hv
      exact ⟨⟨v, lt_of_lt_of_le hv.2 hle⟩, (nat_div_eq_iff hb).mpr hv, rfl⟩
  have := congrArg Finset.card himg
  rw [Finset.card_image_of_injective _ Fin.val_injective, Nat.card_Ico] at this
  rw [this]
  cases b with
  | zero => omega
  | succ c => simp [Nat.succ_mul]

/-! ## 2.  A block carries at most four boundary-crossing pairs -/

/-- **Only the two ends of a block are exposed.**  At most four scored pairs -- the two ordered
pairs cut at each end of the block -- straddle a boundary of a given block. -/
lemma chain_boundaryDeg_le_four (K : ℕ) :
    boundaryDeg (chainGrp n) (chainBlk n b) K ≤ 4 := by
  classical
  set E4 : Finset (ℕ × ℕ) :=
    {(K * b - 1, K * b), (K * b, K * b - 1), ((K + 1) * b - 1, (K + 1) * b),
      ((K + 1) * b, (K + 1) * b - 1)} with hE4
  have hcard4 : E4.card ≤ 4 := by
    calc E4.card ≤ ({(K * b, K * b - 1), ((K + 1) * b - 1, (K + 1) * b),
            ((K + 1) * b, (K + 1) * b - 1)} : Finset (ℕ × ℕ)).card + 1 :=
          Finset.card_insert_le _ _
      _ ≤ (({((K + 1) * b - 1, (K + 1) * b), ((K + 1) * b, (K + 1) * b - 1)} :
            Finset (ℕ × ℕ)).card + 1) + 1 := by
          gcongr
          exact Finset.card_insert_le _ _
      _ ≤ ((({((K + 1) * b, (K + 1) * b - 1)} : Finset (ℕ × ℕ)).card + 1) + 1) + 1 := by
          gcongr
          exact Finset.card_insert_le _ _
      _ ≤ 4 := by simp
  rw [boundaryDeg]
  refine le_trans (Finset.card_le_card_of_injOn
    (fun p : Fin n × Fin n => ((p.1 : ℕ), (p.2 : ℕ))) ?_ ?_) hcard4
  · intro p hp
    simp only [Finset.coe_filter, Set.mem_setOf_eq, boundaryPairs, Finset.mem_filter,
      mem_scoredPairs] at hp
    obtain ⟨⟨⟨hne, hgrp⟩, hbnd⟩, hK⟩ := hp
    have hvne : (p.1 : ℕ) ≠ (p.2 : ℕ) := fun h => hne (Fin.val_injective h)
    simp only [chainGrp] at hgrp
    have hadj : (p.2 : ℕ) = (p.1 : ℕ) + 1 ∨ (p.1 : ℕ) = (p.2 : ℕ) + 1 := by omega
    simp only [chainBlk] at hbnd hK
    simp only [hE4, Finset.coe_insert, Set.mem_insert_iff, Finset.coe_singleton,
      Set.mem_singleton_iff, Prod.mk.injEq]
    rcases hadj with h | h
    · have hform := boundary_pair_form (u := (p.1 : ℕ)) (by rw [← h]; exact hbnd)
        (by rw [← h]; exact hK)
      rcases hform with hf | hf
      · left; constructor <;> omega
      · right; right; left; constructor <;> omega
    · have hform := boundary_pair_form (u := (p.2 : ℕ)) (by rw [← h]; exact hbnd.symm)
        (by rw [← h]; exact hK.symm)
      rcases hform with hf | hf
      · right; left; constructor <;> omega
      · right; right; right; constructor <;> omega
  · intro p _ q _ h
    simp only [Prod.mk.injEq] at h
    exact Prod.ext (Fin.val_injective h.1) (Fin.val_injective h.2)

/-! ## 3.  The chain has at least `n` scored pairs -/

/-- The partner of a residue in the adjacent-pair protocol. -/
def chainPartner (h2 : 2 ∣ n) (x : Fin n) : Fin n :=
  ⟨if (x : ℕ) % 2 = 0 then (x : ℕ) + 1 else (x : ℕ) - 1, by
    have hx := x.isLt
    obtain ⟨m, rfl⟩ := h2
    split <;> omega⟩

lemma chainPartner_val (h2 : 2 ∣ n) (x : Fin n) :
    ((chainPartner h2 x : Fin n) : ℕ) = if (x : ℕ) % 2 = 0 then (x : ℕ) + 1 else (x : ℕ) - 1 :=
  rfl

/-- Each residue is compared with its partner, so the protocol scores at least `n` pairs. -/
lemma chain_card_scoredPairs_ge (h2 : 2 ∣ n) :
    n ≤ (scoredPairs (chainGrp n)).card := by
  classical
  have hmain : (Finset.univ : Finset (Fin n)).card ≤ (scoredPairs (chainGrp n)).card := by
    refine Finset.card_le_card_of_injOn (fun x => (x, chainPartner h2 x)) ?_ ?_
    · intro x _
      have hp := chainPartner_val h2 x
      have hx := x.isLt
      obtain ⟨m, hm⟩ := h2
      refine Finset.mem_coe.mpr (mem_scoredPairs.mpr ⟨?_, ?_⟩)
      · intro h
        have hv := congrArg (fun y : Fin n => (y : ℕ)) h
        simp only at hv
        rw [hp] at hv
        split at hv <;> omega
      · simp only [chainGrp]
        rw [hp]
        split <;> omega
    · intro x _ y _ h
      exact (Prod.ext_iff.mp h).1
  simpa using hmain

/-! ## 4.  The bound -/

/-- **On a chain compared in adjacent pairs, block-correlated noise of block length `b` costs
only `4/b` of its residue-level rate.**  With the truth constant on the blocks and the
annotation flipping whole blocks,

    ν_pair ≤ (4 / b) · ν_label.

The pairwise protocol converts correlated label errors into a pair-level error rate smaller by
the length of the correlated block. -/
theorem chain_pairwise_noise_bound (hn : 0 < n) (hb : 0 < b) (h2 : 2 ∣ n) (hbn : b ∣ n)
    {T L : Finset (Fin n)} (hT : BlockConstant (chainBlk n b) T)
    (hE : BlockConstant (chainBlk n b) (symmDiff T L)) :
    (pairDiscord T L (chainGrp n) : ℝ) / (scoredPairs (chainGrp n)).card
      ≤ (4 / b) * ((symmDiff T L).card / n) := by
  classical
  have hPn : n ≤ (scoredPairs (chainGrp n)).card := chain_card_scoredPairs_ge h2
  have hP : 0 < (scoredPairs (chainGrp n)).card := lt_of_lt_of_le hn hPn
  have hcardn : Fintype.card (Fin n) = n := Fintype.card_fin n
  have hint := card_mul_pairDiscord_le (b := b) (D := 4) (grp := chainGrp n)
    (fun K hK => chain_blockFiber_card hb hbn hK) hT hE
    (fun K _ => chain_boundaryDeg_le_four K)
  rw [hcardn, chain_blockSet hb hbn, Finset.card_range] at hint
  -- pass to the reals
  have hnR : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hbR : (0:ℝ) < (b : ℝ) := by exact_mod_cast hb
  have hPR : (0:ℝ) < ((scoredPairs (chainGrp n)).card : ℝ) := by exact_mod_cast hP
  have hPnR : (n : ℝ) ≤ ((scoredPairs (chainGrp n)).card : ℝ) := by exact_mod_cast hPn
  have hdivb : ((n / b : ℕ) : ℝ) * (b : ℝ) = (n : ℝ) := by
    have : (n / b) * b = n := Nat.div_mul_cancel hbn
    exact_mod_cast congrArg (fun m : ℕ => (m : ℝ)) this
  have hintR : (n : ℝ) * (pairDiscord T L (chainGrp n) : ℝ)
      ≤ 4 * ((n / b : ℕ) : ℝ) * ((symmDiff T L).card : ℝ) := by exact_mod_cast hint
  have hd0 : (0:ℝ) ≤ (pairDiscord T L (chainGrp n) : ℝ) := Nat.cast_nonneg _
  -- `ν_pair ≤ d / n` and `d ≤ 4 |E| / b`
  have hstep1 : (pairDiscord T L (chainGrp n) : ℝ) / ((scoredPairs (chainGrp n)).card : ℝ)
      ≤ (pairDiscord T L (chainGrp n) : ℝ) / (n : ℝ) :=
    div_le_div_of_nonneg_left hd0 hnR hPnR
  have hstep2 : (pairDiscord T L (chainGrp n) : ℝ) / (n : ℝ)
      ≤ (4 / b) * ((symmDiff T L).card / n) := by
    rw [div_le_iff₀ hnR]
    have hb' : (b:ℝ) * ((n / b : ℕ) : ℝ) = (n:ℝ) := by rw [mul_comm]; exact hdivb
    have hkey : (b:ℝ) * ((n : ℝ) * (pairDiscord T L (chainGrp n) : ℝ))
        ≤ (b:ℝ) * (4 * ((n / b : ℕ) : ℝ) * ((symmDiff T L).card : ℝ)) := by
      exact mul_le_mul_of_nonneg_left hintR hbR.le
    have hexp : (b:ℝ) * (4 * ((n / b : ℕ) : ℝ) * ((symmDiff T L).card : ℝ))
        = 4 * (n:ℝ) * ((symmDiff T L).card : ℝ) := by
      rw [show (b:ℝ) * (4 * ((n / b : ℕ) : ℝ) * ((symmDiff T L).card : ℝ))
          = 4 * ((b:ℝ) * ((n / b : ℕ) : ℝ)) * ((symmDiff T L).card : ℝ) by ring, hb']
    rw [hexp] at hkey
    have hgoal : (4 / (b:ℝ)) * (((symmDiff T L).card : ℝ) / n) * n
        = 4 * ((symmDiff T L).card : ℝ) / b := by
      field_simp
    rw [hgoal, le_div_iff₀ hbR]
    nlinarith [hkey, hnR]
  exact hstep1.trans hstep2

/-- **The capacity ceiling improves linearly in the block length.**  In the same setting, the
number of methods the pairwise leaderboard can order, `1 / (2 ν_pair)`, is at least `b / 4`
times the residue-level ceiling `1 / (2 ν_label)`. -/
theorem chain_capacity_ceiling_gain (hn : 0 < n) (hb : 0 < b) (h2 : 2 ∣ n) (hbn : b ∣ n)
    {T L : Finset (Fin n)} (hT : BlockConstant (chainBlk n b) T)
    (hE : BlockConstant (chainBlk n b) (symmDiff T L))
    (hd : 0 < pairDiscord T L (chainGrp n)) (hEpos : 0 < (symmDiff T L).card) :
    ((b : ℝ) / 4) * (1 / (2 * (((symmDiff T L).card : ℝ) / n)))
      ≤ 1 / (2 * ((pairDiscord T L (chainGrp n) : ℝ) / (scoredPairs (chainGrp n)).card)) := by
  classical
  have hbound := chain_pairwise_noise_bound hn hb h2 hbn hT hE
  have hPn : n ≤ (scoredPairs (chainGrp n)).card := chain_card_scoredPairs_ge h2
  have hP : 0 < (scoredPairs (chainGrp n)).card := lt_of_lt_of_le hn hPn
  have hnR : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hbR : (0:ℝ) < (b : ℝ) := by exact_mod_cast hb
  have hPR : (0:ℝ) < ((scoredPairs (chainGrp n)).card : ℝ) := by exact_mod_cast hP
  have hdR : (0:ℝ) < (pairDiscord T L (chainGrp n) : ℝ) := by exact_mod_cast hd
  have hER : (0:ℝ) < ((symmDiff T L).card : ℝ) := by exact_mod_cast hEpos
  set nupair : ℝ := (pairDiscord T L (chainGrp n) : ℝ) / (scoredPairs (chainGrp n)).card with hnp
  set nulab : ℝ := ((symmDiff T L).card : ℝ) / n with hnl
  have hnp0 : 0 < nupair := by rw [hnp]; positivity
  have hnl0 : 0 < nulab := by rw [hnl]; positivity
  have hle : nupair ≤ (4 / b) * nulab := hbound
  have h1 : 1 / (2 * ((4 / (b:ℝ)) * nulab)) ≤ 1 / (2 * nupair) := by
    apply one_div_le_one_div_of_le (by positivity)
    exact mul_le_mul_of_nonneg_left hle (by norm_num)
  refine le_trans (le_of_eq ?_) h1
  field_simp

/-- **Aligned blocks cost nothing at all.**  If the noise blocks have even length they are
unions of comparison groups, no scored pair is cut, and the pairwise protocol reproduces the
truth exactly however many blocks the annotation flips: `ν_pair = 0`. -/
theorem chain_aligned_no_discordance (hbe : 2 ∣ b) {T L : Finset (Fin n)}
    (hT : BlockConstant (chainBlk n b) T) (hE : BlockConstant (chainBlk n b) (symmDiff T L)) :
    pairDiscord T L (chainGrp n) = 0 := by
  classical
  have hempty : boundaryPairs (chainGrp n) (chainBlk n b) = ∅ := by
    rw [Finset.eq_empty_iff_forall_notMem]
    intro p hp
    simp only [boundaryPairs, Finset.mem_filter, mem_scoredPairs, chainGrp, chainBlk] at hp
    obtain ⟨⟨hne, hgrp⟩, hbnd⟩ := hp
    have hvne : (p.1 : ℕ) ≠ (p.2 : ℕ) := fun h => hne (Fin.val_injective h)
    obtain ⟨c, rfl⟩ := hbe
    have hkey : ∀ u : ℕ, u / (2 * c) ≠ (u + 1) / (2 * c) → u % 2 = 0 → False := by
      intro u hne' hu
      rcases boundary_pair_form (K := u / (2 * c)) hne' (Or.inl rfl) with hf | hf
      · exact absurd (show 2 ∣ u + 1 from ⟨u / (2 * c) * c, by rw [hf]; ring⟩) (by omega)
      · exact absurd (show 2 ∣ u + 1 from ⟨(u / (2 * c) + 1) * c, by rw [hf]; ring⟩) (by omega)
    rcases (show (p.2 : ℕ) = (p.1 : ℕ) + 1 ∨ (p.1 : ℕ) = (p.2 : ℕ) + 1 by omega) with h | h
    · exact hkey (p.1 : ℕ) (by rw [← h]; exact hbnd) (by omega)
    · exact hkey (p.2 : ℕ) (by rw [← h]; exact hbnd.symm) (by omega)
  have hsub := discordPairs_subset (blk := chainBlk n b) (grp := chainGrp n) hT hE
  rw [hempty] at hsub
  simp only [Finset.filter_empty, Finset.subset_empty] at hsub
  rw [pairDiscord, hsub, Finset.card_empty]


/-! ## 5.  A worked instance -/

/-- **Nothing above is vacuous.**  Six residues in three comparison groups `{0,1}, {2,3}, {4,5}`,
noise blocks `{0,1,2}` and `{3,4,5}` of length `b = 3`; the truth is the first block, and the
annotation errs by flipping the whole second block.  Half the residues are mislabelled
(`ν_label = 3/6`), and exactly one of the six scored pairs is corrupted (`ν_pair = 1/6`) -- the
pair `(2,3)` cut by the block boundary.  The bound `ν_pair ≤ (4/3)·ν_label` holds with room to
spare, and the pairwise protocol survives an annotation that destroys the residue-level one. -/
theorem chain_worked_instance :
    BlockConstant (chainBlk 6 3) ({0, 1, 2} : Finset (Fin 6)) ∧
      BlockConstant (chainBlk 6 3)
        (symmDiff ({0, 1, 2} : Finset (Fin 6)) ({0, 1, 2, 3, 4, 5} : Finset (Fin 6))) ∧
      (symmDiff ({0, 1, 2} : Finset (Fin 6)) ({0, 1, 2, 3, 4, 5} : Finset (Fin 6))).card = 3 ∧
      (scoredPairs (chainGrp 6)).card = 6 ∧
      pairDiscord ({0, 1, 2} : Finset (Fin 6)) ({0, 1, 2, 3, 4, 5} : Finset (Fin 6))
        (chainGrp 6) = 1 := by
  refine ⟨fun x y h => ?_, fun x y h => ?_, by decide, by decide, by decide⟩
  · revert h; revert x y; decide
  · revert h; revert x y; decide

end Pairwise
end IDR
