/-
# Fragment pipelines V: where the cut goes matters at first order

`RequestProject.SeamCoarsening` proved that merging two adjacent seams never costs more.  This
file shows the law has teeth: there is a chain — the minimal sticker–spacer motif with a spare
block attached — on which

* cutting at the block that separates the two stickers costs
  `½[(1+t)·log(1+t) + (1−t)·log(1−t)] ≥ t²/2` (`seamInfoTotal_stickerPad`), while
* keeping those two blocks in one fragment and cutting one block later costs **nothing**
  (`cmi_pairCur_stickerPad`), and the coarse pipeline is exact.

Same number of fragments, same total length of chain, one block of overlap moved: an
unmodellable chain becomes exactly modellable.  For a disordered region the practical reading
is that cut placement is not a second-order tuning choice — a cut that separates two residues
in contact is paid for in full, and a cut one block away may be free.
-/
import Mathlib
import RequestProject.SeamPhysics
import RequestProject.SeamCoarsening

namespace RequestProject.ChainPipeline

open Finset IDR.Pinsker RequestProject.Modular

/-! ## The sticker motif with a spare block -/

/-- The three-block sticker motif of `RequestProject.SeamPhysics` with an independent uniform
fourth block appended: a contact between blocks `0` and `2`, and nothing else. -/
noncomputable def stickerPad (t : ℝ) : Blocks Bool 3 → ℝ :=
  fun x => sticker t x.1 x.2.1 x.2.2.1 / 2

lemma stickerPad_apply (t : ℝ) (a b c d : Bool) :
    stickerPad t (a, b, c, d) = sticker t a b c / 2 := rfl

lemma stickerPad_pos {t : ℝ} (ht : |t| < 1) (x : Blocks Bool 3) : 0 < stickerPad t x := by
  obtain ⟨a, b, c, d⟩ := x
  rw [stickerPad_apply]
  exact div_pos (sticker_pos ht a b c) (by norm_num)

lemma stickerPad_sum (t : ℝ) : ∑ x : Blocks Bool 3, stickerPad t x = 1 := by
  rw [sum_blocks_succ3]
  have h : ∀ a b c : Bool, ∑ d : Blocks Bool 0, stickerPad t (a, b, c, d)
      = sticker t a b c := by
    intro a b c
    rw [sum_blocks_zero (fun d : Blocks Bool 0 => stickerPad t (a, b, c, d))]
    simp [stickerPad_apply]
    ring
  calc ∑ a : Bool, ∑ b : Bool, ∑ c : Bool, ∑ d : Blocks Bool 0, stickerPad t (a, b, c, d)
      = ∑ a : Bool, ∑ b : Bool, ∑ c : Bool, sticker t a b c :=
        Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ =>
          Finset.sum_congr rfl fun c _ => h a b c
    _ = 1 := sticker_sum t

/-! ## Cutting between the stickers: the full price -/

lemma stickerPad_margXY (t : ℝ) (a b : Bool) : margXY (cur (stickerPad t)) a b = 1 / 4 := by
  have h : margXY (cur (stickerPad t)) a b
      = ∑ c : Bool, ∑ d : Bool, stickerPad t (a, b, c, d) := by
    rw [margXY, sum_blocks_succ (fun z : Blocks Bool 1 => cur (stickerPad t) a b z)]
    exact Finset.sum_congr rfl fun c _ =>
      sum_blocks_zero (fun d : Blocks Bool 0 => stickerPad t (a, b, c, d))
  rw [h]
  simp only [stickerPad_apply, Fintype.sum_bool]
  have := sticker_margXY t a b
  rw [margXY, Fintype.sum_bool] at this
  linarith

lemma stickerPad_margY (t : ℝ) (b : Bool) : margY (cur (stickerPad t)) b = 1 / 2 := by
  rw [margY_eq_sum_margXY]
  simp [stickerPad_margXY]
  norm_num

lemma stickerPad_margYZ (t : ℝ) (b c d : Bool) :
    margYZ (cur (stickerPad t)) b (c, d) = 1 / 8 := by
  have h : margYZ (cur (stickerPad t)) b (c, d) = ∑ a : Bool, sticker t a b c / 2 := rfl
  rw [h, ← Finset.sum_div]
  have := sticker_margYZ t b c
  rw [margYZ] at this
  rw [this]
  norm_num

/-- Cutting the padded motif at the block between the two stickers costs exactly what the
contact costs. -/
theorem cmi_cur_stickerPad (t : ℝ) : cmi (cur (stickerPad t)) = cmi (sticker t) := by
  have hratio : ∀ (a b c d : Bool),
      cur (stickerPad t) a b (c, d) * margY (cur (stickerPad t)) b
        / (margXY (cur (stickerPad t)) a b * margYZ (cur (stickerPad t)) b (c, d))
        = 1 + t * sgn a * sgn c := by
    intro a b c d
    rw [stickerPad_margXY, stickerPad_margY, stickerPad_margYZ,
      show cur (stickerPad t) a b (c, d) = sticker t a b c / 2 from rfl, sticker]
    ring
  have hexp : cmi (cur (stickerPad t))
      = ∑ a : Bool, ∑ b : Bool, ∑ c : Bool, ∑ d : Bool,
          sticker t a b c / 2 * Real.log (1 + t * sgn a * sgn c) := by
    rw [cmi]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    rw [sum_blocks_succ (fun z : Blocks Bool 1 => cur (stickerPad t) a b z *
      Real.log (cur (stickerPad t) a b z * margY (cur (stickerPad t)) b
        / (margXY (cur (stickerPad t)) a b * margYZ (cur (stickerPad t)) b z)))]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [sum_blocks_zero (fun d : Blocks Bool 0 => cur (stickerPad t) a b (c, d) *
      Real.log (cur (stickerPad t) a b (c, d) * margY (cur (stickerPad t)) b
        / (margXY (cur (stickerPad t)) a b * margYZ (cur (stickerPad t)) b (c, d))))]
    exact Finset.sum_congr rfl fun d _ => by rw [hratio a b c d]; rfl
  rw [hexp, sticker_cmi]
  simp only [Fintype.sum_bool, sticker, sgn]
  norm_num
  ring_nf

/-- The chain that starts after the seam is uniform: everything beyond the contact is
structureless. -/
lemma dropFirst_stickerPad (t : ℝ) (y : Blocks Bool 2) : dropFirst (stickerPad t) y = 1 / 8 := by
  obtain ⟨b, c, d⟩ := y
  have h : dropFirst (stickerPad t) (b, c, d) = ∑ a : Bool, sticker t a b c / 2 := rfl
  rw [h, ← Finset.sum_div]
  have := sticker_margYZ t b c
  rw [margYZ] at this
  rw [this]
  norm_num

/-- **The total price of the padded motif**: the contact across the first seam, and nothing
more. -/
theorem seamInfoTotal_stickerPad (t : ℝ) :
    seamInfoTotal (stickerPad t) = cmi (sticker t) := by
  have hci : CondIndep (cur (dropFirst (stickerPad t))) := by
    have hform : cur (dropFirst (stickerPad t))
        = fun (_ : Bool) (_ : Bool) (_ : Blocks Bool 0) => (1 / 8 : ℝ) * 1 := by
      funext b c d
      rw [show cur (dropFirst (stickerPad t)) b c d
        = dropFirst (stickerPad t) ((b, c, d) : Blocks Bool 2) from rfl,
        dropFirst_stickerPad]
      ring
    rw [hform]
    exact condIndep_of_product _ _
  have hsum : ∑ b : Bool, ∑ c : Bool, ∑ d : Blocks Bool 0,
      cur (dropFirst (stickerPad t)) b c d = 1 := by
    have h : ∑ b : Bool, ∑ c : Bool, ∑ d : Blocks Bool 0,
        cur (dropFirst (stickerPad t)) b c d
        = ∑ y : Blocks Bool 2, dropFirst (stickerPad t) y :=
      (sum_blocks_succ2 (dropFirst (stickerPad t))).symm
    rw [h, sum_dropFirst]
    exact stickerPad_sum t
  have hcmi2 : cmi (cur (dropFirst (stickerPad t))) = 0 :=
    (cmi_eq_zero_iff_condIndep (p := cur (dropFirst (stickerPad t)))
      (fun b c d => by
        rw [show cur (dropFirst (stickerPad t)) b c d
          = dropFirst (stickerPad t) ((b, c, d) : Blocks Bool 2) from rfl,
          dropFirst_stickerPad]
        norm_num) hsum).2 hci
  rw [seamInfoTotal_succ2, cmi_cur_stickerPad, seamInfoTotal_succ2, hcmi2,
    seamInfoTotal_one, add_zero, add_zero]

/-! ## Cutting one block later: free -/

/-- **The coarse cut is free.**  Keeping the two stickers in one fragment and cutting at the
next block instead, the seam information vanishes: the pipeline that cuts there is exact. -/
theorem cmi_pairCur_stickerPad {t : ℝ} (ht : |t| < 1) : cmi (pairCur (stickerPad t)) = 0 := by
  have hci : CondIndep (pairCur (stickerPad t)) := by
    have hform : pairCur (stickerPad t)
        = fun (ab : Bool × Bool) (c : Bool) (_ : Blocks Bool 0) =>
            sticker t ab.1 ab.2 c * (1 / 2 : ℝ) := by
      funext ab c d
      rw [show pairCur (stickerPad t) ab c d = stickerPad t (ab.1, ab.2, c, d) from rfl,
        stickerPad_apply]
      ring
    rw [hform]
    exact condIndep_of_product _ _
  have hsum : ∑ ab : Bool × Bool, ∑ c : Bool, ∑ d : Blocks Bool 0,
      pairCur (stickerPad t) ab c d = 1 := by
    have h : ∑ ab : Bool × Bool, ∑ c : Bool, ∑ d : Blocks Bool 0,
        pairCur (stickerPad t) ab c d = ∑ x : Blocks Bool 3, stickerPad t x :=
      (sum_blocks_pair (stickerPad t)).symm
    rw [h]
    exact stickerPad_sum t
  exact (cmi_eq_zero_iff_condIndep (p := pairCur (stickerPad t))
    (fun ab c d => le_of_lt (stickerPad_pos ht _)) hsum).2 hci

/-- **Cut placement matters at first order.**  On one and the same chain, the pipeline that
cuts between the two stickers pays at least `t²/2`, while the pipeline that keeps them
together and cuts one block later pays nothing at all — and is therefore exact.  A cut that
separates residues in contact is paid for in full; the same cut moved one block can be
free. -/
theorem cut_placement_strict {t : ℝ} (ht : |t| < 1) (ht0 : t ≠ 0) :
    cmi (pairCur (stickerPad t)) = 0 ∧
      t ^ 2 / 2 ≤ seamInfoTotal (stickerPad t) ∧
      cmi (pairCur (stickerPad t)) < seamInfoTotal (stickerPad t) := by
  have h0 := cmi_pairCur_stickerPad ht
  have h1 : t ^ 2 / 2 ≤ seamInfoTotal (stickerPad t) := by
    rw [seamInfoTotal_stickerPad t]
    exact sticker_cmi_ge ht
  have h2 : 0 < t ^ 2 / 2 := by positivity
  exact ⟨h0, h1, by linarith⟩

end RequestProject.ChainPipeline
