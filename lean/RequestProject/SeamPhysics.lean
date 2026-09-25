/-
# Fragment pipelines III: which chains are free, and what a contact costs

The two previous files priced a fragment pipeline in information: the relative entropy of the
truth from the best pipeline model is the sum of the seam informations, and no pipeline does
better.  Left open is the physical question: *which chains have cheap seams?*  This file
answers it at both ends.

**Local chains are free.**  A chain whose statistical weight is a product of single-block and
nearest-neighbour-block factors — the transfer-matrix form shared by every helix–coil,
rotamer-library and nearest-neighbour lattice model of a disordered region — is exactly
Markov along its blocks (`isPipeline_of_isLocalChain`), so its total seam information is zero
(`seamInfoTotal_localChain`) and the pipeline reproduces it exactly (`mproj_localChain`).  A
fragment pipeline is not an approximation for such models; it is exact.  Everything a
pipeline loses therefore comes from couplings that reach beyond the neighbouring block.

**A contact across the seam costs an explicit amount.**  The shortest such coupling is a
contact between the two blocks flanking a seam — the sticker–sticker contact of a
sticker–spacer disordered region.  For the two-state chain of three blocks with a contact of
strength `t` between the outer blocks, `stickerSeam_cmi` computes the seam information in
closed form,

  `I = ½[(1+t)·log(1+t) + (1−t)·log(1−t)]`,

and `stickerSeam_cmi_ge` bounds it below by `t²/2`, while `stickerSeam_not_condIndep` shows
it is strictly positive whenever the contact is present.  Combined with the extensivity law
of the previous file this gives `sticker_chain_length_limit`: a chain carrying such a contact
at each of `n` seams admits **no** fragment-pipeline model of relative-entropy accuracy `eps`
once `n > 2·eps/t²`.  Fragment pipelines of a sticker–spacer disordered region have a maximum
usable length, set by the square of the contact strength.
-/
import Mathlib
import RequestProject.ChainPipelineLimits

namespace RequestProject.ChainPipeline

open Finset IDR.Pinsker RequestProject.Modular

universe u

variable {A : Type u} [Fintype A] [Nonempty A]

/-! ## Local chains: the transfer-matrix class -/

omit [Fintype A] [Nonempty A] in
/-- The first block of a chain. -/
def headOf : ∀ {n : ℕ}, Blocks A n → A
  | 0, x => x
  | (_ + 1), x => x.1

/-- The statistical weight of a chain with single-block factors `h` and nearest-neighbour
factors `K`: the transfer-matrix form. -/
def gibbsWeight (h : A → ℝ) (K : A → A → ℝ) : ∀ {n : ℕ}, Blocks A n → ℝ
  | 0, x => h x
  | (_ + 1), x => h x.1 * K x.1 (headOf x.2) * gibbsWeight h K x.2

omit [Fintype A] [Nonempty A] in
lemma gibbsWeight_succ (h : A → ℝ) (K : A → A → ℝ) {n : ℕ} (a : A) (w : Blocks A n) :
    gibbsWeight (n := n + 1) h K (a, w) = h a * K a (headOf w) * gibbsWeight h K w := rfl

/-- A **local chain**: an ensemble proportional to a transfer-matrix weight, possibly with an
extra factor on the first block — which is exactly what marginalising away the block before it
leaves behind.  This is the class of nearest-neighbour chain models of a disordered region. -/
def IsLocalChain {n : ℕ} (p : Blocks A n → ℝ) : Prop :=
  ∃ (m h : A → ℝ) (K : A → A → ℝ) (Z : ℝ), ∀ x, p x = m (headOf x) * gibbsWeight h K x / Z

omit [Nonempty A] in
/-- Marginalising away the first block of a local chain leaves a local chain: the transfer
matrix and the single-block factors are untouched, only the head factor changes. -/
lemma isLocalChain_dropFirst {n : ℕ} {p : Blocks A (n + 1) → ℝ} (hp : IsLocalChain p) :
    IsLocalChain (dropFirst p) := by
  obtain ⟨m, h, K, Z, hform⟩ := hp
  refine ⟨fun c => ∑ a : A, m a * h a * K a c, h, K, Z, fun w => ?_⟩
  rw [dropFirst]
  have hterm : ∀ a : A, p (a, w) = (m a * h a * K a (headOf w)) * gibbsWeight h K w / Z := by
    intro a
    rw [hform (a, w)]
    show m a * (h a * K a (headOf w) * gibbsWeight h K w) / Z = _
    ring
  rw [Finset.sum_congr rfl fun a _ => hterm a, ← Finset.sum_div, ← Finset.sum_mul]

omit [Nonempty A] in
/-- **Nearest-neighbour chain models are exactly modular.**  Every local chain is
conditionally independent across every seam, hence is an ensemble a fragment pipeline can
express exactly. -/
theorem isPipeline_of_isLocalChain : ∀ {n : ℕ} {p : Blocks A n → ℝ}, IsLocalChain p →
    IsPipeline p
  | 0, _, _ => trivial
  | 1, _, _ => trivial
  | (n + 2), p, hp => by
      obtain ⟨m, h, K, Z, hform⟩ := hp
      refine ⟨?_, isPipeline_of_isLocalChain (isLocalChain_dropFirst ⟨m, h, K, Z, hform⟩)⟩
      have hcur : cur p = fun a b (w : Blocks A n) =>
          (m a * h a * K a b / Z) * gibbsWeight (n := n + 1) h K (b, w) := by
        funext a b w
        show p (a, b, w) = (m a * h a * K a b / Z) * gibbsWeight (n := n + 1) h K (b, w)
        rw [hform (a, b, w)]
        show m a * (h a * K a b * gibbsWeight (n := n + 1) h K (b, w)) / Z = _
        ring
      rw [hcur]
      exact condIndep_of_product (fun a b => m a * h a * K a b / Z)
        (fun b w => gibbsWeight (n := n + 1) h K (b, w))

/-- A nearest-neighbour chain model costs nothing to build out of fragments. -/
theorem seamInfoTotal_localChain {n : ℕ} {p : Blocks A n → ℝ} (hp : ∀ x, 0 < p x)
    (hs : ∑ x, p x = 1) (hloc : IsLocalChain p) : seamInfoTotal p = 0 :=
  (seamInfoTotal_eq_zero_iff hp hs).2 (isPipeline_of_isLocalChain hloc)

/-- The fragment pipeline reproduces a nearest-neighbour chain model exactly. -/
theorem mproj_localChain {n : ℕ} {p : Blocks A n → ℝ} (hp : ∀ x, 0 < p x)
    (hloc : IsLocalChain p) : ∀ x, mproj p x = p x :=
  (mproj_eq_self_iff_isPipeline hp).2 (isPipeline_of_isLocalChain hloc)

/-! ## A sticker contact across the seam -/

/-- The `±1` value of a two-state block. -/
def sgn (b : Bool) : ℝ := if b then 1 else -1

/-- Three two-state blocks, uniform except for a contact of strength `t` between the two outer
blocks — the blocks that a cut at the middle block separates.  This is the minimal
sticker–spacer motif: two stickers that touch, with a spacer between them. -/
noncomputable def sticker (t : ℝ) : Bool → Bool → Bool → ℝ :=
  fun a _ c => (1 + t * sgn a * sgn c) / 8

/-- The same ensemble as a chain of three blocks. -/
noncomputable def stickerChain (t : ℝ) : Blocks Bool 2 → ℝ :=
  fun x => sticker t x.1 x.2.1 x.2.2

lemma cur_stickerChain (t : ℝ) : cur (stickerChain t) = sticker t := rfl

lemma sticker_pos {t : ℝ} (ht : |t| < 1) (a b c : Bool) : 0 < sticker t a b c := by
  have h1 : -1 < t := neg_lt_of_abs_lt ht
  have h2 : t < 1 := lt_of_abs_lt ht
  cases a <;> cases c <;> simp [sticker, sgn] <;> linarith

lemma sticker_sum (t : ℝ) : ∑ a : Bool, ∑ b : Bool, ∑ c : Bool, sticker t a b c = 1 := by
  simp [sticker, sgn]
  ring

lemma sticker_margXY (t : ℝ) (a b : Bool) : margXY (sticker t) a b = 1 / 4 := by
  cases a <;> simp [margXY, sticker, sgn] <;> ring

lemma sticker_margYZ (t : ℝ) (b c : Bool) : margYZ (sticker t) b c = 1 / 4 := by
  cases c <;> simp [margYZ, sticker, sgn] <;> ring

lemma sticker_margY (t : ℝ) (b : Bool) : margY (sticker t) b = 1 / 2 := by
  rw [margY_eq_sum_margXY]
  simp [sticker_margXY]
  norm_num

lemma sticker_glue (t : ℝ) (a b c : Bool) : glue (sticker t) a b c = 1 / 8 := by
  rw [glue, sticker_margXY, sticker_margYZ, sticker_margY]
  norm_num

/-- Pointwise, the likelihood ratio of the truth to its glued model is the contact factor. -/
lemma sticker_ratio (t : ℝ) (a b c : Bool) :
    sticker t a b c * margY (sticker t) b / (margXY (sticker t) a b * margYZ (sticker t) b c)
      = 1 + t * sgn a * sgn c := by
  rw [sticker_margXY, sticker_margYZ, sticker_margY, sticker]
  ring

/-- **The information cost of a contact across a seam, in closed form.**  A contact of
strength `t` between the blocks flanking a cut costs the pipeline exactly
`½[(1+t)·log(1+t) + (1−t)·log(1−t)]` nats at that seam. -/
theorem sticker_cmi (t : ℝ) :
    cmi (sticker t) = ((1 + t) * Real.log (1 + t) + (1 - t) * Real.log (1 - t)) / 2 := by
  have hcmi : cmi (sticker t)
      = ∑ a : Bool, ∑ b : Bool, ∑ c : Bool,
          sticker t a b c * Real.log (1 + t * sgn a * sgn c) := by
    rw [cmi]
    exact Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ =>
      Finset.sum_congr rfl fun c _ => by rw [sticker_ratio]
  rw [hcmi]
  simp only [Fintype.sum_bool, sticker, sgn]
  norm_num
  ring_nf

/-- The `ℓ¹` distance between the truth and its glued model is exactly the contact strength. -/
lemma sticker_l1 (t : ℝ) :
    ∑ x : Bool × Bool × Bool, |sticker t x.1 x.2.1 x.2.2 - glue (sticker t) x.1 x.2.1 x.2.2|
      = |t| := by
  have hterm : ∀ x : Bool × Bool × Bool,
      |sticker t x.1 x.2.1 x.2.2 - glue (sticker t) x.1 x.2.1 x.2.2| = |t| / 8 := by
    rintro ⟨a, b, c⟩
    have h1 : sticker t a b c - 1 / 8 = t * sgn a * sgn c / 8 := by rw [sticker]; ring
    rw [sticker_glue, h1, abs_div, abs_mul, abs_mul]
    cases a <;> cases c <;> norm_num [sgn]
  rw [Finset.sum_congr rfl fun x _ => hterm x]
  simp [Finset.sum_const, Fintype.card_prod]
  ring

/-- **A contact across the seam costs at least `t²/2`.**  Through Pinsker's inequality the
closed form of `sticker_cmi` is bounded below by the square of the contact strength. -/
theorem sticker_cmi_ge {t : ℝ} (ht : |t| < 1) : t ^ 2 / 2 ≤ cmi (sticker t) := by
  have hp : ∀ a b c, 0 < sticker t a b c := sticker_pos ht
  have hg : ∀ x : Bool × Bool × Bool, 0 < glue (sticker t) x.1 x.2.1 x.2.2 := by
    intro x; rw [sticker_glue]; norm_num
  have hps : ∑ x : Bool × Bool × Bool, sticker t x.1 x.2.1 x.2.2 = 1 := by
    rw [Fintype.sum_prod_type]
    refine (Finset.sum_congr rfl fun a _ => by rw [Fintype.sum_prod_type]).trans ?_
    exact sticker_sum t
  have hgs : ∑ x : Bool × Bool × Bool, glue (sticker t) x.1 x.2.1 x.2.2 = 1 := by
    simp [sticker_glue]
  have hpin := pinskerG (p := fun x : Bool × Bool × Bool => sticker t x.1 x.2.1 x.2.2)
    (q := fun x : Bool × Bool × Bool => glue (sticker t) x.1 x.2.1 x.2.2)
    (fun x => le_of_lt (hp _ _ _)) hg hps hgs
  rw [sticker_l1] at hpin
  have hkl : klG (fun x : Bool × Bool × Bool => sticker t x.1 x.2.1 x.2.2)
      (fun x : Bool × Bool × Bool => glue (sticker t) x.1 x.2.1 x.2.2) = cmi (sticker t) :=
    klG_glue_eq_cmi (p := sticker t) (fun a b c => le_of_lt (hp a b c))
  rw [hkl, sq_abs] at hpin
  linarith

/-- A contact across the seam genuinely breaks modularity: the ensemble is not conditionally
independent across the cut, so no fragment pipeline reproduces it. -/
theorem sticker_not_condIndep {t : ℝ} (ht : |t| < 1) (ht0 : t ≠ 0) :
    ¬ CondIndep (sticker t) := by
  intro hci
  have hzero : cmi (sticker t) = 0 :=
    (cmi_eq_zero_iff_condIndep (p := sticker t)
      (fun a b c => le_of_lt (sticker_pos ht a b c)) (sticker_sum t)).2 hci
  have hge := sticker_cmi_ge ht
  have hpos : 0 < t ^ 2 := by positivity
  linarith

/-- The three-block sticker chain, read in the pipeline framework: its total seam information
is the cost of its single seam. -/
theorem seamInfoTotal_stickerChain (t : ℝ) : seamInfoTotal (stickerChain t) = cmi (sticker t) := by
  rw [seamInfoTotal_succ2, cur_stickerChain, seamInfoTotal_one, add_zero]

/-! ## The maximum length of a sticker–spacer fragment pipeline -/

/-- **The length ceiling for a sticker–spacer chain.**  If every seam of the chain flanks a
contact of strength at least `t`, so that each seam carries at least `t²/2` of information,
then no fragment pipeline model of the chain — however its fragments were fitted or joined —
is within relative entropy `eps` of the truth once the chain has more than `2·eps/t²` seams.

The three ingredients are physical: nearest-neighbour models are free, a contact across a cut
costs `t²/2`, and the costs of the cuts add.  Together they say that a disordered region with
sticker–sticker contacts cannot be captured by fragments beyond a definite length. -/
theorem sticker_chain_length_limit {n : ℕ} {t eps : ℝ} (ht : t ≠ 0) {p m : Blocks A (n + 1) → ℝ}
    (hp : ∀ x, 0 < p x) (hm : ∀ x, 0 < m x) (hps : ∑ x, p x = 1) (hms : ∑ x, m x = 1)
    (hpipe : IsPipeline m) (hfl : SeamFloor (t ^ 2 / 2) p) (hacc : klG p m ≤ eps) :
    (n : ℝ) ≤ 2 * eps / t ^ 2 := by
  have hc : 0 < t ^ 2 / 2 := by positivity
  have h := pipeline_length_limit hc hp hm hps hms hpipe hfl hacc
  calc (n : ℝ) ≤ eps / (t ^ 2 / 2) := h
    _ = 2 * eps / t ^ 2 := by field_simp

end RequestProject.ChainPipeline
