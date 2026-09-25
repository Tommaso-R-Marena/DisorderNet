/-
# The pairwise protocol: capacity, and why block-correlated label noise costs it almost nothing

`RequestProject.LabelNoise` and `RequestProject.BenchmarkCapacityLabels` price a benchmark that
scores a method residue by residue against an annotation: a comparison is certified only when
the margin exceeds twice the annotation error, and the number of methods a benchmark can order
is therefore capped by `1 / (2 · error rate)`.

This file changes the *protocol*, not the labels.  Instead of scoring residues, score **ordered
pairs of residues inside a group**: for each pair `(i, j)` of distinct residues of the same
group, the reference asserts the verdict "`i` is disordered and `j` is not", and a method is
scored on how many of those verdicts it reproduces.  Two things are proved.

* `pairwise_capacity` -- the capacity theorem for that protocol.  Writing `ν_pair` for the
  fraction of scored pairs on which the annotation's verdict differs from the truth's, a family
  of methods certified at the pair level has at most `⌈1 / (2 ν_pair)⌉` members.  The pairwise
  protocol obeys exactly the same law as the residue protocol, with its own noise rate.

* `correlated_noise_reduces_pair_discordance` -- the reason the protocol is worth changing.
  Suppose the annotation errors are **block-correlated**: the truth is constant on the blocks of
  a partition (regions, not residues, are ordered or disordered) and the annotation errs by
  flipping whole blocks.  Then a scored pair can only be corrupted if it *straddles a block
  boundary*, and a count gives

      ν_pair ≤ (boundary mass / total scored pairs) · ν_label,

  where the boundary mass is the number of boundary-crossing scored pairs the blocks carry
  (`D · #blocks` for a per-block bound `D`) and `ν_label = |T Δ L| / #residues` is the ordinary
  residue-level error rate.  Nothing here is probabilistic: the bound is a deterministic
  counting statement about which pairs block-correlated noise can touch.

The mechanism is structural, and has nothing to do with disorder: whenever noise is constant on
the cells of a partition and the statistic is a within-cell comparison, only the boundary is
exposed.  `RequestProject.PairwiseChain` draws the consequence on a chain.
-/
import Mathlib
import RequestProject.LabelNoise
import RequestProject.BenchmarkCapacity
import RequestProject.BenchmarkCapacityLabels

set_option autoImplicit false

namespace IDR
namespace Pairwise

open Finset
open IDR.LabelNoise
open IDR.BenchCapacity

variable {α : Type*} [Fintype α] [DecidableEq α] {β γ : Type*} [DecidableEq β] [DecidableEq γ]

/-! ## 1.  The protocol -/

/-- The **scored pairs** of a protocol: ordered pairs of distinct residues lying in the same
group.  (`grp` assigns each residue to a group -- a protein, a domain, a window.) -/
def scoredPairs (grp : α → γ) : Finset (α × α) :=
  Finset.univ.filter (fun p => p.1 ≠ p.2 ∧ grp p.1 = grp p.2)

/-- The **verdict** a labelling `S` returns on an ordered pair: "the first residue is disordered
and the second is not". -/
def Verdict (S : Finset α) (p : α × α) : Prop := p.1 ∈ S ∧ p.2 ∉ S

instance decidableVerdict (S : Finset α) (p : α × α) : Decidable (Verdict S p) := by
  unfold Verdict; infer_instance

/-- The scored pairs on which the two labellings return different verdicts. -/
def discordPairs (T L : Finset α) (grp : α → γ) : Finset (α × α) :=
  (scoredPairs grp).filter (fun p => ¬ (Verdict T p ↔ Verdict L p))

/-- **Pair discordance**: the number of scored pairs the annotation gets wrong.  Divided by the
number of scored pairs this is the rate `ν_pair`. -/
def pairDiscord (T L : Finset α) (grp : α → γ) : ℕ := (discordPairs T L grp).card

lemma mem_scoredPairs {grp : α → γ} {p : α × α} :
    p ∈ scoredPairs grp ↔ p.1 ≠ p.2 ∧ grp p.1 = grp p.2 := by
  simp [scoredPairs]

lemma mem_discordPairs {T L : Finset α} {grp : α → γ} {p : α × α} :
    p ∈ discordPairs T L grp ↔ p ∈ scoredPairs grp ∧ ¬ (Verdict T p ↔ Verdict L p) := by
  simp [discordPairs]

/-! ## 2.  Capacity of the pairwise protocol -/

/-- The type of scored pairs, the "targets" of the pairwise benchmark. -/
abbrev ScoredPair (grp : α → γ) := {p : α × α // p ∈ scoredPairs grp}

/-- The reference answer key of the pairwise protocol induced by a residue labelling: the set of
scored pairs whose verdict is positive. -/
def pairLabel (S : Finset α) (grp : α → γ) : Finset (ScoredPair grp) :=
  Finset.univ.filter (fun p => Verdict S p.1)

lemma card_scoredPair (grp : α → γ) :
    Fintype.card (ScoredPair grp) = (scoredPairs grp).card :=
  Fintype.card_coe (scoredPairs grp)

/-- The noise of the pairwise answer key is exactly the pair discordance. -/
lemma noise_pairLabel (T L : Finset α) (grp : α → γ) :
    noise (pairLabel T grp) (pairLabel L grp) = pairDiscord T L grp := by
  classical
  have hset : symmDiff (pairLabel T grp) (pairLabel L grp)
      = Finset.univ.filter (fun p : ScoredPair grp => ¬ (Verdict T p.1 ↔ Verdict L p.1)) := by
    ext p
    simp only [Finset.mem_symmDiff, pairLabel, Finset.mem_filter, Finset.mem_univ, true_and]
    tauto
  rw [noise, hset, pairDiscord]
  refine Finset.card_bij (fun p _ => (p : α × α)) ?_ ?_ ?_
  · intro p hp
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
    exact mem_discordPairs.mpr ⟨p.2, hp⟩
  · intro p _ q _ h
    exact Subtype.ext h
  · intro p hp
    rw [mem_discordPairs] at hp
    exact ⟨⟨p, hp.1⟩, by simp [hp.2], rfl⟩

/-- **The capacity theorem for the pairwise protocol.**  A family of methods whose true
pair-scores are pairwise separated by more than twice the pair discordance -- the certification
condition of the protocol -- has at most `⌈1 / (2 ν_pair)⌉` members, where
`ν_pair = pairDiscord / #scoredPairs` is the pair-level error rate of the annotation.  However
many methods are entered, the leaderboard orders at most that many. -/
theorem pairwise_capacity {iota : Type*} {T L : Finset α} {grp : α → γ}
    {pred : iota → Finset (ScoredPair grp)} {M : Finset iota}
    (hP : 0 < (scoredPairs grp).card) (hnu : 0 < pairDiscord T L grp)
    (hcert : Certified (pairLabel T grp) (pairDiscord T L grp) pred M) :
    M.card ≤ max 1 ⌈1 / (2 * ((pairDiscord T L grp : ℝ) / (scoredPairs grp).card))⌉₊ := by
  classical
  set P : ℕ := (scoredPairs grp).card with hPdef
  set d : ℕ := pairDiscord T L grp with hddef
  have hcard : Fintype.card (ScoredPair grp) = P := card_scoredPair grp
  have hPR : (0:ℝ) < (P : ℝ) := by exact_mod_cast hP
  have heps : (0:ℝ) < (d : ℝ) / P := by
    have : (0:ℝ) < (d : ℝ) := by exact_mod_cast hnu
    positivity
  have hbudget : ((d : ℝ) / P) * (Fintype.card (ScoredPair grp) : ℝ) ≤ (d : ℕ) :=
    le_of_eq (by rw [hcard, div_mul_cancel₀ _ (ne_of_gt hPR)])
  have h1 : M.card ≤ benchCapacity (Fintype.card (ScoredPair grp)) ((d : ℝ) / P) 0 :=
    card_le_benchCapacity_of_rate (by rw [hcard]; exact hP) heps.le hbudget hcert
  have h2 : benchCapacity (Fintype.card (ScoredPair grp)) ((d : ℝ) / P) 0
      ≤ max 1 ⌈1 / (2 * ((d : ℝ) / P))⌉₊ :=
    benchCapacity_noise_only _ (by rw [hcard]; exact hP) heps
  exact h1.trans h2

/-! ## 3.  Block-correlated noise -/

/-- The residues of one block. -/
def blockFiber (blk : α → β) (K : β) : Finset α := Finset.univ.filter (fun x => blk x = K)

/-- The blocks that actually occur. -/
def blockSet (blk : α → β) : Finset β := Finset.univ.image blk

/-- A set of residues is **block-constant** when membership only depends on the block. -/
def BlockConstant (blk : α → β) (S : Finset α) : Prop :=
  ∀ x y : α, blk x = blk y → (x ∈ S ↔ y ∈ S)

/-- The scored pairs that straddle a block boundary. -/
def boundaryPairs (grp : α → γ) (blk : α → β) : Finset (α × α) :=
  (scoredPairs grp).filter (fun p => blk p.1 ≠ blk p.2)

/-- The **boundary mass carried by a block**: the number of boundary-crossing scored pairs with
an endpoint in it. -/
def boundaryDeg (grp : α → γ) (blk : α → β) (K : β) : ℕ :=
  ((boundaryPairs grp blk).filter (fun p => blk p.1 = K ∨ blk p.2 = K)).card

omit [Fintype α] [DecidableEq β] in
/-- If the truth and the error set are both block-constant, so is the annotation. -/
lemma blockConstant_annotation {blk : α → β} {T L : Finset α}
    (hT : BlockConstant blk T) (hE : BlockConstant blk (symmDiff T L)) :
    BlockConstant blk L := by
  intro x y hxy
  have h1 := hT x y hxy
  have h2 := hE x y hxy
  simp only [Finset.mem_symmDiff] at h2
  by_cases hxL : x ∈ L <;> by_cases hyL : y ∈ L <;> by_cases hxT : x ∈ T <;> by_cases hyT : y ∈ T
    <;> tauto

/-- **Only the boundary is exposed.**  Under block-correlated noise every corrupted scored pair
straddles a block boundary and has an endpoint in a flipped block. -/
lemma discordPairs_subset {T L : Finset α} {grp : α → γ} {blk : α → β}
    (hT : BlockConstant blk T) (hE : BlockConstant blk (symmDiff T L)) :
    discordPairs T L grp ⊆
      (boundaryPairs grp blk).filter (fun p => p.1 ∈ symmDiff T L ∨ p.2 ∈ symmDiff T L) := by
  classical
  have hL : BlockConstant blk L := blockConstant_annotation hT hE
  intro p hp
  rw [mem_discordPairs] at hp
  obtain ⟨hps, hpd⟩ := hp
  have hbnd : blk p.1 ≠ blk p.2 := by
    intro hblk
    have hTv : ¬ Verdict T p := fun h => h.2 ((hT p.1 p.2 hblk).mp h.1)
    have hLv : ¬ Verdict L p := fun h => h.2 ((hL p.1 p.2 hblk).mp h.1)
    exact hpd ⟨fun h => absurd h hTv, fun h => absurd h hLv⟩
  have hnoise : p.1 ∈ symmDiff T L ∨ p.2 ∈ symmDiff T L := by
    by_contra hcon
    push_neg at hcon
    obtain ⟨h1, h2⟩ := hcon
    simp only [Finset.mem_symmDiff, not_or, not_and, not_not] at h1 h2
    refine hpd ⟨fun h => ⟨?_, ?_⟩, fun h => ⟨?_, ?_⟩⟩
    · exact h1.1 h.1
    · intro hc; exact h.2 (h2.2 hc)
    · exact h1.2 h.1
    · intro hc; exact h.2 (h2.1 hc)
  simp only [Finset.mem_filter, boundaryPairs]
  exact ⟨⟨hps, hbnd⟩, hnoise⟩

omit [DecidableEq α] in
/-- The residues split into their blocks. -/
lemma card_eq_mul_of_blocks {blk : α → β} {b : ℕ}
    (hsize : ∀ K ∈ blockSet blk, (blockFiber blk K).card = b) :
    Fintype.card α = (blockSet blk).card * b := by
  classical
  have hfib : (Finset.univ : Finset α).card
      = ∑ K ∈ blockSet blk, ((Finset.univ : Finset α).filter (fun x => blk x = K)).card :=
    Finset.card_eq_sum_card_fiberwise (fun x _ => Finset.mem_image_of_mem blk (Finset.mem_univ x))
  have hsum : ∑ K ∈ blockSet blk, ((Finset.univ : Finset α).filter (fun x => blk x = K)).card
      = ∑ _K ∈ blockSet blk, b :=
    Finset.sum_congr rfl (fun K hK => hsize K hK)
  rw [Fintype.card, hfib, hsum, Finset.sum_const, smul_eq_mul]

omit [DecidableEq α] in
/-- A block-constant set is a union of blocks, so its size is a multiple of the block size. -/
lemma card_blockConstant {blk : α → β} {b : ℕ} {S : Finset α}
    (hsize : ∀ K ∈ blockSet blk, (blockFiber blk K).card = b) (hS : BlockConstant blk S) :
    S.card = (S.image blk).card * b := by
  classical
  have hfib : S.card = ∑ K ∈ S.image blk, (S.filter (fun x => blk x = K)).card :=
    Finset.card_eq_sum_card_fiberwise (fun x hx => Finset.mem_image_of_mem blk hx)
  have hsum : ∀ K ∈ S.image blk, (S.filter (fun x => blk x = K)).card = b := by
    intro K hK
    obtain ⟨y, hyS, hyK⟩ := Finset.mem_image.mp hK
    have hEq : S.filter (fun x => blk x = K) = blockFiber blk K := by
      ext x
      simp only [Finset.mem_filter, blockFiber, Finset.mem_univ, true_and]
      constructor
      · exact fun h => h.2
      · intro hx
        exact ⟨(hS y x (by rw [hyK, hx])).mp hyS, hx⟩
    rw [hEq]
    exact hsize K (Finset.mem_image.mpr ⟨y, Finset.mem_univ y, hyK⟩)
  rw [hfib, Finset.sum_congr rfl hsum, Finset.sum_const, smul_eq_mul]

/-- The corrupted pairs are at most the boundary mass of the flipped blocks. -/
lemma pairDiscord_le_boundary {T L : Finset α} {grp : α → γ} {blk : α → β} {D : ℕ}
    (hT : BlockConstant blk T) (hE : BlockConstant blk (symmDiff T L))
    (hD : ∀ K ∈ blockSet blk, boundaryDeg grp blk K ≤ D) :
    pairDiscord T L grp ≤ D * ((symmDiff T L).image blk).card := by
  classical
  set F : Finset β := (symmDiff T L).image blk with hF
  have hsub : discordPairs T L grp ⊆
      F.biUnion (fun K =>
        (boundaryPairs grp blk).filter (fun p => blk p.1 = K ∨ blk p.2 = K)) := by
    intro p hp
    have hp' := discordPairs_subset hT hE hp
    simp only [Finset.mem_filter] at hp'
    obtain ⟨hbp, hnoise⟩ := hp'
    rcases hnoise with h | h
    · exact Finset.mem_biUnion.mpr ⟨blk p.1, Finset.mem_image_of_mem blk h,
        Finset.mem_filter.mpr ⟨hbp, Or.inl rfl⟩⟩
    · exact Finset.mem_biUnion.mpr ⟨blk p.2, Finset.mem_image_of_mem blk h,
        Finset.mem_filter.mpr ⟨hbp, Or.inr rfl⟩⟩
  have hFsub : F ⊆ blockSet blk := by
    intro K hK
    obtain ⟨x, _, hx⟩ := Finset.mem_image.mp hK
    exact Finset.mem_image.mpr ⟨x, Finset.mem_univ x, hx⟩
  calc pairDiscord T L grp ≤ (F.biUnion (fun K =>
        (boundaryPairs grp blk).filter (fun p => blk p.1 = K ∨ blk p.2 = K))).card :=
        Finset.card_le_card hsub
    _ ≤ ∑ K ∈ F, boundaryDeg grp blk K := Finset.card_biUnion_le
    _ ≤ ∑ _K ∈ F, D := Finset.sum_le_sum (fun K hK => hD K (hFsub hK))
    _ = D * F.card := by rw [Finset.sum_const, smul_eq_mul, Nat.mul_comm]

/-- **Block-correlated noise, integer form.**  With blocks of size `b`, a block-constant truth
and a block-correlated annotation, and every block carrying at most `D` boundary-crossing
scored pairs:

    #residues · (corrupted pairs) ≤ D · #blocks · (mislabelled residues).

Dividing by `#residues · #scoredPairs` this is `ν_pair ≤ (boundary mass / pairs) · ν_label`. -/
theorem card_mul_pairDiscord_le {T L : Finset α} {grp : α → γ} {blk : α → β} {b D : ℕ}
    (hsize : ∀ K ∈ blockSet blk, (blockFiber blk K).card = b)
    (hT : BlockConstant blk T) (hE : BlockConstant blk (symmDiff T L))
    (hD : ∀ K ∈ blockSet blk, boundaryDeg grp blk K ≤ D) :
    Fintype.card α * pairDiscord T L grp
      ≤ D * (blockSet blk).card * (symmDiff T L).card := by
  classical
  set F : Finset β := (symmDiff T L).image blk with hF
  have hn : Fintype.card α = (blockSet blk).card * b := card_eq_mul_of_blocks hsize
  have hEcard : (symmDiff T L).card = F.card * b := card_blockConstant hsize hE
  have hdisc : pairDiscord T L grp ≤ D * F.card := pairDiscord_le_boundary hT hE hD
  calc Fintype.card α * pairDiscord T L grp
      = ((blockSet blk).card * b) * pairDiscord T L grp := by rw [hn]
    _ ≤ ((blockSet blk).card * b) * (D * F.card) := Nat.mul_le_mul_left _ hdisc
    _ = D * (blockSet blk).card * (F.card * b) := by ring
    _ = D * (blockSet blk).card * (symmDiff T L).card := by rw [hEcard]

/-- **Correlated noise reduces pair discordance.**  If the annotation errs by flipping whole
blocks of a partition into blocks of `b` residues, and the truth is constant on those blocks,
then the pair-level error rate of the pairwise protocol is at most the boundary fraction times
the residue-level error rate:

    ν_pair ≤ (boundary mass / total scored pairs) · ν_label,

with boundary mass `D · #blocks`, `D` a bound on the boundary-crossing scored pairs carried by
one block.  The gain is the boundary fraction, and it is a property of the block structure
alone -- no assumption on which blocks are flipped, and no measurement. -/
theorem correlated_noise_reduces_pair_discordance
    {T L : Finset α} {grp : α → γ} {blk : α → β} {b D : ℕ}
    (hP : 0 < (scoredPairs grp).card) (hn : 0 < Fintype.card α)
    (hsize : ∀ K ∈ blockSet blk, (blockFiber blk K).card = b)
    (hT : BlockConstant blk T) (hE : BlockConstant blk (symmDiff T L))
    (hD : ∀ K ∈ blockSet blk, boundaryDeg grp blk K ≤ D) :
    (pairDiscord T L grp : ℝ) / (scoredPairs grp).card
      ≤ ((D * (blockSet blk).card : ℝ) / (scoredPairs grp).card)
          * ((symmDiff T L).card / (Fintype.card α : ℝ)) := by
  have hint := card_mul_pairDiscord_le hsize hT hE hD
  have hPR : (0:ℝ) < ((scoredPairs grp).card : ℝ) := by exact_mod_cast hP
  have hnR : (0:ℝ) < (Fintype.card α : ℝ) := by exact_mod_cast hn
  have hintR : (Fintype.card α : ℝ) * (pairDiscord T L grp : ℝ)
      ≤ (D : ℝ) * ((blockSet blk).card : ℝ) * ((symmDiff T L).card : ℝ) := by
    exact_mod_cast hint
  rw [div_mul_div_comm, div_le_div_iff₀ hPR (by positivity)]
  nlinarith [mul_le_mul_of_nonneg_right hintR hPR.le]

end Pairwise
end IDR
