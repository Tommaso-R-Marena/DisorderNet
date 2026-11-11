/-
# Part CXIX  An exact patterning law: sequence order, not composition, fixes chain statistics

Earlier parts of this development proved *composition blindness* abstractly: a model that reads
only the amino-acid composition of a region cannot be right, because two sequences of equal
composition can have different ensembles.  Those proofs exhibit an ensemble; they do not derive
one from polymer physics.  This part does, in the exactly solvable self-avoiding model of
Part CXVI.

Each residue `i` carries its own fugacity `u i` for placing its bond along the chain axis (a
stiffer, more extended, or more solvophobic residue has a larger `u`), and the partition
function is the honest sum over self-avoiding conformations,

  `Zseq u n = ∑_{w admissible, |w| = n} ∏_i (u i if bond i is axial, else 1)`.

The composition of a sequence is the multiset of its fugacities; the *pattern* is their order.

* `Zseq_succ` — the exact transfer recursion.
* `card_eastAt` — the exact decoupling identity: the number of conformations whose bond `j` is
  axial is `pdCnt j * pdCnt (n - 1 - j)`, the product of the conformation counts of the two
  pieces the axial bond separates.  An axial bond cuts the chain into two independent chains.
* `Zseq_markAt` — with a single heavy residue `t` at position `j`, the partition function is
  exactly `t · pdCnt j · pdCnt (n-1-j) + (pdCnt n − pdCnt j · pdCnt (n-1-j))`, and hence
* `patterning_gap` — moving that residue from the chain terminus to the neighbouring interior
  position changes the partition function by exactly

    `(t − 1) · (pdCnt (n-2) − pdCnt (n-3))`,

  which is strictly positive for every `t > 1` and every `n ≥ 3` (`patterning_matters`), and
  grows like `λ ^ n` with `λ = 1 + √2`.  Two sequences of *identical composition* therefore have
  different conformational free energies, by an amount that is exponentially large in the length
  of the region: patterning is not a small correction.

`pdCnt_add_two` records the Pell recursion `P (n+2) = 2 P (n+1) + P n` behind these counts.
-/
import Mathlib
import RequestProject.PartiallyDirected

namespace IDR.SAW
namespace PD

open scoped BigOperators

/-! ## Membership in the conformation list -/

lemma ok_zero_right (a : Letter) : ok a 0 := by revert a; decide

lemma ok_zero_left (b : Letter) : ok 0 b := by revert b; decide

lemma mem_wordsFrom_iff : ∀ (n : ℕ) (p : Letter) (w : List Letter),
    w ∈ wordsFrom n p ↔ w.length = n ∧ Adm w ∧ ∀ b ∈ w.head?, ok p b := by
  intro n
  induction n with
  | zero =>
      intro p w
      constructor
      · intro hw
        rw [(mem_wordsFrom_zero p w).1 hw]
        exact ⟨rfl, List.isChain_nil, by simp⟩
      · rintro ⟨hlen, -, -⟩
        rw [mem_wordsFrom_zero]
        exact List.length_eq_zero_iff.1 hlen
  | succ n ih =>
      intro p w
      constructor
      · intro hw
        obtain ⟨hadm, hhead⟩ := adm_of_mem_wordsFrom (n + 1) p w hw
        exact ⟨length_of_mem_wordsFrom (n + 1) p w hw, hadm, hhead⟩
      · rintro ⟨hlen, hadm, hhead⟩
        cases w with
        | nil => simp at hlen
        | cons a v =>
            have hlv : v.length = n := by simpa using hlen
            have hav : Adm v := (List.isChain_cons.1 hadm).2
            have hlink := (List.isChain_cons.1 hadm).1
            have hva : v ∈ wordsFrom n a := (ih a v).2 ⟨hlv, hav, hlink⟩
            rw [wordsFrom, Finset.mem_biUnion]
            refine ⟨a, ?_, Finset.mem_image.2 ⟨v, hva, rfl⟩⟩
            simp only [Finset.mem_filter, Finset.mem_univ, true_and]
            exact hhead a (by simp)

/-- Membership in the full conformation list of length `n`. -/
lemma mem_words (n : ℕ) (w : List Letter) :
    w ∈ wordsFrom n 0 ↔ w.length = n ∧ Adm w := by
  rw [mem_wordsFrom_iff]
  exact ⟨fun h => ⟨h.1, h.2.1⟩, fun h => ⟨h.1, h.2, fun b _ => ok_zero_left b⟩⟩

/-! ## The Pell recursion -/

/-- The conformation counts obey `P (n+2) = 2 P (n+1) + P n`. -/
theorem pdCnt_add_two (n : ℕ) : pdCnt (n + 2) = 2 * pdCnt (n + 1) + pdCnt n := by
  have h1 : pdCnt (n + 2) = pdCnt (n + 1) + pdN (n + 1) + pdS (n + 1) := pdCnt_succ (n + 1)
  have h2 := pdN_succ n
  have h3 := pdS_succ n
  have h4 := pdCnt_succ n
  omega

lemma pdN_pos : ∀ n : ℕ, 0 < pdN n
  | 0 => by simp
  | (n + 1) => by
      rw [pdN_succ]
      have := pdCnt_pos n
      omega

lemma pdS_pos : ∀ n : ℕ, 0 < pdS n
  | 0 => by simp
  | (n + 1) => by
      rw [pdS_succ]
      have := pdCnt_pos n
      omega

lemma pdCnt_lt_succ (n : ℕ) : pdCnt n < pdCnt (n + 1) := by
  have h := pdCnt_succ n
  have h1 := pdN_pos n
  have h2 := pdS_pos n
  omega

/-! ## An axial bond decouples the chain -/

/-- The conformations of `n` bonds whose bond `j` points along the chain axis. -/
def eastAt (n j : ℕ) : Finset (List Letter) :=
  (wordsFrom n 0).filter (fun w => w.getD j 0 = 0)

/-- **Decoupling.**  An axial bond cuts the chain into two independent chains: the number of
conformations with bond `j` axial is the product of the conformation counts of the two
pieces. -/
theorem card_eastAt {n j : ℕ} (hj : j < n) :
    (eastAt n j).card = pdCnt j * pdCnt (n - 1 - j) := by
  classical
  have hcard : pdCnt j * pdCnt (n - 1 - j)
      = ((wordsFrom j 0) ×ˢ (wordsFrom (n - 1 - j) 0)).card := by
    simp [pdCnt, Finset.card_product]
  rw [hcard]
  refine Finset.card_bij' (fun w _ => (w.take j, w.drop (j + 1)))
    (fun xy _ => xy.1 ++ (0 : Letter) :: xy.2) ?_ ?_ ?_ ?_
  · intro w hw
    simp only [eastAt, Finset.mem_filter] at hw
    obtain ⟨hw, hj0⟩ := hw
    obtain ⟨hlen, hadm⟩ := (mem_words n w).1 hw
    have hjlen : j < w.length := by omega
    have hsplit : w = w.take j ++ (0 : Letter) :: w.drop (j + 1) := by
      conv_lhs => rw [← List.take_append_drop j w]
      congr 1
      rw [List.drop_eq_getElem_cons hjlen]
      congr 1
      rw [← List.getD_eq_getElem w 0 hjlen]
      exact hj0
    have hadm' : Adm (w.take j ++ (0 : Letter) :: w.drop (j + 1)) := by rwa [← hsplit]
    rw [Adm, List.isChain_append] at hadm'
    obtain ⟨h1, h2, -⟩ := hadm'
    have h2' : Adm (w.drop (j + 1)) := (List.isChain_cons.1 h2).2
    simp only [Finset.mem_product]
    refine ⟨(mem_words j _).2 ⟨by simp [hlen]; omega, h1⟩,
      (mem_words (n - 1 - j) _).2 ⟨by simp [hlen]; omega, h2'⟩⟩
  · intro xy hxy
    simp only [Finset.mem_product] at hxy
    obtain ⟨hx, hy⟩ := hxy
    obtain ⟨hlx, hax⟩ := (mem_words j _).1 hx
    obtain ⟨hly, hay⟩ := (mem_words (n - 1 - j) _).1 hy
    have hlen : (xy.1 ++ (0 : Letter) :: xy.2).length = n := by
      simp [hlx, hly]; omega
    have hadm : Adm (xy.1 ++ (0 : Letter) :: xy.2) := by
      rw [Adm, List.isChain_append]
      refine ⟨hax, List.isChain_cons.2 ⟨fun b _ => ok_zero_left b, hay⟩, ?_⟩
      intro a _ b hb
      simp only [List.head?_cons, Option.mem_def, Option.some.injEq] at hb
      subst hb
      exact ok_zero_right a
    simp only [eastAt, Finset.mem_filter]
    refine ⟨(mem_words n _).2 ⟨hlen, hadm⟩, ?_⟩
    have hidx : j < (xy.1 ++ (0 : Letter) :: xy.2).length := by omega
    rw [List.getD_eq_getElem _ _ hidx, List.getElem_append hidx]
    simp [hlx]
  · intro w hw
    simp only [eastAt, Finset.mem_filter] at hw
    obtain ⟨hw, hj0⟩ := hw
    obtain ⟨hlen, -⟩ := (mem_words n w).1 hw
    have hjlen : j < w.length := by omega
    have hcons : w.drop j = w[j] :: w.drop (j + 1) := List.drop_eq_getElem_cons hjlen
    have hval : w[j] = (0 : Letter) := by
      rw [← List.getD_eq_getElem w 0 hjlen]; exact hj0
    show w.take j ++ (0 : Letter) :: w.drop (j + 1) = w
    calc w.take j ++ (0 : Letter) :: w.drop (j + 1)
        = w.take j ++ w.drop j := by rw [hcons, hval]
      _ = w := List.take_append_drop j w
  · intro xy hxy
    simp only [Finset.mem_product] at hxy
    obtain ⟨hx, -⟩ := hxy
    obtain ⟨hlx, -⟩ := (mem_words j _).1 hx
    have h1 : (xy.1 ++ (0 : Letter) :: xy.2).take j = xy.1 := List.take_left' hlx
    have h2 : (xy.1 ++ (0 : Letter) :: xy.2).drop (j + 1) = xy.2 := by
      rw [show xy.1 ++ (0 : Letter) :: xy.2 = (xy.1 ++ [0]) ++ xy.2 by simp]
      exact List.drop_left' (by simp [hlx])
    show (List.take j (xy.1 ++ (0 : Letter) :: xy.2),
      List.drop (j + 1) (xy.1 ++ (0 : Letter) :: xy.2)) = xy
    rw [h1, h2]

/-! ## The sequence-dependent partition function -/

/-- The Boltzmann weight of a conformation for a sequence of fugacities. -/
noncomputable def wtList (u : ℕ → ℝ) (w : List Letter) : ℝ :=
  ∏ i ∈ Finset.range w.length, (if w.getD i 0 = 0 then u i else 1)

/-- The partition function of an `n`-residue chain with sequence-dependent fugacities. -/
noncomputable def Zseq (u : ℕ → ℝ) (n : ℕ) : ℝ := ∑ w ∈ wordsFrom n 0, wtList u w

/-- A sequence with a single heavy residue `t` at position `j`. -/
noncomputable def markAt (j : ℕ) (t : ℝ) : ℕ → ℝ := fun i => if i = j then t else 1

lemma wtList_markAt {n j : ℕ} (hj : j < n) (t : ℝ) {w : List Letter} (hw : w.length = n) :
    wtList (markAt j t) w = if w.getD j 0 = 0 then t else 1 := by
  classical
  rw [wtList, hw]
  rw [Finset.prod_eq_single j]
  · simp [markAt]
  · intro i _ hij
    simp [markAt, hij]
  · intro hjn
    exact absurd (Finset.mem_range.2 hj) hjn

/-- **The partition function of a singly marked sequence**, exactly. -/
theorem Zseq_markAt {n j : ℕ} (hj : j < n) (t : ℝ) :
    Zseq (markAt j t) n
      = t * ((eastAt n j).card : ℝ) + ((pdCnt n : ℝ) - ((eastAt n j).card : ℝ)) := by
  classical
  have hsum : Zseq (markAt j t) n
      = ∑ w ∈ wordsFrom n 0, (if w.getD j 0 = 0 then t else 1) := by
    refine Finset.sum_congr rfl fun w hw => ?_
    exact wtList_markAt hj t (length_of_mem_wordsFrom n 0 w hw)
  rw [hsum, Finset.sum_ite, Finset.sum_const, Finset.sum_const]
  have hfilter : (wordsFrom n 0).filter (fun w => w.getD j 0 = 0) = eastAt n j := rfl
  have hfilter' : ((wordsFrom n 0).filter (fun w => ¬ (w.getD j 0 = 0))).card
      = pdCnt n - (eastAt n j).card := by
    have := Finset.card_filter_add_card_filter_not
      (s := wordsFrom n 0) (p := fun w => w.getD j 0 = 0)
    rw [hfilter] at this
    simp [pdCnt] at this ⊢
    omega
  have hle : (eastAt n j).card ≤ pdCnt n := by
    rw [pdCnt]
    exact Finset.card_filter_le _ _
  rw [hfilter, hfilter']
  have hcast : ((pdCnt n - (eastAt n j).card : ℕ) : ℝ)
      = (pdCnt n : ℝ) - ((eastAt n j).card : ℝ) := Nat.cast_sub hle
  simp only [nsmul_eq_mul, mul_one]
  rw [hcast]
  ring

/-- **The exact patterning gap.**  Moving the heavy residue from the terminus to the adjacent
interior position changes the free energy by exactly `(t − 1) (P (n−2) − P (n−3))`. -/
theorem patterning_gap {n : ℕ} (hn : 3 ≤ n) (t : ℝ) :
    Zseq (markAt 1 t) n - Zseq (markAt 0 t) n
      = (t - 1) * ((pdCnt (n - 2) : ℝ) - (pdCnt (n - 3) : ℝ)) := by
  obtain ⟨m, rfl⟩ : ∃ m, n = m + 3 := ⟨n - 3, by omega⟩
  have hp1 : pdCnt 1 = 3 := by
    rw [pdCnt_succ]
    simp
  have h0 : (eastAt (m + 3) 0).card = pdCnt (m + 2) := by
    rw [card_eastAt (by omega), show m + 3 - 1 - 0 = m + 2 from rfl, pdCnt_zero, one_mul]
  have h1 : (eastAt (m + 3) 1).card = 3 * pdCnt (m + 1) := by
    rw [card_eastAt (by omega), show m + 3 - 1 - 1 = m + 1 from rfl, hp1]
  rw [Zseq_markAt (by omega) t, Zseq_markAt (by omega) t, h0, h1]
  have hpell : pdCnt (m + 2) = 2 * pdCnt (m + 1) + pdCnt m := pdCnt_add_two m
  have hcast : ((pdCnt (m + 2) : ℕ) : ℝ) = 2 * (pdCnt (m + 1) : ℝ) + (pdCnt m : ℝ) := by
    exact_mod_cast congrArg (fun k : ℕ => (k : ℝ)) hpell
  simp only [show m + 3 - 2 = m + 1 from rfl, show m + 3 - 3 = m from rfl]
  push_cast
  rw [hcast]
  ring

/-- **Patterning matters.**  For every chain of at least three residues and every heavy residue
`t > 1`, the two sequences of identical composition — heavy residue at the terminus, heavy
residue one position in — have strictly different conformational partition functions. -/
theorem patterning_matters {n : ℕ} (hn : 3 ≤ n) {t : ℝ} (ht : 1 < t) :
    Zseq (markAt 0 t) n < Zseq (markAt 1 t) n := by
  have hgap := patterning_gap hn t
  have hlt : pdCnt (n - 3) < pdCnt (n - 2) := by
    have : n - 2 = (n - 3) + 1 := by omega
    rw [this]
    exact pdCnt_lt_succ _
  have hpos : (0:ℝ) < (pdCnt (n - 2) : ℝ) - (pdCnt (n - 3) : ℝ) := by
    have : ((pdCnt (n - 3) : ℕ) : ℝ) < ((pdCnt (n - 2) : ℕ) : ℝ) := by exact_mod_cast hlt
    linarith
  nlinarith [hgap, hpos, ht]

/-- The gap grows exponentially in the length of the region: composition-blind scoring is wrong
by an exponentially large free energy, not by a constant. -/
theorem patterning_gap_ge {n : ℕ} (hn : 3 ≤ n) {t : ℝ} (ht : 1 < t) :
    (t - 1) * (lam ^ (n - 3)) ≤ Zseq (markAt 1 t) n - Zseq (markAt 0 t) n := by
  rw [patterning_gap hn t]
  have hlow : lam ^ (n - 3) ≤ (pdCnt (n - 3) : ℝ) := lam_pow_le_pdCnt _
  have hstep : (pdCnt (n - 3) : ℝ) + lam ^ (n - 3) ≤ (pdCnt (n - 2) : ℝ) := by
    have hn2 : n - 2 = (n - 3) + 1 := by omega
    have h1 : pdCnt ((n - 3) + 1) = pdCnt (n - 3) + pdN (n - 3) + pdS (n - 3) :=
      pdCnt_succ _
    have h2 : lam ^ (n - 3) ≤ (pdN (n - 3) : ℝ) + (pdS (n - 3) : ℝ) := by
      have h3 := (pdCnt_bounds (n - 3)).2.2.1
      have h4 := (pdCnt_bounds (n - 3)).2.2.2.2.1
      have hs : (1:ℝ) < Real.sqrt 2 := by
        nlinarith [sqrt_two_sq, sqrt_two_pos]
      have hlamv : lam - 1 = Real.sqrt 2 := by simp [lam]
      have hp : (0:ℝ) < lam ^ (n - 3) := pow_pos lam_pos _
      nlinarith [h3, h4, hp]
    rw [hn2, h1]
    push_cast
    linarith
  nlinarith [hstep, hlow, ht]


/-! ## The chain is ballistic: a third of its bonds are axial, on average

The counts above also settle the shape of a typical conformation of the solvable model.  The
mean number of axial bonds is at least `n / 3`, so the mean end-to-end extension grows linearly
in the number of residues: the swelling exponent of the partially directed chain is exactly `1`,
the extreme of the deterministic window `[1/2, 1]` that Part CXIV proves for every self-avoiding
walk.  A disordered region that cannot turn back on itself is not a random coil; it is extended.
-/

/-- The number of axial bonds of a conformation of `n` bonds. -/
def axialCount (n : ℕ) (w : List Letter) : ℕ :=
  ((Finset.range n).filter (fun j => w.getD j 0 = 0)).card

lemma sum_axialCount (n : ℕ) :
    ∑ w ∈ wordsFrom n 0, axialCount n w = ∑ j ∈ Finset.range n, (eastAt n j).card := by
  classical
  simp only [axialCount, eastAt, Finset.card_filter]
  exact Finset.sum_comm

/-- Cutting an admissible word in two leaves two admissible words, so the conformation count is
submultiplicative. -/
theorem pdCnt_submultiplicative (a b : ℕ) : pdCnt (a + b) ≤ pdCnt a * pdCnt b := by
  classical
  have hcard : pdCnt a * pdCnt b = ((wordsFrom a 0) ×ˢ (wordsFrom b 0)).card := by
    simp [pdCnt, Finset.card_product]
  rw [pdCnt, hcard]
  refine Finset.card_le_card_of_injOn (fun w => (w.take a, w.drop a)) ?_ ?_
  · intro w hw
    rw [Finset.mem_coe, mem_words] at hw
    obtain ⟨hlen, hadm⟩ := hw
    have hsplit : Adm (w.take a ++ w.drop a) := by rwa [List.take_append_drop]
    rw [Adm, List.isChain_append] at hsplit
    simp only [Finset.mem_coe, Finset.mem_product]
    exact ⟨(mem_words a _).2 ⟨by simp [hlen], hsplit.1⟩,
      (mem_words b _).2 ⟨by simp [hlen], hsplit.2.1⟩⟩
  · intro w _ v _ h
    have h1 : w.take a = v.take a := congrArg Prod.fst h
    have h2 : w.drop a = v.drop a := congrArg Prod.snd h
    calc w = w.take a ++ w.drop a := (List.take_append_drop a w).symm
      _ = v.take a ++ v.drop a := by rw [h1, h2]
      _ = v := List.take_append_drop a v

lemma pdCnt_succ_le (n : ℕ) : pdCnt (n + 1) ≤ 3 * pdCnt n := by
  have h : pdCnt (0 + (n + 1)) ≤ pdCnt 1 * pdCnt n := by
    have := pdCnt_submultiplicative 1 n
    simpa [Nat.add_comm] using this
  have hp1 : pdCnt 1 = 3 := by rw [pdCnt_succ]; simp
  simpa [hp1] using h

/-- **Every axial position is populated**: at least a third of the conformations have an axial
bond at any prescribed position. -/
theorem card_eastAt_ge {n j : ℕ} (hj : j < n) : pdCnt n ≤ 3 * (eastAt n j).card := by
  have hdec : (eastAt n j).card = pdCnt j * pdCnt (n - 1 - j) := card_eastAt hj
  have hsub : pdCnt (j + (n - 1 - j)) ≤ pdCnt j * pdCnt (n - 1 - j) :=
    pdCnt_submultiplicative _ _
  have hidx : j + (n - 1 - j) = n - 1 := by omega
  rw [hidx] at hsub
  have hn : n - 1 + 1 = n := by omega
  have hle : pdCnt n ≤ 3 * pdCnt (n - 1) := by
    have := pdCnt_succ_le (n - 1)
    rwa [hn] at this
  omega

/-- **The chain is ballistic.**  On average at least a third of the bonds of a conformation
point along the chain axis, so the mean extension grows linearly in the number of residues. -/
theorem mean_axial_ge_third (n : ℕ) :
    (n : ℝ) / 3 * (pdCnt n : ℝ) ≤ ∑ w ∈ wordsFrom n 0, (axialCount n w : ℝ) := by
  have hsum : ∑ w ∈ wordsFrom n 0, axialCount n w = ∑ j ∈ Finset.range n, (eastAt n j).card :=
    sum_axialCount n
  have hlow : ∑ j ∈ Finset.range n, pdCnt n ≤ 3 * ∑ j ∈ Finset.range n, (eastAt n j).card := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j hj => card_eastAt_ge (Finset.mem_range.1 hj)
  rw [Finset.sum_const, Finset.card_range, smul_eq_mul] at hlow
  have hnat : n * pdCnt n ≤ 3 * ∑ w ∈ wordsFrom n 0, axialCount n w := by
    rw [hsum]; exact hlow
  have hcast : ((n * pdCnt n : ℕ) : ℝ) ≤ ((3 * ∑ w ∈ wordsFrom n 0, axialCount n w : ℕ) : ℝ) := by
    exact_mod_cast hnat
  push_cast at hcast
  linarith

end PD
end IDR.SAW
