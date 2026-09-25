/-
# Part LXXVIII  Phosphorylation as an experiment on the model, not just on the protein

Part LXXIII showed that a pairwise, separation-dependent charge model of a disordered region sees
the sequence only through its charge autocorrelation, and exhibited two sequences the whole model
class cannot tell apart.  That is a statement about *sequence variation* as a probe.  This file
studies the other probe biology actually uses on disordered regions -- **post-translational
modification** -- and finds that it is strictly more informative.

Phosphorylation of site `k` subtracts a charge `z` (about two elementary charges at physiological
pH) from that one position: `phos z k q`.  Within the pairwise model of Part LXXIII this is an
exactly solvable perturbation.

* `pairEnergy_phos` -- **the single-site law.**  `E(phos z k q) = E(q) - z * h_k`, where `h_k` is
  the *local field* `locField` at the modified site: the kernel-weighted sum of all other charges.
  A phosphorylation shifts the patterning energy by exactly the local field it sees.
* `phos_comm` -- **order-independence.**  Phosphorylation of distinct sites commutes, so the
  pairwise model predicts that the outcome of multisite phosphorylation cannot depend on the order
  in which the kinase writes the marks.  Any measured order dependence falsifies the model class.
* `pairEnergy_phos_two` -- **the two-site law**, and with it
* `phospho_epistasis` -- **the central claim of this part.**  The non-additivity of a double
  phosphorylation, `Delta_kl - Delta_k - Delta_l`, is exactly `z^2 * w (dist k l)`.  It does not
  depend on the sequence, on the other charges, on the composition, or on where the pair sits in
  the chain -- only on the *spacing* of the two sites.  `phospho_epistasis_indep_of_sequence`
  states that invariance directly.
* `kernel_of_epistasis` and `pairEnergy_eq_of_epistasis_eq` -- **and therefore a way out of the
  blindness of Part LXXIII.**  Scanning the double-phosphorylation epistasis over all spacings
  determines the interaction kernel `w` on `1 <= d < N` outright, and hence determines the model's
  prediction for *every* sequence.  Sequence-variation experiments only ever see `N - 1`
  autocorrelation coordinates and are blind on the homometric pair; a phosphorylation scan
  identifies the kernel itself.  This is a concrete experimental design consequence of the theory.
* `phospho_no_three_body` -- **a falsifiable prediction.**  The third-order interaction among
  three phosphosites vanishes identically.  Pairwise electrostatics permits pairwise epistasis and
  nothing beyond it, so a measured three-site interaction is evidence of many-body physics rather
  than of a mis-fitted kernel.
* `phospho_dissolves_condensate` -- **and the phenotype.**  A four-residue polycationic patch with
  a contact kernel sits above the demixing threshold of Part LXXV at chain length one; a single
  phosphorylation with the physiological charge `z = 2` takes it below, so the model predicts a
  condensate that one kinase event dissolves.

The relationship this part adds is between *identifiability* and *chemistry*: the coordinates a
pairwise charge model leaves undetermined by sequence comparison are exactly the ones a
modification scan measures, because modification acts on a single site while sequence variation
acts through the autocorrelation.
-/
import Mathlib
import RequestProject.ChargePatterning
import RequestProject.SequencePhase

set_option autoImplicit false

namespace IDR

namespace Phospho

open Finset IDR.Pattern IDR.SeqPhase

/-- Phosphorylation of site `k`: subtract the charge `z` at that position and nowhere else. -/
def phos (z : ℝ) (k : ℕ) (q : ℕ → ℝ) : ℕ → ℝ := fun i => if i = k then q i - z else q i

/-- The local field at site `k`: the kernel-weighted sum of all the other charges of the chain. -/
noncomputable def locField (N : ℕ) (w q : ℕ → ℝ) (k : ℕ) : ℝ :=
  (∑ i ∈ range k, w (k - i) * q i) + ∑ j ∈ Ico (k + 1) N, w (j - k) * q j

/-- **Phosphorylations of distinct sites commute.** -/
theorem phos_comm (y z : ℝ) (k l : ℕ) (hkl : k ≠ l) (q : ℕ → ℝ) :
    phos z k (phos y l q) = phos y l (phos z k q) := by
  funext i
  simp only [phos]
  split_ifs with h1 h2 h2 <;> first | rfl | (exfalso; exact hkl (h1 ▸ h2 ▸ rfl))

/-- The local field is unchanged by phosphorylating the same site (it excludes that site). -/
theorem locField_phos_self (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k : ℕ) :
    locField N w (phos z k q) k = locField N w q k := by
  unfold locField
  congr 1
  · refine Finset.sum_congr rfl fun i hi => ?_
    have : i ≠ k := by simp only [Finset.mem_range] at hi; omega
    simp [phos, this]
  · refine Finset.sum_congr rfl fun j hj => ?_
    have : j ≠ k := by simp only [Finset.mem_Ico] at hj; omega
    simp [phos, this]

/-- **The single-site law.**  Phosphorylating site `k` lowers the pairwise patterning energy by
exactly `z` times the local field at that site. -/
theorem pairEnergy_phos (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k : ℕ) (hk : k < N) :
    pairEnergy N w (phos z k q) = pairEnergy N w q - z * locField N w q k := by
  have hdiff : ∀ j : ℕ,
      (∑ i ∈ range j, w (j - i) * (phos z k q i * phos z k q j))
        = (∑ i ∈ range j, w (j - i) * (q i * q j))
          + (if j = k then -(z * ∑ i ∈ range k, w (k - i) * q i)
              else if k < j then -(z * (w (j - k) * q j)) else 0) := by
    intro j
    by_cases hjk : j = k
    · subst hjk
      have hneg : -(z * ∑ i ∈ range j, w (j - i) * q i)
          = ∑ i ∈ range j, -(z * (w (j - i) * q i)) := by
        rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
      rw [if_pos rfl, hneg, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl fun i hi => ?_
      have hik : i ≠ j := by simp only [Finset.mem_range] at hi; omega
      simp only [phos, if_neg hik, if_true]
      ring
    · rw [if_neg hjk]
      have hqj : phos z k q j = q j := by simp [phos, hjk]
      have hterm : ∀ i, w (j - i) * (phos z k q i * phos z k q j)
          = w (j - i) * (q i * q j) + (if i = k then -(z * (w (j - k) * q j)) else 0) := by
        intro i
        rw [hqj]
        by_cases h : i = k
        · subst h
          rw [if_pos rfl]
          simp only [phos, if_true]
          ring
        · rw [if_neg h]
          simp only [phos, if_neg h]
          ring
      rw [Finset.sum_congr rfl fun i _ => hterm i, Finset.sum_add_distrib,
        Finset.sum_ite_eq' (range j) k (fun _ => -(z * (w (j - k) * q j)))]
      by_cases hkj : k < j
      · rw [if_pos (Finset.mem_range.mpr hkj), if_pos hkj]
      · rw [if_neg (by simpa using hkj), if_neg hkj]
  rw [pairEnergy, pairEnergy, Finset.sum_congr rfl fun j _ => hdiff j, Finset.sum_add_distrib]
  have hsplit : ∑ j ∈ range N,
      (if j = k then -(z * ∑ i ∈ range k, w (k - i) * q i)
        else if k < j then -(z * (w (j - k) * q j)) else 0)
      = -(z * locField N w q k) := by
    rw [Finset.range_eq_Ico, ← Finset.sum_Ico_consecutive _ (Nat.zero_le (k + 1)) hk,
      ← Finset.range_eq_Ico, Finset.sum_range_succ]
    have h0 : ∑ j ∈ range k,
        (if j = k then -(z * ∑ i ∈ range k, w (k - i) * q i)
          else if k < j then -(z * (w (j - k) * q j)) else 0) = 0 := by
      refine Finset.sum_eq_zero fun j hj => ?_
      simp only [Finset.mem_range] at hj
      rw [if_neg (by omega), if_neg (by omega)]
    have h1 : ∑ j ∈ Ico (k + 1) N,
        (if j = k then -(z * ∑ i ∈ range k, w (k - i) * q i)
          else if k < j then -(z * (w (j - k) * q j)) else 0)
        = -(z * ∑ j ∈ Ico (k + 1) N, w (j - k) * q j) := by
      have hcongr : ∀ j ∈ Ico (k + 1) N,
          (if j = k then -(z * ∑ i ∈ range k, w (k - i) * q i)
            else if k < j then -(z * (w (j - k) * q j)) else 0)
          = -(z * (w (j - k) * q j)) := by
        intro j hj
        simp only [Finset.mem_Ico] at hj
        rw [if_neg (by omega), if_pos (by omega)]
      rw [Finset.sum_congr rfl hcongr, Finset.mul_sum, ← Finset.sum_neg_distrib]
    rw [h0, if_pos rfl, h1, locField]
    ring
  rw [hsplit]
  ring

/-- Phosphorylating a *different* site lowers the local field by exactly the kernel at the
spacing of the two sites. -/
theorem locField_phos (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l : ℕ) (hl : l < N) (hkl : k ≠ l) :
    locField N w (phos z l q) k = locField N w q k - z * w (Nat.dist k l) := by
  have hstep : ∀ (m : ℕ) (c : ℕ → ℝ), ∀ i : ℕ,
      c i * (phos z l q i) = c i * q i + (if i = l then -(z * c i) else 0) := by
    intro _ c i
    simp only [phos]
    split_ifs with h <;> ring
  unfold locField
  rw [Finset.sum_congr rfl fun i _ => hstep 0 (fun i => w (k - i)) i,
    Finset.sum_congr rfl fun j _ => hstep 0 (fun j => w (j - k)) j,
    Finset.sum_add_distrib, Finset.sum_add_distrib,
    Finset.sum_ite_eq' (range k) l (fun i => -(z * w (k - i))),
    Finset.sum_ite_eq' (Ico (k + 1) N) l (fun j => -(z * w (j - k)))]
  rcases Nat.lt_or_ge l k with h | h
  · rw [if_pos (Finset.mem_range.mpr h), if_neg (by simp only [Finset.mem_Ico]; omega),
      Nat.dist_comm, Nat.dist_eq_sub_of_le h.le]
    ring
  · have hlk : k < l := lt_of_le_of_ne h hkl
    rw [if_neg (by simp only [Finset.mem_range]; omega),
      if_pos (by simp only [Finset.mem_Ico]; omega), Nat.dist_eq_sub_of_le hlk.le]
    ring

/-- **The two-site law.**  Double phosphorylation is the sum of the two single-site shifts plus an
interaction term set by the spacing alone. -/
theorem pairEnergy_phos_two (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l : ℕ) (hk : k < N) (hl : l < N)
    (hkl : k ≠ l) :
    pairEnergy N w (phos z k (phos z l q))
      = pairEnergy N w q - z * locField N w q k - z * locField N w q l
        + z ^ 2 * w (Nat.dist k l) := by
  rw [pairEnergy_phos N w (phos z l q) z k hk, pairEnergy_phos N w q z l hl,
    locField_phos N w q z k l hl hkl]
  ring

/-- **The three-site law.** -/
theorem pairEnergy_phos_three (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l m : ℕ)
    (hk : k < N) (hl : l < N) (hm : m < N) (hkl : k ≠ l) (hkm : k ≠ m) (hlm : l ≠ m) :
    pairEnergy N w (phos z k (phos z l (phos z m q)))
      = pairEnergy N w q - z * locField N w q k - z * locField N w q l - z * locField N w q m
        + z ^ 2 * w (Nat.dist k l) + z ^ 2 * w (Nat.dist k m) + z ^ 2 * w (Nat.dist l m) := by
  rw [pairEnergy_phos N w (phos z l (phos z m q)) z k hk,
    pairEnergy_phos_two N w q z l m hl hm hlm,
    locField_phos N w (phos z m q) z k l hl hkl,
    locField_phos N w q z k m hm hkm]
  ring

/-- The shift in patterning energy produced by phosphorylating one site. -/
noncomputable def shift1 (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k : ℕ) : ℝ :=
  pairEnergy N w (phos z k q) - pairEnergy N w q

/-- The shift produced by phosphorylating two sites. -/
noncomputable def shift2 (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l : ℕ) : ℝ :=
  pairEnergy N w (phos z k (phos z l q)) - pairEnergy N w q

/-- The shift produced by phosphorylating three sites. -/
noncomputable def shift3 (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l m : ℕ) : ℝ :=
  pairEnergy N w (phos z k (phos z l (phos z m q))) - pairEnergy N w q

/-- **Phosphosite epistasis is the kernel.**  The non-additivity of a double phosphorylation is
exactly `z^2 w(d)` with `d` the spacing of the two sites: it is independent of the sequence, of
the composition, and of the position of the pair along the chain. -/
theorem phospho_epistasis (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l : ℕ) (hk : k < N) (hl : l < N)
    (hkl : k ≠ l) :
    shift2 N w q z k l - shift1 N w q z k - shift1 N w q z l = z ^ 2 * w (Nat.dist k l) := by
  rw [shift2, shift1, shift1, pairEnergy_phos_two N w q z k l hk hl hkl,
    pairEnergy_phos N w q z k hk, pairEnergy_phos N w q z l hl]
  ring

/-- The epistasis is the same for every sequence: it is a property of the model, not the protein. -/
theorem phospho_epistasis_indep_of_sequence (N : ℕ) (w q q' : ℕ → ℝ) (z : ℝ) (k l : ℕ)
    (hk : k < N) (hl : l < N) (hkl : k ≠ l) :
    shift2 N w q z k l - shift1 N w q z k - shift1 N w q z l
      = shift2 N w q' z k l - shift1 N w q' z k - shift1 N w q' z l := by
  rw [phospho_epistasis N w q z k l hk hl hkl, phospho_epistasis N w q' z k l hk hl hkl]

/-- **A phosphorylation scan measures the kernel.**  Taking the two sites at positions `0` and `d`
recovers `w d` from the epistasis, for every spacing `1 <= d < N`. -/
theorem kernel_of_epistasis (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (hz : z ≠ 0) (d : ℕ) (hd1 : 1 ≤ d)
    (hdN : d < N) :
    w d = (shift2 N w q z 0 d - shift1 N w q z 0 - shift1 N w q z d) / z ^ 2 := by
  rw [phospho_epistasis N w q z 0 d (by omega) hdN (by omega)]
  have : Nat.dist 0 d = d := by simp [Nat.dist]
  rw [this]
  field_simp

/-- **Hence the scan pins down every prediction.**  Two kernels with the same double
phosphorylation epistasis at every spacing predict the same patterning energy for every sequence
-- in sharp contrast with sequence-comparison experiments, which by Part LXXIII cannot separate
the homometric pair at all. -/
theorem pairEnergy_eq_of_epistasis_eq (N : ℕ) (w w' q0 : ℕ → ℝ) (z : ℝ) (hz : z ≠ 0)
    (h : ∀ d, 1 ≤ d → d < N →
      shift2 N w q0 z 0 d - shift1 N w q0 z 0 - shift1 N w q0 z d
        = shift2 N w' q0 z 0 d - shift1 N w' q0 z 0 - shift1 N w' q0 z d) :
    ∀ q : ℕ → ℝ, pairEnergy N w q = pairEnergy N w' q := by
  have hker : ∀ d, 1 ≤ d → d < N → w d = w' d := by
    intro d hd1 hdN
    rw [kernel_of_epistasis N w q0 z hz d hd1 hdN, kernel_of_epistasis N w' q0 z hz d hd1 hdN,
      h d hd1 hdN]
  intro q
  rw [pairEnergy_eq_sum_autocorr, pairEnergy_eq_sum_autocorr]
  refine Finset.sum_congr rfl fun d hd => ?_
  simp only [Finset.mem_Ico] at hd
  rw [hker d hd.1 hd.2]

/-- **No three-body phosphorylation interaction.**  The third-order term of the inclusion-exclusion
expansion vanishes identically: a pairwise charge model allows pairwise epistasis between
phosphosites and nothing more.  A measured three-site interaction therefore falsifies the model
class rather than its fitted kernel. -/
theorem phospho_no_three_body (N : ℕ) (w q : ℕ → ℝ) (z : ℝ) (k l m : ℕ)
    (hk : k < N) (hl : l < N) (hm : m < N) (hkl : k ≠ l) (hkm : k ≠ m) (hlm : l ≠ m) :
    shift3 N w q z k l m
        - (shift2 N w q z k l + shift2 N w q z k m + shift2 N w q z l m)
        + (shift1 N w q z k + shift1 N w q z l + shift1 N w q z m) = 0 := by
  rw [shift3, shift2, shift2, shift2, shift1, shift1, shift1,
    pairEnergy_phos_three N w q z k l m hk hl hm hkl hkm hlm,
    pairEnergy_phos_two N w q z k l hk hl hkl,
    pairEnergy_phos_two N w q z k m hk hm hkm,
    pairEnergy_phos_two N w q z l m hl hm hlm,
    pairEnergy_phos N w q z k hk, pairEnergy_phos N w q z l hl, pairEnergy_phos N w q z m hm]
  ring

/-! ## One kinase event dissolves the condensate -/

/-- A four-residue polycationic patch. -/
def patch4 : ℕ → ℝ
  | 0 => 1 | 1 => 1 | 2 => 1 | 3 => 1 | _ => 0

/-- The contact kernel: every pair of residues in the region interacts equally. -/
def contact : ℕ → ℝ := fun _ => 1

theorem pairEnergy_patch4 : pairEnergy 4 contact patch4 = 6 := by
  norm_num [pairEnergy, contact, patch4, Finset.sum_range_succ]

theorem pairEnergy_patch4_phos : pairEnergy 4 contact (phos 2 0 patch4) = 0 := by
  norm_num [pairEnergy, contact, patch4, phos, Finset.sum_range_succ]

/-- **A single phosphorylation dissolves the predicted condensate.**  With the contact kernel and
the physiological phosphate charge `z = 2`, the unmodified polycationic patch sits above the
demixing threshold of Part LXXV at chain length one, and the singly phosphorylated sequence sits
below it: the model predicts a condensate that one kinase event removes. -/
theorem phospho_dissolves_condensate :
    Demixes 1 (chiEff 0 1 (pairEnergy 4 contact patch4)) ∧
      ∀ c, ¬ IDR.Phase.PhaseSeparates (Set.Icc (0 : ℝ) 1)
        (IDR.FH.fh 1 (chiEff 0 1 (pairEnergy 4 contact (phos 2 0 patch4)))) c := by
  have hthr : threshold 1 0 1 = 2 := by
    rw [threshold, IDR.FH.chiC_one]
    norm_num
  have hmain := demixes_iff_gt_threshold (N := 1) (chi0 := 0) (lam := 1) one_pos one_pos
  refine ⟨(hmain (pairEnergy 4 contact patch4)).1 ?_, (hmain _).2 ?_⟩
  · rw [hthr, pairEnergy_patch4]; norm_num
  · rw [hthr, pairEnergy_patch4_phos]; norm_num

end Phospho

end IDR
