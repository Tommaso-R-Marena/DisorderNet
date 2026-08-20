/-
# The per-target mean AUC, and its invariance under per-target recalibration

`RequestProject.AUCInvariance` proves that a per-protein bias — or, more generally, any per-protein
strictly monotone recalibration — leaves the *pair-weighted* within-protein AUC exactly unchanged
(`auc_within_strictMono_invariant`).  A benchmark protocol that scores each target separately and
then averages the per-target numbers is using a different statistic: the **unweighted mean over
targets**, in which a target with three scorable pairs counts as much as one with three hundred.
The two statistics genuinely differ (`weighted_ne_unweighted_instance`), so the published
invariance theorem does not by itself cover the reported number.

This file supplies the missing statement.

* `targetPairs`, `aucTarget`, `aucTargetMean` — the pairs inside one target, that target's AUC, and
  the unweighted mean of those AUCs over a finite set of targets.
* `auc_target_recal_invariant` — each individual target's AUC is invariant under a per-target
  strictly monotone recalibration.
* `auc_target_strictMono_invariant` — **the theorem**: the unweighted per-target mean is invariant
  too, hence so is the number the protocol reports.
* `auc_target_shift_invariant` — the additive-bias special case.
* `auc_within_eq_pair_weighted_mean` — how the two statistics relate: the published, pair-weighted
  within-protein AUC is the *pair-weighted* average of the per-target AUCs, the unweighted mean
  being what one gets by replacing those weights by `1/|K|`.
-/
import Mathlib
import RequestProject.AUCInvariance

set_option autoImplicit false

namespace IDR.GroupedAUC

open Finset

section TargetMean

variable {I G : Type*} [Fintype I] [DecidableEq I] [DecidableEq G]

/-- The (positive, negative) pairs lying inside the single target `k`. -/
def targetPairs (lab : I → Bool) (grp : I → G) (k : G) : Finset (I × I) :=
  (allPairs lab).filter (fun q => grp q.1 = k ∧ grp q.2 = k)

/-- The AUC of one target: the Mann–Whitney statistic over the pairs inside it, normalised. -/
noncomputable def aucTarget (lab : I → Bool) (grp : I → G) (s : I → ℝ) (k : G) : ℝ :=
  auc (targetPairs lab grp k) s

/-- The **unweighted mean per-target AUC** over a set `K` of targets: the number a protocol reports
when it scores each target separately and averages the results, every target counting once. -/
noncomputable def aucTargetMean (lab : I → Bool) (grp : I → G) (s : I → ℝ) (K : Finset G) : ℝ :=
  (∑ k ∈ K, aucTarget lab grp s k) / K.card

/-- The Mann–Whitney kernel is invariant under a strictly monotone recalibration. -/
lemma kern_strictMono (f : ℝ → ℝ) (hf : StrictMono f) (x y : ℝ) : kern (f x) (f y) = kern x y := by
  unfold kern
  have h1 : f y < f x ↔ y < x := hf.lt_iff_lt
  have h2 : f x = f y ↔ x = y := hf.injective.eq_iff
  simp [h1, h2]

omit [DecidableEq I] in
/-- **One target's AUC is invariant** under a per-target strictly monotone recalibration: inside a
target the recalibration is one increasing map applied to every score. -/
theorem auc_target_recal_invariant (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (f : G → ℝ → ℝ) (hf : ∀ k, StrictMono (f k)) (k : G) :
    aucTarget lab grp (recal grp f s) k = aucTarget lab grp s k := by
  unfold aucTarget auc U
  congr 1
  refine Finset.sum_congr rfl (fun q hq => ?_)
  obtain ⟨h1, h2⟩ := (Finset.mem_filter.mp hq).2
  simp only [recal, h1, h2]
  exact kern_strictMono (f k) (hf k) _ _

omit [DecidableEq I] in
/-- **The theorem the protocol needs.**  The *unweighted* mean of the per-target AUCs — the number
reported by a protocol that scores each target separately and averages — is exactly invariant under
any per-target strictly monotone recalibration of the scores. -/
theorem auc_target_strictMono_invariant (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    (f : G → ℝ → ℝ) (hf : ∀ k, StrictMono (f k)) (K : Finset G) :
    aucTargetMean lab grp (recal grp f s) K = aucTargetMean lab grp s K := by
  unfold aucTargetMean
  congr 1
  exact Finset.sum_congr rfl (fun k _ => auc_target_recal_invariant lab grp s f hf k)

omit [DecidableEq I] in
/-- The additive special case: a per-target bias leaves the unweighted mean per-target AUC
unchanged. -/
theorem auc_target_shift_invariant (lab : I → Bool) (grp : I → G) (s : I → ℝ) (b : G → ℝ)
    (K : Finset G) :
    aucTargetMean lab grp (shift grp b s) K = aucTargetMean lab grp s K := by
  have hrec : shift grp b s = recal grp (fun k x => x + b k) s := rfl
  rw [hrec]
  exact auc_target_strictMono_invariant lab grp s _ (fun k => strictMono_id.add_const (b k)) K

/-! ## How the unweighted mean relates to the published, pair-weighted statistic -/

/-- The within-protein pairs are the disjoint union of the per-target pairs, provided every target
that occurs is listed in `K`. -/
lemma withinPairs_eq_biUnion (lab : I → Bool) (grp : I → G) {K : Finset G}
    (hK : ∀ i, grp i ∈ K) :
    withinPairs lab grp = K.biUnion (fun k => targetPairs lab grp k) := by
  ext q
  simp only [withinPairs, targetPairs, Finset.mem_filter, Finset.mem_biUnion]
  constructor
  · rintro ⟨hq, hgrp⟩
    exact ⟨grp q.1, hK q.1, hq, rfl, hgrp.symm⟩
  · rintro ⟨k, _, hq, h1, h2⟩
    exact ⟨hq, by rw [h1, h2]⟩

omit [DecidableEq I] in
lemma targetPairs_disjoint (lab : I → Bool) (grp : I → G) {k l : G} (hkl : k ≠ l) :
    Disjoint (targetPairs lab grp k) (targetPairs lab grp l) := by
  rw [Finset.disjoint_left]
  intro q hq hq'
  have h1 := (Finset.mem_filter.mp hq).2.1
  have h2 := (Finset.mem_filter.mp hq').2.1
  exact hkl (h1 ▸ h2 ▸ rfl)

omit [DecidableEq I] in
/-- The Mann–Whitney statistic of a set of pairs is the per-target AUC times the number of pairs,
including when the target has no pairs. -/
lemma U_targetPairs_eq (lab : I → Bool) (grp : I → G) (s : I → ℝ) (k : G) :
    U (targetPairs lab grp k) s = aucTarget lab grp s k * (targetPairs lab grp k).card := by
  unfold aucTarget auc
  by_cases h : (targetPairs lab grp k).card = 0
  · rw [Finset.card_eq_zero.mp h] at *
    simp [U, h]
  · field_simp

/-- **The published statistic is the pair-weighted mean.**  The within-protein AUC of
`RequestProject.AUCInvariance` is the average of the per-target AUCs weighted by the number of
scorable pairs in each target; the protocol's reported number replaces those weights by `1/|K|`. -/
theorem auc_within_eq_pair_weighted_mean (lab : I → Bool) (grp : I → G) (s : I → ℝ)
    {K : Finset G} (hK : ∀ i, grp i ∈ K) :
    auc (withinPairs lab grp) s
      = (∑ k ∈ K, aucTarget lab grp s k * (targetPairs lab grp k).card)
          / (withinPairs lab grp).card := by
  unfold auc
  congr 1
  rw [withinPairs_eq_biUnion lab grp hK, U,
    Finset.sum_biUnion (fun k _ l _ hkl => targetPairs_disjoint lab grp hkl)]
  exact Finset.sum_congr rfl (fun k _ => U_targetPairs_eq lab grp s k)

end TargetMean

/-! ## The two statistics are genuinely different -/

section Instance

/-- Five residues on two targets: target `0` has one scorable pair, scored correctly; target `1`
has two, both scored backwards. -/
def exGrp : Fin 5 → Fin 2 := ![0, 0, 1, 1, 1]

/-- The labels: one positive and one negative on target `0`, one positive and two negatives on
target `1`. -/
def exLab : Fin 5 → Bool := ![true, false, true, false, false]

/-- The scores: target `0` ranks its positive above its negative, target `1` below both of its
negatives. -/
def exScore : Fin 5 → ℝ := ![1, 0, 0, 1, 1]

/-- **The unweighted mean is not the pair-weighted one.**  On this instance target `0` scores `1`
on its single pair and target `1` scores `0` on its two, so the unweighted mean over targets is
`1/2` while the pair-weighted within-protein AUC — the statistic the published invariance theorem
is about — is `1/3`.  Both invariance theorems are therefore needed: they are statements about
different numbers. -/
theorem weighted_ne_unweighted_instance :
    aucTargetMean exLab exGrp exScore Finset.univ = 1 / 2 ∧
      auc (withinPairs exLab exGrp) exScore = 1 / 3 := by
  have hpos : posSet exLab = ({0, 2} : Finset (Fin 5)) := by decide
  have hneg : negSet exLab = ({1, 3, 4} : Finset (Fin 5)) := by decide
  have ht0 : targetPairs exLab exGrp 0 = ({(0, 1)} : Finset (Fin 5 × Fin 5)) := by decide
  have ht1 : targetPairs exLab exGrp 1 = ({(2, 3), (2, 4)} : Finset (Fin 5 × Fin 5)) := by decide
  have hw : withinPairs exLab exGrp
      = ({(0, 1), (2, 3), (2, 4)} : Finset (Fin 5 × Fin 5)) := by decide
  constructor
  · rw [aucTargetMean, Fin.sum_univ_two, aucTarget, aucTarget, ht0, ht1, auc, auc, U, U,
      Finset.sum_singleton, Finset.sum_insert (by decide), Finset.sum_singleton,
      show (({(0, 1)} : Finset (Fin 5 × Fin 5)).card) = 1 from by decide,
      show (({(2, 3), (2, 4)} : Finset (Fin 5 × Fin 5)).card) = 2 from by decide,
      show ((Finset.univ : Finset (Fin 2)).card) = 2 from by decide]
    simp [exScore, kern]
    norm_num
  · rw [auc, hw, U, Finset.sum_insert (by decide), Finset.sum_insert (by decide),
      Finset.sum_singleton,
      show (({(0, 1), (2, 3), (2, 4)} : Finset (Fin 5 × Fin 5)).card) = 3 from by decide]
    simp [exScore, kern]
    norm_num

end Instance

end IDR.GroupedAUC
