/-
# Part XCIII  A screen whose candidates are *not* independent

Part XCI (`FalseDiscovery.lean`) proves false-discovery-rate control for the Benjamini–Hochberg
list in a **product** experiment: one independent coordinate per candidate region.  A real
proteome screen is not a product.  Candidates share reagents, share a calibration run, share a
plate and share a batch, and the resulting dependence is neither known nor sign-constrained.  The
assumptions list of `PAPER.md` recorded this as an open item: "under arbitrary dependence the
Benjamini–Hochberg procedure needs the harmonic correction, which is not proved here".

This file closes that item, by proving dependence-free error control for the two procedures that
have it.

*The joint law is arbitrary.*  `Law Ω` is any probability distribution on a finite outcome space
`Ω`; no product structure, no independence, no exchangeability, nothing.

* `IsEValue` — the evidence each candidate supplies is an **e-value**: a nonnegative statistic
  whose mean under the null is at most one.  `isEValue_likelihoodRatio` shows the canonical
  example — a likelihood ratio against any alternative law, which is exactly the wealth process of
  the sequential test of Part XCII.  `not_isEValue_of_mean_gt_one` records that the condition is a
  real restriction.
* `SelfConsistent` — the abstract shape of a step-up rule: every reported candidate `i` has
  `E i ≥ m / (q · |R|)`.
* `fdp_le_pointwise` — **the whole argument in one line of algebra**: on *every* outcome, the
  false discovery proportion of a self-consistent rule is at most `(q/m)·Σ_{i null} E i`.  This is
  a pointwise inequality, so no property of the joint law is used at any stage.
* `selfConsistent_fdr_le`, `ebh_fdr_le`, `ebh_fdr_le_level` — **the theorem**: for the e-BH list at
  level `q`, `E[FDP] ≤ q·|H₀|/m ≤ q`, under an arbitrary joint law.  Nothing is assumed about the
  non-null candidates, about the dependence, or about the number of candidates.
* `mem_ebh_of_large` — the list is not empty by construction: any candidate with `E i ≥ m/q` is
  always reported, which is the e-value form of "BH dominates Bonferroni".
* `bonferroni_fwer` — the complementary statement for the stronger criterion: the union bound needs
  no independence either, so family-wise control at level `q/m` per candidate survives arbitrary
  dependence.

What is *not* repaired by any of this — and the file states it rather than hiding it — is the
calibration floor of Part XC.  An e-value is a statement about the null *law* of the read-out; if
the instrument's baseline rate is wrong by `τ·J`, the "e-value" has null mean greater than one
(`not_isEValue_of_mean_gt_one`) and every guarantee above is void.  Multiplicity control is
orthogonal to calibration, and cannot substitute for it.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR.DepScreen

variable {Ω : Type*} [Fintype Ω]

/-- A probability distribution on a finite outcome space.  No product structure is assumed:
this is the *joint* law of the whole screen. -/
structure Law (Ω : Type*) [Fintype Ω] where
  prob : Ω → ℝ
  nonneg : ∀ ω, 0 ≤ prob ω
  total : ∑ ω, prob ω = 1

namespace Law

/-- Expectation under the joint law. -/
def mean (P : Law Ω) (X : Ω → ℝ) : ℝ := ∑ ω, P.prob ω * X ω

/-- Probability of an event. -/
def pr (P : Law Ω) (S : Finset Ω) : ℝ := ∑ ω ∈ S, P.prob ω

lemma mean_mono (P : Law Ω) {X Y : Ω → ℝ} (h : ∀ ω, X ω ≤ Y ω) : P.mean X ≤ P.mean Y := by
  apply Finset.sum_le_sum
  intro ω _
  exact mul_le_mul_of_nonneg_left (h ω) (P.nonneg ω)

lemma mean_nonneg (P : Law Ω) {X : Ω → ℝ} (h : ∀ ω, 0 ≤ X ω) : 0 ≤ P.mean X :=
  Finset.sum_nonneg fun ω _ => mul_nonneg (P.nonneg ω) (h ω)

lemma mean_sum (P : Law Ω) {ι : Type*} (s : Finset ι) (X : ι → Ω → ℝ) :
    P.mean (fun ω => ∑ i ∈ s, X i ω) = ∑ i ∈ s, P.mean (X i) := by
  simp [mean, Finset.mul_sum, Finset.sum_comm (s := (univ : Finset Ω)) (t := s)]

lemma mean_smul (P : Law Ω) (c : ℝ) (X : Ω → ℝ) : P.mean (fun ω => c * X ω) = c * P.mean X := by
  simp only [mean, Finset.mul_sum]
  exact Finset.sum_congr rfl (by intros; ring)

lemma pr_nonneg (P : Law Ω) (S : Finset Ω) : 0 ≤ P.pr S :=
  Finset.sum_nonneg fun ω _ => P.nonneg ω

lemma pr_mono (P : Law Ω) {S T : Finset Ω} (h : S ⊆ T) : P.pr S ≤ P.pr T :=
  Finset.sum_le_sum_of_subset_of_nonneg h fun ω _ _ => P.nonneg ω

/-- The union bound, valid for an arbitrary joint law. -/
lemma pr_biUnion_le (P : Law Ω) {ι : Type*} [DecidableEq ι] (s : Finset ι) (A : ι → Finset Ω) :
    P.pr (s.biUnion A) ≤ ∑ i ∈ s, P.pr (A i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [pr]
  | insert i s hi ih =>
      rw [Finset.biUnion_insert, Finset.sum_insert hi]
      have hsplit : ∑ ω ∈ A i ∪ s.biUnion A, P.prob ω + ∑ ω ∈ A i ∩ s.biUnion A, P.prob ω
          = ∑ ω ∈ A i, P.prob ω + ∑ ω ∈ s.biUnion A, P.prob ω := Finset.sum_union_inter
      have hinter : 0 ≤ ∑ ω ∈ A i ∩ s.biUnion A, P.prob ω :=
        Finset.sum_nonneg fun ω _ => P.nonneg ω
      unfold pr at *
      linarith

end Law

/-- An **e-value**: a nonnegative statistic whose mean under the null law is at most one. -/
def IsEValue (P : Law Ω) (E : Ω → ℝ) : Prop := (∀ ω, 0 ≤ E ω) ∧ P.mean E ≤ 1

/-- The canonical e-value: a likelihood ratio against any alternative law. -/
lemma isEValue_likelihoodRatio (P Q : Law Ω) :
    IsEValue P (fun ω => if P.prob ω = 0 then 0 else Q.prob ω / P.prob ω) := by
  constructor
  · intro ω
    by_cases h : P.prob ω = 0 <;> simp [h]
    exact div_nonneg (Q.nonneg ω) (P.nonneg ω)
  · have hrw : P.mean (fun ω => if P.prob ω = 0 then 0 else Q.prob ω / P.prob ω)
        = ∑ ω, (if P.prob ω = 0 then 0 else Q.prob ω) := by
      refine Finset.sum_congr rfl ?_
      intro ω _
      by_cases h : P.prob ω = 0 <;> simp [h]
      field_simp
    rw [hrw]
    calc ∑ ω, (if P.prob ω = 0 then 0 else Q.prob ω) ≤ ∑ ω, Q.prob ω := by
          apply Finset.sum_le_sum
          intro ω _
          by_cases h : P.prob ω = 0 <;> simp [h, Q.nonneg ω]
      _ = 1 := Q.total

/-- A nonnegative statistic that is *not* an e-value: the condition has content, and a
miscalibrated instrument violates it. -/
lemma not_isEValue_of_mean_gt_one (P : Law Ω) (E : Ω → ℝ) (h : 1 < P.mean E) :
    ¬ IsEValue P E := fun hE => absurd hE.2 (not_le.mpr h)

variable {m : ℕ}

/-- The false discovery proportion of a reported list `R ω` against the set `H0` of true nulls. -/
noncomputable def fdp (H0 : Finset (Fin m)) (R : Ω → Finset (Fin m)) (ω : Ω) : ℝ :=
  if (R ω).card = 0 then 0 else (((R ω) ∩ H0).card : ℝ) / ((R ω).card : ℝ)

omit [Fintype Ω] in
lemma fdp_nonneg (H0 : Finset (Fin m)) (R : Ω → Finset (Fin m)) (ω : Ω) : 0 ≤ fdp H0 R ω := by
  unfold fdp
  split_ifs
  · exact le_rfl
  · positivity

/-- A rejection rule is **self-consistent** at level `q` if every candidate it reports has an
e-value at least `m / (q · |R|)`. -/
def SelfConsistent (q : ℝ) (m : ℕ) (E : Fin m → Ω → ℝ) (R : Ω → Finset (Fin m)) : Prop :=
  ∀ ω, ∀ i ∈ R ω, (m : ℝ) / (q * ((R ω).card : ℝ)) ≤ E i ω

omit [Fintype Ω] in
/-- **The pointwise bound.**  On every single outcome, the false discovery proportion of a
self-consistent rule is dominated by `(q/m)` times the sum of the null e-values.  Because this
holds outcome by outcome, the dependence structure of the screen never enters. -/
lemma fdp_le_pointwise {q : ℝ} (hq : 0 < q) (hm : 0 < m) {E : Fin m → Ω → ℝ}
    (hE : ∀ i ω, 0 ≤ E i ω) {R : Ω → Finset (Fin m)} (hR : SelfConsistent q m E R)
    (H0 : Finset (Fin m)) (ω : Ω) :
    fdp H0 R ω ≤ (q / m) * ∑ i ∈ H0, E i ω := by
  have hsum : 0 ≤ ∑ i ∈ H0, E i ω := Finset.sum_nonneg (fun i _ => hE i ω)
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  unfold fdp
  split_ifs with h
  · positivity
  · have hc : (0:ℝ) < ((R ω).card : ℝ) := by
      have : 0 < (R ω).card := Nat.pos_of_ne_zero h
      exact_mod_cast this
    have key : ∀ i ∈ (R ω) ∩ H0, (1 : ℝ) / ((R ω).card : ℝ) ≤ (q / m) * E i ω := by
      intro i hi
      have hiR : i ∈ R ω := (Finset.mem_inter.mp hi).1
      have hEi := hR ω i hiR
      have h2 : (q / m) * ((m : ℝ) / (q * ((R ω).card:ℝ))) ≤ (q / m) * E i ω :=
        mul_le_mul_of_nonneg_left hEi (by positivity)
      calc (1:ℝ) / ((R ω).card : ℝ) = (q / m) * ((m : ℝ) / (q * ((R ω).card:ℝ))) := by
            field_simp
        _ ≤ _ := h2
    calc (((R ω) ∩ H0).card : ℝ) / ((R ω).card : ℝ)
        = ∑ _i ∈ (R ω) ∩ H0, (1:ℝ) / ((R ω).card : ℝ) := by
          rw [Finset.sum_const]; simp [div_eq_mul_inv]
      _ ≤ ∑ i ∈ (R ω) ∩ H0, (q / m) * E i ω := Finset.sum_le_sum key
      _ ≤ ∑ i ∈ H0, (q / m) * E i ω := by
          apply Finset.sum_le_sum_of_subset_of_nonneg Finset.inter_subset_right
          intro i _ _
          exact mul_nonneg (by positivity) (hE i ω)
      _ = (q / m) * ∑ i ∈ H0, E i ω := by rw [Finset.mul_sum]

/-- **False discovery rate control under arbitrary dependence.**  Any self-consistent rule, fed
e-values for the true nulls, has expected false discovery proportion at most `q·|H₀|/m`. -/
theorem selfConsistent_fdr_le (P : Law Ω) {q : ℝ} (hq : 0 < q) (hm : 0 < m)
    {E : Fin m → Ω → ℝ} (hE : ∀ i ω, 0 ≤ E i ω) {R : Ω → Finset (Fin m)}
    (hR : SelfConsistent q m E R) (H0 : Finset (Fin m))
    (hnull : ∀ i ∈ H0, IsEValue P (E i)) :
    P.mean (fdp H0 R) ≤ q * H0.card / m := by
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  have h1 : P.mean (fdp H0 R) ≤ P.mean (fun ω => (q / m) * ∑ i ∈ H0, E i ω) :=
    P.mean_mono (fdp_le_pointwise hq hm hE hR H0)
  have h2 : P.mean (fun ω => (q / m) * ∑ i ∈ H0, E i ω)
      = (q / m) * ∑ i ∈ H0, P.mean (E i) := by
    rw [P.mean_smul (q / m) (fun ω => ∑ i ∈ H0, E i ω), P.mean_sum H0 E]
  have h3 : ∑ i ∈ H0, P.mean (E i) ≤ (H0.card : ℝ) := by
    calc ∑ i ∈ H0, P.mean (E i) ≤ ∑ _i ∈ H0, (1:ℝ) :=
          Finset.sum_le_sum fun i hi => (hnull i hi).2
      _ = (H0.card : ℝ) := by simp
  have h4 : (q / m) * ∑ i ∈ H0, P.mean (E i) ≤ (q / m) * (H0.card : ℝ) :=
    mul_le_mul_of_nonneg_left h3 (by positivity)
  calc P.mean (fdp H0 R) ≤ (q / m) * ∑ i ∈ H0, P.mean (E i) := by rw [← h2]; exact h1
    _ ≤ (q / m) * (H0.card : ℝ) := h4
    _ = q * H0.card / m := by ring

/-! ### The concrete procedure -/

/-- The candidates whose e-value clears the `k`-th threshold. -/
noncomputable def bigSet (q : ℝ) (m : ℕ) (E : Fin m → Ω → ℝ) (k : ℕ) (ω : Ω) : Finset (Fin m) :=
  Finset.univ.filter (fun i => (m : ℝ) / (q * k) ≤ E i ω)

/-- The e-BH step-up index: the largest `k` for which at least `k` candidates clear the `k`-th
threshold. -/
noncomputable def kHat (q : ℝ) (m : ℕ) (E : Fin m → Ω → ℝ) (ω : Ω) : ℕ :=
  ((Finset.range (m+1)).filter (fun k => k ≤ (bigSet q m E k ω).card)).sup id

/-- The e-BH discovery list. -/
noncomputable def ebh (q : ℝ) (m : ℕ) (E : Fin m → Ω → ℝ) (ω : Ω) : Finset (Fin m) :=
  if kHat q m E ω = 0 then ∅ else bigSet q m E (kHat q m E ω) ω

omit [Fintype Ω] in
lemma kHat_mem (q : ℝ) (m : ℕ) (E : Fin m → Ω → ℝ) (ω : Ω) :
    kHat q m E ω ∈ (Finset.range (m+1)).filter (fun k => k ≤ (bigSet q m E k ω).card) := by
  classical
  set s := (Finset.range (m+1)).filter (fun k => k ≤ (bigSet q m E k ω).card) with hs
  have hne : s.Nonempty := ⟨0, by simp [hs]⟩
  have hEq : s.sup id = s.max' hne :=
    le_antisymm (Finset.sup_le fun b hb => s.le_max' b hb)
      (Finset.le_sup (f := id) (s.max'_mem hne))
  rw [kHat, ← hs, hEq]
  exact s.max'_mem hne

omit [Fintype Ω] in
lemma kHat_le_card (q : ℝ) (m : ℕ) (E : Fin m → Ω → ℝ) (ω : Ω) :
    kHat q m E ω ≤ (bigSet q m E (kHat q m E ω) ω).card :=
  (Finset.mem_filter.mp (kHat_mem q m E ω)).2

omit [Fintype Ω] in
/-- The e-BH list is self-consistent, hence inherits the dependence-free guarantee. -/
theorem ebh_selfConsistent {q : ℝ} (hq : 0 < q) (m : ℕ) (E : Fin m → Ω → ℝ) :
    SelfConsistent q m E (ebh q m E) := by
  intro ω i hi
  unfold ebh at hi
  split_ifs at hi with h0
  · simp at hi
  · have hk : 0 < kHat q m E ω := Nat.pos_of_ne_zero h0
    have hcard : kHat q m E ω ≤ (ebh q m E ω).card := by
      unfold ebh; rw [if_neg h0]; exact kHat_le_card q m E ω
    have hthr : (m : ℝ) / (q * (kHat q m E ω : ℝ)) ≤ E i ω := by
      have := Finset.mem_filter.mp hi
      exact this.2
    have hkR : (0:ℝ) < q * (kHat q m E ω : ℝ) := by
      have : (0:ℝ) < (kHat q m E ω : ℝ) := by exact_mod_cast hk
      positivity
    have hmono : (m : ℝ) / (q * ((ebh q m E ω).card : ℝ)) ≤ (m : ℝ) / (q * (kHat q m E ω : ℝ)) := by
      apply div_le_div_of_nonneg_left (by positivity) hkR
      have : ((kHat q m E ω : ℕ) : ℝ) ≤ (((ebh q m E ω).card : ℕ) : ℝ) := by exact_mod_cast hcard
      nlinarith
    linarith

/-- **The e-BH theorem.**  Under an arbitrary joint law on the screen, the expected false
discovery proportion of the e-BH list at level `q` is at most `q·|H₀|/m`. -/
theorem ebh_fdr_le (P : Law Ω) {q : ℝ} (hq : 0 < q) (hm : 0 < m)
    {E : Fin m → Ω → ℝ} (hE : ∀ i ω, 0 ≤ E i ω) (H0 : Finset (Fin m))
    (hnull : ∀ i ∈ H0, IsEValue P (E i)) :
    P.mean (fdp H0 (ebh q m E)) ≤ q * H0.card / m :=
  selfConsistent_fdr_le P hq hm hE (ebh_selfConsistent hq m E) H0 hnull

/-- ... and in particular at most `q`. -/
theorem ebh_fdr_le_level (P : Law Ω) {q : ℝ} (hq : 0 < q) (hm : 0 < m)
    {E : Fin m → Ω → ℝ} (hE : ∀ i ω, 0 ≤ E i ω) (H0 : Finset (Fin m))
    (hnull : ∀ i ∈ H0, IsEValue P (E i)) :
    P.mean (fdp H0 (ebh q m E)) ≤ q := by
  have h := ebh_fdr_le P hq hm hE H0 hnull
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  have hcard : (H0.card : ℝ) ≤ (m : ℝ) := by
    have : H0.card ≤ m := by
      simpa using (Finset.card_le_card (Finset.subset_univ H0))
    exact_mod_cast this
  have hfin : q * H0.card / m ≤ q := by
    rw [div_le_iff₀ hmR]
    nlinarith
  exact h.trans hfin

omit [Fintype Ω] in
/-- The list is never vacuously empty: a candidate whose e-value reaches `m/q` is always
reported.  (The e-value analogue of "BH contains the Bonferroni list".) -/
theorem mem_ebh_of_large {q : ℝ} (hq : 0 < q) (hm : 0 < m) {E : Fin m → Ω → ℝ} (ω : Ω)
    {i : Fin m} (hi : (m : ℝ) / q ≤ E i ω) : i ∈ ebh q m E ω := by
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  have h1 : i ∈ bigSet q m E 1 ω := by
    simp only [bigSet, Finset.mem_filter, Finset.mem_univ, true_and]
    simpa using hi
  have hcard1 : 1 ≤ (bigSet q m E 1 ω).card := Finset.card_pos.mpr ⟨i, h1⟩
  have hmem1 : 1 ∈ (Finset.range (m+1)).filter (fun k => k ≤ (bigSet q m E k ω).card) := by
    refine Finset.mem_filter.mpr ⟨?_, hcard1⟩
    exact Finset.mem_range.mpr (by omega)
  have hk1 : 1 ≤ kHat q m E ω := by
    have := Finset.le_sup (f := id) hmem1
    simpa [kHat] using this
  have hkne : kHat q m E ω ≠ 0 := by omega
  unfold ebh
  rw [if_neg hkne]
  simp only [bigSet, Finset.mem_filter, Finset.mem_univ, true_and]
  have hkR : (1:ℝ) ≤ (kHat q m E ω : ℝ) := by exact_mod_cast hk1
  have : (m : ℝ) / (q * (kHat q m E ω : ℝ)) ≤ (m : ℝ) / q := by
    apply div_le_div_of_nonneg_left (le_of_lt hmR) hq
    nlinarith
  linarith

/-- **Family-wise control needs no independence either.**  If each candidate's error event has
probability at most `q/m`, the probability of any error at all is at most `q`, under an arbitrary
joint law. -/
theorem bonferroni_fwer (P : Law Ω) {q : ℝ} (hm : 0 < m) (A : Fin m → Finset Ω)
    (h : ∀ i, P.pr (A i) ≤ q / m) :
    P.pr ((Finset.univ : Finset (Fin m)).biUnion A) ≤ q := by
  have hmR : (0:ℝ) < m := by exact_mod_cast hm
  calc P.pr ((Finset.univ : Finset (Fin m)).biUnion A)
      ≤ ∑ i, P.pr (A i) := P.pr_biUnion_le _ A
    _ ≤ ∑ _i : Fin m, q / m := Finset.sum_le_sum fun i _ => h i
    _ = q := by simp; field_simp

end IDR.DepScreen
