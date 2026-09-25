/-
# Part XCIV  The harmonic correction: Benjamini–Hochberg under *arbitrary* dependence

`RequestProject.FalseDiscovery` (Part XCI) proves false-discovery-rate control for the
Benjamini–Hochberg list in a **product** experiment — one independent coordinate per candidate
region.  `RequestProject.DependentScreen` (Part XCIII) removes the independence assumption, at the
price of changing the currency: it asks each candidate for an *e-value* rather than a p-value.

Both files record the same gap, in the same words: *"under arbitrary dependence the
Benjamini–Hochberg procedure needs the harmonic correction, which is not proved here."*  A real
proteome screen for disordered regions reports p-values, not e-values, and its candidates share a
calibration run, a plate and a batch, so neither of the two existing theorems applies to it.  This
file proves the missing statement — the Benjamini–Yekutieli theorem — from scratch, in the same
finite, measure-theory-free setting.

*The joint law is arbitrary.*  `Law Ω` (reused from Part XCIII) is any probability distribution on
a finite outcome space; no product structure, no independence, no positive-regression-dependence
condition, nothing.

* `Superuniform` — the only assumption: each **null** candidate's p-value is valid,
  `P(p_i ≤ t) ≤ t` for every `t ≥ 0`.  Nothing whatsoever is assumed about the non-nulls.
  `superuniform_grid` shows the condition is satisfiable and not vacuous (the exact p-value of a
  discrete uniform read-out), and `not_superuniform_of_const` that it has content.
* `harm` — the harmonic number `H_m = 1 + 1/2 + ⋯ + 1/m`, with `harm_le_one_add_log` and
  `log_le_harm` locating it between `log(m+1)` and `1 + log m`: the correction costs a factor that
  grows only logarithmically in the size of the screen.
* `SelfConsistentP` — the abstract shape of a step-up rule at level `q`: every reported candidate
  has `p_i ≤ |R|·q/m`.  The BH list is of this shape (`bhList_selfConsistent`), and so is every
  step-up rule ever proposed.
* `wgt_decomp` — **the mechanism**, and the only place the harmonic number enters: the weight
  `1/|R|` carried by a discovery is rewritten as a telescoping sum of the indicators of the
  *nested* events `{i ∈ R and |R| ≤ j+1}`, each of which is contained in the single-candidate
  event `{p_i ≤ (j+1)·q/m}` whose probability the null assumption controls.  No independence is
  used, because the events are no longer disjoint slices `{|R| = k}` that would have to be
  recombined.
* `mean_wgt_le` — the per-candidate bound `E[1{i ∈ R}/|R|] ≤ (q/m)·H_m`.
* `selfConsistent_fdr_le_harmonic` — **the general theorem**: any self-consistent step-up rule at
  level `q`, fed valid null p-values, has `E[FDP] ≤ q·H_m·|H₀|/m` under an arbitrary joint law.
* `benjamini_yekutieli`, `benjamini_yekutieli_level` — **the corrected procedure**: running BH at
  the deflated level `α/H_m` gives `E[FDP] ≤ α·|H₀|/m ≤ α`, whatever the dependence.
* `by_dominates_bonferroni` — the corrected list still contains the Bonferroni list at the same
  overall level, so the correction is never worse than the family-wise procedure it replaces.
* `bh_uncorrected_bound` — what the same argument says about *uncorrected* BH under dependence:
  `E[FDP] ≤ q·H_m`, an inflation by the harmonic factor.  That factor is not an artefact of the
  proof: `RequestProject.DependentBHSharp` constructs, for every `m` and every level, a joint law
  with all `m` hypotheses null and all p-values valid on which uncorrected BH attains `q·H_m`
  exactly, and on which the corrected procedure sits exactly at its own bound.

Scope, as elsewhere in Parts XC–XCIII: this is multiplicity control, and it is orthogonal to
calibration.  If the null law of the read-out is wrong, `Superuniform` fails and every bound here
is void.
-/
import Mathlib
import RequestProject.DependentScreen
import RequestProject.FalseDiscovery

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR
namespace DepBH

open IDR.DepScreen

variable {Ω : Type*} [Fintype Ω] {m : ℕ}

/-! ## 1. Harmonic numbers -/

/-- The harmonic number `H_m = 1 + 1/2 + ⋯ + 1/m`. -/
noncomputable def harm (m : ℕ) : ℝ := ∑ j ∈ Finset.range m, (1 : ℝ) / (j + 1)

lemma harm_zero : harm 0 = 0 := by simp [harm]

lemma harm_succ (m : ℕ) : harm (m + 1) = harm m + 1 / (m + 1) := by
  simp [harm, Finset.sum_range_succ]

lemma harm_eq_harmonic (m : ℕ) : harm m = (harmonic m : ℝ) := by
  induction m with
  | zero => simp [harm, harmonic]
  | succ n ih => rw [harm_succ, ih, harmonic_succ]; push_cast; ring

lemma harm_pos {m : ℕ} (hm : 0 < m) : 0 < harm m := by
  apply Finset.sum_pos
  · intro j _; positivity
  · exact ⟨0, by simpa using hm⟩

lemma one_le_harm {m : ℕ} (hm : 0 < m) : 1 ≤ harm m := by
  obtain ⟨n, rfl⟩ : ∃ n, m = n + 1 := ⟨m - 1, by omega⟩
  unfold harm
  rw [Finset.sum_range_succ']
  have h0 : (0 : ℝ) ≤ ∑ j ∈ Finset.range n, (1 : ℝ) / ((j : ℝ) + 1 + 1) := by positivity
  push_cast
  linarith

/-- The correction factor grows only logarithmically: `H_m ≤ 1 + log m`. -/
lemma harm_le_one_add_log (m : ℕ) : harm m ≤ 1 + Real.log m := by
  rw [harm_eq_harmonic]; exact harmonic_le_one_add_log m

/-- ... and it does grow: `log (m+1) ≤ H_m`. -/
lemma log_le_harm (m : ℕ) : Real.log ((m : ℝ) + 1) ≤ harm m := by
  have h := log_add_one_le_harmonic m
  rw [harm_eq_harmonic]
  simpa using h

/-! ## 2. Valid p-values under an arbitrary joint law -/

/-- A **valid (superuniform) p-value**: under the joint law of the screen, the chance that
candidate `i`'s p-value falls below a threshold `t` is at most `t`.  This is the only
distributional assumption made in this file, and it is made only for the true nulls. -/
def Superuniform (P : Law Ω) (X : Ω → ℝ) : Prop :=
  ∀ t : ℝ, 0 ≤ t → P.pr (Finset.univ.filter (fun ω => X ω ≤ t)) ≤ t

/-- A statistic that is *not* a valid p-value: the condition has content. -/
lemma not_superuniform_of_const (P : Law Ω) {c : ℝ} (hc : 0 ≤ c) (hc1 : c < 1) :
    ¬ Superuniform P (fun _ => c) := by
  intro h
  have h1 := h c hc
  have h2 : (Finset.univ.filter (fun _ : Ω => c ≤ c)) = Finset.univ :=
    Finset.filter_true_of_mem (fun _ _ => le_rfl)
  rw [h2] at h1
  have h3 : P.pr Finset.univ = 1 := by simpa [Law.pr] using P.total
  rw [h3] at h1
  linarith

/-! ## 3. Self-consistent step-up rules -/

/-- A rejection rule is **self-consistent at level `q`** if every candidate it reports has a
p-value at most `|R|·q/m`.  Every step-up procedure has this shape. -/
def SelfConsistentP (q : ℝ) (m : ℕ) (p : Fin m → Ω → ℝ) (R : Ω → Finset (Fin m)) : Prop :=
  ∀ ω, ∀ i ∈ R ω, p i ω ≤ ((R ω).card : ℝ) * q / m

/-- The weight a discovery contributes to the false discovery proportion. -/
noncomputable def wgt (R : Ω → Finset (Fin m)) (i : Fin m) (ω : Ω) : ℝ :=
  if i ∈ R ω then 1 / ((R ω).card : ℝ) else 0

omit [Fintype Ω] in
lemma wgt_nonneg (R : Ω → Finset (Fin m)) (i : Fin m) (ω : Ω) : 0 ≤ wgt R i ω := by
  unfold wgt; split_ifs
  · positivity
  · exact le_rfl

omit [Fintype Ω] in
/-- The false discovery proportion is the total weight of the true nulls on the list. -/
lemma fdp_eq_sum_wgt (H0 : Finset (Fin m)) (R : Ω → Finset (Fin m)) (ω : Ω) :
    fdp H0 R ω = ∑ i ∈ H0, wgt R i ω := by
  classical
  unfold fdp wgt
  have hsum : ∑ i ∈ H0, (if i ∈ R ω then 1 / ((R ω).card : ℝ) else 0)
      = ((H0 ∩ R ω).card : ℝ) * (1 / ((R ω).card : ℝ)) := by
    rw [Finset.sum_ite_mem, Finset.sum_const, nsmul_eq_mul]
  rw [hsum, Finset.inter_comm H0 (R ω)]
  split_ifs with h
  · have hR : R ω = ∅ := Finset.card_eq_zero.mp h
    simp [hR]
  · field_simp

/-! ## 4. The telescoping decomposition -/

lemma telescope {a b : ℕ} (h : a ≤ b) :
    ∑ j ∈ Finset.Ico a b, ((1 : ℝ) / (j + 1) - 1 / (j + 2)) = 1 / ((a : ℝ) + 1) - 1 / ((b : ℝ) + 1) := by
  induction b, h using Nat.le_induction with
  | base => simp
  | succ n hn ih =>
      rw [Finset.sum_Ico_succ_top hn, ih]
      push_cast
      ring

omit [Fintype Ω] in
/-- **The mechanism.**  The weight `1/|R|` of a discovery is a telescoping combination of the
indicators of the *nested* events `{i ∈ R and |R| ≤ j+1}`, plus a boundary term.  After
self-consistency each of those events is contained in an event about candidate `i` alone, which is
why no assumption on the joint law is ever needed. -/
lemma wgt_decomp {R : Ω → Finset (Fin m)} (i : Fin m) (ω : Ω) :
    wgt R i ω
      = (∑ j ∈ Finset.range m,
          ((1 : ℝ) / (j + 1) - 1 / (j + 2)) * (if i ∈ R ω ∧ (R ω).card ≤ j + 1 then 1 else 0))
        + (1 / ((m : ℝ) + 1)) * (if i ∈ R ω then 1 else 0) := by
  classical
  unfold wgt
  by_cases h : i ∈ R ω
  · have hk1 : 1 ≤ (R ω).card := Finset.card_pos.mpr ⟨i, h⟩
    have hkm : (R ω).card ≤ m := by
      simpa using Finset.card_le_card (Finset.subset_univ (R ω))
    set k := (R ω).card with hk
    have hfilter : (Finset.range m).filter (fun j => k ≤ j + 1) = Finset.Ico (k - 1) m := by
      ext j; simp only [Finset.mem_filter, Finset.mem_Ico, Finset.mem_range]; omega
    have hsum : (∑ j ∈ Finset.range m,
          ((1 : ℝ) / (j + 1) - 1 / (j + 2)) * (if i ∈ R ω ∧ k ≤ j + 1 then 1 else 0))
        = ∑ j ∈ Finset.Ico (k - 1) m, ((1 : ℝ) / (j + 1) - 1 / (j + 2)) := by
      rw [← hfilter, Finset.sum_filter]
      refine Finset.sum_congr rfl (fun j _ => ?_)
      by_cases hj : k ≤ j + 1 <;> simp [h, hj]
    have hcast : ((k - 1 : ℕ) : ℝ) + 1 = (k : ℝ) := by
      have hk' : (k - 1 : ℕ) + 1 = k := by omega
      exact_mod_cast congrArg (Nat.cast : ℕ → ℝ) hk'
    rw [hsum, telescope (le_trans (Nat.sub_le k 1) hkm), hcast]
    simp only [h, if_true, mul_one]
    ring
  · simp [h]

/-! ## 5. The per-candidate bound -/

lemma mean_add (P : Law Ω) (X Y : Ω → ℝ) :
    P.mean (fun ω => X ω + Y ω) = P.mean X + P.mean Y := by
  unfold Law.mean
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun ω _ => by ring)

lemma mean_indicator (P : Law Ω) (S : Finset Ω) :
    P.mean (fun ω => if ω ∈ S then (1 : ℝ) else 0) = P.pr S := by
  classical
  simp only [Law.mean, Law.pr, mul_ite, mul_one, mul_zero]
  rw [Finset.sum_ite_mem]
  simp

lemma sum_two_inv (m : ℕ) : ∑ j ∈ Finset.range m, (1 : ℝ) / (j + 2) = harm (m + 1) - 1 := by
  unfold harm
  rw [Finset.sum_range_succ']
  push_cast
  ring_nf

/-- **The per-candidate bound.**  For a true null, the expected weight it contributes to the false
discovery proportion of a self-consistent rule is at most `(q/m)·H_m`. -/
lemma mean_wgt_le (P : Law Ω) {q : ℝ} (hq : 0 ≤ q) (hm : 0 < m) {p : Fin m → Ω → ℝ}
    {R : Ω → Finset (Fin m)} (hR : SelfConsistentP q m p R) {i : Fin m}
    (hi : Superuniform P (p i)) :
    P.mean (wgt R i) ≤ q / m * harm m := by
  classical
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  set A : ℕ → Finset Ω := fun j => Finset.univ.filter (fun ω => i ∈ R ω ∧ (R ω).card ≤ j + 1)
    with hA
  set B : Finset Ω := Finset.univ.filter (fun ω => i ∈ R ω) with hB
  have hstep : P.mean (wgt R i)
      = (∑ j ∈ Finset.range m, ((1 : ℝ) / (j + 1) - 1 / (j + 2)) * P.pr (A j))
        + (1 / ((m : ℝ) + 1)) * P.pr B := by
    have hfun : (wgt R i) = fun ω =>
        (∑ j ∈ Finset.range m, ((1 : ℝ) / (j + 1) - 1 / (j + 2)) *
            (if ω ∈ A j then (1 : ℝ) else 0))
          + (1 / ((m : ℝ) + 1)) * (if ω ∈ B then (1 : ℝ) else 0) := by
      funext ω
      rw [wgt_decomp i ω]
      simp [hA, hB]
    rw [hfun, mean_add, Law.mean_sum]
    congr 1
    · refine Finset.sum_congr rfl (fun j _ => ?_)
      rw [Law.mean_smul, mean_indicator]
    · rw [Law.mean_smul, mean_indicator]
  rw [hstep]
  have hAle : ∀ j : ℕ, P.pr (A j) ≤ ((j : ℝ) + 1) * q / m := by
    intro j
    have hsub : A j ⊆ Finset.univ.filter (fun ω => p i ω ≤ ((j : ℝ) + 1) * q / m) := by
      intro ω hω
      simp only [hA, Finset.mem_filter, Finset.mem_univ, true_and] at hω ⊢
      have h1 := hR ω i hω.1
      have h2 : ((R ω).card : ℝ) ≤ (j : ℝ) + 1 := by exact_mod_cast hω.2
      have h3 : ((R ω).card : ℝ) * q / m ≤ ((j : ℝ) + 1) * q / m := by gcongr
      linarith
    exact le_trans (P.pr_mono hsub) (hi _ (by positivity))
  have hBle : P.pr B ≤ q := by
    have hsub : B ⊆ Finset.univ.filter (fun ω => p i ω ≤ q) := by
      intro ω hω
      simp only [hB, Finset.mem_filter, Finset.mem_univ, true_and] at hω ⊢
      have h1 := hR ω i hω
      have h2 : ((R ω).card : ℝ) ≤ (m : ℝ) := by
        exact_mod_cast (by simpa using Finset.card_le_card (Finset.subset_univ (R ω)))
      have h3 : ((R ω).card : ℝ) * q / m ≤ q := by
        rw [div_le_iff₀ hmR]; nlinarith
      linarith
    exact le_trans (P.pr_mono hsub) (hi _ hq)
  have hcoef : ∀ j : ℕ, (0 : ℝ) ≤ (1 : ℝ) / (j + 1) - 1 / (j + 2) := by
    intro j
    have h1 : (0 : ℝ) < (j : ℝ) + 1 := by positivity
    rw [sub_nonneg]
    apply one_div_le_one_div_of_le h1
    linarith
  calc (∑ j ∈ Finset.range m, ((1 : ℝ) / (j + 1) - 1 / (j + 2)) * P.pr (A j))
        + (1 / ((m : ℝ) + 1)) * P.pr B
      ≤ (∑ j ∈ Finset.range m, ((1 : ℝ) / (j + 1) - 1 / (j + 2)) * (((j : ℝ) + 1) * q / m))
        + (1 / ((m : ℝ) + 1)) * q := by
        gcongr with j hj
        · exact hcoef j
        · exact hAle j
    _ = q / m * harm m := by
        have hterm : ∀ j ∈ Finset.range m,
            ((1 : ℝ) / (j + 1) - 1 / (j + 2)) * (((j : ℝ) + 1) * q / m)
              = (q / m) * (1 / ((j : ℝ) + 2)) := by
          intro j _
          have h1 : ((j : ℝ) + 1) ≠ 0 := by positivity
          have h2 : ((j : ℝ) + 2) ≠ 0 := by positivity
          field_simp
          ring
        rw [Finset.sum_congr rfl hterm, ← Finset.mul_sum, sum_two_inv m, harm_succ]
        field_simp
        ring

/-! ## 6. The theorem -/

/-- **False discovery rate control under arbitrary dependence.**  Any self-consistent step-up rule
at level `q`, fed valid p-values for the true nulls, has expected false discovery proportion at
most `q·H_m·|H₀|/m`.  Nothing is assumed about the joint law of the screen, and nothing about the
non-null candidates. -/
theorem selfConsistent_fdr_le_harmonic (P : Law Ω) {q : ℝ} (hq : 0 ≤ q) (hm : 0 < m)
    {p : Fin m → Ω → ℝ} {R : Ω → Finset (Fin m)} (hR : SelfConsistentP q m p R)
    (H0 : Finset (Fin m)) (hnull : ∀ i ∈ H0, Superuniform P (p i)) :
    P.mean (fdp H0 R) ≤ q * harm m * H0.card / m := by
  classical
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hrw : P.mean (fdp H0 R) = ∑ i ∈ H0, P.mean (wgt R i) := by
    rw [show (fdp H0 R) = fun ω => ∑ i ∈ H0, wgt R i ω from
      funext (fun ω => fdp_eq_sum_wgt H0 R ω), Law.mean_sum]
  rw [hrw]
  calc ∑ i ∈ H0, P.mean (wgt R i)
      ≤ ∑ _i ∈ H0, q / m * harm m :=
        Finset.sum_le_sum (fun i hi => mean_wgt_le P hq hm hR (hnull i hi))
    _ = q * harm m * H0.card / m := by
        rw [Finset.sum_const, nsmul_eq_mul]
        field_simp

/-! ## 7. The Benjamini–Hochberg list and its harmonic correction -/

/-- The BH list of Part XCI, read as a function of the outcome through the observed p-values. -/
noncomputable def bhList (q : ℝ) (m : ℕ) (p : Fin m → Ω → ℝ) (ω : Ω) : Finset (Fin m) :=
  IDR.FDR.bhRej m q (fun i => p i ω)

omit [Fintype Ω] in
/-- The BH list is a self-consistent step-up rule: it reports exactly the candidates below the
threshold set by its own length. -/
theorem bhList_selfConsistent {q : ℝ} (hq : 0 ≤ q) (m : ℕ) (p : Fin m → Ω → ℝ) :
    SelfConsistentP q m p (bhList q m p) := by
  intro ω i hi
  have hcard : (bhList q m p ω).card = IDR.FDR.bhR m q (fun i => p i ω) :=
    IDR.FDR.card_bhRej m hq (fun i => p i ω)
  have hmem : p i ω ≤ (IDR.FDR.bhR m q (fun i => p i ω) : ℝ) * q / m := by
    have := Finset.mem_filter.mp hi
    exact this.2
  rw [hcard]
  exact hmem

/-- **The Benjamini–Yekutieli theorem.**  Under an arbitrary joint law on the screen, the BH list
run at the deflated level `α/H_m` has expected false discovery proportion at most `α·|H₀|/m`. -/
theorem benjamini_yekutieli (P : Law Ω) {α : ℝ} (hα : 0 ≤ α) (hm : 0 < m)
    {p : Fin m → Ω → ℝ} (H0 : Finset (Fin m)) (hnull : ∀ i ∈ H0, Superuniform P (p i)) :
    P.mean (fdp H0 (bhList (α / harm m) m p)) ≤ α * H0.card / m := by
  have hH : 0 < harm m := harm_pos hm
  have hq : 0 ≤ α / harm m := div_nonneg hα (le_of_lt hH)
  have h := selfConsistent_fdr_le_harmonic P hq hm (bhList_selfConsistent hq m p) H0 hnull
  calc P.mean (fdp H0 (bhList (α / harm m) m p))
      ≤ (α / harm m) * harm m * H0.card / m := h
    _ = α * H0.card / m := by field_simp

/-- ... and in particular at most `α`, the nominal level. -/
theorem benjamini_yekutieli_level (P : Law Ω) {α : ℝ} (hα : 0 ≤ α) (hm : 0 < m)
    {p : Fin m → Ω → ℝ} (H0 : Finset (Fin m)) (hnull : ∀ i ∈ H0, Superuniform P (p i)) :
    P.mean (fdp H0 (bhList (α / harm m) m p)) ≤ α := by
  have h := benjamini_yekutieli P hα hm H0 hnull
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hcard : (H0.card : ℝ) ≤ (m : ℝ) := by
    exact_mod_cast (by simpa using Finset.card_le_card (Finset.subset_univ H0))
  have hfin : α * H0.card / m ≤ α := by
    rw [div_le_iff₀ hmR]; nlinarith
  exact h.trans hfin

/-- What the same argument gives for *uncorrected* BH under arbitrary dependence: the guarantee
survives, inflated by the harmonic factor. -/
theorem bh_uncorrected_bound (P : Law Ω) {q : ℝ} (hq : 0 ≤ q) (hm : 0 < m)
    {p : Fin m → Ω → ℝ} (H0 : Finset (Fin m)) (hnull : ∀ i ∈ H0, Superuniform P (p i)) :
    P.mean (fdp H0 (bhList q m p)) ≤ q * harm m := by
  have h := selfConsistent_fdr_le_harmonic P hq hm (bhList_selfConsistent hq m p) H0 hnull
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hcard : (H0.card : ℝ) ≤ (m : ℝ) := by
    exact_mod_cast (by simpa using Finset.card_le_card (Finset.subset_univ H0))
  have hH : 0 < harm m := harm_pos hm
  have hfin : q * harm m * H0.card / m ≤ q * harm m := by
    rw [div_le_iff₀ hmR]
    nlinarith [mul_nonneg hq (le_of_lt hH)]
  exact h.trans hfin

omit [Fintype Ω] in
/-- The corrected list still contains the Bonferroni list at the same overall level: paying the
harmonic correction never loses a discovery to the family-wise procedure it replaces. -/
theorem by_dominates_bonferroni {α : ℝ} (hα : 0 ≤ α) {m : ℕ} {p : Fin m → Ω → ℝ} (ω : Ω)
    {i : Fin m} (hi : p i ω ≤ α / (harm m * m)) :
    i ∈ bhList (α / harm m) m p ω := by
  have hm : 0 < m := by
    by_contra h
    have hm0 : m = 0 := by omega
    subst hm0
    exact absurd i.2 (by omega)
  have hH : 0 < harm m := harm_pos hm
  have hq : 0 ≤ α / harm m := div_nonneg hα (le_of_lt hH)
  have hthr : p i ω ≤ (α / harm m) / m := by
    calc p i ω ≤ α / (harm m * m) := hi
      _ = (α / harm m) / m := by rw [div_div]
  exact IDR.FDR.bh_dominates_bonferroni m hq (fun i => p i ω) hthr

/-! ## 8. A tail bound, and what the correction costs at screen scale -/

/-- Markov's inequality for a finite law. -/
lemma markov (P : Law Ω) {X : Ω → ℝ} (hX : ∀ ω, 0 ≤ X ω) {γ : ℝ} (hγ : 0 < γ) :
    P.pr (Finset.univ.filter (fun ω => γ ≤ X ω)) ≤ P.mean X / γ := by
  classical
  rw [le_div_iff₀ hγ]
  unfold Law.pr Law.mean
  calc (∑ ω ∈ Finset.univ.filter (fun ω => γ ≤ X ω), P.prob ω) * γ
      = ∑ ω ∈ Finset.univ.filter (fun ω => γ ≤ X ω), P.prob ω * γ := by rw [Finset.sum_mul]
    _ ≤ ∑ ω ∈ Finset.univ.filter (fun ω => γ ≤ X ω), P.prob ω * X ω := by
        refine Finset.sum_le_sum (fun ω hω => ?_)
        exact mul_le_mul_of_nonneg_left (Finset.mem_filter.mp hω).2 (P.nonneg ω)
    _ ≤ ∑ ω, P.prob ω * X ω :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
          (fun ω _ _ => mul_nonneg (P.nonneg ω) (hX ω))

/-- **A tail bound for the corrected list.**  The false discovery *proportion* of the corrected
list exceeds `γ` with probability at most `α/γ`, again under arbitrary dependence.  (Expected-value
control is what the procedure guarantees; this is what it implies about a single run.) -/
theorem by_fdp_tail (P : Law Ω) {α : ℝ} (hα : 0 ≤ α) (hm : 0 < m) {p : Fin m → Ω → ℝ}
    (H0 : Finset (Fin m)) (hnull : ∀ i ∈ H0, Superuniform P (p i)) {γ : ℝ} (hγ : 0 < γ) :
    P.pr (Finset.univ.filter
        (fun ω => γ ≤ fdp H0 (bhList (α / harm m) m p) ω)) ≤ α / γ := by
  refine le_trans (markov P (fun ω => fdp_nonneg H0 _ ω) hγ) ?_
  exact div_le_div_of_nonneg_right (benjamini_yekutieli_level P hα hm H0 hnull) (le_of_lt hγ)

/-- At the scale of a proteome screen the correction is cheap: with a thousand candidates it costs
less than a factor of eight, against Bonferroni's factor of a thousand. -/
lemma harm_thousand_le_eight : harm 1000 ≤ 8 := by
  have h1 : Real.log 1000 ≤ 7 := by
    have he : (1000 : ℝ) ≤ Real.exp 7 := by
      have h2 : (2.7182818283 : ℝ) < Real.exp 1 := Real.exp_one_gt_d9
      have h3 : Real.exp 7 = (Real.exp 1) ^ (7 : ℕ) := by
        rw [← Real.exp_nat_mul]
        norm_num
      rw [h3]
      nlinarith [pow_le_pow_left₀ (by norm_num : (0:ℝ) ≤ 2.7182818283) (le_of_lt h2) 7]
    calc Real.log 1000 ≤ Real.log (Real.exp 7) := Real.log_le_log (by norm_num) he
      _ = 7 := Real.log_exp 7
  have h2 := harm_le_one_add_log 1000
  norm_num at h2 ⊢
  linarith

/-! ## 9. The assumption is satisfiable -/

/-- The exact p-value of a discrete uniform read-out on `N` levels is a valid p-value, so the
hypotheses of the theorem are satisfiable and its conclusion is not vacuous. -/
lemma superuniform_grid {N : ℕ} (hN : 0 < N) (P : Law (Fin N)) (hP : ∀ ω, P.prob ω = 1 / N) :
    Superuniform P (fun ω => (((ω : ℕ) : ℝ) + 1) / N) := by
  classical
  intro t ht
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  set S := Finset.univ.filter (fun ω : Fin N => (((ω : ℕ) : ℝ) + 1) / N ≤ t) with hS
  have hpr : P.pr S = (S.card : ℝ) / N := by
    unfold Law.pr
    rw [Finset.sum_congr rfl (fun ω _ => hP ω), Finset.sum_const, nsmul_eq_mul]
    ring
  have hmaps : ∀ ω ∈ S, (ω : ℕ) ∈ Finset.range ⌊t * N⌋₊ := by
    intro ω hω
    simp only [hS, Finset.mem_filter, Finset.mem_univ, true_and] at hω
    have h1 : ((ω : ℕ) : ℝ) + 1 ≤ t * N := by
      rw [div_le_iff₀ hNR] at hω; linarith
    have h2 : (((ω : ℕ) + 1 : ℕ) : ℝ) ≤ t * N := by push_cast; linarith
    have h3 : (ω : ℕ) + 1 ≤ ⌊t * N⌋₊ := Nat.le_floor h2
    exact Finset.mem_range.mpr (by omega)
  have hcardN : S.card ≤ ⌊t * N⌋₊ := by
    have h := Finset.card_le_card_of_injOn (fun ω : Fin N => (ω : ℕ))
      (t := Finset.range ⌊t * N⌋₊) (fun ω hω => by simpa using hmaps ω (by simpa using hω))
      (fun a _ b _ hab => Fin.ext hab)
    simpa using h
  have hcard : (S.card : ℝ) ≤ t * N := by
    calc (S.card : ℝ) ≤ (⌊t * N⌋₊ : ℝ) := by exact_mod_cast hcardN
      _ ≤ t * N := Nat.floor_le (by positivity)
  rw [hpr, div_le_iff₀ hNR]
  exact hcard

end DepBH
end IDR
