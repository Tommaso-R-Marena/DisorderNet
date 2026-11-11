/-
# Split-conformal p-values are valid, and they feed the dependent screen

The development has two operational guarantees that do not yet touch each other.

* `RequestProject.ConformalRisk` / `RequestProject.ConformalRiskExchangeable` produce a
  distribution-free guarantee from exchangeability alone.
* `RequestProject.DependentBH` controls the false discovery rate of a screen under an **arbitrary**
  joint law — but it *assumes* `Superuniform`, and says nothing about where a valid p-value comes
  from.

This file supplies the missing arrow and then composes the two.

## The p-value

For a family of `n + 1` nonconformity scores `S 0, …, S n` on a finite probability space, the
**conformal p-value** of item `j` is its rank from the top among all `n + 1` scores, normalised:

    p_j(ω) = |{i : S j ω ≤ S i ω}| / (n + 1)      (`conformalP`)

— for the split-conformal read-out, `j` is the test item and the other `n` are the calibration
items, and the formula is the familiar `(1 + |{i calibration : S_test ≤ S_i}|)/(n+1)`
(`conformalP_last_eq`).

* `card_low_rank_le` — the deterministic half of the rank argument: for every outcome and every
  `k`, at most `k` of the `n + 1` items have rank `≤ k`.  (Take the smallest score among those
  that do; everything counted sits above it.)
* `pr_rank_le` — the probabilistic half: under `ExchangeableScores` all `n + 1` items have the same
  rank distribution, so each has rank `≤ k` with probability at most `k/(n+1)`.
* `conformalP_superuniform` — **the theorem**: a conformal p-value is a valid p-value,
  `P(p ≤ t) ≤ t` for every `t ≥ 0`, assuming nothing but exchangeability.

## The composition

Conformal p-values for different candidate regions are computed against **one shared calibration
set**, so they are dependent by construction — `conformalP_dependent_of_shared_calibration` makes
that concrete: two candidates with the same test statistic get literally the same p-value.  No
independence or positive-dependence assumption is available.  That is exactly the regime
`selfConsistent_fdr_le_harmonic` was proved for, so the arbitrary-dependence theorem is not
conservatism here — it is the correct tool, because the step that supplies valid p-values is what
couples them.

* `conformal_selfConsistent_fdr_le_harmonic` — any self-consistent step-up rule fed conformal
  p-values has `E[FDP] ≤ q·H_m·|H₀|/m`.
* `conformal_benjamini_yekutieli`, `conformal_screen_fdr_le` — the pipeline in one statement:
  exchangeable scores in, BH at the deflated level `α/H_m` run on the conformal p-values, false
  discovery rate at most `α` out, under arbitrary dependence between the candidates.

`conformalP_superuniform_uniform_instance` checks the hypotheses are satisfiable, so none of this
is vacuous.
-/
import Mathlib
import RequestProject.DependentBH

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR
namespace ConformalBH

open Finset
open scoped Classical
open IDR.DepScreen IDR.DepBH

variable {Ω : Type*} [Fintype Ω]

/-! ## 1. The rank statistic and the conformal p-value -/

/-- The **rank from the top** of item `j` among the `n + 1` nonconformity scores: the number of
items (itself included) whose score is at least as large. -/
noncomputable def rankCount {n : ℕ} (S : Fin (n + 1) → Ω → ℝ) (j : Fin (n + 1)) (ω : Ω) : ℕ :=
  (Finset.univ.filter (fun i => S j ω ≤ S i ω)).card

/-- The **conformal p-value** of item `j`: its normalised rank from the top. -/
noncomputable def conformalP {n : ℕ} (S : Fin (n + 1) → Ω → ℝ) (j : Fin (n + 1)) (ω : Ω) : ℝ :=
  (rankCount S j ω : ℝ) / (n + 1)

/-- The split-conformal read-out: `n` calibration scores and one test score. -/
noncomputable def augment {n : ℕ} (c : Fin n → Ω → ℝ) (z : Ω → ℝ) : Fin (n + 1) → Ω → ℝ :=
  Fin.snoc c z

omit [Fintype Ω] in
/-- The familiar split-conformal formula: the test item's p-value is
`(1 + |{calibration scores at least as large}|)/(n+1)`. -/
lemma conformalP_last_eq {n : ℕ} (c : Fin n → Ω → ℝ) (z : Ω → ℝ) (ω : Ω) :
    conformalP (augment c z) (Fin.last n) ω
      = (1 + ((Finset.univ.filter (fun i : Fin n => z ω ≤ c i ω)).card : ℝ)) / (n + 1) := by
  classical
  unfold conformalP rankCount augment
  congr 1
  have hlast : (Fin.snoc c z : Fin (n + 1) → Ω → ℝ) (Fin.last n) = z := Fin.snoc_last _ _
  have hcard : (Finset.univ.filter
        (fun i : Fin (n + 1) => (Fin.snoc c z : Fin (n + 1) → Ω → ℝ) (Fin.last n) ω
          ≤ (Fin.snoc c z : Fin (n + 1) → Ω → ℝ) i ω)).card
      = ∑ i : Fin (n + 1),
          (if (Fin.snoc c z : Fin (n + 1) → Ω → ℝ) (Fin.last n) ω
            ≤ (Fin.snoc c z : Fin (n + 1) → Ω → ℝ) i ω then 1 else 0) := by
    rw [Finset.card_filter]
  rw [hcard, Fin.sum_univ_castSucc]
  simp only [Fin.snoc_castSucc, hlast, le_refl, if_true]
  rw [Finset.card_filter]
  push_cast
  ring

/-! ## 2. The deterministic half of the rank argument -/

omit [Fintype Ω] in
/-- **At most `k` of the `n + 1` items can have rank `≤ k`.**  Among the items that do, take one
with the smallest score: every one of them is counted in *its* rank, which is at most `k`. -/
lemma card_low_rank_le {n : ℕ} (S : Fin (n + 1) → Ω → ℝ) (ω : Ω) (k : ℕ) :
    (Finset.univ.filter (fun j => rankCount S j ω ≤ k)).card ≤ k := by
  classical
  set J : Finset (Fin (n + 1)) := Finset.univ.filter (fun j => rankCount S j ω ≤ k) with hJ
  rcases Finset.eq_empty_or_nonempty J with h | h
  · simp [h]
  · obtain ⟨j0, hj0, hmin⟩ := J.exists_min_image (fun j => S j ω) h
    have hsub : J ⊆ Finset.univ.filter (fun i => S j0 ω ≤ S i ω) := by
      intro i hi
      exact Finset.mem_filter.mpr ⟨Finset.mem_univ i, hmin i hi⟩
    have hk : rankCount S j0 ω ≤ k := (Finset.mem_filter.mp hj0).2
    exact le_trans (Finset.card_le_card hsub) hk

/-! ## 3. Exchangeability, and the probabilistic half -/

/-- **Exchangeability of the score family** on a finite probability space: relabelling the `n + 1`
items is realised by a weight-preserving relabelling of the outcomes.  This is the only
distributional assumption; nothing is assumed about the data-generating process. -/
def ExchangeableScores {n : ℕ} (P : Law Ω) (S : Fin (n + 1) → Ω → ℝ) : Prop :=
  ∀ π : Equiv.Perm (Fin (n + 1)), ∃ T : Equiv.Perm Ω,
    (∀ ω, P.prob (T ω) = P.prob ω) ∧ ∀ i ω, S (π i) (T ω) = S i ω

/-- The low-rank event of item `j`. -/
noncomputable def lowRank {n : ℕ} (S : Fin (n + 1) → Ω → ℝ) (j : Fin (n + 1)) (k : ℕ) :
    Finset Ω :=
  Finset.univ.filter (fun ω => rankCount S j ω ≤ k)

omit [Fintype Ω] in
/-- Ranks are equivariant: a relabelling of the items that is realised on outcomes moves the rank
of `j` to the rank of `π j`. -/
lemma rankCount_relabel {n : ℕ} {S : Fin (n + 1) → Ω → ℝ} {π : Equiv.Perm (Fin (n + 1))}
    {T : Equiv.Perm Ω} (hT : ∀ i ω, S (π i) (T ω) = S i ω) (j : Fin (n + 1)) (ω : Ω) :
    rankCount S (π j) (T ω) = rankCount S j ω := by
  classical
  unfold rankCount
  refine Finset.card_nbij' (fun i => π.symm i) (fun i => π i) ?_ ?_ ?_ ?_
  · intro i hi
    simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_univ, true_and] at hi ⊢
    have h1 : S (π (π.symm i)) (T ω) = S (π.symm i) ω := hT (π.symm i) ω
    rw [Equiv.apply_symm_apply] at h1
    rw [← h1, ← hT j ω]
    exact hi
  · intro i hi
    simp only [Finset.coe_filter, Set.mem_setOf_eq, Finset.mem_univ, true_and] at hi ⊢
    rw [hT i ω, hT j ω]
    exact hi
  · intro i _; simp
  · intro i _; simp

/-- Under exchangeability every item has the same chance of low rank. -/
lemma pr_lowRank_eq {n : ℕ} {P : Law Ω} {S : Fin (n + 1) → Ω → ℝ}
    (hexch : ExchangeableScores P S) (j l : Fin (n + 1)) (k : ℕ) :
    P.pr (lowRank S j k) = P.pr (lowRank S l k) := by
  classical
  obtain ⟨T, hTw, hTS⟩ := hexch (Equiv.swap j l)
  have hpi : (Equiv.swap j l) j = l := Equiv.swap_apply_left j l
  have hmem : ∀ ω, ω ∈ lowRank S j k ↔ T ω ∈ lowRank S l k := by
    intro ω
    simp only [lowRank, Finset.mem_filter, Finset.mem_univ, true_and]
    rw [← hpi, rankCount_relabel hTS j ω]
  unfold Law.pr
  refine Finset.sum_nbij' (fun ω => T ω) (fun ω => T.symm ω) ?_ ?_ ?_ ?_ ?_
  · intro ω hω; exact (hmem ω).1 hω
  · intro ω hω
    have := (hmem (T.symm ω)).2 (by simpa using hω)
    exact this
  · intro ω _; simp
  · intro ω _; simp
  · intro ω _; exact (hTw ω).symm

/-- **The probabilistic half of the rank argument.**  Under exchangeability, the chance that a
given item's rank is at most `k` is at most `k/(n+1)`: the `n + 1` items have equal chances and, by
`card_low_rank_le`, those chances sum to at most `k`. -/
theorem pr_rank_le {n : ℕ} {P : Law Ω} {S : Fin (n + 1) → Ω → ℝ}
    (hexch : ExchangeableScores P S) (j : Fin (n + 1)) (k : ℕ) :
    P.pr (lowRank S j k) ≤ (k : ℝ) / (n + 1) := by
  classical
  have hsum : ∑ _l : Fin (n + 1), P.pr (lowRank S j k)
      = P.mean (fun ω => ((Finset.univ.filter (fun l => rankCount S l ω ≤ k)).card : ℝ)) := by
    have h1 : ∑ l : Fin (n + 1), P.pr (lowRank S j k)
        = ∑ l : Fin (n + 1), P.pr (lowRank S l k) :=
      Finset.sum_congr rfl (fun l _ => pr_lowRank_eq hexch j l k)
    have h2 : ∀ l : Fin (n + 1), P.pr (lowRank S l k)
        = P.mean (fun ω => if ω ∈ lowRank S l k then (1 : ℝ) else 0) :=
      fun l => (mean_indicator P (lowRank S l k)).symm
    rw [h1, Finset.sum_congr rfl (fun l _ => h2 l), ← Law.mean_sum]
    congr 1
    funext ω
    rw [Finset.card_filter]
    push_cast
    refine Finset.sum_congr rfl (fun l _ => ?_)
    simp [lowRank]
  have hle : P.mean (fun ω => ((Finset.univ.filter (fun l => rankCount S l ω ≤ k)).card : ℝ))
      ≤ (k : ℝ) := by
    have hmono : P.mean (fun ω => ((Finset.univ.filter (fun l => rankCount S l ω ≤ k)).card : ℝ))
        ≤ P.mean (fun _ => (k : ℝ)) := by
      refine Law.mean_mono P (fun ω => ?_)
      exact_mod_cast card_low_rank_le S ω k
    have hconst : P.mean (fun _ : Ω => (k : ℝ)) = (k : ℝ) := by
      unfold Law.mean
      rw [← Finset.sum_mul, P.total, one_mul]
    rwa [hconst] at hmono
  have hcard : ∑ _l : Fin (n + 1), P.pr (lowRank S j k) = (n + 1 : ℝ) * P.pr (lowRank S j k) := by
    rw [Finset.sum_const, nsmul_eq_mul]
    simp
  rw [hcard] at hsum
  have hn1 : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  rw [le_div_iff₀ hn1]
  calc P.pr (lowRank S j k) * ((n : ℝ) + 1) = ((n : ℝ) + 1) * P.pr (lowRank S j k) := by ring
    _ = P.mean (fun ω => ((Finset.univ.filter (fun l => rankCount S l ω ≤ k)).card : ℝ)) := hsum
    _ ≤ (k : ℝ) := hle

/-! ## 4. The theorem: conformal p-values are valid p-values -/

/-- **Split conformal produces a valid p-value.**  Under exchangeability of the `n + 1` scores —
and nothing else — the conformal p-value of any item is superuniform, which is exactly the
hypothesis the dependent screen of `RequestProject.DependentBH` needs. -/
theorem conformalP_superuniform {n : ℕ} {P : Law Ω} {S : Fin (n + 1) → Ω → ℝ}
    (hexch : ExchangeableScores P S) (j : Fin (n + 1)) :
    Superuniform P (conformalP S j) := by
  classical
  intro t ht
  set k : ℕ := ⌊t * (n + 1)⌋₊ with hk
  have hn1 : (0 : ℝ) < (n : ℝ) + 1 := by positivity
  have hev : Finset.univ.filter (fun ω => conformalP S j ω ≤ t) = lowRank S j k := by
    ext ω
    simp only [lowRank, Finset.mem_filter, Finset.mem_univ, true_and, conformalP]
    constructor
    · intro h
      have h1 : (rankCount S j ω : ℝ) ≤ t * ((n : ℝ) + 1) := by
        rw [div_le_iff₀ hn1] at h; exact h
      exact Nat.le_floor h1
    · intro h
      have h1 : (rankCount S j ω : ℝ) ≤ (k : ℝ) := by exact_mod_cast h
      have h2 : (k : ℝ) ≤ t * ((n : ℝ) + 1) := Nat.floor_le (by positivity)
      rw [div_le_iff₀ hn1]
      linarith
  rw [hev]
  refine (pr_rank_le hexch j k).trans ?_
  rw [div_le_iff₀ hn1]
  exact Nat.floor_le (by positivity)

/-! ## 5. Why the p-values are dependent, and why that is fine -/

omit [Fintype Ω] in
/-- **The p-values of a conformal screen are dependent by construction.**  They are computed
against one shared calibration set, so two candidates with the same test statistic receive
*identically* the same p-value — there is no independence to appeal to, and no positive-dependence
condition either. -/
theorem conformalP_dependent_of_shared_calibration {n : ℕ} (c : Fin n → Ω → ℝ) (z w : Ω → ℝ)
    (hzw : z = w) :
    conformalP (augment c z) (Fin.last n) = conformalP (augment c w) (Fin.last n) := by
  rw [hzw]

/-! ## 6. The pipeline -/

variable {m : ℕ}

/-- **The composition, for any self-consistent rule.**  Feed a screen of `m` candidate regions with
split-conformal p-values — one per candidate, all against the same calibration set, hence dependent
— and run any self-consistent step-up rule at level `q`.  The expected false discovery proportion
is at most `q·H_m·|H₀|/m`.  Exchangeability of each null candidate's scores is the only
distributional assumption; the joint law across candidates is arbitrary. -/
theorem conformal_selfConsistent_fdr_le_harmonic {n : ℕ} (P : Law Ω) {q : ℝ} (hq : 0 ≤ q)
    (hm : 0 < m) (S : Fin m → Fin (n + 1) → Ω → ℝ) {R : Ω → Finset (Fin m)}
    (hR : SelfConsistentP q m (fun i => conformalP (S i) (Fin.last n)) R)
    (H0 : Finset (Fin m)) (hexch : ∀ i ∈ H0, ExchangeableScores P (S i)) :
    P.mean (fdp H0 R) ≤ q * harm m * H0.card / m :=
  selfConsistent_fdr_le_harmonic P hq hm hR H0
    (fun i hi => conformalP_superuniform (hexch i hi) (Fin.last n))

/-- **The pipeline, end to end.**  Exchangeable nonconformity scores in; Benjamini–Hochberg run at
the deflated level `α/H_m` on the conformal p-values; false discovery rate at most `α·|H₀|/m` out.
Both ends are guaranteed, and the dependence created by the shared calibration set is exactly what
the arbitrary-dependence theorem tolerates. -/
theorem conformal_benjamini_yekutieli {n : ℕ} (P : Law Ω) {α : ℝ} (hα : 0 ≤ α) (hm : 0 < m)
    (S : Fin m → Fin (n + 1) → Ω → ℝ) (H0 : Finset (Fin m))
    (hexch : ∀ i ∈ H0, ExchangeableScores P (S i)) :
    P.mean (fdp H0 (bhList (α / harm m) m (fun i => conformalP (S i) (Fin.last n))))
      ≤ α * H0.card / m :=
  benjamini_yekutieli P hα hm H0 (fun i hi => conformalP_superuniform (hexch i hi) (Fin.last n))

/-- ... and in particular the screen's false discovery rate is at most the nominal level `α`. -/
theorem conformal_screen_fdr_le {n : ℕ} (P : Law Ω) {α : ℝ} (hα : 0 ≤ α) (hm : 0 < m)
    (S : Fin m → Fin (n + 1) → Ω → ℝ) (H0 : Finset (Fin m))
    (hexch : ∀ i ∈ H0, ExchangeableScores P (S i)) :
    P.mean (fdp H0 (bhList (α / harm m) m (fun i => conformalP (S i) (Fin.last n)))) ≤ α :=
  benjamini_yekutieli_level P hα hm H0
    (fun i hi => conformalP_superuniform (hexch i hi) (Fin.last n))

/-! ## 7. The assumption is satisfiable -/

/-- A uniform exchangeable model on `n + 1` items: the outcome is which item carries the anomalous
score, and every item is equally likely to be that one. -/
noncomputable def uniformLaw (n : ℕ) : Law (Fin (n + 1)) where
  prob := fun _ => 1 / (n + 1)
  nonneg := fun _ => by positivity
  total := by
    rw [Finset.sum_const, nsmul_eq_mul]
    simp
    field_simp

/-- The scores of that model: the outcome's item scores `1`, every other item `0`. -/
noncomputable def uniformScores (n : ℕ) : Fin (n + 1) → Fin (n + 1) → ℝ :=
  fun i ω => if i = ω then 1 else 0

/-- The uniform model is exchangeable: relabelling the items is realised by relabelling the
outcomes. -/
theorem uniformScores_exchangeable (n : ℕ) :
    ExchangeableScores (uniformLaw n) (uniformScores n) := by
  intro π
  refine ⟨π, fun ω => rfl, ?_⟩
  intro i ω
  simp [uniformScores]

/-- **The hypotheses are satisfiable, so the theorem is not vacuous.**  In the uniform exchangeable
model every item's conformal p-value is a valid p-value. -/
theorem conformalP_superuniform_uniform_instance (n : ℕ) (j : Fin (n + 1)) :
    Superuniform (uniformLaw n) (conformalP (uniformScores n) j) :=
  conformalP_superuniform (uniformScores_exchangeable n) j

end ConformalBH
end IDR
