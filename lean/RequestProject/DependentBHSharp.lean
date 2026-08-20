/-
# Part XCIV.2  The harmonic correction is *necessary*: a sharp instance

`RequestProject.DependentBH` proves that under an arbitrary joint law the Benjamini–Hochberg list
at level `q` still controls the false discovery rate, but only up to the harmonic factor
(`bh_uncorrected_bound` : `E[FDP] ≤ q·H_m`), which is why the corrected procedure must be run at
`α/H_m`.  A guarantee of that shape is worth having only if the factor is real: if `q·H_m` were an
artefact of the proof, deflating the level would be throwing away power for nothing.

This file settles the question by constructing, for every screen size `m` and every level `q`, a
joint law on which uncorrected BH attains `q·H_m` **exactly**.

*The construction.*  The outcome space is `none` plus the pairs `(j, s)` with `j, s : Fin m`.  In
outcome `(j, s)` the candidates in the cyclic window `win j s` of length `j+1` starting at `s` are
given the p-value `(j+1)q/m` and all others the p-value `1`; in outcome `none` every p-value is
`1`.  Outcome `(j, s)` has probability `q/((j+1)·m)`, so the outcomes with a window of length
`j+1` carry total mass `q/(j+1)` and the whole family carries `q·H_m`.

* `card_win`, `card_windowsThrough` — the window has `j+1` members, and each candidate lies in
  exactly `j+1` of the `m` windows of that length.  This is the balance that makes the p-values
  valid: each candidate sees the value `(j+1)q/m` with probability exactly `q/m`, for every `j`.
* `pval_superuniform` — **every** candidate's p-value is valid, `P(p_i ≤ t) ≤ t` for all `t ≥ 0`.
  So all `m` hypotheses are true nulls and the hypotheses of the Benjamini–Yekutieli theorem hold.
* `bhR_pv`, `bhRej_pv` — BH run at level `q` rejects *exactly* the window: the observed p-values
  are `j+1` copies of `(j+1)q/m`, which is precisely the step-up fixed point.
* `fdp_some`, `fdp_none` — every discovery is false, so the false discovery proportion is `1` on
  every outcome that produces a list at all.
* `bh_fdr_eq_harmonic` — **the theorem**: `E[FDP] = q·H_m` for uncorrected BH.
* `harmonic_factor_sharp` — hence the bound of `bh_uncorrected_bound` is an equality on this
  instance: the harmonic factor cannot be lowered.
* `uncorrected_bh_exceeds_level` — the practical consequence: for `m ≥ 2` uncorrected BH at level
  `q` has false discovery rate strictly above `q`, so under arbitrary dependence the deflation to
  `q/H_m` is not conservatism but necessity.  `by_level_is_attained` records the matching statement
  for the corrected procedure: at level `α/H_m` this same instance sits exactly at `α`, so the
  Benjamini–Yekutieli bound is itself attained and cannot be improved either.

Only `0 < q` and `q·H_m ≤ 1` are needed (the second only so that the construction is a probability
distribution — it is the statement that the total mass of the discovery outcomes does not exceed
one).
-/
import Mathlib
import RequestProject.DependentScreen
import RequestProject.FalseDiscovery
import RequestProject.DependentBH

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR
namespace DepBHSharp

open IDR.DepScreen IDR.DepBH

variable {m : ℕ}

/-! ## 1. Windows -/

/-- The cyclic window of length `j+1` starting at `s`. -/
def win (j s : Fin m) : Finset (Fin m) :=
  Finset.univ.filter (fun i => ((i - s : Fin m) : ℕ) ≤ (j : ℕ))

lemma card_win (j s : Fin m) : (win j s).card = (j : ℕ) + 1 := by
  haveI : NeZero m := ⟨by have := j.pos; omega⟩
  have h1 : (win j s).card = (Finset.univ.filter (fun r : Fin m => (r : ℕ) ≤ (j : ℕ))).card := by
    apply Finset.card_bij' (fun i _ => i - s) (fun r _ => r + s) <;> intros <;>
      simp_all [win, Finset.mem_filter, add_sub_cancel_right, sub_add_cancel]
  rw [h1]
  have h : (Finset.univ.filter (fun r : Fin m => (r : ℕ) ≤ (j : ℕ)))
      = (Finset.Iic j : Finset (Fin m)) := by
    ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iic, Fin.le_def]
  rw [h, Fin.card_Iic]

/-- Each candidate lies in exactly `j+1` of the `m` windows of length `j+1`: the construction is
balanced, which is what makes every p-value valid. -/
lemma card_windowsThrough (i j : Fin m) :
    (Finset.univ.filter (fun s : Fin m => i ∈ win j s)).card = (j : ℕ) + 1 := by
  haveI : NeZero m := ⟨by have := j.pos; omega⟩
  have h1 : (Finset.univ.filter (fun s : Fin m => i ∈ win j s)).card
      = (Finset.univ.filter (fun r : Fin m => (r : ℕ) ≤ (j : ℕ))).card := by
    apply Finset.card_bij' (fun s _ => i - s) (fun r _ => i - r) <;> intros <;>
      simp_all [win, Finset.mem_filter, sub_sub_cancel]
  rw [h1]
  have h : (Finset.univ.filter (fun r : Fin m => (r : ℕ) ≤ (j : ℕ)))
      = (Finset.Iic j : Finset (Fin m)) := by
    ext r
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iic, Fin.le_def]
  rw [h, Fin.card_Iic]

/-! ## 2. The p-values -/

/-- The observed p-vector in outcome `(j, s)`: the window gets `(j+1)q/m`, everyone else `1`. -/
noncomputable def pv (q : ℝ) (j s : Fin m) : Fin m → ℝ :=
  fun i => if i ∈ win j s then (((j : ℕ) : ℝ) + 1) * q / m else 1

/-- The p-value of candidate `i` as a function of the outcome. -/
noncomputable def pval (q : ℝ) (i : Fin m) : Option (Fin m × Fin m) → ℝ
  | none => 1
  | some (j, s) => pv q j s i

/-! ## 3. What BH does on this instance -/

lemma below_pv (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (j s : Fin m) {k : ℕ} (hk : k ≤ m) :
    IDR.FDR.below m q (pv q j s) k = if (j : ℕ) + 1 ≤ k then (j : ℕ) + 1 else 0 := by
  have hm : 0 < m := j.pos
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hkq : (k : ℝ) * q / m < 1 := by
    have hkm : (k : ℝ) ≤ m := by exact_mod_cast hk
    rw [div_lt_one hmR]
    nlinarith
  unfold IDR.FDR.below
  by_cases hjk : (j : ℕ) + 1 ≤ k
  · rw [if_pos hjk]
    have hset : (Finset.univ.filter (fun i => pv q j s i ≤ (k : ℝ) * q / m)) = win j s := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, pv]
      constructor
      · intro h
        by_contra hi
        rw [if_neg hi] at h
        linarith
      · intro hi
        rw [if_pos hi]
        have hle : ((j : ℕ) : ℝ) + 1 ≤ (k : ℝ) := by exact_mod_cast hjk
        gcongr
    rw [hset, card_win]
  · rw [if_neg hjk]
    have hset : (Finset.univ.filter (fun i => pv q j s i ≤ (k : ℝ) * q / m)) = ∅ := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, pv, Finset.notMem_empty, iff_false,
        not_le]
      by_cases hi : i ∈ win j s
      · rw [if_pos hi]
        have hlt : (k : ℝ) < ((j : ℕ) : ℝ) + 1 := by
          have hkj : k < (j : ℕ) + 1 := by omega
          exact_mod_cast hkj
        gcongr
      · rw [if_neg hi]; exact hkq
    rw [hset]; simp

/-- BH stops at the length of the window. -/
lemma bhR_pv (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (j s : Fin m) :
    IDR.FDR.bhR m q (pv q j s) = (j : ℕ) + 1 := by
  have hjm : (j : ℕ) + 1 ≤ m := j.2
  have hge : (j : ℕ) + 1 ≤ IDR.FDR.bhR m q (pv q j s) := by
    apply IDR.FDR.le_bhR m q _ hjm
    rw [below_pv q hq hq1 j s hjm, if_pos le_rfl]
  have hle := IDR.FDR.bhR_le_below m q (pv q j s)
  rw [below_pv q hq hq1 j s (IDR.FDR.bhR_le m q (pv q j s))] at hle
  split_ifs at hle
  omega

/-- BH rejects exactly the window — and every one of those rejections is false. -/
lemma bhRej_pv (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (j s : Fin m) :
    IDR.FDR.bhRej m q (pv q j s) = win j s := by
  have hm : 0 < m := j.pos
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  unfold IDR.FDR.bhRej
  rw [bhR_pv q hq hq1 j s]
  ext i
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, pv]
  have hjm : ((j : ℕ) : ℝ) + 1 ≤ m := by exact_mod_cast j.2
  have hlt1 : (((j : ℕ) : ℝ) + 1) * q / m < 1 := by
    rw [div_lt_one hmR]; nlinarith
  constructor
  · intro h
    by_contra hi
    rw [if_neg hi] at h
    push_cast at h
    linarith
  · intro hi
    rw [if_pos hi]
    push_cast
    exact le_rfl

/-- In the outcome with no small p-values BH reports nothing. -/
lemma bhRej_none (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (hm : 0 < m) :
    IDR.FDR.bhRej m q (fun _ : Fin m => (1 : ℝ)) = ∅ := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hR : IDR.FDR.bhR m q (fun _ : Fin m => (1 : ℝ)) = 0 := by
    by_contra h
    have hpos : 0 < IDR.FDR.bhR m q (fun _ : Fin m => (1 : ℝ)) := Nat.pos_of_ne_zero h
    have hle := IDR.FDR.bhR_le_below m q (fun _ : Fin m => (1 : ℝ))
    have hbelow : IDR.FDR.below m q (fun _ : Fin m => (1 : ℝ))
        (IDR.FDR.bhR m q (fun _ : Fin m => (1 : ℝ))) = 0 := by
      unfold IDR.FDR.below
      have hset : (Finset.univ.filter
          (fun _ : Fin m => (1 : ℝ) ≤ (IDR.FDR.bhR m q (fun _ : Fin m => (1 : ℝ)) : ℝ) * q / m))
          = ∅ := by
        ext i
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, iff_false,
          not_le]
        have hkm : ((IDR.FDR.bhR m q (fun _ : Fin m => (1 : ℝ))) : ℝ) ≤ (m : ℝ) := by
          exact_mod_cast IDR.FDR.bhR_le m q (fun _ : Fin m => (1 : ℝ))
        rw [div_lt_one hmR]
        nlinarith
      rw [hset]; simp
    omega
  unfold IDR.FDR.bhRej
  rw [hR]
  ext i
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.notMem_empty, iff_false, not_le]
  norm_num

/-! ## 4. The law -/

/-- The joint law: outcome `(j, s)` carries mass `q/((j+1)·m)`, and the remaining mass sits on the
outcome in which nothing is discovered. -/
noncomputable def probs (q : ℝ) {m : ℕ} : Option (Fin m × Fin m) → ℝ
  | none => 1 - q * harm m
  | some (j, _) => q / (((j : ℕ) : ℝ) + 1) / m

lemma sum_probs (q : ℝ) (hm : 0 < m) : ∑ ω : Option (Fin m × Fin m), probs q ω = 1 := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  rw [Fintype.sum_option]
  have h1 : ∑ x : Fin m × Fin m, probs q (some x) = q * harm m := by
    rw [Fintype.sum_prod_type]
    have hj : ∀ j : Fin m, ∑ _s : Fin m, probs q (some (j, _s)) = q / (((j : ℕ) : ℝ) + 1) := by
      intro j
      simp only [probs, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      field_simp
    rw [Finset.sum_congr rfl (fun j _ => hj j), harm, Finset.mul_sum,
      Fin.sum_univ_eq_sum_range (fun j => q / ((j : ℝ) + 1))]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  rw [h1]
  simp [probs]

/-- The construction as a probability distribution.  `q · H_m ≤ 1` is exactly the requirement that
the discovery outcomes do not carry more than the whole mass. -/
noncomputable def lawSharp (q : ℝ) (hq : 0 ≤ q) (hm : 0 < m) (hqh : q * harm m ≤ 1) :
    Law (Option (Fin m × Fin m)) where
  prob := probs q
  nonneg := by
    intro ω
    match ω with
    | none => simpa [probs] using hqh
    | some (j, s) =>
        have hmR : (0 : ℝ) < m := by exact_mod_cast hm
        have : (0 : ℝ) < ((j : ℕ) : ℝ) + 1 := by positivity
        simp only [probs]
        positivity
  total := sum_probs q hm

/-! ## 5. Every p-value is valid -/

lemma pval_superuniform (q : ℝ) (hq : 0 < q) (hm : 0 < m) (hqh : q * harm m ≤ 1) (i : Fin m) :
    Superuniform (lawSharp q (le_of_lt hq) hm hqh) (pval q i) := by
  intro t ht
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  by_cases ht1 : 1 ≤ t
  · refine le_trans ?_ ht1
    have hsub := (lawSharp q (le_of_lt hq) hm hqh).pr_mono
      (Finset.subset_univ (Finset.univ.filter (fun ω => pval q i ω ≤ t)))
    have huniv : (lawSharp q (le_of_lt hq) hm hqh).pr Finset.univ = 1 := by
      simpa [Law.pr] using (lawSharp q (le_of_lt hq) hm hqh).total
    rw [huniv] at hsub
    exact hsub
  · push_neg at ht1
    -- the event is a union of whole window families
    set J : Finset (Fin m) :=
      Finset.univ.filter (fun j : Fin m => (((j : ℕ) : ℝ) + 1) * q / m ≤ t) with hJ
    have hpr : (lawSharp q (le_of_lt hq) hm hqh).pr
        (Finset.univ.filter (fun ω => pval q i ω ≤ t)) = (J.card : ℝ) * (q / m) := by
      unfold Law.pr
      rw [Finset.sum_filter]
      rw [Fintype.sum_option]
      have hnone : (if pval q i none ≤ t then (lawSharp q (le_of_lt hq) hm hqh).prob none else 0)
          = 0 := by
        have : ¬ (pval q i none ≤ t) := by simp only [pval]; linarith
        rw [if_neg this]
      rw [hnone, zero_add, Fintype.sum_prod_type]
      have hj : ∀ j : Fin m,
          (∑ s : Fin m, if pval q i (some (j, s)) ≤ t then
            (lawSharp q (le_of_lt hq) hm hqh).prob (some (j, s)) else 0)
            = if j ∈ J then q / m else 0 := by
        intro j
        by_cases hjt : (((j : ℕ) : ℝ) + 1) * q / m ≤ t
        · have hmem : j ∈ J := by simp [hJ, hjt]
          rw [if_pos hmem]
          have hterm : ∀ s : Fin m,
              (if pval q i (some (j, s)) ≤ t then
                (lawSharp q (le_of_lt hq) hm hqh).prob (some (j, s)) else 0)
                = if i ∈ win j s then q / (((j : ℕ) : ℝ) + 1) / m else 0 := by
            intro s
            by_cases hi : i ∈ win j s
            · rw [if_pos hi, if_pos]
              · rfl
              · simp only [pval, pv, if_pos hi]; exact hjt
            · rw [if_neg hi, if_neg]
              simp only [pval, pv, if_neg hi]
              linarith
          rw [Finset.sum_congr rfl (fun s _ => hterm s), ← Finset.sum_filter,
            Finset.sum_const, card_windowsThrough i j, nsmul_eq_mul]
          push_cast
          field_simp
        · have hmem : j ∉ J := by simp [hJ, hjt]
          rw [if_neg hmem]
          apply Finset.sum_eq_zero
          intro s _
          by_cases hi : i ∈ win j s
          · rw [if_neg]
            simp only [pval, pv, if_pos hi]
            exact hjt
          · rw [if_neg]
            simp only [pval, pv, if_neg hi]
            linarith
      rw [Finset.sum_congr rfl (fun j _ => hj j), Finset.sum_ite_mem, Finset.univ_inter,
        Finset.sum_const, nsmul_eq_mul]
    rw [hpr]
    rcases Finset.eq_empty_or_nonempty J with hJe | hJne
    · rw [hJe]; simpa using ht
    · obtain ⟨jmax, hjmax, hmax⟩ := J.exists_max_image (fun j => (j : ℕ)) hJne
      have hsub : J ⊆ Finset.Iic jmax := by
        intro j hjJ
        simp only [Finset.mem_Iic, Fin.le_def]
        exact hmax j hjJ
      have hcard : (J.card : ℝ) ≤ ((jmax : ℕ) : ℝ) + 1 := by
        have h1 : J.card ≤ (jmax : ℕ) + 1 := by
          calc J.card ≤ (Finset.Iic jmax).card := Finset.card_le_card hsub
            _ = (jmax : ℕ) + 1 := Fin.card_Iic jmax
        exact_mod_cast h1
      have hjt : (((jmax : ℕ) : ℝ) + 1) * q / m ≤ t := by
        have := hjmax
        simp only [hJ, Finset.mem_filter] at this
        exact this.2
      calc (J.card : ℝ) * (q / m) ≤ (((jmax : ℕ) : ℝ) + 1) * (q / m) := by
            apply mul_le_mul_of_nonneg_right hcard (by positivity)
        _ = (((jmax : ℕ) : ℝ) + 1) * q / m := by ring
        _ ≤ t := hjt

/-! ## 6. Every discovery is false -/

lemma fdp_some (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (j s : Fin m) :
    fdp (Finset.univ : Finset (Fin m)) (bhList q m (fun i ω => pval q i ω)) (some (j, s)) = 1 := by
  have hR : bhList q m (fun i ω => pval q i ω) (some (j, s)) = win j s := by
    unfold bhList
    rw [show (fun i => pval q i (some (j, s))) = pv q j s from rfl]
    exact bhRej_pv q hq hq1 j s
  unfold fdp
  rw [hR, Finset.inter_univ, card_win]
  norm_num
  positivity

lemma fdp_none (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (hm : 0 < m) :
    fdp (Finset.univ : Finset (Fin m)) (bhList q m (fun i ω => pval q i ω)) none = 0 := by
  have hR : bhList q m (fun i ω => pval q i ω) none = ∅ := by
    unfold bhList
    rw [show (fun i : Fin m => pval q i none) = (fun _ : Fin m => (1 : ℝ)) from rfl]
    exact bhRej_none q hq hq1 hm
  unfold fdp
  rw [hR]
  simp

/-! ## 7. The theorem -/

/-- **The harmonic factor is attained.**  On this instance the false discovery rate of
*uncorrected* Benjamini–Hochberg at level `q` is exactly `q·H_m`. -/
theorem bh_fdr_eq_harmonic (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (hm : 0 < m)
    (hqh : q * harm m ≤ 1) :
    (lawSharp q (le_of_lt hq) hm hqh).mean
        (fdp (Finset.univ : Finset (Fin m)) (bhList q m (fun i ω => pval q i ω)))
      = q * harm m := by
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  unfold Law.mean
  rw [Fintype.sum_option]
  rw [fdp_none q hq hq1 hm]
  rw [Fintype.sum_prod_type]
  have hj : ∀ j : Fin m,
      ∑ s : Fin m, (lawSharp q (le_of_lt hq) hm hqh).prob (some (j, s)) *
        fdp (Finset.univ : Finset (Fin m)) (bhList q m (fun i ω => pval q i ω)) (some (j, s))
        = q / (((j : ℕ) : ℝ) + 1) := by
    intro j
    have hterm : ∀ s : Fin m, (lawSharp q (le_of_lt hq) hm hqh).prob (some (j, s)) *
        fdp (Finset.univ : Finset (Fin m)) (bhList q m (fun i ω => pval q i ω)) (some (j, s))
        = q / (((j : ℕ) : ℝ) + 1) / m := by
      intro s
      rw [fdp_some q hq hq1 j s, mul_one]
      rfl
    rw [Finset.sum_congr rfl (fun s _ => hterm s), Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul]
    field_simp
  rw [Finset.sum_congr rfl (fun j _ => hj j)]
  rw [harm, Finset.mul_sum, Fin.sum_univ_eq_sum_range (fun j => q / ((j : ℝ) + 1))]
  rw [Finset.sum_congr rfl (fun j (_ : j ∈ Finset.range m) => (by ring :
    q * ((1 : ℝ) / ((j : ℝ) + 1)) = q / ((j : ℝ) + 1)))]
  simp

/-- **Sharpness.**  The bound `bh_uncorrected_bound` is an equality on this instance, so the
harmonic factor in it cannot be lowered. -/
theorem harmonic_factor_sharp (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (hm : 0 < m)
    (hqh : q * harm m ≤ 1) :
    ∃ (P : Law (Option (Fin m × Fin m))) (p : Fin m → Option (Fin m × Fin m) → ℝ),
      (∀ i, Superuniform P (p i)) ∧
      P.mean (fdp (Finset.univ : Finset (Fin m)) (bhList q m p)) = q * harm m := by
  exact ⟨lawSharp q (le_of_lt hq) hm hqh, fun i ω => pval q i ω,
    fun i => pval_superuniform q hq hm hqh i, bh_fdr_eq_harmonic q hq hq1 hm hqh⟩

/-- **Why the correction is not optional.**  From two candidates on, uncorrected BH at level `q`
overshoots its nominal level under dependence. -/
theorem uncorrected_bh_exceeds_level (q : ℝ) (hq : 0 < q) (hq1 : q < 1) (hm : 2 ≤ m)
    (hqh : q * harm m ≤ 1) :
    q < (lawSharp q (le_of_lt hq) (by omega) hqh).mean
        (fdp (Finset.univ : Finset (Fin m)) (bhList q m (fun i ω => pval q i ω))) := by
  rw [bh_fdr_eq_harmonic q hq hq1 (by omega) hqh]
  have hsub : Finset.range 2 ⊆ Finset.range m := by
    intro x hx; simp only [Finset.mem_range] at *; omega
  have h2 : harm 2 ≤ harm m :=
    Finset.sum_le_sum_of_subset_of_nonneg hsub (fun j _ _ => by positivity)
  have h2' : (3 : ℝ) / 2 ≤ harm m := by
    have : harm 2 = 3 / 2 := by norm_num [harm, Finset.sum_range_succ]
    linarith
  nlinarith

/-- At the corrected level the construction is still a probability distribution. -/
lemma mass_le_one {α : ℝ} {m : ℕ} (hm : 0 < m) (hα1 : α ≤ 1) : α / harm m * harm m ≤ 1 := by
  rw [div_mul_cancel₀ _ (ne_of_gt (harm_pos hm))]
  exact hα1

/-- The corrected procedure is itself exactly at its bound on the same instance: run at level
`α/H_m` the false discovery rate is exactly `α`.  So `benjamini_yekutieli_level` cannot be
improved either. -/
theorem by_level_is_attained (α : ℝ) (hα : 0 < α) (hm : 0 < m) (hα1 : α / harm m < 1)
    (hqh : α ≤ 1) :
    (lawSharp (α / harm m) (le_of_lt (div_pos hα (harm_pos hm))) hm
        (mass_le_one hm hqh)).mean
        (fdp (Finset.univ : Finset (Fin m))
          (bhList (α / harm m) m (fun i ω => pval (α / harm m) i ω)))
      = α := by
  have hH : 0 < harm m := harm_pos hm
  rw [bh_fdr_eq_harmonic (α / harm m) (div_pos hα hH) hα1 hm (mass_le_one hm hqh)]
  field_simp

/-- The hypotheses of the sharp instance are satisfiable for every screen of two or more
candidates: at `q = 1/(2 H_m)` the construction is a probability distribution, the level is below
one, and uncorrected BH overshoots. -/
lemma hypotheses_satisfiable (hm : 2 ≤ m) :
    0 < 1 / (2 * harm m) ∧ 1 / (2 * harm m) < 1 ∧ (1 / (2 * harm m)) * harm m ≤ 1 := by
  have hm0 : 0 < m := by omega
  have hH : 0 < harm m := harm_pos hm0
  have h1 : 1 ≤ harm m := one_le_harm hm0
  refine ⟨by positivity, ?_, ?_⟩
  · rw [div_lt_one (by positivity)]
    linarith
  · rw [div_mul_eq_mul_div, one_mul, div_le_one (by positivity)]
    linarith

end DepBHSharp
end IDR
