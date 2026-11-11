/-
# Part XCI.2  A p-value with no distributional assumption, and what the screen costs

Benjamini–Hochberg needs superuniform null p-values.  A counting reporter supplies an exact mean
and an exact variance (`RequestProject.NoisyDetection`) and, without further modelling, nothing
else: no exact binomial tail, because the per-molecule firing probabilities of a real instrument
are not identical, and no normal approximation, because the sample sizes at which a screen would
be run are exactly the sizes at which that approximation is unjustified.

This file builds the p-value that uses only what is known.

* `chebP` — the Chebyshev p-value `n·q₀·(1-q₀)/(count - n·q₀)²`, truncated at one.
* `chebP_superuniform` — **it is a valid p-value**: `P(p ≤ t) ≤ t` at every level, from two
  moments alone.  The degenerate zero-variance case is handled, not excluded.
* `chebP_power`, `screenSamples`, `screenSamples_spec` — and it has power: a candidate whose
  reporter contrast is `Δ` reaches level `u` with probability `1 - α` after
  `⌈1/(min(u,α)·Δ²)⌉` molecules.
* `screen_fdr_control` — the screen: `N` candidates, each read by its own reporter against its own
  calibrated baseline, with BH applied to the resulting p-values, has expected false discovery
  fraction at most `q`.
* `screen_budget_quadratic` — **the price.**  Because a candidate must reach `q/N` in the worst
  case and Chebyshev buys tail probability only quadratically, the screen costs at least
  `N²/(q·Δ²)` molecules in total: quadratic in the number of candidates screened.

`RequestProject.ChernoffScreen` shows what changes when the analysis is willing to prove an
exponential tail bound instead.
-/
import Mathlib
import RequestProject.NoisyDetection
import RequestProject.FalseDiscovery

set_option autoImplicit false
set_option maxHeartbeats 1000000

open Finset
open scoped Classical

namespace IDR
namespace Screen

open IDR.Noisy IDR.FDR

/-! ## 1. The Chebyshev p-value -/

/-- The one-sided p-value a counting experiment can report using *nothing but* the exact mean
and variance of its read-out law: `n·q₀·(1-q₀)/(count - n·q₀)²`, truncated at one, and one when
the count is at or below the baseline mean.  No normal approximation, no exact binomial tail. -/
noncomputable def chebP (n : ℕ) (q₀ : ℝ) (s : Fin n → Bool) : ℝ :=
  if cnt s ≤ n * q₀ then 1 else min 1 ((n : ℝ) * q₀ * (1 - q₀) / (cnt s - n * q₀) ^ 2)

lemma chebP_le_one (n : ℕ) (q₀ : ℝ) (s : Fin n → Bool) : chebP n q₀ s ≤ 1 := by
  unfold chebP
  split
  · exact le_rfl
  · exact min_le_left _ _

lemma chebP_nonneg {q₀ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (n : ℕ) (s : Fin n → Bool) :
    0 ≤ chebP n q₀ s := by
  unfold chebP
  split
  · norm_num
  · refine le_min (by norm_num) ?_
    have : (0:ℝ) ≤ (n : ℝ) * q₀ * (1 - q₀) := by
      have : (0:ℝ) ≤ 1 - q₀ := by linarith
      positivity
    positivity

/-- A Chebyshev bound in the form the p-value needs: the mass of the set on which the squared
deviation from the mean is at least `c`. -/
lemma tail_sq_bound {q₀ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (n : ℕ) {c : ℝ} (hc : 0 < c) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ (cnt s - n * q₀) ^ 2), recProb q₀ s
      ≤ (n : ℝ) * q₀ * (1 - q₀) / c := by
  have hkey : ∀ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ (cnt s - n * q₀) ^ 2),
      recProb q₀ s ≤ recProb q₀ s * (cnt s - n * q₀) ^ 2 / c := by
    intro s hs
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hs
    have hp := recProb_nonneg h0 h1 s
    rw [le_div_iff₀ hc]
    nlinarith
  calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ (cnt s - n * q₀) ^ 2), recProb q₀ s
      ≤ ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => c ≤ (cnt s - n * q₀) ^ 2),
          recProb q₀ s * (cnt s - n * q₀) ^ 2 / c := Finset.sum_le_sum hkey
    _ ≤ ∑ s : Fin n → Bool, recProb q₀ s * (cnt s - n * q₀) ^ 2 / c := by
        refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) ?_
        intro s _ _
        have := recProb_nonneg h0 h1 s
        positivity
    _ = (n : ℝ) * q₀ * (1 - q₀) / c := by rw [← Finset.sum_div, sum_recProb_var]

/-- The p-value is superuniform at every level strictly between zero and one. -/
lemma chebP_superuniform_pos {q₀ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (n : ℕ) {t : ℝ}
    (ht : 0 < t) (ht1 : t < 1) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t), recProb q₀ s ≤ t := by
  set v : ℝ := (n : ℝ) * q₀ * (1 - q₀) with hv
  have hvnn : 0 ≤ v := by
    have h1' : (0:ℝ) ≤ 1 - q₀ := by linarith
    rw [hv]; positivity
  -- on the rejection set the count strictly exceeds the mean and `v ≤ t·(deviation)²`
  have hmem : ∀ s : Fin n → Bool, chebP n q₀ s ≤ t →
      (n : ℝ) * q₀ < cnt s ∧ v ≤ t * (cnt s - n * q₀) ^ 2 := by
    intro s hs
    unfold chebP at hs
    by_cases hc : cnt s ≤ (n : ℝ) * q₀
    · rw [if_pos hc] at hs; linarith
    · rw [if_neg hc] at hs
      push_neg at hc
      refine ⟨hc, ?_⟩
      have hmin : v / (cnt s - n * q₀) ^ 2 ≤ t := by
        rcases min_cases (1 : ℝ) (v / (cnt s - n * q₀) ^ 2) with ⟨he, _⟩ | ⟨he, hle⟩
        · rw [he] at hs; linarith
        · rw [he] at hs; exact hs
      have hd : (0:ℝ) < (cnt s - n * q₀) ^ 2 := by
        have : (0:ℝ) < cnt s - n * q₀ := by linarith
        positivity
      rw [div_le_iff₀ hd] at hmin
      linarith [hmin]
  rcases eq_or_lt_of_le hvnn with hv0 | hvpos
  · -- degenerate case: zero variance, so the rejection set carries no mass at all
    have hzero : ∀ s : Fin n → Bool, recProb q₀ s * (cnt s - n * q₀) ^ 2 = 0 := by
      have hsum : ∑ s : Fin n → Bool, recProb q₀ s * (cnt s - n * q₀) ^ 2 = 0 := by
        rw [sum_recProb_var]; rw [hv] at hv0; linarith
      intro s
      by_contra hne
      have hnn : ∀ s : Fin n → Bool, 0 ≤ recProb q₀ s * (cnt s - n * q₀) ^ 2 := by
        intro s; have := recProb_nonneg h0 h1 s; positivity
      have := (Finset.sum_eq_zero_iff_of_nonneg (fun s _ => hnn s)).mp hsum s (Finset.mem_univ s)
      exact hne this
    have : ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t),
        recProb q₀ s = 0 := by
      refine Finset.sum_eq_zero ?_
      intro s hsm
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm
      have hgt := (hmem s hsm).1
      have hd : (cnt s - (n : ℝ) * q₀) ^ 2 ≠ 0 := by
        have : (0:ℝ) < cnt s - n * q₀ := by linarith
        positivity
      have := hzero s
      rcases mul_eq_zero.mp this with h | h
      · exact h
      · exact absurd h hd
    rw [this]; exact ht.le
  · have hc : 0 < v / t := div_pos hvpos ht
    have hsub : Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t)
        ⊆ Finset.univ.filter (fun s : Fin n → Bool => v / t ≤ (cnt s - n * q₀) ^ 2) := by
      intro s hsm
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm ⊢
      rw [div_le_iff₀ ht]
      have := (hmem s hsm).2
      linarith
    calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t), recProb q₀ s
        ≤ ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => v / t ≤ (cnt s - n * q₀) ^ 2),
            recProb q₀ s := by
          refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
          intro s _ _
          exact recProb_nonneg h0 h1 s
      _ ≤ v / (v / t) := tail_sq_bound h0 h1 n hc
      _ = t := by field_simp

/-- **A valid p-value from two moments.**  Under the baseline read-out law the Chebyshev p-value
is superuniform at every level: `P(p ≤ t) ≤ t`.  Nothing about the shape of the count
distribution is used — only its exact mean and variance. -/
theorem chebP_superuniform {q₀ : ℝ} (h0 : 0 ≤ q₀) (h1 : q₀ ≤ 1) (n : ℕ) {t : ℝ} (ht : 0 ≤ t) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t), recProb q₀ s ≤ t := by
  rcases le_or_gt 1 t with hge | hlt
  · calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t), recProb q₀ s
        ≤ ∑ s : Fin n → Bool, recProb q₀ s := by
          refine Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _) ?_
          intro s _ _
          exact recProb_nonneg h0 h1 s
      _ = 1 := sum_recProb q₀ n
      _ ≤ t := hge
  · rcases eq_or_lt_of_le ht with ht0 | htpos
    · -- level zero: squeeze from all positive levels
      subst_vars
      by_contra hcon
      push_neg at hcon
      set M : ℝ := ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ 0),
        recProb q₀ s with hM
      set t' : ℝ := min (M / 2) (1 / 2) with ht'
      have ht'pos : 0 < t' := lt_min (by linarith) (by norm_num)
      have ht'lt : t' < 1 := lt_of_le_of_lt (min_le_right _ _) (by norm_num)
      have hsub : Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ 0)
          ⊆ Finset.univ.filter (fun s : Fin n → Bool => chebP n q₀ s ≤ t') := by
        intro s hsm
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm ⊢
        linarith
      have hle : M ≤ t' := by
        refine le_trans (Finset.sum_le_sum_of_subset_of_nonneg hsub ?_) ?_
        · intro s _ _
          exact recProb_nonneg h0 h1 s
        · exact chebP_superuniform_pos h0 h1 n ht'pos ht'lt
      have : t' ≤ M / 2 := min_le_left _ _
      linarith
    · exact chebP_superuniform_pos h0 h1 n htpos hlt

/-! ## 2. Power: how many molecules a candidate needs to reach a given p-value -/

/-- The molecules per candidate: `⌈1/(min(u,α)·Δ²)⌉`, where `u` is the p-value the candidate must
reach and `Δ` the contrast its reporter delivers. -/
noncomputable def screenSamples (u alpha delta : ℝ) : ℕ := ⌈1 / (min u alpha * delta ^ 2)⌉₊

/-- **The Chebyshev p-value has power.**  A candidate whose true read rate exceeds the baseline
by `Δ = q₁ - q₀` returns a p-value above `u` with probability at most `α`, as soon as the
molecule count satisfies `u·n·Δ² ≥ 1` and `α·n·Δ² ≥ 1`. -/
theorem chebP_power {q₀ q₁ u alpha : ℝ} (h10 : 0 ≤ q₁) (h11 : q₁ ≤ 1) (hlt : q₀ < q₁) (hu : 0 < u) {n : ℕ} (hn : 0 < n)
    (hnu : 1 ≤ u * n * (q₁ - q₀) ^ 2) (hna : 1 ≤ alpha * n * (q₁ - q₀) ^ 2) :
    ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => ¬ (chebP n q₀ s ≤ u)), recProb q₁ s
      ≤ alpha := by
  set d : ℝ := q₁ - q₀ with hd
  have hdpos : 0 < d := by rw [hd]; linarith
  have hnpos : (0:ℝ) < (n : ℝ) := by exact_mod_cast hn
  set r : ℝ := (n : ℝ) * d / 2 with hr
  have hrpos : 0 < r := by rw [hr]; positivity
  -- a molecule record close to its own mean returns a small p-value
  have hclose : ∀ s : Fin n → Bool, |cnt s - (n : ℝ) * q₁| < r → chebP n q₀ s ≤ u := by
    intro s hs
    have habs := abs_lt.mp hs
    have hgap : cnt s - (n : ℝ) * q₀ > r := by
      have : cnt s - (n : ℝ) * q₀ = (cnt s - (n : ℝ) * q₁) + (n : ℝ) * d := by
        rw [hd]; ring
      rw [this, hr]
      have := habs.1
      rw [hr] at this
      linarith
    have hgt : ¬ (cnt s ≤ (n : ℝ) * q₀) := by
      have : (0:ℝ) < r := hrpos
      intro hcon; linarith
    unfold chebP
    rw [if_neg hgt]
    refine le_trans (min_le_right _ _) ?_
    have hsq : r ^ 2 ≤ (cnt s - (n : ℝ) * q₀) ^ 2 := by nlinarith
    have hq4 : q₀ * (1 - q₀) ≤ 1 / 4 := by nlinarith [sq_nonneg (q₀ - 1 / 2)]
    have hv : (n : ℝ) * q₀ * (1 - q₀) ≤ (n : ℝ) / 4 := by nlinarith [hnpos.le]
    have hur : (n : ℝ) / 4 ≤ u * r ^ 2 := by
      rw [hr]
      have : u * ((n : ℝ) * d / 2) ^ 2 = (u * n * d ^ 2) * (n : ℝ) / 4 := by ring
      rw [this]
      nlinarith
    have hdgt : (0:ℝ) < cnt s - (n : ℝ) * q₀ := by linarith
    have hd2 : (0:ℝ) < (cnt s - (n : ℝ) * q₀) ^ 2 := by positivity
    rw [div_le_iff₀ hd2]
    nlinarith
  have hsub : Finset.univ.filter (fun s : Fin n → Bool => ¬ (chebP n q₀ s ≤ u))
      ⊆ devSet n ((n : ℝ) * q₁) r := by
    intro s hsm
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hsm
    rw [mem_devSet]
    by_contra hcon
    push_neg at hcon
    exact hsm (hclose s hcon)
  calc ∑ s ∈ Finset.univ.filter (fun s : Fin n → Bool => ¬ (chebP n q₀ s ≤ u)), recProb q₁ s
      ≤ ∑ s ∈ devSet n ((n : ℝ) * q₁) r, recProb q₁ s := by
        refine Finset.sum_le_sum_of_subset_of_nonneg hsub ?_
        intro s _ _
        exact recProb_nonneg h10 h11 s
    _ ≤ (n : ℝ) * q₁ * (1 - q₁) / r ^ 2 := chebyshev h10 h11 n hrpos
    _ ≤ alpha := by
        have hq4 : q₁ * (1 - q₁) ≤ 1 / 4 := by nlinarith [sq_nonneg (q₁ - 1 / 2)]
        have hv : (n : ℝ) * q₁ * (1 - q₁) ≤ (n : ℝ) / 4 := by nlinarith [hnpos.le]
        have hr2 : r ^ 2 = (n : ℝ) ^ 2 * d ^ 2 / 4 := by rw [hr]; ring
        rw [div_le_iff₀ (by positivity)]
        rw [hr2]
        nlinarith

/-- The molecule count `screenSamples` delivers both hypotheses of `chebP_power`. -/
lemma screenSamples_spec {u alpha delta : ℝ} (hu : 0 < u) (ha : 0 < alpha) (hdelta : 0 < delta)
    {n : ℕ} (hn : screenSamples u alpha delta ≤ n) :
    1 ≤ u * n * delta ^ 2 ∧ 1 ≤ alpha * n * delta ^ 2 := by
  have hmin : 0 < min u alpha := lt_min hu ha
  have hceil : (1 : ℝ) / (min u alpha * delta ^ 2) ≤ (screenSamples u alpha delta : ℝ) :=
    Nat.le_ceil _
  have hnle : (screenSamples u alpha delta : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hkey : 1 ≤ min u alpha * n * delta ^ 2 := by
    have h1 : (1 : ℝ) / (min u alpha * delta ^ 2) ≤ (n : ℝ) := le_trans hceil hnle
    rw [div_le_iff₀ (by positivity)] at h1
    nlinarith
  constructor
  · nlinarith [min_le_left u alpha, Nat.cast_nonneg (α := ℝ) n, sq_nonneg delta,
      mul_nonneg (Nat.cast_nonneg (α := ℝ) n) (sq_nonneg delta)]
  · nlinarith [min_le_right u alpha, Nat.cast_nonneg (α := ℝ) n, sq_nonneg delta,
      mul_nonneg (Nat.cast_nonneg (α := ℝ) n) (sq_nonneg delta)]

/-! ## 3. The screen: many candidate regions at once -/

/-- **The proteome-scale screen has a controlled false discovery rate.**

`N` candidate regions are screened independently; region `i` is read `n i` times by its own
reporter, whose true positive rate is `rate i`, and reports the Chebyshev p-value against its own
calibrated baseline `q0 i`.  For a null region — one whose reporter fires at the disorder-free
baseline rate — the two rates agree.  Then the expected fraction of the Benjamini–Hochberg
discovery list at level `q` that consists of null regions is at most `q`, no matter how many
regions there are, how they are distributed, or what the non-null regions do. -/
theorem screen_fdr_control {N : ℕ} (n : Fin N → ℕ) (q0 rate : Fin N → ℝ)
    (h0 : ∀ i, 0 ≤ q0 i) (h1 : ∀ i, q0 i ≤ 1)
    (hr0 : ∀ i, 0 ≤ rate i) (hr1 : ∀ i, rate i ≤ 1)
    {q : ℝ} (hq : 0 ≤ q) (H₀ : Finset (Fin N)) (hnull : ∀ i ∈ H₀, rate i = q0 i) :
    EE (V := fun i => (Fin (n i) → Bool)) (fun i s => recProb (rate i) s)
        (fun ω => fdp N q (pvec (fun i s => chebP (n i) (q0 i) s) ω) H₀) ≤ q := by
  refine bh_fdr_le (V := fun i => (Fin (n i) → Bool)) hq
    (fun i s => recProb_nonneg (hr0 i) (hr1 i) s)
    (fun i => sum_recProb (rate i) (n i))
    (fun i s => chebP_nonneg (h0 i) (h1 i) (n i) s) H₀ ?_
  intro i hi t ht
  have := chebP_superuniform (h0 i) (h1 i) (n i) (t := t) ht
  rw [hnull i hi]
  exact this

/-! ## 4. What the screen costs -/

/-- **The screen is quadratic in the number of candidates.**  To be discoverable a candidate must
reach the smallest BH threshold `q/N` in the worst case; sizing every candidate for that level
costs at least `N²/(q·Δ²)` molecules in total.  A ten-thousand-region screen at `q = 0.05` with a
reporter contrast of `Δ = 0.1` is therefore a `2·10¹⁰`-molecule experiment — the multiplicity, not
the physics, is what makes a proteome-wide claim expensive. -/
theorem screen_budget_quadratic {N : ℕ} (hN : 0 < N) {q alpha delta : ℝ} (hq : 0 < q)
    (hdelta : 0 < delta) (hle : q / N ≤ alpha) :
    (N : ℝ) ^ 2 / (q * delta ^ 2)
      ≤ (N : ℝ) * (screenSamples (q / N) alpha delta : ℝ) := by
  have hNpos : (0:ℝ) < (N : ℝ) := by exact_mod_cast hN
  have hu : 0 < q / N := div_pos hq hNpos
  have hmin : min (q / N) alpha = q / N := min_eq_left hle
  have hceil : (1 : ℝ) / (min (q / N) alpha * delta ^ 2)
      ≤ (screenSamples (q / N) alpha delta : ℝ) := Nat.le_ceil _
  rw [hmin] at hceil
  have hkey : (N : ℝ) / (q * delta ^ 2) ≤ (screenSamples (q / N) alpha delta : ℝ) := by
    refine le_trans (le_of_eq ?_) hceil
    field_simp
  calc (N : ℝ) ^ 2 / (q * delta ^ 2) = (N : ℝ) * ((N : ℝ) / (q * delta ^ 2)) := by
        field_simp
    _ ≤ (N : ℝ) * (screenSamples (q / N) alpha delta : ℝ) := by
        exact mul_le_mul_of_nonneg_left hkey hNpos.le

end Screen
end IDR
