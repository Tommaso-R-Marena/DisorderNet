/-
# Conformal risk control, and what a *grouped* guarantee does not say

Two questions about the calibration layer of an IDR predictor are answered here.

**1.  Does the conformal risk control (CRC) recipe actually control the risk?**  The recipe: a
monotone family of prediction rules indexed by a threshold `t` running along a finite grid, a
loss `L i t` for calibration item `i` that is *non-increasing* in `t` and bounded above by `B`,
and the calibrated threshold

  `t̂_j = min { t : (Σ_{i ≠ j} L i t) + B ≤ α·(n+1) }`   (the whole grid's last point if no such `t`)

computed from the `n` calibration items other than the held-out item `j`.  `crc_validity` proves

  `Σ_j L j (t̂_j) ≤ α·(n+1)`,

i.e. the *average over which item is held out* of the realised loss is at most `α`.  This is the
exchangeable statement in its finite, deterministic form: for exchangeable random losses the
theorem follows by averaging this inequality, since the average over the held-out index is exactly
what exchangeability buys.  The only hypotheses are monotonicity, the bound `B`, and that the last
grid point is safe (`L i t_max ≤ α`) — no distributional assumption whatsoever.

**2.  Does a group-level guarantee give an element-level one?**  No, and the failure is
quantitative.  With `cov k` the covered fraction of group `k` and `sizes k` its size,

* `macroCov` (the group-level number a grouped report quotes) is the unweighted mean of `cov`;
* `microCov` (what a user experiences per residue) is the size-weighted mean.

`macroCov_eq_microCov_of_equal_sizes` — they agree exactly when the groups are equally sized.
`microCov_ge_of_macroCov` — in general
`1 - microCov ≤ (n_max·K / n_total)·(1 - macroCov)`: the element-level miss rate is the group-level
miss rate inflated by the size imbalance.  A panel whose largest protein is `r` times the average
length can therefore report `1 - α` at the group level and deliver `1 - r·α` per residue.
`grouped_validity_underdetermines_element_coverage` is the sharp negative companion: two grouped
panels with *identical* group-level coverage, both valid at level `α`, one of which covers a
`1 - α` fraction of residues and the other an arbitrarily small fraction.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.ConformalRisk

open Finset

/-! ## Conformal risk control on a threshold grid -/

section CRC

variable {n m : ℕ}

open Classical in
/-- The grid points at which the *inflated* leave-one-out empirical risk is already below the
target: `(Σ_{i ≠ j} L i t) + B ≤ α·(n+1)`. -/
noncomputable def crcSet (α B : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ) (j : Fin (n + 1)) :
    Finset (Fin (m + 1)) :=
  univ.filter (fun t => (∑ i ∈ univ.erase j, L i t) + B ≤ α * (n + 1))

open Classical in
/-- The calibrated threshold when item `j` is held out: the first safe grid point, or the last
grid point if none is safe. -/
noncomputable def calib (α B : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ) (j : Fin (n + 1)) :
    Fin (m + 1) :=
  if h : (crcSet α B L j).Nonempty then (crcSet α B L j).min' h else Fin.last m

open Classical in
/-- The grid points at which the risk over *all* `n+1` items is below the target. -/
noncomputable def crcSetFull (α : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ) :
    Finset (Fin (m + 1)) :=
  univ.filter (fun t => (∑ i, L i t) ≤ α * (n + 1))

open Classical in
/-- The (unrealisable) threshold calibrated on all `n+1` items, including the held-out one. -/
noncomputable def calibFull (α : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ) : Fin (m + 1) :=
  if h : (crcSetFull α L).Nonempty then (crcSetFull α L).min' h else Fin.last m

theorem mem_crcSet {α B : ℝ} {L : Fin (n + 1) → Fin (m + 1) → ℝ} {j : Fin (n + 1)}
    {t : Fin (m + 1)} :
    t ∈ crcSet α B L j ↔ (∑ i ∈ univ.erase j, L i t) + B ≤ α * (n + 1) := by
  classical
  simp [crcSet]

theorem mem_crcSetFull {α : ℝ} {L : Fin (n + 1) → Fin (m + 1) → ℝ} {t : Fin (m + 1)} :
    t ∈ crcSetFull α L ↔ (∑ i, L i t) ≤ α * (n + 1) := by
  classical
  simp [crcSetFull]

/-- The oracle threshold is never later than the one the recipe returns: inflating the risk by the
bound `B` in place of the missing item can only delay the stopping point. -/
theorem calibFull_le_calib (α B : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ)
    (hB : ∀ i t, L i t ≤ B) (j : Fin (n + 1)) :
    calibFull α L ≤ calib α B L j := by
  classical
  unfold calib
  split_ifs with h
  · set t := (crcSet α B L j).min' h with ht
    have hmem : (∑ i ∈ univ.erase j, L i t) + B ≤ α * (n + 1) :=
      mem_crcSet.mp (Finset.min'_mem _ _)
    have hfull : t ∈ crcSetFull α L := by
      rw [mem_crcSetFull]
      have hsum : (∑ i, L i t) = L j t + ∑ i ∈ univ.erase j, L i t :=
        (Finset.add_sum_erase _ _ (Finset.mem_univ j)).symm
      have hBj := hB j t
      rw [hsum]
      linarith
    unfold calibFull
    rw [dif_pos ⟨_, hfull⟩]
    exact Finset.min'_le _ _ hfull
  · unfold calibFull
    split_ifs with h'
    · exact Fin.le_last _
    · exact le_rfl

/-- The risk at the oracle threshold is below the target. -/
theorem sum_le_at_calibFull (α : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ)
    (hlast : ∀ i, L i (Fin.last m) ≤ α) :
    (∑ i, L i (calibFull α L)) ≤ α * (n + 1) := by
  classical
  unfold calibFull
  split_ifs with h
  · exact mem_crcSetFull.mp (Finset.min'_mem _ _)
  · calc (∑ i, L i (Fin.last m)) ≤ ∑ _i : Fin (n + 1), α := Finset.sum_le_sum fun i _ => hlast i
      _ = α * (n + 1) := by simp [mul_comm]

/-- **Conformal risk control is valid.**  For monotone losses bounded by `B` with a safe last grid
point, the average over the held-out item of the realised loss at the calibrated threshold is at
most the target `α`.  No assumption on the data-generating distribution is used; for exchangeable
random losses the usual statement `𝔼[L_{n+1}(t̂)] ≤ α` follows by averaging this inequality over
the exchangeable permutations. -/
theorem crc_validity (α B : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ)
    (hanti : ∀ i, Antitone (L i)) (hB : ∀ i t, L i t ≤ B)
    (hlast : ∀ i, L i (Fin.last m) ≤ α) :
    (∑ j, L j (calib α B L j)) ≤ α * (n + 1) := by
  classical
  calc (∑ j, L j (calib α B L j))
      ≤ ∑ j, L j (calibFull α L) :=
        Finset.sum_le_sum fun j _ => hanti j (calibFull_le_calib α B L hB j)
    _ ≤ α * (n + 1) := sum_le_at_calibFull α L hlast

/-- The same statement in mean form: the average realised loss is at most `α`. -/
theorem crc_validity_mean (α B : ℝ) (L : Fin (n + 1) → Fin (m + 1) → ℝ)
    (hanti : ∀ i, Antitone (L i)) (hB : ∀ i t, L i t ≤ B)
    (hlast : ∀ i, L i (Fin.last m) ≤ α) :
    (∑ j, L j (calib α B L j)) / (n + 1) ≤ α := by
  have hpos : (0:ℝ) < (n + 1) := by positivity
  rw [div_le_iff₀ hpos]
  exact crc_validity α B L hanti hB hlast

end CRC

/-! ## Group-level validity versus element-level coverage -/

section Grouping

variable {K : ℕ}

/-- The group-level (macro) coverage a grouped report quotes: the unweighted mean over groups of
the covered fraction. -/
noncomputable def macroCov (cov : Fin K → ℝ) : ℝ := (∑ k, cov k) / K

/-- The element-level (micro) coverage a user experiences: the size-weighted mean. -/
noncomputable def microCov (sizes : Fin K → ℕ) (cov : Fin K → ℝ) : ℝ :=
  (∑ k, (sizes k : ℝ) * cov k) / (∑ k, (sizes k : ℝ))

/-- With equally sized groups the two coverages agree. -/
theorem macroCov_eq_microCov_of_equal_sizes (hK : 0 < K) (c : ℕ) (hc : 0 < c)
    (cov : Fin K → ℝ) : microCov (fun _ => c) cov = macroCov cov := by
  have hKR : (0:ℝ) < K := by exact_mod_cast hK
  have hcR : (0:ℝ) < c := by exact_mod_cast hc
  unfold microCov macroCov
  rw [← Finset.mul_sum]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [mul_comm ((K:ℝ)) ((c:ℝ))]
  rw [mul_div_mul_left _ _ (ne_of_gt hcR)]

/-- **Size imbalance is exactly what separates the two numbers.**  The element-level miss rate is
the group-level miss rate inflated by the ratio of the largest group to the average group. -/
theorem microCov_ge_of_macroCov (hK : 0 < K) (sizes : Fin K → ℕ) (cov : Fin K → ℝ)
    (hpos : 0 < ∑ k, sizes k) (nmax : ℕ) (hmax : ∀ k, sizes k ≤ nmax)
    (hcov : ∀ k, cov k ≤ 1) :
    1 - microCov sizes cov
      ≤ ((nmax : ℝ) * K / (∑ k, (sizes k : ℝ))) * (1 - macroCov cov) := by
  have hKR : (0:ℝ) < K := by exact_mod_cast hK
  have hsum : (0:ℝ) < ∑ k, (sizes k : ℝ) := by
    have : (0:ℝ) < ((∑ k, sizes k : ℕ) : ℝ) := by exact_mod_cast hpos
    simpa using this
  have hkey : (∑ k, (sizes k : ℝ)) - ∑ k, (sizes k : ℝ) * cov k
      ≤ (nmax : ℝ) * (K - ∑ k, cov k) := by
    have h1 : (∑ k, (sizes k : ℝ)) - ∑ k, (sizes k : ℝ) * cov k
        = ∑ k, (sizes k : ℝ) * (1 - cov k) := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun k _ => by ring
    have hKsum : ((K:ℝ) - ∑ k, cov k) = ∑ k, (1 - cov k) := by
      rw [Finset.sum_sub_distrib]
      simp
    have h2 : (nmax : ℝ) * (K - ∑ k, cov k) = ∑ k, (nmax : ℝ) * (1 - cov k) := by
      rw [hKsum, Finset.mul_sum]
    rw [h1, h2]
    refine Finset.sum_le_sum fun k _ => ?_
    have hs : (sizes k : ℝ) ≤ (nmax : ℝ) := by exact_mod_cast hmax k
    have hc : 0 ≤ 1 - cov k := by linarith [hcov k]
    exact mul_le_mul_of_nonneg_right hs hc
  unfold microCov macroCov
  have hexp : ((nmax : ℝ) * K / (∑ k, (sizes k : ℝ))) * (1 - (∑ k, cov k) / K)
      = ((nmax : ℝ) * (K - ∑ k, cov k)) / (∑ k, (sizes k : ℝ)) := by
    field_simp
  have hleft : 1 - (∑ k, (sizes k : ℝ) * cov k) / (∑ k, (sizes k : ℝ))
      = ((∑ k, (sizes k : ℝ)) - ∑ k, (sizes k : ℝ) * cov k) / (∑ k, (sizes k : ℝ)) := by
    field_simp
  rw [hexp, hleft]
  gcongr

theorem sum_uncovered_last (J : ℕ) :
    ∑ k : Fin (J + 1), (if k = Fin.last J then (0:ℝ) else 1) = (J : ℝ) := by
  have h1 : ∀ k : Fin (J + 1), (if k = Fin.last J then (0:ℝ) else 1)
      = 1 - (if k = Fin.last J then (1:ℝ) else 0) := by
    intro k; split <;> norm_num
  rw [Finset.sum_congr rfl (fun k _ => h1 k), Finset.sum_sub_distrib]
  simp

theorem sum_sizes_last (J N : ℕ) :
    ∑ k : Fin (J + 1), ((if k = Fin.last J then N else 1 : ℕ) : ℝ) = (J : ℝ) + N := by
  have h1 : ∀ k : Fin (J + 1), ((if k = Fin.last J then N else 1 : ℕ) : ℝ)
      = 1 + (if k = Fin.last J then ((N:ℝ) - 1) else 0) := by
    intro k; split <;> norm_num
  rw [Finset.sum_congr rfl (fun k _ => h1 k), Finset.sum_add_distrib]
  simp

theorem sum_sizes_mul_cov_last (J N : ℕ) :
    ∑ k : Fin (J + 1), ((if k = Fin.last J then N else 1 : ℕ) : ℝ)
      * (if k = Fin.last J then (0:ℝ) else 1) = (J : ℝ) := by
  have h1 : ∀ k : Fin (J + 1), ((if k = Fin.last J then N else 1 : ℕ) : ℝ)
      * (if k = Fin.last J then (0:ℝ) else 1) = 1 - (if k = Fin.last J then (1:ℝ) else 0) := by
    intro k; split <;> norm_num
  rw [Finset.sum_congr rfl (fun k _ => h1 k), Finset.sum_sub_distrib]
  simp

theorem sum_sizes_pos_last {J : ℕ} (hJ : 0 < J) (N : ℕ) :
    0 < ∑ k : Fin (J + 1), (if k = Fin.last J then N else 1) := by
  refine Finset.sum_pos' (fun i _ => Nat.zero_le _)
    ⟨⟨0, by omega⟩, Finset.mem_univ _, ?_⟩
  have hne : ¬ (J = 0) := by omega
  simp [hne]

/-- **A group-level guarantee does not pin down element-level coverage.**  For every target level
`α ∈ (0,1)` and every `ε > 0` there are two grouped panels reporting the *same* group-level
coverage, both valid at level `α`, one covering at least a `1 - α` fraction of residues and the
other at most an `ε` fraction: the group-level number is silent about the residue-level one. -/
theorem grouped_validity_underdetermines_element_coverage
    {alpha eps : ℝ} (ha : 0 < alpha) (ha1 : alpha < 1) (he : 0 < eps) :
    ∃ (K : ℕ) (sizes sizes' : Fin K → ℕ) (cov cov' : Fin K → ℝ),
      0 < K ∧ (∀ k, 0 ≤ cov k ∧ cov k ≤ 1) ∧ (∀ k, 0 ≤ cov' k ∧ cov' k ≤ 1) ∧
      (0 < ∑ k, sizes k) ∧ (0 < ∑ k, sizes' k) ∧
      macroCov cov = macroCov cov' ∧ 1 - alpha ≤ macroCov cov ∧
      microCov sizes cov ≤ eps ∧ 1 - alpha ≤ microCov sizes' cov' := by
  classical
  -- `J + 1` groups, of which one is uncovered; the uncovered group holds `N` residues
  set J : ℕ := ⌈1 / alpha⌉₊ with hJdef
  have halphainv : 1 / alpha ≤ (J : ℝ) := Nat.le_ceil _
  have hJR : (0:ℝ) < (J : ℝ) := lt_of_lt_of_le (by positivity) halphainv
  have hJpos : 0 < J := by exact_mod_cast hJR
  set N : ℕ := ⌈(J : ℝ) / eps⌉₊ with hNdef
  have hNge : (J : ℝ) / eps ≤ (N : ℝ) := Nat.le_ceil _
  have hJ1 : (0:ℝ) < (J : ℝ) + 1 := by linarith
  -- the group-level number both panels report
  have hmacro : macroCov (fun k : Fin (J + 1) => if k = Fin.last J then (0:ℝ) else 1)
      = (J : ℝ) / ((J : ℝ) + 1) := by
    unfold macroCov
    rw [sum_uncovered_last]
    push_cast
    ring
  have hmacro' : macroCov (fun _ : Fin (J + 1) => (J : ℝ) / ((J : ℝ) + 1))
      = (J : ℝ) / ((J : ℝ) + 1) := by
    unfold macroCov
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    push_cast
    field_simp
  have hvalid : 1 - alpha ≤ (J : ℝ) / ((J : ℝ) + 1) := by
    rw [le_div_iff₀ hJ1]
    have h1 : 1 ≤ alpha * J := by
      rw [div_le_iff₀ ha] at halphainv
      linarith
    nlinarith
  refine ⟨J + 1, (fun k => if k = Fin.last J then N else 1), (fun _ => 1),
    (fun k => if k = Fin.last J then 0 else 1), (fun _ => (J : ℝ) / ((J : ℝ) + 1)),
    Nat.succ_pos _, ?_, ?_, sum_sizes_pos_last hJpos N, ?_, ?_, ?_, ?_, ?_⟩
  · intro k; dsimp only; split <;> norm_num
  · intro k
    constructor
    · positivity
    · rw [div_le_one hJ1]; linarith
  · simp
  · rw [hmacro, hmacro']
  · rw [hmacro]; exact hvalid
  · -- the element-level coverage of the first panel is tiny
    unfold microCov
    rw [sum_sizes_mul_cov_last, sum_sizes_last]
    rw [div_le_iff₀ (by linarith : (0:ℝ) < (J:ℝ) + N)]
    have hNJ : (J : ℝ) ≤ eps * N := by
      rw [div_le_iff₀ he] at hNge
      linarith
    nlinarith [he.le, hJR.le]
  · -- the element-level coverage of the second panel is the reported one
    unfold microCov
    simp only [Nat.cast_one, one_mul, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul]
    push_cast
    have hEq : ((J:ℝ) + 1) * ((J:ℝ) / ((J:ℝ) + 1)) / (((J:ℝ) + 1) * 1)
        = (J:ℝ) / ((J:ℝ) + 1) := by
      field_simp
    rw [hEq]
    exact hvalid

end Grouping

end IDR.ConformalRisk
