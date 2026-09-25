/-
# No finite panel of ensemble averages can bound structural error

Every experiment routinely used on a disordered region reports an *ensemble average*: the
mean radius of gyration from SAXS, a mean FRET efficiency, averaged scalar couplings and
chemical shifts, an averaged RDC.  `RequestProject.TransportDuality` shows that each such
average, when the underlying observable is Lipschitz, yields a lower bound on the structural
transport error, and `RequestProject.TransportOneDim` shows that a *distribution* of one
descriptor yields the exact transport distance along that coordinate.

This file proves the negative half, quantitatively.  Fix **any** finite panel of real-valued
observables -- arbitrary functions, not even continuous -- and any error budget `eps`.  Then
there are two ensembles that agree on every average in the panel exactly and are nevertheless
at transport distance at least `eps`.  Averages cannot be made to work by taking more of
them: the deficiency is not statistical but geometric, and it is repaired only by measuring
distributions.

* `exists_kernel_vector` -- the linear-algebra core: `m + 1` linear constraints on `m + 2`
  grid populations always have a nonzero solution.
* `exists_large_cum` -- a normalised signed population has a cumulative sum of size at least
  `1 / (2n)` somewhere: a perturbation that is invisible to the panel is still visible to
  the cumulative distribution.
* `no_finite_panel_of_means_certifies` -- the theorem.

Design consequence: a model of an intrinsically disordered region cannot be validated by a
panel of averages, however large.  The validation data must be distributional, and the
`RequestProject.TransportOneDim` certificate is then exact.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportOneDim

namespace IDR

open Finset
open scoped Classical

/-! ## Two elementary ingredients -/

/-- The average of an observable under a grid ensemble. -/
lemma gridEns_expect (n : ℕ) (t : ℕ → ℝ) (p : ℕ → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hs : ∑ i ∈ Finset.range n, p i = 1) (f : ℝ → ℝ) :
    (gridEns n t p hp hs).expect f = ∑ i ∈ Finset.range n, p i * f (t i) := by
  simp only [Ens.expect, gridEns]
  exact Fin.sum_univ_eq_sum_range (fun i => p i * f (t i)) n

/-- **The linear-algebra core.**  Any `m + 1` linear functionals on the populations of
`m + 2` grid sites annihilate a common nonzero signed population. -/
lemma exists_kernel_vector {m : ℕ} (g : Fin (m + 1) → Fin (m + 2) → ℝ) :
    ∃ v : Fin (m + 2) → ℝ, v ≠ 0 ∧ ∀ i, ∑ k, v k * g i k = 0 := by
  let phi : (Fin (m + 2) → ℝ) →ₗ[ℝ] (Fin (m + 1) → ℝ) :=
    { toFun := fun v i => ∑ k, v k * g i k
      map_add' := by intro a b; funext i; simp [add_mul, Finset.sum_add_distrib]
      map_smul' := by intro c a; funext i; simp [Finset.mul_sum, mul_assoc] }
  have hnotinj : ¬ Function.Injective phi := by
    intro h
    have hle := LinearMap.finrank_le_finrank_of_injective (f := phi) h
    simp at hle
  have hker : LinearMap.ker phi ≠ ⊥ := fun h => hnotinj (LinearMap.ker_eq_bot.1 h)
  obtain ⟨v, hv, hv0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  refine ⟨v, hv0, fun i => ?_⟩
  have h1 : phi v = 0 := hv
  simpa [phi] using congrFun h1 i

/-- **A normalised signed population is visible to the cumulative distribution.**  If the
absolute populations sum to one, some cumulative sum has size at least `1 / (2n)`. -/
lemma exists_large_cum {u : ℕ → ℝ} {n : ℕ} (hn : 0 < n)
    (hsum : ∑ k ∈ Finset.range n, |u k| = 1) :
    ∃ k₀ < n, 1 / (2 * (n : ℝ)) ≤ |cumW u (k₀ + 1)| := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  have h1 : ∃ k₁ ∈ Finset.range n, 1 / (n : ℝ) ≤ |u k₁| := by
    by_contra hcon
    push_neg at hcon
    have hlt : ∑ k ∈ Finset.range n, |u k| < ∑ _k ∈ Finset.range n, (1 / (n : ℝ)) :=
      Finset.sum_lt_sum_of_nonempty (Finset.nonempty_range_iff.2 (by omega))
        (fun k hk => hcon k hk)
    rw [hsum, Finset.sum_const, Finset.card_range, nsmul_eq_mul] at hlt
    rw [mul_one_div, div_self (ne_of_gt hnR)] at hlt
    exact lt_irrefl 1 hlt
  obtain ⟨k₁, hk₁, hu⟩ := h1
  have hk₁n : k₁ < n := Finset.mem_range.1 hk₁
  have hdiff : u k₁ = cumW u (k₁ + 1) - cumW u k₁ := by rw [cumW_succ]; ring
  have hsplit : |u k₁| ≤ |cumW u (k₁ + 1)| + |cumW u k₁| := by
    rw [hdiff]
    exact abs_sub _ _
  rcases le_or_gt (1 / (2 * (n : ℝ))) |cumW u (k₁ + 1)| with h | h
  · exact ⟨k₁, hk₁n, h⟩
  · have h2 : 1 / (2 * (n : ℝ)) ≤ |cumW u k₁| := by
      have hhalf : 1 / (n : ℝ) = 2 * (1 / (2 * (n : ℝ))) := by field_simp
      rw [hhalf] at hu
      linarith
    have hk1pos : 0 < k₁ := by
      rcases Nat.eq_zero_or_pos k₁ with h0 | h0
      · exfalso
        rw [h0, cumW_zero, abs_zero] at h2
        have hpos : (0 : ℝ) < 1 / (2 * (n : ℝ)) := by positivity
        linarith
      · exact h0
    refine ⟨k₁ - 1, by omega, ?_⟩
    rwa [show k₁ - 1 + 1 = k₁ by omega]

/-! ## The theorem -/

/-- **No finite panel of ensemble averages can bound structural error.**  Given any finite
list `f` of real-valued observables -- radii of gyration, FRET efficiencies, scalar
couplings, arbitrary functions of the conformation -- and any error budget `eps`, there are
two conformational ensembles that reproduce every average in the panel *exactly* and yet lie
at transport distance at least `eps`.

The construction is explicit: put `m + 2` conformations on a line at spacing `eps · (m+2)²`,
observe that the `m + 1` linear conditions (`m` panel averages plus normalisation) cannot
pin down `m + 2` populations, and split the resulting free direction symmetrically between
the two ensembles.

A disorder model can therefore never be validated by averages, no matter how many are
measured; only distributional data can certify it.  Compare
`transportCost_line_eq_cdfL1`, where a single measured *distribution* determines the
transport distance exactly. -/
theorem no_finite_panel_of_means_certifies {m : ℕ} (f : Fin m → ℝ → ℝ) {eps : ℝ}
    (heps : 0 < eps) :
    ∃ E F : Ens ℝ, (∀ j, E.expect (f j) = F.expect (f j)) ∧
      eps ≤ transportCost lineCost E F := by
  classical
  set n : ℕ := m + 2 with hn
  have hnpos : 0 < n := by omega
  have hnR : (0 : ℝ) < n := by exact_mod_cast hnpos
  -- the grid
  set s : ℝ := eps * (n : ℝ) ^ 2 with hs
  have hspos : 0 < s := by positivity
  set t : ℕ → ℝ := fun k => s * k with htdef
  have ht : Monotone t := by
    intro a b hab
    have : (a : ℝ) ≤ b := by exact_mod_cast hab
    exact mul_le_mul_of_nonneg_left this (le_of_lt hspos)
  -- the free direction
  obtain ⟨v, hv0, hvg⟩ :=
    exists_kernel_vector (m := m) (fun i k => Fin.cases (1 : ℝ) (fun j => f j (t k)) i)
  have hvsum : ∑ k, v k = 0 := by
    have := hvg 0
    simpa using this
  have hvf : ∀ j : Fin m, ∑ k, v k * f j (t k) = 0 := by
    intro j
    have := hvg j.succ
    simpa using this
  -- transfer to `ℕ`-indexed weights
  set vN : ℕ → ℝ := fun k => if h : k < n then v ⟨k, h⟩ else 0 with hvN
  have hvNfin : ∀ k : Fin n, vN (k : ℕ) = v k := by
    intro k
    simp [hvN, k.isLt]
  have hvNsum : ∑ k ∈ Finset.range n, vN k = 0 := by
    rw [← Fin.sum_univ_eq_sum_range vN n, Finset.sum_congr rfl (fun k _ => hvNfin k)]
    exact hvsum
  have hvNf : ∀ j : Fin m, ∑ k ∈ Finset.range n, vN k * f j (t k) = 0 := by
    intro j
    rw [← Fin.sum_univ_eq_sum_range (fun k => vN k * f j (t k)) n,
      Finset.sum_congr rfl (fun k _ => by rw [hvNfin k])]
    exact hvf j
  -- normalise
  set B : ℝ := ∑ k ∈ Finset.range n, |vN k| with hB
  have hBpos : 0 < B := by
    rcases lt_or_eq_of_le (Finset.sum_nonneg (fun k _ => abs_nonneg (vN k)) : (0:ℝ) ≤ B) with h | h
    · exact h
    · exfalso
      apply hv0
      funext k
      have hzero : ∀ k ∈ Finset.range n, |vN k| = 0 :=
        (Finset.sum_eq_zero_iff_of_nonneg (fun k _ => abs_nonneg (vN k))).1 h.symm
      have := hzero (k : ℕ) (Finset.mem_range.2 k.isLt)
      rw [hvNfin k] at this
      simpa using abs_eq_zero.1 this
  set u : ℕ → ℝ := fun k => vN k / B with hu
  have husum : ∑ k ∈ Finset.range n, |u k| = 1 := by
    have : ∀ k, |u k| = |vN k| / B := by
      intro k; rw [hu]; simp [abs_div, abs_of_pos hBpos]
    rw [Finset.sum_congr rfl (fun k _ => this k), ← Finset.sum_div, ← hB,
      div_self (ne_of_gt hBpos)]
  have huabs : ∀ k, |u k| ≤ 1 := by
    intro k
    by_cases hk : k < n
    · rw [← husum]
      exact Finset.single_le_sum (fun i _ => abs_nonneg (u i)) (Finset.mem_range.2 hk)
    · have : vN k = 0 := by simp [hvN, hk]
      simp [hu, this]
  have huzero : ∑ k ∈ Finset.range n, u k = 0 := by
    simp only [hu]
    rw [← Finset.sum_div, hvNsum, zero_div]
  have huf : ∀ j : Fin m, ∑ k ∈ Finset.range n, u k * f j (t k) = 0 := by
    intro j
    have hstep : ∀ k ∈ Finset.range n, u k * f j (t k) = (vN k * f j (t k)) / B := by
      intro k _; simp only [hu]; ring
    rw [Finset.sum_congr rfl hstep, ← Finset.sum_div, hvNf j, zero_div]
  -- the two ensembles
  set unif : ℕ → ℝ := fun k => if k < n then (1 : ℝ) / n else 0 with hunif
  set pW : ℕ → ℝ := fun k => unif k + u k / n with hpW
  set qW : ℕ → ℝ := fun k => unif k - u k / n with hqW
  have hunifsum : ∑ k ∈ Finset.range n, unif k = 1 := by
    have hstep : ∀ k ∈ Finset.range n, unif k = 1 / (n : ℝ) := by
      intro k hk
      simp [hunif, Finset.mem_range.1 hk]
    rw [Finset.sum_congr rfl hstep, Finset.sum_const, Finset.card_range, nsmul_eq_mul,
      mul_one_div, div_self (ne_of_gt hnR)]
  have hunifzero : ∀ k, ¬ k < n → unif k = 0 := by
    intro k hk; simp [hunif, hk]
  have huNzero : ∀ k, ¬ k < n → u k = 0 := by
    intro k hk
    have : vN k = 0 := by simp [hvN, hk]
    simp [hu, this]
  have hpnn : ∀ k, 0 ≤ pW k := by
    intro k
    simp only [hpW]
    by_cases hk : k < n
    · have h1 : unif k = 1 / (n : ℝ) := by simp [hunif, hk]
      have h2 : -1 ≤ u k := neg_le_of_abs_le (huabs k)
      have h3 : (1 : ℝ) / n + u k / n = (1 + u k) / n := by ring
      rw [h1, h3]
      exact div_nonneg (by linarith) (le_of_lt hnR)
    · rw [hunifzero k hk, huNzero k hk]
      simp
  have hqnn : ∀ k, 0 ≤ qW k := by
    intro k
    simp only [hqW]
    by_cases hk : k < n
    · have h1 : unif k = 1 / (n : ℝ) := by simp [hunif, hk]
      have h2 : u k ≤ 1 := le_of_abs_le (huabs k)
      have h3 : (1 : ℝ) / n - u k / n = (1 - u k) / n := by ring
      rw [h1, h3]
      exact div_nonneg (by linarith) (le_of_lt hnR)
    · rw [hunifzero k hk, huNzero k hk]
      simp
  have hpsum : ∑ k ∈ Finset.range n, pW k = 1 := by
    simp only [hpW]
    rw [Finset.sum_add_distrib, hunifsum, ← Finset.sum_div, huzero, zero_div, add_zero]
  have hqsum : ∑ k ∈ Finset.range n, qW k = 1 := by
    simp only [hqW]
    rw [Finset.sum_sub_distrib, hunifsum, ← Finset.sum_div, huzero, zero_div, sub_zero]
  refine ⟨gridEns n t pW hpnn hpsum, gridEns n t qW hqnn hqsum, ?_, ?_⟩
  · -- every panel average agrees
    intro j
    rw [gridEns_expect, gridEns_expect]
    have hstep : ∀ k ∈ Finset.range n,
        pW k * f j (t k) - qW k * f j (t k) = (2 / (n : ℝ)) * (u k * f j (t k)) := by
      intro k _
      simp only [hpW, hqW]
      ring
    have hzero : ∑ k ∈ Finset.range n, (pW k * f j (t k) - qW k * f j (t k)) = 0 := by
      rw [Finset.sum_congr rfl hstep, ← Finset.mul_sum, huf j, mul_zero]
    rw [Finset.sum_sub_distrib] at hzero
    linarith
  · -- but the transport distance is at least `eps`
    rw [transportCost_line_eq_cdfL1 n ht hpnn hqnn hpsum hqsum]
    obtain ⟨k₀, hk₀, hcum⟩ := exists_large_cum (u := u) hnpos husum
    have hcumdiff : ∀ k, cumW pW k - cumW qW k = (2 / (n : ℝ)) * cumW u k := by
      intro k
      simp only [cumW, hpW, hqW]
      rw [← Finset.sum_sub_distrib, Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => by ring
    have hterm : ∀ k ∈ Finset.range n,
        (0 : ℝ) ≤ (t (k + 1) - t k) * |cumW pW (k + 1) - cumW qW (k + 1)| := by
      intro k _
      refine mul_nonneg ?_ (abs_nonneg _)
      simpa using sub_nonneg.2 (ht (Nat.le_succ k))
    have hsingle : (t (k₀ + 1) - t k₀) * |cumW pW (k₀ + 1) - cumW qW (k₀ + 1)|
        ≤ cdfL1 n t pW qW :=
      Finset.single_le_sum hterm (Finset.mem_range.2 hk₀)
    have hgap : t (k₀ + 1) - t k₀ = s := by
      simp only [htdef]
      push_cast
      ring
    have hval : |cumW pW (k₀ + 1) - cumW qW (k₀ + 1)| = (2 / (n : ℝ)) * |cumW u (k₀ + 1)| := by
      rw [hcumdiff, abs_mul, abs_of_pos (by positivity : (0:ℝ) < 2 / (n : ℝ))]
    rw [hgap, hval] at hsingle
    refine le_trans ?_ hsingle
    have hlow : (2 / (n : ℝ)) * (1 / (2 * (n : ℝ))) ≤ (2 / (n : ℝ)) * |cumW u (k₀ + 1)| :=
      mul_le_mul_of_nonneg_left hcum (by positivity)
    have hcalc : s * ((2 / (n : ℝ)) * (1 / (2 * (n : ℝ)))) = eps := by
      rw [hs]
      field_simp
    calc eps = s * ((2 / (n : ℝ)) * (1 / (2 * (n : ℝ)))) := hcalc.symm
      _ ≤ s * ((2 / (n : ℝ)) * |cumW u (k₀ + 1)|) :=
          mul_le_mul_of_nonneg_left hlow (le_of_lt hspos)

end IDR
