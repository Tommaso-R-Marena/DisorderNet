/-
# Part XLVIII.1  The entropy meter: what an order parameter bounds

The generalised order parameter `S²` measured by NMR spin relaxation is routinely converted into
a conformational entropy, and differences of `S²` between a free and a bound state into an
entropy of binding -- the "entropy meter".  The conversion needs a model, and this file says
exactly which part of the conversion is model-free and which is not.

The setting is the one already fixed in `RequestProject.NMR`: a weighted ensemble of unit bond
vectors, its order tensor and its Lipari--Szabo order parameter `S²`.  Here the ensemble also
carries a Shannon entropy, and the two are compared.

* `orderParam_orthogonal` -- the computational bridge.  When the populated orientations are
  mutually orthogonal, `S² = (3/2)Σ_k w_k² − 1/2`: the order parameter is an affine function of
  the *purity* of the population, and of nothing else.
* `purity_of_orderParam` -- equivalently `Σ_k w_k² = (2S² + 1)/3`.  A single order parameter
  measures one number about the population: its collision probability.
* `shannon_ge_neg_log_purity` -- the Rényi-2 bound, from Jensen's inequality for `log`: the
  Shannon entropy is at least `−log Σ_k w_k²`.
* `orderParam_entropy_lower_bound` -- **what the entropy meter certifies**: for orthogonal
  orientations, `H ≥ −log((2S² + 1)/3)`.  A measured order parameter is a rigorous *lower* bound
  on the conformational entropy of the bond, with no model and no calibration.
* `shannon_le_log_card` -- and the only general upper bound is the number of states, which `S²`
  does not constrain.
* `aligned_entropy_unbounded` -- the negative half, and the reason the meter needs a model: an
  ensemble of `N` distinct conformations that happen to share one bond orientation has
  `S² = 1` -- a perfectly rigid bond -- and entropy `log N`, unbounded.  An order parameter
  bounds the entropy of the *orientational marginal it measures*, never the conformational
  entropy of the chain.
-/
import Mathlib
import RequestProject.NMR

set_option autoImplicit false

namespace IDR

open Finset

namespace EntropyMeter

open IDR.NMR

/-- The Shannon entropy of a finite population, in nats. -/
noncomputable def shannon {ι : Type*} [Fintype ι] (p : ι → ℝ) : ℝ :=
  ∑ i, -(p i * Real.log (p i))

/-- The purity, or collision probability, of a finite population. -/
noncomputable def purity {ι : Type*} [Fintype ι] (p : ι → ℝ) : ℝ := ∑ i, p i ^ 2

/-! ## The order parameter measures the purity -/

/-- **The order parameter of a population of mutually orthogonal orientations.**  It is an
affine function of the purity `Σ_k w_k²` of the population, and of nothing else: a single `S²`
is a single number about the distribution of orientations. -/
theorem orderParam_orthogonal {m : ℕ} (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ)
    (horth : ∀ k l, dot (u k) (u l) ^ 2 = if k = l then 1 else 0) :
    orderParam w u = 3 / 2 * purity w - 1 / 2 := by
  rw [orderParam_eq_pair]
  have hrow : ∀ k : Fin m, ∑ l, w k * w l * dot (u k) (u l) ^ 2 = w k ^ 2 := by
    intro k
    rw [Finset.sum_eq_single k]
    · rw [horth k k]; simp [sq]
    · intro l _ hlk
      rw [horth k l, if_neg (Ne.symm hlk), mul_zero]
    · intro h
      exact absurd (Finset.mem_univ k) h
  rw [Finset.sum_congr rfl (fun k _ => hrow k)]
  rfl

/-- **The purity is the order parameter.**  Inverting the previous identity: a measured `S²`
determines `Σ_k w_k²` exactly, and only that. -/
theorem purity_of_orderParam {m : ℕ} (w : Fin m → ℝ) (u : Fin m → Fin 3 → ℝ)
    (horth : ∀ k l, dot (u k) (u l) ^ 2 = if k = l then 1 else 0) :
    purity w = (2 * orderParam w u + 1) / 3 := by
  rw [orderParam_orthogonal w u horth]
  ring

/-! ## What the purity bounds -/

/-- **The Rényi-2 bound on the Shannon entropy.**  For any population, `H ≥ −log Σ_k w_k²`.
This is Jensen's inequality for the logarithm, applied to the population itself. -/
theorem shannon_ge_neg_log_purity {m : ℕ} {w : Fin m → ℝ} (hw : ∀ k, 0 < w k)
    (hsum : ∑ k, w k = 1) : -Real.log (purity w) ≤ shannon w := by
  have hjensen := (strictConcaveOn_log_Ioi.concaveOn).le_map_sum
    (t := (univ : Finset (Fin m))) (w := w) (p := w)
    (fun i _ => (hw i).le) hsum (fun i _ => Set.mem_Ioi.mpr (hw i))
  simp only [smul_eq_mul] at hjensen
  have hpur : ∑ i, w i * w i = purity w := by
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [sq]
  rw [hpur] at hjensen
  have hsh : shannon w = -∑ i, w i * Real.log (w i) := by
    rw [shannon, ← Finset.sum_neg_distrib]
  rw [hsh]
  linarith

/-- **What a measured order parameter certifies.**  For a population of mutually orthogonal
orientations, the conformational entropy of the bond is at least `−log((2S² + 1)/3)`.  No model,
no calibration: a small order parameter is a rigorous lower bound on entropy. -/
theorem orderParam_entropy_lower_bound {m : ℕ} {w : Fin m → ℝ} {u : Fin m → Fin 3 → ℝ}
    (hw : ∀ k, 0 < w k) (hsum : ∑ k, w k = 1)
    (horth : ∀ k l, dot (u k) (u l) ^ 2 = if k = l then 1 else 0) :
    -Real.log ((2 * orderParam w u + 1) / 3) ≤ shannon w := by
  rw [← purity_of_orderParam w u horth]
  exact shannon_ge_neg_log_purity hw hsum

/-- The only general upper bound on the entropy is the number of populated states. -/
theorem shannon_le_log_card {m : ℕ} {w : Fin m → ℝ} (hw : ∀ k, 0 < w k) (hsum : ∑ k, w k = 1) :
    shannon w ≤ Real.log m := by
  have hjensen := (strictConcaveOn_log_Ioi.concaveOn).le_map_sum
    (t := (univ : Finset (Fin m))) (w := w) (p := fun i => (w i)⁻¹)
    (fun i _ => (hw i).le) hsum (fun i _ => Set.mem_Ioi.mpr (inv_pos.mpr (hw i)))
  simp only [smul_eq_mul] at hjensen
  have hpt : ∀ i : Fin m, w i * ((w i)⁻¹) = 1 := fun i => mul_inv_cancel₀ (hw i).ne'
  have hrhs : ∑ i, w i * (w i)⁻¹ = (m : ℝ) := by
    rw [Finset.sum_congr rfl (fun i _ => hpt i)]
    simp
  have hlhs : ∑ i, w i * Real.log (w i)⁻¹ = shannon w := by
    rw [shannon]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Real.log_inv]
    ring
  rw [hrhs, hlhs] at hjensen
  exact hjensen

/-! ## What it does not bound -/

/-- The uniform population on `N` conformations. -/
noncomputable def uniform (N : ℕ) : Fin N → ℝ := fun _ => (N : ℝ)⁻¹

theorem uniform_sum {N : ℕ} (hN : 0 < N) : ∑ i, uniform N i = 1 := by
  simp only [uniform, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  field_simp

theorem shannon_uniform {N : ℕ} (hN : 0 < N) : shannon (uniform N) = Real.log N := by
  have hNne : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  simp only [shannon, uniform, Real.log_inv, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul]
  field_simp

/-- **A rigid bond vector says nothing about conformational entropy.**  An ensemble of `N`
distinct conformations that share a single bond orientation has the maximal order parameter
`S² = 1` and entropy `log N`; the entropy is unbounded above at fixed `S²`.  An order parameter
constrains the orientational marginal of the bond it reports on, and the conversion of `ΔS²`
into a conformational entropy is therefore a property of the model that performs it. -/
theorem aligned_entropy_unbounded (v : Fin 3 → ℝ) (hv : IsBondVector v) (B : ℝ) :
    ∃ N : ℕ, 0 < N ∧
      orderParam (uniform N) (fun _ => v) = 1 ∧
      shannon (uniform N) = Real.log N ∧ B < shannon (uniform N) := by
  obtain ⟨N, hN⟩ := exists_nat_gt (Real.exp B)
  have hNpos : 0 < N := by
    have hpos : (0:ℝ) < N := (Real.exp_pos B).trans hN
    exact_mod_cast hpos
  refine ⟨N, hNpos, ?_, shannon_uniform hNpos, ?_⟩
  · refine orderParam_eq_one_of_aligned (uniform_sum hNpos) ?_
    intro k l
    have : dot v v = 1 := hv
    rw [this]; norm_num
  · rw [shannon_uniform hNpos]
    have hNR : (0:ℝ) < N := by exact_mod_cast hNpos
    calc B = Real.log (Real.exp B) := by rw [Real.log_exp]
      _ < Real.log N := Real.log_lt_log (Real.exp_pos B) hN

end EntropyMeter

end IDR
