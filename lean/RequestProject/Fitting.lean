/-
# Part XLIV.1  Fitting an ensemble to data: how much agreement is evidence

The standard way to produce a model of a disordered region is to take a pool of candidate
conformations and fit weights on them so that the ensemble averages of `n` measured observables
match the data.  The fit is then reported as agreement with experiment.  This file asks the
question a referee should ask: how much of that agreement is evidence about the ensemble, and
how much is arithmetic?

Throughout, `A i j` is the value of observable `i` on conformation `j` of the pool, `b i` is the
measurement, and a *fit* is a weight vector `w ≥ 0` with `Σ_j w j = 1` and
`Σ_j w j · A i j = b i` for every `i` (`IsFit`).

* `fit_with_few_structures` -- **Carathéodory's theorem, read as a statement about ensembles.**
  If the data can be fitted at all, they can be fitted *exactly* by an ensemble supported on at
  most `n + 1` conformations, no matter how large the pool is.  Exact agreement with `n`
  measurements is therefore reproduced by `n + 1` structures; it constrains the ensemble only
  through feasibility.
* `fit_not_unique` -- **and with more conformations than that, the fit is a continuum.**  If the
  pool has more than `n + 1` members and some fit uses all of them, there is a direction `u` in
  weight space -- summing to zero and invisible in every measured observable -- along which the
  whole segment `w + t·u` consists of exact fits.  Nothing in the data chooses between them; the
  choice is made by the regulariser.
* `fit_blind_to_unmeasured` -- **an explicit case.**  Three conformations, one measured
  observable, one unmeasured one: two exact fits of the same datum differ by the whole range of
  the unmeasured observable.
* `fit_unique_of_ker_trivial` -- **the positive counterpart.**  Uniqueness is exactly the
  triviality of that kernel: if no nonzero weight direction is invisible to the measurements,
  the fit is unique.  This is the condition an experimental design has to establish, and
  `fit_not_unique` says that a pool larger than `n + 1` never satisfies it.
-/
import Mathlib

set_option autoImplicit false

namespace IDR

open Finset Module

namespace Fitting

variable {n k : ℕ}

/-- `w` is an exact fit of the data `b` for the forward model `A`. -/
def IsFit (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) (w : Fin k → ℝ) : Prop :=
  (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ ∀ i, ∑ j, w j * A i j = b i

/-! ## Carathéodory: `n + 1` structures fit `n` measurements -/

/-- **If the data can be fitted, they can be fitted by `n + 1` conformations.**  Exact agreement
with `n` measured averages is reproduced by an ensemble of at most `n + 1` structures, however
large the pool of candidates is: the size of the fitted ensemble is not evidence, and neither is
the agreement itself beyond the fact that the data are feasible. -/
theorem fit_with_few_structures (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) {w : Fin k → ℝ}
    (hfit : IsFit A b w) :
    ∃ w' : Fin k → ℝ, IsFit A b w' ∧
      (Finset.univ.filter (fun j => w' j ≠ 0)).card ≤ n + 1 := by
  classical
  obtain ⟨hw, hsum, hfit⟩ := hfit
  set col : Fin k → (Fin n → ℝ) := fun j i => A i j with hcol
  have hb : b = ∑ j, w j • col j := by
    funext i
    rw [Finset.sum_apply]
    simpa [hcol, smul_eq_mul] using (hfit i).symm
  have hmem : b ∈ convexHull ℝ (Set.range col) := by
    rw [hb]
    have := Finset.centerMass_mem_convexHull (t := (Finset.univ : Finset (Fin k))) (w := w)
      (z := col) (fun i _ => hw i) (by rw [hsum]; norm_num) (fun i _ => Set.mem_range_self i)
    rwa [Finset.centerMass_eq_of_sum_1 _ _ hsum] at this
  obtain ⟨ι, hfin, z, v, hrange, haff, hvpos, hvsum, hvb⟩ :=
    eq_pos_convex_span_of_mem_convexHull hmem
  have hcard : Fintype.card ι ≤ n + 1 := by
    have h1 := haff.card_le_finrank_succ
    have h2 : Module.finrank ℝ ↥(vectorSpan ℝ (Set.range z)) ≤ Module.finrank ℝ (Fin n → ℝ) :=
      Submodule.finrank_le _
    have h3 : Module.finrank ℝ (Fin n → ℝ) = n := by simp
    omega
  have hchoose : ∀ i : ι, ∃ j : Fin k, col j = z i := by
    intro i
    obtain ⟨j, hj⟩ := hrange (Set.mem_range_self i)
    exact ⟨j, hj⟩
  choose g hg using hchoose
  refine ⟨fun j => ∑ i ∈ Finset.univ.filter (fun i : ι => g i = j), v i,
    ⟨fun j => Finset.sum_nonneg fun i _ => (hvpos i).le, ?_, ?_⟩, ?_⟩
  · rw [Finset.sum_fiberwise]
    exact hvsum
  · intro i
    have hterm : ∀ j : Fin k, (∑ i' ∈ Finset.univ.filter (fun i' : ι => g i' = j), v i') * A i j
        = ∑ i' ∈ Finset.univ.filter (fun i' : ι => g i' = j), v i' * A i (g i') := by
      intro j
      rw [Finset.sum_mul]
      refine Finset.sum_congr rfl fun i' hi' => ?_
      rw [(Finset.mem_filter.mp hi').2]
    rw [Finset.sum_congr rfl (fun j _ => hterm j), Finset.sum_fiberwise]
    have hz : ∀ i' : ι, A i (g i') = z i' i := by
      intro i'
      simpa [hcol] using congrFun (hg i') i
    rw [Finset.sum_congr rfl (fun i' _ => by rw [hz i'])]
    have hvbi := congrFun hvb i
    rw [Finset.sum_apply] at hvbi
    simpa [smul_eq_mul] using hvbi
  · have hsub : (Finset.univ.filter
        (fun j => (∑ i ∈ Finset.univ.filter (fun i : ι => g i = j), v i) ≠ 0))
        ⊆ Finset.univ.image g := by
      intro j hj
      simp only [Finset.mem_filter] at hj
      by_contra hcon
      have hempty : Finset.univ.filter (fun i : ι => g i = j) = ∅ := by
        refine Finset.filter_eq_empty_iff.mpr fun i _ => ?_
        intro hgi
        exact hcon (Finset.mem_image.mpr ⟨i, Finset.mem_univ i, hgi⟩)
      rw [hempty] at hj
      simp at hj
    calc _ ≤ (Finset.univ.image g).card := Finset.card_le_card hsub
      _ ≤ Fintype.card ι := Finset.card_image_le
      _ ≤ n + 1 := hcard

/-! ## Underdetermination -/

/-- The linear map a fit has to invert: a weight vector goes to its total together with the
`n` predicted averages. -/
noncomputable def fitMap (A : Fin n → Fin k → ℝ) : (Fin k → ℝ) →ₗ[ℝ] ℝ × (Fin n → ℝ) where
  toFun u := (∑ j, u j, fun i => ∑ j, u j * A i j)
  map_add' u u' := by
    refine Prod.ext ?_ ?_
    · simp [Finset.sum_add_distrib]
    · funext i
      simp [add_mul, Finset.sum_add_distrib]
  map_smul' c u := by
    refine Prod.ext ?_ ?_
    · simp [Finset.mul_sum]
    · funext i
      simp [Finset.mul_sum, mul_assoc]

/-- **A fit is unique exactly when no weight direction is invisible to the measurements.** -/
theorem fit_unique_of_ker_trivial (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ)
    (hker : LinearMap.ker (fitMap A) = ⊥) {w w' : Fin k → ℝ}
    (h : IsFit A b w) (h' : IsFit A b w') : w = w' := by
  have hval : ∀ v : Fin k → ℝ, IsFit A b v → fitMap A v = (1, b) := by
    intro v hv
    refine Prod.ext ?_ ?_
    · simpa [fitMap] using hv.2.1
    · funext i
      simpa [fitMap] using hv.2.2 i
  have hmem : w - w' ∈ LinearMap.ker (fitMap A) := by
    refine LinearMap.mem_ker.mpr ?_
    rw [map_sub, hval w h, hval w' h', sub_self]
  rw [hker, Submodule.mem_bot, sub_eq_zero] at hmem
  exact hmem

/-- **More conformations than measurements: the exact fits form a continuum.**  If the pool has
more than `n + 1` members and some fit gives every member positive weight, then there is a
nonzero direction `u` in weight space that sums to zero and leaves every measured average
unchanged, and a whole interval of `t` for which `w + t·u` is again an exact fit.  The data
cannot distinguish any two of them. -/
theorem fit_not_unique (A : Fin n → Fin k → ℝ) (b : Fin n → ℝ) {w : Fin k → ℝ}
    (hwpos : ∀ j, 0 < w j) (hfit : IsFit A b w) (hk : n + 1 < k) :
    ∃ u : Fin k → ℝ, u ≠ 0 ∧ (∑ j, u j = 0) ∧ (∀ i, ∑ j, u j * A i j = 0) ∧
      ∃ eps > (0 : ℝ), ∀ t : ℝ, |t| ≤ eps → IsFit A b (fun j => w j + t * u j) := by
  classical
  have hkpos : 0 < k := by omega
  have hne : Nonempty (Fin k) := ⟨⟨0, hkpos⟩⟩
  -- a nonzero weight direction invisible to the measurements
  have hrank : Module.finrank ℝ (ℝ × (Fin n → ℝ)) < Module.finrank ℝ (Fin k → ℝ) := by
    simp only [Module.finrank_prod, Module.finrank_self, Module.finrank_pi, Fintype.card_fin]
    omega
  have hker : LinearMap.ker (fitMap A) ≠ ⊥ := LinearMap.ker_ne_bot_of_finrank_lt hrank
  obtain ⟨u, hu, hune⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hker
  have hu' := LinearMap.mem_ker.mp hu
  have husum : ∑ j, u j = 0 := congrArg Prod.fst hu'
  have huobs : ∀ i, ∑ j, u j * A i j = 0 := by
    intro i
    have := congrArg Prod.snd hu'
    exact congrFun this i
  -- the interval of admissible step sizes
  obtain ⟨jmin, -, hjmin⟩ := Finset.exists_min_image (Finset.univ : Finset (Fin k)) w
    ⟨Classical.arbitrary (Fin k), Finset.mem_univ _⟩
  obtain ⟨jmax, -, hjmax⟩ := Finset.exists_max_image (Finset.univ : Finset (Fin k))
    (fun j => |u j|) ⟨Classical.arbitrary (Fin k), Finset.mem_univ _⟩
  have hUpos : 0 < |u jmax| := by
    rcases Function.ne_iff.mp hune with ⟨j, hj⟩
    have : 0 < |u j| := abs_pos.mpr (by simpa using hj)
    exact lt_of_lt_of_le this (hjmax j (Finset.mem_univ j))
  have hepspos : 0 < w jmin / |u jmax| := div_pos (hwpos jmin) hUpos
  refine ⟨u, hune, husum, huobs, w jmin / |u jmax|, hepspos, fun t ht => ?_⟩
  refine ⟨fun j => ?_, ?_, ?_⟩
  · have h1 : |t * u j| ≤ (w jmin / |u jmax|) * |u jmax| := by
      rw [abs_mul]
      exact mul_le_mul ht (hjmax j (Finset.mem_univ j)) (abs_nonneg _) hepspos.le
    have h2 : (w jmin / |u jmax|) * |u jmax| = w jmin := by
      field_simp
    have h3 : |t * u j| ≤ w j := le_trans (h1.trans_eq h2) (hjmin j (Finset.mem_univ j))
    have h4 : -(w j) ≤ t * u j := by
      have := neg_abs_le (t * u j)
      linarith
    linarith
  · rw [Finset.sum_add_distrib, hfit.2.1, ← Finset.mul_sum, husum]
    ring
  · intro i
    have hexp : ∀ j, (w j + t * u j) * A i j = w j * A i j + t * (u j * A i j) := by
      intro j
      ring
    rw [Finset.sum_congr rfl (fun j _ => hexp j), Finset.sum_add_distrib, hfit.2.2 i,
      ← Finset.mul_sum, huobs i]
    ring

/-! ## An explicit case -/

/-- Three conformations; the measured observable takes the values `0, 1, 2` on them. -/
def poolMeasured : Fin 1 → Fin 3 → ℝ := ![![0, 1, 2]]

/-- An unmeasured observable on the same three conformations. -/
def poolUnmeasured : Fin 3 → ℝ := ![0, 1, 0]

/-- **Two exact fits of the same datum, disagreeing completely about what was not measured.**
The single conformation of intermediate value and the equal mixture of the two extremes both
reproduce the measured average `1` exactly, and predict `1` and `0` for the unmeasured
observable: the full range it takes on the pool. -/
theorem fit_blind_to_unmeasured :
    IsFit poolMeasured ![1] ![0, 1, 0] ∧
    IsFit poolMeasured ![1] ![1 / 2, 0, 1 / 2] ∧
    (∑ j, (![0, 1, 0] : Fin 3 → ℝ) j * poolUnmeasured j) = 1 ∧
    (∑ j, (![1 / 2, 0, 1 / 2] : Fin 3 → ℝ) j * poolUnmeasured j) = 0 := by
  refine ⟨⟨fun j => ?_, ?_, fun i => ?_⟩, ⟨fun j => ?_, ?_, fun i => ?_⟩, ?_, ?_⟩ <;>
    first
      | (fin_cases j <;> norm_num)
      | (fin_cases i <;>
          norm_num [poolMeasured, Fin.sum_univ_three, Matrix.cons_val_two, Matrix.tail_cons])
      | norm_num [poolUnmeasured, Fin.sum_univ_three, Matrix.cons_val_two, Matrix.tail_cons]

end Fitting

end IDR
