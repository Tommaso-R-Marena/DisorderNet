/-
# Part IX.1  Real chain statistics: stiffness, persistence length, radius of gyration

The earlier chain calculation (`RequestProject.Chain`) used the crudest caricature of a
disordered region, the *freely jointed* chain, in which successive bonds are statistically
independent.  Real polypeptide backbones are not freely jointed: steric exclusion and the
partial double-bond character of the peptide bond correlate successive bond directions, and
the correlation decays exponentially along the contour with a *persistence length*
`l_p ≈ 4 Å` for a denatured or disordered polypeptide.  This file develops the standard
statistics of such a chain exactly, with no continuum or large-`N` approximation, and reads
off the consequences for what an ensemble model of a disordered region has to reproduce.

The physical input is a single, experimentally meaningful assumption, the one that defines
the worm-like / freely-rotating chain:

  the unit tangents obey `⟪u_i, u_j⟫ = a^{|i-j|}` with `0 ≤ a < 1`,

i.e. the tangent–tangent correlation decays exponentially in the contour separation
(`corr_eq_exp_of_persistence`, which identifies `a = exp(-b / l_p)`).

Results.

* `corrSum_closed_form` -- the exact double geometric sum.  Hence
  `msd_eq` : `⟨R²⟩ = b²·[ N(1+a)/(1-a) - 2a(1-a^N)/(1-a)² ]`, the discrete worm-like-chain
  formula, valid for every `N`, not only asymptotically.
* `msd_ideal` (`a = 0`, `⟨R²⟩ = N b²`, the random walk) and `msd_rod` (`a = 1`,
  `⟨R²⟩ = N² b²`, the rigid rod) are the two limits, and `msd_mono_stiffness` shows the
  chain swells monotonically with stiffness in between.
* `kuhn_length_limit` -- `⟨R²⟩/N → b²(1+a)/(1-a)`: an effective Kuhn segment
  `b_K = b(1+a)/(1-a)`, so on long enough scales a stiff chain is a random walk with a
  renormalised step.  This is why global observables cannot see local stiffness --
* `stiffness_bondlength_degeneracy`, the design consequence: for *any* two stiffnesses there
  are bond lengths making the mean-square end-to-end distances exactly equal.  A single
  global size measurement (a SAXS `Rg`, a smFRET efficiency) determines one number and
  therefore cannot separate local stiffness from local geometry: it is not, by itself, a
  restraint on the conformational ensemble.
* `gyration_eq_pair_sum` -- the exact identity `Rg² = (1/2N²)·Σ_{ij} |r_i - r_j|²` in any
  real inner-product space: the radius of gyration *is* a pair statistic, which is why SAXS
  (a pair-distance experiment) measures it and why it is blind to everything else.
* `ideal_gyration` -- for the ideal chain `⟨Rg²⟩ = b²(N²-1)/(6N)` exactly, hence
  `ideal_gyration_ratio` : `6⟨Rg²⟩ ≤ ⟨R²⟩` with the classical `⟨Rg²⟩ → ⟨R²⟩/6` recovered as
  `ideal_gyration_limit`.
-/
import Mathlib

namespace IDR

open Finset

namespace Polymer

/-! ## The tangent-correlation sum -/

/-- The double sum of tangent–tangent correlations `Σ_{i,j<N} a^{|i-j|}`.  For a chain of
`N` bonds of length `b` whose unit tangents satisfy `⟪u_i,u_j⟫ = a^{|i-j|}`, the mean-square
end-to-end distance is `b²` times this sum. -/
noncomputable def corrSum (a : ℝ) (N : ℕ) : ℝ :=
  ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N, a ^ (Nat.dist i j)

/-- Geometric sum of the correlations of the last bond with all previous ones. -/
lemma sum_pow_sub (a : ℝ) (ha : a ≠ 1) (N : ℕ) :
    ∑ i ∈ Finset.range N, a ^ (N - i) = a * (1 - a ^ N) / (1 - a) := by
  have h : ∑ i ∈ Finset.range N, a ^ (N - i) = ∑ j ∈ Finset.range N, a ^ (j + 1) := by
    rw [← Finset.sum_range_reflect (fun i => a ^ (N - i)) N]
    refine Finset.sum_congr rfl (fun j hj => ?_)
    simp only [Finset.mem_range] at hj
    congr 1
    omega
  have h2 : ∑ j ∈ Finset.range N, a ^ (j + 1) = a * ∑ j ∈ Finset.range N, a ^ j := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun j _ => by ring)
  have key : (a ^ N - 1) / (a - 1) = (1 - a ^ N) / (1 - a) := by
    rw [← neg_sub 1 (a ^ N), ← neg_sub 1 a, neg_div_neg_eq]
  rw [h, h2, geom_sum_eq ha, key, mul_div_assoc]

/-- Adding one bond adds its correlation with itself and, twice, with each earlier bond. -/
lemma corrSum_succ (a : ℝ) (N : ℕ) :
    corrSum a (N + 1) = corrSum a N + 2 * (∑ i ∈ Finset.range N, a ^ (N - i)) + 1 := by
  unfold corrSum
  rw [Finset.sum_range_succ]
  have h1 : ∀ i ∈ Finset.range N, ∑ j ∈ Finset.range (N + 1), a ^ (Nat.dist i j)
      = (∑ j ∈ Finset.range N, a ^ (Nat.dist i j)) + a ^ (N - i) := by
    intro i hi
    simp only [Finset.mem_range] at hi
    rw [Finset.sum_range_succ]
    congr 2
    unfold Nat.dist
    omega
  rw [Finset.sum_congr rfl h1, Finset.sum_add_distrib, Finset.sum_range_succ]
  have h2 : ∀ j ∈ Finset.range N, a ^ (Nat.dist N j) = a ^ (N - j) := by
    intro j hj
    simp only [Finset.mem_range] at hj
    congr 1
    unfold Nat.dist
    omega
  rw [Finset.sum_congr rfl h2]
  simp [Nat.dist_self]
  ring

/-- **The discrete worm-like chain sum.**  `Σ_{i,j<N} a^{|i-j|}` in closed form. -/
theorem corrSum_closed_form (a : ℝ) (ha : a ≠ 1) (N : ℕ) :
    corrSum a N = N * (1 + a) / (1 - a) - 2 * a * (1 - a ^ N) / (1 - a) ^ 2 := by
  have h1 : (1 : ℝ) - a ≠ 0 := sub_ne_zero.mpr (Ne.symm ha)
  induction N with
  | zero => simp [corrSum]
  | succ N ih =>
      rw [corrSum_succ, ih, sum_pow_sub a ha]
      push_cast
      field_simp
      ring

/-- A freely jointed chain: uncorrelated tangents. -/
@[simp] lemma corrSum_zero (N : ℕ) : corrSum 0 N = N := by
  unfold corrSum
  have h : ∀ i ∈ Finset.range N, ∑ j ∈ Finset.range N, (0 : ℝ) ^ (Nat.dist i j) = 1 := by
    intro i hi
    rw [Finset.sum_eq_single i]
    · simp [Nat.dist_self]
    · intro j _ hj
      have hd : Nat.dist i j ≠ 0 := fun h => hj (Nat.eq_of_dist_eq_zero h).symm
      simp [zero_pow hd]
    · intro h; exact absurd hi h
  rw [Finset.sum_congr rfl h]
  simp

/-- A rigid rod: perfectly correlated tangents. -/
@[simp] lemma corrSum_one (N : ℕ) : corrSum 1 N = (N : ℝ) ^ 2 := by
  unfold corrSum
  simp
  ring

/-- With nonnegative correlations every off-diagonal term helps: the chain is at least as
extended as a random walk. -/
lemma corrSum_ge (a : ℝ) (ha : 0 ≤ a) (N : ℕ) : (N : ℝ) ≤ corrSum a N := by
  have : corrSum 0 N ≤ corrSum a N := by
    unfold corrSum
    refine Finset.sum_le_sum (fun i _ => Finset.sum_le_sum (fun j _ => ?_))
    exact pow_le_pow_left₀ le_rfl ha _
  simpa using this

lemma corrSum_pos (a : ℝ) (ha : 0 ≤ a) {N : ℕ} (hN : 0 < N) : 0 < corrSum a N :=
  lt_of_lt_of_le (by exact_mod_cast hN) (corrSum_ge a ha N)

/-! ## Mean-square end-to-end distance -/

/-- The mean-square end-to-end distance of a chain of `N` bonds of length `b` with
tangent correlation `⟪u_i,u_j⟫ = a^{|i-j|}`. -/
noncomputable def msd (a b : ℝ) (N : ℕ) : ℝ := b ^ 2 * corrSum a N

/-- **The discrete worm-like chain formula.** -/
theorem msd_eq (a b : ℝ) (ha : a ≠ 1) (N : ℕ) :
    msd a b N = b ^ 2 * (N * (1 + a) / (1 - a) - 2 * a * (1 - a ^ N) / (1 - a) ^ 2) := by
  unfold msd
  rw [corrSum_closed_form a ha]

/-- Ideal (freely jointed) chain: `⟨R²⟩ = N b²`. -/
theorem msd_ideal (b : ℝ) (N : ℕ) : msd 0 b N = N * b ^ 2 := by
  unfold msd; rw [corrSum_zero]; ring

/-- Rigid rod: `⟨R²⟩ = N² b²`. -/
theorem msd_rod (b : ℝ) (N : ℕ) : msd 1 b N = (N : ℝ) ^ 2 * b ^ 2 := by
  unfold msd; rw [corrSum_one]; ring

/-- A chain swells monotonically with stiffness. -/
theorem msd_mono_stiffness {a a' b : ℝ} (ha : 0 ≤ a) (haa : a ≤ a') (N : ℕ) :
    msd a b N ≤ msd a' b N := by
  unfold msd
  refine mul_le_mul_of_nonneg_left ?_ (sq_nonneg b)
  unfold corrSum
  exact Finset.sum_le_sum (fun i _ => Finset.sum_le_sum
    (fun j _ => pow_le_pow_left₀ ha haa _))

/-- Every chain is at least a random walk and at most a rod. -/
theorem msd_bounds {a b : ℝ} (ha : 0 ≤ a) (ha1 : a ≤ 1) (N : ℕ) :
    N * b ^ 2 ≤ msd a b N ∧ msd a b N ≤ (N : ℝ) ^ 2 * b ^ 2 := by
  constructor
  · rw [← msd_ideal b N]; exact msd_mono_stiffness le_rfl ha N
  · rw [← msd_rod b N]; exact msd_mono_stiffness ha ha1 N

/-! ## Persistence length -/

/-- The persistence length of a chain of bond length `b` with tangent correlation ratio `a`:
`l_p = -b / log a`, the contour length over which the tangent correlation falls by `e`. -/
noncomputable def persistenceLength (a b : ℝ) : ℝ := -b / Real.log a

/-- **Exponential decay of the tangent correlation in contour length.**  The correlation
between bonds separated by `d` bonds, i.e. by a contour length `s = b·d`, is
`exp(-s / l_p)`. -/
theorem corr_eq_exp_of_persistence {a b : ℝ} (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (d : ℕ) :
    a ^ d = Real.exp (-(b * d) / persistenceLength a b) := by
  have hlog : Real.log a < 0 := Real.log_neg ha ha1
  unfold persistenceLength
  rw [div_div_eq_mul_div, neg_mul, neg_div_neg_eq]
  rw [show (b * d) * Real.log a / b = d * Real.log a by field_simp]
  rw [Real.exp_nat_mul, Real.exp_log ha]

/-- The persistence length is positive for a genuine (flexible, `0 < a < 1`) chain. -/
lemma persistenceLength_pos {a b : ℝ} (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) :
    0 < persistenceLength a b := by
  have hlog : Real.log a < 0 := Real.log_neg ha ha1
  unfold persistenceLength
  exact div_pos_of_neg_of_neg (by linarith) hlog

/-! ## The Kuhn length: long chains forget their stiffness -/

/-- **Effective Kuhn segment.**  `⟨R²⟩/N → b²(1+a)/(1-a)`: on scales long compared with the
persistence length any chain of finite stiffness is a random walk with a renormalised step
length `b_K = b(1+a)/(1-a)`. -/
theorem kuhn_length_limit {a b : ℝ} (ha : |a| < 1) :
    Filter.Tendsto (fun N : ℕ => msd a b N / N) Filter.atTop
      (nhds (b ^ 2 * (1 + a) / (1 - a))) := by
  have ha1 : a ≠ 1 := by
    intro h; rw [h] at ha; simp at ha
  have h1 : (1 : ℝ) - a ≠ 0 := sub_ne_zero.mpr (Ne.symm ha1)
  have hpow : Filter.Tendsto (fun N : ℕ => a ^ N) Filter.atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_abs_lt_one ha
  have hnum : Filter.Tendsto
      (fun N : ℕ => b ^ 2 * (2 * a * (1 - a ^ N) / (1 - a) ^ 2)) Filter.atTop
      (nhds (b ^ 2 * (2 * a * (1 - 0) / (1 - a) ^ 2))) := by
    exact (((tendsto_const_nhds.sub hpow).const_mul (2 * a)).div_const _).const_mul _
  have hden : Filter.Tendsto (fun N : ℕ => (N : ℝ)) Filter.atTop Filter.atTop :=
    tendsto_natCast_atTop_atTop
  have hzero : Filter.Tendsto
      (fun N : ℕ => b ^ 2 * (2 * a * (1 - a ^ N) / (1 - a) ^ 2) / N) Filter.atTop (nhds 0) :=
    hnum.div_atTop hden
  have : Filter.Tendsto
      (fun N : ℕ => b ^ 2 * (1 + a) / (1 - a)
        - b ^ 2 * (2 * a * (1 - a ^ N) / (1 - a) ^ 2) / N) Filter.atTop
      (nhds (b ^ 2 * (1 + a) / (1 - a) - 0)) := tendsto_const_nhds.sub hzero
  rw [sub_zero] at this
  refine this.congr' ?_
  filter_upwards [Filter.eventually_gt_atTop 0] with N hN
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  rw [msd_eq a b ha1]
  field_simp

/-- **A single global size measurement cannot separate stiffness from geometry.**  Given any
chain (`a`, `b`) and any other stiffness `a'`, there is a bond length `b'` for which the two
chains have exactly the same mean-square end-to-end distance at length `N`.  A SAXS radius of
gyration, or a single FRET efficiency, is one number: it constrains the product, never the
factors, and so is not by itself a restraint on the conformational ensemble. -/
theorem stiffness_bondlength_degeneracy {a a' b : ℝ} (ha : 0 ≤ a) (ha' : 0 ≤ a') (hb : 0 < b)
    {N : ℕ} (hN : 0 < N) :
    ∃ b' : ℝ, 0 < b' ∧ msd a' b' N = msd a b N := by
  have hS : 0 < corrSum a N := corrSum_pos a ha hN
  have hS' : 0 < corrSum a' N := corrSum_pos a' ha' hN
  refine ⟨Real.sqrt (b ^ 2 * corrSum a N / corrSum a' N), ?_, ?_⟩
  · exact Real.sqrt_pos.mpr (div_pos (by positivity) hS')
  · unfold msd
    rw [Real.sq_sqrt (le_of_lt (div_pos (by positivity) hS'))]
    field_simp

/-! ## Radius of gyration -/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The squared radius of gyration of a configuration `r` of `N` monomers. -/
noncomputable def gyrationSq {N : ℕ} (r : Fin N → E) : ℝ :=
  (1 / (N : ℝ)) * ∑ i, ‖r i - (1 / (N : ℝ)) • ∑ k, r k‖ ^ 2

/-- **The radius of gyration is a pair statistic.**  `Rg² = (1/2N²)·Σ_{ij}|r_i - r_j|²`, so a
pair-distance experiment (SAXS, and its Debye formula) measures exactly this and nothing
else about the configuration. -/
theorem gyration_eq_pair_sum {N : ℕ} (hN : 0 < N) (r : Fin N → E) :
    gyrationSq r = (1 / (2 * (N : ℝ) ^ 2)) * ∑ i, ∑ j, ‖r i - r j‖ ^ 2 := by
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  unfold gyrationSq
  set m : E := (1 / (N : ℝ)) • ∑ k, r k with hm
  have expand : ∀ x y : E, ‖x - y‖ ^ 2 = ‖x‖ ^ 2 - 2 * (inner ℝ x y) + ‖y‖ ^ 2 := by
    intro x y; rw [@norm_sub_sq_real]
  have hsum : ∑ k, r k = (N : ℝ) • m := by
    rw [hm, smul_smul, mul_one_div, div_self hN', one_smul]
  have L : ∑ i, ‖r i - m‖ ^ 2 = (∑ i, ‖r i‖ ^ 2) - (N : ℝ) * ‖m‖ ^ 2 := by
    simp only [expand]
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, ← sum_inner,
      hsum, real_inner_smul_left, real_inner_self_eq_norm_sq]
    simp
    ring
  have R : ∑ i, ∑ j, ‖r i - r j‖ ^ 2
      = 2 * (N : ℝ) * (∑ i, ‖r i‖ ^ 2) - 2 * (N : ℝ) ^ 2 * ‖m‖ ^ 2 := by
    simp only [expand]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum]
    have hinner : ∑ i, ∑ j, (inner ℝ (r i) (r j) : ℝ) = (inner ℝ (∑ i, r i) (∑ j, r j) : ℝ) := by
      rw [sum_inner]
      exact Finset.sum_congr rfl (fun i _ => (inner_sum _ _ _).symm)
    rw [hinner, hsum, real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq]
    ring
  rw [L, R]
  field_simp

/-! ## The ideal chain's radius of gyration -/

/-- `Σ_{i<N} i = N(N-1)/2`, over the reals. -/
lemma sum_range_cast (N : ℕ) :
    ∑ i ∈ Finset.range N, (i : ℝ) = (N : ℝ) * ((N : ℝ) - 1) / 2 := by
  induction N with
  | zero => simp
  | succ M ih => rw [Finset.sum_range_succ, ih]; push_cast; ring

/-- `Σ_{i,j<N} |i-j| = N(N²-1)/3`. -/
theorem pair_dist_sum (N : ℕ) :
    ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N, (Nat.dist i j : ℝ)
      = (N : ℝ) * ((N : ℝ) ^ 2 - 1) / 3 := by
  induction N with
  | zero => simp
  | succ N ih =>
      rw [Finset.sum_range_succ]
      have h1 : ∀ i ∈ Finset.range N, ∑ j ∈ Finset.range (N + 1), (Nat.dist i j : ℝ)
          = (∑ j ∈ Finset.range N, (Nat.dist i j : ℝ)) + ((N : ℝ) - i) := by
        intro i hi
        simp only [Finset.mem_range] at hi
        rw [Finset.sum_range_succ]
        congr 1
        have hd : Nat.dist i N = N - i := by unfold Nat.dist; omega
        have hle : i ≤ N := le_of_lt hi
        rw [hd, Nat.cast_sub hle]
      rw [Finset.sum_congr rfl h1, Finset.sum_add_distrib, ih, Finset.sum_range_succ]
      have h2 : ∀ j ∈ Finset.range N, ((Nat.dist N j : ℕ) : ℝ) = (N : ℝ) - j := by
        intro j hj
        simp only [Finset.mem_range] at hj
        have hd : Nat.dist N j = N - j := by unfold Nat.dist; omega
        have hle : j ≤ N := le_of_lt hj
        rw [hd, Nat.cast_sub hle]
      rw [Finset.sum_congr rfl h2]
      have hgauss : ∑ i ∈ Finset.range N, ((N : ℝ) - i) = (N : ℝ) * ((N : ℝ) + 1) / 2 := by
        rw [Finset.sum_sub_distrib, sum_range_cast, Finset.sum_const, Finset.card_range,
          nsmul_eq_mul]
        ring
      rw [hgauss]
      simp only [Nat.dist_self, Nat.cast_zero]
      push_cast
      ring

/-- **The ideal chain's radius of gyration**, exactly: `⟨Rg²⟩ = b²(N²-1)/(6N)`, from the
pair identity together with `⟨(r_i-r_j)²⟩ = |i-j| b²`. -/
theorem ideal_gyration (b : ℝ) {N : ℕ} (hN : 0 < N) :
    (1 / (2 * (N : ℝ) ^ 2)) * ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N,
        (Nat.dist i j : ℝ) * b ^ 2
      = b ^ 2 * ((N : ℝ) ^ 2 - 1) / (6 * N) := by
  have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
  have hfac : ∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N, (Nat.dist i j : ℝ) * b ^ 2
      = (∑ i ∈ Finset.range N, ∑ j ∈ Finset.range N, (Nat.dist i j : ℝ)) * b ^ 2 := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl (fun i _ => (Finset.sum_mul _ _ _).symm)
  rw [hfac, pair_dist_sum]
  field_simp
  ring

/-- `6⟨Rg²⟩ ≤ ⟨R²⟩`: the classical ratio, as an inequality valid at every finite `N`. -/
theorem ideal_gyration_ratio (b : ℝ) {N : ℕ} (hN : 0 < N) :
    6 * (b ^ 2 * ((N : ℝ) ^ 2 - 1) / (6 * N)) ≤ msd 0 b N := by
  rw [msd_ideal]
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  have h6 : 6 * (b ^ 2 * ((N : ℝ) ^ 2 - 1) / (6 * N)) = b ^ 2 * ((N : ℝ) ^ 2 - 1) / N := by
    field_simp
  rw [h6, div_le_iff₀ hN']
  nlinarith [sq_nonneg b, hN']

/-- `⟨Rg²⟩ / (N b²) → 1/6`. -/
theorem ideal_gyration_limit (b : ℝ) :
    Filter.Tendsto (fun N : ℕ => (b ^ 2 * ((N : ℝ) ^ 2 - 1) / (6 * N)) / (N * b ^ 2))
      Filter.atTop (nhds (b ^ 2 / (6 * b ^ 2))) := by
  have h : ∀ᶠ N : ℕ in Filter.atTop,
      (b ^ 2 * ((N : ℝ) ^ 2 - 1) / (6 * N)) / (N * b ^ 2)
        = (b ^ 2 * (1 - 1 / (N : ℝ) ^ 2)) / (6 * b ^ 2) := by
    filter_upwards [Filter.eventually_gt_atTop 0] with N hN
    have hN' : (N : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hN.ne'
    by_cases hb : b = 0
    · simp [hb]
    · field_simp
  refine Filter.Tendsto.congr' (Filter.EventuallyEq.symm h) ?_
  have h1 : Filter.Tendsto (fun N : ℕ => 1 / (N : ℝ) ^ 2) Filter.atTop (nhds 0) := by
    simpa using (tendsto_natCast_atTop_atTop (R := ℝ)).inv_tendsto_atTop.pow 2
  have heq : b ^ 2 * (1 - (0 : ℝ)) / (6 * b ^ 2) = b ^ 2 / (6 * b ^ 2) := by ring
  rw [← heq]
  exact ((tendsto_const_nhds.sub h1).const_mul (b ^ 2)).div_const (6 * b ^ 2)


end Polymer

end IDR
