/-
# Part LXXVII  The geometry a radius of gyration can and cannot report

Every experimental statement about the size of a disordered region -- a SAXS radius of gyration,
a hydrodynamic radius, a "compaction" -- is a *single number* extracted from a conformational
ensemble whose true content is a whole matrix of interatomic distances.  This file makes that
relation exact, in a general real inner-product space, and then prices it.

* `rg2_eq` -- the moment form of the squared radius of gyration: `Rg^2` is the mean square norm
  minus the square norm of the mean.
* `pairSum_eq` -- the same moments computed from the matrix of squared distances.
* `rg2_eq_pairSum` -- **the readout identity.**  `Rg^2 = (1 / 2 n^2) * sum_{i,j} |x_i - x_j|^2`:
  the radius of gyration is one fixed *linear* functional of the squared-distance matrix, with no
  reference to the coordinates themselves.
* `rg2_congr_of_dist_eq` -- **and therefore a blind readout.**  Two conformations with the same
  distance matrix -- mirror images, and every other isometric or non-isometric coincidence -- have
  literally the same radius of gyration.  One measured number stands in for `n(n-1)/2` numbers.
* `rg2_union` -- **the two-module (parallel-axis) law.**  For a protein split into two blocks --
  the realistic case of a folded domain plus a disordered tail -- the global second moment is the
  exact sum `n Rg^2 = n_A Rg_A^2 + n_B Rg_B^2 + (n_A n_B / n) d^2` with `d` the distance between
  the two block centroids.  Global size, block sizes and module separation are one equation, not
  three.
* `inter_module_dist_sq_le` -- **a usable experimental inequality.**  Because block radii are
  nonnegative, that identity bounds the inter-module distance by the global radius of gyration:
  `d^2 <= (n^2 / (n_A n_B)) Rg^2`.  A measured global `Rg` is a hard ceiling on how far the folded
  domain can sit from the centre of its disordered partner.
* `rg2_tradeoff` -- **and a hard degeneracy.**  Two explicit four-bead chains have exactly the same
  global `Rg^2 = 2` while one has a compact block pair held `2` apart and the other has a fourfold
  more expanded block with coincident centroids.  Tail expansion and module separation are
  exchangeable at fixed global size: a single `Rg` cannot report either one.
* `norm_sub_le_bond`, `rg2_le_rod` -- **the chain-connectivity ceiling.**  If consecutive beads are
  within `b`, then `|x_j - x_i| <= b (j - i)` and hence `Rg^2 <= b^2 (n^2 - 1) / 12`.
* `rod_rg2`, `rg2_rod_sharp` -- **and it is attained.**  The straight rod `x_i = i b` has
  `Rg^2 = b^2 (n^2 - 1)/12` exactly, so the bound is the true supremum, not an estimate.

Read together with Parts LXIX-LXXII: `Rg` is exactly the kind of report those parts license -- a
linear functional, recomputable from the data, with a stated blind spot.  Read against Part LXXIII:
the blindness is of the same kind as the homometric charge sequences, a many-to-one map from
structure to observable which no amount of extra precision inverts.
-/
import Mathlib

set_option autoImplicit false

namespace IDR.ChainGeo

open Finset

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The centroid of the beads indexed by `s`. -/
noncomputable def cen (s : Finset ℕ) (x : ℕ → E) : E := (s.card : ℝ)⁻¹ • ∑ i ∈ s, x i

/-- The squared radius of gyration of the beads indexed by `s`. -/
noncomputable def rg2 (s : Finset ℕ) (x : ℕ → E) : ℝ :=
  (s.card : ℝ)⁻¹ * ∑ i ∈ s, ‖x i - cen s x‖ ^ 2

/-- The sum of all squared interbead distances: the full distance-matrix content. -/
noncomputable def pairSum (s : Finset ℕ) (x : ℕ → E) : ℝ :=
  ∑ i ∈ s, ∑ j ∈ s, ‖x i - x j‖ ^ 2

/-- The squared radius of gyration is nonnegative. -/
theorem rg2_nonneg (s : Finset ℕ) (x : ℕ → E) : 0 ≤ rg2 s x := by
  apply mul_nonneg (by positivity)
  exact Finset.sum_nonneg fun i _ => by positivity

/-- The distance-matrix sum in terms of second moments. -/
theorem pairSum_eq (s : Finset ℕ) (x : ℕ → E) :
    pairSum s x = 2 * s.card * (∑ i ∈ s, ‖x i‖ ^ 2) - 2 * ‖∑ i ∈ s, x i‖ ^ 2 := by
  have h : ∀ i j : ℕ, ‖x i - x j‖ ^ 2 = ‖x i‖ ^ 2 - 2 * (inner ℝ (x i) (x j)) + ‖x j‖ ^ 2 := by
    intro i j; rw [norm_sub_sq_real]
  simp only [pairSum, h, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  have h1 : ∑ _i ∈ s, ∑ j ∈ s, ‖x j‖ ^ 2 = (s.card : ℝ) * ∑ i ∈ s, ‖x i‖ ^ 2 := by
    simp [Finset.sum_const, nsmul_eq_mul]
  have h2 : ∑ i ∈ s, ∑ _j ∈ s, ‖x i‖ ^ 2 = (s.card : ℝ) * ∑ i ∈ s, ‖x i‖ ^ 2 := by
    rw [Finset.sum_comm]; exact h1
  have h3 : ∑ i ∈ s, ∑ j ∈ s, 2 * (inner ℝ (x i) (x j) : ℝ) = 2 * ‖∑ i ∈ s, x i‖ ^ 2 := by
    have e : ∀ i ∈ s, ∑ j ∈ s, 2 * (inner ℝ (x i) (x j) : ℝ)
        = 2 * (inner ℝ (x i) (∑ j ∈ s, x j) : ℝ) := by
      intro i _; rw [inner_sum, Finset.mul_sum]
    rw [Finset.sum_congr rfl e, ← Finset.mul_sum, ← sum_inner, real_inner_self_eq_norm_sq]
  rw [h1, h2, h3]; ring

/-- **Moment form.**  `Rg^2` is the mean square norm minus the square norm of the mean. -/
theorem rg2_eq (s : Finset ℕ) (x : ℕ → E) :
    rg2 s x = (s.card : ℝ)⁻¹ * (∑ i ∈ s, ‖x i‖ ^ 2)
      - ((s.card : ℝ)⁻¹) ^ 2 * ‖∑ i ∈ s, x i‖ ^ 2 := by
  set n : ℝ := (s.card : ℝ) with hn
  set S : E := ∑ i ∈ s, x i with hS
  have hexp : ∀ i : ℕ, ‖x i - cen s x‖ ^ 2
      = ‖x i‖ ^ 2 - 2 * (inner ℝ (x i) (cen s x) : ℝ) + ‖cen s x‖ ^ 2 := by
    intro i; rw [norm_sub_sq_real]
  have hsum : ∑ i ∈ s, ‖x i - cen s x‖ ^ 2
      = (∑ i ∈ s, ‖x i‖ ^ 2) - 2 * n⁻¹ * ‖S‖ ^ 2 + n * (n⁻¹ ^ 2 * ‖S‖ ^ 2) := by
    simp only [hexp, Finset.sum_add_distrib, Finset.sum_sub_distrib]
    have e1 : ∑ i ∈ s, 2 * (inner ℝ (x i) (cen s x) : ℝ) = 2 * n⁻¹ * ‖S‖ ^ 2 := by
      rw [← Finset.mul_sum, ← sum_inner, cen, real_inner_smul_right, ← hS,
        real_inner_self_eq_norm_sq]
      ring
    have e2 : ∑ _i ∈ s, ‖cen s x‖ ^ 2 = n * (n⁻¹ ^ 2 * ‖S‖ ^ 2) := by
      rw [Finset.sum_const, nsmul_eq_mul, cen, norm_smul, ← hS]
      simp [mul_pow, hn]
    rw [e1, e2]
  rw [rg2, hsum]
  rcases Nat.eq_zero_or_pos s.card with h0 | h0
  · simp [hn, h0]
  · have hne : n ≠ 0 := by rw [hn]; exact Nat.cast_ne_zero.mpr h0.ne'
    field_simp
    ring

/-- **The readout identity.**  The radius of gyration is a fixed linear functional of the matrix
of squared interbead distances. -/
theorem rg2_eq_pairSum (s : Finset ℕ) (x : ℕ → E) :
    rg2 s x = pairSum s x / (2 * (s.card : ℝ) ^ 2) := by
  rcases Nat.eq_zero_or_pos s.card with h0 | h0
  · simp [rg2, pairSum, Finset.card_eq_zero.mp h0]
  · have hne : ((s.card : ℝ)) ≠ 0 := Nat.cast_ne_zero.mpr h0.ne'
    rw [rg2_eq, pairSum_eq]
    field_simp

/-- **The blind spot.**  Conformations with the same distance matrix on `s` have the same radius
of gyration; the map from structure to `Rg` factors through the distances and then collapses
`n(n-1)/2` numbers into one. -/
theorem rg2_congr_of_dist_eq {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    (s : Finset ℕ) (x : ℕ → E) (y : ℕ → F)
    (h : ∀ i ∈ s, ∀ j ∈ s, ‖x i - x j‖ = ‖y i - y j‖) :
    rg2 s x = rg2 s y := by
  rw [rg2_eq_pairSum, rg2_eq_pairSum, pairSum, pairSum]
  congr 1
  refine Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => ?_
  rw [h i hi j hj]

/-- Reflection through the origin is one instance of the blindness. -/
theorem rg2_neg (s : Finset ℕ) (x : ℕ → E) : rg2 s (fun i => -x i) = rg2 s x := by
  refine (rg2_congr_of_dist_eq s _ x fun i _ j _ => ?_)
  rw [show -x i - -x j = -(x i - x j) by abel, norm_neg]

/-- **The two-module (parallel-axis) law.**  For a chain split into two disjoint blocks -- a
folded domain and a disordered partner, say -- the global second moment is the sum of the block
second moments plus the squared separation of the block centroids, weighted by the reduced count
`n_A n_B / n`. -/
theorem rg2_union (s t : Finset ℕ) (x : ℕ → E) (hst : Disjoint s t)
    (hs : 0 < s.card) (ht : 0 < t.card) :
    ((s ∪ t).card : ℝ) * rg2 (s ∪ t) x
      = (s.card : ℝ) * rg2 s x + (t.card : ℝ) * rg2 t x
        + ((s.card : ℝ) * t.card / ((s.card : ℝ) + t.card)) * ‖cen s x - cen t x‖ ^ 2 := by
  have hcard : ((s ∪ t).card : ℝ) = (s.card : ℝ) + t.card := by
    rw [Finset.card_union_of_disjoint hst]; push_cast; ring
  have ha : ((s.card : ℝ)) ≠ 0 := Nat.cast_ne_zero.mpr hs.ne'
  have hb : ((t.card : ℝ)) ≠ 0 := Nat.cast_ne_zero.mpr ht.ne'
  have hsum : ∀ f : ℕ → ℝ, ∑ i ∈ s ∪ t, f i = (∑ i ∈ s, f i) + ∑ i ∈ t, f i :=
    fun f => Finset.sum_union hst
  have hvec : ∑ i ∈ s ∪ t, x i = (∑ i ∈ s, x i) + ∑ i ∈ t, x i := Finset.sum_union hst
  have hnorm1 : ‖(∑ i ∈ s, x i) + ∑ i ∈ t, x i‖ ^ 2
      = ‖∑ i ∈ s, x i‖ ^ 2 + 2 * (inner ℝ (∑ i ∈ s, x i) (∑ i ∈ t, x i) : ℝ)
        + ‖∑ i ∈ t, x i‖ ^ 2 := by
    rw [norm_add_sq_real]
  have hnorm2 : ‖cen s x - cen t x‖ ^ 2
      = ((s.card : ℝ)⁻¹) ^ 2 * ‖∑ i ∈ s, x i‖ ^ 2
        - 2 * ((s.card : ℝ))⁻¹ * ((t.card : ℝ))⁻¹
            * (inner ℝ (∑ i ∈ s, x i) (∑ i ∈ t, x i) : ℝ)
        + ((t.card : ℝ)⁻¹) ^ 2 * ‖∑ i ∈ t, x i‖ ^ 2 := by
    rw [cen, cen, norm_sub_sq_real, real_inner_smul_left, real_inner_smul_right, norm_smul,
      norm_smul]
    simp [mul_pow]
    ring_nf
  rw [rg2_eq, rg2_eq, rg2_eq, hcard, hsum, hvec, hnorm1, hnorm2]
  field_simp
  ring

/-- **A measured global radius of gyration caps the module separation.**  Since block radii are
nonnegative, the parallel-axis law turns into a one-sided experimental inequality: the squared
distance between the two block centroids is at most `n^2 / (n_A n_B)` times the global `Rg^2`. -/
theorem inter_module_dist_sq_le (s t : Finset ℕ) (x : ℕ → E) (hst : Disjoint s t)
    (hs : 0 < s.card) (ht : 0 < t.card) :
    ‖cen s x - cen t x‖ ^ 2
      ≤ (((s.card : ℝ) + t.card) ^ 2 / ((s.card : ℝ) * t.card)) * rg2 (s ∪ t) x := by
  have ha : (0 : ℝ) < (s.card : ℝ) := by exact_mod_cast hs
  have hb : (0 : ℝ) < (t.card : ℝ) := by exact_mod_cast ht
  have hcard : ((s ∪ t).card : ℝ) = (s.card : ℝ) + t.card := by
    rw [Finset.card_union_of_disjoint hst]; push_cast; ring
  have key := rg2_union s t x hst hs ht
  rw [hcard] at key
  have h1 : ((s.card : ℝ) * t.card / ((s.card : ℝ) + t.card)) * ‖cen s x - cen t x‖ ^ 2
      ≤ ((s.card : ℝ) + t.card) * rg2 (s ∪ t) x := by
    rw [key]
    have := mul_nonneg ha.le (rg2_nonneg s x)
    have := mul_nonneg hb.le (rg2_nonneg t x)
    linarith
  rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)] at h1
  rw [div_mul_eq_mul_div, le_div_iff₀ (by positivity)]
  nlinarith [h1, sq_nonneg (‖cen s x - cen t x‖)]

/-! ### Chain connectivity: the rod ceiling -/

omit [InnerProductSpace ℝ E] in
/-- Bonded beads: the distance from bead `i` to bead `j` grows at most linearly in `j - i`. -/
theorem norm_sub_le_bond (b : ℝ) (x : ℕ → E) (hb : ∀ k, ‖x (k + 1) - x k‖ ≤ b) :
    ∀ i j : ℕ, i ≤ j → ‖x j - x i‖ ≤ b * ((j : ℝ) - i) := by
  intro i j hij
  induction j with
  | zero =>
      have : i = 0 := Nat.le_zero.mp hij
      subst this; simp
  | succ m ih =>
      rcases Nat.lt_or_ge i (m + 1) with h | h
      · have hm : i ≤ m := Nat.lt_succ_iff.mp h
        have hstep := ih hm
        calc ‖x (m + 1) - x i‖ ≤ ‖x (m + 1) - x m‖ + ‖x m - x i‖ := by
              rw [show x (m + 1) - x i = (x (m + 1) - x m) + (x m - x i) by abel]
              exact norm_add_le _ _
          _ ≤ b + b * ((m : ℝ) - i) := add_le_add (hb m) hstep
          _ = b * (((m + 1 : ℕ) : ℝ) - i) := by push_cast; ring
      · have : i = m + 1 := le_antisymm hij h
        subst this; simp

omit [InnerProductSpace ℝ E] in
/-- Symmetric form of the bond bound. -/
theorem norm_sub_le_bond_abs (b : ℝ) (x : ℕ → E) (hb : ∀ k, ‖x (k + 1) - x k‖ ≤ b) (i j : ℕ) :
    ‖x i - x j‖ ≤ b * |(i : ℝ) - j| := by
  rcases le_total i j with h | h
  · have hle : ((i : ℝ)) ≤ j := by exact_mod_cast h
    rw [norm_sub_rev, abs_sub_comm, abs_of_nonneg (sub_nonneg.mpr hle)]
    exact norm_sub_le_bond b x hb i j h
  · have hle : ((j : ℝ)) ≤ i := by exact_mod_cast h
    rw [abs_of_nonneg (sub_nonneg.mpr hle)]
    exact norm_sub_le_bond b x hb j i h

/-- Gauss's sum, cast to the reals. -/
theorem sum_range_cast (n : ℕ) : ∑ i ∈ range n, (i : ℝ) = n * (n - 1) / 2 := by
  induction n with
  | zero => simp
  | succ m ih => rw [Finset.sum_range_succ, ih]; push_cast; ring

/-- The sum of squares, cast to the reals. -/
theorem sum_range_sq_cast (n : ℕ) :
    ∑ i ∈ range n, (i : ℝ) ^ 2 = n * (n - 1) * (2 * n - 1) / 6 := by
  induction n with
  | zero => simp
  | succ m ih => rw [Finset.sum_range_succ, ih]; push_cast; ring

/-- The double sum of squared index differences. -/
theorem sum_sq_index_diff (n : ℕ) :
    ∑ i ∈ range n, ∑ j ∈ range n, ((i : ℝ) - j) ^ 2 = (n : ℝ) ^ 2 * ((n : ℝ) ^ 2 - 1) / 6 := by
  have h := pairSum_eq (E := ℝ) (range n) (fun i => (i : ℝ))
  simp only [pairSum, Real.norm_eq_abs, sq_abs, Finset.card_range] at h
  rw [h, sum_range_sq_cast, sum_range_cast]
  ring

/-- **The rod ceiling.**  A chain whose consecutive beads are within `b` has
`Rg^2 <= b^2 (n^2 - 1)/12`. -/
theorem rg2_le_rod (b : ℝ) (x : ℕ → E) (hb : ∀ k, ‖x (k + 1) - x k‖ ≤ b) (n : ℕ) (hn : 0 < n) :
    rg2 (range n) x ≤ b ^ 2 * ((n : ℝ) ^ 2 - 1) / 12 := by
  have hb0 : 0 ≤ b := le_trans (norm_nonneg _) (hb 0)
  have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hps : pairSum (range n) x ≤ b ^ 2 * ((n : ℝ) ^ 2 * ((n : ℝ) ^ 2 - 1) / 6) := by
    rw [← sum_sq_index_diff n, Finset.mul_sum]
    refine Finset.sum_le_sum fun i _ => ?_
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun j _ => ?_
    have h1 : ‖x i - x j‖ ≤ b * |(i : ℝ) - j| := norm_sub_le_bond_abs b x hb i j
    have h2 : ‖x i - x j‖ ^ 2 ≤ (b * |(i : ℝ) - j|) ^ 2 :=
      pow_le_pow_left₀ (norm_nonneg _) h1 2
    calc ‖x i - x j‖ ^ 2 ≤ (b * |(i : ℝ) - j|) ^ 2 := h2
      _ = b ^ 2 * ((i : ℝ) - j) ^ 2 := by rw [mul_pow, sq_abs]
  rw [rg2_eq_pairSum]
  simp only [Finset.card_range]
  rw [div_le_div_iff₀ (by positivity) (by norm_num)]
  nlinarith [hps, sq_nonneg ((n : ℝ))]

/-! ### The rod attains the ceiling -/

/-- The straight rod with bond length `b`, living in `ℝ`. -/
noncomputable def rod (b : ℝ) : ℕ → ℝ := fun i => (i : ℝ) * b

theorem rod_bonds (b : ℝ) (hb : 0 ≤ b) (k : ℕ) : ‖rod b (k + 1) - rod b k‖ ≤ b := by
  simp [rod, Real.norm_eq_abs, add_mul, abs_of_nonneg hb]

/-- The rod's radius of gyration, exactly. -/
theorem rod_rg2 (b : ℝ) (n : ℕ) (hn : 0 < n) :
    rg2 (range n) (rod b) = b ^ 2 * ((n : ℝ) ^ 2 - 1) / 12 := by
  have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn
  have hps : pairSum (range n) (rod b) = b ^ 2 * ((n : ℝ) ^ 2 * ((n : ℝ) ^ 2 - 1) / 6) := by
    have hij : ∀ i ∈ range n, ∀ j ∈ range n,
        ‖rod b i - rod b j‖ ^ 2 = b ^ 2 * ((i : ℝ) - j) ^ 2 := by
      intro i _ j _
      simp only [rod, Real.norm_eq_abs, sq_abs]
      ring
    rw [pairSum, Finset.sum_congr rfl fun i hi =>
      Finset.sum_congr rfl fun j hj => hij i hi j hj]
    simp only [← Finset.mul_sum]
    rw [sum_sq_index_diff n]
  rw [rg2_eq_pairSum, hps]
  simp only [Finset.card_range]
  field_simp
  ring

/-- **The ceiling is sharp.**  The rod bound is attained, so `b^2 (n^2 - 1)/12` is the exact
supremum of `Rg^2` over bonded chains of `n` beads. -/
theorem rg2_rod_sharp (b : ℝ) (hb : 0 ≤ b) (n : ℕ) (hn : 0 < n) :
    rg2 (range n) (rod b) = b ^ 2 * ((n : ℝ) ^ 2 - 1) / 12 ∧
      (∀ k, ‖rod b (k + 1) - rod b k‖ ≤ b) :=
  ⟨rod_rg2 b n hn, rod_bonds b hb⟩

/-! ### The size-separation degeneracy -/

/-- A four-bead chain: two compact modules with centroids `2` apart. -/
noncomputable def chainSep : ℕ → ℝ := fun i => if i = 0 then -1 else if i = 1 then 1 else
  if i = 2 then 1 else 3

/-- A four-bead chain: one expanded module and one collapsed one, centroids coincident. -/
noncomputable def chainExp : ℕ → ℝ := fun i => if i = 0 then -2 else if i = 1 then 2 else 0

/-- **Size does not report architecture.**  Two four-bead chains have exactly the same global
`Rg^2`, while the first has compact modules held apart and the second has a fourfold expanded
module with coincident centroids.  Module separation and tail expansion trade off exactly at fixed
global radius of gyration, so no measurement of the global size can separate them. -/
theorem rg2_tradeoff :
    rg2 (range 4) chainSep = 2 ∧ rg2 (range 4) chainExp = 2 ∧
      rg2 {0, 1} chainSep = 1 ∧ rg2 {0, 1} chainExp = 4 ∧
      ‖cen {0, 1} chainSep - cen {2, 3} chainSep‖ = 2 ∧
      ‖cen ({0, 1} : Finset ℕ) chainExp - cen {2, 3} chainExp‖ = 0 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [rg2, cen, chainSep, chainExp, Finset.sum_range_succ, Real.norm_eq_abs]

end IDR.ChainGeo
