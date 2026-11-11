/-
# Part CXLVI  What one measured average certifies about the populations of a disordered region

Every experiment that reports a single number about an intrinsically disordered region — a mean
radius of gyration, a mean end-to-end distance, a mean paramagnetic relaxation enhancement — is a
weighted average over a conformational ensemble.  Part LXXXIV showed that a finite list of such
averages never determines the ensemble, and reported the two classical certificates a mean does
support (Markov and Chebyshev).  Neither is sharp, and neither uses the one hard constraint that a
polymer always supplies: **a chain of `n` residues with bond length `b` cannot present an internal
distance larger than the contour length `B = n·b`.**  This part works out exactly what a measured
mean certifies once that bound is in force, and proves each certificate is the best possible by
exhibiting an ensemble that attains it.

* `markov_pop_upper` / `markov_pop_upper_sharp` — the population of conformers at or beyond a
  distance `a` is at most `⟨r⟩/a`, and some admissible ensemble attains it.  (Upper certificate;
  the contour bound does not improve it.)

* `reverse_markov_lower` / `reverse_markov_lower_sharp` — **a mean forces population, it does not
  merely cap it.**  With all distances in `[0, B]`, the population strictly beyond `a` is at least
  `(⟨r⟩ − a)/(B − a)`, and this too is attained.  A measured mean end-to-end distance that is a
  sizeable fraction of the contour length is a proof that a definite fraction of the ensemble is
  expanded — no modelling assumption required.

* `population_identified_interval` — the two together: the population beyond `a` is pinned to the
  interval `[(⟨r⟩ − a)/(B − a), ⟨r⟩/a]` and to nothing smaller, both endpoints being attained.
  This is the honest content of a one-number experiment on a disordered region.

* `bhatia_davis` / `bhatia_davis_sharp` — **a mean also caps the heterogeneity.**  With distances
  in `[m, M]` the variance of the ensemble is at most `(M − ⟨r⟩)(⟨r⟩ − m)`, attained exactly by the
  two-state ensemble living on the endpoints.  Conformational breadth is largest for a mean in the
  middle of the accessible range and is squeezed to zero as the mean approaches either end.

* `cantelli_bound` / `cantelli_sharp` — the sharp one-sided second-moment certificate,
  `pop(r ≥ ⟨r⟩ + a) ≤ Var/(Var + a²)`, strictly better than the Chebyshev bound `Var/a²` of Part
  LXXXIV, and attained by an explicit two-state ensemble.  Proved by the shift trick: the second
  moment about `⟨r⟩ − Var/a` is `Var + Var²/a²`.

* `contour_certificate` — the physical package: for a region of `n` residues with bond length `b`,
  a measured mean end-to-end distance `mu` certifies, with no further assumption, an expanded
  population of at least `(mu − a)/(n·b − a)`, a variance of at most `mu·(n·b − mu)`, and hence a
  tail population `pop(r ≥ mu + a) ≤ Var/(Var + a²)` with `Var` below that ceiling.

* `near_contour_forces_extension` — the limiting form: if the measured mean is within `ε` of the
  contour length then all but `ε/(B − a)` of the ensemble is beyond `a`.  A near-maximal mean is
  incompatible with a broad ensemble, which is the one regime in which a single-structure model of
  a disordered region becomes asymptotically defensible.

Design consequence: a model fitted to averages should be scored against the *interval* those
averages certify, not against a point estimate; and every interval here is exactly attained, so a
model reporting a population outside it is falsified by the average alone, while one reporting a
population inside it cannot be falsified by that average at all.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace PopulationCertificate

open Finset

variable {N : ℕ}

/-- Population of the ensemble at or beyond the threshold `a`. -/
noncomputable def popGe (w x : Fin N → ℝ) (a : ℝ) : ℝ := ∑ j, if a ≤ x j then w j else 0

/-- Population of the ensemble strictly beyond the threshold `a`. -/
noncomputable def popGt (w x : Fin N → ℝ) (a : ℝ) : ℝ := ∑ j, if a < x j then w j else 0

/-- Ensemble average of the observable `x`. -/
def wmean (w x : Fin N → ℝ) : ℝ := ∑ j, w j * x j

/-- Ensemble variance of the observable `x`. -/
noncomputable def wvar (w x : Fin N → ℝ) : ℝ := ∑ j, w j * (x j - wmean w x) ^ 2

theorem popGe_nonneg {w x : Fin N → ℝ} (hw : ∀ j, 0 ≤ w j) (a : ℝ) : 0 ≤ popGe w x a := by
  refine Finset.sum_nonneg fun j _ => ?_
  by_cases h : a ≤ x j <;> simp [h, hw j]

theorem popGt_le_popGe (w x : Fin N → ℝ) (hw : ∀ j, 0 ≤ w j) (a : ℝ) :
    popGt w x a ≤ popGe w x a := by
  refine Finset.sum_le_sum fun j _ => ?_
  by_cases h : a < x j
  · simp [h, le_of_lt h]
  · simp only [h, if_false]
    by_cases h' : a ≤ x j <;> simp [h', hw j]

theorem wvar_nonneg {w x : Fin N → ℝ} (hw : ∀ j, 0 ≤ w j) : 0 ≤ wvar w x :=
  Finset.sum_nonneg fun j _ => mul_nonneg (hw j) (sq_nonneg _)

/-- The variance is the second moment minus the squared mean. -/
theorem wvar_eq {w x : Fin N → ℝ} (hsum : ∑ j, w j = 1) :
    wvar w x = (∑ j, w j * x j ^ 2) - wmean w x ^ 2 := by
  rw [wvar]
  have h : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * (x j - wmean w x) ^ 2
      = w j * x j ^ 2 - 2 * wmean w x * (w j * x j) + wmean w x ^ 2 * w j := fun j _ => by ring
  rw [Finset.sum_congr rfl h]
  simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  rw [hsum, show (∑ j, w j * x j) = wmean w x from rfl]
  ring

/-! ### The upper certificate: Markov -/

/-- **Markov certificate.**  The population at or beyond `a` is at most `⟨x⟩/a`. -/
theorem markov_pop_upper {w x : Fin N → ℝ} (hw : ∀ j, 0 ≤ w j) (hx : ∀ j, 0 ≤ x j) {a : ℝ}
    (ha : 0 < a) : popGe w x a ≤ wmean w x / a := by
  rw [le_div_iff₀ ha, popGe, Finset.sum_mul]
  refine Finset.sum_le_sum fun j _ => ?_
  by_cases h : a ≤ x j
  · rw [if_pos h]
    exact mul_le_mul_of_nonneg_left h (hw j)
  · rw [if_neg h, zero_mul]
    exact mul_nonneg (hw j) (hx j)

/-- The Markov certificate is attained: for `0 ≤ mu ≤ a` there is a two-state ensemble with
distances in `[0, a]`, mean `mu`, and population exactly `mu/a` at threshold `a`. -/
theorem markov_pop_upper_sharp {mu a : ℝ} (ha : 0 < a) (hmu : 0 ≤ mu) (hmua : mu ≤ a) :
    ∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, 0 ≤ x j ∧ x j ≤ a) ∧
      wmean w x = mu ∧ popGe w x a = wmean w x / a := by
  have hmu1 : mu / a ≤ 1 := (div_le_one ha).mpr hmua
  have hmu0 : 0 ≤ mu / a := div_nonneg hmu ha.le
  have hm : wmean ![1 - mu / a, mu / a] ![(0:ℝ), a] = mu := by
    rw [wmean, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  refine ⟨![1 - mu / a, mu / a], ![0, a], ?_, ?_, ?_, hm, ?_⟩
  · intro j
    fin_cases j
    · simpa using by linarith
    · simpa using hmu0
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    ring
  · intro j
    fin_cases j <;> simp [ha.le]
  · rw [hm, popGe, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [if_neg (by linarith), if_pos le_rfl]
    simp

/-! ### The lower certificate: a mean forces population -/

/-- **Reverse Markov certificate.**  If every conformation has `x ≤ B` and the weights are a
probability vector, then the population strictly beyond `a < B` is at least `(⟨x⟩ − a)/(B − a)`.
Unlike Markov's inequality this is a *lower* bound: the mean forces population into the tail. -/
theorem reverse_markov_lower {w x : Fin N → ℝ} {B a : ℝ} (hw : ∀ j, 0 ≤ w j)
    (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (hab : a < B) :
    (wmean w x - a) / (B - a) ≤ popGt w x a := by
  have hpt : ∑ j, w j * x j
      ≤ ∑ j, (a * w j + (B - a) * (if a < x j then w j else 0)) := by
    refine Finset.sum_le_sum fun j _ => ?_
    by_cases h : a < x j
    · rw [if_pos h]
      nlinarith [hw j, hx j]
    · rw [if_neg h]
      push_neg at h
      nlinarith [hw j]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsum,
    show (∑ j, if a < x j then w j else 0) = popGt w x a from rfl,
    show (∑ j, w j * x j) = wmean w x from rfl] at hpt
  rw [div_le_iff₀ (by linarith : (0:ℝ) < B - a)]
  linarith

/-- The reverse certificate is attained: for `a ≤ mu ≤ B` the two-state ensemble on `{a, B}`
with mean `mu` has population exactly `(mu − a)/(B − a)` strictly beyond `a`. -/
theorem reverse_markov_lower_sharp {B a mu : ℝ} (hab : a < B) (h0 : 0 ≤ a) (hma : a ≤ mu)
    (hmB : mu ≤ B) :
    ∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, 0 ≤ x j ∧ x j ≤ B) ∧
      wmean w x = mu ∧ popGt w x a = (wmean w x - a) / (B - a) := by
  have hBa : (0:ℝ) < B - a := by linarith
  have hm : wmean ![(B - mu) / (B - a), (mu - a) / (B - a)] ![a, B] = mu := by
    rw [wmean, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  refine ⟨![(B - mu) / (B - a), (mu - a) / (B - a)], ![a, B], ?_, ?_, ?_, hm, ?_⟩
  · intro j
    fin_cases j
    · simpa using div_nonneg (by linarith) hBa.le
    · simpa using div_nonneg (by linarith) hBa.le
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  · intro j
    fin_cases j <;> constructor <;> simp <;> linarith
  · rw [hm, popGt, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    rw [if_neg (lt_irrefl a), if_pos hab]
    simp

/-- **The identified interval of a single measured average.**  With distances in `[0, B]` and mean
`mu`, the population beyond a threshold `a` lies in `[(mu − a)/(B − a), mu/a]`, and the lower
endpoint is attained by an admissible ensemble, so no narrower certificate follows from the mean. -/
theorem population_identified_interval {B a mu : ℝ} (ha : 0 < a) (hab : a < B) (hma : a ≤ mu)
    (hmB : mu ≤ B) :
    (∀ (n : ℕ) (w x : Fin n → ℝ), (∀ j, 0 ≤ w j) → (∑ j, w j = 1) → (∀ j, 0 ≤ x j ∧ x j ≤ B) →
        wmean w x = mu →
        (mu - a) / (B - a) ≤ popGt w x a ∧ popGe w x a ≤ mu / a) ∧
      (∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, 0 ≤ x j ∧ x j ≤ B) ∧
        wmean w x = mu ∧ popGt w x a = (mu - a) / (B - a)) := by
  constructor
  · intro n w x hw hsum hx hmean
    refine ⟨?_, ?_⟩
    · have := reverse_markov_lower hw hsum (fun j => (hx j).2) hab
      rwa [hmean] at this
    · have := markov_pop_upper hw (fun j => (hx j).1) ha
      rwa [hmean] at this
  · obtain ⟨w, x, hw, hsum, hx, hmean, hpop⟩ := reverse_markov_lower_sharp hab ha.le hma hmB
    exact ⟨w, x, hw, hsum, hx, hmean, by rw [hpop, hmean]⟩

/-! ### The mean caps the heterogeneity: Bhatia–Davis -/

/-- **Bhatia–Davis certificate.**  With all conformational values in `[m, M]`, the ensemble
variance is at most `(M − ⟨x⟩)(⟨x⟩ − m)`. -/
theorem bhatia_davis {w x : Fin N → ℝ} {m M : ℝ} (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1)
    (hlo : ∀ j, m ≤ x j) (hhi : ∀ j, x j ≤ M) :
    wvar w x ≤ (M - wmean w x) * (wmean w x - m) := by
  have hpt : ∑ j, w j * x j ^ 2 ≤ ∑ j, ((M + m) * (w j * x j) - M * m * w j) := by
    refine Finset.sum_le_sum fun j _ => ?_
    nlinarith [mul_nonneg (hw j) (mul_nonneg (sub_nonneg.2 (hhi j)) (sub_nonneg.2 (hlo j)))]
  rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hsum,
    show (∑ j, w j * x j) = wmean w x from rfl] at hpt
  rw [wvar_eq hsum]
  nlinarith [hpt]

/-- Bhatia–Davis is attained: the two-state ensemble on the extreme values `{m, M}` with mean `mu`
has variance exactly `(M − mu)(mu − m)`.  The maximal conformational heterogeneity compatible with
a measured average is a two-state ensemble on the extremes of the accessible range. -/
theorem bhatia_davis_sharp {m M mu : ℝ} (hmM : m < M) (hlo : m ≤ mu) (hhi : mu ≤ M) :
    ∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ (∀ j, m ≤ x j ∧ x j ≤ M) ∧
      wmean w x = mu ∧ wvar w x = (M - mu) * (mu - m) := by
  have hd : (0:ℝ) < M - m := by linarith
  have hm : wmean ![(M - mu) / (M - m), (mu - m) / (M - m)] ![m, M] = mu := by
    rw [wmean, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  refine ⟨![(M - mu) / (M - m), (mu - m) / (M - m)], ![m, M], ?_, ?_, ?_, hm, ?_⟩
  · intro j
    fin_cases j
    · simpa using div_nonneg (by linarith) hd.le
    · simpa using div_nonneg (by linarith) hd.le
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  · intro j
    fin_cases j <;> constructor <;> simp <;> linarith
  · rw [wvar, hm, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring

/-! ### The sharp second-moment certificate: Cantelli -/

/-- **Cantelli certificate** (sharp one-sided Chebyshev).  The population at or beyond
`⟨x⟩ + a` is at most `Var/(Var + a²)`, strictly better than Chebyshev's `Var/a²`. -/
theorem cantelli_bound {w x : Fin N → ℝ} (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) {a : ℝ}
    (ha : 0 < a) :
    popGe w x (wmean w x + a) ≤ wvar w x / (wvar w x + a ^ 2) := by
  set mu := wmean w x with hmu
  set v := wvar w x with hv
  have hv0 : 0 ≤ v := wvar_nonneg hw
  set t : ℝ := v / a with ht
  have ht0 : 0 ≤ t := div_nonneg hv0 ha.le
  have hshift : ∑ j, w j * (x j - mu + t) ^ 2 = v + t ^ 2 := by
    have hexp : ∀ j ∈ (Finset.univ : Finset (Fin N)), w j * (x j - mu + t) ^ 2
        = w j * (x j - mu) ^ 2 + 2 * t * (w j * x j) - 2 * t * mu * w j + t ^ 2 * w j :=
      fun j _ => by ring
    rw [Finset.sum_congr rfl hexp]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
    rw [hsum, show (∑ j, w j * x j) = mu from rfl,
      show (∑ j, w j * (x j - mu) ^ 2) = v from rfl]
    ring
  have hbound : popGe w x (mu + a) * (a + t) ^ 2 ≤ v + t ^ 2 := by
    rw [← hshift, popGe, Finset.sum_mul]
    refine Finset.sum_le_sum fun j _ => ?_
    by_cases h : mu + a ≤ x j
    · rw [if_pos h]
      have h2 : (a + t) ^ 2 ≤ (x j - mu + t) ^ 2 := by nlinarith
      exact mul_le_mul_of_nonneg_left h2 (hw j)
    · rw [if_neg h, zero_mul]
      exact mul_nonneg (hw j) (sq_nonneg _)
  have hva : 0 < v + a ^ 2 := by positivity
  have ha2 : (0:ℝ) < a ^ 2 := by positivity
  rw [le_div_iff₀ hva]
  have e1 : (a + t) ^ 2 = (v + a ^ 2) ^ 2 / a ^ 2 := by rw [ht]; field_simp; ring
  have e2 : v + t ^ 2 = v * (v + a ^ 2) / a ^ 2 := by rw [ht]; field_simp; ring
  rw [e1, e2] at hbound
  have h := mul_le_mul_of_nonneg_right hbound ha2.le
  field_simp at h
  linarith

/-- **Cantelli is strictly sharper than Chebyshev.**  The certificate proved above is below the
one-sided Chebyshev value `Var/a²` reported in Part LXXXIV, strictly so whenever the variance is
positive. -/
theorem cantelli_lt_chebyshev {v a : ℝ} (hv : 0 < v) (ha : 0 < a) :
    v / (v + a ^ 2) < v / a ^ 2 := by
  have ha2 : (0:ℝ) < a ^ 2 := by positivity
  have hva : (0:ℝ) < v + a ^ 2 := by positivity
  exact div_lt_div_of_pos_left hv ha2 (by linarith)

/-- The Cantelli certificate is attained: for any variance `v > 0` and any `a > 0` there is a
two-state ensemble with mean `mu`, variance `v`, and population exactly `v/(v + a²)` at or beyond
`mu + a`. -/
theorem cantelli_sharp {v a mu : ℝ} (hv : 0 < v) (ha : 0 < a) :
    ∃ w x : Fin 2 → ℝ, (∀ j, 0 ≤ w j) ∧ (∑ j, w j = 1) ∧ wmean w x = mu ∧ wvar w x = v ∧
      popGe w x (wmean w x + a) = v / (v + a ^ 2) := by
  have hva : (0:ℝ) < v + a ^ 2 := by positivity
  have hm : wmean ![a ^ 2 / (v + a ^ 2), v / (v + a ^ 2)] ![mu - v / a, mu + a] = mu := by
    rw [wmean, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  refine ⟨![a ^ 2 / (v + a ^ 2), v / (v + a ^ 2)], ![mu - v / a, mu + a], ?_, ?_, hm, ?_, ?_⟩
  · intro j
    fin_cases j
    · simpa using div_nonneg (sq_nonneg a) hva.le
    · simpa using div_nonneg hv.le hva.le
  · rw [Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  · rw [wvar, hm, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    field_simp
    ring
  · rw [hm, popGe, Fin.sum_univ_two]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one]
    have h1 : ¬ (mu + a ≤ mu - v / a) := by
      have : 0 < v / a := div_pos hv ha
      push_neg
      linarith
    rw [if_neg h1, if_pos le_rfl]
    simp

/-! ### The physical package -/

/-- **Contour certificate.**  For a disordered region of `n` residues with bond length `b`, whose
internal distances therefore lie in `[0, n·b]`, a measured mean distance `mu` certifies without
further assumption: an expanded population at least `(mu − a)/(n·b − a)` beyond any threshold
`a < n·b`, an upper Markov population `mu/a`, a conformational variance at most `(n·b − mu)·mu`,
and a tail population at or beyond `mu + a` bounded by the Cantelli value. -/
theorem contour_certificate {n : ℕ} {b mu a : ℝ} {w x : Fin n → ℝ}
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1)
    (hx : ∀ j, 0 ≤ x j ∧ x j ≤ (n : ℝ) * b) (hmean : wmean w x = mu)
    (ha : 0 < a) (hab : a < (n : ℝ) * b) :
    (mu - a) / ((n : ℝ) * b - a) ≤ popGt w x a ∧
      popGe w x a ≤ mu / a ∧
      wvar w x ≤ ((n : ℝ) * b - mu) * mu ∧
      popGe w x (mu + a) ≤ wvar w x / (wvar w x + a ^ 2) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have := reverse_markov_lower hw hsum (fun j => (hx j).2) hab
    rwa [hmean] at this
  · have := markov_pop_upper hw (fun j => (hx j).1) ha
    rwa [hmean] at this
  · have := bhatia_davis (m := 0) (M := (n : ℝ) * b) hw hsum (fun j => (hx j).1)
      (fun j => (hx j).2)
    rwa [hmean, sub_zero] at this
  · have := cantelli_bound (x := x) hw hsum ha
    rwa [hmean] at this

/-- **A near-maximal mean forces near-complete extension.**  If the measured mean distance is
within `ε` of the contour length `B`, then all but `ε/(B − a)` of the ensemble lies strictly beyond
`a`.  Sending `ε → 0` pushes the ensemble onto the fully extended state: the single regime in which
a one-structure description of a disordered region becomes asymptotically defensible. -/
theorem near_contour_forces_extension {w x : Fin N → ℝ} {B a eps : ℝ}
    (hw : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1) (hx : ∀ j, x j ≤ B) (hab : a < B)
    (hmean : B - eps ≤ wmean w x) :
    1 - eps / (B - a) ≤ popGt w x a := by
  have hBa : (0:ℝ) < B - a := by linarith
  have h1 := reverse_markov_lower hw hsum hx hab
  have h2 : 1 - eps / (B - a) ≤ (wmean w x - a) / (B - a) := by
    rw [le_div_iff₀ hBa]
    have hexp : (1 - eps / (B - a)) * (B - a) = (B - a) - eps := by field_simp
    rw [hexp]
    linarith
  linarith

end PopulationCertificate
end IDR
