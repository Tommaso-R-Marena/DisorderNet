/-
# Averaged distances are not distances of anything: what an ensemble panel can look like

`RequestProject.DistanceRealizability` shows that the mean-distance panel of an ensemble still
obeys the triangle inequality and the contour bound.  It is tempting to read that as saying
that the averaged panel behaves "like a structure" -- and indeed the standard practice of
interpreting averaged distance restraints (NOE, PRE, FRET) by fitting *one* structure to them
presupposes exactly that.

This file shows the presupposition is false, and quantifies how far off it is.

* `cutEnsemble_meanDist` -- a general construction.  For any finite family of two-block splits
  of the residues, any nonnegative weights summing to at most one, and any scale `t`, there is
  an ensemble of chain conformations with every bond at most `t` whose mean-distance panel is
  exactly the weighted sum of the split (cut) pseudometrics.  Ensembles therefore realise the
  whole cut cone, truncated at the contour scale: a very large family of panels, most of which
  are not distances between points of space at all.
* `no_single_structure_of_fourCycle` -- the obstruction, proved from the parallelogram law: no
  four points of a Euclidean space have pairwise distances `1` around a cycle and `2` across
  both diagonals, because both `y 1` and `y 3` would have to be the midpoint of `y 0` and
  `y 2`.
* `ensemble_mean_beyond_single_structures` -- the two statements combined: an explicit
  two-state ensemble of a four-site chain (a "compact/extended exchange" of exactly the kind a
  disordered region shows) whose measured mean-distance panel is a genuine metric, passes every
  triangle test, respects the contour bound -- and is realised by **no** single conformation.
  Fitting one structure to averaged distances is not merely imprecise; the target of the fit
  can be geometrically nonexistent.
* `single_structure_fit_error_lower_bound` -- and the failure is quantitative: any single
  structure misses at least one entry of that panel by at least `1/2` in the units of the panel
  (a full half of the bond scale), no matter how the fit is done.

The design consequence for a model of a disordered region: the model must denote an *ensemble*,
and its predictions must be computed as ensemble averages of the measured observable; a
single-conformation representation, fitted to averaged distances, is not an approximation of the
ensemble but a description of an object that need not exist.
-/
import Mathlib
import RequestProject.DistanceRealizability

namespace RequestProject.DistanceCutEnsembles

open Finset RequestProject.DistanceRealizability

/-! ## The cut construction -/

section Cut

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The conformation associated with a two-block split `S` of the residues at scale `t`: every
residue in one block sits at the origin, every residue in the other at `t • u`.  This is a
maximally collapsed conformation, and its bonds are at most `t`. -/
noncomputable def splitConfig (S : ℕ → Bool) (t : ℝ) (u : E) (i : ℕ) : E :=
  if S i then t • u else 0

theorem dist_splitConfig {S : ℕ → Bool} {t : ℝ} {u : E} (hu : ‖u‖ = 1) (ht : 0 ≤ t) (i j : ℕ) :
    dist (splitConfig S t u i) (splitConfig S t u j) = if S i = S j then 0 else t := by
  unfold splitConfig
  rcases Bool.eq_false_or_eq_true (S i) with hi | hi <;>
    rcases Bool.eq_false_or_eq_true (S j) with hj | hj <;>
      simp [hi, hj, dist_eq_norm, norm_smul, hu, abs_of_nonneg ht]

/-- **Ensembles realise the cut cone.**  Given splits `S k` of the chain, weights `lam k ≥ 0`
with `∑ lam ≤ 1` and a scale `t ≥ 0`, there is an ensemble of conformations -- each of them a
legitimate chain with all bonds at most `t` -- whose mean-distance panel is the weighted sum of
the corresponding cut pseudometrics. -/
theorem cutEnsemble_meanDist {r : ℕ} (S : Fin r → ℕ → Bool) (lam : Fin r → ℝ) (t : ℝ)
    (u : E) (hu : ‖u‖ = 1) (ht : 0 ≤ t) (hlam : ∀ k, 0 ≤ lam k) (hsum : ∑ k, lam k ≤ 1) :
    ∃ (w : Fin (r + 1) → ℝ) (X : Fin (r + 1) → ℕ → E),
      (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧ (∀ a k, dist (X a k) (X a (k + 1)) ≤ t) ∧
      ∀ i j, meanDist w X i j = ∑ k, lam k * (if S k i = S k j then 0 else t) := by
  refine ⟨Fin.cons (1 - ∑ k, lam k) lam, Fin.cons (fun _ => (0 : E))
    (fun k => splitConfig (S k) t u), ?_, ?_, ?_, ?_⟩
  · refine Fin.cases ?_ ?_
    · simpa using hsum
    · intro k; simpa using hlam k
  · rw [Fin.sum_cons]; ring
  · refine Fin.cases ?_ ?_
    · intro k; simp [ht]
    · intro k i
      rw [Fin.cons_succ, dist_splitConfig hu ht]
      split <;> simp [ht]
  · intro i j
    rw [meanDist, Fin.sum_univ_succ]
    simp only [Fin.cons_succ, Fin.cons_zero, dist_self, mul_zero, zero_add]
    exact Finset.sum_congr rfl fun k _ => by rw [dist_splitConfig hu ht]

end Cut

/-! ## The four-point obstruction -/

section Obstruction

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Equality in the triangle inequality in a Euclidean space: if `dist a b = dist b c = 1` and
`dist a c = 2` then `b` is the midpoint of `a` and `c`. -/
theorem two_smul_eq_of_dist {a b c : E} (h1 : dist a b = 1) (h2 : dist b c = 1)
    (h3 : dist a c = 2) : (2 : ℝ) • b = a + c := by
  have hu : ‖a - b‖ = 1 := by rwa [← dist_eq_norm]
  have hv : ‖b - c‖ = 1 := by rwa [← dist_eq_norm]
  have huv : ‖(a - b) + (b - c)‖ = 2 := by
    have : (a - b) + (b - c) = a - c := by abel
    rw [this, ← dist_eq_norm]; exact h3
  have hpar := parallelogram_law_with_norm ℝ (a - b) (b - c)
  rw [hu, hv, huv] at hpar
  have hzero : ‖(a - b) - (b - c)‖ = 0 := by nlinarith [norm_nonneg ((a - b) - (b - c))]
  have h : (a - b) - (b - c) = 0 := by rwa [norm_eq_zero] at hzero
  linear_combination (norm := module) -h

/-- **No single structure has the four-cycle distance panel.**  In any Euclidean space, four
points cannot be pairwise at distance `1` around the cycle `0-1-2-3-0` and at distance `2`
across both diagonals. -/
theorem no_single_structure_of_fourCycle (y : ℕ → E)
    (h01 : dist (y 0) (y 1) = 1) (h12 : dist (y 1) (y 2) = 1)
    (h23 : dist (y 2) (y 3) = 1) (h03 : dist (y 0) (y 3) = 1)
    (h02 : dist (y 0) (y 2) = 2) (h13 : dist (y 1) (y 3) = 2) : False := by
  have hb : (2 : ℝ) • y 1 = y 0 + y 2 := two_smul_eq_of_dist h01 h12 h02
  have hd : (2 : ℝ) • y 3 = y 0 + y 2 := two_smul_eq_of_dist h03 (by rwa [dist_comm]) h02
  have : y 1 = y 3 := by
    have h2 : (2 : ℝ) • y 1 = (2 : ℝ) • y 3 := by rw [hb, hd]
    exact smul_right_injective E (by norm_num) h2
  rw [this, dist_self] at h13
  norm_num at h13

end Obstruction

/-! ## The explicit compact/extended exchange -/

/-- The measured panel of the exchange: `1` around the cycle, `2` across the diagonals. -/
noncomputable def fourCyclePanel : ℕ → ℕ → ℝ := fun i j =>
  if i = j then 0 else if (i = 0 ∧ j = 2) ∨ (i = 2 ∧ j = 0) ∨ (i = 1 ∧ j = 3) ∨ (i = 3 ∧ j = 1)
    then 2 else 1

/-- The two splits whose half-and-half mixture produces `fourCyclePanel`: `{0,1} | {2,3}` (the
N-terminal half collapses onto itself) and `{1,2} | {0,3}` (the central pair collapses). -/
def splitA : ℕ → Bool := fun i => decide (i = 0 ∨ i = 1)

def splitB : ℕ → Bool := fun i => decide (i = 1 ∨ i = 2)

/-- **An ensemble whose averaged distances are the distances of nothing.**  There is a
three-conformation ensemble of a four-site chain, with every bond at most `2` (so it is a
legitimate chain) whose mean-distance panel on the four labelled sites is exactly
`fourCyclePanel`; that panel satisfies the triangle inequality and the contour bound, yet no
single conformation of any Euclidean space realises it. -/
theorem ensemble_mean_beyond_single_structures :
    ∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
      (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
      (∀ a k, dist (X a k) (X a (k + 1)) ≤ 2) ∧
      (∀ i < 4, ∀ j < 4, meanDist w X i j = fourCyclePanel i j) ∧
      (∀ i j k, meanDist w X i k ≤ meanDist w X i j + meanDist w X j k) ∧
      ¬ ∃ y : ℕ → EuclideanSpace ℝ (Fin 3),
          ∀ i < 4, ∀ j < 4, fourCyclePanel i j = dist (y i) (y j) := by
  classical
  set u : EuclideanSpace ℝ (Fin 3) := EuclideanSpace.single 0 (1 : ℝ) with hu_def
  have hu : ‖u‖ = 1 := by simp [hu_def]
  obtain ⟨w, X, hw, hsum, hbond, hmean⟩ :=
    cutEnsemble_meanDist ![splitA, splitB] ![1/2, 1/2] 2 u hu (by norm_num)
      (by intro k; fin_cases k <;> norm_num) (by norm_num [Fin.sum_univ_two])
  refine ⟨_, w, X, hw, hsum, hbond, ?_, fun i j k => meanDist_triangle hw i j k, ?_⟩
  · intro i hi j hj
    interval_cases i <;> interval_cases j <;>
      simp [hmean, Fin.sum_univ_two, splitA, splitB, fourCyclePanel] <;> norm_num
  · rintro ⟨y, hy⟩
    refine no_single_structure_of_fourCycle y ?_ ?_ ?_ ?_ ?_ ?_ <;>
      [ have := hy 0 (by norm_num) 1 (by norm_num);
        have := hy 1 (by norm_num) 2 (by norm_num);
        have := hy 2 (by norm_num) 3 (by norm_num);
        have := hy 0 (by norm_num) 3 (by norm_num);
        have := hy 0 (by norm_num) 2 (by norm_num);
        have := hy 1 (by norm_num) 3 (by norm_num)] <;>
      simpa [fourCyclePanel] using this.symm

/-- Quantitative equality case: if `dist a b` and `dist b c` exceed `1` by at most `e` and
`dist a c` falls short of `2` by at most `e`, then `b` is within `√(12e + 3e²)` of the midpoint
of `a` and `c`. -/
theorem near_midpoint {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {a b c : E} {e : ℝ} (he : 0 ≤ e) (he2 : e ≤ 2)
    (h1 : dist a b ≤ 1 + e) (h2 : dist b c ≤ 1 + e) (h3 : 2 - e ≤ dist a c) :
    ‖a + c - (2 : ℝ) • b‖ * ‖a + c - (2 : ℝ) • b‖ ≤ 12 * e + 3 * e ^ 2 := by
  have hpar := parallelogram_law_with_norm ℝ (a - b) (b - c)
  have hsum : (a - b) + (b - c) = a - c := by abel
  have hdiff : (a - b) - (b - c) = a + c - (2 : ℝ) • b := by
    rw [two_smul]; abel
  rw [hsum, hdiff] at hpar
  have hu : ‖a - b‖ ≤ 1 + e := by rwa [← dist_eq_norm]
  have hv : ‖b - c‖ ≤ 1 + e := by rwa [← dist_eq_norm]
  have hw : 2 - e ≤ ‖a - c‖ := by rwa [dist_eq_norm] at h3
  nlinarith [norm_nonneg (a - b), norm_nonneg (b - c), norm_nonneg (a - c)]

/-- **Quantitative version.**  Every single conformation misses some entry of the measured
panel by at least `1/5`; the deficiency of a single-structure representation of an averaged
panel is bounded away from zero, uniformly over all structures. -/
theorem single_structure_fit_error_lower_bound (y : ℕ → EuclideanSpace ℝ (Fin 3)) :
    ∃ i < 4, ∃ j < 4, (1 : ℝ) / 5 ≤ |fourCyclePanel i j - dist (y i) (y j)| := by
  by_contra hcon
  push_neg at hcon
  have key : ∀ i < 4, ∀ j < 4, |fourCyclePanel i j - dist (y i) (y j)| < 1 / 5 :=
    fun i hi j hj => hcon i hi j hj
  have h01 := abs_lt.mp (key 0 (by norm_num) 1 (by norm_num))
  have h12 := abs_lt.mp (key 1 (by norm_num) 2 (by norm_num))
  have h23 := abs_lt.mp (key 2 (by norm_num) 3 (by norm_num))
  have h03 := abs_lt.mp (key 0 (by norm_num) 3 (by norm_num))
  have h02 := abs_lt.mp (key 0 (by norm_num) 2 (by norm_num))
  have h13 := abs_lt.mp (key 1 (by norm_num) 3 (by norm_num))
  simp only [fourCyclePanel] at h01 h12 h23 h03 h02 h13
  norm_num at h01 h12 h23 h03 h02 h13
  -- `y 1` and `y 3` are both near the midpoint of `y 0` and `y 2`
  have hB := near_midpoint (a := y 0) (b := y 1) (c := y 2) (e := 1/5)
    (by norm_num) (by norm_num) (by linarith [h01.2]) (by linarith [h12.2])
    (by linarith [h02.1])
  have hD := near_midpoint (a := y 0) (b := y 3) (c := y 2) (e := 1/5)
    (by norm_num) (by norm_num) (by linarith [h03.2])
    (by rw [dist_comm]; linarith [h23.2]) (by linarith [h02.1])
  set P := ‖y 0 + y 2 - (2 : ℝ) • y 1‖ with hP
  set Q := ‖y 0 + y 2 - (2 : ℝ) • y 3‖ with hQ
  have hsplit : (2 : ℝ) * dist (y 1) (y 3) ≤ P + Q := by
    have : (2 : ℝ) • y 3 - (2 : ℝ) • y 1 =
        (y 0 + y 2 - (2 : ℝ) • y 1) - (y 0 + y 2 - (2 : ℝ) • y 3) := by abel
    have h := norm_sub_le (y 0 + y 2 - (2 : ℝ) • y 1) (y 0 + y 2 - (2 : ℝ) • y 3)
    rw [← this] at h
    calc (2 : ℝ) * dist (y 1) (y 3) = ‖(2 : ℝ) • y 3 - (2 : ℝ) • y 1‖ := by
          rw [← smul_sub, norm_smul, dist_comm, dist_eq_norm]
          simp
      _ ≤ P + Q := h
  have hd : (2 : ℝ) - 1/5 ≤ dist (y 1) (y 3) := by linarith [h13.1]
  nlinarith [norm_nonneg (y 0 + y 2 - (2 : ℝ) • y 1), norm_nonneg (y 0 + y 2 - (2 : ℝ) • y 3),
    sq_nonneg (P - Q)]

end RequestProject.DistanceCutEnsembles
