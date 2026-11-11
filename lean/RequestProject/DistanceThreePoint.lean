/-
# The three-site test is complete

`RequestProject.DistanceRealizability` gives a necessary condition on a measured panel of mean
distances -- the triangle inequality -- and `RequestProject.DistanceCutEnsembles` shows that
passing it does not deliver a structure.  What is the *exact* strength of the test?

For three labelled sites, this file answers: the triangle inequality is not merely necessary, it
is the whole story.

* `threePointEnsemble_meanDist` -- given three nonnegative numbers satisfying the three triangle
  inequalities, an explicit three-conformation ensemble whose mean-distance panel is exactly
  those numbers.  The construction is the classical decomposition of a three-point metric into
  the three one-versus-rest cuts, with the cut weights `(p + q - r)/2` read directly off the
  measured panel; each conformation is a collapse in which one site is displaced and the other
  two coincide -- exactly the picture of a disordered region sampling three interconverting
  local arrangements.
* `three_point_feasible_iff` -- consequently, a three-site panel is the mean-distance panel of
  some ensemble **if and only if** it is nonnegative, symmetric and satisfies the triangle
  inequality.

Two consequences for design.  First, the triangle test is exactly the right test at three sites:
a panel that passes it cannot be refuted by any further geometric argument, so refuting a model
there requires distributional data, not more geometry.  Second, the "structure" that a
three-site panel appears to determine is an illusion of small numbers: the realising ensemble is
a three-state exchange, not a triangle of fixed positions, and by
`RequestProject.DistanceCutEnsembles` the illusion breaks at four sites.
-/
import Mathlib
import RequestProject.DistanceRealizability

namespace RequestProject.DistanceThreePoint

open Finset RequestProject.DistanceRealizability

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The conformation in which site `k` is displaced by `s` along `u` and the other sites sit at
the origin. -/
noncomputable def spike (k : ℕ) (s : ℝ) (u : E) (i : ℕ) : E := if i = k then s • u else 0

theorem dist_spike {k : ℕ} {s : ℝ} {u : E} (hu : ‖u‖ = 1) (hs : 0 ≤ s) {i j : ℕ} (hij : i ≠ j) :
    dist (spike k s u i) (spike k s u j) = if i = k ∨ j = k then s else 0 := by
  unfold spike
  by_cases hi : i = k <;> by_cases hj : j = k
  · exact absurd (hi.trans hj.symm) hij
  all_goals simp [hi, hj, dist_eq_norm, norm_smul, hu, abs_of_nonneg hs]

/-- **Every three-site metric panel is an ensemble average.**  Given measured distances `p`
(sites `0,1`), `q` (sites `0,2`) and `r` (sites `1,2`) obeying the triangle inequality, the
three-state exchange with weights `1/3` and cut amplitudes `3·(p+q-r)/2`, `3·(p+r-q)/2`,
`3·(q+r-p)/2` reproduces the panel exactly. -/
theorem threePointEnsemble_meanDist (u : E) (hu : ‖u‖ = 1) {p q r : ℝ}
    (h0 : r ≤ p + q) (h1 : q ≤ p + r) (h2 : p ≤ q + r) :
    ∃ (w : Fin 3 → ℝ) (X : Fin 3 → ℕ → E),
      (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
      meanDist w X 0 1 = p ∧ meanDist w X 0 2 = q ∧ meanDist w X 1 2 = r := by
  classical
  set l0 : ℝ := (p + q - r) / 2 with hl0
  set l1 : ℝ := (p + r - q) / 2 with hl1
  set l2 : ℝ := (q + r - p) / 2 with hl2
  have hl0n : 0 ≤ l0 := by rw [hl0]; linarith
  have hl1n : 0 ≤ l1 := by rw [hl1]; linarith
  have hl2n : 0 ≤ l2 := by rw [hl2]; linarith
  refine ⟨fun _ => 1 / 3, ![spike 0 (3 * l0) u, spike 1 (3 * l1) u, spike 2 (3 * l2) u],
    fun _ => by norm_num, by simp, ?_, ?_, ?_⟩
  · rw [meanDist, Fin.sum_univ_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.tail_cons, Matrix.head_cons]
    rw [dist_spike hu (by linarith) (by norm_num : (0:ℕ) ≠ 1),
      dist_spike hu (by linarith) (by norm_num : (0:ℕ) ≠ 1),
      dist_spike hu (by linarith) (by norm_num : (0:ℕ) ≠ 1)]
    norm_num
    rw [hl0, hl1]
    ring
  · rw [meanDist, Fin.sum_univ_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.tail_cons, Matrix.head_cons]
    rw [dist_spike hu (by linarith) (by norm_num : (0:ℕ) ≠ 2),
      dist_spike hu (by linarith) (by norm_num : (0:ℕ) ≠ 2),
      dist_spike hu (by linarith) (by norm_num : (0:ℕ) ≠ 2)]
    norm_num
    rw [hl0, hl2]
    ring
  · rw [meanDist, Fin.sum_univ_three]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.tail_cons, Matrix.head_cons]
    rw [dist_spike hu (by linarith) (by norm_num : (1:ℕ) ≠ 2),
      dist_spike hu (by linarith) (by norm_num : (1:ℕ) ≠ 2),
      dist_spike hu (by linarith) (by norm_num : (1:ℕ) ≠ 2)]
    norm_num
    rw [hl1, hl2]
    ring

/-- **The three-site feasibility test is exactly the triangle inequality.**  A panel of three
measured mean distances is realised by some ensemble if and only if it satisfies the three
triangle inequalities (which already force the three numbers to be nonnegative). -/
theorem three_point_feasible_iff {p q r : ℝ} :
    (∃ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)),
        (∀ a, 0 ≤ w a) ∧ (∑ a, w a = 1) ∧
        meanDist w X 0 1 = p ∧ meanDist w X 0 2 = q ∧ meanDist w X 1 2 = r)
      ↔ (r ≤ p + q ∧ q ≤ p + r ∧ p ≤ q + r) := by
  constructor
  · rintro ⟨m, w, X, hw, hsum, hp, hq, hr⟩
    refine ⟨?_, ?_, ?_⟩
    · have := meanDist_triangle (X := X) hw 1 0 2
      rw [hr, meanDist_comm (w := w) (X := X) 1 0, hp, hq] at this
      exact this
    · have := meanDist_triangle (X := X) hw 0 1 2
      rw [hq, hp, hr] at this
      exact this
    · have := meanDist_triangle (X := X) hw 0 2 1
      rw [hp, hq, meanDist_comm (w := w) (X := X) 2 1, hr] at this
      exact this
  · rintro ⟨h0, h1, h2⟩
    obtain ⟨w, X, hw, hsum, hp, hq, hr⟩ :=
      threePointEnsemble_meanDist (EuclideanSpace.single (0 : Fin 3) (1 : ℝ)) (by simp) h0 h1 h2
    exact ⟨3, w, X, hw, hsum, hp, hq, hr⟩

end RequestProject.DistanceThreePoint
