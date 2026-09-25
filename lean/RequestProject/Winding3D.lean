/-
# The wrapping budget for a chain in three dimensions

`RequestProject.Winding` proved the geometry of threading for a chain in the plane transverse to
an axis.  A real disordered region lives in space, and its residues move along the axis as well
as around it.  This file shows that nothing is lost: the whole budget, and topological
protection, transfer verbatim to a chain in `ℝ³` wrapping the `z` axis, because projecting onto
the transverse plane can only shorten bonds and cannot increase the distance to the axis.

* `transverse` -- the transverse component of a point of `ℝ³`, as a complex number, and
  `winding3` -- the number of turns a chain of residues in space makes around the `z` axis.
* `norm_transverse_sub_le` -- projection is 1-Lipschitz: a bond of length `b` in space projects
  to a step of length at most `b`.
* `norm_transverse_le_dist_axis` -- `‖transverse u‖` really is the distance from `u` to the
  axis, so the exclusion hypothesis is the physical one: no residue enters the cylinder of
  radius `R` around the partner rod, pore or filament.
* `abs_winding3_le` -- **the wrapping budget in space**: `|winding3| ≤ n b / (4 R)`.
* `length_demand3` -- `k` turns of threading require at least `4 k R / b` residues.
* `winding3_invariant_of_deformation` -- topological protection in space: a continuous motion of
  a closed chain that never enters the exclusion cylinder and never stretches a bond past `2 R`
  cannot change the number of turns.
-/
import Mathlib
import RequestProject.Winding

namespace RequestProject.Winding3D

open RequestProject.Winding

/-- The transverse component of a point of `ℝ³`, read as a complex number: the position seen
looking down the axis. -/
noncomputable def transverse (u : EuclideanSpace ℝ (Fin 3)) : ℂ := ⟨u 0, u 1⟩

/-- The number of turns the chain `p` of residues in space makes around the `z` axis. -/
noncomputable def winding3 (p : ℕ → EuclideanSpace ℝ (Fin 3)) (n : ℕ) : ℝ :=
  winding (fun i => transverse (p i)) n

/-- Projecting onto the transverse plane cannot lengthen a bond. -/
theorem norm_transverse_sub_le (u v : EuclideanSpace ℝ (Fin 3)) :
    ‖transverse u - transverse v‖ ≤ ‖u - v‖ := by
  have h1 : ‖transverse u - transverse v‖ ^ 2 = (u 0 - v 0) ^ 2 + (u 1 - v 1) ^ 2 := by
    simp [transverse, Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im]
    ring
  have h2 : ‖u - v‖ ^ 2 = (u 0 - v 0) ^ 2 + (u 1 - v 1) ^ 2 + (u 2 - v 2) ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
    simp [Fin.sum_univ_three]
  nlinarith [norm_nonneg (transverse u - transverse v), norm_nonneg (u - v),
    sq_nonneg (u 2 - v 2)]

/-- `‖transverse u‖` is the distance from `u` to the axis: it is at most the distance from `u` to
every point of the axis, and it is attained.  So "the residue stays at distance at least `R` from
the axis" is exactly the hypothesis `R ≤ ‖transverse (p i)‖`. -/
theorem norm_transverse_le_dist_axis (u : EuclideanSpace ℝ (Fin 3)) (t : ℝ) :
    ‖transverse u‖ ≤ ‖u - EuclideanSpace.single (2 : Fin 3) t‖ := by
  have h0 : transverse (EuclideanSpace.single (2 : Fin 3) t) = 0 := by
    simp [transverse, EuclideanSpace.single_apply]
    rfl
  calc ‖transverse u‖ = ‖transverse u - transverse (EuclideanSpace.single (2 : Fin 3) t)‖ := by
        rw [h0, sub_zero]
    _ ≤ ‖u - EuclideanSpace.single (2 : Fin 3) t‖ := norm_transverse_sub_le _ _

theorem dist_axis_attained (u : EuclideanSpace ℝ (Fin 3)) :
    ‖transverse u‖ = ‖u - EuclideanSpace.single (2 : Fin 3) (u 2)‖ := by
  have h2 : ‖u - EuclideanSpace.single (2 : Fin 3) (u 2)‖ ^ 2
      = (u 0) ^ 2 + (u 1) ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
    simp [Fin.sum_univ_three, EuclideanSpace.single_apply]
  have h1 : ‖transverse u‖ ^ 2 = (u 0) ^ 2 + (u 1) ^ 2 := by
    simp [transverse, Complex.sq_norm, Complex.normSq_apply]
    ring
  have hnn : (0 : ℝ) ≤ ‖u - EuclideanSpace.single (2 : Fin 3) (u 2)‖ := norm_nonneg _
  nlinarith [norm_nonneg (transverse u)]

variable {R b : ℝ} {p : ℕ → EuclideanSpace ℝ (Fin 3)} {n : ℕ}

/-- **The wrapping budget in space.**  A chain of `n` bonds of length at most `b` in `ℝ³`, no
residue of which enters the cylinder of radius `R` about the axis, winds at most `n b / (4 R)`
times around it. -/
theorem abs_winding3_le (hR : 0 < R) (hfar : ∀ i, R ≤ ‖transverse (p i)‖)
    (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b) : |winding3 p n| ≤ n * b / (4 * R) :=
  abs_winding_le hR hfar fun i =>
    le_trans (norm_transverse_sub_le _ _) (hstep i)

/-- **Residue demand in space.**  `k` turns of threading around a rod of exclusion radius `R`
with bonds of length at most `b` require at least `4 k R / b` residues. -/
theorem length_demand3 {k : ℝ} (hR : 0 < R) (hb : 0 < b) (hfar : ∀ i, R ≤ ‖transverse (p i)‖)
    (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b) (hk : k ≤ |winding3 p n|) : 4 * k * R / b ≤ n :=
  length_demand_of_winding hR hb hfar
    (fun i => le_trans (norm_transverse_sub_le _ _) (hstep i)) hk

/-- The transverse component in the usual complex form. -/
theorem transverse_eq (u : EuclideanSpace ℝ (Fin 3)) :
    transverse u = ((u 0 : ℝ) : ℂ) + ((u 1 : ℝ) : ℂ) * Complex.I := by
  apply Complex.ext <;> simp [transverse]

/-- The transverse projection of a continuously moving residue moves continuously. -/
theorem continuous_transverse {f : ℝ → EuclideanSpace ℝ (Fin 3)} (hf : Continuous f) :
    Continuous fun t => transverse (f t) := by
  have h0 : Continuous fun t => (f t) 0 := ((EuclideanSpace.proj (0 : Fin 3)).continuous).comp hf
  have h1 : Continuous fun t => (f t) 1 := ((EuclideanSpace.proj (1 : Fin 3)).continuous).comp hf
  simp only [transverse_eq]
  fun_prop

/-- **Topological protection in space.**  A continuous motion of a closed chain in `ℝ³` that
never puts a residue inside the exclusion cylinder and never stretches a bond past `2 R` leaves
the number of turns around the axis unchanged. -/
theorem winding3_invariant_of_deformation (hR : 0 < R) (hbR : b < 2 * R)
    (P : ℝ → ℕ → EuclideanSpace ℝ (Fin 3)) (hcont : ∀ i, Continuous fun t => P t i)
    (hfar : ∀ t i, R ≤ ‖transverse (P t i)‖) (hstep : ∀ t i, ‖P t (i + 1) - P t i‖ ≤ b)
    (hclosed : ∀ t, P t n = P t 0) (s u : ℝ) :
    winding3 (P s) n = winding3 (P u) n :=
  winding_invariant_of_deformation hR hbR (fun t i => transverse (P t i))
    (fun i => continuous_transverse (hcont i)) hfar
    (fun t i => le_trans (norm_transverse_sub_le _ _) (hstep t i))
    (fun t => congrArg transverse (hclosed t)) s u

end RequestProject.Winding3D
