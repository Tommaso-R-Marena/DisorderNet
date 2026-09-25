/-
# Threading costs extension: a topological state read from a metric measurement

The results of `RequestProject.Winding` and `RequestProject.Threading` are about a quantity --
the winding number -- that is hard to measure directly.  This file closes that gap.  It proves a
quantitative trade-off between how much a chain is wrapped around an axis and how far it can
reach along it:

`axial_extension_le` -- for a chain of `n` bonds of length at most `b` in `ℝ³` that never enters
the exclusion cylinder of radius `R` about the axis,

    |z(n) - z(0)|  ≤  n b  -  8 R² w² / (b n),          w = winding3 p n.

The first term is the familiar contour limit; the second is the extension a chain *forfeits* by
being wrapped.  The mechanism is exact and local: a bond that turns the chain by an angle at the
axis must spend transverse length to do it (the chord bound of `Winding`), and transverse length
is subtracted in quadrature from the length available along the axis.

This makes threading an inference from ordinary metric data.  A single-molecule FRET or force
measurement that reports an end-to-end extension `D` along the axis refutes every wrapped state
with `w² > b n (n b - D) / (8 R²)` (`winding_bound_of_extension`), and in particular a chain
observed at more than `n b - 8 R² / (b n)` of extension is not wrapped even once
(`unwrapped_of_large_extension`).  Conversely a model that places a disordered region in a
threaded state predicts a *shorter* accessible extension, by an amount the same inequality
quantifies, so the topological clause of a model has metric consequences that experiment can
already check.
-/
import Mathlib
import RequestProject.Winding
import RequestProject.Winding3D
import RequestProject.Threading

namespace RequestProject.ThreadingExtension

open Finset RequestProject.Winding RequestProject.Winding3D RequestProject.Threading

/-- Pythagoras along the axis: the square of a bond splits into its axial and transverse parts. -/
theorem sq_norm_split (u v : EuclideanSpace ℝ (Fin 3)) :
    (u 2 - v 2) ^ 2 + ‖transverse u - transverse v‖ ^ 2 = ‖u - v‖ ^ 2 := by
  have h1 : ‖transverse u - transverse v‖ ^ 2 = (u 0 - v 0) ^ 2 + (u 1 - v 1) ^ 2 := by
    simp [transverse, Complex.sq_norm, Complex.normSq_apply, Complex.sub_re, Complex.sub_im]
    ring
  have h2 : ‖u - v‖ ^ 2 = (u 0 - v 0) ^ 2 + (u 1 - v 1) ^ 2 + (u 2 - v 2) ^ 2 := by
    rw [EuclideanSpace.norm_eq, Real.sq_sqrt (by positivity)]
    simp [Fin.sum_univ_three]
  rw [h1, h2]; ring

/-- **Transverse length is subtracted from axial reach.**  A bond of length at most `b` whose
transverse part has length `c` advances along the axis by at most `b - c² / (2 b)`. -/
theorem abs_axial_step_le {b : ℝ} (hb : 0 < b) {u v : EuclideanSpace ℝ (Fin 3)}
    (hstep : ‖u - v‖ ≤ b) :
    |u 2 - v 2| ≤ b - ‖transverse u - transverse v‖ ^ 2 / (2 * b) := by
  set c := ‖transverse u - transverse v‖ with hc
  have hc0 : 0 ≤ c := norm_nonneg _
  have hsplit := sq_norm_split u v
  have hcb : c ≤ b := by
    nlinarith [sq_nonneg (u 2 - v 2), norm_nonneg (u - v)]
  have hax : (u 2 - v 2) ^ 2 ≤ b ^ 2 - c ^ 2 := by
    nlinarith [norm_nonneg (u - v)]
  have hrhs : 0 ≤ b - c ^ 2 / (2 * b) := by
    rw [sub_nonneg, div_le_iff₀ (by positivity)]
    nlinarith
  have hexp : (b - c ^ 2 / (2 * b)) ^ 2 = b ^ 2 - c ^ 2 + c ^ 4 / (4 * b ^ 2) := by
    field_simp; ring
  have h4 : 0 ≤ c ^ 4 / (4 * b ^ 2) := by positivity
  have hx2 : (u 2 - v 2) ^ 2 ≤ (b - c ^ 2 / (2 * b)) ^ 2 := by rw [hexp]; linarith
  nlinarith [abs_nonneg (u 2 - v 2), sq_abs (u 2 - v 2)]

variable {R b : ℝ} {p : ℕ → EuclideanSpace ℝ (Fin 3)} {n : ℕ}

/-- The transverse chord of the `i`-th bond. -/
noncomputable def chord (p : ℕ → EuclideanSpace ℝ (Fin 3)) (i : ℕ) : ℝ :=
  ‖transverse (p (i + 1)) - transverse (p i)‖

/-- **Wrapping is paid for in transverse length.**  The transverse chords of a chain that winds
`w` times around the axis, staying outside radius `R`, sum to at least `4 R |w|`. -/
theorem sum_chord_ge (hR : 0 < R) (hfar : ∀ i, R ≤ ‖transverse (p i)‖) :
    4 * R * |winding3 p n| ≤ ∑ i ∈ range n, chord p i := by
  have hpi := Real.pi_pos
  have hturn : ∀ i, |turn (transverse (p i)) (transverse (p (i + 1)))|
      ≤ Real.pi * chord p i / (2 * R) := fun i =>
    abs_turn_le hR (hfar i) (hfar (i + 1)) (le_of_eq rfl)
  have htot : |totalTurn (fun i => transverse (p i)) n|
      ≤ ∑ i ∈ range n, Real.pi * chord p i / (2 * R) := by
    calc |totalTurn (fun i => transverse (p i)) n|
        ≤ ∑ i ∈ range n, |turn (transverse (p i)) (transverse (p (i + 1)))| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ i ∈ range n, Real.pi * chord p i / (2 * R) := Finset.sum_le_sum fun i _ => hturn i
  have hsum : ∑ i ∈ range n, Real.pi * chord p i / (2 * R)
      = Real.pi / (2 * R) * ∑ i ∈ range n, chord p i := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun i _ => by ring
  rw [hsum] at htot
  have hw : |winding3 p n| = |totalTurn (fun i => transverse (p i)) n| / (2 * Real.pi) := by
    rw [winding3, winding, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 * Real.pi)]
  rw [hw]
  have hid : 4 * R * (|totalTurn (fun i => transverse (p i)) n| / (2 * Real.pi))
      = (2 * R / Real.pi) * |totalTurn (fun i => transverse (p i)) n| := by
    field_simp; ring
  rw [hid]
  calc (2 * R / Real.pi) * |totalTurn (fun i => transverse (p i)) n|
      ≤ (2 * R / Real.pi) * (Real.pi / (2 * R) * ∑ i ∈ range n, chord p i) :=
        mul_le_mul_of_nonneg_left htot (by positivity)
    _ = ∑ i ∈ range n, chord p i := by field_simp

/-- **Threading costs extension.**  A chain of `n` bonds of length at most `b` that stays outside
the cylinder of radius `R` and winds `w` times around the axis reaches at most
`n b - 8 R² w² / (b n)` along the axis: the contour limit `n b`, less the length forfeited to
wrapping. -/
theorem axial_extension_le (hR : 0 < R) (hb : 0 < b) (hn : 0 < n)
    (hfar : ∀ i, R ≤ ‖transverse (p i)‖) (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b) :
    |p n 2 - p 0 2| ≤ n * b - 8 * R ^ 2 * (winding3 p n) ^ 2 / (b * n) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  -- axial displacement telescopes and is bounded bond by bond
  have htel : (p n 2 - p 0 2) = ∑ i ∈ range n, (p (i + 1) 2 - p i 2) :=
    (Finset.sum_range_sub (fun i => p i 2) n).symm
  have hstepbd : ∀ i, |p (i + 1) 2 - p i 2| ≤ b - chord p i ^ 2 / (2 * b) := fun i =>
    abs_axial_step_le hb (hstep i)
  have hax : |p n 2 - p 0 2| ≤ ∑ i ∈ range n, (b - chord p i ^ 2 / (2 * b)) := by
    rw [htel]
    exact le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum fun i _ => hstepbd i)
  have hsplit : ∑ i ∈ range n, (b - chord p i ^ 2 / (2 * b))
      = n * b - (∑ i ∈ range n, chord p i ^ 2) / (2 * b) := by
    rw [Finset.sum_sub_distrib]
    · rw [Finset.sum_div]
      simp [mul_comm]
  -- Cauchy--Schwarz turns the sum of chords into the sum of their squares
  have hcs : (∑ i ∈ range n, chord p i) ^ 2 ≤ n * ∑ i ∈ range n, chord p i ^ 2 := by
    simpa using sq_sum_le_card_mul_sum_sq (s := range n) (f := chord p)
  have hchord := sum_chord_ge (p := p) (n := n) hR hfar
  have hwnn : 0 ≤ |winding3 p n| := abs_nonneg _
  have hsq : 16 * R ^ 2 * (winding3 p n) ^ 2 ≤ (∑ i ∈ range n, chord p i) ^ 2 := by
    have h1 : 0 ≤ 4 * R * |winding3 p n| := by positivity
    nlinarith [sq_abs (winding3 p n)]
  have hfinal : 16 * R ^ 2 * (winding3 p n) ^ 2 ≤ n * ∑ i ∈ range n, chord p i ^ 2 :=
    le_trans hsq hcs
  have hbound : |p n 2 - p 0 2| ≤ n * b - (∑ i ∈ range n, chord p i ^ 2) / (2 * b) := by
    rw [← hsplit]; exact hax
  have hdiv : 8 * R ^ 2 * (winding3 p n) ^ 2 / (b * n)
      ≤ (∑ i ∈ range n, chord p i ^ 2) / (2 * b) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_le_mul_of_nonneg_left hfinal hb.le]
  linarith

/-- **Reading topology off a metric measurement.**  An observed axial extension `D` caps how much
the chain can be wrapped: any conformation compatible with the measurement has
`w² ≤ b n (n b - D) / (8 R²)`. -/
theorem winding_bound_of_extension (hR : 0 < R) (hb : 0 < b) (hn : 0 < n)
    (hfar : ∀ i, R ≤ ‖transverse (p i)‖) (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b)
    {D : ℝ} (hD : D ≤ |p n 2 - p 0 2|) :
    (winding3 p n) ^ 2 ≤ b * n * (n * b - D) / (8 * R ^ 2) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have h := axial_extension_le hR hb hn hfar hstep
  have h2 : 8 * R ^ 2 * (winding3 p n) ^ 2 / (b * n) ≤ n * b - D := by linarith
  rw [div_le_iff₀ (by positivity)] at h2
  rw [le_div_iff₀ (by positivity)]
  nlinarith

/-- A chain seen extended to more than `n b - 8 R² / (b n)` along the axis is not wrapped even
once: the measurement excludes the threaded sectors outright. -/
theorem unwrapped_of_large_extension (hR : 0 < R) (hb : 0 < b) (hn : 0 < n)
    (hfar : ∀ i, R ≤ ‖transverse (p i)‖) (hstep : ∀ i, ‖p (i + 1) - p i‖ ≤ b)
    (hext : n * b - 8 * R ^ 2 / (b * n) < |p n 2 - p 0 2|) :
    |winding3 p n| < 1 := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have h := axial_extension_le hR hb hn hfar hstep
  by_contra hcon
  push_neg at hcon
  have hsq : 1 ≤ (winding3 p n) ^ 2 := by nlinarith [sq_abs (winding3 p n), abs_nonneg (winding3 p n)]
  have hmono : 8 * R ^ 2 / (b * n) ≤ 8 * R ^ 2 * (winding3 p n) ^ 2 / (b * n) := by
    rw [div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_nonneg (mul_nonneg (by positivity : (0 : ℝ) ≤ 8 * R ^ 2)
      (le_of_lt (mul_pos hb hn0))) (sub_nonneg.2 hsq)]
  linarith

/-! ### Ensembles: a measured extension caps the threaded population -/

/-- The transverse projection of an ensemble of chains in space, as an ensemble of planar
chains -- the object the results of `RequestProject.Threading` speak about. -/
noncomputable def proj {m : ℕ} (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)) :
    Fin m → ℕ → ℂ := fun a i => transverse (X a i)

theorem winding_proj {m : ℕ} (X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)) (a : Fin m) (n : ℕ) :
    winding (proj X a) n = winding3 (X a) n := rfl

/-- **A measured mean extension caps the mean squared winding.**  Averaging the trade-off over an
ensemble: if the mean axial extension of the ensemble is `D`, then its mean squared winding is at
most `b n (n b - D) / (8 R²)`.  An ordinary FRET or force-extension mean, with no topological
assay at all, is a quantitative constraint on how entangled the ensemble can be. -/
theorem meanSqWinding_le_of_extension {m : ℕ} {w : Fin m → ℝ}
    {X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)} (hR : 0 < R) (hb : 0 < b) (hn : 0 < n)
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hfar : ∀ a i, R ≤ ‖transverse (X a i)‖) (hstep : ∀ a i, ‖X a (i + 1) - X a i‖ ≤ b)
    {D : ℝ} (hD : D ≤ ∑ a, w a * |X a n 2 - X a 0 2|) :
    ∑ a, w a * (winding3 (X a) n) ^ 2 ≤ b * n * (n * b - D) / (8 * R ^ 2) := by
  have hn0 : (0 : ℝ) < n := by exact_mod_cast hn
  have hpt : ∀ a : Fin m, 8 * R ^ 2 * (winding3 (X a) n) ^ 2 / (b * n)
      ≤ n * b - |X a n 2 - X a 0 2| := fun a => by
    have h := axial_extension_le (p := X a) (n := n) hR hb hn (hfar a) (hstep a)
    linarith
  have hmean : ∑ a, w a * (8 * R ^ 2 * (winding3 (X a) n) ^ 2 / (b * n))
      ≤ ∑ a, w a * (n * b - |X a n 2 - X a 0 2|) :=
    Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_left (hpt a) (hw a)
  have hlhs : ∑ a, w a * (8 * R ^ 2 * (winding3 (X a) n) ^ 2 / (b * n))
      = (8 * R ^ 2 / (b * n)) * ∑ a, w a * (winding3 (X a) n) ^ 2 := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun a _ => by ring
  have hrhs : ∑ a, w a * (n * b - |X a n 2 - X a 0 2|)
      = n * b - ∑ a, w a * |X a n 2 - X a 0 2| := by
    have hdist : ∀ a : Fin m, w a * (n * b - |X a n 2 - X a 0 2|)
        = w a * (n * b) - w a * |X a n 2 - X a 0 2| := fun a => by ring
    rw [Finset.sum_congr rfl fun a _ => hdist a, Finset.sum_sub_distrib, ← Finset.sum_mul,
      hsum, one_mul]
  rw [hlhs, hrhs] at hmean
  have hstep2 : (8 * R ^ 2 / (b * n)) * ∑ a, w a * (winding3 (X a) n) ^ 2 ≤ n * b - D := by
    linarith
  rw [le_div_iff₀ (by positivity : (0 : ℝ) < 8 * R ^ 2)]
  rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity : (0 : ℝ) < b * n)] at hstep2
  nlinarith [hstep2]

/-- **Threaded population from an extension measurement.**  Chebyshev on the mean squared
winding: at level `k`, the population wound `k` times or more is at most
`b n (n b - D) / (8 R² k²)`.  For a well-extended ensemble this is a far sharper ceiling than the
kinematic budget of `RequestProject.Threading`, and it needs only metric data. -/
theorem threadedFraction_le_of_extension {m : ℕ} {w : Fin m → ℝ}
    {X : Fin m → ℕ → EuclideanSpace ℝ (Fin 3)} (hR : 0 < R) (hb : 0 < b) (hn : 0 < n)
    (hw : ∀ a, 0 ≤ w a) (hsum : ∑ a, w a = 1)
    (hfar : ∀ a i, R ≤ ‖transverse (X a i)‖) (hstep : ∀ a i, ‖X a (i + 1) - X a i‖ ≤ b)
    {D k : ℝ} (hk : 0 < k) (hD : D ≤ ∑ a, w a * |X a n 2 - X a 0 2|) :
    threadedFraction w (proj X) n k ≤ b * n * (n * b - D) / (8 * R ^ 2 * k ^ 2) := by
  classical
  have hmean := meanSqWinding_le_of_extension (w := w) (X := X) hR hb hn hw hsum hfar hstep hD
  -- Chebyshev step
  have hcheb : k ^ 2 * threadedFraction w (proj X) n k ≤ ∑ a, w a * (winding3 (X a) n) ^ 2 := by
    have hsub : (Finset.univ.filter (fun a => k ≤ |winding (proj X a) n|)) ⊆ Finset.univ :=
      Finset.filter_subset _ _
    calc k ^ 2 * threadedFraction w (proj X) n k
        = ∑ a ∈ Finset.univ.filter (fun a => k ≤ |winding (proj X a) n|), w a * k ^ 2 := by
          rw [threadedFraction, Finset.mul_sum]
          exact Finset.sum_congr rfl fun a _ => by ring
      _ ≤ ∑ a ∈ Finset.univ.filter (fun a => k ≤ |winding (proj X a) n|),
            w a * (winding3 (X a) n) ^ 2 := by
          refine Finset.sum_le_sum fun a ha => ?_
          have hka := (Finset.mem_filter.1 ha).2
          rw [winding_proj] at hka
          have : k ^ 2 ≤ (winding3 (X a) n) ^ 2 := by
            nlinarith [abs_nonneg (winding3 (X a) n), sq_abs (winding3 (X a) n)]
          exact mul_le_mul_of_nonneg_left this (hw a)
      _ ≤ ∑ a, w a * (winding3 (X a) n) ^ 2 :=
          Finset.sum_le_sum_of_subset_of_nonneg hsub
            (fun a _ _ => mul_nonneg (hw a) (sq_nonneg _))
  have hfinal : k ^ 2 * threadedFraction w (proj X) n k ≤ b * n * (n * b - D) / (8 * R ^ 2) :=
    le_trans hcheb hmean
  rw [le_div_iff₀ (by positivity : (0 : ℝ) < 8 * R ^ 2 * k ^ 2)]
  rw [le_div_iff₀ (by positivity : (0 : ℝ) < 8 * R ^ 2)] at hfinal
  nlinarith [hfinal]

end RequestProject.ThreadingExtension
