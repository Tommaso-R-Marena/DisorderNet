/-
# Capstone: the topological clause of a model of a disordered region

One statement, `entanglement_design_law`, collecting the results of `RequestProject.Winding` and
`RequestProject.Threading`.  Read as the topological clause of the specification a model of an
intrinsically disordered region has to satisfy -- the one clause that no amount of energy
refinement or ensemble reweighting can supply after the fact:

0. **Wrapping is budgeted even with no excluded volume at all**: `n` bonds never make more than
   `n / 2` turns, because a single bond cannot sweep half a turn about a point it misses.
1. **Excluded volume sharpens the budget.**  Connectivity and excluded volume alone cap the winding of a chain
   of `n` bonds of length `b` around an object of exclusion radius `R` at `n b / (4 R)` turns.
   Read backwards this is a residue demand: `k` turns of observed threading require at least
   `4 k R / b` residues of disordered chain.
2. **The budget is tight to within `π / 2`.**  An explicit regular wrap achieves exactly `k`
   turns with any `n ≥ 2π k R / b`, so the design law brackets the true demand rather than
   merely bounding it.
3. **Threading is protected.**  Along any continuous deformation that keeps residues out of the
   excluded region and bonds shorter than `2 R`, the winding number of a closed chain is
   constant.  Topology is not a soft degree of freedom.
4. **Populations inherit the budget.**  The threaded population at level `k` is at most
   `n b / (4 R k)`, so a measured threaded fraction above that ceiling refutes every admissible
   ensemble at once, with no model in between.
5. **All of it holds in space, and it is measurable.**  The budget and protection transfer to a
   chain in `ℝ³` wrapping a straight axis, and threading is paid for in reach: a wrapped chain
   forfeits `8 R² w² / (b n)` of the contour limit on its axial extension, so an ordinary
   extension measurement bounds the topological state.
6. **Reweighting cannot repair the wrong sector.**  If the conformations a model samples are
   unthreaded, then for *every* weight vector its threaded population is exactly zero and its
   discrepancy against a measured threaded population `phi` is exactly `phi`.  Combined with
   clause 3: neither reweighting nor admissible dynamics can move a model into the right
   topological sector.  It has to be built there -- which is a statement about how the ensemble
   is *generated*, not about how it is scored.

Together with the metric clauses proved elsewhere in this project, this says that a model of a
disordered region is specified by an ensemble whose support is chosen in the correct topological
sectors and whose weights then carry the metric data; and that the sector populations, unlike
distances and contacts, are data the model must be constructed to reproduce.
-/
import Mathlib
import RequestProject.Winding
import RequestProject.Threading
import RequestProject.Winding3D
import RequestProject.ThreadingExtension

namespace RequestProject.EntanglementVerdict

open Finset RequestProject.Winding RequestProject.Threading
open RequestProject.Winding3D RequestProject.ThreadingExtension

/-- **The entanglement design law.**  Five clauses: the wrapping budget and the residue demand
it implies, the explicit realisation that makes the demand tight to within `π / 2`, topological
protection of the winding number, the population ceiling for threading, and the impossibility of
creating threading by reweighting. -/
theorem entanglement_design_law :
    -- (i) the wrapping budget, and the residue demand it implies
    (∀ (R b : ℝ) (p : ℕ → ℂ) (n : ℕ), 0 < R → (∀ i, R ≤ ‖p i‖) →
        (∀ i, ‖p (i + 1) - p i‖ ≤ b) → |winding p n| ≤ n * b / (4 * R))
    ∧ (∀ (R b k : ℝ) (p : ℕ → ℂ) (n : ℕ), 0 < R → 0 < b → (∀ i, R ≤ ‖p i‖) →
        (∀ i, ‖p (i + 1) - p i‖ ≤ b) → k ≤ |winding p n| → 4 * k * R / b ≤ n)
    ∧ -- the exclusion-free half of the budget: `n` bonds never exceed `n / 2` turns
      (∀ (p : ℕ → ℂ) (n : ℕ), |winding p n| ≤ n / 2)
    ∧ -- (ii) the demand is tight: an explicit chain realises `k` turns once `n ≥ 2π k R / b`
      (∀ (R b : ℝ) (k n : ℕ), 0 < R → 0 < b → b < 2 * R → 0 < k →
        2 * Real.pi * k * R / b ≤ n →
        ∃ p : ℕ → ℂ, (∀ i, R ≤ ‖p i‖) ∧ (∀ i, ‖p (i + 1) - p i‖ ≤ b) ∧ winding p n = k)
    ∧ -- (iii) topological protection under admissible deformation
      (∀ (R b : ℝ) (n : ℕ) (P : ℝ → ℕ → ℂ), 0 < R → b < 2 * R →
        (∀ i, Continuous fun t => P t i) → (∀ t i, R ≤ ‖P t i‖) →
        (∀ t i, ‖P t (i + 1) - P t i‖ ≤ b) → (∀ t, P t n = P t 0) →
        ∀ s u, winding (P s) n = winding (P u) n)
    ∧ -- (iv) the population ceiling, and the refutation it gives
      (∀ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → ℂ) (R b k : ℝ) (n : ℕ), 0 < R →
        (∀ a, 0 ≤ w a) → (∑ a, w a = 1) → Admissible X R b → 0 < k →
        threadedFraction w X n k ≤ n * b / (4 * R * k))
    ∧ (∀ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → ℂ) (R b k phi e : ℝ) (n : ℕ), 0 < R →
        (∀ a, 0 ≤ w a) → (∑ a, w a = 1) → Admissible X R b → 0 < k →
        (n : ℝ) * b / (4 * R * k) < phi - e →
        e < |threadedFraction w X n k - phi|)
    ∧ -- (v) everything above survives in three dimensions, about the true axis
      (∀ (R b : ℝ) (p : ℕ → EuclideanSpace ℝ (Fin 3)) (n : ℕ), 0 < R →
        (∀ i, R ≤ ‖transverse (p i)‖) → (∀ i, ‖p (i + 1) - p i‖ ≤ b) →
        |winding3 p n| ≤ n * b / (4 * R))
    ∧ -- (vi) threading is paid for in extension, so a metric measurement bounds the topology
      (∀ (R b : ℝ) (p : ℕ → EuclideanSpace ℝ (Fin 3)) (n : ℕ), 0 < R → 0 < b → 0 < n →
        (∀ i, R ≤ ‖transverse (p i)‖) → (∀ i, ‖p (i + 1) - p i‖ ≤ b) →
        |p n 2 - p 0 2| ≤ n * b - 8 * R ^ 2 * (winding3 p n) ^ 2 / (b * n))
    ∧ -- (vii) reweighting cannot create threading
      (∀ (m : ℕ) (w : Fin m → ℝ) (X : Fin m → ℕ → ℂ) (k phi : ℝ) (n : ℕ), 0 < k →
        (∀ a, winding (X a) n = 0) →
        threadedFraction w X n k = 0 ∧ |threadedFraction w X n k - phi| = |phi|) := by
  refine ⟨fun R b p n hR hfar hstep => abs_winding_le hR hfar hstep,
    fun R b k p n hR hb hfar hstep hk => length_demand_of_winding hR hb hfar hstep hk,
    fun p n => abs_winding_le_half p n,
    fun R b k n hR hb hbR hk hn => ?_, fun R b n P hR hbR hcont hfar hstep hclosed s u =>
      winding_invariant_of_deformation hR hbR P hcont hfar hstep hclosed s u,
    fun m w X R b k n hR hw hsum hX hk => threadedFraction_le_budget hR hw hsum hX hk,
    fun m w X R b k phi e n hR hw hsum hX hk hex =>
      no_ensemble_of_excess_threading hR hw hsum hX hk hex,
    fun R b p n hR hfar hstep => abs_winding3_le hR hfar hstep,
    fun R b p n hR hb hn hfar hstep => axial_extension_le hR hb hn hfar hstep,
    fun m w X k phi n hk hzero =>
      ⟨reweighting_cannot_create_threading hk hzero, unthreaded_model_gap hk hzero⟩⟩
  exact (wrap_threshold_window hR hb hbR hk hn).2

end RequestProject.EntanglementVerdict
