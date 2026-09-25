/-
# Part LXI  Dye photophysics and linkers: what a single-molecule FRET number is worth

Part XXXIII models a burst as clean photon counting and Part LIII removed the `κ² = 2/3`
substitution.  Two idealisations survived: the photophysics that relates photon counts to a
transfer efficiency, and the linker that separates the dye from the residue whose distance is
being reported.  `RequestProject.Photophysics` removes both, and both carry an exact price.

`IDR.photophysics_laws` bundles six statements:

1. *The raw ratio is the efficiency only in the calibrated case.*  The observed proximity ratio
   equals the true transfer efficiency at some interior efficiency exactly when the detection
   correction factor `g` is `1`.
2. *Rankings survive.*  The proximity ratio is strictly increasing in the efficiency, so the
   ordering of conformers and the direction of any change are correct whatever `g` is.
3. *Magnitudes do not.*  Since `E/(1−E) = (R₀/r)⁶`, an uncorrected analysis returns
   `(1−P)/P = (1/g)·(1−E)/E`: the inferred `r⁶` is the true `r⁶` divided by `g`, exactly.
4. *And `g` is not in the data*: every observed ratio in `(0,1)` is produced by every `g > 0` at
   a suitable true efficiency.  A single-colour measurement determines a ranking of distances and
   no distance.
5. *Background biases towards compaction.*  With any acceptor background at all, a state with
   zero transfer is measured at strictly positive efficiency -- and the extended states of a
   disordered region are precisely the zero-transfer ones.
6. *The linker is a bracket, not a correction.*  The dye--dye distance determines the
   residue--residue distance to `±2L` and no better: the bound follows from the triangle
   inequality alone, and it is attained at both ends by two configurations with the same
   attachment distance whose dye distances differ by `4L`.

The three multiplicative corrections a single-molecule distance carries -- `κ²` (Part LIII),
`g` (here), and the linker range (here) -- are independent, and none of them shrinks when more
photons are collected.
-/
import Mathlib
import RequestProject.Photophysics

set_option autoImplicit false

namespace IDR

open IDR.Photophysics

/-- **The single-molecule photophysics laws.**

1. the proximity ratio equals the efficiency iff the correction factor is one;
2. it is strictly increasing in the efficiency, so rankings survive;
3. the inferred sixth power is off by exactly the correction factor;
4. which is not identifiable from the observed ratio;
5. acceptor background makes a zero-transfer state measure positive;
6. and the linker brackets the residue distance to `±2L`, sharply. -/
theorem photophysics_laws :
    (∀ g E : ℝ, 0 < g → 0 < E → E < 1 → (proximityRatio g E = E ↔ g = 1)) ∧
    (∀ g E F : ℝ, 0 < g → 0 ≤ E → E < F → F ≤ 1 →
        proximityRatio g E < proximityRatio g F) ∧
    (∀ g E : ℝ, 0 < g → 0 < E → E < 1 →
        (1 - proximityRatio g E) / proximityRatio g E = (1 / g) * ((1 - E) / E)) ∧
    (∀ g P : ℝ, 0 < g → 0 < P → P < 1 →
        ∃ E : ℝ, 0 < E ∧ E < 1 ∧ proximityRatio g E = P) ∧
    (∀ nD bD bA : ℝ, 0 ≤ nD → 0 ≤ bD → 0 < bA → 0 < apparentEff nD 0 bD bA) ∧
    ((∀ (X : Type) (_ : MetricSpace X) (a1 a2 b1 b2 : X) (L : ℝ),
        dist a1 b1 ≤ L → dist a2 b2 ≤ L → |dist b1 b2 - dist a1 a2| ≤ 2 * L) ∧
      (∀ d L : ℝ, 0 < L → 2 * L ≤ d →
        dist (0:ℝ) d = d ∧
        (dist (0:ℝ) (-L) ≤ L ∧ dist d (d + L) ≤ L ∧ dist (-L : ℝ) (d + L) = d + 2 * L) ∧
        (dist (0:ℝ) L ≤ L ∧ dist d (d - L) ≤ L ∧ dist (L : ℝ) (d - L) = d - 2 * L))) := by
  refine ⟨fun g E hg h0 h1 => proximityRatio_eq_self_iff hg h0 h1,
    fun g E F hg h0 hEF h1 => proximityRatio_strictMono hg h0 hEF h1,
    fun g E hg h0 h1 => sixthPower_off_by_gamma hg h0 h1,
    fun g P hg h0 h1 => gamma_unidentifiable hg h0 h1,
    fun nD bD bA hD hbD hbA => apparentEff_pos_of_background hD hbD hbA,
    ⟨fun X _ a1 a2 b1 b2 L h1 h2 => linker_bound h1 h2,
      fun d L hL hd => linker_bound_sharp hL hd⟩⟩

end IDR
