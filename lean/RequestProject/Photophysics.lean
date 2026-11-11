/-
# Part LXI.1  What a FRET experiment actually measures: photophysics and linkers

Part XXXIII models a burst as clean photon counting and Part LIII removed the `κ² = 2/3`
substitution.  Two idealisations named there remain: dye photophysics -- the detection
efficiencies and quantum yields that relate photon counts to a transfer efficiency -- and the
linker, the flexible arm of a dozen or so bonds that separates the dye from the residue the
distance is reported for.  This file removes both, and both turn out to have exact, quantitative
prices.

**The gamma factor.**  Photon counts give the *proximity ratio*, `nA/(nA + nD)`; the transfer
efficiency needs the correction factor `g = (η_A φ_A)/(η_D φ_D)`.  With `proximityRatio g E`
the observed ratio at true efficiency `E`:

* `proximityRatio_eq_self_iff` -- the raw ratio equals the efficiency exactly when `g = 1`, and
  otherwise at no interior efficiency at all.
* `proximityRatio_strictMono` -- but it is strictly increasing in `E`.  This is the positive
  result: *rankings* of conformers, and the direction of any change, survive an unknown `g`
  intact.
* `sixthPower_off_by_gamma` -- **and the magnitude does not.**  Since `E/(1−E) = (R₀/r)⁶`, the
  uncorrected analysis returns `(1−P)/P = (1/g)·(1−E)/E`: the inferred `r⁶` is the true `r⁶`
  divided by `g`, exactly.  A factor two in `g` is a factor two in `r⁶`.
* `gamma_unidentifiable` -- and `g` is not in the data: every observed ratio in `(0,1)` is
  produced by every `g > 0` at a suitable true efficiency.  A single-colour experiment therefore
  determines a *ranking* of distances and no distance.

**Background.**  `apparentEff_pos_of_background`: with any acceptor background at all, a state
with zero transfer is measured at strictly positive efficiency.  A disordered region's extended
states -- exactly the ones that carry the information about the coil -- are the ones this
distorts, and it distorts them upwards, so the measured distance distribution is biased towards
compaction in a direction no amount of averaging removes.

**The linker.**  The dye is not the residue.  With attachment points `a₁, a₂` and dyes `b₁, b₂`
constrained to lie within `L` of their attachment points:

* `linker_bound` -- `|dist b₁ b₂ − dist a₁ a₂| ≤ 2L`, in any metric space, from the triangle
  inequality alone.  That is the whole positive content: a measured dye--dye distance brackets
  the residue--residue distance, to `±2L`.
* `linker_bound_sharp` -- and the bracket is attained at both ends: two configurations with the
  *same* attachment distance whose dye distances differ by `4L`.  The correction is not a shift,
  a scale factor, or any function of the attachment distance; it is a range.

For a typical linker (`L ≈ 1` nm) and a typical `R₀` (`≈ 5` nm) the `±2` nm bracket is not a
refinement -- it is comparable with the difference between a compact and an expanded state of a
short disordered region.  Together with Part LIII's factor `6` in `r⁶` from `κ²` and the factor
`g` here, the three corrections a single-molecule measurement carries are multiplicative and
independent, and none of them is removed by collecting more photons.
-/
import Mathlib

set_option autoImplicit false

namespace IDR
namespace Photophysics

/-! ## The gamma factor -/

/-- The proximity ratio observed at true transfer efficiency `E` with detection-correction
factor `g = (η_A φ_A)/(η_D φ_D)`. -/
noncomputable def proximityRatio (g E : ℝ) : ℝ := g * E / (g * E + (1 - E))

lemma proximityRatio_den_pos {g E : ℝ} (hg : 0 < g) (h0 : 0 ≤ E) (h1 : E ≤ 1) :
    0 < g * E + (1 - E) := by
  rcases eq_or_lt_of_le h1 with h | h
  · subst h; simpa using hg
  · nlinarith

/-- **The raw ratio is the efficiency only when the correction factor is one.** -/
theorem proximityRatio_eq_self_iff {g E : ℝ} (hg : 0 < g) (h0 : 0 < E) (h1 : E < 1) :
    proximityRatio g E = E ↔ g = 1 := by
  have hden : 0 < g * E + (1 - E) := proximityRatio_den_pos hg h0.le h1.le
  unfold proximityRatio
  rw [div_eq_iff hden.ne']
  constructor
  · intro h
    have h2 : g * E = E * (g * E) + E * (1 - E) := by linarith
    have h3 : (g - 1) * (E * (1 - E)) = 0 := by nlinarith
    have h4 : E * (1 - E) ≠ 0 := (mul_pos h0 (by linarith : (0:ℝ) < 1 - E)).ne'
    have := mul_eq_zero.1 h3
    rcases this with h5 | h5
    · linarith
    · exact absurd h5 h4
  · intro h
    subst h
    ring

/-- **But the ratio is strictly increasing in the efficiency**: rankings survive an unknown
correction factor. -/
theorem proximityRatio_strictMono {g E F : ℝ} (hg : 0 < g) (h0 : 0 ≤ E) (hEF : E < F)
    (h1 : F ≤ 1) : proximityRatio g E < proximityRatio g F := by
  have hE1 : E ≤ 1 := le_trans hEF.le h1
  have hF0 : 0 ≤ F := le_trans h0 hEF.le
  have hdE : 0 < g * E + (1 - E) := proximityRatio_den_pos hg h0 hE1
  have hdF : 0 < g * F + (1 - F) := proximityRatio_den_pos hg hF0 h1
  unfold proximityRatio
  rw [div_lt_div_iff₀ hdE hdF]
  nlinarith [hg, hEF]

/-- **The inferred sixth power is off by exactly the correction factor.**  Since
`E/(1−E) = (R₀/r)⁶`, an uncorrected analysis returns the true `r⁶` divided by `g`. -/
theorem sixthPower_off_by_gamma {g E : ℝ} (hg : 0 < g) (h0 : 0 < E) (h1 : E < 1) :
    (1 - proximityRatio g E) / proximityRatio g E = (1 / g) * ((1 - E) / E) := by
  have hden : 0 < g * E + (1 - E) := proximityRatio_den_pos hg h0.le h1.le
  have hnum : 0 < g * E := by positivity
  have hD : (g * E + (1 - E)) ≠ 0 := hden.ne'
  have hgE : g * E ≠ 0 := hnum.ne'
  have hE : E ≠ 0 := h0.ne'
  unfold proximityRatio
  have hL : (1 - g * E / (g * E + (1 - E))) = (1 - E) / (g * E + (1 - E)) := by
    field_simp
    ring
  have hsplit : (1 - E) / (g * E + (1 - E)) / (g * E / (g * E + (1 - E)))
      = (1 - E) / (g * E) := by
    rw [div_div_div_comm, div_self hD, div_one]
  rw [hL, hsplit]
  field_simp

/-- **The correction factor is not in the data.**  Every observed ratio in `(0,1)` is produced by
every `g > 0` at a suitable true efficiency. -/
theorem gamma_unidentifiable {g P : ℝ} (hg : 0 < g) (h0 : 0 < P) (h1 : P < 1) :
    ∃ E : ℝ, 0 < E ∧ E < 1 ∧ proximityRatio g E = P := by
  have h1' : (0:ℝ) < 1 - P := by linarith
  have hd : 0 < g * (1 - P) + P := by positivity
  refine ⟨P / (g * (1 - P) + P), div_pos h0 hd, ?_, ?_⟩
  · rw [div_lt_one hd]
    nlinarith
  · have hEval : g * (P / (g * (1 - P) + P)) + (1 - P / (g * (1 - P) + P))
        = g / (g * (1 - P) + P) := by
      field_simp
      ring
    unfold proximityRatio
    rw [hEval]
    field_simp

/-! ## Background -/

/-- The apparent efficiency computed from photon counts with background in both channels. -/
noncomputable def apparentEff (nD nA bD bA : ℝ) : ℝ := (nA + bA) / (nA + bA + nD + bD)

/-- **A zero-transfer state is measured at strictly positive efficiency.**  With any acceptor
background at all, the extended states of a disordered region are biased towards compaction. -/
theorem apparentEff_pos_of_background {nD bD bA : ℝ} (hD : 0 ≤ nD) (hbD : 0 ≤ bD)
    (hbA : 0 < bA) : 0 < apparentEff nD 0 bD bA := by
  unfold apparentEff
  have : 0 < 0 + bA + nD + bD := by linarith
  positivity

/-! ## The linker -/

/-- **The dye--dye distance brackets the residue--residue distance to `±2L`.**  Nothing but the
triangle inequality. -/
theorem linker_bound {X : Type*} [MetricSpace X] {a1 a2 b1 b2 : X} {L : ℝ}
    (h1 : dist a1 b1 ≤ L) (h2 : dist a2 b2 ≤ L) :
    |dist b1 b2 - dist a1 a2| ≤ 2 * L := by
  have hup : dist b1 b2 ≤ dist a1 a2 + 2 * L := by
    have h := dist_triangle4 b1 a1 a2 b2
    rw [dist_comm b1 a1] at h
    linarith
  have hlo : dist a1 a2 ≤ dist b1 b2 + 2 * L := by
    have := dist_triangle4 a1 b1 b2 a2
    rw [dist_comm b2 a2] at this
    linarith
  rw [abs_le]
  constructor <;> linarith

/-- **And the bracket is attained at both ends.**  Two configurations with the same attachment
distance whose dye distances differ by `4L`: the linker correction is a range, not a function of
the attachment distance. -/
theorem linker_bound_sharp {d L : ℝ} (hL : 0 < L) (hd : 2 * L ≤ d) :
    dist (0:ℝ) d = d ∧
    (dist (0:ℝ) (-L) ≤ L ∧ dist d (d + L) ≤ L ∧ dist (-L : ℝ) (d + L) = d + 2 * L) ∧
    (dist (0:ℝ) L ≤ L ∧ dist d (d - L) ≤ L ∧ dist (L : ℝ) (d - L) = d - 2 * L) := by
  have hd0 : (0:ℝ) ≤ d := by linarith
  refine ⟨?_, ⟨?_, ?_, ?_⟩, ?_, ?_, ?_⟩
  · rw [Real.dist_eq, abs_of_nonpos (by linarith)]; ring
  · rw [Real.dist_eq, abs_of_nonneg (by linarith)]; linarith
  · rw [Real.dist_eq, abs_of_nonpos (by linarith)]; linarith
  · rw [Real.dist_eq, abs_of_nonpos (by linarith)]; ring
  · rw [Real.dist_eq, abs_of_nonpos (by linarith)]; linarith
  · rw [Real.dist_eq, abs_of_nonneg (by linarith)]; linarith
  · rw [Real.dist_eq, abs_of_nonpos (by linarith)]; ring

end Photophysics
end IDR
