/-
# Part XXIX.1  The finite periodic box compacts the ensemble

Every simulated ensemble of a disordered region is generated in a periodic cell of finite side
`L`, in which the chain interacts with its own periodic images.  For a folded domain this is a
small correction; for a disordered region whose end-to-end distance is comparable with the box
it is not, and the direction of the artefact is universal.  This file proves that direction
from a single inequality.

* `sum_antivary_le` -- a weighted Chebyshev inequality: if `f` is *antitone in* `R` (larger
  `R` never has larger `f`) then `(Σ w R f)(Σ w) ≤ (Σ w R)(Σ w f)`.
* `wmean_tilt_le` -- consequently, reweighting any ensemble by a positive factor that decreases
  with `R` can only decrease `⟨R⟩`.  This is a general statement about Boltzmann tilts and is
  used twice below.
* `boxMean_le_freeMean` -- **the box compacts.**  If the image interaction `g` is nondecreasing
  in the chain dimension `R` -- larger conformations sit closer to their images and pay more,
  the defining property of a repulsive excluded-volume/electrostatic image term -- then the
  ensemble sampled in the box has `⟨R⟩_box ≤ ⟨R⟩_∞` at every inverse temperature `β ≥ 0`.
* `boxMean_lt_freeMean_two` -- an explicit two-state instance with a strict gap at every
  `β > 0`, so this is a genuine bias and not a boundary case.
* `boxMean_mono_in_box` -- **and it is monotone in the box size.**  If going from box `L` to
  the larger box `L'` weakens the image term by an amount that is itself nondecreasing in `R`,
  then `⟨R⟩_L ≤ ⟨R⟩_{L'}`: the reported dimension of a disordered region increases with the
  box until the images are gone.  A single box size therefore cannot validate an ensemble;
  the box-size dependence must be shown to be flat.

The statements are about *any* ensemble, not a particular force field: no amount of sampling,
and no reweighting against a global restraint that is itself computed in the same box, removes
the artefact.
-/
import Mathlib

set_option autoImplicit false

namespace Box

open Finset

variable {m : ℕ}

/-- Weighted mean of `R` under unnormalised nonnegative weights `w`. -/
noncomputable def wmean (w R : Fin m → ℝ) : ℝ := (∑ k, w k * R k) / ∑ k, w k

/-- `f` is antitone in `R`: a conformation with larger `R` never has a larger `f`. -/
def AntitoneIn (R f : Fin m → ℝ) : Prop := ∀ i j, R i ≤ R j → f j ≤ f i

/-- **Weighted Chebyshev inequality.**  If `f` is antitone in `R` then `R` and `f` are
negatively correlated under every nonnegative weighting. -/
theorem sum_antivary_le {w R f : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (hRf : AntitoneIn R f) :
    (∑ k, w k * (R k * f k)) * (∑ k, w k) ≤ (∑ k, w k * R k) * ∑ k, w k * f k := by
  have hterm : ∀ i j : Fin m, w i * w j * ((R i - R j) * (f i - f j)) ≤ 0 := by
    intro i j
    have hprod : (R i - R j) * (f i - f j) ≤ 0 := by
      rcases le_total (R i) (R j) with h | h
      · have : f j ≤ f i := hRf i j h
        nlinarith
      · have : f i ≤ f j := hRf j i h
        nlinarith
    exact mul_nonpos_of_nonneg_of_nonpos (mul_nonneg (hw i) (hw j)) hprod
  have hrow : ∀ i : Fin m, ∑ j, w i * w j * ((R i - R j) * (f i - f j))
      = w i * (R i * f i * (∑ k, w k) - R i * (∑ k, w k * f k) - f i * (∑ k, w k * R k)
        + ∑ k, w k * (R k * f k)) := by
    intro i
    have hpt : ∀ j, w i * w j * ((R i - R j) * (f i - f j))
        = w i * (R i * f i * w j - R i * (w j * f j) - f i * (w j * R j) + w j * (R j * f j)) := by
      intro j; ring
    rw [Finset.sum_congr rfl fun j _ => hpt j, ← Finset.mul_sum]
    congr 1
    rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, Finset.sum_sub_distrib,
      ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum]
  have htotal : ∑ i, ∑ j, w i * w j * ((R i - R j) * (f i - f j))
      = 2 * ((∑ k, w k * (R k * f k)) * (∑ k, w k)
        - (∑ k, w k * R k) * ∑ k, w k * f k) := by
    rw [Finset.sum_congr rfl fun i _ => hrow i]
    have hpt : ∀ i, w i * (R i * f i * (∑ k, w k) - R i * (∑ k, w k * f k)
          - f i * (∑ k, w k * R k) + ∑ k, w k * (R k * f k))
        = (∑ k, w k) * (w i * (R i * f i)) - (∑ k, w k * f k) * (w i * R i)
          - (∑ k, w k * R k) * (w i * f i) + (∑ k, w k * (R k * f k)) * w i := by
      intro i; ring
    rw [Finset.sum_congr rfl fun i _ => hpt i, Finset.sum_add_distrib, Finset.sum_sub_distrib,
      Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum,
      ← Finset.mul_sum]
    ring
  have hle : ∑ i, ∑ j, w i * w j * ((R i - R j) * (f i - f j)) ≤ 0 :=
    Finset.sum_nonpos fun i _ => Finset.sum_nonpos fun j _ => hterm i j
  rw [htotal] at hle
  linarith

/-- **A tilt that decreases with `R` decreases `⟨R⟩`.** -/
theorem wmean_tilt_le {w R f : Fin m → ℝ} (hw : ∀ k, 0 ≤ w k) (hf : ∀ k, 0 < f k)
    (hpos : 0 < ∑ k, w k) (hRf : AntitoneIn R f) :
    wmean (fun k => w k * f k) R ≤ wmean w R := by
  have hwf : 0 < ∑ k, w k * f k := by
    obtain ⟨k0, hk0⟩ : ∃ k, 0 < w k := by
      by_contra hcon
      push_neg at hcon
      exact absurd hpos (not_lt.mpr (Finset.sum_nonpos fun k _ => hcon k))
    exact lt_of_lt_of_le (mul_pos hk0 (hf k0))
      (Finset.single_le_sum (fun k _ => mul_nonneg (hw k) (hf k).le) (Finset.mem_univ k0))
  have hkey := sum_antivary_le (w := w) (R := R) (f := f) hw hRf
  have hnum : ∑ k, w k * f k * R k = ∑ k, w k * (R k * f k) :=
    Finset.sum_congr rfl fun k _ => by ring
  rw [wmean, wmean, hnum, div_le_div_iff₀ hwf hpos]
  exact hkey

/-! ### The periodic image term -/

/-- Boltzmann weights of the ensemble `w` tilted by the image energy `g` at inverse
temperature `beta`. -/
noncomputable def boxWeights (beta : ℝ) (w g : Fin m → ℝ) : Fin m → ℝ :=
  fun k => w k * Real.exp (-beta * g k)

/-- The chain dimension reported by a simulation in the box. -/
noncomputable def boxMean (beta : ℝ) (w g R : Fin m → ℝ) : ℝ := wmean (boxWeights beta w g) R

/-- **The box compacts the ensemble.**  If the image interaction is nondecreasing in the chain
dimension, the mean dimension measured in the box never exceeds the free-chain value. -/
theorem boxMean_le_freeMean {beta : ℝ} (hbeta : 0 ≤ beta) {w g R : Fin m → ℝ}
    (hw : ∀ k, 0 ≤ w k) (hpos : 0 < ∑ k, w k) (hg : ∀ i j, R i ≤ R j → g i ≤ g j) :
    boxMean beta w g R ≤ wmean w R := by
  refine wmean_tilt_le hw (fun k => Real.exp_pos _) hpos ?_
  intro i j hij
  have : -beta * g j ≤ -beta * g i := by
    have := hg i j hij
    nlinarith
  exact Real.exp_le_exp.mpr this

/-- **Strictly, in the smallest instance.**  Two conformations of dimension `1` and `2` with
equal free-chain weights and image energies `0` and `1`: the box reports a strictly smaller
mean dimension at every positive inverse temperature. -/
theorem boxMean_lt_freeMean_two {beta : ℝ} (hbeta : 0 < beta) :
    boxMean beta (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![0, 1] : Fin 2 → ℝ) (![1, 2] : Fin 2 → ℝ)
      < wmean (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 2] : Fin 2 → ℝ) := by
  have hx : Real.exp (-beta * 1) < 1 := by
    rw [show -beta * (1 : ℝ) = -beta by ring]
    exact Real.exp_lt_one_iff.mpr (by linarith)
  have hxpos : 0 < Real.exp (-beta * 1) := Real.exp_pos _
  set x := Real.exp (-beta * 1) with hxdef
  have hfree : wmean (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![1, 2] : Fin 2 → ℝ) = 3 / 2 := by
    simp [wmean, Fin.sum_univ_two]
    norm_num
  have hbox : boxMean beta (![1 / 2, 1 / 2] : Fin 2 → ℝ) (![0, 1] : Fin 2 → ℝ)
      (![1, 2] : Fin 2 → ℝ) = (1 / 2 + x) / (1 / 2 + x / 2) := by
    simp [boxMean, wmean, boxWeights, Fin.sum_univ_two, hxdef]
    ring_nf
  rw [hfree, hbox, div_lt_iff₀ (by linarith)]
  linarith

/-- **The artefact is monotone in the box size.**  If enlarging the box lowers the image energy
by an amount `g - g'` that is itself nondecreasing in `R` (the images matter most for the
largest conformations), then the smaller box reports the smaller mean dimension. -/
theorem boxMean_mono_in_box {beta : ℝ} (hbeta : 0 ≤ beta) {w g g' R : Fin m → ℝ}
    (hw : ∀ k, 0 ≤ w k) (hpos : 0 < ∑ k, w k)
    (hdiff : ∀ i j, R i ≤ R j → g i - g' i ≤ g j - g' j) :
    boxMean beta w g R ≤ boxMean beta w g' R := by
  have hrewrite : boxWeights beta w g
      = fun k => boxWeights beta w g' k * Real.exp (-beta * (g k - g' k)) := by
    funext k
    simp only [boxWeights]
    rw [mul_assoc, ← Real.exp_add]
    ring_nf
  have hposw : 0 < ∑ k, boxWeights beta w g' k := by
    obtain ⟨k0, hk0⟩ : ∃ k, 0 < w k := by
      by_contra hcon
      push_neg at hcon
      exact absurd hpos (not_lt.mpr (Finset.sum_nonpos fun k _ => hcon k))
    refine lt_of_lt_of_le (mul_pos hk0 (Real.exp_pos (-beta * g' k0))) ?_
    exact Finset.single_le_sum
      (f := fun k => boxWeights beta w g' k)
      (fun k _ => mul_nonneg (hw k) (Real.exp_pos _).le) (Finset.mem_univ k0)
  rw [boxMean, hrewrite]
  refine wmean_tilt_le (fun k => mul_nonneg (hw k) (Real.exp_pos _).le)
    (fun k => Real.exp_pos _) hposw ?_
  intro i j hij
  have : -beta * (g j - g' j) ≤ -beta * (g i - g' i) := by
    have := hdiff i j hij
    nlinarith
  exact Real.exp_le_exp.mpr this

end Box
