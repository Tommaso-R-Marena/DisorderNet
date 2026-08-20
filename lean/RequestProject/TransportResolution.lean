/-
# Finite instrument resolution: what a *binned* histogram still proves

`RequestProject.TransportOneDim` computes the transport distance between two ensembles of
descriptor values exactly, from their cumulative distributions.  A real measurement never
delivers those distributions: a FRET efficiency histogram, a SAXS-derived `Rg`
distribution, an NMR chemical-shift readout all arrive *binned*, at a finite resolution
`w` set by the instrument.  This file quantifies what survives the binning.

* `transportCost_map_le_displacement` -- moving every conformation by at most `d` costs at
  most `d`: the diagonal plan bounds the transport distance by the largest displacement.
* `transportCost_binning_stability` -- consequently, replacing both ensembles by their
  images under *any* map that displaces points by at most `d` changes the transport distance
  by at most `2d`.  Coarse-graining an ensemble description is a `2d`-perturbation of the
  geometry, not a loss of it.
* `binR_displacement` -- rounding a descriptor to the nearest multiple of the bin width `w`
  displaces it by at most `w/2`.
* `resolution_certificate` -- the design law: a *binned* histogram of an `L`-Lipschitz
  descriptor still certifies a structural transport error of at least
  `(cdfL1 - w) / L` ångströms.  The bin width is subtracted from the certificate, one for
  one, before the Lipschitz constant divides it.
* `bin_width_design_rule` -- read as an instrument specification: to certify a structural
  error of `eps` the measured cumulative-histogram discrepancy must exceed `w + L·eps`.  A
  disorder experiment must therefore be designed with `w` well below `L·eps`, and no amount
  of averaging can substitute for resolution.
-/
import Mathlib
import RequestProject.EnsembleCore
import RequestProject.CoarseGraining
import RequestProject.Transport
import RequestProject.TransportGeometry
import RequestProject.TransportProcessing
import RequestProject.TransportTotalVariation
import RequestProject.TransportOneDim

namespace IDR

open Finset
open scoped Classical

variable {X : Type*}

/-! ## Displacing every conformation a little costs a little -/

/-- **The diagonal plan.**  If a relabelling `r` moves every conformation by at most `d` in
the structural metric, the transport distance from an ensemble to its image is at most `d`:
keep every component where it is and pay only for the displacement. -/
theorem transportCost_map_le_displacement {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (E : Ens X) (r : X → X) {d : ℝ} (hr : ∀ x, c x (r x) ≤ d) :
    transportCost c E (E.map r) ≤ d := by
  have hcoup : IsCoupling E (E.map r) (fun i j => if i = j then E.w i else 0) := by
    refine ⟨fun i j => ?_, fun i => ?_, fun j => ?_⟩
    · by_cases h : i = j <;> simp [h, E.w_nonneg]
    · simp
    · show ∑ i, (if i = j then E.w i else 0) = E.w j
      simp
  refine (transportCost_le_of_coupling hc hcoup).trans ?_
  have hcost : planCost E (E.map r) c (fun i j => if i = j then E.w i else 0)
      = ∑ i, E.w i * c (E.pt i) (r (E.pt i)) := by
    simp only [planCost]
    refine Finset.sum_congr rfl fun i _ => ?_
    simp only [ite_mul, zero_mul]
    simp [Ens.map]
  rw [hcost]
  calc ∑ i, E.w i * c (E.pt i) (r (E.pt i))
      ≤ ∑ i, E.w i * d :=
        Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hr _) (E.w_nonneg i)
    _ = d := by rw [← Finset.sum_mul, E.w_sum, one_mul]

/-- **Binning is a bounded perturbation of the geometry.**  If `r` displaces every
conformation by at most `d`, then comparing the two ensembles after applying `r` -- reading
them at finite resolution -- changes the transport distance by at most `2d`. -/
theorem transportCost_binning_stability {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    (hsymm : ∀ x y, c x y = c y x) (htri : ∀ x y z, c x z ≤ c x y + c y z)
    (E F : Ens X) (r : X → X) {d : ℝ} (hr : ∀ x, c x (r x) ≤ d) :
    |transportCost c E F - transportCost c (E.map r) (F.map r)| ≤ 2 * d := by
  have hE : transportCost c E (E.map r) ≤ d := transportCost_map_le_displacement hc E r hr
  have hF : transportCost c F (F.map r) ≤ d := transportCost_map_le_displacement hc F r hr
  have hEs : transportCost c (E.map r) E ≤ d := by
    rwa [transportCost_comm hc hsymm]
  have hFs : transportCost c (F.map r) F ≤ d := by
    rwa [transportCost_comm hc hsymm]
  have h1 : transportCost c E F
      ≤ transportCost c E (E.map r) + transportCost c (E.map r) F :=
    transportCost_triangle hc htri _ _ _
  have h2 : transportCost c (E.map r) F
      ≤ transportCost c (E.map r) (F.map r) + transportCost c (F.map r) F :=
    transportCost_triangle hc htri _ _ _
  have h3 : transportCost c (E.map r) (F.map r)
      ≤ transportCost c (E.map r) E + transportCost c E (F.map r) :=
    transportCost_triangle hc htri _ _ _
  have h4 : transportCost c E (F.map r) ≤ transportCost c E F + transportCost c F (F.map r) :=
    transportCost_triangle hc htri _ _ _
  rw [abs_sub_le_iff]
  constructor <;> linarith

/-! ## Rounding a descriptor to the instrument grid -/

/-- Reading a descriptor at resolution `w`: round it to the nearest multiple of the bin
width. -/
noncomputable def binR (w : ℝ) (x : ℝ) : ℝ := w * (round (x / w) : ℝ)

/-- Binning displaces a descriptor value by at most half a bin. -/
lemma binR_displacement {w : ℝ} (hw : 0 < w) (x : ℝ) : lineCost x (binR w x) ≤ w / 2 := by
  have hw' : w ≠ 0 := ne_of_gt hw
  have hx : x - binR w x = w * (x / w - (round (x / w) : ℝ)) := by
    simp only [binR]
    field_simp
  rw [lineCost, hx, abs_mul, abs_of_pos hw]
  calc w * |x / w - (round (x / w) : ℝ)| ≤ w * (1 / 2) :=
        mul_le_mul_of_nonneg_left (abs_sub_round (x / w)) (le_of_lt hw)
    _ = w / 2 := by ring

/-! ## The design law -/

/-- **What a binned histogram proves.**  Let `h` be a descriptor that is `L`-Lipschitz with
respect to the structural metric, measured at resolution `w`, and let `p` and `q` be the
binned histograms of the true region and of the candidate model.  Then the structural
transport distance between them is at least `(cdfL1 p q - w) / L`.  Finite resolution costs
exactly one bin width off the certificate -- and nothing more. -/
theorem resolution_certificate {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    {L : ℝ} (hL : 0 < L) {h : X → ℝ} (hlip : ∀ x y, |h x - h y| ≤ L * c x y)
    (E F : Ens X) {w : ℝ} (hw : 0 < w)
    (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) {p q : ℕ → ℝ}
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    (hE : ((E.map h).map (binR w)).Same (gridEns n t p hp hps))
    (hF : ((F.map h).map (binR w)).Same (gridEns n t q hq hqs)) :
    (cdfL1 n t p q - w) / L ≤ transportCost c E F := by
  have hdp : transportCost lineCost (E.map h) (F.map h) ≤ L * transportCost c E F :=
    transportCost_map_le hc lineCost_nonneg (fun x x' => by simpa [lineCost] using hlip x x') E F
  have hbin : |transportCost lineCost (E.map h) (F.map h)
      - transportCost lineCost ((E.map h).map (binR w)) ((F.map h).map (binR w))| ≤ 2 * (w / 2) :=
    transportCost_binning_stability lineCost_nonneg lineCost_comm lineCost_triangle
      (E.map h) (F.map h) (binR w) (fun x => binR_displacement hw x)
  have hcdf : transportCost lineCost ((E.map h).map (binR w)) ((F.map h).map (binR w))
      = cdfL1 n t p q := by
    rw [transportCost_congr_same lineCost_nonneg lineCost_self lineCost_triangle hE hF]
    exact transportCost_line_eq_cdfL1 n ht hp hq hps hqs
  rw [hcdf] at hbin
  rw [abs_sub_le_iff] at hbin
  rw [div_le_iff₀ hL]
  linarith [hbin.2]

/-- **The instrument design rule.**  To certify a structural error of at least `eps`
ångströms from a binned histogram of an `L`-Lipschitz descriptor, the measured discrepancy
between the cumulative histograms must exceed the bin width plus `L·eps`.  Resolution enters
the budget additively and cannot be bought back by better statistics. -/
theorem bin_width_design_rule {c : X → X → ℝ} (hc : ∀ x y, 0 ≤ c x y)
    {L : ℝ} (hL : 0 < L) {h : X → ℝ} (hlip : ∀ x y, |h x - h y| ≤ L * c x y)
    (E F : Ens X) {w eps : ℝ} (hw : 0 < w)
    (n : ℕ) {t : ℕ → ℝ} (ht : Monotone t) {p q : ℕ → ℝ}
    (hp : ∀ i, 0 ≤ p i) (hq : ∀ i, 0 ≤ q i)
    (hps : ∑ i ∈ Finset.range n, p i = 1) (hqs : ∑ i ∈ Finset.range n, q i = 1)
    (hE : ((E.map h).map (binR w)).Same (gridEns n t p hp hps))
    (hF : ((F.map h).map (binR w)).Same (gridEns n t q hq hqs))
    (hmeas : w + L * eps ≤ cdfL1 n t p q) :
    eps ≤ transportCost c E F := by
  have hcert := resolution_certificate hc hL hlip E F hw n ht hp hq hps hqs hE hF
  have : eps ≤ (cdfL1 n t p q - w) / L := by
    rw [le_div_iff₀ hL]
    linarith
  linarith

end IDR
