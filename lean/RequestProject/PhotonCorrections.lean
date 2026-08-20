/-
# Part CI  What the detector actually counts: leakage, direct excitation, gamma and background

Parts XXXIII and LXI treat single-molecule FRET with clean photon counting: a burst is `N` photons
with binomial acceptor counts at the conformer's transfer efficiency.  The assumptions list
recorded what was left out — "Background, crosstalk, detector dead time, gamma/beta correction,
the burst-size distribution and dye photophysics are not modelled".  This file models the four
that a corrected efficiency is computed from: donor **leakage** into the acceptor channel `l`,
**direct excitation** of the acceptor `dex`, the detection-efficiency/quantum-yield ratio
**gamma** `g`, and channel **backgrounds** `bA, bD`.

`prPhys` is the raw proximity ratio those parameters produce from a conformer of true efficiency
`E`, and the results are about it.

* `prPhys_moebius` — **the raw ratio is a Möbius function of the true efficiency**,
  `(a·E + b)/(c·E + e)`, with the four coefficients given explicitly in terms of the instrument
  parameters.  Everything else follows from that one fact.
* `pr_strictMono` — it is strictly increasing exactly when `a·e − b·c > 0`, so the ranking of two
  conformers by raw ratio is trustworthy even before correction.
* `corr_prPhys` — **and the correction is exact when the parameters are known**: inverting the
  Möbius map recovers `E` on the nose.  Nothing in this part says the corrected number is wrong.
* `gamma_leakage_unidentifiable` — **what is wrong is computing it from uncalibrated parameters.**
  Two physically distinct instruments — leakage `0.100` with `gamma = 1` and no direct excitation,
  against leakage `0.055` with `gamma = 1.05` and `5%` direct excitation — produce *identical* raw
  ratios at every efficiency.  No amount of data from the experiment itself separates them, and the
  efficiency each assigns to the same burst differs.  Gamma and leakage must come from a
  calibration measurement; this is the FRET form of the calibration floor of Part XC.
* `avg_correction_noncommute` — **and averaging does not commute with correcting.**  For a
  two-conformer ensemble the mean of the raw ratios is `6/11`, while the raw ratio of the mean
  efficiency is `11/21`.  A histogram of *uncorrected* burst ratios is therefore not a distorted
  picture of the efficiency distribution that a single monotone relabelling repairs: the per-burst
  correction has to be applied to each burst, and the mean of a corrected histogram is not the
  correction of the mean.  For a disordered region, whose efficiency distribution is broad by
  construction, this is the regime where the difference is largest.
-/
import Mathlib

set_option autoImplicit false
set_option maxHeartbeats 1000000

namespace IDR.PhotonCorr

/-- The raw (uncorrected) proximity ratio produced by a conformer of true transfer efficiency `E`
on an instrument with donor leakage `l`, direct acceptor excitation `dex`, detection ratio `g`,
channel backgrounds `bA`, `bD`, and burst size `N`. -/
noncomputable def prPhys (N l dex g bA bD E : ℝ) : ℝ :=
  (N * (E + l * (1 - E) + dex) + bA) /
    (N * (E + l * (1 - E) + dex) + bA + N * g * (1 - E) + bD)

/-- The Möbius coefficients of the raw ratio. -/
noncomputable def coefA (N l : ℝ) : ℝ := N * (1 - l)
noncomputable def coefB (N l dex bA : ℝ) : ℝ := N * (l + dex) + bA
noncomputable def coefC (N l g : ℝ) : ℝ := N * (1 - l) - N * g
noncomputable def coefE (N l dex g bA bD : ℝ) : ℝ := N * (l + dex) + bA + N * g + bD

/-- **The raw proximity ratio is a Möbius function of the true efficiency.** -/
theorem prPhys_moebius (N l dex g bA bD E : ℝ) :
    prPhys N l dex g bA bD E =
      (coefA N l * E + coefB N l dex bA) /
        (coefC N l g * E + coefE N l dex g bA bD) := by
  unfold prPhys coefA coefB coefC coefE
  congr 1 <;> ring

/-- A Möbius ratio, in the abstract. -/
noncomputable def mob (a b c e E : ℝ) : ℝ := (a * E + b) / (c * E + e)

/-- The raw ratio is strictly increasing in the efficiency exactly when `a·e − b·c > 0`. -/
theorem pr_strictMono {a b c e E1 E2 : ℝ} (hdet : 0 < a * e - b * c)
    (h1 : 0 < c * E1 + e) (h2 : 0 < c * E2 + e) (hE : E1 < E2) :
    mob a b c e E1 < mob a b c e E2 := by
  unfold mob
  rw [div_lt_div_iff₀ h1 h2]
  nlinarith

/-- Inverting the Möbius map: the corrected efficiency. -/
noncomputable def corr (a b c e P : ℝ) : ℝ := (b - e * P) / (c * P - a)

/-- **With the instrument parameters known, the correction is exact.** -/
theorem corr_mob {a b c e E : ℝ} (hden : c * E + e ≠ 0) (hdet : a * e - b * c ≠ 0) :
    corr a b c e (mob a b c e E) = E := by
  unfold corr
  set P : ℝ := mob a b c e E with hPdef
  have hP : P * (c * E + e) = a * E + b := by
    rw [hPdef]; unfold mob; exact div_mul_cancel₀ _ hden
  have hne : c * P - a ≠ 0 := by
    intro h
    apply hdet
    have hc : c * P = a := by linarith
    have h2 : c * (P * (c * E + e)) = c * (a * E + b) := by rw [hP]
    rw [← mul_assoc, hc] at h2
    linear_combination h2
  rw [div_eq_iff hne]
  linear_combination (-1 : ℝ) * hP

/-- **With the parameters unknown, the raw data do not determine them.**  Two physically distinct
instruments produce the identical raw ratio at every efficiency. -/
theorem gamma_leakage_unidentifiable {E : ℝ} (h1 : E ≤ 1) :
    prPhys 1 (1/10) 0 1 0 0 E = prPhys 1 (11/200) (1/20) (21/20) 0 0 E := by
  have hE : (0:ℝ) < 11 - E := by linarith
  have e1 : prPhys 1 (1/10) 0 1 0 0 E = (9*E + 1)/(11 - E) := by
    unfold prPhys
    rw [div_eq_div_iff (by nlinarith) (by nlinarith)]
    ring
  have e2 : prPhys 1 (11/200) (1/20) (21/20) 0 0 E = (9*E + 1)/(11 - E) := by
    unfold prPhys
    rw [div_eq_div_iff (by nlinarith) (by nlinarith)]
    ring
  rw [e1, e2]

/-- **Averaging and correcting do not commute.**  For the two-conformer ensemble
`{E = 0, E = 1}` with equal weights on the instrument of `gamma_leakage_unidentifiable`, the mean
raw ratio is `6/11` while the raw ratio of the mean efficiency is `11/21`. -/
theorem avg_correction_noncommute :
    (prPhys 1 (1/10) 0 1 0 0 0 + prPhys 1 (1/10) 0 1 0 0 1) / 2 = 6/11 ∧
    prPhys 1 (1/10) 0 1 0 0 (1/2) = 11/21 ∧
    (6:ℝ)/11 ≠ 11/21 := by
  refine ⟨?_, ?_, by norm_num⟩
  · unfold prPhys
    norm_num
  · unfold prPhys
    norm_num

end IDR.PhotonCorr
