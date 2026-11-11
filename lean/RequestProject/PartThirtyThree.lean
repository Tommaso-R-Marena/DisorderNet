/-
# Part XXXIII  Single-molecule histograms: how much of the width is the molecule?

Single-molecule FRET is the experiment most often quoted as direct evidence that a disordered
region is heterogeneous.  `RequestProject.PhotonCounting` derives, from the photon statistics of
a burst, exactly how much of a histogram's width belongs to the ensemble.

The burst model is elementary and explicit: `N` photons, each detected in the acceptor channel
with the transfer efficiency of the conformation emitting them, so that the acceptor count is
binomial (`Photon.binom`, with its normalisation, mean and variance obtained from Mathlib's
Bernstein-polynomial identities).  Everything else is a theorem.

* The histogram is centred correctly (`measured_unbiased`) but its width obeys an exact law of
  total variance, `Var = varConf + shotNoise` (`shotnoise_decomposition`).
* A single conformation therefore produces a histogram of strictly positive width
  (`homogeneous_histogram_has_width`), and an explicit pair of experiments — one homogeneous
  with 100 photons per burst, one genuinely two-state with 276 — produce histograms of exactly
  the same variance `1/400` (`width_not_evidence_of_heterogeneity`).
* Conversely the shot-noise term is at most `1/(4N)`, so excess width *is* evidence
  (`heterogeneity_detected`) and a photon budget `N ≥ 1/(4·varConf)` suffices to see the
  ensemble (`photons_needed`).
* And if the chain interconverts faster than the burst lasts, the conformational term is
  averaged away while the detector term is not (`dynamic_averaging`): a *narrow* histogram is no
  more evidence of homogeneity than a broad one is of heterogeneity.

`IDR.single_molecule_laws` bundles the six statements.
-/
import Mathlib
import RequestProject.PhotonCounting

set_option autoImplicit false

namespace IDR

/-- **The design laws of a single-molecule histogram.**

1. *Unbiased centre*: the mean measured efficiency is the ensemble mean efficiency.
2. *Exact law of total variance*: histogram width = conformational width + shot noise.
3. *Width is not evidence*: a homogeneous ensemble at 100 photons per burst and a two-state
   ensemble at 276 photons per burst give the same histogram variance `1/400`.
4. *Excess width is evidence*: whatever the photon budget, the histogram variance minus
   `1/(4N)` is a lower bound on the conformational variance.
5. *What it costs to see the ensemble*: `N ≥ 1/(4·varConf)` photons per burst put the detector
   term below the conformational term.
6. *Dynamic averaging*: if a burst samples two independent conformations, the conformational
   term is halved while the shot noise is unchanged. -/
theorem single_molecule_laws :
    -- 1  the histogram is centred on the ensemble mean efficiency
    (∀ (m N : ℕ), 0 < N → ∀ w eff : Fin m → ℝ,
        Photon.measuredMean w eff N = Photon.meanEff w eff) ∧
    -- 2  its width is the conformational width plus shot noise
    (∀ (m N : ℕ), 0 < N → ∀ w eff : Fin m → ℝ,
        Photon.measuredVar w eff N = Photon.varConf w eff + Photon.shotNoise w eff N) ∧
    -- 3  width alone does not distinguish a homogeneous from a heterogeneous ensemble
    (Photon.varConf ![1] ![1/2] = 0 ∧
      Photon.varConf ![1/2, 1/2] ![23/50, 27/50] = 1/625 ∧
      Photon.measuredVar ![1] ![(1/2 : ℝ)] 100 = 1/400 ∧
      Photon.measuredVar ![1/2, 1/2] ![(23/50 : ℝ), 27/50] 276 = 1/400) ∧
    -- 4  excess width is a lower bound on the conformational variance
    (∀ (m N : ℕ), 0 < N → ∀ w eff : Fin m → ℝ, (∀ k, 0 ≤ w k) → ∑ k, w k = 1 →
        Photon.measuredVar w eff N - 1 / (4 * N) ≤ Photon.varConf w eff) ∧
    -- 5  the photon budget that resolves the ensemble
    (∀ (m N : ℕ), 0 < N → ∀ w eff : Fin m → ℝ, (∀ k, 0 ≤ w k) → ∑ k, w k = 1 →
        0 < Photon.varConf w eff → 1 / (4 * Photon.varConf w eff) ≤ N →
        Photon.shotNoise w eff N ≤ Photon.varConf w eff) ∧
    -- 6  dynamic averaging inside a burst destroys the conformational width, not the shot noise
    (∀ (m n : ℕ), 0 < n → ∀ w eff : Fin m → ℝ, ∑ k, w k = 1 →
        Photon.splitVar w eff n = Photon.varConf w eff / 2 + Photon.shotNoise w eff (2 * n)) := by
  refine ⟨fun m N hN w eff => Photon.measured_unbiased hN w eff,
    fun m N hN w eff => Photon.shotnoise_decomposition hN w eff,
    Photon.width_not_evidence_of_heterogeneity,
    fun m N hN w eff hw hsum => Photon.heterogeneity_detected hN hw hsum,
    fun m N hN w eff hw hsum hv hbudget => Photon.photons_needed hN hw hsum hv hbudget,
    fun m n hn w eff hw => Photon.dynamic_averaging hn hw⟩

end IDR
