# Fit distributions, not averages

## What this round adds

The earlier transport development (`RequestProject/Transport*.lean`) established that the
right notion of error for an ensemble model of an intrinsically disordered region is a
*transport* distance against a structural metric, made it a genuine metric on ensembles,
and turned a measured discrepancy in the **mean** of a Lipschitz observable into a lower
bound on that distance, in ångströms.

This round asks the next question a designer must answer: *what data does the certificate
need?*  The answer is proved in four steps.

### 1. Averages are provably insufficient — at any panel size

`IDR.no_finite_panel_of_means_certifies` (`RequestProject/TransportPanelLimits.lean`).

> For **any** finite list `f₀, …, f_{m-1}` of real-valued observables — arbitrary functions,
> not required to be continuous, let alone Lipschitz — and **any** error budget `ε > 0`,
> there exist two conformational ensembles `E`, `F` with
> `E.expect (f j) = F.expect (f j)` for every `j`, and `transportCost lineCost E F ≥ ε`.

The construction is explicit: place `m + 2` conformations on a line at spacing `ε·(m+2)²`;
the `m + 1` linear conditions (the `m` panel averages plus normalisation) cannot pin down
`m + 2` populations, so a nonzero signed direction survives; split it symmetrically between
the two ensembles and normalise it so that its cumulative distribution is guaranteed to
reach `1/(2n)` somewhere.

The point is not the familiar statement that averages fail to *determine* an ensemble — the
project already had that in population space — but that the failure is **unbounded in the
structural metric**.  Adding SAXS, plus FRET, plus RDCs, plus chemical shifts, plus any
number of further averaged readouts, never yields a finite bound on structural error.

### 2. One measured distribution suffices, and gives the exact answer

`IDR.transportCost_line_eq_cdfL1` (`RequestProject/TransportOneDim.lean`).

> For two ensembles of descriptor values on a common ordered grid `t 0 ≤ t 1 ≤ …` with
> populations `p` and `q`,
> `transportCost |·-·| = ∑ₖ (t (k+1) − t k)·|cum p (k+1) − cum q (k+1)|`,
> the `L¹` distance between the two cumulative distribution functions.

Both directions come from a single decomposition, proved here from scratch: the cost of
*any* plan is the sum, over the elementary gaps of the grid, of the gap length times the
mass the plan drags across that gap; the marginal constraints force the **net** mass
crossing gap `k` to be the cumulative discrepancy there; hence every plan pays at least
`|cumulative discrepancy|` per gap, and the monotone (quantile) plan — written down
explicitly as an overlap of quantile intervals — pays exactly that.

Two consequences are recorded:

* `IDR.mean_gap_le_cdfL1`: the histogram certificate is never weaker than the mean
  certificate.
* `IDR.bimodal_vs_unimodal_transport`: and it is strictly stronger.  A collapsed/extended
  two-state region and a single-state model placed at the measured mean have *identical*
  means — the mean certificate returns `0` — while their exact transport distance is one
  full grid unit.  This is precisely the failure mode of average-fitted IDR models.

### 3. Finite instrument resolution costs one bin width — and no more

`RequestProject/TransportResolution.lean`.

* `IDR.transportCost_map_le_displacement`: relabelling every conformation by a map that
  moves it at most `d` costs at most `d` (diagonal plan).
* `IDR.transportCost_binning_stability`: hence reading both ensembles through such a map
  changes their transport distance by at most `2d`.
* `IDR.resolution_certificate`: a histogram of an `L`-Lipschitz descriptor **binned at width
  `w`** still certifies a structural transport error of at least `(cdfL1 − w)/L`.
* `IDR.bin_width_design_rule`: read as an instrument specification — to certify `ε` of
  structural error, the measured cumulative-histogram discrepancy must exceed `w + L·ε`.
  Resolution enters additively and cannot be bought back with more averaging.

### 4. A panel of probes certifies additively, not competitively

`RequestProject/TransportFeatures.lean`.

The earlier certificate converts *one* Lipschitz observable into a bound, so a panel of
probes yields one bound per probe and only the largest can be kept.  Instead:

* `IDR.featureCost_le_transportCost`: if the weighted `ℓ¹` distance between the feature
  vectors of two conformations never exceeds their structural distance, then
  `∑ᵢ wᵢ·|⟨Aᵢ⟩_model − ⟨Aᵢ⟩_truth| ≤ transportCost c model truth`.  Every probe
  contributes its full measured discrepancy.
* `IDR.featureCost_le_transportCost_of_lipschitz`: the usable budget form — if probe `i` is
  `Lᵢ`-Lipschitz and `∑ᵢ wᵢLᵢ ≤ 1`, the additive certificate holds.  Probes reporting on
  tightly coupled features must share the weight budget; probes on structurally independent
  features each get their own.
* `IDR.additive_certificate_is_sharp`: and the gain is real — for two structures differing
  by one unit in each of two independent descriptors, the additive certificate returns the
  exact structural distance `2` while either descriptor alone certifies only `1`.

This is a *lower* bound and so is consistent with §1: adding probes strengthens the
certificate monotonically, but never turns it into an upper bound on structural error.

### 5. The capstone

`IDR.distributional_design_law` (`RequestProject/DistributionVerdict.lean`) bundles the
five statements about one and the same quantity.

## Design conclusion

A model of a disordered region must be trained and falsified against measured
*distributions* of structural descriptors, at a bin width small compared with `L·ε` for the
error `ε` one intends to detect.  A panel of ensemble averages, however large, is incapable
in principle of bounding the structural error; a single distribution, at adequate
resolution, determines it.

All statements are proved in Lean 4 with Mathlib, with no `sorry` and with the standard
axioms only (`propext`, `Classical.choice`, `Quot.sound`).
