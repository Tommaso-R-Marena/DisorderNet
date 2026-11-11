# From a declared distance law to a measured distance distribution (Parts CXXXIX–CXLI)

Files: `RequestProject/SwellingExponent.lean`, `RequestProject/DistanceProfile.lean`,
`RequestProject/DistanceEnsemble.lean`.  Everything below is proved in Lean, with no `sorry`, and
every theorem depends only on the standard axioms `propext`, `Classical.choice`, `Quot.sound`.

Part CXXXVIII ended with the instruction that a model of a charged disordered region must
*declare* its distance law — `R(d) = b√d` for an ideal chain — and showed that the bond length
inside that law is confounded with sequence separation unless the correlation profile is known
independently.  These three parts take the realism of the distance law apart, one layer at a time:
the ideal chain becomes a polymer with solvent quality, the parametric family disappears
altogether, and finally the single distance per separation becomes a distribution over the
conformational ensemble.

## Part CXXXIX — Solvent quality is part of the distance law

Real disordered regions are not ideal chains: their internal distances follow `R(d) = b·d^ν`, with
`ν ≈ 3/5` for a swollen region in good solvent, `ν = 1/2` at the theta point, `ν ≈ 1/3` for a
collapsed globule.  The titration reading becomes
`swellCurve N b ν κ c = ∑_{d=1}^{N−1} c d · e^{−κ b d^ν}/(b d^ν)`, which contains Part CXXXVIII's
kernel at `ν = 1/2` (`swellCurve_half_eq_debyeCurve`).

* `swell_profile_identifiable` — nothing is lost by moving to the realistic law: with `b` and `ν`
  declared, a complete titration still determines every charge correlation in the window.
* `even_lag_swelling_confound` — the Part CXXXVIII confound survives, deformed: a profile supported
  on even separations, read at `(b, ν)`, is reproduced exactly by the profile of its even lags read
  at `(b·2^ν, ν)`.  The ideal-chain factor `√2` is the special case of `2^ν`.
* `single_lag_exponent_confound` — a single active separation determines *nothing* about solvent
  quality: for every exponent `ν'` the bond length `b·D^ν/D^{ν'}` reproduces the curve identically.
  `swollen_and_collapsed_indistinguishable_at_one_lag` makes it concrete — a good-solvent model
  (`ν = 3/5`) and a collapsed globule (`ν = 1/3`), at different bond lengths, give literally the
  same titration curve at every ionic strength.
* `swell_bond_and_exponent_identified` — **two** active separations suffice.  If the correlation
  profile is known independently (the bond-length-free panel of Part CXXXVI) and is non-zero at
  separations `1` and `2`, a complete titration determines the bond length *and* the exponent.  The
  mechanism is a two-stage slowest-rate argument: separation `1` has rate `b·1^ν = b` whatever the
  exponent, so it fixes `b`; with `b` known those terms cancel identically and separation `2`, rate
  `b·2^ν`, fixes `ν`.  Hence `good_solvent_distinguishable_from_theta`.
* `exponent_difference_bound` and `solvent_quality_resolution_horizon` — the price.  Two exponents
  `ν, ν' ≥ ν_min` with the same profile differ by at most
  `2·M·e^{−κ b 2^{ν_min}}/(b 2^{ν_min})`, where `M` is the correlation mass beyond the first
  separation.  So above `κ* = log(2M/(b·2^{ν_min}·ε))/(b·2^{ν_min})` no pair of exponents is
  separable at precision `ε`: exact identification is a low-salt statement, and the cost of moving
  up the titration is exponential.

## Part CXL — The internal-distance profile, with no chain model at all

Local stiffness, prolines, transient helices and charge blocks deform `R(d)` away from any
two-parameter form, and it is `R` itself that a model must reproduce.  So drop the family: the
curve is `kCurve 1 N R c κ = ∑_{d=1}^{N−1} c d · e^{−κ R d}/(R d)` for an arbitrary positive,
strictly increasing profile `R`.

* `internal_distance_profile_identifiable` — with the correlation profile known and vanishing
  nowhere, agreement of the titration curves at every ionic strength forces two such profiles to
  agree at *every* separation.  The distance law is data, not a modelling choice.  The proof is an
  induction on separation: the shortest separation carries the slowest screening rate and so is
  determined first; its term then cancels identically and the next separation becomes the slowest.
* `swap_rates_confound` — monotonicity is not a technicality.  Two separations carrying *equal*
  correlations may exchange their distances with no effect on the curve at any ionic strength; the
  physical requirement that distance grow with separation is exactly what removes this degeneracy.
* `polymer_params_of_rates` — matching rates at separations `1` and `2` already pins `b` and `ν`,
  so Part CXXXIX's identification theorem is a corollary, and one sees what it was using: two
  active short separations and a monotone law.

## Part CXLI — A disordered region is an ensemble

One distance per separation is still one structure.  A disordered region realises a *distribution*
of distances at each separation, and the kernel is averaged over it:
`ensCurve S r p κ = ∑_{k ∈ S} p k · e^{−κ r k}/(r k)`.

* `ens_weights_identifiable` — over a fixed finite set of distinct positive candidate distances, a
  complete titration determines the weight of every candidate.  The experiment measures the
  distance *distribution*, not a mean or an apparent distance.
* `distance_distribution_identifiable` — the candidate set is not needed either: two finitely
  supported distributions of positive distances whose curves agree at every ionic strength have the
  same weight at every distance, so the support is identified along with the weights.  The tool is
  `realRate_amplitudes_zero`, the independence lemma re-indexed by the rates themselves.
* `two_conformer_not_single_distance` — if a separation realises two distinct distances with
  positive weights then for *every* candidate single distance `R` some ionic strength separates the
  single-distance model from the ensemble.  A single-structure model of a disordered region is
  falsifiable, and a complete titration falsifies it.
* `mean_distance_model_falsified` — in particular the single-distance model placed at the
  ensemble's mean distance is refuted: `⟨R⟩` is not what the experiment reports.
* Tools: `two_exp_indep`, `three_exp_indep`, the two- and three-rate specialisations of the
  exponential-independence lemma of Part CXXXVIII.

## What this adds to the design of a model

Four requirements, on top of Parts CXXXIV–CXXXVIII:

1. If a scaling exponent is quoted, say from how many active separations it was fitted.  From one
   separation it is prior, not measurement: bond length absorbs it exactly.
2. Quote the ionic-strength range over which the exponent was read.  Above the explicit horizon
   `κ*` above, solvent quality is below the precision of the experiment, whatever the number of
   conditions.
3. Do not treat the distance law as an assumption.  With an independently measured, nowhere-zero
   correlation profile the whole profile `R(d)` is identifiable, so a parametric law should be
   reported as a *summary* of a calibrated profile — and monotone in separation, since without
   monotonicity separations with equal correlations may swap their distances undetectably.
4. Calibrate a distribution, not a structure.  The weights of the internal-distance distribution
   at a separation are identified by a complete titration; a model reporting one distance per
   separation — the mean included — makes a prediction that the data can refute.
