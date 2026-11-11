# Designing the experiment, not just the model (Parts CXLII–CXLV)

Files: `RequestProject/TitrationLadder.lean`, `RequestProject/SaltResponsiveEnsemble.lean`,
`RequestProject/EnsembleFunctionals.lean`, `RequestProject/LadderSpacing.lean`.  Everything below is
proved in Lean with no `sorry`, and every theorem depends only on the standard axioms `propext`,
`Classical.choice`, `Quot.sound`.

Part CXLI ended with the requirement that a model of a charged disordered region be calibrated
against a *distribution* of internal distances per sequence separation, identified by a **complete**
salt titration — agreement of the curves at every ionic strength on a half-line.  A complete
titration is not an experiment.  These four parts replace it by the experiment an experimenter can
actually run: finitely many salt conditions, prepared at a chosen spacing, read at finite precision,
on an ensemble that itself responds to the salt.  Throughout, the reading is the ensemble kernel of
Part CXLI, `measCurve T w κ = ∑_{t ∈ T} w t · e^{−κt}/t`, and a ladder condition is
`ladderPoint κ₀ h j = κ₀ + j·h`.

## Part CXLII — How many salt conditions, and where?

*File: `TitrationLadder.lean`.*

The key observation is that at **equally spaced** conditions the screening factors become powers,
`e^{−κ_j t} = e^{−κ₀t}·(e^{−ht})^j`, so the design matrix is a Vandermonde matrix in the distinct
nodes `e^{−ht}` and can be inverted exactly by Lagrange interpolation
(`power_amplitudes_zero`, `ladder_amplitudes_zero`).

* `ladder_identifies_distribution` — **finitely many conditions suffice.**  An arithmetic ladder of
  `n` ionic strengths determines every weight of a distance distribution supported on at most `n`
  distances.  No limit, no half-line, and the reachable ionic-strength range is irrelevant; only the
  spacing being uniform and the conditions being enough of them matter.
* `ensemble_ladder_design` — the practical corollary: `2m` equally spaced conditions separate any
  two ensembles of at most `m` conformers each.
* `finite_conditions_insufficient` — **and the count is essentially optimal.**  For *any* `n` ionic
  strengths, chosen however one likes, and any `n+2` candidate distances, there are two *different*
  strictly positive probability distributions on those distances with identical readings at every
  condition.  The obstruction is dimensional: `n` readings plus the normalisation are `n+1` linear
  constraints on `n+2` weights, and the kernel vector is scaled small enough to keep every weight
  positive, so the two ensembles are physically legitimate.
* `two_distance_resolution_horizon` — **identifiability is not resolvability.**  For two distances
  `r₁ ≠ r₂` and any tolerance `δ > 0` there are two strictly positive weightings whose readings
  agree to within `δ` at both ladder conditions (exactly, at the first) while their weights at `r₁`
  differ by exactly `δ·r₁·e^{κ₀r₁}/|e^{−hr₁} − e^{−hr₂}|`.  `resolution_horizon_unbounded` shows
  that factor exceeds any prescribed bound once the distances are close enough.
* `titration_ladder_law` collects sufficiency, necessity and conditioning.

## Part CXLIII — The ensemble responds to the salt

*File: `SaltResponsiveEnsemble.lean`.*

Parts CXLI–CXLII treat the distance distribution as fixed and the ionic strength as changing only
how it is *read*.  A polyelectrolyte violates that: screening reweights the conformers.

* `arbitrary_salt_response_unidentifiable` — **with an unconstrained response the titration is
  vacuous.**  A single internal distance whose weight may depend freely on ionic strength reproduces
  *any* target curve exactly, with strictly positive weight wherever the curve is positive.  The
  identification theorems are therefore theorems about the declared response model.
* `ladder_weight_stability` — **the quantitative inverse.**  If the ladder readings of two
  distributions agree to within `δ`, their weights agree at every distance `t` to within
  `t·e^{κ₀t}·nodeAmp·δ`, where `nodeAmp` (`nodeAmp`) is the ℓ¹ norm of the Lagrange basis
  coefficients at the node of `t` — the exact amplification constant of the inversion.  At `δ = 0`
  this is Part CXLII's exact identification.
* `drift_bias_bound` — **slow response is a bounded bias.**  If the true weights drift with salt by
  at most `ε` at each distance across the ladder, and a *static* distribution is fitted to the data
  and reproduces it exactly, the fitted weights differ from the reference by at most
  `t·e^{κ₀t}·nodeAmp·(ε·S)` with `S = ∑_s e^{−κ₀s}/s` the kernel mass at the lowest condition.
* `salt_response_law` collects the three.

## Part CXLIV — Which properties of the ensemble are actually measured

*File: `EnsembleFunctionals.lean`.*

A model is rarely asked for a weight; it is asked for a property of the ensemble, i.e. a linear
functional `L(w) = ∑_t c_t w_t`.

* `functional_stability` — **the reliability of a property is the ℓ¹ norm of its representer.**  If
  a polynomial `P` of degree below the number of conditions solves the representer equation
  `P(e^{−ht}) = c_t · t · e^{κ₀t}` at every distance, then two distributions whose ladder readings
  agree to within `δ` agree on `L` to within `representerNorm P n · δ`.  Part CXLIII's weight bound
  is the case of a point mass, whose representer is a Lagrange basis polynomial.
* `screened_population_stable` — the screened population read at the `j`-th condition has
  representer `X^j` and hence amplification exactly `1`.  These are the properties a titration
  measures outright; everything else is read through its representer, and Part CXLII showed the
  representer norm of an individual weight of nearby distances is unbounded.
* `ensemble_functional_law` collects the two, with the rule: report the property *and* its
  representer norm.  A property with a large representer norm is a prediction of the model, not a
  measurement.

## Part CXLV — Where to put the conditions

*File: `LadderSpacing.lean`.*

The spacing `h` is free, and the node gap depends on it non-monotonically: small `h` puts both nodes
near `1`, large `h` puts both near `0`.

* `optimalSpacing r₁ r₂ = log(r₂/r₁)/(r₂ − r₁)` and `optimalSpacing_pos`.
* `node_gap_maximised` — the node gap `|e^{−hr₁} − e^{−hr₂}|` is largest at exactly that spacing.
  The turning point is where `r₂e^{−hr₂} = r₁e^{−hr₁}`, i.e. `log r₂ − hr₂ = log r₁ − hr₁`, and the
  gap increases below it and decreases above it.
* `amplification_minimised_at_optimal_spacing` — hence the resolution horizon of Part CXLII, being
  inversely proportional to the gap, is smallest there: no ladder at any spacing separates the two
  distances better.  A titration meant to resolve two conformers should step the inverse Debye
  length by `log(r₂/r₁)/(r₂ − r₁)` — a step set by the *ratio* of the distances as well as their
  difference.
* `ladder_spacing_law` collects the statement, and `optimalSpacing_twenty_twentyfive` works out a
  concrete number: to separate `20 Å` from `25 Å` the optimal step in inverse Debye length is
  `log(1.25)/5`, proved to lie between `0.04 Å⁻¹` and `0.05 Å⁻¹`.

## What this adds to the design of a model

Four requirements, on top of Parts CXXXIV–CXLI:

1. **Count the conditions against the conformers.**  A model claiming `m` distinguishable
   conformers at a separation needs at least `m` salt conditions and, if they are equally spaced,
   `2m` of them always suffice to falsify a wrong distribution.  Below that count the data cannot
   distinguish the claim from an infinity of alternatives with strictly positive weights.
2. **Quote a precision, not just an identification.**  Exact identification is a noiseless
   statement.  At precision `δ` the weight error is `t·e^{κ₀t}·nodeAmp·δ`, and for two nearby
   distances that constant is unbounded: "we resolve two conformers" is meaningless without the
   node gap and the precision.
3. **Declare the salt response.**  If the ensemble may reweight arbitrarily with ionic strength, a
   titration constrains nothing at all; if the response is asserted to be weak, the residual bias
   of a static analysis is bounded by the drift times the same amplification constant.
4. **Report properties with their representer norms, and space the ladder deliberately.**  The
   screened populations are measured with unit amplification; any other property costs its
   representer norm; and the spacing that minimises that cost for a target pair of distances is
   `log(r₂/r₁)/(r₂ − r₁)`.
