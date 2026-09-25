# Certificates: what a measurement of a disordered region proves outright

Parts CXLVI–CLII, added in this round, take the development from *what a model must predict* and
*what an experiment can identify* to a third question: **given the numbers an experiment actually
returns, what is provably true of the ensemble, with no model in between?**  Every statement below
is a theorem in the Lean development, proved with no `sorry` and with only the standard axioms
(`propext`, `Classical.choice`, `Quot.sound`).  The formalization treats an ensemble as finitely
many conformations with probability weights and an observable (an internal distance, a transfer
efficiency) attached to each.

## Part CXLVI — one measured average (`RequestProject/PopulationCertificate.lean`)

A polymer supplies one hard constraint free of charge: a chain of `n` residues with bond length `b`
cannot present an internal distance beyond its contour length `B = n·b`.  With that in force, a
single measured mean `mu` pins the population beyond a threshold `a` from *both* sides.

* `markov_pop_upper` — `pop(r ≥ a) ≤ mu/a`, and `markov_pop_upper_sharp` exhibits an ensemble
  attaining it.
* `reverse_markov_lower` — **a mean forces population, it does not merely cap it**:
  `pop(r > a) ≥ (mu − a)/(B − a)`, attained by the two-state ensemble on `{a, B}`
  (`reverse_markov_lower_sharp`).  A mean end-to-end distance that is a sizeable fraction of the
  contour length is by itself a proof that a definite fraction of the ensemble is expanded.
* `population_identified_interval` — the two together, with the lower endpoint attained: the honest
  content of a one-number experiment is an interval of populations, and no narrower one.
* `bhatia_davis` / `bhatia_davis_sharp` — the same average caps heterogeneity:
  `Var ≤ (M − mu)(mu − m)`, attained exactly by the two-state ensemble on the extremes of the
  accessible range.
* `cantelli_bound` / `cantelli_sharp` / `cantelli_lt_chebyshev` — the sharp one-sided second-moment
  certificate `pop(r ≥ mu + a) ≤ Var/(Var + a²)`, strictly better than the Chebyshev bound used
  earlier in the development, and attained.
* `contour_certificate` — the four statements packaged for a region of `n` residues with bond
  length `b`; `near_contour_forces_extension` is the limiting form, in which a mean within `ε` of
  the contour length leaves at most `ε/(B − a)` of the ensemble below `a`.

## Part CXLVII — one measured FRET efficiency (`RequestProject/FretCertificate.lean`)

The transfer efficiency needs no external bound: `E(r) = R₀⁶/(R₀⁶ + r⁶)` lies in `[0,1]` by
construction and decreases strictly with distance, so a threshold in efficiency *is* a threshold in
distance (`eff_lt_eff_iff`).  Applying Part CXLVI to it gives, for any chosen distance `s`:

* `compact_population_certificate` — at least `(Ē − E(s))/(1 − E(s))` of the ensemble is more
  compact than `s`;
* `expanded_population_certificate` — at least `(E(s) − Ē)/E(s)` of it is more expanded than `s`;
* `compact_certificate_sharp` — the compact bound is attained, so it cannot be improved;
* `fret_single_number_law` — the three together;
* `worked_certificate` — with `R₀ = 54 Å`, a measured mean efficiency of `0.5` proves that at least
  45% of the ensemble is more compact than 80 Å.

No polymer model, Gaussian-chain assumption or fitted distribution is used anywhere.

## Part CXLVIII — error bars and joint tightness (`RequestProject/RobustCertificate.lean`)

* `certificate_under_mean_error` — an average known to within `δ` costs the certificate exactly
  `δ/(B − a)`: the bound degrades continuously, and still proves a positive population whenever the
  measured mean exceeds `a + δ`.
* `certificate_conservative_in_contour` / `certificate_monotone_in_contour` — overestimating the
  maximum extension is safe, and only weakens the claim.  The one modelling input is one-sided.
* `equality_forces_two_point_support` — an ensemble saturating the certificate at a threshold has
  every populated conformation either exactly at the threshold or exactly at the contour bound.
* `equality_at_two_thresholds_forces_extended` / `strict_slack_at_some_threshold` — and it cannot
  saturate at two thresholds unless it is the fully extended chain.  The worst case is
  threshold-specific, so reading the certificate across thresholds is strictly stronger than
  reading it at any one.

## Part CXLIX — why FRET calls the region more compact than scattering does (`RequestProject/FretSaxsOrdering.lean`)

Efficiency is a function of the *squared* distance `u = r²`, the variable scattering averages, and
that function `A/(A + u³)` (with `A = R₀⁶`) changes curvature at `2u³ = A`, i.e. at
`r = R₀·2^{−1/6}`.

* `tangent_line_ineq` — the supporting-line inequality on the convex region, proved algebraically:
  the difference factors as `(u − m)²·(m²(3u² + 2mu + m²) − A(u + 2m))` over a positive denominator.
  No differentiation is used.
* `jensen_expanded`, `mean_in_convex_region` — Jensen's inequality for the ensemble, with the
  observation that an ensemble inside the convex region has its mean square distance there too.
* `saxs_fret_ordering` / `apparent_le_rms_sixth` — **the theorem**: for an ensemble whose distances
  all exceed `R₀·2^{−1/6}`, the measured mean efficiency is at least the efficiency at the
  scattering root-mean-square distance; equivalently the FRET-apparent distance never exceeds the
  root-mean-square distance.  `apparent_le_rms_le_sixth` places this strictly inside the general
  moment ordering proved earlier.
* `ordering_needs_expanded_regime` — the hypothesis is necessary: an explicit two-state ensemble
  with a collapsed member (`R₀ = 1`, distances `0` and `1`, equal weights) has mean efficiency
  `3/4`, below the efficiency `8/9` of its own root-mean-square distance.  So an observed FRET size
  *above* the scattering size is not a calibration failure — it is evidence for populated compact
  conformations.

## Part CL — the entropy floor (`RequestProject/EntropyFloor.lean`)

A mean alone never forces conformational entropy: a single conformation at the mean fits it.  A
measured spread does.

* `entropy_ge_two_block` — any block of weight `P` gives `H ≥ 2P(1 − P)` nats, proved from
  `log t ≤ t − 1` alone.
* `mass_above_mean_lower`, `mass_below_mean_lower` — a variance forces mass on both sides of the
  mean, at least `Var/(2B²)` on each, by `Var ≤ B⟨|r − ⟨r⟩|⟩ = 2B⟨(r − ⟨r⟩)₊⟩ ≤ 2B²·pop(r > ⟨r⟩)`.
* `entropy_floor_from_variance` — hence `H ≥ Var²/(2B⁴)` nats, and `entropy_floor_positive`: any
  measured heterogeneity is quantitative evidence of conformational entropy.
* `ordering_cost_lower_bound` — a binding event that renders the region fully ordered destroys all
  of it, so it costs at least `k_B T · Var²/(2B⁴)` of free energy, whatever the mechanism.

## Part CLI — from photon counts to entropy (`RequestProject/ShotNoise.lean`)

A single-molecule experiment reports neither a mean nor a variance but a histogram of burst
efficiencies `k/n`, `k` acceptor photons out of `n`.  The binomial machinery is developed from
scratch here, by a transfer recursion (`binPmf_succ_succ`, `moment_succ`, `binPmf_sum`,
`binPmf_mean`, `binPmf_sq`, `binPmf_dev_sq`).

* `burst_mean_eq` — the histogram is unbiased in the mean: the mean burst efficiency is the
  ensemble mean efficiency for any photon budget.
* `burst_var_decomposition` — and biased in the width by exactly the shot noise:
  `Var(histogram) = Var(ensemble) + ⟨E(1 − E)⟩/n`.
* `ensemble_var_lower` — hence the deconvolution certificate
  `Var(ensemble) ≥ Var(histogram) − 1/(4n)`.
* `homogeneous_histogram_has_width` — which cannot be improved: a homogeneous ensemble at
  efficiency `p` produces a histogram of variance `p(1−p)/n`, so a histogram no broader than shot
  noise is consistent with a single conformation.
* `entropy_floor_from_burst_histogram` — the chain closes.  A histogram broader than shot noise
  certifies conformational entropy at least `(Var(histogram) − 1/(4n))²/2` nats, and through Part
  CL an ordering free energy of at least `k_B T` times that.

The photon budget `n` is thereby a design parameter of the same standing as the number of salt
conditions in Parts CXLII–CXLV: it fixes the smallest heterogeneity, and hence the smallest
conformational entropy, an experiment can certify.

## Part CLII — the verdict (`RequestProject/CertificateVerdict.lean`)

`single_experiment_verdict` states Parts CXLVI–CLI as one theorem about one experiment: a labelled
region, a burst-efficiency histogram at photon budget `n`, and a chosen distance `s`.  It certifies
simultaneously a compact population, an expanded population, heterogeneity beyond the shot-noise
ceiling, a conformational entropy floor, and a free-energy cost of ordering.  `verdict_is_sharp`
records the other half: the compact bound is attained, and a homogeneous ensemble reproduces a
shot-noise-width histogram, so nothing positive is certifiable below that ceiling.

## What these seven parts add to the design

The earlier parts said that a model of a disordered region must output an ensemble, and that no
finite collection of experiments identifies one.  These parts say what to do about it: report the
*certified interval* for each ensemble property, computed from the measured numbers, the contour
bound and the error bar, all three of which enter explicitly and one-sidedly.  A model whose
prediction falls outside the interval is falsified by the datum alone; a model inside it cannot be
distinguished from the truth by that datum, because the interval endpoints are attained by genuine
ensembles.
