# How to design a model that fully captures intrinsically disordered regions

**A machine-checked specification, and its proof.**

All statements below are theorems of the Lean 4 development in `RequestProject/`, checked
against Mathlib. The whole project builds with no `sorry`, and each capstone theorem has been
confirmed to depend only on Lean's standard axioms (`propext`, `Classical.choice`,
`Quot.sound`). This document is the reader's map: it states the specification, names the
formal statement of every claim, records the assumptions each one needs, and says explicitly
what is *not* proved. `ANSWER.md` is the long-form narrative version, section by section.

---

## 1. The question, made precise

An intrinsically disordered region (IDR) does not have *a* structure. It has a thermodynamic
ensemble: a probability distribution on conformation space, depending on the sequence and on
the thermodynamic context (temperature, salt, binding partners, phase). "Design a model that
fully captures it" is therefore a question about the *specification* of a predictor:

> What must a model output, what capacity must it have, in what metric must it be scored,
> what may its architecture be invariant to, what data are needed to fit it, and which of
> these constraints are consequences of physics rather than of taste?

The development answers this in the form of two bundled theorems — a positive specification
`IDR.model_must_be` and a list of impossibilities `IDR.model_cannot_be`
(`RequestProject/Verdict.lean`) — together with the quantitative capstones that put numbers
on both:

| capstone | file | content |
|---|---|---|
| `quantitative_design_laws` | `Design.lean` | metric, capacity, entropy, invariance costs, transport loss |
| `physical_design_laws` | `PartFour.lean` | free energy, rate–distortion, ideal chain, measurement channel, latent rank, kinetics |
| `statistical_design_laws` | `PartFive.lean` | sample complexity of fitting an ensemble |
| `precision_design_laws` | `PartSix.lean` | Fisher information: what an ensemble can report about its conditions |
| `collective_design_laws` | `PartSeven.lean` | condensates, multivalency, two-phase samples |
| `temporal_design_laws` | `PartEight.lean` | relaxation time, run length, correlated frames |
| `physical_realism_design_laws` | `PartNine.lean` | chain statistics, SAXS/PRE/FRET, electrostatics, temperature, Rouse dynamics |
| `solution_state_design_laws` | `PartTen.lean` | NMR order parameters and RDCs, hydrodynamics, helix–coil |
| `continuum_design_laws`, `equivariance_design_law` | `PartEleven.lean` | continuum target, well-posedness, symmetry, learnability |
| `excluded_volume_design_laws` | `PartTwelve.lean` | exact self-avoidance in two and three dimensions |
| `sequence_hamiltonian_design_laws`, `cubic_sequence_hamiltonian_laws` | `PartThirteen.lean` | order and disorder from a microscopic sequence Hamiltonian |
| `evaluation_design_laws` | `PartFourteen.lean` | which score may a disorder model be judged by |
| `situated_design_laws` | `PartFifteen.lean` | crowding, force spectroscopy, single trajectories |
| `coupling_design_laws` | `PartSixteen.lean` | linkage reciprocity and the disordered tether |
| `sample_design_laws` | `PartSeventeen.lean` | concentration and multisite modification |
| `nonequilibrium_design_laws` | `PartNineteen.lean` | driven steady states and what they dissipate |
| `reported_ensemble_belongs_to_the_integrator` | `Integrator.lean` | the finite-timestep bias of a simulated ensemble |
| `two_state_model_cannot_fit_two_timescales` | `Memory.lean` | one relaxation time is not enough |
| `uncertainty_design_laws` | `PartTwentyTwo.lean` | calibration, resolution and prediction sets |
| `mutational_design_laws` | `PartTwentyThree.lean` | mutations: energy adds, populations do not |

---

## 2. Setting

`Ens X` (`RequestProject/EnsembleCore.lean`) is a finitely supported probability distribution
on a conformation space `X`: finitely many conformations with weights. Read as a *model* it is
a finite mixture: latent states, latent distribution, decoder. Two ensembles are
observationally equal (`Same`) when they agree on every observable, and `ApproxSame eps` when
they agree to `eps` on every observable bounded by one; `Metric.lean` proves that this is
exactly the population-space `ℓ¹` (twice total-variation) distance. `Continuum.lean` and
`GibbsMeasure.lean` replace the finite picture by measures on a conformation space and by the
Gibbs measure of an energy, and show that the finite statements are the honest approximations
of the continuous ones.

---

## 3. The specification

**(S1) The output is a distribution, and this is not a modelling choice.** Every ensemble is a
convex combination of single structures, and a class of outputs solves every prediction
problem *iff* it realises every ensemble up to observational equality (`model_must_be`,
clauses 1–4). A single structure — with or without error bars — is off by the population it
misses (`model_cannot_be`, clause 1), and this floor survives the continuum limit
(`ContinuumLoss.lean`) and holds even for a perfectly equivariant predictor
(`Symmetry.lean`: an equivariant single-structure output must be a fixed point of every
symmetry of its target).

**(S2) Capacity must scale with the disorder.** A model of at most `k` components is at `ℓ¹`
distance `(m − k)·δ` from a target populating `m` conformations of weight `≥ δ`
(`Ens.ell1_ge_of_card_le`), an exactly correct model carries `exp H` components where `H` is
the conformational entropy (`Ens.card_ge_exp_entropy`), and for the freely jointed chain of
`N` bonds this is `2^N` (`Chain.lean`). Excluded volume does not rescue this: for a chain that
genuinely cannot overlap itself, the count is still exponential (`PartTwelve.lean`, §5 below).

**(S3) The loss must see the geometry.** `ℓ¹`/entropy fix the capacity budget, but two
ensembles can be at maximal `ℓ¹` distance while their transport cost is arbitrarily small
(`Transport.lean`), and against a continuous truth every finite model is at total-variation
distance exactly 1 while still reproducing all Lipschitz observables to any accuracy
(`Continuum.lean`). Training and scoring must therefore be transport-/Lipschitz-based.

**(S4) Context is an input, not a nuisance.** One sequence with two thermodynamic contexts
defeats any predictor of the sequence alone (`Binding.lean`); conditioning on context repairs
it. Quantitatively, any invariance has a price: a predictor that conflates two inputs is off
by half the distance between their targets (`Invariance.lean`), which prices bounded receptive
fields, composition-only sequence features, and two-tower factorisations.

**(S5) The energy must be known to a fraction of `kT`.** The excess free energy of a model
*is* its KL divergence to the truth (`FreeEnergy.lean`), and energies agreeing to `d` in units
of `kT` give populations within `e^{2d}` (`GibbsMeasure.lean`). Conversely, giving one
conformation half the population requires an energy gap of order `kT·log(number of
competitors)`, i.e. linear in the length of the region — which is why disordered regions are
disordered (`folded_needs_entropic_gap`, `chain_folding_gap`, `saw_folding_gap`).

**(S6) The data have a cost in samples and in time.** Fitting an ensemble to `ℓ¹` accuracy
`eps` needs `Θ(m/eps²)` independent frames (`SampleComplexity.lean`, `SharpBound.lean`), and
frames are not independent inside a relaxation time; a barrier `B` forces run lengths
`~ e^B`, a chain of `N` beads forces `~ N²` (`Relaxation.lean`, `Rouse.lean`).

**(S7) The physics is in the forward models, not in the ensemble alone.** SAXS, PRE/NOE, FRET,
NMR order parameters and RDCs, and translational diffusion each enter through an exact
nonlinear functional of the ensemble, each with a proved bias (Jensen compaction for `r^{-6}`
averaging, harmonic averaging for diffusion, cancellation for RDCs), and each is invariant to
things the ensemble is not (`Observables.lean`, `NMR.lean`, `Hydrodynamics.lean`).

**(S8) Collective behaviour is not a single-chain property.** Demixing is exactly the failure
of convexity of the free-energy density, and is invariant under adding any affine function of
concentration — so two systems with identical single-chain thermodynamics can sit on opposite
sides of a phase boundary (`Condensate.lean`, `TwoPhase.lean`, `Valence.lean`).

---

## 4. Removing the idealisations

Part XI (`Continuum.lean`, `GibbsMeasure.lean`, `ContinuumLoss.lean`, `Symmetry.lean`,
`Generalisation.lean`, `PartEleven.lean`) replaces "finite ensemble" by "probability measure",
proves the target exists and is well posed (the Gibbs measure of a bounded energy, atomless
whenever the conformational coordinates are continuous, exponentially stable in the energy),
and adds learnability across sequences: covering sequence space to `delta` buys `K·delta`
accuracy everywhere, while a switch of gap `G` between inputs at distance `d` forces error
`(G − L·d)/2` on any predictor of smoothness `L`. `FlatChain.flat_chain_design_laws` verifies
that the hypotheses of the capstone are satisfiable by a standard object, so no clause is an
empty implication.

---

## 5. Excluded volume, exactly (Part XII)

The one physical objection that could have overturned (S2) is excluded volume: a chain that
cannot overlap has fewer conformations than a freely jointed one, and it is often suggested
that this is what makes a small structural library viable. Part XII settles it by counting,
for a general lattice (`LatticeWalk.lean`) and for the square and cubic lattices
(`SelfAvoiding.lean`, `CubicLattice.lean`):

* subchains of a self-avoiding chain are self-avoiding (`isSAW_append`), so the conformation
  count is submultiplicative (`cntOf_submultiplicative`) and the entropy per residue exists by
  Fekete's lemma (`tendsto_connectiveConstantOf`);
* it is strictly below the ideal-chain value — `μ₂ < log 4` and `μ₃ < log 6`, from the exact
  counts `cnt 4 = 100` and `cnt3 2 = 30` — so excluded volume costs a fixed amount of entropy
  per residue;
* and strictly positive — `μ₂ ≥ log 2`, `μ₃ ≥ log 3`, from the directed walks — so the
  ensemble is still exponentially large, and accuracy `eps` still costs `2^n(1 − eps)`,
  respectively `3^n(1 − eps)`, mixture components (`saw_capacity_lower_bound`,
  `saw3_capacity_lower_bound`);
* an ideal-chain generator is wrong about the *support*, by an exponentially large factor
  (`saw_fraction_tendsto_zero`);
* and sequential generation is constrained: growth can dead-end (`exists_trapped_walk`), and
  self-avoidance is not decidable from a bounded context window (`unbounded_memory`), so an
  autoregressive sampler with a fixed receptive field cannot be correct.

---

## 6. Order and disorder from a microscopic Hamiltonian (Part XIII)

Parts I–XII take the target ensemble as given, or as the Gibbs measure of an abstract energy.
Part XIII derives it. On the self-avoiding chains of Part XII it puts the hydrophobic/polar
contact energy — `-eps` per pair of hydrophobic residues that are lattice neighbours without
being bonded neighbours (`HPModel.lean`) — and takes its Boltzmann ensemble
(`Boltzmann.lean`). No mean-field step enters.

* **Excluded volume caps the energy.** A self-avoiding chain of `n` bonds on a lattice with `q`
  bond vectors has at most `(n+1)·q` contacts (`contacts_card_le`), because a residue has at
  most `q` neighbouring sites and no two residues share a site. The landscape therefore spans
  at most `eps·(n+1)·q`: at most `eps·q` per residue.
* **Hence a sequence-independent threshold for order.** Ordering one conformation to half the
  population needs a gap of `(1/beta)·log(cnt n − 1)` (S5), while the conformational entropy
  grows at least `log r` per residue. Comparing the two slopes gives `hp_no_folding_of_growth`:
  if `2·q·beta·eps < log r` then for *every* sequence and every conformation the equilibrium
  population is below one half — `8·beta·eps < log 2` on the square lattice, and
  `12·beta·eps < log 3` on the cubic lattice. Sequence design cannot evade a bound that never
  mentions the sequence.
* **Composition decides.** Only hydrophobic residues supply contact energy, so the cap improves
  to `h·q` with `h` the number of hydrophobic residues (`contacts_card_le_hCount`), and a
  sequence whose hydrophobic *fraction* is at most `f` is disordered as soon as
  `2·q·f·beta·eps < log r` — `8·f·beta·eps < log 2` on the square lattice,
  `12·f·beta·eps < log 3` on the cubic one. This is the microscopic counterpart of the
  empirical rule that regions of low mean hydrophobicity are disordered.
* **The criterion is sharp.** A unique lowest-energy conformation separated by
  `beta·gap ≥ log (cnt n)` does hold half the population (`hp_folding_sufficient`), so
  `beta·gap ≈ log(number of conformations)` is exactly the condition for order.
* **Cooling is not an escape.** A polar region has an exactly flat landscape, so its
  equilibrium ensemble is the athermal self-avoiding ensemble at *every* temperature
  (`polar_ens_same_saw`), costing `2^n` (in three dimensions `3^n`) components; and in general
  the low-temperature capacity requirement is the ground-state degeneracy
  (`Boltz.ground_state_capacity`).

---

## 6b. The evaluation (Part XIV)

A specification of what the model must output is only half of a design: a pipeline returns
whatever its score selects. Part XIV (`Scoring.lean`, `ScoreFloor.lean`, `PartFourteen.lean`)
scores predictions on a finite conformation library, a prediction being a population vector.

* **An honest score exists.** The quadratic (Brier) score has excess risk exactly
  `∑_x (p x − q x)²` (`brier_excess`), so it is strictly proper (`brier_strictly_proper`), and
  under *any* strictly proper score a single-structure prediction loses to the true ensemble
  (`strictlyProper_rejects_point_prediction`, `brier_rejects_deterministic_model`).
* **The sample-distance score rewards collapse.** Drawing a structure and measuring its squared
  deviation from the observed one has expectation `var(model) + var(truth) + bias²`
  (`sqDistScore_expected`); at fixed mean it strictly falls as the model becomes less dispersed
  (`sqDistScore_rewards_collapse`), and it is not strictly proper — on three equally populated
  rotamers the truth scores `4/3` and the collapsed structure `2/3` (`sqDist_three_state`).
* **The best-of-`N` score rewards hedging.** It does not see the weights
  (`bestOfScore_eq_of_supp_eq`), falls when the reported set is enlarged
  (`bestOfScore_antitone_supp`), and gives a perfect zero to every prediction covering the
  truth (`bestOfScore_expected_eq_zero`); it is not strictly proper either.
* **An honest score still confounds difficulty with quality.** The score splits as
  `∑_x (p x − q x)² + gini q` (`brier_decomposition`), the second term depending on the target
  alone: zero for an ordered region, `1 − 1/|X|` at most, `2/3` for three equal rotamers. So an
  exactly correct disorder model can score worse than a wrong model of an ordered region
  (`benchmark_confounded`): report the excess over the floor, not the raw number.
* **Context blindness, at training time.** If one prediction must serve several contexts, the
  optimum under an honest score is their weighted average
  (`brier_optimal_prediction_is_average`) — the truth in none of them.

---

## 6c. The situated region (Part XV)

Everything above describes a region in dilute buffer, at equilibrium, observed for as long as
one likes. Part XV (`Crowding.lean`, `Force.lean`, `Trajectory.lean`, `PartFifteen.lean`)
removes those three idealisations, and each removal changes the specification.

* **In the cell.** A crowded background at osmotic pressure `Pi` reweights each conformation by
  the work of opening a cavity of its excluded volume, i.e. exponentially tilts the ensemble
  (`crowded`, and `crowded_eq_boltz`: crowding is a shift of the landscape). Every structural
  average then responds as `d⟨f⟩/dPi = −beta·Cov(f, v)` (`hasDerivAt_crowdedMean`), so if larger
  conformations exclude more volume the mean size is *strictly* decreasing in the pressure at
  every pressure (`crowding_strictly_compacts`, via the two-point covariance identity
  `cov_two_point_of_weights` and `cov_pos_of_comonotone`). The measured ensemble is therefore
  never the functional one (`in_cell_ne_in_vitro`), while a region with a single excluded volume
  is untouched (`rigid_region_ignores_crowding`) — the correction is specific to disorder. And
  it is a new parameter: for an explicit three-state region *no* inverse temperature reproduces
  the crowded populations (`crowding_is_not_a_temperature`).
* **Under force.** The slope of a force–extension curve is `beta` times the variance of the
  extension coordinate (`stiffness_eq_beta_var`): compliance *is* fluctuation. A single-structure
  model is exactly inextensible (`rigid_is_inextensible`), while two populated conformations of
  different extension force a strictly rising curve (`extension_strictMono`). The two-state bond
  gives the closed form `b·tanh(beta·F·b)` (`two_state_bond_extension`) with entropic stiffness
  `beta·b²` (`two_state_stiffness_zero_force`) and the contour bound (`two_state_extension_lt`).
  The whole curve is a functional of the *law of the pulling coordinate* alone
  (`same_extension_law_same_force_curve`), so structurally different ensembles share it exactly
  (`force_curve_blind_to_structure`).
* **In time.** Time averaging is unbiased from equilibrium (`timeAvg_stationary`), but a
  trajectory started in a kinetically closed set never leaves it (`supportedOn_distAt`), so two
  kinetic models agreeing on that set give identical statistics for every observable and every
  window length (`distAt_eq_of_agree`, `timeAvg_eq_of_agree`). The explicit witness has two such
  models with *different* equilibrium ensembles
  (`single_molecule_cannot_identify_the_ensemble`), and when exchange is absent the equilibrium
  ensemble is not even a function of the kinetics (`reducible_stationary_not_unique`). Between-
  basin weights must come from elsewhere, or be reported as unidentified.

`IDR.situated_design_laws` bundles the three clauses.

---

## 6d. Coupling: linkage and the tether (Part XVI)

Part XVI (`Linkage.lean`, `Tether.lean`, `PartSixteen.lean`) turns to what a disordered region
is for.

* **Linkage is reciprocal.** In the doubly tilted ensemble — a structural field `lam·A` and a
  ligand chemical potential `mu·B` — both partial responses are covariances in the same ensemble
  (`hasDerivAt_mean2_mu`, `hasDerivAt_mean2_lam`), and covariance is symmetric, so
  `∂⟨A⟩/∂mu = ∂⟨B⟩/∂lam` (`linkage_reciprocity`). The coupling vanishes exactly when the partner
  does not discriminate between conformations (`no_linkage_of_uniform_affinity`) and is strictly
  positive as soon as it does (`linkage_of_comonotone`).
* **One coupling constant, finite form.** On the four-state folding/binding cycle the
  cross-product of the populations is `exp(−w)` regardless of the intrinsic stability and
  affinity (`thermodynamic_box`), so the ligand's stabilisation of the folded state *is* the
  folded state's enhancement of binding (`folding_stabilization_eq_binding_enhancement`), and
  with favourable coupling the folded fraction strictly rises on saturation
  (`apo_folded_fraction_lt_holo`). A model may not fit a conformational shift and an affinity
  change as independent parameters.
* **The linker is part of the binding site.** For the ideal three-dimensional tether the contact
  probability obeys the exact recursion `ret1_succ`, decreases strictly with length
  (`contactProb_strictAnti`), and satisfies the two-sided `N^{−3/2}` law (`le_contactProb`,
  `contactProb_le`, from the elementary bounds `one_le_ret1_sq_mul` and `ret1_sq_mul_le_one`,
  which pin the exponent from both sides). Hence the apparent affinity of an otherwise
  identical motif is strictly decreasing in linker length
  (`avidity_not_a_property_of_the_motif`): a measured affinity is not a property of the motif
  and does not transfer between constructs.

`IDR.coupling_design_laws` bundles the two clauses.

---

## 6e. The sample and the proteoform (Part XVII)

Part XVII (`SelfAssociation.lean`, `Multisite.lean`, `PartSeventeen.lean`) closes two gaps
between what is measured and what is meant.

* **A reported ensemble belongs to a concentration.** Solving monomer–dimer mass action
  (`mass_action`) gives the monomer fraction `2/(1 + sqrt(1 + 8Kc))`: a genuine fraction
  (`monoFrac_pos`, `monoFrac_le_one`), one only at infinite dilution (`monoFrac_zero`), strictly
  decreasing in total concentration at every concentration (`monoFrac_strictAnti`) and vanishing
  at high concentration (`monoFrac_tendsto_zero`). Any observable distinguishing monomer from
  dimer therefore drifts strictly with concentration
  (`measured_observable_depends_on_concentration`, `apparentObs_strictAnti`); only the
  infinite-dilution value is a property of the molecule (`apparentObs_at_zero_eq_monomer`).
* **Single-site modification data do not add up.** The interaction free energy of two
  modifications vanishes when one of them does not discriminate between conformations
  (`coupling_eq_zero_of_constant`) and is strictly positive for an explicit two-conformation
  witness (`coupling_pos_of_correlated`), so the joint effect is not the sum of the single
  effects (`multisite_effects_not_additive`). Every pairwise coupling of a multiply modifiable
  region is a separate parameter — and, by Part XVI, the same parameter that fixes the
  partner's affinity.

`IDR.sample_design_laws` bundles the two clauses.

---

## 6f. Out of equilibrium (Parts XVIII--XIX)

Every equilibrium statement above presupposes detailed balance; a cell does not supply it.
Part XVIII (`Driven.lean`) takes the smallest system that can show the difference, a three-state
conformational cycle `cycleP a b` stepping forward with probability `a` and backward with `b`.

* It is a legitimate kinetics (`cycleP_stochastic`) with an ordinary stationary distribution
  (`cycle_stationary_unif`), so the observation theory of Part XV survives unchanged: a
  stationary time average still equals the ensemble average (`Trajectory.timeAvg_stationary`).
* But the steady state carries a current `(a − b)/3` around the cycle (`cycle_current`), and by
  Kolmogorov's criterion **no** strictly positive distribution whatsoever satisfies detailed
  balance with the driven kernel (`no_detailed_balance_of_driven`). The driven steady state is
  the reversible equilibrium of no energy function at all, so Boltzmann weights, the variational
  free energy and exponential tilting have no target to fit.

Part XIX (`EntropyProduction.lean`) prices the drive. For a strictly positive kinetics in a
strictly positive steady state the entropy production rate — the relative entropy of the forward
one-step process against its time reverse — is nonnegative (`epRate_nonneg`, proved by
symmetrising the double sum into `∑ (x − y)(log x − log y)`), and vanishes *exactly* at detailed
balance (`epRate_eq_zero_iff_detailedBalance`). A landscape model is therefore committed to
predicting zero dissipation. For the driven cycle the rate is exactly `(a − b) log(a/b) > 0`
(`cycle_epRate`, `cycle_epRate_pos`), and two kinetics with the *same* stationary populations can
have zero and positive dissipation (`dissipation_not_determined_by_populations`): the energy
budget is a separate parameter, not a functional of the reported ensemble.
`IDR.nonequilibrium_design_laws` (`PartNineteen.lean`) bundles the three clauses.

---

## 6g. The integrator (Part XX)

Every simulated ensemble came from a discrete integrator at a finite timestep, and
`Integrator.lean` proves that this changes the ensemble itself. For overdamped Langevin
(Euler–Maruyama) dynamics in a harmonic well of stiffness `k` — the normal-mode description of a
flexible chain (`Rouse.lean`) — the variance evolves by `v ↦ (1 − k·dt)² v + 2 dt/beta`, whose
closed-form solution (`varSeq_eq`) relaxes geometrically (`varSeq_tendsto`) to
`2/(beta·k·(2 − k·dt))`.

* That is **not** the Boltzmann variance `1/(beta·k)`: the sampled ensemble is too broad by
  exactly `dt/(beta(2 − k·dt))` (`bias_eq`, `exactVar_lt_discVar`, `bias_pos`), a discrepancy
  that vanishes only as `dt → 0` (`bias_tendsto_zero_at_zero_timestep`).
* The excess is not noise. The simulation is *exactly* the equilibrium ensemble of a different
  force field, one of stiffness `effK = k(2 − k·dt)/2 < k`
  (`integrator_samples_a_softer_force_field`): a finite timestep is a silent change of
  Hamiltonian.
* It is a bias, not a variance: beyond some run length the sampled variance stays at least half
  the bias away from the target, from any initial condition
  (`no_sampling_removes_the_timestep_bias`). Past the stability limit `k·dt > 2` the variance
  diverges outright (`unstable_of_large_timestep`).
* And the positive half: for *any* scheme acting affinely on the variance, `v ↦ r v + s`,
  sampling the Boltzmann ensemble exactly is the single algebraic condition
  `s = (1 − r)/(beta·k)` (`unbiased_iff_consistency`), which the exact Ornstein–Uhlenbeck
  propagator satisfies at every timestep (`ou_unbiased`). The bias is a property of the chosen
  scheme, not of discreteness.

`reported_ensemble_belongs_to_the_integrator` is the design statement: a reported ensemble is a
statement about an integrator as much as about an energy function, so the integrator (and its
consistency condition) belongs in the model.

---

## 6h. Timescales (Part XXI)

A two-state kinetic model is the default reading of a correlation experiment on a disordered
region. `Memory.lean` shows it cannot carry what such experiments report. The two-state chain is
reversible and unobjectionable at equilibrium (`twoP_stochastic`, `piTwo_stationary`,
`twoP_detailedBalance`), but its propagator relaxes every observable by the single eigenvalue
`λ = 1 − a − b` (`prop_eq`), so the equilibrium autocorrelation of *every* observable is one
geometric `Var(f)·λ^t` (`autocorr_eq`) — one relaxation time, a property of the model rather than
of the probe. A two-exponential decay with positive weights and distinct rates cannot be matched
by any geometric, already at `t = 0, 1, 2` (`no_single_geometric`, by strict Cauchy–Schwarz), so
`two_state_model_cannot_fit_two_timescales`: measured timescales are a capacity statement about
the kinetic model, exactly as populated conformations are a capacity statement about the ensemble.

## 6i. Uncertainty: calibration and prediction sets (Part XXII)

Part XIV fixes the score; Part XXII fixes the two things reported *about* the score.

A model computes a finite internal code `k = b i` of its input and answers with a population
vector `A k`; the benchmark weights inputs by `mu` and the truth at input `i` is `T i`.
`Calibration.lean` proves the exact identity `risk = calError + resolution`
(`risk_decomposition`): the averaged squared population error — which is the excess risk of the
strictly proper score of Part XIV (`risk_eq_brier_excess`) — splits into what a reliability
diagram measures and the variance of the truth *inside* the code classes. A calibrated model has
paid the first term and nothing else (`risk_eq_resolution_of_calibrated`); the second is strictly
positive whenever two weighted inputs with different targets share a code
(`resolution_pos_of_conflated`); and recalibration attains, but cannot beat, that floor
(`recalibrated_risk_eq_resolution`, `recalibration_improves`). The extreme case is a model that
ignores its input and reports the population-averaged ensemble: perfectly calibrated, wrong
everywhere (`calibrated_but_wrong_everywhere`, and in ensemble language
`ens_context_blind_calibrated_but_wrong`).

`PredictionSets.lean` does the same for coverage. The whole library covers at every level
(`univ_covers`, `validity_is_free`), so validity ranks no model and the content of a prediction set
is its size; coverage transfers with `l¹` accuracy at half the error (`mass_diff_le_half_ell1`,
`covers_of_close`); a set covering at level `alpha` needs at least `(1 − alpha)/pmax` conformations
(`card_ge_of_covers`), which for `m` equally populated conformations is an equivalence
(`unif_covers_iff`) and rules out single-structure answers (`no_singleton_covers`); and a reported
coverage is a marginal — 90% over two contexts, zero in one of them
(`marginal_not_conditional`). `IDR.uncertainty_design_laws` (`PartTwentyTwo.lean`) bundles the six
clauses.

## 6j. Mutations (Part XXIII)

A model is used to predict differences, and the standard analysis sums single-mutant effects on
measured populations. `Epistasis.lean` proves that the additive quantity is the energy. An affine
quantity has zero double-mutant cycle (`cyc_of_affine`); the energy cycle of a two-site model
`h₁s₁ + h₂s₂ + Js₁s₂` is exactly the coupling (`energy_cycle_eq_coupling`); and since the log-odds
of a two-state population is the energy up to `−beta` (`logit_pop`), the log-odds cycle returns
`−beta·J` exactly (`logodds_cycle_eq_coupling`) and vanishes iff the sites are uncoupled
(`coupling_iff_logodds_cycle`).

Populations, by contrast, never add: two uncoupled favourable substitutions have a strictly
negative population cycle (`sig_cyc_neg`, `population_cycle_not_evidence_of_coupling`), and the
sign of the measured epistasis flips with the background (`cyc_reflect`,
`epistasis_sign_depends_on_background`). On an extreme background arbitrarily large energetic
effects move the population by arbitrarily little (`saturation`). `mutational_design_laws`
(`PartTwentyThree.lean`) bundles the five clauses: fit, report and combine in energy; treat every
predicted population as a non-linear, background-dependent, saturating read-out of it.

## 6k. The molecular model: continuous space, force field, solvent, Gibbs measure (Part XXIV)

The earlier parts use the smallest carrier of each phenomenon. Part XXIV rebuilds the
load-bearing statements in the setting a model is actually built in: atoms at real coordinates in
`R³`, a class-I molecular-mechanics Hamiltonian, implicit solvent, and a probability measure on
`R^(3N)` with a density.

`Context.lean` makes the environment a coordinate of the model — temperature, ionic strength,
partner concentration, SI constants, and the Debye length, which decreases with salt
(`debyeLength_strictAnti_ionicStrength`) and increases with temperature
(`debyeLength_strictMono_temperature`). All three coordinates are load-bearing
(`context_coordinates_load_bearing`) and no constant predictor survives (`no_context_free_model`).

`Potentials.lean` and `Hamiltonian.lean` give the untruncated 12-6, Coulomb-with-dielectric and
harmonic terms and the Hamiltonian on `(R³)^N` with Lorentz–Berthelot mixing. The `r^-12` core
dominates any Coulomb attraction (`pair_bddBelow`), so the energy is bounded below on
non-degenerate configurations (`H_bddBelow`); excluded volume becomes a theorem
(`H_repulsive_core`) rather than a lattice postulate; the energy is exactly E(3) invariant
(`H_rigid_invariant`); and exact atom overlaps form a Lebesgue-null set (`coincident_null`).

`Solvation.lean` adds Generalized Born (with the Still interpolation `fGB` and its exact bounds)
and a surface-area term whose surface is the two-dimensional Hausdorff measure of the accessible
set. A charge is strictly stabilised by the solvent (`born_self_neg`), desolvation grows as the
Born radius shrinks (`born_self_strictAnti_radius`), burial lowers the nonpolar term
(`nonpolar_le_of_occlusion`), and the solvated Hamiltonian is still stable and exactly invariant
(`Hsolv_bddBelow`, `Hsolv_rigid_invariant`).

`GibbsField.lean` then produces the ensemble: a strictly positive partition function on a
container (`Zpart_pos`), a probability measure absolutely continuous with respect to Lebesgue
measure whose Radon–Nikodym derivative is the Boltzmann density (`gibbs_rnDeriv`), the
partition-function-free ratio `f(x₁)/f(x₂) = exp(−βΔU)` (`density_ratio`) and score identity
`∇ log f = −β∇U` (`score_eq_neg_beta_grad`), SE(3) invariance of the measure
(`gibbsMeasure_rigid_invariant`), and the exclusion of point predictions
(`dirac_not_absolutelyContinuous`). `ContinuousScore.lean` redoes the scoring theory for
densities and prices non-equivariance exactly (`symmetrization_identity`,
`equal_risk_iff_equivariant`), and `Generative.lean` states the push-forward programme and proves
that every Borel ensemble is representable (`exists_generator`). `molecular_design_laws`
(`PartTwentyFour.lean`) bundles the nine clauses.

## 6l. The residual assumptions, priced (Part XXV)

Four idealisations remain inside the Part XXIV model, and Part XXV removes each.

`ManyBody.lean`: pairwise additivity of the solvent-averaged energy is exactly the vanishing of a
mixed second difference (`mixed_eq_zero_of_additive`), so a cooperative three-body potential of
mean force is not representable by any pair decomposition (`not_additive_cooperative`), and the
best additive surrogate errs by at least a quarter of that difference
(`additive_error_lower_bound`).

`Protonation.lean`: the Henderson–Hasselbalch occupancy is strictly decreasing in pH
(`protonatedFraction_strictAnti_pH`), the mean charge of a titratable region with it
(`netCharge_strictAnti_pH`), and no constant charge reproduces a titration curve
(`no_fixed_charge_model`) — pH is a context coordinate. The linkage cycle closes exactly
(`linkage_cycle`, `pKa_shift_eq_binding_shift`): a pKa shift on binding is a pH dependence of the
affinity.

`BrokenErgodicity.lean`: the two-state master equation of a slow isomerisation is solved exactly
(`popB_hasDerivAt`) and relaxes only asymptotically (`popB_tendsto_equilibrium`); a run whose
horizon times the total rate is at most `δ` retains a fraction `1 − δ` of its initial deviation
(`popB_stuck`), and averaging the whole trajectory does not help (`timeAverage_eq`,
`timeAverage_stuck`, `finite_run_not_boltzmann`).

`Constraints.lean`: constraining a stiff coordinate is not the stiff limit of the unconstrained
ensemble. The marginal is computed in closed form (`softMarginal_eq`) and the relative weights
differ by exactly the Fixman factor `sqrt(w(q₂)/w(q₁))` for every stiffness (`soft_ratio`), with
equality precisely where the stiffness is constant (`soft_eq_rigid_iff`, `rigid_ne_soft`).

`residual_assumption_laws` (`PartTwentyFive.lean`) bundles the four.

## 6m. Tractability: evaluating and sampling the model (Part XXVI)

Everything above constrains what a model must *denote*. Part XXVI adds the requirement that the
denoted distribution be *evaluable*, and prices it.

`TransferMatrix.lean`: a residue-level model of `n+1` residues with `k` local states denotes a
distribution over `k^(n+1)` conformations (`card_chain_states`), so every population is a ratio of
exponentially long sums. `Z_eq_vecMul_pow` is the transfer-matrix theorem in the general `k`-state
case: for any nearest-neighbour weight the configuration sum equals `v·Mⁿ·u`, i.e. `n` matrix
multiplications. With strictly positive local weights the normalized measure is a genuine ensemble
(`Z_pos`, `chainProb_sum_one`, `chainProb_pos`), and `Z_boltzmann` states the identity for a
nearest-neighbour energy at inverse temperature `beta`. The price of that tractability is exact:
a nearest-neighbour model makes the two ends of a three-residue region conditionally independent
given the middle residue (`pairFactored_cross`), while the explicit contact ensemble `contactDist`
is a bona fide distribution with correlated ends (`contactDist_sum_one`,
`contactDist_ends_correlated`) that admits no nearest-neighbour factorization whatsoever
(`contactDist_not_pairFactored`, `chainProb_ne_contactDist`).

`Tensorization.lean`: the usual repair — sample a reference model and reweight — does not scale.
The chi-squared reweighting cost of Part V.1 tensorizes over sites (`chiSq_prodDist`), so a
per-residue mismatch `c > 0` repeated along `n` residues leaves an effective sample size fraction
of exactly `(1+c)^(−n)` (`essFrac_prodDist_iid`) and demands `neff·(1+c)^n` frames
(`frames_needed_prodDist`); relative entropy, by contrast, is additive (`kl_prodDist`). Local
observables of a factorized model, on the other hand, cost one sum of `k` terms
(`sum_prodDist_local`).

`tractability_laws` (`PartTwentySix.lean`) bundles the four statements. The architectural
conclusion: the model must be factorized enough to be evaluated and sampled, and its factorization
must already contain the long-range structure it is meant to predict, because neither a local
factorization nor a reweighting of a mis-specified reference can supply it afterwards.

## 6n. Sampling: turning an energy function into conformations (Part XXVII)

Part XXVI leaves a dilemma: exponentially many conformations, and cheap factorizations that
cannot carry a long-range contact. Part XXVII treats the route real programs take.

`Metropolis.lean`: the Metropolis kernel `mhK` of a target under a symmetric proposal is a
transition kernel (`mhK_stochastic`), is reversible with respect to the target
(`mhK_detailedBalance`) and therefore leaves it stationary (`mhK_stationary`, via
`Kinetics.stationary_of_detailedBalance`). It depends on the target only through *ratios*
(`mhK_smul`), so the kernel of a Boltzmann ensemble is literally the kernel of its unnormalized
weights (`mhK_of_unnormalized`) and the acceptance probability is a function of the energy
difference alone (`acc_boltzmann`). The intractable configuration sum never appears.

`Mixing.lean`: what that does not buy is time. The bottleneck lemma (`massOut_evolve_le`,
`massOut_iterate_le`) bounds the population that can leave a region of conformation space by
`eps` per step, so escaping takes at least `m/eps` steps (`steps_needed`). On the two-well
landscape with a barrier of height `B` the Metropolis escape probability is `e^{−βB}/2`
(`barrier_escape_le`), the population outside the initial well after `t` steps is at most
`t·e^{−βB}/2` (`barrier_mass_le`), and equilibrating the wells takes at least `e^{βB}` steps
(`barrier_steps_needed`) — the discrete Monte Carlo counterpart of the broken ergodicity of
Part XXV.

`sampling_laws` (`PartTwentySeven.lean`) bundles the four. Evaluation of a factorized model is
linear in the length of the region; sampling a rugged one is exponential in its barriers. A model
must therefore report which ensemble its sampler actually produced.

## 6o. Local restraints: scalar couplings and hydrogen exchange (Part XXVIII)

Parts IX and IX.2 treated the global restraints (SAXS, PRE/NOE, smFRET). Part XXVIII treats the
two local, residue-resolved restraints that carry most of the remaining experimental weight.

`Karplus.lean`: the measured three-bond coupling of an interconverting torsion is exactly
`A⟨cos²θ⟩ + B⟨cos θ⟩ + C` (`avgJ_eq`). It is therefore *two numbers* about the torsion
distribution, and any two ensembles matching those two moments agree on every Karplus
parametrisation at once (`avgJ_congr_of_moments`): measuring more couplings of the same torsion
adds nothing. The single-angle inversion is biased by exactly `A·Var(cos θ)`
(`avgJ_eq_karplusOfCos_add_var`), a variance that is strictly positive whenever two populated
conformers differ in `cos θ` (`cosVar_pos_of_two`); and a coupling cannot see the sign of the
torsion (`avgJ_reflect`), so α_R and α_L are indistinguishable. Explicitly, two five-basin
torsion distributions with the same first two `cos` moments predict identical couplings
(`karplus_two_ensembles_agree`), the whole segment between them does too, and the population of
the `θ = π/2` basin can be set to any value in `[0, 1/2]` without changing any measurement
(`karplus_any_population_consistent`). The positive counterpart, and the reason the classical
two-state analysis is legitimate, is `twoBasin_identifiable`: with two basins and a
discriminating coupling the populations are unique.

`HydrogenExchange.lean`: amide exchange averages *rates*. Under EX2 the protection factor is the
reciprocal of the mean openness (`protectionFactor_eq`), so the apparent free energy
`-RT log⟨p_open⟩` is at most the mean local stability `⟨-RT log p_open⟩` (`deltaGapp_le_mean`,
Jensen for `log`), strictly so in an explicit heterogeneous instance (`deltaGapp_lt_mean_two`),
and it is capped by any minority open state: a conformer of weight `w` and openness `p` limits
the apparent stability to `-RT log p - RT log w` whatever the rest of the ensemble does
(`deltaGapp_le_of_weight`) — the exchange analogue of the `r^{-6}` minority-report bound. Worse,
the observable is not a functional of the equilibrium ensemble at all: the steady-state rate is
`k_op k_int/(k_cl + k_int)`, whose deviation from the EX2 reading is exact (`kex_eq_EX2_sub`),
which saturates at the opening rate in the EX1 limit (`kex_le_kop`), and two schemes with the
same opening equilibrium constant — hence the same Boltzmann ensemble — exchange at different
rates (`equilibrium_does_not_determine_kex`).

`local_restraint_laws` (`PartTwentyEight.lean`) bundles the five statements. Local restraints
must enter as forward-modelled ensemble averages with their exact nonlinear kernels, must be
reported together with the degeneracy they leave, and hydrogen exchange must either be
restricted to a verified EX2 regime or modelled kinetically.

## 6p. What the simulation actually reports: box, cutoff, error bars (Part XXIX)

Parts XX and XXVII priced the dynamics of generating an ensemble. Part XXIX prices the three
remaining systematic differences between the ensemble a paper reports and the ensemble its model
denotes. All three are properties of the protocol, and all three have a known sign or cost.

`PeriodicBox.lean`: a weighted Chebyshev inequality (`sum_antivary_le`) gives that reweighting
any ensemble by a positive factor decreasing in the chain dimension can only decrease `⟨R⟩`
(`wmean_tilt_le`). The image interaction of a periodic cell is exactly such a factor, so
`⟨R⟩_box ≤ ⟨R⟩_∞` at every temperature (`boxMean_le_freeMean`), strictly in an explicit two-state
instance at every `β > 0` (`boxMean_lt_freeMean_two`), and the reported dimension increases
monotonically with box size (`boxMean_mono_in_box`). A single box size cannot validate a reported
dimension.

`Cutoff.lean`: a conformation all of whose pair distances exceed the cutoff has zero truncated
interaction energy (`truncEnergy_eq_zero_of_beyond`), so any two such conformations receive
exactly equal Boltzmann weight at every temperature (`cutoff_blind`) while the untruncated
potential separates them (`cutoff_loses_true_ranking`). A cutoff model does not misestimate the
long-range landscape; beyond `rc` it has none. Quantitatively, with a `C/r` tail the neglected
energy of an `n`-site conformation is at most `n²C/rc` (`truncation_error_le`): quadratic in the
length of the region at fixed cutoff.

`CorrelatedSampling.lean`: for an exponentially correlated stationary series — the autocorrelation
of the two-state exchange of Part VIII.1 and of a Rouse mode sampled at interval `Δt` — the
variance of a trajectory average is exactly `σ²/N²·(N(1+ρ)/(1-ρ) - 2ρ(1-ρ^N)/(1-ρ)²)`
(`corrSum_eq`). It reduces to `σ²/N` only at `ρ = 0` (`varMean_eq_iid`), equals `σ²` at `ρ = 1`
for every `N` (`varMean_frozen`: a trajectory shorter than the relaxation time contains one
sample however often it is written), is never below the independent-sample value
(`varMean_ge_iid`), and exceeds it by the statistical inefficiency `(1+ρ)/(1-ρ) ≈ 2τ/Δt`
(`varMean_ge_inflated`), which sets the number of frames an error bar costs (`frames_needed`).

`protocol_laws` (`PartTwentyNine.lean`) bundles the seven statements. Every capacity and
sample-complexity bound of Parts V–VII is stated in independent samples; this is the conversion
factor, and the box and cutoff results are the two protocol parameters a reported ensemble must
be shown to be stationary in.

## 6q. The entropy price of ordering (Part XXX)

Part XVI proved the reciprocity between conformational populations and affinity; Part X proved
that a binding constant is not a function of a mean structure with error bars. Part XXX supplies
the accounting they leave open, in the standard finite-conformer partition-function formalism:
free-state populations `p`, a binding-competent subset `S`, a per-conformer interaction `-eps k`,
and a bound-state conformational sum restricted to `S`.

`Selection.lean`: with a uniform interaction `e` over the competent set,
`deltaG_eq_selection` gives the conformational-selection decomposition
`ΔG = -e + kT log (1/P_S)` — the intrinsic interaction plus the free energy of the population
restriction, strictly positive whenever the competent set is not the whole ensemble
(`selection_penalty_pos`), and equal to `kT log m` — `kT` times the conformational entropy — for a
uniform free ensemble binding through a single conformer (`penalty_uniform_eq_log_card`). Two
models that agree on the bound structure and disagree on the free-state breadth therefore predict
affinities differing by exactly the entropy they disagree on. Three consequences follow.
`deltaG_antitone_subset`: a complex that tolerates more conformers binds at least as well, so a
fuzzy complex is thermodynamically favoured whenever the interaction survives it, not a defect of
the model. `deltaG_le_neg_mean`: by Jensen for `exp`, the true binding free energy is at most the
population average of the per-conformer interaction energies, so scoring a designed binder by a
mean interaction energy is systematically conservative. `deltaG_le_of_conformer`: a single
competent conformer of population `p_k` guarantees `ΔG ≤ -eps_k + kT log (1/p_k)` whatever the rest
of the ensemble does — the binding counterpart of the `r^{-6}` and hydrogen-exchange
minority-report bounds.

`selection_laws` (`PartThirty.lean`) bundles the six. An affinity prediction for a disordered
region is a difference of two *ensemble* free energies: a model must report the free-state
populations of the competent conformers, not a bound pose and not a mean structure.

## 6r. The scaling exponent is fitted, not measured (Part XXXI)

The Flory exponent is the single number most often quoted for a disordered region, and it is a
derived quantity: inferred from a few chain lengths through a relation that carries corrections
to scaling. `ScalingExponent.lean` computes what the standard two-point log–log estimator returns
when the truth is `R_g² = A N^{2ν}(1 + B/N)`. `nuHat_eq` is the exact identity: the fitted
exponent is `ν` plus `(log(1+B/N₂) − log(1+B/N₁))/(2 log(N₂/N₁))`, a term built entirely from the
correction. The bias therefore has a determined sign — a positive correction amplitude makes the
region look less swollen than it is (`nuHat_lt_of_pos_correction`), a negative one more
(`nuHat_gt_of_neg_correction`) — and a determined size, at most `B/(2N₁ log(N₂/N₁))`
(`nuHat_bias_le`), controlled by the *shortest* chain used and by the lever arm in log length.
And it is invisible: `two_point_fit_exact` exhibits a pure power law with the fitted exponent and
no correction that reproduces both measurements exactly, so no goodness-of-fit argument at two
lengths can defend a reported exponent.

`scaling_laws` (`PartThirtyOne.lean`) bundles the four. A model of a disordered region should be
compared with the radii actually measured, forward-modelled at the lengths studied, rather than
with a fitted `ν`.

## 6s. Dynamics from NMR: spectral densities and relaxation dispersion (Part XXXII)

Motion enters `R₁`, `R₂` and the heteronuclear NOE only through the spectral density
`J(ω) = Σ_k w_k·2τ_k/(1+(ωτ_k)²)` (`SpinRelaxation.lean`). Three exact facts follow. `J(0)` is
twice the *mean* correlation time (`specDens_zero`), so it is dominated by the slow tail: a `1%`
population `10³` times slower than the bulk already raises it above ten times the bulk value
(`minority_slow_state_dominates`). Each Lorentzian is two-to-one in the correlation time,
`lorentz_tau_ambiguity` and `lorentz_eq_iff` giving the exact `τ ↦ 1/(ω²τ)` degeneracy — the
classical `τ_c` ambiguity. And a one-field data set does not determine the motion:
`relaxation_underdetermined` exhibits two four-component motional models with strictly positive
populations agreeing at three frequencies — as many independent numbers as `R₁`, `R₂` and the
NOE supply — whose values of `J(0)` differ by `77/240`. The Lipari–Szabo "model-free" form is
exactly the two-component case (`modelFree_eq_specDens`), so `S²` is a fitted population, not a
readout; two `(S², τ_e)` pairs give identical data (`modelFree_underdetermined`). Identifiability
is not hopeless — two known correlation times off the reflection are identified from one
frequency (`twoComponent_identifiable`) — but it is a property of the model, not of the data.

`ChemicalExchange.lean` does the same for the minority state. Under the standard fast-exchange
(Luz–Meiboom) forward model the CPMG profile has amplitude `Φ_ex/k_ex` with `Φ_ex = p_A p_B Δω²`:
strictly positive and bounded (`disp_nonneg`, `R2eff_lt_plateau`), refocused linearly in the
cycle time (`R2eff_sub_le_short_cycle`), approaching the plateau within `2Φ_ex/(k_ex²t_cp)`
(`plateau_gap_le`). A measured amplitude *does* bound the population from below once `Δω` is
capped by the spectral range (`population_lower_bound`) — exchange broadening is evidence that a
minority state exists. It does not say how much of it there is: every population in `(0,1/2]`
reproduces the entire profile with a suitable `Δω` (`invisible_state_population_unidentifiable`),
and at large `k_ex` a state of any population contributes less than any threshold at every cycle
time (`arbitrarily_populated_invisible_state`). `nmr_dynamics_laws` (`PartThirtyTwo.lean`)
bundles the eight statements.

## 6t. Single-molecule histograms: how much of the width is the molecule? (Part XXXIII)

`PhotonCounting.lean` derives the width of a single-molecule FRET histogram from the photon
statistics of a burst: `N` photons, binomial acceptor counts, with the normalisation, mean and
variance of that distribution obtained from the Bernstein-polynomial identities (`binom_sum`,
`binom_mean`, `binom_var`). The histogram is centred correctly (`measured_unbiased`) and its
width obeys an exact law of total variance, `Var = varConf + shotNoise`
(`shotnoise_decomposition`). Consequently a *single* conformation produces a histogram of
strictly positive width `e(1−e)/N` (`homogeneous_histogram_has_width`), and
`width_not_evidence_of_heterogeneity` makes it quantitative: one conformation at `e = 1/2` with
100 photons per burst, and a genuine two-state ensemble at `e = 0.46, 0.54` with 276, give
histograms of exactly the same variance `1/400`. Conversely the detector term never exceeds
`1/(4N)`, so excess width *is* evidence (`heterogeneity_detected`) and `N ≥ 1/(4·varConf)`
photons per burst put the molecule above the noise (`photons_needed`). Finally, if the chain
interconverts faster than the burst lasts the conformational term is averaged away while the
detector term is not: with two independently sampled conformations per burst the width is
`varConf/2 + shotNoise` (`dynamic_averaging`). A narrow histogram is no more evidence of
homogeneity than a broad one is of heterogeneity. `single_molecule_laws`
(`PartThirtyThree.lean`) bundles the six.

## 6u. Association kinetics: what fly-casting would have to mean (Part XXXIV)

`Association.lean` composes Smoluchowski capture with the Stokes–Einstein relation of Part X.2.
The diffusion-limited rate is exactly `(2kT/3η)·(R_c/R_h)` (`smoluchowski_stokes`): the capture
radius and the hydrodynamic radius enter only through their ratio, and the prefactor is a
property of the solvent. So swelling at fixed shape is kinetically free
(`rate_scale_invariant`); a speed-up happens if and only if the capture radius outgrows the
hydrodynamic radius (`flycasting_iff`); and a chain whose two radii follow the same scaling law
binds at a length-independent rate (`no_flycasting_from_scaling`) — a fly-casting effect must
come from a capture radius set by something other than the chain's own size, and a model that
predicts on-rates must say what. Rates average over the ensemble (`ensembleRate_eq`), and the
two radii are averaged differently by the experiment — harmonically and arithmetically — so a
surrogate structure carrying the measured hydrodynamic radius and the mean capture radius binds
`4/3` times too fast (`rate_not_determined_by_apparent_size`). `association_laws`
(`PartThirtyFour.lean`) bundles the six.

## 6v. Titration curves: the cooperativity is fitted, not measured (Part XXXV)

The last derived quantity in common use is the `m`-value of a chemical denaturation curve, read
as the cooperativity of a transition that, for a region with no folded state, is not a transition
at all. `Denaturant.lean` computes what the two-state linear-extrapolation fit returns. The
logistic curve `frac dG m RT x = sigmoid((m x − ΔG)/RT)` crosses `1/2` at `x = ΔG/m`
(`frac_midpoint`) with slope exactly `m/(4RT)` (`frac_deriv_midpoint`), so the reported `m` *is*
four `RT` times the measured midpoint slope (`m_eq_four_RT_slope`) — a repackaging of one point
and one slope. Consequently `fit_matches_any_curve`: every signal crossing the midpoint with a
positive slope is matched there, in value and in slope, by a two-state model. And the agreement
is not only first order: `sigmoid_tangent_cubic` bounds the departure of the logistic curve from
its own midpoint tangent by `|u|³/48` in the reduced variable `u = m(x − x½)/RT` — proved from
`tanh y ≤ y` and the cubic bound `y − y³/3 ≤ tanh y`, both established from scratch — so a
strictly linear, non-cooperative expansion is reproduced by a two-state fit to cubic accuracy
(`frac_close_to_linear`). The one qualitative difference is saturation (`frac_mem_Ioo`), which
lives at the ends of the titration where the baselines are fitted. `titration_laws`
(`PartThirtyFive.lean`) bundles the five.

## 6w. Circular dichroism: the secondary-structure content is a projection (Part XXXVI)

`Dichroism.lean` treats the experiment most often reported for a disordered region and the
numbers extracted from it. The forward model is the mixture used by every deconvolution program,
`mix B f i = ∑ⱼ fⱼ Bⱼᵢ`. A *null direction* is a change of composition that leaves both the total
weight and every measured wavelength unchanged, and along one the fit is literally blind
(`mix_perturb`, `sum_perturb`). `exists_null` shows that a null direction exists whenever the
number of wavelengths plus one is below the number of classes, whatever the reference spectra
are — the underdetermination is linear algebra, not bad luck; `segment_of_null` places a whole
segment of nonnegative, normalised, equally fitting compositions around *any* strictly positive
composition, so it is not a boundary effect; and `two_wavelength_witness` exhibits, for a
realistic four-class basis at two wavelengths, two compositions differing by more than 13
percentage points of β-sheet with identical spectra. The case that matters most for a disordered
region is the near-coincidence of the polyproline II and statistical-coil reference spectra: were
they equal the split between them would be entirely free (`equal_basis_split_free`), and as it is
the split is bounded by the noise rather than by the physics — two spectra differing by at most
`ε` at every wavelength make a transfer of weight `t` move the data by at most `|t|·ε`
(`near_degenerate_tolerance`), so a tolerance `η` admits every split with `|t| ≤ η/ε`. What *is*
determined is exactly the affine read-outs: `readout_determined` shows that `∑ⱼ wⱼ fⱼ` takes the
same value on every fitting composition as soon as `wⱼ = c₀ + ∑ᵢ cᵢ Bⱼᵢ`. `dichroism_laws`
(`PartThirtySix.lean`) bundles the six. The design consequence is the one already drawn for the
scaling exponent and the `m`-value: predict the spectrum, and compare with the spectrum.

## 6x. Aggregation kinetics: the lag time is a logarithm (Part XXXVII)

`Aggregation.lean` treats the fate that makes disordered regions worth modelling, and the number
extracted from a thioflavin curve. In the exactly solvable early-time nucleation–elongation model
`M'' = κ²M`, `M(0) = 0`, `M'(0) = v` — `v` set by primary nucleation, `κ` by elongation and
secondary nucleation — the mass is `(v/κ)·sinh(κt)` (`mass_hasDerivAt`, `mass_second_deriv`).
`mass_pos` records that there is no lag *phase*: the mass is strictly positive at every positive
time, and the flat portion of a trace is the interval below the detection threshold.
`mass_invariant` gives the exact constant of the motion `(M')² − κ²M² = v²`, which is the
identifiable content of a curve. The threshold is crossed at `lagTime = arsinh(κM_c/v)/κ`
(`mass_lagTime`) — a logarithm of the nucleation flux — and the two insensitivity laws follow:
multiplying either the detection threshold or the nucleation rate by `r ≥ 1` moves the apparent
lag by at most `log r/κ` (`lag_shift_le`, `lag_rate_shift_le`), and by at least
`(log r − log(3/2))/κ` once the threshold is at or above `v/κ` (`lag_shift_ge`), so the dependence
is logarithmic and no weaker. Finally `lag_underdetermined` and `two_models_one_lag`: for *every*
growth rate there is a nucleation flux reproducing an observed lag time exactly, and two such fits
have genuinely different fluxes. `aggregation_laws` (`PartThirtySeven.lean`) bundles the seven.
An ensemble model is therefore not confirmed or falsified by a lag time; the quantity it can be
held to is the pair `(v, κ)`, equivalently the curve, and the nucleation rate it predicts enters
the observable only through its logarithm.

## 6y. Density maps: occupancy and disorder are the same parameter (Part XXXVIII)

`Density.lean` treats the operational definition of a disordered region — the part of the chain
with no interpretable density in a crystallographic or cryo-EM map — in the standard harmonic
(Debye–Waller) treatment. In reciprocal space, `single_shell_degenerate` shows that one
resolution shell determines nothing: whatever occupancy is assumed, the displacement parameter
`B + 4 log(q'/q)/s0^2` reproduces the measured amplitude exactly. `two_shells_identify` shows
that two distinct shells determine both parameters, so the model is identifiable in principle.
`formFactor_close` prices the practice: two models agreeing at a shell `s0` differ at a higher
shell by at most `F0 |B - B'| (s^2 - s0^2)/4`, proved from `|e^x - e^y| <= |x - y|` for
nonpositive exponents, so separating occupancy from disorder requires data quality proportional
to the *span* of resolution actually measured — precisely what a region whose scattering has
decayed away does not provide.

In real space the degeneracy is exact. `peak_scale_invariant` states the classical occupancy-`B`
correlation as an identity: multiplying the occupancy by `c^3` and the displacement parameter by
`c^2` leaves the peak height `q (4 pi/B)^(3/2)` unchanged, for every `c > 0`. A missing side
chain and a fully occupied but mobile one are the same map. `peak_antitone` and `peak_hundred`
quantify the cost of disorder — `B = 100` gives one eighth of the peak at `B = 25` — and
`visibility_bound` with `invisible_of_large_B` turn the operational definition into a threshold
statement: a peak at contour level `tau` forces `B^3 tau^2 <= (4 pi)^3 q^2`, and any `B` beyond
that bound is invisible at that contour. "No density" is a bound on `q^2/B^3`, not the absence of
a residue. `density_laws` (`PartThirtyEight.lean`) bundles the seven. The design consequence is
the one that motivates the whole development: a model of a disordered region cannot be validated
against a deposited structure, because the deposited structure does not contain the region; what
a map can be compared with is a density forward-modelled from the ensemble.

## 6z. The entropy per residue, bracketed from both sides (Parts XXXIX-XL)

Part XII left the conformational entropy per residue of a self-avoiding square-lattice chain
bracketed between `log 2` and `log 4`. Both ends are now tightened, by finite computations at
each end.

`ConnectiveBound.lean` computes the exact seven-bond conformation count, `cnt_seven : c7 = 2172`,
by exhaustive kernel enumeration of all `4^7 = 16384` bond sequences. Since `2172 < 2187 = 3^7`
and the count is submultiplicative, `connectiveConstant_lt_log_three` gives `mu < log 3`. The
significance of `log 3` is that it is the entropy left by the crudest excluded-volume rule alone,
that a chain may not immediately retrace its previous bond, which leaves three continuations of
each bond and so at most `4*3^(n-1)` conformations (an elementary count, quoted here as the point
of comparison rather than formalised); the theorem says that real excluded volume costs strictly
more than that.

`Bridge.lean` supplies the matching device at the other end, Hammersley's bridges. A *bridge* for
an additive functional `phi` of position is a self-avoiding chain whose every site after the
first has `phi > 0` and no site exceeds the far end (`IsBridge`). `isBridge_append` proves that
two bridges concatenate to a bridge — the first chain lies weakly below its endpoint and the
second strictly above its start, so they cannot collide — whence bridge counts are
*super*multiplicative (`brCntOf_supermultiplicative`), the exact opposite of the
submultiplicativity of the full count. Iterating gives `brCnt N ^ k <= cnt (kN)`, and since the
entropy per residue is a limit, `log_brCntOf_div_le_connectiveConstantOf` turns *every* single
chain length into a lower bound as well as an upper one. On the square lattice the six-bond
bridge count is `101` (`brCnt_six`, again by exhaustive enumeration), giving
`mu >= (log 101)/6 = 0.769...`, a strict improvement on `log 2 = 0.693...`.

`entropy_per_residue_laws` (`PartForty.lean`) bundles the five statements: the entropy per
residue of a self-avoiding chain lies in `[(log 101)/6, log 3)`, both ends strictly interior to
the ideal-chain value `log 4 = 1.386`. The design consequence is quantitative: the conformational
freedom a generative model of a disordered region has to reproduce is a definite, and definitely
reduced, amount, certified on both sides.

## 6aa. Neutral evolution: conserved physics in a diverged sequence (Part XLI)

Disordered regions diverge in sequence far faster than in function, and the working resolution in
the field is that what is conserved is a set of sequence-level physical descriptors rather than
the residues themselves. `Evolution.lean` turns that into theorems about the mean-field
Debye--Hueckel descriptor already introduced in `Electrostatics.lean`.

The first observation is that any descriptor built from pairwise terms `q_i q_j w(|i-j|)` is
invariant under reading the chain backwards, because reversal preserves every separation:
`pairSum_rev` proves it once, and `screenedEnergy_rev`, `scd_rev`, `netCharge_rev` specialise it
to the screened electrostatic energy at every salt concentration, the sequence charge decoration
and the net charge. The invariance is a statement about that level of description and not a
triviality: `headCharge_not_rev_invariant` exhibits a directional descriptor, the net charge of
the N-terminal half, that reversal changes. The physics of a real chain is not reversal
invariant either -- the backbone is directional -- so the theorems apply exactly to models built
on descriptors that are.

The consequence is `diverged_sequences_same_physics`: the block polyampholyte `+^m -^m` and its
reversal have sequence identity exactly `0` -- they differ at *every* position
(`block_rev_identity_zero`) -- and identical screened energy, charge decoration and net charge.
Percent identity and the conserved descriptor can sit at opposite extremes simultaneously.
And the neutral set is not small: `composition_class_card` counts the zero-net-charge
composition class of a chain of length `2m` as `C(2m,m)`, `composition_class_exponential` gives
`4^m < m*C(2m,m)`, and `training_set_covers_vanishing_fraction` concludes that a training set of
`T` sequences drawn from the class covers at most a fraction `T*m/4^m` of it.

What a model should do about it is also a theorem. If the target is invariant under an involution
of sequence space and the model is not, `asymmetry_forces_error` bounds the sum of the two errors
below by the model's own asymmetry and `asymmetry_error_half` turns that into an error of at
least half the asymmetry on one of the two sequences; `symmetrised_error_le` and
`symmetrised_error_lt` show that averaging the model over the group never increases the squared
error and strictly decreases it whenever the model is asymmetric. `neutral_evolution_laws`
(`PartFortyOne.lean`) bundles the five statements.

## 6bb. Synthesis: the nascent chain is not the free chain (Part XLII)

Every disordered region is first present as a nascent chain emerging from the ribosome, while
models are built and validated against the mature chain at equilibrium.
`Cotranslational.lean` quantifies the two distinct errors that entails.

The first is lag. Writing `p t` for the actual distribution after `t` elongation events and
`pi t` for the equilibrium ensemble of the `t`-residue chain, with a per-step contraction factor
`delta < 1` towards the current equilibrium (the standard Dobrushin/spectral-gap input, supplied
as a hypothesis) and per-step equilibrium drift at most `d`, `tracking_error_le` proves the
adiabatic bound `dist(p t, pi t) <= delta^t dist(p 0, pi 0) + d(1-delta^t)/(1-delta)`, and
`tracking_error_le_steady` the steady form `... + d/(1-delta)`. The bound is not an artefact of
the estimates: `sharp_saturates` exhibits a chain meeting every hypothesis with equality whose
lag is exactly `d(1-delta^t)/(1-delta)`, and `sharp_tracking_error_tendsto` shows it converges to
`d/(1-delta)`. In rate variables, with `delta = e^{-lambda tau}` for relaxation rate `lambda` and
codon time `tau`, `quasi_static_limit` gives the quasi-static limit -- the bound tends to `d`, one
elongation step of drift, as synthesis slows -- while `lag_bound_ge_of_fast_synthesis` bounds the
same quantity below by `d/(lambda tau)`, which grows without limit as synthesis speeds up.

The second error survives infinitely slow synthesis. The equilibrium of the emerged fragment is
not the corresponding marginal of the mature chain's equilibrium, because partners not yet
synthesised are absent from the Hamiltonian. For the two-state caricature, `vectorial_gap`
computes the difference in the population of one fragment conformation exactly as
`(e^{beta eps} - 1)/(2(e^{beta eps} + 1))`, `vectorial_gap_pos` shows it is positive for every
stabilising contact, and `vectorial_gap_tendsto_half` that it saturates at `1/2`: the whole
population can move. `mature_model_error_on_nascent` reads that as the error of a model that is
exact on the mature chain. `cotranslational_laws` (`PartFortyTwo.lean`) bundles the five
statements: co-translational data are a separate observable, not a consistency check.

## 6cc. The material state: what condensate rheology excludes (Part XLIII)

Condensates formed by disordered regions are described first as liquid droplets -- one viscosity,
one relaxation time, Stokes--Einstein diffusion inside -- and measured to be nothing of the kind.
`Rheology.lean` proves what the three standard measurements exclude.

For a finite Maxwell spectrum `G(t) = sum_k g_k e^{-t/tau_k}`, which is what a coarse-grained
model with finitely many slow variables produces, `maxwell_le_exp` bounds the modulus by
`G(0) e^{-t/T}` with `T` the slowest mode, and `maxwell_lt_power_law` concludes that for every
power law `C t^{-alpha}` the model eventually falls strictly below it. The failure is not only
asymptotic in shape but categorical in the transport coefficient: `maxwell_integrableOn` and
`maxwell_viscosity` show a finite spectrum always has a terminal viscosity, equal to
`sum_k g_k tau_k`, whereas `power_law_not_integrableOn` shows a power law with `alpha <= 1` has
none at all.

For probe motion, `msd_of_uncorrelated` proves that uncorrelated stationary steps give exactly
`MSD(m) = m sigma^2` -- no freedom whatever -- so a measured sublinear displacement forces the
off-diagonal covariances to sum to a negative number (`subdiffusion_forces_anticorrelation`) and
some pair of steps to be correlated (`subdiffusion_forces_memory`);
`power_law_msd_forces_memory` applies this at every `m >= 2` for a measured `MSD(m) = A m^alpha`
with `alpha < 1`. Finally `aging_forces_error` prices non-stationarity: if the modulus measured
at two waiting times differs at some lag, a single waiting-time-independent predicted curve is
wrong by at least half that difference on one of them. `material_state_laws`
(`PartFortyThree.lean`) bundles the six statements.

## 6dd. Fitting an ensemble to data: how much agreement is evidence (Part XLIV)

The standard construction of an ensemble model reweights a pool of candidate conformations until
the predicted averages of `n` measured observables match the data, and reports the agreement.
`Fitting.lean` asks how much of that agreement is information about the ensemble.

`fit_with_few_structures` is Carathéodory's theorem read as a statement about ensembles: if the
data can be fitted at all, they can be fitted *exactly* by an ensemble supported on at most
`n + 1` conformations, however large the pool. Exact agreement with `n` measurements is
therefore reproduced by `n + 1` structures and constrains the ensemble only through feasibility.
`fit_not_unique` goes further: if the pool exceeds `n + 1` members and some fit gives them all
positive weight, the kernel of the fit map is nontrivial, so there is a nonzero weight direction
that sums to zero and is invisible in every measured observable, and a whole interval of steps
along it consists of exact fits. `fit_blind_to_unmeasured` makes this concrete with three
conformations and one datum: two exact fits assign the unmeasured observable the two extreme
values on the pool. The positive counterpart is `fit_unique_of_ker_trivial`: uniqueness of the
fit is exactly triviality of that kernel, which is the property an experimental design has to
establish and which a pool larger than `n + 1` never has. `ensemble_fitting_laws`
(`PartFortyFour.lean`) bundles the four.

## 6ee. The benchmark: what an incomplete annotation certifies (Part XLV)

Disorder predictors are compared on annotated benchmarks, and the annotation is one-sided: a
residue is annotated disordered when an experiment has shown it to be, and left unannotated
otherwise, whether it was shown to be ordered or never examined. `Benchmark.lean` derives what
follows for the reported numbers when the labels are sound but incomplete.

`precision_label_le_precision_truth` shows the reported precision is a lower bound on the true
precision, and `falsePos_label_eq` decomposes the reported false positives exactly into the true
false positives plus the predicted disordered residues the annotation has not yet recorded: a
predictor is penalised, residue by residue, for being ahead of the database. What the benchmark
does certify is quantitative -- `accuracy_close` bounds the difference between true and reported
accuracy by the annotation error rate `d/N`. What it does not certify is the ranking:
`benchmark_can_invert_ranking` exhibits a four-residue chain with sound labels on which the
predictor that is exactly right about the truth scores `3/4` while the predictor that merely
reproduces the annotation scores `1`. Since models are selected by benchmark order, incomplete
annotation is a mechanism that selects against generalisation. `benchmark_laws`
(`PartFortyFive.lean`) bundles the four statements.

## 6ff. Enthalpy--entropy compensation is a property of the fit (Part XLVI)

Coupled folding and binding is reported as a `(ΔH, ΔS)` pair obtained by fitting a van 't Hoff
line to log-constants measured at inverse temperatures `x i = 1/T i`, and across variants,
ligands and conditions those pairs are strongly correlated. `Compensation.lean` shows what the
correlation is. `intercept_eq_mean_sub` records the identity everything follows from: the fitted
intercept is `ȳ − slope · x̄`, so the entropy is the response extrapolated out of the
experimental window. `compensation_of_equal_mean` then gives the compensation identity — two
data sets with the same mean log-constant have intercepts differing by exactly `−x̄` times the
difference of their slopes, so the `(ΔH, ΔS)` points lie on an exact straight line whatever the
molecules are and whether or not anything compensates — and
`compensation_temperature_eq_harmonic_mean` identifies its slope as `n / Σ_i T_i⁻¹`, the harmonic
mean of the temperatures at which the experiment was run, a property of the design alone.
`deviation_from_compensation` isolates what is left: the departure from the line is exactly the
difference in mean log-constant over the window, and that is the only part of a compensation plot
that is about the molecules. `compensation_amplifies_error` reads the same identity as error
propagation — `|Δintercept| = |x̄|·|Δslope|`, so a fitted entropy is never better determined than
the fitted enthalpy divided by the harmonic mean temperature — and `compensation_example` gives
a two-point instance with both differences nonzero, so none of this is vacuous.
`compensation_laws` (`PartFortySix.lean`) bundles the five statements.

## 6gg. Which moment does the experiment weigh? (Part XLVII)

The two experiments that characterise a disordered region most often do not measure the same
thing. Scattering reports a second moment of the distance distribution; single-molecule FRET
reports the ensemble average of `R₀⁶/(R₀⁶ + r⁶)`, an observable dominated by the short distances.
`SaxsFret.lean` sets up the forward models (`eff_pos`, `eff_le_one`, `eff_antitone`), the
standard inversion in root-free form (`apparentSixth`, `apparentDistance`,
`apparentDistance_pow_six`), and derives what the inversion returns. `meanEff_ge` is the key
inequality, proved from Cauchy--Schwarz in Engel form: the efficiency of a heterogeneous
ensemble is at least that of the conformation whose sixth power of distance is the ensemble
mean. `apparentSixth_le_meanSixth` restates it as the statement an experimentalist uses -- the
inferred distance never exceeds the sixth-moment mean distance, so heterogeneity always makes a
chain look more compact than it is -- and `apparentSixth_eq_of_homogeneous` and
`apparentSixth_mem_Icc` show that this is a width effect, exact for a single conformation and
bracketed in general by the extreme distances present. `meanSq_cube_le_meanSixth` gives the only
general relation between the two experiments (`⟨r²⟩³ ≤ ⟨r⁶⟩`), and it is only an inequality:
`saxs_blind_to_fret` exhibits two two-state ensembles with the same mean square distance and
efficiencies `1/2` and `5/9`, and `fret_blind_to_saxs` two with the same efficiency `1/2` and
different mean square distances. A model must therefore be compared with each experiment through
its own forward model; agreement with one is not agreement with the other, and a reported
"FRET distance" and a reported "scattering size" are not comparable numbers. `saxs_fret_laws`
(`PartFortySeven.lean`) bundles the five statements.

## 6hh. The landscape does not determine the rate (Part XLVIII)

A free-energy profile along a reaction coordinate is what a simulation or a reweighted experiment
reports, and it is routinely read as if it fixed the kinetics. `FirstPassage.lean` solves the
associated hopping dynamics exactly and separates what the profile fixes from what it does not.
The setting is states `0, …, n` along the coordinate with forward rates `kp`, backward rates
`km`, a reflecting lower end, and detailed balance `p i · kp i = p (i+1) · km (i+1)`, i.e. the
profile is the equilibrium of the dynamics. `mfptFun_isMFPT` shows the closed form
`T m = Σ_{m ≤ i < n} (Σ_{j ≤ i} p j)/(p i · kp i)` solves the first-step-analysis system, and
`flux_eq`, `mfpt_unique`, `mfpt_eq` show it is the only solution: the discrete Kramers formula,
proved from detailed balance by the telescoping of the flux `p i · kp i · (T i − T (i+1))`.
Reading the formula, `detailedBalance_smul` and `mfptFormula_smul` give the central negative
result: multiplying every rate by `c > 0` leaves the profile — and hence every equilibrium
observable — exactly unchanged and divides every first-passage time by `c`
(`landscape_does_not_determine_rate`). The freedom is not just an overall prefactor:
`kinetics_not_determined_by_profile` gives two rate profiles on one flat landscape, differing
only at the second step, with crossing times `3` and `5`. What the profile does fix is a
bracket: `mfptFormula_ge_barrier` and `mfpt_ge_exp_barrier` give the Arrhenius lower bound
`T ≥ exp(β(F b − F 0))/kp b` for every intermediate state `b`, and `mfptFormula_le` the matching
upper bound `Σ_i 1/(p i · kp i)`. So the barrier height controls the exponential part and the
kinetic profile the rest. `first_passage_laws` (`PartFortyEight.lean`) bundles the six
statements.

## 6ii. The committor: nor the mechanism (Part XLIX)

Mechanism, in a hopping model, is the committor `q i`: the probability of reaching the product
end before returning to the reactant end, with "the transition state" defined as `q = 1/2`.
`Committor.lean` solves the harmonic system for it. `committorFun_isCommittor` and
`committor_unique` give the closed form
`q i = (Σ_{k<i} 1/(p k · kp k))/(Σ_{k<n} 1/(p k · kp k))` as its unique solution, by the
constancy of the reactive flux (`flux_const`) — the electrical analogy, with detailed balance as
Kirchhoff's law; `committor_strictMono` and `committor_mem_Icc` show it rises strictly from `0`
to `1`. What the committor sees is the resistance profile `1/(p k · kp k)`, not the weights
alone. Hence `committor_uniform_rate`: at constant diffusion coefficient the committor *is* a
functional of the equilibrium profile, `q i = (Σ_{k<i} 1/p k)/(Σ_{k<n} 1/p k)`, dominated by the
states of least weight — the precise sense in which "the transition state is the top of the
barrier" is a theorem. And hence also `mechanism_not_determined_by_landscape`: on one flat
landscape the uniform rate profile gives `q 1 = 1/2` and the profile with a slower second step
gives `q 1 = 1/3`. Neither the rate (Part XLVIII) nor the mechanism is a functional of the
landscape. `committor_laws` (`PartFortyNine.lean`) bundles the five statements.

## 6jj. A barrier is neither necessary nor sufficient for slow kinetics (Part L)

`BarrierRate.lean` states the practical form of Parts XLVIII-XLIX.
`flatLandscape_arbitrarily_slow`: for every `M` there is a detailed-balanced hopping model on a
perfectly flat three-state profile -- no barrier anywhere -- whose crossing time exceeds `M`, so
slowness is not evidence of a barrier (it is equally consistent with a small local diffusion
coefficient).  `barrier_arbitrarily_fast`: for every barrier height `B` and every `eps > 0` there
is a model whose profile has a barrier of exactly height `B` (`p 0 / p 1 = exp B`) and whose
crossing time is exactly `eps`, so a barrier is not evidence of slowness.  What survives is
`barrier_brackets_rate`: with the prefactor bounded, `kmin ≤ kp i ≤ kmax`, the time lies between
`p 0/(p b · kmax)` -- the Arrhenius factor of any intermediate state -- and
`(Σ_{i<n} 1/p i)/kmin`.  Arrhenius reasoning is a theorem about a landscape *plus* a bounded
diffusion profile.  `barrier_rate_laws` (`PartFifty.lean`) bundles the three statements.

## 6kk. Electronic polarisability, priced exactly (Part LI)

`Polarisability.lean` removes the fixed-charge idealisation that Part XXV named and left in
place. The induced dipoles of a point-polarisable model are defined implicitly, by the
self-consistent system `mu i = a i · (E i + Σ_j T i j · mu j)` (`SelfConsistent`), so the first
question is whether the repair is even well posed. Under the standard damping condition
`|a i| · Σ_j |T i j| ≤ c < 1` (`Damped` -- the exclusion of the polarisation catastrophe),
`selfConsistent_unique` and `selfConsistent_exists` prove there is exactly one solution for every
external field: injectivity of `I − aT` by a maximum argument, hence surjectivity in finite
dimension. The symmetric `m`-site cluster in a uniform field is then solved in closed form
(`clusterDipole_selfConsistent`, `energy_uniform_eq`): `mu = a/(1 − (m−1)at)`, `U = −(m/2)·mu`.
The inclusion--exclusion residue of the three-site cluster -- what is left after the *best
possible* one- and two-body terms -- is computed exactly, `threeBody_eq`:
`U₃ − 3U₂ + 3U₁ = −3a³t²/((1 − 2at)(1 − at))`. It is strictly negative whenever the coupling is
nonzero (`threeBody_neg`), so polarisation is cooperative, and it is second order in the coupling,
which is why a pairwise-fitted force field can be numerically decent and structurally wrong.
`polarisable_not_pairwise_additive` draws the consequence: *no* assignment of one- and two-body
energies reproduces the cluster energies of a polarisable model. A fixed-charge model is not a
polarisable model with bad parameters. `polarisability_laws` (`PartFiftyOne.lean`) bundles the
four statements.

## 6ll. Nuclear quantum effects and the isotope effect (Part LII)

`NuclearQuantum.lean` prices the other assumption Part XXV left outside: that the nuclei are
classical. For one harmonic mode of quantum `w = ħω` at inverse temperature `b`,
`qFree_eq_zpe_add` gives `F_q = w/2 + (1/b) log(1 − e^{−bw})`, and `clFree_lt_qFree`,
`qFree_lt_zpe` the sandwich `F_cl < F_q < w/2` -- the first from `sinh x > x`, so there is no
temperature at which a classical harmonic mode has the right free energy. `qEnergy_gt_kT` is the
same failure for the mean energy (from `sinh x < x cosh x`, proved from the derivative). The
failure is not universal: `tendsto_partition_ratio_one` shows quantum and classical partition
functions agree in the soft-mode limit, so the collective motions of a disordered chain are safely
classical and the stiff stretches are not. The sharp statement is the pair
`classical_isotope_independent` / `quantum_isotope_effect`: a mass substitution scales every
frequency by one factor `s`, which cancels *exactly* from every classical free-energy difference,
so a classical force field predicts an equilibrium isotope effect of identically zero -- not a
small one -- while the quantum model's low-temperature limits are the zero-point differences
`(wA − wB)/2` and `s(wA − wB)/2`, which differ. A measured H/D equilibrium effect is a measurement
of something a classical model assigns the value zero. `nuclear_quantum_laws`
(`PartFiftyTwo.lean`) bundles five statements.

## 6mm. The orientation factor: an efficiency is not a distance (Part LIII)

`Orientation.lean` removes the `κ² = 2/3` substitution Part XLVII left standing.
`kappaSq_le_four` proves the exact range `0 ≤ κ² ≤ 4` for unit dipoles and a unit separation
direction, from the Gram determinant of the three vectors (`gram_identity`: the determinant is the
squared triple product, so nonnegativity is a polynomial identity rather than an assumption).
`meanKappaSq_eq` exhibits a finite orientational model -- donor and acceptor each uniform over the
six axis directions -- whose mean orientation factor is exactly `2/3`. And that is all `2/3` is:
`meanEff_ne_effMean` shows that for the same model, at the distance where the `κ² = 2/3` formula
returns `E = 2/5`, the true mean efficiency is `1/5`, because efficiency is not linear in `κ²`;
read as a distance (`apparentSixth_of_meanEff`) the measurement returns `r⁶ = 8/3` where the truth
is `1`. The residual uncertainty does not shrink with photon statistics:
`apparentSixth_ratio_six` shows the distances inferred at fixed efficiency with `κ² = 4` and with
`κ² = 2/3` differ by a factor `6` in `r⁶`. For a disordered region -- short linkers, sticky dyes,
incomplete rotational averaging -- an ensemble is not comparable with a FRET efficiency unless the
model carries the dye orientational distribution or the analysis reports the bracket.
`orientation_laws` (`PartFiftyThree.lean`) bundles five statements.

## 6nn. Transition-path times: the crossing is not the waiting (Part LIV)

`TransitionPath.lean` brings inside the development the object a single-molecule experiment
actually resolves, with no new dynamical assumption. The committor `h`-transform is constructed
explicitly -- `reactP p q i = p i · q i²`, `reactKp i = kp i · q(i+1)/q(i)`,
`reactKm i = km i · q(i−1)/q(i)` -- and `reactive_detailedBalance` proves the conditioned dynamics
is *again* a detailed-balanced hopping chain, `reactKm_zero` that its lower end is automatically
reflecting: a reactive trajectory cannot return. Everything proved in Part XLVIII therefore
applies verbatim: `tpt_eq` gives existence, uniqueness and the closed form of the mean
transition-path time (the discrete Kramers formula for the reweighted profile `p q²`) and
`tptFormula_pos` its positivity. `slow_reaction_fast_paths` is the law the two times violate
together: for every `M` there is a detailed-balanced three-state model whose mean first-passage
time is at least `M` and whose mean transition-path time is at most `1/M`. The witness is solved
in closed form -- on the profile `(1, e, 1)` with unit forward rates the times are `1 + (1+e)/e`
and `e/(1+e)` -- so raising the barrier makes the reaction slower and the crossings faster. No
function of one returns the other. `transition_path_laws` (`PartFiftyFour.lean`) bundles four
statements.

## 6oo. Heat capacity: the curved van 't Hoff plot (Part LV)

`HeatCapacity.lean` covers the case Part XLVI excluded. With a constant `ΔCp`,
`ΔH(T) = ΔH₀ + ΔCp(T − T₀)` and `ΔS(T) = ΔS₀ + ΔCp log(T/T₀)`, everything is an identity.
`vantHoff_eq_enthalpy_at`: the two-point van 't Hoff enthalpy of a window `[T₁, T₂]` is the *true*
enthalpy at one interior temperature, `ΔH_vH = ΔH(M)` with
`M = T₁T₂ log(T₁/T₂)/(T₁ − T₂)`, and `logMeanRecip_mem_Ioo` places `M` strictly between the
endpoints (from `log x < x − 1` applied in both directions), whence `vantHoff_ne_endpoints`: once
`ΔCp ≠ 0` the fitted number is the enthalpy at neither temperature of the window. `secondDiff_eq`
computes the curvature exactly -- sampling `log K` at three temperatures equally spaced in `1/T`
gives `(ΔCp/R) log((T₁+T₂)²/(4T₁T₂))` -- which vanishes identically when `ΔCp = 0` and otherwise
carries its sign, by strict arithmetic--geometric mean. `ΔCp` is thus identifiable from `log K`
alone and invisible to the linear fit whose algebra Part XLVI analysed. For a disordered region,
where burial of apolar surface on binding makes `|ΔCp|` large, this is the ordinary case.
`heat_capacity_laws` (`PartFiftyFive.lean`) bundles four statements.

## 6pp. Aggregation after the early-time regime (Part LVI)

`Depletion.lean` treats the saturated kinetics Part XXXVII excluded: `M' = κM(1 − M/m₀)`, solved
by `M(t) = m₀/(1 + e^{−κ(t − t½)})`. `fibrilMass_hasDerivAt` verifies the solution and
`fibrilMass_pos`, `fibrilMass_lt_total`, `fibrilMass_strictMono`, `tendsto_fibrilMass_total` give
positivity, the bound by the total protein, monotonicity and the plateau; `fibrilMass_le_exp`
shows depletion only ever slows growth, so the early-time treatment is an upper bound of known
sign rather than an approximation of unknown sign. `crossing_eq` is the threshold identity: the
fraction `f` is reached at `t½ + log(f/(1−f))/κ`, and `crossing_threshold_shift` shows two
detection thresholds give genuinely different crossing times, so a quoted lag is a statement about
the instrument as much as about the sample. `tangent_slope` and `tangentLag_eq` give the standard
construction -- maximal slope `κm₀/4` at `t½`, tangent meeting the baseline at `t½ − 2/κ` -- and
`lag_does_not_determine_rate` shows the lag fixes neither `κ` nor `t½`. In the early-time regime
the lag is a logarithm of the nucleation rate, in the saturated regime a two-parameter shadow of
the whole curve, and in neither is it a rate. `depletion_laws` (`PartFiftySix.lean`) bundles seven
statements.

## 6qq. Anisotropic displacement and discrete conformers (Part LVII)

`Anisotropy.lean` removes the two restrictions Part XXXVIII named: one occupancy and one
*isotropic* displacement parameter.  The two-conformer density factorises exactly,
`twoSite s d x = gauss s x · e^{-d²/2s²} · cosh(xd/s²)` (`twoSite_eq`), and `twoSite_ne_gauss`
draws the consequence: for *every* single-site width `u > 0` whatsoever the two densities differ
somewhere, so the refinement program's one-parameter family does not contain the two-site density
at all.  Inflating `B` is not an approximation to splitting a site.  The error has a sign where
one looks: at the second-moment-matched width `u² = s² + d²` the fit is strictly too high at the
midpoint (`twoSite_center_lt_matched`, which is `1 + t < e^t` in disguise), and once the sites are
resolved, `2s² ≤ d²`, the true density dips at the midpoint (`twoSite_bimodal`) -- a shape no
Gaussian has anywhere.  For the tensor, `anisoDensity_inj` is the positive result -- an
anisotropic density determines its principal widths -- and `equivB_not_determining` with
`aniso_ratio_unbounded` the price of the isotropic fit: two tensors with the same isotropic `B`
have different densities, and at fixed `B` the ratio of principal widths is unbounded.
`occupancy_anisotropy_degenerate` carries the Part XXXVIII occupancy trade-off into tensor form.
`anisotropy_laws` (`PartFiftySeven.lean`) bundles five statements.

## 6rr. Orientational NMR: residual dipolar couplings (Part LVIII)

`Rdc.lean` adds the observable the development lacked: every other one here is a distance, a
rate, a population or a coupling constant, and none constrains a *direction*.  With
`rdc A u = (3uᵀAu − tr A)/2` and `A` symmetric traceless, `rdc_neg` gives the reversal invariance
of a bond vector; `rdc_level_set_circle` shows that for an axially symmetric tensor the coupling
is constant on a whole circle of unit directions, so one medium leaves a *continuous* ambiguity;
`rdc_axes_mean_zero` shows the coupling averaged over three orthogonal directions vanishes for
every alignment tensor.  `order_population_degenerate` exhibits a fully ordered ensemble at an
intermediate angle and a half-ordered one at the pole with exactly equal couplings: population and
order parameter enter only through their product.  For ensembles,
`meanRdc_eq_quad_secondMoment` shows the average is a linear functional of the second moment
`⟨u_i u_j⟩`, so `meanRdc_eq_of_secondMoment_eq` -- ensembles sharing a second moment are
indistinguishable in *every* medium at once -- and `secondMoment_not_injective` exhibits two
ensembles with no conformer in common that share it.  What is determined is five numbers:
`alignment_decomposition` decomposes a symmetric traceless tensor on five basis tensors and
`rdc_five_directions_determine` recovers it from five bond directions, which is the linear algebra
behind "five independent alignment media", proved rather than asserted.  `rdc_laws`
(`PartFiftyEight.lean`) bundles six statements.

## 6ss. Benchmarks with unsound annotations (Part LIX)

`LabelNoise.lean` drops the soundness assumption of Part XLV: errors in the annotation now run in
both directions and nothing is assumed about their direction.  `errors_le_add_noise` and its
converse bracket the measured error count by the true one to within `noise T L`, the number of
mislabelled residues, which is the entire positive content of an unsound benchmark.
`perfect_predictor_penalised` is the sharp form of the problem: a model that reproduces the truth
residue by residue is recorded as making exactly `noise` mistakes.  `ranking_certified` is the
usable statement -- a margin of more than twice the noise certifies the ranking -- and it cannot
be improved: `ranking_inversion_false_positive` and `ranking_inversion_single_label` invert a
ranking with, respectively, two and one mislabelled residues, and `ranking_certificate_sharp`
records a tie at a margin of exactly twice the noise.  A leaderboard whose margins are smaller
than the annotation error rate reports a ranking the data do not contain.  `label_noise_laws`
(`PartFiftyNine.lean`) bundles five statements.

## 6tt. Explicit solvent: the potential of mean force (Part LX)

`Pmf.lean` supplies the item Part XXV named and left outside -- the structure of the solvent --
and does so exactly.  For a finite solvent state space and an arbitrary solute--solvent
interaction, `marginal_eq` proves that the solute marginal of the joint Boltzmann measure is
*identically* the Boltzmann measure of `Usol + pmf`, at every temperature: integrating out the
water is not an approximation, and the only question is what the resulting object is.  It is not
a force field.  In the smallest model that can show it -- one water molecule with a bound and a
free state, stabilised additively by each of three solutes in contact, so that the *interaction*
is exactly pairwise -- `solventThreeBody_eq` computes the inclusion--exclusion residue in closed
form, `(1/b)·log(250/243)` at the coupling `b·eps = log 2`, and `solventThreeBody_pos` gives its
sign: strictly positive, so the solvent-mediated three-body force is anti-cooperative.  A
logarithm of a sum is not a sum.  Nor is it a potential: `pmf_temperature_dependent` shows the
solvation contribution of a single solute takes different values at two temperatures, so it is a
free energy carrying an entropy and cannot be tabulated once and transferred.  For a region that
is mostly surface, this is where collapse, the temperature dependence of the radius of gyration
and hydrophobic cooperativity live.  `pmf_laws` (`PartSixty.lean`) bundles four statements.

## 6uu. What a single-molecule FRET number is worth: photophysics and linkers (Part LXI)

Part XXXIII models a burst as clean photon counting and Part LIII removes the `κ² = 2/3`
substitution; `Photophysics.lean` removes the two idealisations that survived.  Photon counts give
the *proximity ratio* `nA/(nA+nD)`, and the transfer efficiency needs the detection correction
factor `g`.  `proximityRatio_eq_self_iff` -- the raw ratio is the efficiency exactly when `g = 1`,
and at no interior efficiency otherwise.  `proximityRatio_strictMono` -- but it is strictly
increasing in the efficiency, so the *ranking* of conformers and the direction of any change
survive an unknown `g` intact.  `sixthPower_off_by_gamma` -- the magnitude does not: since
`E/(1−E) = (R₀/r)⁶`, the uncorrected analysis returns `(1−P)/P = (1/g)·(1−E)/E`, so the inferred
`r⁶` is the true `r⁶` divided by `g`, exactly; and `gamma_unidentifiable` -- every observed ratio
in `(0,1)` is produced by every `g > 0` at a suitable true efficiency, so `g` is not in the data.
`apparentEff_pos_of_background` -- with any acceptor background at all, a state with zero transfer
measures at strictly positive efficiency, and the extended states of a disordered region are
exactly the zero-transfer ones, so the bias is towards compaction.  `linker_bound` --
`|dist b₁ b₂ − dist a₁ a₂| ≤ 2L` in any metric space, from the triangle inequality alone, and
`linker_bound_sharp` -- the bracket is attained at both ends by two configurations with the same
attachment distance whose dye distances differ by `4L`.  The linker correction is a range, not a
shift or a scale factor.  The three multiplicative corrections a single-molecule distance carries
-- `κ²`, `g`, and the linker range -- are independent, and none of them shrinks when more photons
are collected.  `photophysics_laws` (`PartSixtyOne.lean`) bundles six statements.

## 6vv. The Markov state model that is actually estimated (Part LXII)

Part IV.6 (`Kinetics.lean`) settles the qualitative question about clustering microstates: under
Dynkin's condition the coarse variable is Markov, and generically it is not.  What is done in
practice is not to check that condition but to count transitions between clusters along an
equilibrium trajectory at some lag and normalise, giving the stationary-weighted lumping
`macroW a b = (Σ_{i∈a} π i · Σ_{j∈b} P i j)/(Σ_{i∈a} π i)` (`MarkovStateModel.lean`).  What that
matrix is a statement about splits into three.  *It is a consistent estimator*:
`macroW_eq_of_lumpable` -- when the clustering happens to be lumpable, the estimate is the lumped
matrix.  *Its thermodynamics is unconditionally right*: `macroW_nonneg`, `macroW_stochastic`,
`macroW_stationary`, `macroW_detailedBalance` -- with no condition on the clustering whatever, the
estimate is a stochastic matrix, the coarse equilibrium populations are stationary for it, and
detailed balance is inherited.  A Markov state model reproduces the populations of its own
clustering by construction, so agreement there is no evidence that the clustering is good.  *Its
kinetics is one-sided*: `mean_lump_eq`, `nrm_lump_eq` and `form_lump_eq` show the coarse mean,
mean square and quadratic form of a macro observable `g` are exactly the microscopic ones of
`g ∘ phi`, so by `macro_gap_bound` every bound on the microscopic Rayleigh quotient is inherited.
Coarse-graining cannot invent a slow mode; a reported implied timescale is a lower bound on the
truth.  Finally the lag: `chapman_kolmogorov_of_lumpable` -- under lumpability the lag-two
estimate is the square of the lag-one estimate, so the implied-timescale test is a genuine test of
the modelling assumption -- and `chapman_kolmogorov_fails` -- for an explicit three-state chain
that is stochastic, reversible and uniformly populated, with no absorbing state and no vanishing
population, the lag-two estimate of the escape probability is `1/4` while the square of the
lag-one estimate is `5/16`.  The matrix a Markov state model reports is a function of a modelling
choice.  For a disordered region, whose clusters are cuts through a continuum rather than genuine
metastable basins, the reading is: quote populations without qualification, quote timescales as
lower bounds, and do not quote a state-decomposition kinetic model at all without the lag it was
estimated at.  `markov_state_model_laws` (`PartSixtyTwo.lean`) bundles five statements.

## 6ww. Pulling out of equilibrium: work, dissipation and the free energy (Part LXIII)

Part XV.2 treats force spectroscopy at equilibrium; a real optical-tweezer or AFM pull is not at
equilibrium, and what is recorded is the *work* of each pull.  `WorkTheorem.lean` takes the single
physical hypothesis -- microscopic reversibility, `p γ = q(rev γ)·exp(b·(W γ − dF))`, with `q` the
time-reversed protocol -- and derives what the work distribution determines.
`crooks_satisfiable` shows the hypothesis is not vacuous and fixes `dF` as the logarithm of the
exponential average.  `jarzynski` -- `⟨exp(−b·W)⟩ = exp(−b·dF)` exactly, however fast the pull, so
a nonequilibrium experiment does determine an equilibrium free energy.
`dissipation_eq_relEntropy` -- the dissipated work is exactly a relative entropy,
`b·(⟨W⟩ − dF) = Σ p·log(p / q∘rev)`: irreversibility is the statistical distinguishability of the
pull from its own time reverse.  `second_law` -- hence `dF ≤ ⟨W⟩`, and
`no_dissipation_iff_deterministic` -- with equality exactly when the work is the same on every
trajectory.  For a disordered chain, which has no folded state to hold the work distribution
narrow, the mean work is an upper bound and not an estimate.  `crooks_histogram` and
`crooks_crossing` -- the forward and reverse work histograms satisfy
`P_F(w) = exp(b·(w − dF))·P_R(−w)` and therefore cross exactly at `dF`: the one construction that
returns the free energy with no model in between.  Against that, `rare_trajectories_dominate` --
for every `M` there is a work distribution whose rare branch has probability at most `exp(−M)`,
contributes more than all the rest to the exponential average, and puts the free energy at `−M` or
below, while the estimate from the typical branch alone is `0`; and `omitting_low_work_overestimates` -- the error has a sign, since discarding any set of trajectories whose work is below
all the retained ones can only raise the estimate.  `work_theorem_laws` (`PartSixtyThree.lean`)
bundles six statements.

## 6xx. Structure-based coarse-graining: the inverse problem (Part LXIV)

Part LX treats the forward direction; `InversePotential.lean` treats the inverse one, which is
how coarse-grained models of disordered regions are built: fit a potential so that the model
reproduces a measured structural statistic, then use it elsewhere.
`gibbs_unique_of_meanFeature_eq` -- **the fit is well posed**: Henderson's uniqueness theorem in
the discrete setting, two parameter vectors whose Boltzmann distributions have the same mean
features have the same Boltzmann distribution.  The proof is the symmetrised relative entropy,
`KL(p‖p') + KL(p'‖p) = b·⟨theta' − theta, ⟨n⟩_p − ⟨n⟩_p'⟩`, together with Gibbs' inequality; a
fitted potential is therefore not one arbitrary choice among many that fit.
`pair_potentials_blind_to_three_body` -- **but the model it determines is blind beyond the
statistics it was fitted to**: on three spins the parity ensemble (uniform on the four
configurations with `s₁s₂s₃ = +1`) has all three pair correlations zero, exactly like the
uniform ensemble, and triple correlation `1`; by uniqueness, every pair-potential model matching
those pair correlations *is* the uniform ensemble, whose triple correlation is `0`.  Fitting the
pair structure exactly gets the three-body structure maximally wrong, and
`zero_matches_pair_structure` confirms the hypothesis is satisfiable, so the statement is not
vacuous.  `inverse_potential_temperature_dependent` -- **and the fitted potential is a free
energy**: in the smallest two-state model the potential reproducing the feature value `1/3` at
inverse temperature `1` is `log 2`, the one reproducing the same value at inverse temperature
`2` is `(log 2)/2`, and the first transferred to the second temperature does not reproduce the
target.  Together with Part LX this brackets coarse-graining from both sides.
`inverse_potential_laws` (`PartSixtyFour.lean`) bundles four statements.

## 6yy. Replica exchange: unbiased, and not a cure (Part LXV)

Every reported ensemble of a disordered region is the output of an enhanced-sampling run, and in
practice that means replica exchange.  `ReplicaExchange.lean` treats the two-replica extended
system exactly on a finite conformation space.  `swapK_detailedBalance`, `swapK_stationary`,
`prodW_marginal_fst`, `prodW_marginal_snd`, `reK_stationary` -- **it is unbiased**: the swap move
is reversible with respect to the product of the two Boltzmann ensembles, hence stationary for
it, the marginal at each temperature is exactly that temperature's Boltzmann ensemble, and the
whole chain (within-temperature moves plus swaps) preserves the extended ensemble.  The
correction to reported populations from swapping is exactly zero, not small.
`accSwap_boltz` -- the acceptance is `min 1 exp((b2−b1)(E j − E i))`, with no partition function
in it, which is why the move is implementable.  `meanAcc_eq_overlap`, `meanAcc_eq_one_sub_tv` --
**the acceptance rate is exactly an overlap**, `1 − TV(pi, pi∘swap)`, and measures nothing else.
`meanAcc_le_exp_neg_gap`, `meanAcc_two_state`, `deltaBeta_le_of_meanAcc`, `replicas_needed` --
**a temperature step costs exponentially in the energy gap**: histograms separated by `g` accept
at most `exp(−(b2−b1) g)`, attained exactly in a two-state instance, so holding acceptance `al`
forces `(b2−b1) g ≤ log(1/al)` and a ladder spanning `[b 0, b K]` needs at least
`(b K − b 0) g / log(1/al)` rungs; the gap is extensive, so the replica count grows with the size
of the region.  `meanAcc_flat_eq_one` -- and a high acceptance rate is no evidence of anything:
at equal temperatures every swap is accepted while the two replicas sample the same ensemble.
`reK_conserves_count`, `re_iterate_support`, `replica_exchange_cannot_repair_ergodicity` -- **the
negative result**: if no within-temperature move crosses between a set `A` of conformations and
its complement, the number of replicas inside `A` is a conserved quantity of the whole extended
chain, so a run started inside `A` never leaves it, for any number of sweeps and at any
temperature, although the Boltzmann ensemble puts positive weight outside.  Tempering lowers
barriers; it does not connect what the move set disconnects.  `replica_exchange_laws`
(`PartSixtyFive.lean`) bundles five statements.

## 6zz. Umbrella sampling: overlap is exactly the condition (Part LXVI)

A free-energy profile along a coordinate is assembled from biased windows; `Umbrella.lean` treats
the assembly exactly.  `winDist_unbias` -- within a window there is nothing left to argue about:
reweighting the biased histogram by `exp(V)` returns exactly the conditional distribution of the
target on the window support.  `winDist_smul` -- but a window fixes only the *shape* on its own
support, which is the precise reason WHAM and MBAR return offsets up to one global constant.
`free_energy_unidentifiable` -- **the sharp negative**: if the conformation space splits into `A`
and its complement with no window straddling the split, then for *every* positive ratio `r` there
is a target with that ratio of populations reproducing every window histogram exactly.  The
relative free energy is not noisy, it is arbitrary.  `umbrella_identifiable`,
`population_determined` -- **and the matching positive result**: if consecutive window supports
meet and the windows cover the space, two targets with the same window data are proportional,
hence give identical populations and identical free-energy differences.  Connectivity of the
window supports is therefore exactly the right condition, necessary and sufficient.
`overlap_missed_ge`, `half_of_runs_blind`, `thin_overlap_example` -- and overlap on paper is not
overlap in the data: a shared region of probability `m` is missed entirely by an `N`-frame run
with probability at least `1 − N m`, so runs shorter than `1/(2 m)` frames return, at least half
the time, data whose *recorded* supports do not straddle -- and then the negative theorem applies
to what was actually recorded.  `umbrella_laws` (`PartSixtySix.lean`) bundles five statements.

## 6aaa. How much sequence a pairwise theory can carry (Part LXVII)

The analytic sequence-to-ensemble theories in use -- sequence charge decoration, random-phase
free energies, preaveraged Debye-Hückel treatments -- share one structural feature: the charge
sequence enters through pairwise terms whose strength depends on the separation along the chain.
`SequenceDegeneracy.lean` measures how much sequence information survives that structure.
`pairSum_eq_shell` -- such a model sees the sequence **only** through its autocorrelation
`shell d = Σ_i q i q (i+d)`; the identity holds for every kernel, hence at every screening length,
salt concentration and temperature.  `confEnergy_eq_of_shell_eq`, `sequence_blind`,
`scd_congr_of_shell_eq` -- consequently two sequences with equal autocorrelation have the same
conformational energy function, and therefore the same value of *every* functional of the model:
same partition function, same Boltzmann ensemble at every temperature, same radius of gyration,
same value of every observable, same sequence charge decoration.  `shell_seqA_eq_seqB`,
`seqA_ne_seqB` -- **and the autocorrelation does not determine the sequence**: the two explicit
12-residue charge patterns `+ + + + − + − − + + − −` and `+ + − + + − + + + − − −` have equal net
charge, equal composition and equal autocorrelation at every separation, while being distinct,
not each other's reverse, and not each other's charge inversion.  `triple_seqA`, `triple_seqB`,
`three_body_resolves` -- their nearest-neighbour three-body correlations are `2` and `−6`, so a
third-order term separates exactly the pair that every pairwise separation-dependent theory
conflates.  A model is sequence-resolved only in so far as its sequence dependence is not
pairwise-in-separation.  `sequence_resolution_laws` (`PartSixtySeven.lean`) bundles four
statements.

## 6bbb. The full ladder: `K` replicas, and the invariant that survives (Part LXVIII)

Part LXV treats one exchanging pair; a production run is a ladder.  `ReplicaLadder.lean` proves
the same facts for `K` replicas, through a general lemma that isolates the mechanism.
`invK_stochastic`, `invK_detailedBalance`, `invK_stationary`, `invAcc_mean_eq_overlap` -- **a
Metropolis move along an involution** of any finite state space is a transition kernel, is
reversible with respect to any positive weight, leaves it stationary, and accepts at exactly the
overlap rate; a replica exchange is such a move, the involution being "swap the configurations
held by replicas `a` and `c`".  `ladderW_ratio`, `ladder_acc_boltz` -- on the ladder the
acceptance collapses to the two replicas involved, `min 1 exp((b c − b a)(E (x c) − E (x a)))`,
with no partition function and no dependence on the other `K − 2` replicas.  `ladder_unbiased`,
`ladder_swap_stationary` -- **every rung is unbiased**: the extended ensemble returns exactly the
Boltzmann average at each replica's own temperature, for every observable, and the exchange
preserves the extended ensemble.  `ladder_swap_conserves_count`, `ladder_within_conserves_count`,
`ladder_iterate_support`, `ladder_cannot_repair_ergodicity` -- **and the conserved count survives
the whole ladder**: with no within-temperature move crossing the boundary of `A`, the number of
replicas holding a configuration in `A` is unchanged by every exchange and every update, so a run
started with all `K` replicas inside `A` never leaves it.  Adding rungs or widening the
temperature range cannot help, because the obstruction is a conserved quantity of the move set and
the temperatures do not enter it.  `replica_ladder_laws` (`PartSixtyEight.lean`) bundles four
statements.

## 6ccc. How many experiments an ensemble costs (Part LXIX)

An ensemble model is a population vector over a library of `m` conformations, fitted against `k`
experimental averages.  `Restraints.lean` settles the relation between the two numbers.
`exists_null_direction` -- `k` observables together with normalisation are `k + 1` linear
functionals on an `m`-dimensional population space, so once `k + 1 < m` they annihilate a nonzero
signed population direction.  `exists_perturbation` -- and against an *interior* target, one
carrying weight at least `d` on every conformation, which is what disorder means, that direction
can be followed a finite distance without leaving the simplex.  `restraints_insufficient` -- the
design statement: below the threshold there is a genuine ensemble, nonnegative and normalised,
reproducing **every** measured average exactly at population distance at least `2 d` from the
truth; uniformly, `2/m`, the weight of two whole conformations
(`uniform_restraints_insufficient`).  `experiments_needed`, `restraints_exp_entropy` -- so
determining an interior target needs at least `m - 1` restraints, one per conformation, and since
a library realising conformational entropy `H` has `exp H` members the restraint count needed is
exponential in the conformational entropy.  `cross_validation_cannot_certify` -- and a held-out
validation set buys nothing: if the *total* count is below threshold, one ensemble far from the
truth matches the fitting data and the validation data alike.  `agree_except_one`,
`indicator_restraints_determine` -- the threshold is sharp, since the `m - 1` indicator
observables do determine the ensemble.  `feasible_convex` -- and there is no lucky restraint set
below it, because the feasible set is a convex slice of the simplex whose deficient directions
form a subspace.  `restraint_counting_laws` (`PartSixtyNine.lean`) bundles six statements.

## 6ddd. What is verifiable in a reported ensemble (Part LXX)

Given the restraints one does have, which reported numbers are consequences of the data?
`Identifiability.lean` answers exactly, for the linear functionals of the ensemble -- a substate
population, a contact frequency, a secondary-structure content, a mean radius of gyration.
`determined_of_mem` -- every functional in `measured g`, the span of the constant function
(normalisation is a restraint too) together with the measured observables, is determined: all
ensembles matching the data report the same value.  `exists_vector_of_dual`,
`not_determined_of_notMem` -- and nothing else is, because a functional outside the span is
separated from it by a linear functional, a linear functional on population space *is* a signed
population direction, and being null on the span it is invisible to the data while the interior
target can be moved along it.  `determined_iff_mem` -- so identifiability is exactly membership in
a subspace fixed by the experiment list: not a matter of regularisation, prior width, or goodness
of fit.  `identifiable_dim` -- and that subspace has dimension at most `k + 1`, so `k` experiments
support at most `k + 1` independent verifiable numbers, whatever the library or the fitting
method; `exists_population_not_determined` -- with `k + 1 < m`, at least one single-conformation
population is not among them.  `identifiability_laws` (`PartSeventy.lean`) bundles five
statements.

## 6eee. The precision floor of an ensemble measurement (Part LXXI)

Parts LXIX and LXX match the data exactly; real data are matched to a tolerance.
`Tolerance.lean` prices it.  `exists_pair_perturbation` -- transferring population `c` from
conformation `b` to conformation `a` changes the `j`-th measured average by exactly
`c (g j a - g j b)` and moves the ensemble by at least `2 c`: only the *contrast* of an observable
between two conformations is visible.  `pair_degeneracy` -- hence the conformational counterpart
of Part LXVII's sequence degeneracy: two conformations on which every measured observable takes
the same value have a completely undetermined relative population, an ensemble `2 d` away
matching every measurement exactly, for every restraint count.  `precision_floor` -- and the
quantitative law: with observables bounded by `G` and data matched to tolerance `eps`, the
ensemble is pinned down only to population distance `min (2 d) (eps / G)`.  The bound contains no
`k`; more experiments cannot lower it, only a smaller `eps` can.  `tolerance_ceiling` -- the
matching upper bound, `2 (m - 1) eps` when the populations themselves are known to `eps`, so the
resolution is linear in the tolerance from both sides.  `precision_laws`
(`PartSeventyOne.lean`) bundles four statements.

## 6fff. The design and reporting theorem (Part LXXII)

Parts LXIX-LXXI price the ensemble; `ReportDesign.lean` turns the price into a procedure and
proves the procedure optimal.  `mem_measured_iff_combo` -- a functional is identifiable exactly
when it carries a certificate, an explicit representation `c0 + sum_j c j g j` in terms of
normalisation and the measured observables.  `report_value_from_data` -- and the certificate
computes the number from the data: every ensemble consistent with the measured averages `D`
reports `c0 + sum_j c j D j`, so the identifiable content of a fit is a linear readout of the
deposited data and the fitted ensemble contributes nothing to it; `consistent_ensembles_agree`
states the same fact as agreement of any two consistent fits.  `report_sufficient` -- the design
half: measuring the functionals one intends to report determines all of them and every
combination of them with normalisation, so `r` experiments buy `r + 1` verifiable dimensions.
`report_necessary`, `report_min_experiments`, `report_needs_r` -- and that is optimal: any
experiment set making all `r` reports verifiable against an interior target satisfies
`finrank (span {1, f 1, ..., f r}) <= k + 1`, so with linearly independent reports at least `r`
experiments are needed, and the minimum is exactly `r` -- independently of the library size.
Determining the ensemble costs `m - 1` experiments and is hopeless; determining `r` reported
numbers costs exactly `r` and is routine.  `report_design_laws` (`PartSeventyTwo.lean`) bundles
four statements.

## 6ggg. How much sequence a pairwise charge model can carry (Part LXXIII)

`ChargePatterning.lean` prices the sequence side of a condensation model.  Every patterning
parameter in use -- `SCD`, `kappa`, a screened-Coulomb chain energy -- has the form
`E(q) = sum_{i<j} w(j-i) q_i q_j` for some kernel `w`.  `pairEnergy_eq_sum_autocorr` regroups that
sum by separation: `E(q) = sum_{d=1}^{N-1} w d * C(d)`, so *every* such model is a linear
functional of the `N - 1` charge autocorrelation coordinates.  `autocorr_eq_iff_pairEnergy_eq`
makes the converse precise -- two sequences agree under every kernel exactly when their
autocorrelations agree, since `pairEnergy_delta_kernel` shows each `C(d)` is itself realised by a
Kronecker kernel.  The blindness is therefore exactly measured, and it is not empty:
`homometric_same_composition`, `homometric_autocorr` and `homometric_blind` exhibit two explicit
length-nine charge sequences -- `+ + - + + - - - +` and `- + + - + + + - -` -- with the same
composition and the same autocorrelation at every lag, hence the same energy under every kernel,
the same `SCD` (`homometric_scd`) and the same Debye-screened energy at every salt concentration
(`homometric_debye`).  They are genuinely different sequences (`chargeB_ne_chargeA` and the three
companion statements rule out equality with the reversal, the negation and the reversed negation),
and a three-body descriptor separates them: `homometric_triple_ne` computes `+2` against `-2`.
Completeness therefore requires many-body sequence terms or an explicit ensemble.
`charge_patterning_laws` (`PartSeventyThree.lean`) bundles the statements.

## 6hhh. Multivalent binding: what a Hill slope can be (Part LXXIV)

`BindingPolynomial.lean` treats the binding polynomial of a disordered multivalent region as an
analytic object.  `hasDerivAt_mom` and `hasDerivAt_meanOcc` establish the two classical
derivative identities -- the mean occupancy is `d ln Z / d ln x`, and its derivative is the
occupancy *variance* -- in Lean, from which `varOcc_nonneg` and `meanOcc_nondecreasing_deriv` give
that occupancy is nondecreasing in ligand activity with no assumption on the site energies.
`hill_le_valence` bounds the Hill slope by the valence, and `hill_eq_valence_iff` and
`hill_allOrNone` identify the equality case exactly: the maximal slope is attained only by the
all-or-none polynomial, so a fitted Hill coefficient equal to the valence is a statement that no
partially bound state is populated.  `hillInd_le_one` is the complementary bound: independent
sites can never give a slope above one, by Cauchy-Schwarz, with equality exactly for identical
sites (`hillInd_identical`).  A measured slope strictly between one and the valence therefore
measures the *deviation from independence*, and nothing else.  `binding_polynomial_laws`
(`PartSeventyFour.lean`) bundles the statements.

## 6iii. The exact critical point of a chain-length Flory-Huggins model (Part LXXV)

`FloryHuggins.lean` computes, rather than assumes, the demixing point of the standard mean-field
free energy of a chain of `N` segments.  `curvature_identity` writes the second derivative in a
form whose minimum over the composition interval is explicit, giving `curvature_min` and the
critical composition `phiC N = 1 / (1 + sqrt N)`; `chiC N = (1 + sqrt N)^2 / (2N)` is then the
exact critical coupling.  `fh_convexOn` and `no_demixing_below_chiC` prove there is no demixing at
or below it, and `fh_demixes_above_chiC` produces an explicit demixing witness above it, so the
threshold is sharp in both directions.  Three corollaries are the physically useful ones:
`chiC_strictAnti` (longer chains demix at weaker coupling), `chiC_gt_half` with
`chiC_tendsto_half` (the coupling can never fall below the polymer-solution limit `1/2`, and
approaches it), and `phiC_tendsto_zero` (the critical composition of a long chain is dilute).
`flory_huggins_laws` (`PartSeventyFive.lean`) bundles the statements.

## 6jjj. From sequence to phase diagram, and the blind spot it inherits (Part LXXVI)

`SequencePhase.lean` composes the previous two parts through the affine sequence-to-coupling law
`chiEff chi0 lam P = chi0 + lam P` that every published patterning-versus-condensation correlation
assumes.  `demixes_iff_gt_threshold` turns it into an exact, testable sequence threshold: the model
predicts a condensate precisely above `threshold N chi0 lam = (chiC N - chi0)/lam`.
`threshold_strictAnti` states the resulting length-sequence trade-off -- a longer region condenses
at lower blockiness, so chain length and patterning are interchangeable within the model.
`blocky_demixes_alternating_stable` shows the link has content: two length-four sequences of
identical composition sit on opposite sides of the threshold at one chain length and one
chemistry.  And `homometric_same_phase_diagram` shows what it inherits: for every kernel, every
sequence-to-coupling law and every chain length, the homometric pair of Part LXXIII has the same
free-energy density and the same verdict.  A measured difference between those two sequences
falsifies the entire model class rather than its fitted parameters.  `sequence_phase_laws`
(`PartSeventySix.lean`) bundles the statements.

## 6kkk. What a radius of gyration reports (Part LXXVII)

`ChainGeometry.lean` treats the most-quoted single number about a disordered region exactly.
`rg2_eq_pairSum` proves the readout identity `Rg^2 = (1/2n^2) sum_{i,j} |x_i - x_j|^2` in any real
inner-product space: the radius of gyration is one fixed linear functional of the squared-distance
matrix.  `rg2_congr_of_dist_eq` is the immediate corollary -- conformations with the same distance
matrix have the same size, so the map from structure to `Rg` is many-to-one before any
experimental error enters.  `rg2_union` is the two-module parallel-axis law, the realistic case of
a folded domain with a disordered partner: `n Rg^2 = n_A Rg_A^2 + n_B Rg_B^2 + (n_A n_B/n) d^2`.
It has one useful direction and one cautionary one.  `inter_module_dist_sq_le` extracts a genuine
structural constraint from a single scalar -- a measured global `Rg` caps the module separation by
`d^2 <= (n^2/(n_A n_B)) Rg^2` -- while `rg2_tradeoff` exhibits two four-bead chains with the same
global `Rg^2 = 2`, one with compact modules held apart and one with a fourfold expanded module and
coincident centroids: expansion and separation are exchangeable at fixed size.  `rg2_le_rod` and
`rod_rg2` close the part with the connectivity ceiling `Rg^2 <= b^2 (n^2-1)/12` and its exact
attainment by the straight rod.  `chain_geometry_laws` (`PartSeventySeven.lean`) bundles the
statements.

## 6lll. Modification scanning: the experiment that breaks the blind spot (Part LXXVIII)

`Phosphorylation.lean` studies the probe biology actually applies to disordered regions.  Removing
a charge `z` at a single site is an exactly solvable perturbation of the model of Part LXXIII:
`pairEnergy_phos` gives `E(phos z k q) = E(q) - z h_k` with `h_k` the local field at the modified
site, and `phos_comm` shows modifications of distinct sites commute, so the model forbids any
dependence on the order in which multisite marks are written.  The central statement is
`phospho_epistasis`: the non-additivity of a *double* phosphorylation is exactly `z^2 w(d)`, `d`
the spacing of the two sites -- independent of the sequence, the composition and the position of
the pair along the chain (`phospho_epistasis_indep_of_sequence`).  Two consequences follow.
`phospho_no_three_body`: the third-order interaction of three sites vanishes identically, so a
measured three-body effect falsifies the pairwise class rather than its fitted kernel.  And
`kernel_of_epistasis` with `pairEnergy_eq_of_epistasis_eq`: a scan over spacings determines the
kernel `w` outright, hence the predicted energy of *every* sequence -- exactly the information
sequence comparison cannot supply, since by Part LXXIII the homometric pair is invisible under
every kernel.  `phospho_dissolves_condensate` closes with the phenotype: a polycationic patch under
a contact kernel lies above the threshold of Part LXXV at chain length one, and one phosphorylation
of physiological charge `z = 2` takes it below.  `phosphorylation_laws`
(`PartSeventyEight.lean`) bundles six statements.

## 6mmm. Salt: the high-salt limit and reentrant condensation (Part LXXIX)

`Screening.lean` puts the salt dependence of the pairwise charge model into one parameter, the
screening fugacity `x = exp(-kappa b)`, with kernel `screen x d = x^d/d`; `energy_eq_sum` then
writes the patterning energy as `sum_{d=1}^{N-1} (x^d/d) C(d)`, a power series in the screening
parameter whose coefficients are the autocorrelations of Part LXXIII.  Salt reweights the same
`N - 1` coordinates; it does not give a pairwise model any new sequence information.
`energy_high_salt` bounds the difference from `C(1) x` by `x^2 sum_{d>=2}|C(d)|/d`, so at high salt
the only surviving sequence information is the nearest-neighbour charge correlation, with an
explicit rate; `energy_neg_of_small` turns this into a sign statement.  The substantive result is
`reentrant_window`: for an explicit twelve-residue sequence the energy is negative at zero salt
(`energy_reent_one`), positive at intermediate salt (`energy_reent_four_fifths`) and negative again
at high salt (`energy_reent_half`), so the intermediate value theorem gives a zero on each side of
the favourable window.  Non-monotone salt dependence therefore needs no ion-specific or hydration
physics -- a sign pattern in the charge autocorrelation is enough.  `reentrant_demixing` states the
phase consequence against the threshold of Part LXXV: condensation at intermediate salt, stability
at both zero and high salt.  `screening_laws` (`PartSeventyNine.lean`) bundles four statements.

## 6nnn. Crosslinking mass spectrometry: the yield is a population (Part LXXX)

`Crosslink.lean` reads a crosslink experiment as it is: the reported quantity `freq E d r` is the
fraction of the ensemble whose site-site distance `d` is within the spacer reach `r`, i.e. one
value of the cumulative distribution of that distance (`freq_nonneg`, `freq_le_one`, `freq_mono`),
and it is an ensemble average, hence a linear restraint of the kind Parts LXIX-LXX count.  Three
consequences follow.  `freq_dirac` says a single structure predicts every yield to be `0` or `1`,
so `not_deterministic_of_fractional`: one sub-stoichiometric yield refutes every single-structure
model, with no geometry and no second crosslink.  `sum_freq_le_one` says the yields of mutually
exclusive crosslinks (`Exclusive`: no conformation satisfies two) sum to at most one -- a
falsifiable consistency test on the data itself.  `card_ge_of_exclusive` and
`card_ge_of_exclusive_model` say `k` observed mutually exclusive crosslinks force at least `k`
conformations in every model reproducing the data, upgrading "incompatible with one structure" to a
lower bound on ensemble size; `exclusive_of_separated` supplies the geometric input, and the
three-anchor tail (`triad_freq`, `triad_card_ge`) is an explicit instance forcing three states.
The positive statement is `mean_distance_from_series`: because yields are values of one cumulative
distribution, a series of crosslinkers with reaches `0, h, 2h, ...` recovers the mean site-site
distance by a layer-cake sum to within the spacing `h`.  One crosslinker reports one number; a
spacer-length series reports a distribution.  `crosslink_laws` (`PartEighty.lean`) bundles five
statements.

## 6ooo. Ensemble reweighting as convex duality (Part LXXXI)

`Dual.lean` analyses the objective actually minimised in ensemble refinement,
`dual q f d lam = log Z(lam) - <lam, d>`.  `logPartition_convex` proves the log-partition function
convex in the multipliers by a termwise weighted arithmetic-geometric mean inequality, and
`dual_convex` extends this to the objective: refinement is a convex problem with no spurious local
minima.  `dual_gap` is the substantive identity: if the tilt at `lam` matches the data then
`dual mu - dual lam = KL(tilt lam || tilt mu)` for every `mu`, so the duality gap is exactly the
information distance between the two reweighted ensembles -- a certificate in the units of the
objective.  Hence `dual_min_of_matches` (global optimality) and `tilt_eq_of_both_match`: multipliers
may be non-identifiable, the fitted ensemble is not.  Finally `dual_ge_of_feasible` bounds the
objective below by `log c` (`c` the smallest prior weight) whenever some ensemble on the pool
reproduces the data, while `dual_le_of_separated` and `dual_unbounded_of_separated` show that a
direction separating the data from every pool conformation by a margin `eps` gives
`dual(t u) <= -t eps`: the objective is unbounded below and the multipliers diverge.  Divergence in
a refinement run is therefore evidence of infeasibility against the pool, not a numerical
pathology to be regularised away.  `reweighting_duality_laws` (`PartEightyOne.lean`) bundles five
statements.

## 6ppp. Heterogeneous kinetics: what a single rate constant reports (Part LXXXII)

`Heterokinetics.lean` applies the ensemble view to kinetics.  In the slow-exchange limit a
disordered region reacting at a site it exposes only in some conformations decays as a mixture,
`surv w k t = sum_j w_j exp(-k_j t)`.  `surv_ge_exp_mean` proves `exp(-<k> t) <= surv t`:
heterogeneity mimics stability, with no protective mechanism present.  `apparentRate_zero` shows
the initial apparent rate is exactly the mean rate, and `apparentRate_antitone` -- proved by a
Chebyshev-type symmetrisation of a double sum rather than by differentiation, hence exactly and at
all times -- shows that the apparent rate then falls monotonically as the fragile conformations are
consumed first; `apparentRate_ge_min`, `apparentRate_le_max` bracket it by the extreme substate
rates, `surv_ge_slowest` gives the long-time tail to the slowest substate, and `surv_log_convex`
states the same fact as log-convexity of the decay curve.  The explicit instance `twoW`, `twoK`
(half at rate `1`, half at rate `1/100`) has initial apparent rate `101/200` and apparent rate below
`1/10` by `t = 10`: a fivefold change in the fitted rate constant of one sample, obtained by
measuring later.  A single fitted rate constant is therefore a property of the ensemble and of the
measurement window, not of the region; kinetics of a disordered region must be reported as a rate
distribution.  `heterogeneous_kinetics_laws` (`PartEightyTwo.lean`) bundles five statements.

## 6qqq. Ion mobility: the arrival-time distribution is the datum (Part LXXXIII)

`IonMobility.lean` applies the ensemble view to the one experiment that measures a size and a
width in the same trace.  Each conformer family contributes a peak at its own drift time `t j`,
proportional to its collision cross section, with instrumental width `sig j`, and the recorded
arrival-time distribution is the `w`-weighted mixture.  `width2_eq_instr_add_spread` is the law of
total variance in the instrument's own units: the measured squared width is the mean instrumental
variance plus the conformational spread of the drift times, exactly.  Hence
`spread_eq_width2_sub_instr` -- the spread is identified, but only as the *excess* width over the
instrument function -- and `width2_ge_instr`.  A model committed to a single conformation predicts
the instrumental width and nothing more (`width2_single`), so any excess width whatsoever refutes
it (`excess_width_refutes_single`); this is the ion-mobility analogue of a sub-stoichiometric
crosslink yield.  The centroid, by contrast, selects nothing: `min_le_meanTime` and
`meanTime_le_max` bracket it by the conformer drift times, and `mixture_hits_intermediate` shows
that every intermediate arrival time is reproduced exactly by a compact/extended two-state mixture
with both weights strictly between `0` and `1`.  `moment_ambiguity` goes further: two explicitly
different ensembles on drift times `0,1,2,3,4` -- a symmetric two-state mixture at `1` and `3`, and
a three-state mixture at `0`, `2` and `4` -- share both centroid and spread, so even mean *and*
width together do not determine the ensemble.  Finally the price of resolution is exact:
`resolved_iff` shows that peaks of width `t/R` are separated at criterion `k` precisely when
`k(t_1+t_2) <= R|t_1-t_2|`, so by `resolved_two_percent` a two-percent cross-section difference
needs resolving power `R >= 101`; and `card_ge_of_distinct_peaks` turns resolved peaks into a lower
bound on model complexity, `k` peaks forcing at least `k` conformers.  An ion-mobility measurement
of a disordered region is a distribution: a model of the region must predict a mean *and* an excess
width, not the single cross section a single structure has.  `ion_mobility_laws`
(`PartEightyThree.lean`) bundles five statements.

## 6rrr. The moment problem: what a finite list of averages can fix (Part LXXXIV)

`MomentProblem.lean` asks what the whole apparatus presupposes: given `k` measured moments of a
conformational observable, what is determined?  `moment_indeterminacy` answers, for every `k`, with
an explicit pair.  Take the `k+2` conformations with observable values `0,1,...,k+1` and weight the
even-numbered ones by `C(k+1,i)/2^k` and the odd-numbered ones likewise; both are probability
vectors (`wEven_sum`, `wOdd_sum`), they share no conformation at all (`supports_disjoint`), and
every moment of order `p <= k` agrees exactly (`moments_agree`), because the alternating binomial
sum `sum_i (-1)^i C(k+1,i) i^p` vanishes for `p < k+1` -- the `(k+1)`-st forward difference of a
polynomial of degree `p` (`alt_choose_pow_sum`).  `two_state_instance` is the smallest case: half a
population at `0` and half at `2`, against all of it at `1`, same mean, no shared structure,
distinguished only at second order.  So no finite list of averages certifies an ensemble: a model
disjoint from the truth reproduces the data exactly as well.  What repairs this is the support.
`weights_eq_of_moments_eq` proves by Lagrange interpolation that on a *fixed* set of `k+1` distinct
conformations the moments of orders `0,...,k` determine the populations uniquely -- structural prior
knowledge is not a convenience but the ingredient that makes a fit an answer, which is why the
support assumed must be part of the report.  Finally `markov_bound` and `chebyshev_bound` isolate
what a moment certifies with no prior at all: at most `<x>/a` of the ensemble has `x >= a`, and at
most `Var/a^2` of it deviates from the mean by `a`.  Those population bounds are the surviving,
falsifiable content of an average, and `markov_sharp` shows the first of them is attained by an
explicit two-conformation ensemble, so nothing sharper follows from a mean.
`moment_problem_laws` (`PartEightyFour.lean`) bundles five statements.

## 6sss. The capacity threshold as a testable design rule (Part LXXXV)

Every capacity statement before this part is one-sided: too few components and the model is wrong.
That rules architectures out; it does not say what to build, and it makes no number a measurement
could contradict.  `CapacityExact.lean` closes both gaps.  Fix a target populating `m` states with
measured populations `w_0 >= ... >= w_{m-1}` -- the form in which populations are reported -- and
write `tail P k = w_k + ... + w_{m-1}`.  `ell1_ge_two_tail` proves that every model with at most `k`
components, whatever its parameters and however trained, is at population-space `l1` distance at
least `2 tail P k` from the target; `ell1_trunc_le_two_tail` proves the explicit
keep-the-top-`k`-and-renormalise model attains exactly that, so `minErr P k = 2 tail P k` is not a
bound but *the* best attainable error at capacity `k` (`minErr_eq`).  Two-sided, and computable from
measured populations before any model is fitted.

From the equality the shape of the fit-quality curve follows term by term.  `minErr_step`: adding
the `(k+1)`-st component improves the attainable error by exactly `2 w_k`.  `minErr_eq_zero_iff` and
`minErr_pos`: the attainable error is strictly positive for every `k < m` and exactly zero from
`k = m` on.  So the theory predicts a kink at the measured state count and a plateau after it -- not
the smooth diminishing returns that generic "more capacity helps" intuition predicts, which is what
makes the curve a discriminating measurement rather than a consistency check.  `optimalK` turns the
law around into a design rule: the least component count achieving a target accuracy, sufficient by
`optimalK_spec`, minimal by `optimalK_min`, never above `m` by `optimalK_le`.

And the rule bites harder the broader the ensemble: on a target with `m` equally populated states
the attainable error at capacity `k` is exactly `2(m-k)/m` (`minErr_uniform`), so for any fixed
component count `K` there are targets on which the best a `K`-component model can do is arbitrarily
close to the maximal error `2` (`fixed_capacity_degrades`).  A fixed architecture is not merely
suboptimal on broad ensembles; its accuracy tends to the trivial, at a rate the law states.

Three theorems make the rule usable on real measurements rather than on idealised labels.
`stateLevel_floor`: state assignment is a push-forward and a push-forward cannot increase component
count, so the floor still applies after structures have been assigned to experimentally reported
states.  `minErr_perturb`: profiles differing by `eta` in `l1` have floors differing by at most
`2 eta`, so error bars on the reported populations propagate without amplification.
`missed_states_of_under_capacity`: an under-capacity model assigns population zero to an
identifiable set of conformations that really carries at least `tail P k` of the population -- a
named subensemble the model calls unoccupied and the experiment finds occupied, which is what an
experimenter actually measures.

`Falsification.lean` then encodes the test itself, and its point is to keep two halves apart.
`baseline_must_fail` (and `baseline_must_fail_robust`, which absorbs the error bars) proves that on
an admissible record no model with the practitioner's fixed component count can reach the
pre-registered tolerance: the predicted failure of an under-capacity baseline is a theorem, so an
experiment reporting otherwise has an error elsewhere -- in the state count, the populations or the
read-out -- and the protocol says where to look.  The other half is open, and provably so:
`at_capacity_not_sufficient` exhibits an `m`-component model that is maximally wrong, and
`confirms_refutable` (with `Panel.refutable` at panel level) exhibits outcomes consistent with
everything proved here on which the pre-registered criterion is false.  All the empirical content of
the test sits in whether the threshold-respecting model actually fits -- exactly the statement
`confirms_iff_threshold_fits` makes formal.  The worked records `panelA`, `panelB`, `panelC` show
the arithmetic on stipulated populations, with every numerical claim checked by computation, and
`illustrativePanel_selective` records that the rule fires on two of them and stays silent on the
third: a rule that fired everywhere would carry no information.  `PREREGISTRATION.md` is the
protocol these definitions encode -- systems, independent source of the state count, baseline,
metric, analysis and the outcomes that would refute the prediction -- and it is a protocol only: no
data has been analysed anywhere in this development.
`capacity_threshold_laws` and `prereg_test_laws` (`PartEightyFive.lean`) bundle the two groups.

## 6ttt. A cryo-EM map is an occupancy, not a structure (Part LXXXVI)

Single-particle reconstruction averages over the particles it was built from, so the map of an atom
is its occupancy distribution over voxels.  `CryoEM.lean` proves what that object determines.  It is
a probability distribution over the grid (`dens_nonneg`, `sum_dens_eq_one`, `dens_le_one`), so its
units are populations: a low value is a low population, not a blurred coordinate.  Peak height is
then a conformation count -- if no voxel exceeds `p` the atom occupies at least `1/p` distinct
voxels (`card_support_ge_inv_peak`), and at most `1/t` voxels per atom survive a contour level `t`
(`card_above_le_inv_threshold`).  The disappearance of disordered regions follows as a theorem
rather than an artefact: an atom spread uniformly over `m` voxels with `1/m < t` has no voxel above
the level (`uniform_spread_invisible`) while the sub-threshold voxels carry the whole population
(`invisible_mass`).  On the positive side every single-atom average is an exact linear read-out of
the map (`expect_from_map`), so mean position and positional variance are recoverable and two
ensembles with the same map agree on all such averages (`expect_eq_of_dens_eq`).  Nothing joint is:
`map_blind_to_correlation` gives two two-residue ensembles with identical maps for both residues,
one with the residues always in contact and one never.  Classification does not repair this by
itself, since the class maps of *any* partition sum back to the consensus map (`sum_classDens`), and
a `K`-class reconstruction stays a `K`-point model -- with `n > K` distinct atom positions realised,
every one-structure-per-class assignment leaves a strictly positive mean squared error
(`resid_pos_of_injective`).  `cryoem_occupancy_laws` and `cryoem_blindness_laws`
(`PartEightySix.lean`) bundle the two halves.

## 6uuu. How much of the ensemble has the sampling seen? (Part LXXXVII)

An ensemble model is built from a finite sample, and the population the sample never visited is
invisible to every diagnostic computed from the sample.  `Coverage.lean` works on the space
`Fin N -> Fin m` of `N` independent draws from an ensemble of `m` states.  The sample weights are a
probability distribution (`sum_sw_eq_one`), the probability that a state is never drawn is exactly
`(1 - w x)^N` (`prob_unseen`), and hence the expected unseen population is exactly the missing mass
`Sum_x w x (1 - w x)^N` (`expected_unseenMass`) -- an equality, not a bound.  It is strictly
positive at every sample size as soon as more than one state is populated (`missingMass_pos`): no
finite sample certifies that it has seen a disordered ensemble.  On a uniform ensemble the missing
mass is `(1 - 1/m)^N`, bounded below by `1 - N/m` (`missingMass_uniform`,
`missingMass_uniform_ge_one_sub`), so leaving at most `eps` unseen costs at least `(1 - eps)m` draws
(`sample_size_needed`) -- linear in the state count, which is itself exponential in chain length.
The positive half is `good_turing`: the expected number of states drawn exactly once in a sample of
size `N + 1` is `(N + 1)` times the missing mass at size `N`, so the singleton fraction of a sample
is an unbiased estimate of the population that sample is missing, computable without knowing `m` or
`w`.  `coverage_laws` (`PartEightySeven.lean`) bundles the six statements.

## 6vvv. Conditioning: what a regularised ensemble fit reports (Part LXXXVIII)

Every practical ensemble fit is regularised, and `Conditioning.lean` computes exactly what that
returns.  Mode by mode, with sensitivity `s`, datum `d = s c + n` and penalty `lam`, the
reconstruction `s d/(s^2 + lam)` equals `filt * c + gain * n` with `filt = s^2/(s^2 + lam)` and
`gain = s/(s^2 + lam)` (`recon_eq_filt_add_noise`).  The filter factor is always below one and is
monotone in sensitivity and in the penalty (`filt_lt_one`, `filt_mono_sens`, `filt_anti_pen`), so
regularisation is a systematic shrinkage towards the prior rather than a neutral technicality.  The
penalty caps the noise gain at `1/(2 sqrt lam)` uniformly in `s`, and the cap is attained at
`s = sqrt lam` (`gain_le_inv_two_sqrt`, `gain_eq_at_sqrt`); without a penalty the gain is `1/s` and
exceeds every bound (`gain_unbounded_of_no_penalty`).  The exchange rate is exact:
`s (1 - filt) = lam * gain` (`stability_bias_identity`).  The resolution boundary sits at
`s^2 = lam` (`filt_ge_half_iff`), below which the reported amplitude is the prior's
(`prior_dominates_of_insensitive`, `recon_le_of_insensitive`), and the total error is
`lam |c|/(s^2 + lam) + |n|/(2 sqrt lam)` (`error_le`).  Across a spectrum, `resolvedModes` -- the
modes with `s^2 >= lam` -- is the set of features the data control; it only shrinks as the penalty
rises, and outside it the fit reports the prior whatever the data say.  `conditioning_laws` and
`resolved_modes_laws` (`PartEightyEight.lean`) bundle the two groups.

## 6www. From a threshold to a study design: power, visibility and the four numbers (Part LXXXIX)

A threshold is not a test.  Part LXXXV made the capacity statement exact -- the best `l1` error a
`k`-component model can reach on a target with measured populations is exactly twice the population
outside its top `k` states -- but an exact law still does not say how many molecules to watch, with
what probe, or through how rich a read-out.  Part LXXXIX supplies those, each as a computation on
the independently measured populations and each tied by a theorem to the behaviour of real
ensembles.

*Power.*  The set of conformations an under-capacity model omits carries true population `tau`, and
the probability that `n` independent observations all avoid it is exactly `(1 - tau)^n`
(`Power.missProb_eq`).  A single observation inside it makes the model's likelihood exactly `0`
(`Power.likelihood_zero_of_hit`): the outcome is a refutation, not a worse score.  Hence the
pre-registered sample size `samplesFor alpha tau = ceil(log(1/alpha)/tau)` (`samplesFor_spec`), and
the honesty clause that with `n` draws the chance of seeing nothing is still at least `1 - n tau`
(`missProb_ge_one_sub`), so a short run is not evidence for the baseline.

*Visibility.*  An observable with values in `[a, b]` differs between two ensembles by at most
`(b-a)/2` times their population `l1` distance (`Visible.expect_gap_le_range`), so a model sitting
on the capacity floor moves it by at most `(b-a) tau`.  That is the mechanism behind an
under-capacity model with an excellent chi-squared: the probe cannot see the failure.  The probe
that can has dynamic range at least `sigma/tau` (`Visible.contrast_requirement`), and the extremal
one is the indicator of the omitted states, range `1`, which reports the whole discrepancy
(`Visible.optimal_reporter`).  Separately, unless the suite carries at least `m-1` independent
observables, some scored population is not a consequence of the data at all
(`Visible.observables_needed`).

*The instrument.*  `Instrument.Study` is the frozen record and `Instrument.report` computes from it
the required component count, the required number of observations, the required number of
independent observables and the required probe contrast, in exact rational arithmetic, each with a
soundness theorem (`requiredComponents_min`, `requiredComponents_sound`, `requiredSamples_sound`,
`requiredObservables_sound`, `requiredContrast_sound`).  On the stipulated illustrative record it
returns five components, fourteen observations, four observables and contrast `1/10`.

*The panel.*  Freeze `n` records at level `alpha/n`, take that level's sample size on each, and the
probability that every system refutes its under-capacity baseline is at least `1 - alpha`
(`panel_design_laws`) -- multiplicity paid in planned observations rather than in a post-hoc
correction.

*Where to spend.*  `budget_laws` prices the two routes on the same target: on `m` equally
populated states accuracy `eps` costs exactly `ceil(m(1 - eps/2))` components
(`Budget.optimalK_uniform`), payable by an explicit model with no data at all once the populated
states are known, while any support-honest learner needs `m(1 - eps)` samples
(`Learn.support_honest_needs_coverage`) of a state space exponential in the length of the region.
Buy capacity and state identification, not pool size.

*And what it does not buy.*  `design_limits` records the two limits: being at capacity never
entails fitting well, and a read-out poorer than `m-1` independent observables cannot certify the
populations the test is scored against.  `capacity_power_laws`, `capacity_visibility_laws`,
`design_window`, `panel_design_laws` and `design_limits` (`PartEightyNine.lean`) bundle the groups.
No number in this part is a measurement, and no claim is made about any real system.

## 6xxx. What a discovery list means when the whole proteome is screened (Part XCI)

Part XC prices one system's test on one realistic reporter.  A claim about disordered regions in
general is not made that way: thousands of candidate regions are screened at once and the ones
that fire are reported, and that report — a *list* — has an error rate that no per-system
calculation controls.  Splitting the level by the number of candidates (Part LXXXIX.4) controls
the family-wise error, but at twenty thousand candidates it demands a per-candidate level of
`2.5·10⁻⁶`.  Part XCI supplies the quantity a screen can afford instead, and prices it.

*The procedure.*  `IDR.FDR.bhR` is the Benjamini–Hochberg step-up index and `IDR.FDR.bhRej` the
discovery list it defines; `IDR.FDR.below_bhR` proves the list has exactly `bhR` entries, and
`IDR.FDR.bh_dominates_bonferroni` that it always contains the Bonferroni list, so the extra power
costs nothing in what is reported.

*The theorem.*  `IDR.FDR.bh_fdr_control`: in a finite product experiment with one independent
coordinate per candidate and superuniform null p-values, the expected false discovery proportion
of the list at level `q` is at most `|H₀|·q/N`, hence at most `q` — with nothing assumed about the
non-null candidates.  The proof is the classical leave-one-out argument made explicit:
`IDR.FDR.bhR_eq_iff_pZero` shows that on the event that candidate `i` lies below the `k`-th
threshold, the stopping index is unchanged by setting `p i` to zero, which turns `{BH stops at k}`
into an event about the other candidates and lets `IDR.FDR.EE_factor` factorise the term.

*The p-values.*  A counting reporter supplies exact moments and nothing else that is free of
modelling assumptions.  `IDR.Screen.chebP_superuniform` shows the two-moment statistic is a valid
p-value at every level; `IDR.Chernoff.sum_recProb_exp` computes the read-out law's moment
generating function exactly, and `IDR.Chernoff.chernoff_tail` turns it into
`P(count ≥ n·q + a) ≤ exp(-a²/4n)` at every finite `n`, whence `IDR.Chernoff.expP_superuniform`.
Both feed the same screen (`IDR.Screen.screen_fdr_control`,
`IDR.Chernoff.screen_fdr_control_exp`), because BH cares only about superuniformity.

*The cost.*  `IDR.Screen.screen_budget_quadratic`: the two-moment screen costs at least
`N²/(q·Δ²)` molecules, quadratic in the number of candidates.
`IDR.Chernoff.screen_exp_cheaper`: the exponential-tail screen costs
`⌈(16·log(N/q) + 1/α)/Δ²⌉` per candidate, logarithmic in the size of the screen.  On the worked
record — twenty thousand regions, `q = 0.05`, `Δ = 0.1` — that is `4·10⁷` molecules per candidate
against at most `22 800` (`IDR.PartXCI.worked_cheb_cost`, `IDR.PartXCI.worked_exp_cost`,
`IDR.PartXCI.worked_cost_ratio`): a factor of more than a thousand decided by which inequality the
analysis is willing to prove, not by the physics.  `IDR.PartXCI.screen_laws` is the single
statement.

## 6yyy. Sequential counting: a run that may be watched (Part XCII)

Every sample size in Parts XC and XCI is fixed before the run and the data scored once, at the
end.  Real runs are watched.  Under the fixed-`n` theory a data-dependent stopping time voids the
error bound, and not marginally: two looks, each at level `1/2`, give a combined null error of
`3/4` (`IDR.Seq.peeking_inflates_error`).

*What is watched.*  The wealth of a bet on the alternative read-out rate — the running likelihood
ratio `IDR.Seq.wealth`, equal to `probL q₁ / probL q₀` on the molecules scored so far
(`IDR.Seq.wealth_eq_ratio`).

*The level.*  `IDR.Seq.ville` proves Ville's inequality from scratch, by induction over the
read-out word: under the null the wealth reaches `c` at some point of a run of `n` reads with
probability at most `1/c`.  Hence `IDR.Seq.anytime_valid` and
`IDR.Seq.anytime_valid_all_horizons`: stopping when the likelihood ratio exceeds `1/α` has
type-I error at most `α`, at every horizon and for every stopping rule.

*The power.*  The expected log wealth after `n` molecules is exactly `n·KL(q₁‖q₀)`
(`IDR.Seq.expected_logWealth`) with variance exactly `n·klVar` (`IDR.Seq.variance_logWealth`),
both strictly positive off the null (`IDR.Seq.kl2_pos`, `IDR.Seq.klVar_pos`).  Chebyshev on those
exact moments bounds the probability of failing to stop by `n·klVar/(n·KL − log c)²`
(`IDR.Seq.sequential_power`), and `IDR.Seq.powerSamples_spec` converts it into a molecule count
with power `1 − β` at level `α`.

*Every rule.*  `IDR.Seq.stopping_rule_valid`: a run stopped by an arbitrary rule — any function
of the reads seen so far — declares a refutation under the null with probability at most `α`, so
the guarantee covers the rule an experimenter actually uses, not only the crossing rule
(`IDR.PartXCII.capacity_test_under_any_stopping_rule`).

*An estimate.*  Reporting the candidate omitted populations not yet excluded gives a confidence
sequence: nested (`IDR.Seq.excluded_of_prefix`) and covering at every horizon
(`IDR.Seq.confSeq_coverage`, `IDR.PartXCII.sequential_population_estimate`).

*The cost.*  Pinsker on the reporter's two rates (`IDR.Seq.kl2_ge_two_sq`) bounds the horizon by
`⌈log(1/α)/(2Δ²)⌉` molecules with `Δ = τ·J` (`IDR.Seq.sequentialHorizon_le_of_contrast`): the
same contrast the fixed design pays for, with `log(1/α)` in place of `1/α`, and
`IDR.Seq.sequential_cheaper` is the explicit comparison.
`IDR.PartXCII.sequential_capacity_laws` states level, power, signal and evidence rate together
for the capacity test.

## 7. Assumptions, and what is not claimed

* **Sequential analysis does not repair calibration.** Part XCII frees the run length from being
  fixed in advance, under two hypotheses it does not remove: the reads are independent, and the
  null read rate `1 − sp` is known. A reporter miscalibrated by `τ·J` defeats the wealth test
  exactly as it defeats the counting test of Part XC, since the wealth is then a bet on an
  instrument artefact. Its power bound uses two moments only and is therefore conservative.

* **Independence across screened candidates.** The false discovery rate theorem of Part XCI is
  proved for a product experiment: one independent coordinate per candidate region. Shared
  reagents, shared calibration and batch effects couple candidates in a real screen, and under
  arbitrary dependence the Benjamini–Hochberg procedure needs the harmonic correction. That
  correction is now proved (Part XCIV, `RequestProject/DependentBH.lean`), together with the fact
  that its factor cannot be lowered (Part XCIV.2, `RequestProject/DependentBHSharp.lean`), so this
  item is closed. Superuniformity of the nulls still requires each candidate's baseline read rate
  to be known — the calibration floor of Part XC, which is per candidate and is not softened by
  multiplicity control.

* **Finite versus continuous.** Parts I–X are stated for finitely supported ensembles. Part XI
  proves the continuum versions of the load-bearing statements (error floor, metric choice,
  well-posedness) rather than assuming they transfer; the finite statements are used
  thereafter as the quantitative, resolution-limited form of the continuous ones.
* **Lattice models.** The exact excluded-volume results are for chains on a lattice with a
  finite set of bond vectors. This is the standard rigorous setting for self-avoidance; the
  numerical constants (`log 4`, `log 6`, `100`, `30`) are lattice-specific, but the
  structure — submultiplicativity, a strictly positive entropy per residue strictly below the
  free-chain value, trapping, unbounded memory — is proved for a general lattice.
* **Equilibrium.** Ensembles are equilibrium (Boltzmann/Gibbs) objects except where kinetics
  is explicitly modelled (`Dynamics.lean`, `Kinetics.lean`, `Relaxation.lean`, `Rouse.lean`,
  `Trajectory.lean`). Parts XVIII–XIX lift the equilibrium assumption itself, but only for
  *driven steady states* of a finite-state Markov kinetics: the three-state cycle is an explicit
  witness rather than a general theory, and genuinely non-stationary (ageing, transient) cellular
  behaviour is still outside the scope.
* **The integrator results are Gaussian.** Part XX is exact for the linear (harmonic) case, where
  the variance map of an additive-noise scheme is affine; it is the normal-mode statement, not a
  theorem about anharmonic force fields, and the constant-`dt`, single-mode setting is explicit
  in the statements.
* **Crowding and tethers are modelled, not derived.** The crowding results of Part XV are
  theorems about the standard excluded-volume (osmotic-work) reweighting, with the excluded
  volume taken as a given conformational observable; the tether results of Part XVI are exact
  statements about the ideal (non-self-avoiding) three-dimensional chain, whose contact
  probability is the cube of the one-dimensional return probability. Self-avoidance changes the
  exponent and is treated separately in Part XII.
* **Mean-field steps are labelled as such.** Flory theory (`Flory.lean`), Debye–Hückel
  screening (`Electrostatics.lean`), Flory–Huggins demixing (`Condensate.lean`) and the
  helix–coil model (`HelixCoil.lean`) are theorems *about those models*, not derivations of
  them from a microscopic Hamiltonian.
* **The local restraints are the standard forward models, not derivations.** Part XXVIII takes
  the Karplus relation as a quadratic in `cos θ` and assumes fast exchange on the coupling
  timescale, so that the measured coupling is the population average; and it takes hydrogen
  exchange in the Linderstrøm-Lang scheme, with a per-conformer probability of being
  exchange-competent and a sequence-dependent intrinsic rate supplied from outside. The theorems
  are statements about those forward models — which are the ones used to refine ensembles — and
  not about their derivation from a quantum-chemical or chemical-kinetic first principle.
* **The protocol results are schematic in the artefact, exact in its consequence.** Part XXIX
  does not compute a periodic image energy: it assumes only that the image term is nondecreasing
  in the chain dimension, which is what makes the compaction result independent of the force
  field, and correspondingly it gives a sign and a monotonicity rather than a magnitude. The
  cutoff results are for a plain spherical truncation, not for shifted, switched or reaction-field
  variants, and they bound the *neglected* energy rather than the induced population error. The
  correlated-sampling results assume a single exponential autocorrelation; Part XXI shows that a
  single relaxation time is itself an idealisation, so `(1+ρ)/(1-ρ)` should be read as the
  inefficiency of the slowest mode the observable couples to.
* **The binding accounting is conformational, not total.** Part XXX prices the *conformational*
  free energy of a disordered region on binding, in a fixed-conformer-library, single-site,
  ideal-dilution setting with the interaction energies supplied from outside. Translational and
  rotational entropy, desolvation, the standard-state concentration convention, and the partner's
  own conformational change are not part of the accounting; the statements are about the term the
  disorder contributes, which is the one a model of the region is responsible for.
* **The scaling results assume the leading correction.** Part XXXI takes the correction to
  scaling in the standard one-term form `A N^{2ν}(1 + B/N)` and the estimator in its two-point
  log–log form. The sign and size statements are theorems about that pair of choices, which are
  the ones in common use; they are not a derivation of the correction exponent, and a
  multi-length fit with a fitted correction amplitude is outside them.
* **The NMR dynamics results are statements about the standard forward models.** Part XXXII
  takes the correlation function of the motion to be a finite sum of exponentials, so that the
  spectral density is a finite mixture of Lorentzians, and takes CPMG relaxation dispersion in
  the fast-exchange (Luz–Meiboom) form with a two-site exchange process. It does not derive
  either from the Bloch–Redfield or Bloch–McConnell equations, and the statements are about what
  those forward models — the ones used to fit real data — can and cannot determine.
* **The single-molecule results assume clean photon counting.** Part XXXIII models a burst as `N`
  photons with binomial acceptor counts at the conformer's transfer efficiency. Background,
  crosstalk, detector dead time, gamma/beta correction, the burst-size distribution and dye
  photophysics are not modelled; the dynamic-averaging statement models fast interconversion as
  two independently sampled conformations per burst, which gives the `varConf/2` factor but not
  a full treatment of continuous exchange within a burst.
* **The association results are diffusion-limited.** Part XXXIV assumes Smoluchowski capture at a
  given radius and the Stokes–Einstein relation, both in their spherical, uncharged, homogeneous
  form; reaction-limited binding, long-range electrostatic steering, hydrodynamic interactions
  between the partners and the internal dynamics of the encounter complex are outside it. What
  is proved is that within that regime the rate is fixed by the ratio of two radii, which is
  what makes the fly-casting criterion sharp.
* **The titration results are about the standard fit.** Part XXXV takes the two-state
  linear-extrapolation model with a linear free energy in denaturant concentration and ideal
  baselines. It does not model denaturant binding, activity coefficients, or the sloping
  pre- and post-transition baselines that a real fit also carries; the statements are about
  what the fitted midpoint and `m`-value are, given that model.
* **The CD results are about the standard linear mixture.** Part XXXVI assumes the forward model
  used by every deconvolution program: the measured spectrum is a nonnegative, sum-to-one mixture
  of fixed reference spectra sampled at the measured wavelengths, with no wavelength-dependent
  scaling and no chain-length correction. The results are about the identifiability of that
  model; the reference bases themselves are inputs, and the concrete two-wavelength witness uses
  realistic but illustrative ellipticities.
* **The aggregation results split by regime.** Part XXXVII solves the unsaturated
  nucleation–elongation equation `M'' = κ²M` exactly, and says nothing about the plateau or about
  monomer depletion; Part LVI (`Depletion.lean`) supplies the saturated regime, solving
  `M' = κM(1 − M/m₀)` exactly, proving the early-time treatment is an upper bound of known sign,
  and redoing the lag-time laws there. What remains outside both is fibril fragmentation beyond
  its lumped contribution to `κ`, secondary nucleation as a separate channel, and any
  distribution over fibril lengths: both parts track the total fibril mass only.
* **The density results are Gaussian.** Part XXXVIII models a smeared atom by a single occupancy
  and a single isotropic displacement parameter; Part LVII (`Anisotropy.lean`) removes both
  restrictions, proving that no inflated isotropic width reproduces a two-conformer density, that
  an anisotropic map determines its displacement tensor while the isotropic equivalent does not,
  and that the anisotropy discarded at fixed `B` is unbounded. Both parts keep the Gaussian
  real-space form and the two-conformer case as the witness for multi-conformer disorder;
  bulk-solvent modelling, map sharpening, resolution truncation and the refinement procedure
  itself remain outside. The results are about what the reported parameters are identifiable
  from, not about map refinement practice.
* **No empirical claim.** Nothing here asserts what any particular protein does. The results
  are statements about what any model of a disordered region must do to be correct on targets
  that disordered regions demonstrably realise, and the concrete targets used as witnesses
  (two-state ensembles, uniform libraries, lattice chains, three-fold rotamers) are exhibited
  explicitly.
* **The molecular model of Part XXIV is class-I with implicit solvent.** The Hamiltonian is
  additive and non-polarisable; the solvent enters through Generalized Born and a surface-area
  term. Electronic polarisability, explicit-solvent many-body potentials of mean force,
  protonation equilibria (pH-dependent charge regulation), rigid-constraint (Fixman) corrections
  and broken ergodicity (for example cis/trans proline isomerisation) are not part of that
  model. What Part XXIV establishes is that the standard model is well posed: the ensemble
  exists, has a density, is exactly E(3) invariant, and its log-gradient is the force. Part XXV
  then prices four of those residual idealisations — pairwise additivity, fixed charges,
  finite-time sampling and rigid constraints. Two of the items that stood outside the development
  have since been brought inside: Part LI (`Polarisability.lean`) proves the point-polarisable
  repair well posed and computes the exact three-body residue a pairwise fit must miss, and
  Part LII (`NuclearQuantum.lean`) prices the classical treatment of the nuclei, including the
  equilibrium isotope effect a classical force field assigns the value zero. What still remains
  outside is quantum-mechanical bond making and breaking. Explicit-solvent structure is treated in
  Part LX (`Pmf.lean`), which constructs the potential of mean force for a finite solvent exactly,
  proves it reproduces the solute marginal identically, and shows on a minimal hydration model
  that it is neither pairwise nor temperature independent even when the underlying interaction is
  exactly pairwise; that model is a single two-state solvent molecule, chosen to be the smallest
  system in which the obstruction can be computed in closed form, and it is not a theory of water.
  The polarisability results are for the point-dipole model with damping, not for
  Drude oscillators or fluctuating-charge schemes, and the nuclear-quantum results are for a
  single harmonic mode, which is the mode-by-mode form of the standard estimate and not a
  path-integral treatment of the full chain.
* **Evolution is treated at the level of a descriptor, not of function.** The reversal
  invariance of Part XLI is proved for pairwise, separation-dependent sequence descriptors
  (the mean-field Debye--Hueckel energy, the charge decoration, the net charge); the full
  physics of a real chain is directional, and the file exhibits a directional descriptor that
  reversal changes, precisely so that the scope of the invariance is explicit. Nothing is
  claimed about which descriptors evolution actually conserves in any particular protein.
* **The co-translational contraction factor is a hypothesis.** Part XLII takes the per-step
  contraction towards the current equilibrium as given, which is the standard spectral-gap or
  Dobrushin input; it is not derived from a microscopic dynamics here. The statements are
  therefore conditional on that input, and an explicit chain is exhibited that satisfies every
  hypothesis with equality, so the bound is attained and not merely valid. The vectorial-context
  result is a two-state caricature: exact within the model, illustrative of the mechanism.
* **The rheology results are about model classes, not about specific materials.** Part XLIII
  proves what a *finite* Maxwell spectrum and *uncorrelated* steps can and cannot produce, and
  contrasts them with power-law forms; whether a given condensate is a power-law material is an
  experimental question, and no such claim is made here.
* **The fitting results are about exact fits and linear observables.** Part XLIV treats
  observables that are ensemble averages, which is the case in ensemble reweighting, and exact
  agreement rather than agreement within error bars; the practical statement -- that the data
  leave a large set of ensembles admissible -- is the same, but the theorems are proved for the
  exact-fit version.
* **The benchmark results come in two versions.** Part XLV assumes every annotated residue really
  is disordered and only the coverage is partial; Part LIX (`LabelNoise.lean`) drops that
  assumption entirely, allowing errors in both directions and assuming nothing about their
  direction, and replaces it with a bracket by the annotation error count, a certificate at a
  margin of twice that count, and explicit inversions showing the certificate is sharp. Both
  parts count residues with equal weight: class imbalance, per-protein aggregation, and the
  particular figures of merit used by real assessments (balanced accuracy, Matthews correlation)
  are not modelled, and the ranking-inversion results are explicit examples, so they show the
  failure is possible, not that it is typical.
* **The compensation results are about the ordinary least-squares van 't Hoff fit.** Part XLVI
  proves an identity for unweighted linear regression of log-constants on inverse temperature;
  it does not assert that no physical compensation mechanism exists, only that the linear
  correlation such a fit produces, and its slope, are already forced by the design. The
  heat-capacity case is treated separately and exactly in Part LV (`HeatCapacity.lean`): with a
  constant `ΔCp` the two-point van 't Hoff enthalpy is the true enthalpy at an interior log-mean
  temperature, hence at neither endpoint, and the curvature of the plot is computed in closed
  form. Weighted fits, calorimetric (model-free) enthalpies, and a temperature-dependent `ΔCp`
  have their own error structure and are not covered.
* **The scattering/FRET comparison is about the labelled pair.** Part XLVII treats the
  donor--acceptor distance distribution of a finite weighted ensemble and the ideal transfer
  efficiency `R₀⁶/(R₀⁶+r⁶)`, with the orientation factor set to `2/3`. Part LIII
  (`Orientation.lean`) removes that substitution: it proves the exact range `0 ≤ κ² ≤ 4`, exhibits
  a finite orientational model whose mean `κ²` is exactly `2/3` and for which the mean efficiency
  is nonetheless `1/5` where the substituted formula gives `2/5`, and bounds the residual
  ambiguity at a factor `6` in `r⁶`. That model is a discrete six-direction caricature of the dye
  orientational distribution, chosen so that the mean is exactly the textbook value; dye linker
  dynamics, photophysics, and the reconstruction of a radius of gyration from a full scattering
  curve remain outside. What is proved in Part XLVII is the moment mismatch itself, which no
  improvement in either measurement removes.
* **The orientational results are about the ideal alignment model.** Part LVIII treats the
  standard second-rank form `rdc A u = (3uᵀAu − tr A)/2` with a symmetric traceless alignment
  tensor and exact ensemble averaging. It does not model the physical origin of the alignment
  (steric or electrostatic interaction with the medium), the possibility that the alignment tensor
  itself depends on the conformer -- which for a disordered chain it does -- dynamic frequency
  shifts, or the experimental separation of the residual coupling from the scalar one. What is
  proved is what the ideal observable determines: the second moment of the orientational
  distribution of each bond, and nothing beyond it.
* **The kinetic results are about a one-dimensional hopping chain.** Parts XLVIII, XLIX and L model
  motion along a chosen coordinate as a nearest-neighbour Markov jump process obeying detailed
  balance, and define the mean first-passage time and the committor as the solutions of the
  corresponding first-step-analysis systems — the files prove those systems are uniquely
  solvable, but the reduction of the full dynamics to such a chain is a modelling assumption (its
  general failure is the subject of Part IV.6, `Kinetics.lean`). Part LIV (`TransitionPath.lean`)
  adds the transition-path time within the same class, by constructing the committor `h`-transform
  and proving the conditioned process is again detailed-balanced, so the Part XLVIII machinery
  applies to it verbatim; it exhibits a model whose reaction is arbitrarily slow while its
  crossings are arbitrarily fast. Non-Markovian memory and genuinely multidimensional pathways are
  still outside what is proved here; what is proved is that even granting the reduction, the
  equilibrium profile leaves the rate free by an arbitrary factor, the committor free step by
  step, and the transition-path time unrelated to the rate.
* **The Markov state model results are about a finite chain and a fixed clustering.** Part LXII
  takes the microscopic kinetics to be a finite-state Markov chain with a known stationary
  distribution and the clustering to be a given map onto macrostates; the estimated matrix is the
  exact stationary-weighted lumping, i.e. the infinite-sampling limit of transition counting.
  Statistical error in the counts, the choice of clustering, and the continuous-time case are
  outside what is proved. What is proved is that even with perfect statistics the estimate is
  exactly right about the coarse thermodynamics, one-sided about the timescales, and dependent on
  the lag.
* **The nonequilibrium work results assume microscopic reversibility.** Part LXIII takes Crooks'
  relation as a hypothesis on a finite set of trajectories rather than deriving it from an
  underlying dynamics, and proves it is satisfiable so that nothing is vacuous. The Jarzynski
  equality, the relative-entropy form of the dissipation, the second law and the histogram
  crossing are all consequences of that hypothesis; the sampling statements are exact statements
  about which trajectories carry the exponential average, not a distributional theory of the
  estimator's error.
* **The inverse coarse-graining results are discrete and finite.** Part LXIV works on a finite
  configuration space with finitely many structural features and a linear (pairwise-in-features)
  energy, which is the exact discrete analogue of a radial distribution function fit; the
  continuum Henderson theorem, the existence of a fitting parameter vector for a prescribed
  target, and the convergence of iterative Boltzmann inversion are not treated. What is proved is
  uniqueness, blindness beyond the fitted statistics, and state-point dependence.
* **The replica-exchange results are for a finite state space and two replicas.** Part LXV models
  one exchanging pair -- which is what an exchange attempt is -- with the swap treated as the
  Metropolis move on the extended ensemble; the within-temperature dynamics enters only through
  its stationarity and, for the conservation law, through which moves it allows. Mixing rates of
  the extended chain, optimal temperature placement and the multi-replica round-trip time are not
  treated. The acceptance bound assumes energy histograms separated by a gap; that separation is
  a hypothesis about the system, satisfied in an explicit instance, not derived here from an
  extensivity argument. Part LXVIII lifts the two-replica restriction for the statements that do
  not depend on it -- exactness, the acceptance formula and the conserved count -- for a ladder of
  `K` replicas with one exchanging pair per sweep; the scheduling of which pairs attempt exchanges,
  and the round-trip statistics that scheduling controls, are not modelled.
* **The umbrella-sampling results are exact statements about window data, not about estimators.**
  Part LXVI treats the population that each window's data determines, on a finite conformation
  space with exactly known biases. The identifiability theorem assumes a chain of overlapping
  supports covering the space; the unidentifiability theorem assumes only that no window straddles
  the split. Estimator variance, the WHAM/MBAR self-consistency iteration and its convergence, and
  correlated frames within a window are treated elsewhere (`CorrelatedSampling.lean`,
  `Reweighting.lean`) or not at all; the `1 − N m` bound is a statement about independent frames.
* **The sequence-degeneracy result is about a class of models, not about nature.** Part LXVII
  proves that any model whose charge-sequence dependence is pairwise and depends on a pair only
  through its separation along the chain is blind to the difference between two explicit charge
  patterns. It does not claim the two patterns behave identically in reality -- that is precisely
  the point, since the three-body correlation separates them -- nor that every published sequence
  parameter is of this form; what is proved is that sequence charge decoration and every
  separation-dependent pairwise energy are, and that the blindness is exact at every state point.

* **The restraint-counting law is a statement about linear restraints on a fixed library.**
  Parts LXIX-LXXI model an experimental restraint as a known linear functional of the population
  vector over a fixed finite conformational library -- which is what a back-calculated ensemble
  average is -- and the target as interior, every conformation carrying weight at least `d`. The
  counting bound `m - 1` is a statement about that linear inverse problem; it is not a claim that
  published ensembles are wrong, but a claim about what their data can and cannot determine. A
  smaller library, or a prior, closes the gap -- and the theorems say precisely that the closure
  comes from the library or the prior rather than from the experiment.

* **Identifiability is proved for linear functionals.** Part LXX characterises exactly the
  *linear* functionals of the ensemble that the data determine. Nonlinear summaries (a mode of a
  distribution, a ratio of populations, an exponent extracted by fitting) are not covered by the
  characterisation; the necessary direction still applies to them wherever they can be moved
  along a null direction, but the clean iff is for the linear case.

* **The precision floor is a worst-case bound over conformation pairs.** Part LXXI exhibits, for
  any two distinct conformations, an ensemble consistent to tolerance `eps` at population distance
  `min (2 d) (eps / G)`. It is a guarantee that resolution *cannot* be better than that, not a
  claim that every reported ensemble is that badly determined in every direction; the matching
  ceiling shows the `eps`-scaling is right.

* **The design theorem is about linear reports and exact matching.** Part LXXII shows that `r`
  independent reported functionals cost exactly `r` experiments. "Cost" here means identifiability
  in the sense of Part LXX -- exact matching of the measured averages -- so the resolution floor of
  Part LXXI still applies to the numbers so obtained, and the optimality statement is about how
  many observables must be measured, not about how accurately.

* **The charge-patterning blindness is a statement about pairwise kernels.** Part LXXIII shows
  that any model of the form `sum_{i<j} w(j-i) q_i q_j` reads the sequence through its
  autocorrelation only. Models with explicit many-body sequence terms, or with a conformational
  ensemble, are not covered by the blindness -- indeed Part LXXIII's own three-body descriptor
  separates the homometric pair.

* **The Flory-Huggins critical point is mean-field.** Part LXXV computes the exact critical point
  of the standard chain-length free energy `fh N chi`; it is a statement about that free energy,
  not about the true critical behaviour of a polymer solution, and no claim is made about critical
  exponents. Part LXXVI inherits this, and additionally assumes the affine sequence-to-coupling
  law that published correlations use; its threshold is exact *within* that model.

* **The radius-of-gyration results are geometric, not ensemble-averaged.** Part LXXVII states
  identities and bounds for one conformation of `n` beads. The blindness statement
  (`rg2_congr_of_dist_eq`) and the rod ceiling are therefore per-conformation facts; an ensemble
  average of `Rg^2` inherits them by linearity, but the relation between `<Rg^2>` and other
  ensemble observables is the subject of Part XLVII, not of this part.

* **The phosphorylation results hold inside the pairwise charge model.** Part LXXVIII's exact
  epistasis law `z^2 w(d)` and its vanishing three-body term are predictions *of* that model
  class; they are stated so as to be falsifiable, and a measured three-body interaction would
  refute the class. The identifiability statement determines the kernel on `1 <= d < N` from
  idealised, noiseless epistasis measurements; the precision floor of Part LXXI applies to any
  actual measurement of them.

* **The screening model is a chain-distance Debye kernel.** Part LXXIX models added salt by the
  factor `x^d = exp(-kappa b d)` along the chain and computes the exact consequence. It contains no
  ion-specific effects, no charge condensation and no hydration; the reentrance it exhibits is
  therefore a *sufficient* mechanism for non-monotone salt dependence within the pairwise model,
  not a claim that observed reentrance has this cause.

* **The cryo-EM part models the map as an exact ensemble average on a finite voxel grid.** Part
  LXXXVI contains no image-formation model, no contrast transfer function, no noise and no
  alignment error; the point-spread of the microscope would only broaden the occupancies further,
  so the peak-height and contour-level bounds are the best case for a real reconstruction rather
  than an idealisation that flatters it. The blindness to correlations is a property of averaging
  over particles and is independent of resolution.

* **The coverage results assume independent draws.** Part LXXXVII models a conformational sample
  as `N` independent draws from the ensemble. A molecular-dynamics trajectory is correlated, so its
  effective sample size is smaller and the missing mass correspondingly larger; the bounds proved
  here are therefore optimistic for trajectory data; the effect of correlation on trajectory error
  bars is treated in Part XXIX.3 (`CorrelatedSampling.lean`).

* **The conditioning results are stated mode by mode.** Part LXXXVIII assumes the measurement has
  been diagonalised, so that each conformational mode has a single sensitivity; it does not derive
  that diagonalisation, and the penalty is the quadratic one. The statements are exact for the
  Tikhonov reconstruction and are not claimed for other regularisers.

* **Not proved.** The Flory exponent is not derived for genuine self-avoiding walks (only for
  the Flory free energy); the connective constants are bracketed, not computed (Parts XXXIX-XL narrow the square-lattice
  bracket to `[(log 101)/6, log 3)`, but the exact value is not determined); the numerical
  constants in the Part XIII ordering threshold (`log 2 / 8`, `log 3 / 12`) follow from the
  bracketing bounds on the conformation count and the worst-case coordination bound, and are
  not claimed to be optimal; no claim is
  made about the sharpness of the sample-complexity constants beyond the matching bounds in
  `SharpBound.lean`.


### 7.1 Items on this list that have since been closed

The assumptions above are the ones the development records against itself, and seventeen of them
have since been discharged by new parts rather than restated.  Each entry names the item, the part that
closes it, and the theorem that does the work.  `LIMITATIONS_CLOSED.md` describes them in prose.

* *Sequential analysis assumes independent reads* — Part CIX (`DependentReads.lean`), Ville's
  inequality and anytime validity for a null given by predictable conditional rates, with no
  assumption whatsoever about the dependence (`IDR.DepSeq.villeD`, `anytime_validD`).  The other
  hypothesis of that item — that the null read rate is known — is unchanged.
* *Independence across screened candidates* — Part XCIII (`DependentScreen.lean`), false discovery
  rate control under **arbitrary dependence** by e-values and e-BH
  (`IDR.DepScreen.ebh_fdr_le_level`).  The per-candidate calibration floor of Part XC is untouched
  by this and is stated as such in the file.
* *Class imbalance, per-protein aggregation, balanced accuracy and Matthews correlation* — Part
  XCIV (`Imbalance.lean`), including the imbalance and macro/micro ranking inversions
  (`IDR.Imbalance.imbalance_inversion`, `macro_micro_inversion`, `mcc_le_one`).
* *Coverage assumes independent draws* — Part XCV (`CorrelatedCoverage.lean`), missing mass for a
  correlated (Markov) trajectory by Doeblin minorisation (`IDR.CorrCoverage.unseen_le`).
* *Crooks' relation assumed rather than derived* — Part XCVI (`CrooksDerivation.lean`), derived
  from local detailed balance (`IDR.CrooksDeriv.crooks_derived`).
* *Identifiability proved for linear functionals only* — Part XCVII
  (`NonlinearIdentifiability.lean`), the reachable interval of a nonlinear ensemble report
  (`IDR.NonlinearIdentify.reachable_interval`).
* *Fitting results assume exact fits* — Part XCVIII (`ToleranceFit.lean`), fitting within error
  bars (`IDR.ToleranceFit.tol_fit_perturb`).
* *Genuinely non-stationary (ageing) behaviour outside the scope* — Part XCIX (`Ageing.lean`), the
  lag bound `‖p_t − π_t‖₁ ≤ (1−ε)^t‖p₀−π₀‖₁ + δ/ε` (`IDR.Ageing.ageing_lag`).
* *Markov state models: statistical error in the counts, and the continuous-time case* — Part C
  (`MarkovEstimation.lean`), stationary-population error propagation
  (`IDR.MSMEstimation.stationary_perturb`) and the generator bridge.
* *Single-molecule FRET assumes clean photon counting* — Part CI (`PhotonCorrections.lean`),
  leakage, direct excitation, gamma and background, with the calibration-parameter
  unidentifiability (`IDR.PhotonCorr.gamma_leakage_unidentifiable`).
* *Multi-replica scheduling and round-trip statistics* — Part CII (`ReplicaRoundTrip.lean`), the
  ballistic bound and the diffusive one (`IDR.ReplicaTrip.reach_prob_le`).
* *Aggregation: fragmentation, secondary nucleation as a separate channel, and the fibril length
  distribution* — Part CIII (`FibrilLength.lean`), the two-channel identifiability
  (`IDR.FibrilLength.channels_identified`) and the length-distribution results.
* *Cryo-EM: no image formation model, contrast transfer function, noise or alignment error; bulk
  solvent and resolution truncation* — Part CIV (`ImageFormation.lean`), the transfer-function
  zeros and the defocus pair (`IDR.ImageFormation.incommensurate_defocus_fills_zeros`), the solvent
  occupancy bias and the alignment peak drop.
* *The WHAM/MBAR self-consistency iteration and its convergence* — Part CV (`MBAR.lean`),
  consistency, uniqueness up to the scale freedom and stability of the iteration
  (`IDR.MBAR.mbar_consistent`, `mbar_unique_up_to_scale`, `mbar_iter_bracket`).  A convergence
  *rate* is still not proved.
* *The choice of clustering in a Markov state model* — Part CVI (`Clustering.lean`), the lumped
  chain as an exact restriction and the one-sided timescale bound for every clustering
  (`IDR.Clustering.lump_gap_transfer`, `lump_comp`).
* *The integrator results are Gaussian* — Part CVII (`AnharmonicIntegrator.lean`), exact
  reversibility for every force, the harmonic shadow energy, and the absence of any
  unconditionally stable timestep for a quartic force
  (`IDR.AnharmonicIntegrator.quartic_unstable`).
* *Quantum-mechanical bond making and breaking* — Part CVIII (`BondBreaking.lean`), the two-state
  reactive surface, the barrier lowered by exactly the coupling, the avoided crossing, and the
  unbounded error of a harmonic bond at dissociation (`IDR.BondBreaking.barrier_lowering`).

What remains open is stated without softening: the exact connective constants and the Flory
exponent for genuine self-avoiding walks (open mathematics, not a modelling choice); a theory of
water rather than the exact finite-solvent potential of mean force of Part LX; a many-electron
treatment of reactive chemistry beyond the two-state model of Part CVIII; a convergence rate for
the MBAR iteration; and, throughout, the empirical question of what any particular protein does,
about which nothing here is claimed.

---

## 8. Reading the formal development

Suggested order: `EnsembleCore.lean` → `ModelNature.lean` → `Verdict.lean` (the qualitative
answer) → `Metric.lean`, `Transport.lean`, `Invariance.lean`, `Design.lean` (the quantitative
laws) → `FreeEnergy.lean`, `Chain.lean`, `PartFour.lean` (the physics) →
`SampleComplexity.lean`, `Fisher.lean` (data) → `Condensate.lean`, `Relaxation.lean` (collective
behaviour and time) → `Polymer.lean`, `Observables.lean`, `Electrostatics.lean`, `Thermo.lean`,
`Flory.lean`, `Rouse.lean`, `NMR.lean`, `Hydrodynamics.lean`, `HelixCoil.lean` (measured
physics) → `Continuum.lean`, `GibbsMeasure.lean`, `Symmetry.lean`, `Generalisation.lean` (the
idealisations removed) → `LatticeWalk.lean`, `SelfAvoiding.lean`, `CubicLattice.lean`,
`PartTwelve.lean` (excluded volume) → `Boltzmann.lean`, `HPModel.lean`, `PartThirteen.lean` (a
microscopic sequence Hamiltonian) → `Scoring.lean`, `ScoreFloor.lean`, `PartFourteen.lean` (the
evaluation: which score may a disorder model be judged by) → `Crowding.lean`, `Force.lean`,
`Trajectory.lean`, `PartFifteen.lean` (the situated region: the cell, the pulling curve, the
single trajectory) → `Linkage.lean`, `Tether.lean`, `PartSixteen.lean` (coupling: reciprocity
and the disordered linker) → `SelfAssociation.lean`, `Multisite.lean`, `PartSeventeen.lean` (the
sample and the proteoform) → `Driven.lean`, `EntropyProduction.lean`, `PartNineteen.lean` (out of
equilibrium: the driven cycle and what it dissipates) → `Integrator.lean` (the finite timestep,
and which schemes escape it) → `Memory.lean` (one relaxation time is not enough) →
`Calibration.lean`, `PredictionSets.lean`, `PartTwentyTwo.lean` (uncertainty: calibration,
resolution and prediction sets) → `Epistasis.lean`, `PartTwentyThree.lean` (mutations: energy
adds, populations do not) → `Context.lean`, `Potentials.lean`, `Hamiltonian.lean`,
`Solvation.lean`, `GibbsField.lean`, `ContinuousScore.lean`, `Generative.lean`,
`PartTwentyFour.lean` (the molecular model: continuous space, force field, solvent, and the Gibbs
measure) → `ManyBody.lean`, `Protonation.lean`, `BrokenErgodicity.lean`, `Constraints.lean`,
`PartTwentyFive.lean` (the residual assumptions, priced) → `TransferMatrix.lean`,
`Tensorization.lean`, `PartTwentySix.lean` (tractability: what makes the model evaluable, and what
that costs) → `Metropolis.lean`, `Mixing.lean`, `PartTwentySeven.lean` (sampling: the chain that
needs no partition function, and the barriers that slow it) → `Karplus.lean`,
`HydrogenExchange.lean`, `PartTwentyEight.lean` (local restraints: what a coupling and an
exchange rate can and cannot pin down) → `PeriodicBox.lean`, `Cutoff.lean`,
`CorrelatedSampling.lean`, `PartTwentyNine.lean` (the protocol: box, cutoff and honest error
bars) → `Selection.lean`, `PartThirty.lean` (the entropy price of ordering, and what an affinity
prediction must report) → `ScalingExponent.lean`, `PartThirtyOne.lean` (what a reported Flory
exponent is a statement about) → `SpinRelaxation.lean`, `ChemicalExchange.lean`,
`PartThirtyTwo.lean` (dynamics from NMR: the spectral density, the `τ_c` ambiguity and the
invisible state) → `PhotonCounting.lean`, `PartThirtyThree.lean` (single-molecule histograms:
shot noise, detection thresholds and dynamic averaging) → `Association.lean`,
`PartThirtyFour.lean` (diffusion-limited association and the fly-casting criterion) →
`Denaturant.lean`, `PartThirtyFive.lean` (what a fitted `m`-value is a statement about) →
`Dichroism.lean`, `PartThirtySix.lean` (circular dichroism: which secondary-structure content the
spectrum determines and which the regulariser chooses) → `Aggregation.lean`,
`PartThirtySeven.lean` (aggregation kinetics: the lag time as a logarithm) → `Density.lean`,
`PartThirtyEight.lean` (why a disordered region is missing from a map, and what that absence
bounds) → `ConnectiveBound.lean`, `Bridge.lean`, `PartForty.lean` (the entropy per residue,
bracketed from both sides by finite counts) → `Evolution.lean`, `PartFortyOne.lean` (neutral
evolution: conserved descriptors, diverged sequences, and the symmetry a model should carry) →
`Cotranslational.lean`, `PartFortyTwo.lean` (synthesis: the lag behind a moving equilibrium and
the vectorial context) → `Rheology.lean`, `PartFortyThree.lean` (the material state: what
condensate rheology, probe diffusion and ageing exclude) → `Fitting.lean`,
`PartFortyFour.lean` (how much agreement with data is evidence) → `Benchmark.lean`,
`PartFortyFive.lean` (what an incomplete annotation certifies) → `Compensation.lean`,
`PartFortySix.lean` (why fitted enthalpies and entropies are correlated) → `SaxsFret.lean`,
`PartFortySeven.lean` (which moment each experiment weighs) → `FirstPassage.lean`,
`PartFortyEight.lean` (exact first-passage times: the landscape does not determine the rate) →
`Committor.lean`, `PartFortyNine.lean` (the committor: nor the mechanism) → `BarrierRate.lean`,
`PartFifty.lean` (a barrier is neither necessary nor sufficient for slow kinetics) →
`Polarisability.lean`, `PartFiftyOne.lean` (electronic polarisability: well posed, and not
pairwise) → `NuclearQuantum.lean`, `PartFiftyTwo.lean` (nuclear quantum effects and the isotope
effect a classical model cannot have) → `Orientation.lean`, `PartFiftyThree.lean` (the FRET
orientation factor: why an efficiency is not a distance) → `TransitionPath.lean`,
`PartFiftyFour.lean` (transition-path times: the crossing is not the waiting) →
`HeatCapacity.lean`, `PartFiftyFive.lean` (the curved van 't Hoff plot) → `Depletion.lean`,
`PartFiftySix.lean` (aggregation after the early-time regime) → `Anisotropy.lean`,
`PartFiftySeven.lean` (anisotropic displacement and discrete conformers in a map) → `Rdc.lean`,
`PartFiftyEight.lean` (orientational NMR: what a residual dipolar coupling determines) →
`LabelNoise.lean`, `PartFiftyNine.lean` (benchmarks whose annotations are wrong both ways) →
`Pmf.lean`, `PartSixty.lean` (explicit solvent: the potential of mean force is not a force
field) → `Photophysics.lean`, `PartSixtyOne.lean` (what a single-molecule FRET number is worth:
the detection factor, the background and the linker) → `MarkovStateModel.lean`,
`PartSixtyTwo.lean` (the Markov state model that is actually estimated: right thermodynamics,
one-sided kinetics, and a lag) → `WorkTheorem.lean`, `PartSixtyThree.lean` (pulling out of
equilibrium: the work, the dissipation and the free energy) → `InversePotential.lean`,
`PartSixtyFour.lean` (structure-based coarse-graining: unique, blind, and state-point specific) →
`ReplicaExchange.lean`, `PartSixtyFive.lean` (replica exchange: unbiased, priced, and unable to
repair broken ergodicity) → `Umbrella.lean`, `PartSixtySix.lean` (umbrella sampling: overlap is
exactly the condition for a free-energy difference to be determined) →
`SequenceDegeneracy.lean`, `PartSixtySeven.lean` (the sequence-to-ensemble map: what a pairwise
theory can never resolve) → `ReplicaLadder.lean`, `PartSixtyEight.lean` (the full ladder: `K`
replicas, and the invariant that survives) → `Restraints.lean`, `PartSixtyNine.lean` (the
restraint-counting law: how many experiments an ensemble costs) → `Identifiability.lean`,
`PartSeventy.lean` (what is verifiable in a reported ensemble: the measured span and its
dimension) → `Tolerance.lean`, `PartSeventyOne.lean` (the precision floor: structural degeneracy,
the resolution `eps / G` that no further experiment removes, and its matching ceiling) →
`ReportDesign.lean`, `PartSeventyTwo.lean` (the design and reporting theorem: certified readouts
of the data, and the exact cost of a report) → `ChargePatterning.lean`,
`PartSeventyThree.lean` (how much sequence a pairwise charge model can carry, and two sequences it
cannot separate) → `BindingPolynomial.lean`, `PartSeventyFour.lean` (multivalent binding: what a
Hill slope can be) → `FloryHuggins.lean`, `PartSeventyFive.lean` (the exact critical point of a
chain-length demixing model) → `SequencePhase.lean`, `PartSeventySix.lean` (sequence to phase
diagram: the threshold, the trade-off and the blind spot) → `ChainGeometry.lean`,
`PartSeventySeven.lean` (what a radius of gyration reports: the readout identity, the two-module
law and the rod ceiling) → `Phosphorylation.lean`, `PartSeventyEight.lean` (modification scanning:
epistasis is the kernel, and the experiment that breaks the sequence blind spot) →
`Screening.lean`, `PartSeventyNine.lean` (salt: the high-salt limit, and reentrant condensation
from a sign change in the charge autocorrelation) -> `Crosslink.lean`, `PartEighty.lean`
(crosslinking mass spectrometry: a yield is a population, a partial yield proves multiplicity, and
a spacer-length series measures a distribution) -> `Dual.lean`, `PartEightyOne.lean` (ensemble
reweighting as convex duality: the relative-entropy certificate, and the diverging multiplier as a
proof of infeasibility) -> `Heterokinetics.lean`, `PartEightyTwo.lean` (heterogeneous kinetics: the
apparent rate constant falls with time, and heterogeneity mimics stability) -> `IonMobility.lean`,
`PartEightyThree.lean` (ion mobility: the measured width splits into instrument plus ensemble, and
the resolving power a two-percent difference costs) -> `MomentProblem.lean`,
`PartEightyFour.lean` (the moment problem: `k` averages never fix an ensemble, a known support
does, and the bounds an average certifies outright) -> `CapacityExact.lean`,
`Falsification.lean`, `PartEightyFive.lean` (the capacity threshold as a testable design rule: the
exact attainable error at each component count, the kink at the measured state count, the design
rule, and the pre-registered test whose success is provably not a theorem) -> `CryoEM.lean`,
`PartEightySix.lean` (a cryo-EM map is an occupancy: peak height counts conformations, invisibility
of disordered density is a theorem, single-atom averages are read-outs of the map and correlations
are not) -> `Coverage.lean`, `PartEightySeven.lean` (what a finite sample has seen: the missing mass
exactly, the linear cost of coverage, and the Good-Turing estimator of the unseen population) ->
`Conditioning.lean`, `PartEightyEight.lean` (what a regularised fit reports: the exact shrinkage,
the capped noise gain, the stability-bias exchange rate and the resolution boundary) ->
`DetectionPower.lean`, `Visibility.lean`, `Instrument.lean`, `PanelPower.lean`,
`PartEightyNine.lean` (from the threshold to a study design: the exact miss law and the
pre-registered sample size, what a bounded observable can reveal and the probe to build, the four
numbers a frozen record fixes, family-wise power across a panel, and the budget window between
capacity and data).
Each `PartN.lean` file states the
capstone of its part and nothing else, so the capstones can be read first and the supporting
files consulted as needed.

Later parts continue the same reading order: `NoisyDetection.lean`, `ReporterPhysics.lean`,
`DetectionLowerBound.lean`, `NoisyInstrument.lean`, `ProbePanel.lean`, `PartNinety.lean` (the same
test on a real instrument) -> `FalseDiscovery.lean`, `ScreenBudget.lean`, `ChernoffScreen.lean`,
`PartNinetyOne.lean` (the discovery list of a proteome-scale screen) -> `Sequential.lean`,
`SequentialPower.lean`, `StoppingRules.lean`, `ConfidenceSequence.lean`, `PartNinetyTwo.lean` (a run that may be watched: Ville's inequality, the
anytime-valid capacity test, its exact evidence rate and variance, and the molecule count that
gives it power).

The limitation-closing parts of §7.1 continue it: `DependentScreen.lean` (Part XCIII, screening
without independence) -> `Imbalance.lean` (Part XCIV, imbalanced benchmarks) ->
`CorrelatedCoverage.lean` (Part XCV, coverage for a correlated trajectory) ->
`CrooksDerivation.lean` (Part XCVI, Crooks derived from local detailed balance) ->
`NonlinearIdentifiability.lean` (Part XCVII, nonlinear reports) -> `ToleranceFit.lean` (Part
XCVIII, fitting within error bars) -> `Ageing.lean` (Part XCIX, a region that never reaches
equilibrium) -> `MarkovEstimation.lean` (Part C, estimated Markov state models) ->
`PhotonCorrections.lean` (Part CI, what the FRET detector counts) -> `ReplicaRoundTrip.lean` (Part
CII, what a temperature ladder costs) -> `FibrilLength.lean` (Part CIII, fragmentation, secondary
nucleation and fibril lengths) -> `ImageFormation.lean` (Part CIV, the contrast transfer function,
solvent, truncation, alignment and noise) -> `MBAR.lean` (Part CV, the estimator behind the
umbrella windows) -> `Clustering.lean` (Part CVI, whatever the clustering, the timescales come out
short) -> `AnharmonicIntegrator.lean` (Part CVII, integrators beyond the Gaussian case) ->
`BondBreaking.lean` (Part CVIII, bonds that break) -> `DependentReads.lean` (Part CIX, stopping
when you like, with reads that are not independent).  `LIMITATIONS_CLOSED.md` is the prose summary
of these seventeen parts.

Parts CX–CXV take up the five items that §7 had left open: `MBARRate.lean` (Part CX, a geometric
convergence rate for the MBAR/WHAM iteration, governed by window overlap) -> `ManyElectron.lean`
(Part CXI, the Hubbard dimer, its exact Feshbach downfolding, and the derivation of the two-state
model of Part CVIII from it) -> `WaterTheory.lean` (Part CXII, an exactly solved hydrogen-bond
chain, the density anomaly, the hydrophobic effect and cold denaturation) ->
`NullCalibration.lean` (Part CXIII, why an uncalibrated null forces powerlessness, and the finite
control-read protocol that buys calibration at an explicit price) -> `ConnectiveExact.lean` (Part
CXIV, the exactly solvable directed lattice, an improved rigorous upper bound for the square
lattice, and the deterministic Flory window `1/2 ≤ ν ≤ 1`) -> `ConnectiveLower.lean` (Part CXV,
bridges, the supermultiplicativity of their count, and the two-sided bracket
`log 251 / 7 ≤ μ ≤ log 780 / 6`).  `OPEN_ITEMS_CLOSED.md` is the prose summary of these six parts
and states precisely what remains open.

Parts CXVI–CXIX carry that programme further: `PartiallyDirected.lean` (Part CXVI, the partially
directed chain solved exactly — its conformations are genuinely self-avoiding, their count is
Pell-like, its entropy per residue is exactly `log (1 + √2)`, and that is a new lower bound for
`μ`) -> `StretchedChain.lean` (Part CXVII, the same chain under tension: the exact free energy
per residue `log λ(u)`, and the force–extension law, with extension exactly one half of the
contour length at zero force and saturating at full stretching) -> `SawEnumerate.lean` (Part
CXVIII, a depth-first enumerator proved correct, the exact counts `cnt 8 = 5916` and
`cnt 10 = 44100`, and the sharpened bracket `log (1 + √2) ≤ μ ≤ log 44100 / 10`) ->
`SequencePatterningExact.lean` (Part CXIX, an axial bond decouples the chain, and moving one
heavy residue from the terminus to the neighbouring interior position changes the conformational
free energy by an exponentially large amount at fixed composition).
`EXACT_POLYMER_RESULTS.md` is the prose summary of these four parts.

Parts CXX–CXXIII solve the charge-patterning problem exactly.  `ChargePatterningExact.lean` (Part CXX)
proves the dipole identity — for any neutral charge assignment and any conformation whatsoever,
the pairwise squared-distance charge coupling is exactly minus the squared dipole — performs the
conformational average of the ideal chain exactly (`⟨M²⟩ = b² ∑_k S_k²`, with `S_k` the charge of
the first `k` residues) and brackets that sequence functional between the perfectly mixed and the
diblock patterns, which attain the two ends; at fixed composition the diblock's mean squared
dipole is larger by a factor growing like the square of the length.  `DielectricResponse.lean`
(Part CXXI) switches on a field and solves the model in closed form, `Z(u) = ∏_i 2 cosh(u b
S_{i+1})`, so that the field free energy is, up to `N log 2`, the total absolute prefix charge.
`PolarizationLaw.lean` (Part CXXII) differentiates it into the Langevin polarization law and
proves fluctuation–dissipation for the model: the zero-field susceptibility *is* the mean squared
dipole of Part CXX.  `ElectroStretching.lean` (Part CXXIII) adds the conjugate stretching field
and obtains the exact force–extension law; because half the bonds of the perfectly mixed sequence
sit at zero prefix charge and never respond, it cannot be stretched past half its contour length,
while the diblock of identical composition reaches the whole of it.
`DipoleAutocorrelation.lean` (Part CXXIV) closes the loop with Part LXXIII: the exactly computed
mean squared dipole is itself a pairwise separation-dependent charge energy, so it is a readout of
the charge autocorrelation alone and two neutral sequences with equal autocorrelation share it
exactly.  Even an exactly solved observable of a disordered region does not determine its sequence.
`FluctuationSuppression.lean` (Part CXXV) differentiates the two-field solution once more and
computes the conformational fluctuation of the extension exactly, `b² ∑_i (1 − tanh²(u b
S_{i+1}))`: the perfectly mixed sequence keeps a fluctuation of at least `(t−1) b²` at every field
strength, because a fixed fraction of its bonds sits at zero prefix charge and never feels the
field, while the diblock of the same composition is squeezed below `N b² (1 − tanh²(u b))` and so
frozen.

Parts CXXVI–CXXVIII put the salt back into the exact solutions.  Every statement of Parts
CXX–CXXV is made at zero ionic strength, with the bare chain kernel; a real disordered region
sits in an electrolyte.  Part LXXIX already analysed the salt dependence of the pairwise model in
the fugacity variable and identified what survives at high salt; what these three parts add is
the *size* of the patterning effect as a function of the screening length, uniformly over all
patterns, and the crossover between the low- and high-salt regimes.
`SaltCrossover.lean` (Part CXXVI) damps the linear chain kernel by the Debye factor `exp(−κ d)` and
proves the screening bound `|E| ≤ 4 N / κ²` for every sequence of unit charges — the screened
electrostatic energy is at most *linear* in the length of the region, uniformly over all charge
patterns, and it vanishes as the salt concentration grows.  Against that, at zero salt the
diblock and the perfectly mixed sequence — identical composition, opposite charge order — differ
by at least `t³/3 − t`, a *cubic* contrast.  The two bounds cross: `salt_crossover` and
`patterning_needs_long_screening_length` show that once `κ t > 12` the contrast has fallen below
half its unscreened value, so charge patterning is visible only while the Debye screening length
is comparable to the length of the region itself.  `no_salt_blind_prediction` turns that into a
statement about models: any predictor that assigns an energy to a sequence *and to a sequence
only* is wrong by at least `(t³/3 − t)/8` on one of four sequence/condition pairs.
`DebyeScreening.lean` (Part CXXVII) repeats the collapse for the standard Debye–Hückel
Gaussian-chain kernel `exp(−κ b √d)/(b √d)` of Part LXXIII — `|E| ≤ 54 N / (κ³ b⁴)`, hence
`debye_pattern_collapse`: *any two* unit-charge sequences of the same length, not merely the two
extremal ones, differ by at most `108 N / (κ³ b⁴)`.  The conclusion is a property of screening,
not of a convenient kernel.  `SaltAwareDesign.lean` (Part CXXVIII) states the design rule that
follows: context is indispensable (1), the charge autocorrelation together with the solution
condition reproduces every pairwise electrostatic energy exactly (2), and even that
representation is provably incomplete, since the homometric pair of Part LXXIII shares its
autocorrelation at every lag — and therefore its Debye energy at every ionic strength — while a
three-body correlator separates the two sequences (3).  A model of a charged disordered region
must be conditional on the solution condition and many-body in the sequence; neither requirement
can be traded for more data or more parameters of the wrong kind.

The crossover of Part CXXVI is two-sided: `energy_perturb` bounds the difference between the
screened and the unscreened energy by `κ N⁴`, so for `κ ≤ 1/(288 t)` the patterning contrast is
*still* at least half its unscreened value (`patterning_survives_at_low_salt`), while for
`κ t > 12` it is less than half.  `crossover_two_sided` states the two together: the transition
happens at `κ t` of order one, neither sooner nor later.

Parts CXXXIV–CXXXVI add the noise to the identifiability analysis of Parts CXXIX–CXXXIII, and it
changes the conclusion.  `TitrationResolution.lean` (Part CXXXIV) observes that the lag-`D` charge
correlation reaches the titration curve only through the factor `D e^{−κ d}`: perturbing the
correlation profile at that lag by `δ` shifts the whole curve by exactly `D e^{−κD} δ`, so with
an energy resolution `eps` and conditions confined to `κ ≥ κ₀` a perturbation as large as
`eps·e^{κ₀D}/D` is invisible at *every* accessible condition (`resolution_floor`).  The threshold
is exact in both directions: a condition that separates a lag-`D` discrepancy `δ` must satisfy
`κ < log(D|δ|/eps)/D` (`separating_condition_below_threshold`), and every condition at or below
that value does separate it (`detectable_at_low_salt`).  Over regions of growing length there is
no uniform recovery at all (`no_uniform_profile_recovery`).  The concrete instance is a contact
between the two ends of the region: the two sequences differ by exactly `(N−1)e^{−κ(N−1)}` at
every ionic strength, a signal of `N − 1` at zero salt that falls below any resolution once the
Debye length is shorter than the region by a logarithmic factor.  `ResolutionDesign.lean`
(Part CXXXV) turns this into numbers — for a twenty-residue region read to `10⁻³ kT` from
conditions with a Debye length of one residue spacing, the lag-10 correlation is undetermined
over two charge units, the lag-15 correlation over its entire physical range, and the end-to-end
contact, worth `19 kT` unscreened, is invisible — and into the general horizon
`high_lags_unconstrained`: beyond a lag of order `4B/(eps κ₀²)` a salt titration constrains
nothing, whatever the region, buffer and instrument.  `ContactPanel.lean` (Part CXXXVI) supplies
the positive complement, so that the moral is a choice of probe rather than a counsel of despair:
a crosslinking-style panel whose kernel is a step in separation reads the cumulative correlation
`S(D) = ∑_{d≤D} C(d)`, inverts by the first difference `C(D) = S(D) − S(D−1)`, and is
`2`-Lipschitz — uniformly in the lag, the length of the region and the ionic strength
(`panel_stability`).  On the same worked case the panel pins the lag-15 correlation to `±2·10⁻³`
where the titration leaves it free over `[−20, 20]`.  Long-range charge correlations of a
disordered region are identifiable; screened energies are simply the wrong observable for them,
because screening is an exponential filter on sequence separation.  `RESOLUTION_CONTEXT.md`
gives the prose account.

`DebyeResolution.lean` (Part CXXXVII) asks how severe the ill-conditioning of Part CXXXIV really
is, and finds that the answer is a property of the assumed distance law rather than of the
region.  The chain kernel `d e^{−κd}` used there has a screening exponent proportional to the
sequence separation; the standard Debye–Hückel coupling on a Gaussian chain has exponent
`κ b √d`, since residues `d` apart sit a spatial distance `b√d` apart.  Redoing the analysis with
that kernel gives the floor `eps · b√D · e^{κ₀ b √D}` (`debye_single_lag_invisible`), exact in
both directions (`debye_detectable`), and a horizon (`debye_high_lags_unconstrained`) that sits
at separation of order `(log(B/eps)/(κb))²` — the square of the chain-kernel horizon.  On the
worked case (`demo_debye_lag16_resolved`), at bond length and inverse screening length equal to
one and a resolution of `10⁻³`, a lag-16 discrepancy of a quarter of a charge unit is detected at
that single condition, where the exponentially screened chain kernel leaves an entire physical
range free at lag 15.  Ill-conditioning is generic — every kernel that decays in separation
filters the long lags — but a model must say which distance law it assumes before it can quote a
resolution on a long-range correlation.

## 6zzz. Feasibility geometry of a measured distance panel (Part CL)

The transport part of this development asks how far a candidate ensemble is from the measured
one. `DistanceRealizability.lean`, `DistanceCutEnsembles.lean`, `ContactBudget.lean` and
`ContactPacking.lean` (Part CL) ask the prior question: which panels of averaged pairwise
distances are the panel of *any* ensemble. Two constraints survive averaging and hold for the
average itself — the triangle inequality (`meanDist_triangle`) and the contour bound
`⟨d(i,j)⟩ ≤ b·|i−j|` (`meanDist_le_chain`) — so a triangle defect exceeding `3e` refutes every
ensemble at error bar `e` (`no_ensemble_of_triangle_defect`), with the instrument specification
`e < D/3` for an observed defect `D`.

Those tests are far from sufficient. Ensembles realise the whole cut cone up to the contour scale
(`cutEnsemble_meanDist`); the explicit two-state exchange of four labelled sites whose mean panel
is `1` around the cycle and `2` across the diagonals is a metric that respects the contour bound
and is the distance panel of no configuration of any Euclidean space
(`no_single_structure_of_fourCycle`), with every single structure missing an entry by at least
`1/5` (`single_structure_fit_error_lower_bound`). Fitting one structure to averaged restraints
targets an object that need not exist.

The population side is budgeted by excluded volume: a Markov bound forces population from a short
measured mean (`contactPop_ge`), a counting bound caps the total (`sum_contactPop_le_multiplicity`),
and a grid-pigeonhole packing bound supplies the cap `(2⌈2D/σ⌉+1)³` (`packing_card_le`), giving
`∑ₖ (1 − Mₖ/D) ≤ (2⌈2D/σ⌉+1)³` for every ensemble of chains with hard-core separation `σ`
(`hard_core_contact_budget`, `hard_core_panel_falsified`) and the lower bound `r` on the number of
conformations demanded by `r` mutually exclusive populated contacts (`min_ensemble_size`). The
capstone is `distance_panel_design_law`; the prose account is `FEASIBILITY_CONTEXT.md`.

At three labelled sites the necessary condition is also sufficient: `DistanceThreePoint.lean`
realises any triangle-consistent triple of mean distances by an explicit three-state exchange
(`threePointEnsemble_meanDist`), so `three_point_feasible_iff` characterises three-site
feasibility exactly, and clause (vi) of `distance_panel_design_law` records it.

## 7zzz. Topological entanglement: the clause reweighting cannot supply (Part CLI)

Working in the plane transverse to a straight axis (partner helix, pore, filament) with exclusion
radius `R` and maximum bond length `b`, `Winding.lean` defines the discrete turning angle of a
bond as seen from the axis and the winding number `winding p n` in turns. The chord bound
`2R|sin(θ/2)| ≤ b` (`two_mul_abs_sin_half_le`) gives, with Jordan's inequality, `|turn| ≤ πb/(2R)`
per bond (`abs_turn_le`), hence the wrapping budget `|winding| ≤ n b/(4R)` (`abs_winding_le`) and
the exclusion-free polygon bound `|winding| ≤ n/2` (`abs_winding_le_half`). Read as a design rule,
`k` turns of threading demand at least `4kR/b` residues (`length_demand_of_winding`). For a closed
chain the winding number is an integer (`winding_int_of_closed`, via the argument of the
telescoping product in `Real.Angle`), and it is invariant along any continuous deformation
respecting the constraints (`winding_invariant_of_deformation`) — the total turn is continuous
because with `b < 2R` no bond reaches the branch cut, and a continuous integer-valued function on
an interval is constant.

`Threading.lean` gives the ensemble and data statements: `threadedFraction_le_budget` (population
ceiling `n b/(4Rk)`), `no_ensemble_of_excess_threading` (model-free refutation with error bars),
`length_demand_of_threaded_fraction`, `sector_population_invariant`, and the modelling no-go
`reweighting_cannot_create_threading` with its quantitative form `unthreaded_model_gap`. Tightness
is exhibited rather than assumed: `wrap_admissible` realises exactly `k` turns with `n ≥ 2πkR/b`
bonds on a circle of radius `R`, so `wrap_threshold_window` brackets the residue demand within a
factor `π/2`. `meanWinding_realises` records that a measured mean threading level is reproduced by
a two-state mixture, so it reports a population rather than a structure. The capstone is
`EntanglementVerdict.lean`'s `entanglement_design_law`; the prose account, including the caveat
that this is the winding invariant about an axis and not a knot invariant, is
`ENTANGLEMENT_CONTEXT.md`.

`Winding3D.lean` carries the whole development into three dimensions: projection to the plane
transverse to the axis is 1-Lipschitz (`norm_transverse_sub_le`) and `‖transverse u‖` is the
distance from the residue to the axis (`norm_transverse_le_dist_axis`, `dist_axis_attained`), so
`abs_winding3_le`, `length_demand3` and `winding3_invariant_of_deformation` hold for a chain in
`ℝ³` wrapping a rod, pore or filament. `ThreadingExtension.lean` then converts the topological
invariant into a metric prediction: `abs_axial_step_le` (a bond of length `b` with transverse
chord `c` advances at most `b − c²/(2b)` along the axis), `sum_chord_ge` (the chords sum to at
least `4R|w|`) and Cauchy–Schwarz give `axial_extension_le`,
`|z(n) − z(0)| ≤ n b − 8R²w²/(b n)`, with the two readings `winding_bound_of_extension` and
`unwrapped_of_large_extension`. Clauses (v) and (vi) of `entanglement_design_law` record them.

The ensemble forms are `meanSqWinding_le_of_extension` (a measured mean extension `D` caps the
mean squared winding at `b n (n b − D)/(8R²)`) and `threadedFraction_le_of_extension` (Chebyshev:
the population wound `k` times or more is at most `b n (n b − D)/(8R²k²)`).

## Modular construction: gluing fragments, and what it costs

Cut a disordered chain into (segment, seam, segment). The modular model
`glue p (x,y,z) = p(x,y)·p(y,z)/p(y)` reproduces both fragment panels exactly
(`glue_margXY`, `glue_margYZ`), is conditionally independent across the seam (`condIndep_glue`),
and is exact precisely for ensembles with that property (`glue_eq_self_iff_condIndep`). Its
relative entropy from the truth is exactly the conditional mutual information across the seam
(`klG_glue_eq_cmi`), which is nonnegative and vanishes only in that case (`cmi_nonneg`,
`cmi_eq_zero_iff_condIndep`); Pinsker turns this into the cut-placement rule `seam_design_rule`.
`no_modular_model_beats_seam_information` shows the floor is intrinsic to the cut: every
strictly positive conditionally independent ensemble, however fitted, pays at least `cmi p`. Its
ingredients — `log_sum_inequality` and the marginalisation bound `klXY_ge_klY` — are proved from
scratch. `l1_le_of_conditional_defect` is the safe-side bound (`delta`-decoupled segments give
population error at most `delta·|X|·|Z|`); `longRange` is the unsafe-side witness, a long-range
terminal contact invisible to both fragment panels whose glued model is at population distance
`1`, halves the end-to-end contact probability, and costs `log 2` for every modular model
(`longRange_l1`, `longRange_contact`, `longRange_cmi`,
`longRange_no_modular_model_is_close`). `cmi_coarse_le` and
`coarse_validation_can_hide_everything` record that coarse validation understates the price, and
can hide it completely. The capstone is `ModularDesignLaw.lean`'s `modular_design_law`, with the
prose account in `MODULAR_CONTEXT.md`.

`fragment_panels_never_falsify_modularity` records that fragment data can never test the
assumption: for every ensemble the glued model matches both fragment panels and has zero seam
information.

## 8zzz. The pairwise protocol: capacity, and why block-correlated noise barely touches it

Scoring a benchmark residue by residue caps the number of methods it can order at about
`1/(2·ν_label)`, where `ν_label` is the annotation error rate (Part on benchmark capacity,
`RequestProject/BenchmarkCapacity.lean`). `RequestProject/PairwiseProtocol.lean` changes the
protocol: score **within-group ordered pairs** of residues, each carrying the verdict "the first
residue is disordered and the second is not". `pairwise_capacity` shows the same law governs the
new protocol with its own rate — a certified family has at most `⌈1/(2·ν_pair)⌉` members, where
`ν_pair` is the fraction of scored pairs whose reference verdict the annotation gets wrong.

`correlated_noise_reduces_pair_discordance` is the reason to change it. If the truth is constant
on the blocks of a partition and the annotation errs by flipping whole blocks — the realistic
error model, since regions rather than residues are annotated — then a scored pair can only be
corrupted when it straddles a block boundary and touches a flipped block, so

    ν_pair ≤ (boundary mass / total scored pairs) · ν_label,

deterministically, with no assumption about which blocks flip. On a chain compared in adjacent
pairs with blocks of `b` consecutive residues the boundary fraction is `4/b`
(`RequestProject/PairwiseChain.lean`, `chain_pairwise_noise_bound`), so the capacity ceiling of
the leaderboard grows linearly in the block length (`chain_capacity_ceiling_gain`); when the
blocks are unions of comparison groups the protocol is exactly noise-free
(`chain_aligned_no_discordance`). `PAIRWISE_PROTOCOL_CONTEXT.md` gives the prose account.

## Part XCIV. Multiplicity control when the screen is not a product

The false discovery rate theorem of Part XCI assumes a product experiment, and Part XCIII buys
dependence-freedom by asking each candidate for an e-value rather than a p-value. A real screen
reports p-values and its candidates are coupled through shared reagents, a shared calibration run
and batch effects. `RequestProject/DependentBH.lean` proves the statement that covers that case —
the harmonic correction — under an arbitrary joint law, with validity of the null p-values
(`Superuniform`) as the only assumption.

The mechanism is a telescoping identity, `wgt_decomp`: on every outcome,

    1{i ∈ R}/|R| = Σ_{j=1}^{m} (1/j − 1/(j+1))·1{i ∈ R, |R| ≤ j} + (1/(m+1))·1{i ∈ R}.

The events on the right are nested rather than disjoint, and self-consistency of a step-up rule
(`SelfConsistentP` : every reported candidate has `p_i ≤ |R|·q/m`) contains each of them in the
single-candidate event `{p_i ≤ j·q/m}`. Validity then bounds each term, and the coefficients sum
to the harmonic number: `E[1{i ∈ R}/|R|] ≤ (q/m)·H_m` (`mean_wgt_le`). No property of the joint
law is used at any stage. Summing over the nulls gives `selfConsistent_fdr_le_harmonic`
(`E[FDP] ≤ q·H_m·|H₀|/m` for any step-up rule) and hence the Benjamini–Yekutieli theorem
(`benjamini_yekutieli`, `benjamini_yekutieli_level`): BH run at the deflated level `α/H_m`
controls the false discovery rate at `α` whatever the dependence. The price is logarithmic,
`log(m+1) ≤ H_m ≤ 1 + log m`, against Bonferroni's factor `m`, and the corrected list still
contains the Bonferroni list (`by_dominates_bonferroni`).

`RequestProject/DependentBHSharp.lean` shows the factor is not an artefact of the proof. For every
`m` and every level it constructs a joint law — outcome `(j, s)` gives the cyclic window of length
`j+1` starting at `s` the p-value `(j+1)q/m` and everyone else `1`, with probability
`q/((j+1)·m)` — in which all `m` hypotheses are null, all p-values are valid
(`pval_superuniform`, by the balance lemma `card_windowsThrough`), BH rejects exactly the window
(`bhRej_pv`), every discovery is false, and therefore

    E[FDP] = q·H_m     (`bh_fdr_eq_harmonic`).

So uncorrected BH at level `q` exceeds its nominal level from two candidates on
(`uncorrected_bh_exceeds_level`) and the corrected procedure sits exactly at its bound
(`by_level_is_attained`): under arbitrary dependence the deflation to `α/H_m` is necessary, not
conservative. `DEPENDENT_BH_CONTEXT.md` gives the prose account.
