# Can structure-aware models ever fully predict intrinsically disordered regions and their binding?

**Short answer: no — not if "structure-aware model" means *sequence in, one structure
(with or without a confidence score) out*. Yes — if the model keeps the structural
machinery but changes what it predicts: a *context-conditional probability distribution*
over conformations, with capacity allowed to grow with the target.**

Everything asserted below is proved in Lean 4 in this repository; each claim is followed
by the name of the theorem that establishes it. The build is free of `sorry` and uses only
Lean's standard axioms.

## 1. The setting (generalised)

A conformation is a point of an arbitrary space `X` (`Conf n = Fin n → ℝ` when we need
geometry). An **ensemble** `Ens X` is a finitely supported probability distribution on `X`.
The *same* structure is also the definition of a **model**: a finite mixture /
latent-variable generative model, whose components are the decoded structures and whose
weights are the latent distribution. A model is *correct* when it is observationally
identical to the target, `Ens.Same`: **every** observable has the same average. That is the
strongest possible notion of correctness — no measurement can tell model and target apart.

A **prediction problem** is a family of targets `T : I → Ens X` indexed by the model's
inputs; a **predictor** is `A : I → Ens X`; and `Solves A T` says `A` is right on every
input.

## 2. Why single-structure models fail — sharply

* A single-structure output is a point mass. It is exactly right **iff** every degree of
  freedom has zero variance, i.e. iff the region is rigidly ordered
  (`Ens.deterministic_iff_var_zero`). Disorder is precisely the complement of the domain of
  validity of structure prediction; the boundary is not fuzzy.
* Quantitatively, the squared error of any predicted structure is the total variance plus
  the squared bias (`Ens.pointLoss_eq`), so it is bounded below by the variance of any one
  disordered degree of freedom (`Ens.var_le_pointLoss`) — an irreducible floor, independent
  of the model.
* The loss-optimal structure is the ensemble mean, and the mean is typically **not a
  physically admissible conformation**: if every populated conformation obeys a hard
  geometric constraint (fixed bond lengths, a definite radius — modelled as lying on a
  sphere), the mean lies strictly inside it (`Ens.mean_sq_lt_of_disordered`). Regression
  towards a structure produces geometry that no molecule ever adopts.
* Adding a per-residue confidence score does not change any of this
  (`confidence_score_does_not_help`): a confidence can *flag* disorder, it cannot
  *represent* it.

## 3. Why the other usual model families fail

The general principle (`exists_failure_of_not_separating`): a predictor whose output
depends on the target only through a family of statistics `S` that does **not determine**
the ensemble must be wrong on some target. This is information-theoretic — more data, more
parameters and better optimisation cannot repair it. Instances, each proved by exhibiting
two ensembles the family cannot distinguish:

| family | what it models | theorem |
|---|---|---|
| affine observables | coordinate regression, "predict the mean structure" | `not_separates_linearObs` |
| degree ≤ 2 observables | mean + covariance, B-factors, predicted error bars | `not_separates_quadraticObs` |
| per-residue marginals | mean-field / factorised per-residue predictions | `not_separates_marginalObs` |
| functions of pairwise distances | distograms, contact maps | `not_separates_distanceObs` |

* Mean-field models fail specifically because they cannot carry **correlations**
  (`no_meanField_captures_corrEns`) — and correlated disorder is exactly what couples a
  disordered region to a partner.
* No **fixed capacity** suffices: for every `k` there is an ensemble that no `k`-component
  model reproduces (`exists_ensemble_beyond_capacity`, `Ens.card_le_of_same`,
  `not_solves_of_bounded_capacity`). Model size must scale with the breadth of the
  ensemble.
* Most generally: **no finite family of observables whatsoever determines an ensemble**
  (`not_separates_of_finite`, proved by a dimension count). Hence any predictor fitted to
  finitely many statistics is wrong on some target (`finite_statistics_predictor_fails`),
  and experimental ensemble refinement against finitely many restraints (SAXS, NMR, FRET
  averages) is intrinsically underdetermined (`finite_experiment_underdetermined`).

## 4. Why binding is harder still

* **Context dependence.** A disordered region does not have *an* ensemble; it has one per
  thermodynamic context. Any predictor that reads the sequence but not the context is wrong
  in at least one context (`not_solves_of_context_blind`, `exists_context_failure`), and
  such contexts genuinely differ (`contexts_can_differ`). The cure is structural, not
  quantitative: condition on the context (`conditional_predictor_exact`).
* **Conformational selection versus induced fit.** Modelling binding as Boltzmann
  reweighting of the free-state ensemble can only redistribute weight among conformations
  already populated (`Ens.reweight_prob_pos_iff`). Hence a reweighting model can never
  describe induced fit, where the complex populates new geometry
  (`no_reweight_of_unpopulated`, `no_boltzmann_of_unpopulated`, `induced_fit_example`).
* **Reweighting never fully orders.** Conversely, at any finite interaction strength and
  temperature, a degree of freedom disordered in the free state stays disordered in the
  complex (`reweight_var_pos`, `boltzmann_var_pos`). Fuzzy complexes are the generic
  outcome; "the structure of the complex" often has no correct answer at all
  (`fuzzy_complex_not_deterministic`).
* **Affinity is not a function of a structure with error bars.** Two ensembles with
  identical mean and identical variance can have different equilibrium binding constants
  (`binding_not_determined_by_two_moments`).

## 5. What does work

* **Ensemble models are universal.** Every prediction problem is solved exactly — on every
  observable — by a predictor that outputs a context-conditional distribution over
  conformations (`solvable_by_conditional_ensembles`); in latent-variable form,
  `exists_latentModel_captures`.
* Disorder *forces* that form: any model reproducing a disordered degree of freedom must
  place positive probability on at least two latent states with different geometry there
  (`captures_disordered_two_states`, `two_le_k_of_captures_disordered`).
* **The design specification in one theorem** (`disorder_model_design`): condition on the
  context, output a distribution, let the number of components scale with the target
  (it must be at least the number of populated conformations), and fit the weights by
  maximum likelihood.
* **Fitting is well posed.** Given a conformational library, maximum likelihood on the
  weights is consistent: the true weights are the unique minimiser of the expected negative
  log-likelihood (`crossEntropy_min_iff`, from Gibbs' inequality `klDiv_nonneg`,
  `klDiv_pos_of_ne`).

## 6. The verdict, in one theorem

`structure_aware_verdict` bundles the five statements: single-structure output fails;
every fixed capacity fails; context blindness fails; no finite family of statistics
(in particular none of the four families above) determines the target; and
context-conditional ensemble predictors solve every problem exactly.

So the honest answer to "will structure-aware models ever fully predict disordered regions
and their binding?" is: **the structural machinery is not the problem — the output type,
the input type and the fixed capacity are.** A model that predicts one structure per
sequence is provably wrong on every disordered region, and no amount of scale changes that.
A model that predicts a context-conditional ensemble can in principle be exactly right, but
even then, what can be *pinned down from finitely many measurements or statistics* is
strictly less than the ensemble itself.

---

# Part II. The nature of the model, in general

Part I was about *structure-aware* models. Part II drops that qualifier and asks what
**any** model of a disordered region must be, and what no model can be. The statements
below are architecture-free: they constrain the model's *type* — its input, its output, its
resolution, its scope — and never its internals. All of them are proved in Lean in this
repository, with no `sorry` and only Lean's standard axioms.

Two capstones collect them: `model_must_be` and `model_cannot_be` (in
`RequestProject/Verdict.lean`), together with `model_scope` for the questions of
resolution and time.

## 7. What the output object has to be

* **A point of the simplex over conformation space, not a point of conformation space.**
  Every ensemble is a convex combination of single structures
  (`Ens.expect_eq_sum_dirac`): single structures are exactly the *extreme points* of the
  set of possible answers, which is why committing to one is a strict loss of generality.
* **The exact criterion for a model class to be enough.** A class of outputs solves *every*
  prediction problem if and only if it realises every ensemble up to observational equality
  (`expressive_iff_solves_all`). The design problem is entirely a question about the model's
  *range*; the computational architecture is unconstrained.
* **Convexity is forced.** Such a class must represent every population ratio between any
  two of its outputs (`expressive_closed_under_mix`), so "return one of finitely many
  templates" is excluded as a matter of type, not of accuracy.
* **Capacity must scale with the target** (`Ens.card_le_of_same`), and no fixed capacity
  suffices (`exists_ensemble_beyond_capacity`).

## 8. The impossibilities are quantitative, not just exact

A natural objection to Part I is that models are only ever approximately right. `ApproxSame
eps` compares model and target on *all* observables bounded by one — the strongest uniform
notion of approximate correctness available. The obstructions survive:

* **A single structure is wrong by a fixed margin.** Against a two-state target, every
  single-structure output — whatever structure it picks, with whatever confidence — differs
  from the truth by at least the population it misses (`not_approx_dirac`). The gap is set
  by the thermodynamics of the target, not by the quality of training.
* **Bounded capacity is wrong by at least `1/(k+1)`.** A model emitting at most `k`
  structures cannot approximate the uniform ensemble over `k+1` conformations
  (`not_approx_of_bounded_capacity`); sampling more decoys from it does not converge.
* **Finite ensembles are themselves an idealisation.** No finitely supported model equals an
  atomless (continuous) conformational distribution: they differ maximally on the indicator
  of the model's own support (`no_finite_ensemble_eq_atomless`). This is why approximate
  correctness — and not exact equality — is the honest target, and why the two quantitative
  statements above are the operative ones.

## 9. Three further things the model cannot be

* **It cannot route its answer through a finite internal code.** If the model's dependence
  on the target passes through a finite set of internal states — a disorder class, a chosen
  template, a discrete vocabulary of conformations — then it fails on one of any
  `card + 1` observationally distinct targets, whatever the decoder
  (`no_finite_representation`). Capacity has to be counted at the bottleneck, not only at
  the output.
* **It cannot respect a free symmetry and still output a structure.** If the landscape is
  invariant under a transformation with no fixed conformation — mirror-image states, a
  register shift between equivalent binding modes — then the equilibrium ensemble is
  symmetric and no single conformation is (`not_deterministic_of_symmetric`,
  `fixed_point_of_symmetric_deterministic`). This argument uses no notion of variance or
  distance; it is purely structural, and it is not vacuous
  (`exists_symmetric_nondeterministic`).
* **It cannot be factorised over parts of the chain.** Independent per-residue or
  per-module prediction, and equivalently any one-body (separable) energy, produces an
  ensemble with identically zero coupling (`prod_factorised`, `separable_energy_prod`), so
  a correlated target is not approximated badly but excluded outright
  (`no_factorised_captures_corrPair`).

## 10. Energy-based models: right in form, with two hard limits

* **The form is right, and costs nothing.** Every ensemble with positive weights is exactly
  the Boltzmann ensemble of some energy function, at any temperature one likes
  (`exists_energy_representation`).
* **Only differences are learnable.** Two energies give the same predictions exactly when
  they differ by an additive constant on the library (`energy_unique_up_to_const`), so
  absolute energies are neither trainable nor testable.
* **Finite energies cannot forbid anything.** At any finite temperature every library
  conformation keeps a positive population (`energyEns_prob_pos`), so a hard constraint —
  excluded volume, chain connectivity, an obligate contact — has to be built into the
  *support* of the model, a modelling decision the energy cannot make
  (`no_energy_model_excludes`). The same mechanism is why Boltzmann reweighting of a fixed
  reference ensemble cannot produce induced fit (`no_boltzmann_of_unpopulated`).
* **What rescues factorisation is a latent variable.** Every two-part ensemble is a
  *mixture* of factorised ones (`mixture_of_products_universal`): conditional independence
  given a latent state is fully general even though unconditional independence is fatal.
  The latent variable is precisely the carrier of the correlations, which is the formal
  reason latent-variable generative models are the right shape.

## 11. Resolution and time: what the model must be *about*

* **Coarse descriptors are necessary but never sufficient.** Correctness at full resolution
  implies correctness for every descriptor (`map_same_of_same`), but two observationally
  different ensembles can share a coarse description
  (`exists_same_coarse_ne_fine`); and if a descriptor merges even one pair of conformations,
  every model supervised only through it is wrong on some target
  (`coarse_trained_model_fails`). Coarse data also never singles out a fine model: every
  coarse target is the shadow of some full-resolution ensemble
  (`exists_lift_of_surjective`).
* **Populations do not determine kinetics.** Two Markov dynamics — one frozen, one
  exchanging every step — have identical equilibrium ensembles and different two-time
  observables (`stationary_does_not_determine_dynamics`), so any predictor of dynamics that
  is a function of the equilibrium distribution alone is wrong about some system whose
  populations it gets exactly right (`no_static_model_predicts_kinetics`). A correct
  ensemble is necessary but not sufficient.
* **The repair needs no new theory.** Because the whole framework is stated over an
  *arbitrary* conformation space, taking that space to be trajectory space makes every
  theorem above a statement about kinetic models: a model outputting a distribution over
  trajectories is universal (`trajectory_ensembles_universal`), a single predicted
  trajectory is not (`no_single_trajectory`).

## 12. The nature of the model, in one paragraph

The model must be a **conditional generative model of a distribution over the objects whose
behaviour is being claimed** — conformations if the claim is thermodynamic, trajectories if
it is kinetic — taking the thermodynamic context as an input, with an output class rich
enough to realise every distribution (hence convex, hence of unbounded capacity, hence with
no finite internal bottleneck), carrying correlations through a latent variable rather than
factorising them away, defined and supervised at the resolution at which its predictions
are stated, optionally parametrised by an energy defined up to a constant over a support
chosen in advance, and fitted by maximum likelihood. It cannot be a single structure, a
structure with error bars, a bounded set of decoys, a context-blind map, a function of
finitely many statistics, a per-residue independent model, a reweighting of a fixed
reference ensemble, or a model whose claims are validated only through a lossy descriptor.
Each of these "cannot"s is a theorem, and each holds against arbitrary computational power
and arbitrary data.

---

# Part III — The quantitative design laws

Parts I and II answer *whether* a family of models can be right. Part III answers the two
questions a designer asks next: **how is error measured**, and **how big must the model
be**? The new material lives in `RequestProject/Metric.lean`, `RequestProject/MaxEnt.lean`,
`RequestProject/Invariance.lean` and the capstone `RequestProject/Design.lean`
(`quantitative_design_laws`, `locality_law`, `sampling_law`). Everything is proved over a
finite conformational library, with no `sorry` and only Lean's standard axioms.

## 13. The loss is the population-space ℓ¹ distance

On a finite conformation space, an ensemble is exactly its population vector
(`Ens.expect_eq_sum_prob`, `Ens.same_iff_prob_eq`), and the uniform observational error
`ApproxSame` used throughout Part II *is* the ℓ¹ distance between population vectors:

* `Ens.approxSame_iff_ell1_le` — `ApproxSame eps E F ↔ ℓ¹(E,F) ≤ eps`;
* `Ens.ell1_eq_sSup_expect_diff` — ℓ¹ is the supremum of the discrepancies over all
  observables bounded by one, so no experiment can do better than the ℓ¹ error;
* `Ens.ell1_eq_zero_iff` — ℓ¹ vanishes exactly on observationally equal ensembles, and
  `Ens.ell1_triangle`, `Ens.ell1_comm`, `Ens.ell1_le_two` make it a metric of diameter two.

There is therefore no freedom in the choice of loss: RMSD to a single structure is the
wrong object, and the right one is a distance between distributions.

## 14. Capacity must grow linearly with the ensemble — and exponentially with its entropy

* `Ens.ell1_ge_of_card_le` — a model built from at most `k` components is at ℓ¹ distance at
  least `(m − k)·δ` from any target populating `m` distinct conformations with weight `≥ δ`.
* `Ens.capacity_lower_bound` — equivalently, uniform accuracy `eps` **forces**
  `k ≥ m − eps/δ`. Capacity is not "large enough"; it is linear in the number of populated
  conformations.
* `ell1_truncated_unif` — the law is sharp: keeping `k` of `m` equally populated
  conformations gives ℓ¹ error exactly `2(m−k)/m`.
* `entropy_le_log_card` and `card_ge_exp_entropy` — the intrinsic scale is conformational
  entropy: an exactly correct model must carry at least `exp H` components, `H` the
  Gibbs–Shannon entropy of the target ensemble (`entropy`), which is a property of the
  ensemble and not of its parametrisation (`entropy_eq_of_same`). Disorder *is* large `H`.
* `sampling_law` — a model that answers with `N` snapshots or decoys is a model of capacity
  `N`, so all of the above applies to sampling protocols verbatim.

## 15. Ensemble refinement: maximum entropy is well posed, but the prior never washes out

`RequestProject/MaxEnt.lean` formalises minimum-relative-entropy reweighting of a reference
ensemble against `r` linear restraints (SAXS, NMR, FRET averages):

* `MaxEnt.relEnt_pythagoras` — the **Pythagorean identity** `KL(p‖q) = KL(p‖t) + KL(t‖q)`
  for every `p` matching the same restraints as the exponential tilt `t`;
* `MaxEnt.tilt_is_min`, `MaxEnt.tilt_unique` — hence the tilt is the *unique*
  minimum-relative-entropy ensemble consistent with the data: refinement is well posed once
  a reference ensemble is fixed;
* `MaxEnt.tilt_tilt` — **sequential refinement equals joint refinement**: the exponential
  family is closed, so fitting one experiment after another (with re-optimised multipliers)
  equals fitting them together, and the order of experiments is irrelevant;
* `MaxEnt.maxent_full_support` — a reweighted ensemble keeps *every* library conformation
  populated, so restraints can never switch a conformation off; incompatibility must be
  expressed by choosing the library (the same phenomenon as `no_energy_model_excludes`);
* `MaxEnt.maxent_no_data`, `MaxEnt.maxent_prior_dependence` — where the data are silent the
  maximum-entropy answer *is* the prior. The underdetermination of §9 is not removed by the
  maximum-entropy principle; it is relocated into the reference ensemble, which must be
  reported as part of the model's output.

## 16. Every architectural invariance has a price

`RequestProject/Invariance.lean` proves one master bound and reads several architectural
failures off it.

* `invariance_cost`, `invariance_cost_pair` — if a predictor returns observationally equal
  answers on two inputs, then on one of them its ℓ¹ error is at least **half the distance
  between the two targets**. `approx_of_invariant` is the contrapositive form.
* `receptive_field_error`, `distal_switch_error` — **locality**. A predictor whose output
  depends only on the residues in a window `S` misplaces at least half of the population
  whenever the conformational state is switched by a single residue outside `S`. A bounded
  receptive field is not an approximation issue; it is provably wrong on long-range
  coupling, and only enlarging the field helps.
* `composition_blind_error` — **patterning beats composition**. Two sequences of identical
  composition can have different ensembles, so a model reading only net charge or residue
  fractions is off by at least `1` in ℓ¹ on one of them: sequence patterning must be an
  input feature.
* `context_blind_error` — the quantitative version of context blindness (free vs bound,
  unmodified vs phosphorylated).
* `no_two_tower_model` — two independent per-segment heads produce a product distribution
  and cannot represent inter-segment correlation, however each head is computed.

## 17. The specification, quantified

`quantitative_design_laws` bundles the six laws in one theorem: measure error in population
space; size the model linearly in the number of populated conformations and exponentially
in the conformational entropy; refine by maximum entropy and report the prior; pay for
every invariance you build in; and carry inter-segment coupling in a shared latent
variable. Together with `model_must_be` and `model_cannot_be` of Part II, this is a
complete, architecture-free specification of what it takes to capture an intrinsically
disordered region — and a set of numbers against which any proposed model can be checked.

## 18. …but the loss must also see the geometry

The ℓ¹ laws of §13–14 count populations; they are blind to how *similar* the predicted
structures are to the true ones. `RequestProject/Transport.lean` adds the transport
(earth-mover) cost of an arbitrary structural dissimilarity `c` — RMSD, a contact-map
distance, whatever structures are compared with:

* `transportCost`, `IsCoupling`, `planCost` — the cost of the cheapest transport plan
  between the predicted and the true ensemble; `transportCost_le_of_coupling` and
  `transportCost_le_indep` make every explicit matching a certificate, and
  `transportCost_nonneg` records positivity.
* `transportCost_dirac` — between two single structures the transport cost is exactly their
  structural dissimilarity, so it does see the geometry.
* `ell1_geometry_blind` — two ensembles can sit at the **maximal** ℓ¹ distance `2` while
  their transport cost is as small as one likes. A model whose predicted structures are all
  but identical to the true ones gets no credit at all in ℓ¹.

The consequence is not that the ℓ¹ results are wrong — they are lower bounds on error and
remain valid, and the capacity budget must still be set by them — but that the objective an
ensemble model is *trained* and *scored* on must be a transport cost against a structural
dissimilarity, with the ℓ¹/entropy laws used to size the model.

---

# Part IV — the physics, the measurement process, and a worked disordered polymer

Parts I–III are informational: they say what object a model of an intrinsically disordered
region must output, what it can never be, and how big it must be in population space.
Part IV adds the two things a physicist would ask for next — the *thermodynamics* that
decides whether a region is disordered at all, and the *measurement theory* that decides
what can be learned about it — and then computes every one of the abstract laws for the
standard physical caricature of a disordered region, the freely jointed chain. Everything
below is proved in Lean with no `sorry` and only Lean's standard axioms. The capstone is
`physical_design_laws` in `RequestProject/PartFour.lean`.

## 19. The variational principle: what a model is trained to minimise

`RequestProject/FreeEnergy.lean`. Over a conformational library with energies `U` and
inverse temperature `β`, the variational free energy of a candidate ensemble `p` is
`F[p] = ⟨U⟩_p − S[p]/β`.

* `freeEnergy_eq` — the exact decomposition `F[p] = −(1/β) log Z + (1/β)·KL(p‖Boltzmann)`.
* `freeEnergy_ge`, `freeEnergy_boltz`, `freeEnergy_min_unique` — **Gibbs' variational
  principle**: `F` is minimised at the Boltzmann ensemble, at value `−(1/β) log Z`, and
  nowhere else.
* `freeEnergy_gap` — the excess free energy of a model is *exactly* `1/β` times its relative
  entropy to the truth. So minimising variational free energy over any class of models is
  the same optimisation as minimising KL divergence to the true ensemble: the objective
  itself is unbiased, and everything a variational fit gets wrong is the model class.

## 20. Why a disordered region is disordered: entropy has to be paid for

`folded_needs_entropic_gap`. If one conformation is to carry at least half of the Boltzmann
population while `|D|` competitors have energy at most `Uu`, then

    Uu − U(folded) ≥ (1/β)·log |D| .

A region with exponentially many accessible conformations can therefore be folded only by
an energy gap *linear in its length*. Folded domains pay it; disordered regions do not,
which is exactly why their prediction target is irreducibly an ensemble.
`flat_landscape_uniform` is the opposite extreme: a flat landscape gives the uniform
ensemble, maximal disorder.

## 21. Rate–distortion: how many structures, in the geometric metric

`RequestProject/Quantization.lean`. §18 showed that ℓ¹ is blind to structural similarity.
The repair is a genuine geometric capacity law. Let the structural dissimilarity `c` be a
pseudometric (`StructDist`) and let the target populate `m` substates that are pairwise at
least `Δ` apart (`Separated`).

* `card_near_le` — each structure the model carries is within `Δ/2` of at most one substate.
* `transportCost_ge_of_card_le` — a model of at most `k` structures pays transport cost at
  least `(Δ/2)·(m−k)/m`.
* `quantization_capacity` — **the rate–distortion law**: distortion `D` forces
  `k ≥ m·(1 − 2D/Δ)`.
* `bit_rate_lower_bound` — in information units, at least `log m + log(1 − 2D/Δ)` nats.

* `exists_model_of_covering` — **the matching upper bound**: an `eps`-net of the populated
  region *is* a model within transport distortion `eps`, obtained by collapsing each
  conformation onto its representative. So the number of structures a disorder model needs
  is the covering number of the populated region at the resolution demanded, up to the
  factor two between the two bounds.

Unlike the ℓ¹ bounds this cannot be evaded by predicting near-copies of the true
structures: it is a statement about geometry. `discreteDist_structDist` shows the earlier
counting bounds are the special case of the crudest dissimilarity, so the two families of
laws are one family.

## 22. A worked disordered region: the ideal chain

`RequestProject/Chain.lean`. Conformation space is `Chain N = Fin N → Bool` — one binary
torsion per bond — and the ensemble `chainEns N` is uniform over all `2^N` of them.

* `chain_endToEnd_mean = 0`, `chain_endToEnd_msd = N·b²` — the random-walk law `⟨R²⟩ = N b²`
  (proved by a bond-flipping involution that kills the cross terms). The mean structure is
  a point the chain essentially never visits, and the discrepancy grows as `b√N`.
* `chain_single_structure_floor` — every single-structure answer has squared error at least
  `N·b²`: the irreducible error of a structure prediction grows **linearly with the length
  of the disordered region**.
* `chain_entropy = N·log 2`, `chain_capacity` — an exactly correct model must carry `2^N`
  components. Libraries, decoy banks and snapshot sets do not scale; only a *generative*
  model, encoding `2^N` conformations in `O(N)` parameters, can.
* `chain_rate_distortion`, `chain_bit_rate` — and this is not an artefact of demanding
  exactness: at transport distortion `D` a model still needs `2^N(1 − 2D)` structures, i.e.
  a bit rate linear in `N`.
* `chain_folding_gap` — folding the chain would take an energy gap of order `N·kT·log 2`.
* `ideal_chain_design_laws` — the six statements bundled.

This is the whole development in numbers: for a disordered region the output object must be
a generative distribution, never an explicit list of structures.

## 23. What an experiment can resolve: contraction under measurement

`RequestProject/Channel.lean`. An experiment is a stochastic channel `K` from conformation
space to outcome space (a SAXS profile, an NOE, a FRET histogram; a coarse-graining map is
the deterministic case), and `push K` is the induced distribution of outcomes.

* `log_sum_ineq` — the log-sum inequality, from Gibbs' inequality.
* `dataProcessing` — **the data-processing inequality** `KL(Kp‖Kq) ≤ KL(p‖q)`: two candidate
  ensembles are never easier to tell apart after an experiment than before. No re-analysis
  of the data recovers ensemble information the experiment did not transmit.
* `ell1_push_le` — the same in the operational ℓ¹ metric of §13.
* `ell1_push_contraction` — **Dobrushin contraction**: if every conformation has probability
  at least `alpha` of producing the same outcome distribution — the insensitivity of the
  experiment — then a distance `d` between candidate ensembles arrives in the data squeezed
  to at most `(1−alpha)·d`.
* `indistinguishable_radius` — with precision `eps`, every pair of ensembles within ℓ¹
  distance `eps/(1−alpha)` is indistinguishable. An experiment defines a *ball of
  underdetermination* whose radius grows with its insensitivity, which is the quantitative
  reason an ensemble model must report its prior (§15) and why fitting an ensemble to data
  is never recovery of a unique answer.
* `blind_channel_loses_everything` — the extreme case.

## 24. How many latent states? Coupling is nonnegative rank

`RequestProject/LatentRank.lean`. Part II showed that a factorised model cannot carry
correlation and that a latent variable is what does. The quantitative form: a mixture of
`k` product ensembles is a decomposition of the two-segment population table into `k`
nonnegative rank-one terms, so the latent capacity required is the table's nonnegative rank.

* `latent_dim_lower_bound` — if the target couples the segments through `m` mutually
  exclusive substates, every mixture-of-products model needs `k ≥ m` latent states.
* `diagEns_mixture` — and `m` suffice, so the bound is exact.
* `latent_rank_theorem` — both halves for the canonical coupled ensemble. At `k = 1` this
  recovers "no factorised, mean-field or two-tower model represents coupling"; for general
  `k` it prices coupling in latent states, one per coupled substate.

## 25. The specification, with the physics in it

`physical_design_laws` (`RequestProject/PartFour.lean`) bundles the six clauses: train by
free energy (it *is* training by KL, with a unique optimum); expect disorder wherever the
energy gap does not pay for the conformational entropy; size the model by a geometric
rate–distortion law; for an `N`-bond disordered polymer expect an `N·b²` error floor for
any single structure and a `2^N` capacity requirement; remember that every experiment
contracts, leaving a ball of ensembles that fit equally well; and pay one latent state per
coupled substate. Together with `quantitative_design_laws` (Part III) and `model_must_be` /
`model_cannot_be` (Part II) this is the specification: what to output, what it costs, how to
train it, how to fit it to data, and what no model of an intrinsically disordered region
can ever be.

## 26. Coarse-grained kinetics: when is a reduced model still a model?

`RequestProject/Kinetics.lean` treats the reduction every practical kinetic model of a
disordered region performs — lumping microstates into a few macrostates (secondary-structure
classes, FRET-resolvable states, Markov-state-model clusters) and running a chain on those.

* `stationary_of_detailedBalance` — a kernel obeying detailed balance leaves the Boltzmann
  populations invariant.
* `Lumpable` — **Dynkin's lumpability condition**: microstates in the same macrostate must
  have the same total transition probability into each macrostate.
* `lump_evolve` — under it, coarse-graining commutes with the dynamics: the reduced chain
  predicts the reduced populations exactly, for *every* initial condition, which is what it
  means for the coarse variable to be Markov. `lump_stationary` — and it inherits the right
  equilibrium.
* `no_markov_coarse_graining` — the generic situation is failure: there is a three-state
  chain and a two-state partition for which **no** reduced transition matrix whatsoever
  reproduces the coarse dynamics. Two microstates that the coarse observable cannot tell
  apart have different fates, so the coarse variable has memory.

The design consequence is sharper than "the reduced rates are inaccurate": without checking
lumpability, a coarse-grained kinetic model of a disordered region is not a Markov model of
its own observable. Either verify the condition, or keep the memory — which, in the language
of §12, means modelling trajectories rather than states.

# Part V — the loss that is optimised and the data that are available

Parts I–IV describe the object a model must output and the physics it must respect. Part V
is about the two things a practitioner actually controls: the loss the model is trained on,
and the number of conformations available to train and validate it.

## 27. The training loss controls the operational error

`RequestProject/Pinsker.lean`. §13 shows that the operational error of a disorder model is
the population-space ℓ¹ distance, but no one trains on it: training minimises relative
entropy (maximum likelihood, variational free energy, §19). The connection is **Pinsker's
inequality**, proved here from scratch on a finite conformational library.

* `log_pointwise` — the pointwise inequality `(3/2)(p−q)²/(p+2q) ≤ p log(p/q) − p + q`,
  obtained by a two-level derivative argument (`Fbridge`, `Hbridge`).
* `pinskerG` — summing it and applying Cauchy–Schwarz gives `‖p − q‖₁² ≤ 2 KL(p‖q)` on an
  arbitrary finite conformation space, `pinsker` on a library `Fin m`.
* `ell1_le_of_kl_le` — so training to relative entropy `eps²/2` certifies operational error
  `eps`: *every* capacity, resolution and data law proved earlier applies verbatim to a
  KL-trained model.
* `ell1_le_of_freeEnergy_gap` — the force-field version, via §19: an excess variational free
  energy `ΔF` certifies `‖p − q‖₁ ≤ sqrt (2 β ΔF)`. A model that is thermodynamically good
  is operationally good, at a square-root rate.

## 28. Reweighting has an exact price

`RequestProject/Reweighting.lean`. Refining a simulated ensemble by importance weights — the
standard move in ensemble refinement (§15) — costs sample size, and the cost is exactly
computable.

* `essFrac_eq` — **Kish's effective sample size is exactly `N/(1 + χ²)`**, with `χ²` the
  chi-squared divergence of target from reference.
* `kl_le_log_one_add_chiSq`, `essFrac_le_exp_neg_kl` — hence `essFrac ≤ exp(−KL)`:
  *reweighting a trajectory by `K` nats costs a factor `e^K` in simulation length*
  (`frames_needed`).
* `ell1_sq_le_chiSq` — the cost is already visible in the operational metric.
* `chiSq_unif_dirac` — moving a maximally disordered reference onto one conformation costs
  `χ² = m − 1`: linear in the library, i.e. exponential in the length of the region.

## 29. Response is fluctuation, and a model is a certified free energy

`RequestProject/Response.lean` develops the exponentially tilted family — the perturbation
of an ensemble by a coupling `A` of strength `lam`, which by `tilted_eq_boltz_of_unif` is
exactly a Boltzmann reweighting of a landscape.

* `linear_response` — **the fluctuation–response theorem**: `d⟨f⟩/dλ = Cov(f, A)` for every
  observable. Susceptibilities to ligands, crowders, denaturants and modifications are
  properties of the *unperturbed* fluctuations.
* `rigid_no_response` / `response_of_disordered` — a region responds to a perturbation
  exactly to the extent that it fluctuates along it. Conformational entropy is the resource;
  a model that misrepresents correlations mispredicts all response behaviour, by exactly the
  covariance error.
* `bogoliubov`, `bogoliubov_gap` — the Gibbs–Bogoliubov–Feynman inequality, with the slack
  identified as `(1/β)·KL`: an approximate model is a *certified* upper bound on the true
  free energy, and its thermodynamic penalty is its information-theoretic error.

## 30. How much data? A lower bound no estimator escapes

`RequestProject/SampleComplexity.lean`. Let an estimator see `n` independent conformations
drawn from the true ensemble and return a population vector.

* `l1_prodP_le` — tensorisation: `n` draws separate two ensembles by at most `n` times their
  single-draw distance.
* `le_cam` — **Le Cam's two-point bound**: the summed risk at two ensembles is at least
  `‖p − q‖₁ (1 − ‖p^{⊗n} − q^{⊗n}‖₁/2)`.
* `sample_complexity_two_point`, `exists_hard_pair` — hence `eps`-accuracy on a pair at
  distance `4 eps` forces `n ≥ 1/(4 eps)`, and such pairs exist on any library with two
  conformations. No architecture, prior or amount of computation evades it.
* `support_honest_needs_coverage` — and a model that can only re-emit conformations it has
  *seen* (a retrieval model, a fixed structural pool, a nearest-neighbour ensemble) needs
  `n ≥ m(1 − eps)` frames on a maximally disordered target: it must observe essentially the
  whole library, which is exponentially large in the length of the region.

## 31. And the data suffice: the empirical ensemble

`RequestProject/Estimation.lean`. A lower bound alone is a half-theory; it leaves open
whether the obstruction is information or ingenuity. It is information.

* `expect_coord`, `expect_coord_pair` — the independence calculus of the `n`-fold product
  law.
* `expect_emp`, `variance_emp` — the empirical populations are unbiased with the exact
  multinomial variance `p(1 − p)/n`.
* `empirical_risk_le` — **just counting the sampled conformations attains expected ℓ¹ risk
  `sqrt (m/n)`**, so `n = m/eps²` frames suffice (`ensemble_sample_complexity`).

## 32. The sharp rate: quadratic in the tolerance

`RequestProject/SharpBound.lean` closes the gap between §30 (`1/eps`) and §31 (`m/eps²`) on
the tolerance.

* `klG_prodP` — relative entropy is **additive** over independent frames.
* `tv_prodP_le` — with Pinsker (§27), `‖p^{⊗n} − q^{⊗n}‖₁ ≤ sqrt (2 n KL(p‖q))`, replacing
  the crude linear tensorisation.
* `bernoulli_kl_le` — the hard pair `(1/2 ± tau)` has relative entropy at most `24 tau²`:
  nearby ensembles are *quadratically* hard to tell apart.
* `sharp_sample_complexity` — therefore every estimator, of any kind, needs
  `n ≥ 1/(48 eps²)` frames to be `eps`-accurate on both members of the pair. Together with
  §31 the data cost of an ensemble is `Θ(1/eps²)`, with a library-size factor between the
  two sides: **halving the tolerance quadruples the data**, and the library size multiplies
  it.

## 33. The specification, with the loss and the data in it

`statistical_design_laws` (`RequestProject/PartFive.lean`) bundles the eight clauses of Part
V: the loss controls the error; reweighting has an exact exponential price; response is
fluctuation; a tractable model is a certified free energy; data are needed unconditionally,
at the sharp quadratic rate; counting already achieves the order; and a model that can only
reuse observed conformations must observe nearly all of them.

# Part VI — precision: what a disordered ensemble can say about its conditions

Parts I–V ask how well an ensemble can be *learned*. Most experiments on disorder ask the
inverse question: given the ensemble, how precisely can the *conditions* be read off it — a
ligand activity, a denaturant concentration, a phosphorylation state, a crowding fugacity?
Each enters the statistical mechanics as the strength of a coupling in the tilted family of
§29, so this is parameter estimation, and it has a precision limit.

## 34. Information is fluctuation

`RequestProject/Fisher.lean`.

* `fisher` — the Fisher information a single conformation carries about the perturbation
  strength, and `fisher_eq_susceptibility`: it is the variance of the conjugate observable,
  which by §29 is also the susceptibility. **Response, fluctuation and information are one
  quantity**, so the same modelling error corrupts all three.
* `logPart_convexOn` — the log-partition function is convex in the coupling: the tilted
  family is a regular exponential family and the susceptibility can never be negative.

## 35. The Cramér–Rao bound, and the blindness of rigidity

* `cov_eq_one_of_unbiased` — unbiasedness forces the estimator's covariance with the
  coupling to be exactly `1` (the fluctuation–response theorem read backwards).
* `cramer_rao`, `cramer_rao_var_ge` — with Cauchy–Schwarz, **every unbiased estimate of the
  perturbation strength from a conformation has variance at least `1/fisher`**: precision
  about the context is bought with conformational disorder.
* `no_unbiased_estimator_of_rigid` — if the coupling cannot move the region, *no unbiased
  estimator exists at all*: no architecture, no data set, no amount of computation. The
  disorder is not noise around a structure; it is the entire measurement channel from
  context to observation.
* `fisher_pos_of_disordered` — and any two conformations differing along the coupling
  already give strictly positive information.

## 36. Assumption-free, and with many frames

* `chapman_robbins`, `context_discrimination` — for **any** readout of a conformation
  whatsoever — a FRET efficiency, a radius of gyration, a chemical shift, the output of a
  trained classifier — the shift between two contexts is at most
  `sqrt (Var · χ²)`. No unbiasedness, differentiability or model assumption is used.
* `chiSqG_prodP` (`RequestProject/Repeats.lean`) — **the chi-squared cost tensorises
  multiplicatively**: `1 + χ²` is raised to the `n`-th power by `n` independent frames, the
  exact counterpart of the additivity of relative entropy in §32, and the reason the
  effective sample size of a reweighted trajectory (§28) decays geometrically.
* `frames_needed_to_discriminate`, `log_frames_needed` — inverting it, separating two
  contexts by `delta` with a readout that fluctuates by `V` needs
  `n ≥ log(1 + delta²/V) / log(1 + χ²)` frames, i.e. `n ≍ 1/χ²` for nearby contexts.
  **Detection and reweighting are reciprocal**: the cost of moving a simulated ensemble from
  one context to the other is exactly what governs how many frames an experiment needs to
  tell them apart.

## 37. The specification, complete

`precision_design_laws` (`RequestProject/PartSix.lean`) bundles Part VI: information is
fluctuation; the free energy is convex in the coupling; Cramér–Rao; rigidity is blindness
and disorder is sensitivity; Chapman–Robbins without assumptions; multiplicative
tensorisation of the chi-squared cost; and the reciprocity of detection and reweighting.

With `model_must_be` and `model_cannot_be` (Part II), `quantitative_design_laws` (Part III),
`physical_design_laws` (Part IV) and `statistical_design_laws` (Part V), the specification is
now complete on all six axes: what the model must output, what it can never be, how its
error and capacity are measured, what the physics and the experiment impose, what the loss
and the data cost, and how precisely the resulting model can ever report on the conditions
its disordered region is in.

# Part VII — collective behaviour: condensates, multivalency, and two-phase samples

Parts I–VI treat the disordered region one molecule at a time: the object to be modelled is
a distribution over the conformations of a single chain in a fixed context. Much of the
biology of intrinsically disordered regions is collective — multivalent disordered proteins
condense into dense phases, and their binding curves are switches rather than graded
isotherms. Part VII adds that axis, and its point for a designer is a negative one: the
collective behaviour is carried by exactly the information a single-chain model does not
contain.

## 38. Demixing is the failure of convexity

`RequestProject/Condensate.lean`.

* `PhaseSeparates` — a system whose homogeneous free-energy density is `f` demixes at overall
  composition `c` when two distinct compositions that mix back to `c` (mass is conserved)
  have a strictly lower total free energy.
* `not_phaseSeparates_of_convexOn` and `exists_phaseSeparates_iff_not_convexOn` — **a convex
  free-energy density never demixes, and every failure of convexity is a demixing
  composition.** The common-tangent construction, stated as an equivalence: the quantity a
  model must predict, if it is to predict condensation, is the *curvature* of the free
  energy in the concentration.
* `lever_rule` — once the coexisting compositions are fixed, the phase fractions are forced:
  `t = (c₂ − c)/(c₂ − c₁)`. No freedom to fit remains at that level.

## 39. The single-chain ensemble cannot decide whether a condensate forms

* `phaseSeparates_add_affine` — **the phase diagram is invariant under adding any affine
  function of the concentration.** The free energy of the isolated chain enters the
  free-energy density proportionally to the amount of material, i.e. affinely.
* `chain_free_energy_blind_to_demixing` — hence two systems with exactly the same
  single-chain thermodynamics, differing only in their interchain coupling, sit on opposite
  sides of the phase boundary. However exactly a model reproduces the ensemble of the
  isolated chain, it says nothing about condensation.
* `floryFE`, `floryFE_convexOn`, `no_demixing_of_weak_coupling`, `flory_demixes` — the worked
  critical point: the Flory–Huggins density is convex for coupling `chi ≤ 2` (its curvature
  is `1/c + 1/(1−c) − 2·chi ≥ 4 − 2·chi`), so the solution is stable at every composition,
  while at `chi = 4` the half-filled solution demixes because `1 − log 2 > 0`.

## 40. Multivalency: why a switch needs coupled sites

`RequestProject/Valence.lean`. Everything is read off the binding polynomial through the
thermodynamic occupancy `occupancy Z x = x·Z′(x)/Z(x)`.

* `occupancy_independent`, `perSite_occupancy_independent` — `n` independent motifs give
  `Z = (1+x)^n` and mean occupancy `n·x/(1+x)`: **the per-site curve is the Langmuir
  isotherm at every valence.** Adding motifs multiplies the amount bound; it does not
  sharpen the response.
* `occupancy_allOrNone`, `hill_log`, `hill_slope` — fully coupled motifs give `Z = 1 + x^n`,
  and the Hill plot of `x^n/(1+x^n)` is the straight line `n·log x`: **Hill coefficient
  exactly `n`**, against exactly `1` for independent sites (`hill_slope_independent`).
* `no_cooperativity_from_independent_sites` — consequently, for valence at least two, no
  independent-site model reproduces the coupled response: cooperativity is a statement about
  the coupling between motifs, the same structure that Part II proves a factorised model
  cannot carry.
* `ligand_window_independent`, `ligand_window_coupled`, `ligand_window_tendsto_one` — the
  quantitative form: independent sites need an **81-fold** change in activity to go from 10%
  to 90% bound; coupled sites need only `81^{1/n}`-fold, and that window tends to `1` with
  the valence. Multivalency is what turns a graded isotherm into the switch a sharp
  concentration threshold requires.

## 41. What a bulk experiment sees when the sample has two phases

`RequestProject/TwoPhase.lean` — where Part VII rejoins the ensemble theory of Parts I–VI.

* `Ens.prob_mix` (with `Ens.expect_mix`) — **the bulk-measurement law**: every observable of
  a two-phase sample is the mass-weighted average of its values in the two phases.
* `Ens.ell1_mix_left`, `Ens.ell1_mix_right`, `Ens.bulk_fit_error` — the mixture lies on the
  segment between the phase ensembles, at `ℓ¹` distance `(1−t)·d` from one and `t·d` from
  the other, the two errors summing to the full separation `d`. Accuracy on one phase is
  bought at the price of the other.
* `Ens.no_single_ensemble_fits_both_phases` — and this is unavoidable: *any* single
  ensemble, fitted by any means, is off by at least `d/2` on one of the two phases. A model
  of a condensing disordered protein must be conditional on the local concentration.
* `bulk_cannot_detect_demixing` — worse, averaged data do not even reveal that there are two
  phases: a demixed sample and a homogeneous sample with the averaged populations are
  observationally identical. Phase structure is a latent variable, and only spatially
  resolved or single-molecule data expose it — the latent capacity priced in §24.

## 42. The specification, with the collective behaviour in it

`collective_design_laws` (`RequestProject/PartSeven.lean`) bundles Part VII: demixing is
non-convexity; the lever rule; affine — hence single-chain — blindness; the Flory–Huggins
critical point; cooperativity requires coupled motifs and sharpens with valence; the
bulk-measurement law with the exact cost of fitting one ensemble to two phases; and the
invisibility of demixing to averaged data.

Together with `model_must_be` and `model_cannot_be` (Part II), `quantitative_design_laws`
(Part III), `physical_design_laws` (Part IV), `statistical_design_laws` (Part V) and
`precision_design_laws` (Part VI), the specification now also covers what a model must
contain in order to say anything about the collective behaviour — condensation and
switch-like binding — of the disordered region it describes.

# Part VIII — time: the cost of obtaining the data

Parts I–VII treat the target as an equilibrium ensemble and the training data as independent
draws from it. Neither is free. A disordered region whose states are separated by a barrier
is sampled only after the corresponding relaxation time, and the frames of a trajectory are
not independent until they are separated by that same time. Part VIII prices both on the
smallest system that has them, a two-state exchange, in `RequestProject/Relaxation.lean`.

## 43. Relaxation, and the run length it forces

* `twoStep`, `twoStep_iterate_sub_stat` — the exact solution: the deviation from the
  stationary population `b/(a+b)` decays as `lam^t` with `lam = 1 − a − b`.
* `relaxation_lower_bound` — **the run must be at least as long as the relaxation time.** A
  trajectory of `t` steps started a distance `d` from equilibrium that reports the
  populations to accuracy `eps` satisfies `1 − eps/d ≤ t·(1 − lam)`. No estimator,
  smoothing or reweighting scheme applied to the same trajectory evades it: the trajectory
  has simply not visited the other state.
* `barrier_cost` — writing the exchange rate of an activated process as `1 − lam = exp(−B)`
  for a barrier `B` in units of `kT`, the required run length is at least
  `(1 − eps/d)·exp B`: **the sampling cost is exponential in the barrier height.**
* `ell1_error_of_unequilibrated` — a too-short run is not merely slow to converge; its
  populations are at `ℓ¹` distance `2·lam^t·d` from the truth, so it enters the error budget
  of Part III on the same footing as a modelling error.

## 44. Frames are not samples

* `autocorrelation` — the two-time correlation of the state decays at the same rate,
  `pi(1−pi)·lam^t`.
* `frames_correlated_within_relaxation_time` — at lags shorter than half a relaxation time
  at least half the variance survives in the covariance. The **independent** frames counted
  by the tensorisation law of §36 are therefore `T/(2·tau)`, not the number of stored
  snapshots; the sample-complexity bounds of Part V must be read in those units.

## 45. The specification, with the cost of the data in it

`temporal_design_laws` (`RequestProject/PartEight.lean`) bundles Part VIII: exact
relaxation; the run-length lower bound; the exponential cost of a barrier; the `ℓ¹` error of
an unequilibrated run; and the decay of the autocorrelation with the counting rule it
imposes on frames.

# Part IX — physical realism: chains, experiments, charges, temperature, dynamics

Parts I–VIII are architecture-free and information-theoretic. They constrain *what kind of
object* a model of a disordered region must be, how big, how well fitted, and how expensive
the data are — but they never mention a polypeptide in water. Part IX supplies the physics
in the form the earlier parts can consume: exact chain statistics, exact forward models of
the experiments that actually constrain disordered ensembles, an explicit screened
electrostatic Hamiltonian, the temperature axis, the excluded-volume balance that fixes the
size exponent, and the chain dynamics that fixes the sampling cost. Everything below is
proved with no continuum, small-parameter or large-`N` approximation; where an asymptotic
statement is made, it is stated as a limit theorem with the finite-`N` identity behind it.

## 46. Real chain statistics: stiffness, persistence length, radius of gyration

`RequestProject/Polymer.lean`. The physical input is the defining property of the
worm-like / freely-rotating chain: the tangent correlation decays exponentially in contour
separation, `⟪u_i,u_j⟫ = a^{|i−j|}` — equivalently `exp(−s/l_p)` with the persistence length
`l_p = −b/log a` (`corr_eq_exp_of_persistence`, `persistenceLength_pos`).

* `corrSum_closed_form`, `msd_eq` — the exact discrete worm-like-chain formula
  `⟨R²⟩ = b²[N(1+a)/(1−a) − 2a(1−a^N)/(1−a)²]`, valid at every `N`.
* `msd_ideal`, `msd_rod`, `msd_mono_stiffness`, `msd_bounds` — the random walk `N b²` and the
  rigid rod `N²b²` are its two limits, and the chain swells monotonically with stiffness in
  between.
* `kuhn_length_limit` — `⟨R²⟩/N → b²(1+a)/(1−a)`: on long scales a stiff chain is a random
  walk with a renormalised Kuhn step, which is precisely why global size measurements cannot
  see local stiffness.
* `stiffness_bondlength_degeneracy` — the design consequence: for *any* two stiffnesses there
  are bond lengths giving exactly the same `⟨R²⟩`. One global number (a SAXS `Rg`, one FRET
  efficiency) constrains a product, never its factors.
* `gyration_eq_pair_sum` — the exact identity `Rg² = (1/2N²)Σ_{ij}|r_i−r_j|²` in any real
  inner-product space: the radius of gyration *is* a pair statistic.
* `pair_dist_sum`, `ideal_gyration`, `ideal_gyration_ratio`, `ideal_gyration_limit` — for the
  ideal chain `⟨Rg²⟩ = b²(N²−1)/(6N)` exactly, with the classical `⟨R²⟩/6` as the limit.

## 47. What the experiments measure: SAXS

`RequestProject/Observables.lean`. `debye` is the Debye scattering function of `N` point
scatterers, `I(q)/I(0) = (1/N²)Σ_{ij} sinc(q r_ij)`.

* `debye_isometry_invariant`, `debye_perm_invariant` — the curve depends on the configuration
  only through its unordered set of pair distances. Relabelling the chain changes nothing:
  SAXS carries no information about which residue is where.
* `sinc_taylor_bound`, `guinier` — the Guinier law *with an explicit error bound*:
  `|I(q)/I(0) − (1 − q²Rg²/3)| ≤ (5/96)(qD)³` whenever `q·D ≤ 1`. In the Guinier regime the
  entire experiment is one number.

## 48. What the experiments measure: `r^{-6}` averaging and FRET

* `preApparent_le_of_weight` — a conformer of weight `w` at distance `d` forces the apparent
  PRE/NOE distance below `w^{-1/6}·d`, *whatever the rest of the ensemble does*: a 1%
  population at 15 Å pins the reading below 32 Å.
* `preApparent_ge_min` — and it is never below the closest approach. The observable is a
  minority report on the compact members of the ensemble, not an average.
* `preApparent_le_mean` — Jensen: the apparent distance never exceeds the mean distance, so
  an `r^{-6}` restraint read as a mean distance always makes the model too compact.
* `fret_ensemble_ne_mean` — an explicit symmetric two-state ensemble whose mean distance is
  exactly `R₀` transfers with efficiency strictly above `1/2 = fretEff 1`. Inverting a
  measured efficiency returns something that is not the mean distance, and the discrepancy is
  a property of the *width* of the ensemble.

The common conclusion: restraints must be applied by forward-modelling the observable from
the candidate ensemble. The exact forward models are the ones proved here.

## 49. Charge patterning and salt

`RequestProject/Electrostatics.lean`. The mean-field Debye–Hückel energy of a charge sequence
on a Gaussian chain, `E = Σ_{i<j} q_i q_j exp(−κ b√|i−j|)/(b√|i−j|)` (`screenedEnergy`),
together with the sequence charge decoration `scd`.

* `netCharge_perm_invariant`, `absCharge_perm_invariant` — net charge and composition are
  blind to the pattern.
* `scd_blocky_lt_alternating` — the blocky `(+ + − −)` and alternating `(+ − + −)`
  arrangements of the same composition differ in charge decoration by exactly `4 − 4√2 < 0`,
  the blocky sequence lower (more compact).
* `screenedEnergy_blocky_sub_alternating`, `screenedEnergy_perm_not_invariant` — and their
  Debye–Hückel *energies* differ, by `(4 − 2√2)/b`. This is the Hamiltonian-level instance of
  the abstract composition-blindness bound of Part III.
* `screenedEnergy_abs_le`, `screening_tendsto_zero`, `patterning_washed_out` — every
  electrostatic energy, and every difference between two sequences, is bounded by
  `(Σ|q_i|)²exp(−κb)/b` and vanishes at high salt. Charge patterning is a salt-tunable
  effect: a model fitted at one ionic strength is not a model of the region.

## 50. Temperature: heat capacity, van 't Hoff, cold denaturation

`RequestProject/Thermo.lean`. Taking the inverse temperature as the tilting parameter of
Part V makes the Boltzmann family an exponential family in `β`.

* `hasDerivAt_meanE`, `heatCapacity_eq_variance`, `heatCapacity_nonneg` — `d⟨E⟩/dβ = −Var(E)`
  and `C/k = β²Var(E) ≥ 0`: thermodynamic stability as an identity.
* `meanE_antitone` — the mean energy rises with temperature.
* `rigid_zero_heatCapacity`, `heatCapacity_pos_of_disordered` — a single-conformation model
  predicts *zero* heat capacity, while any library with two distinct energies has a strictly
  positive one. Calorimetry measures exactly the fluctuation that makes the region
  disordered.
* `log_pop_ratio`, `vantHoff` — `log(p_i/p_j) = −β(E_i−E_j)` exactly: a two-state analysis
  returns an energy gap, never a structure.
* `gibbsHelmholtz_strictConcaveOn`, `denaturation_two_temperatures`,
  `no_three_transition_temperatures` — with a positive `ΔCp` the Gibbs–Helmholtz stability
  curve is strictly concave, so an ordered state that is stable somewhere must melt on
  *both* sides: **cold denaturation is forced**, and there are never three transition
  temperatures.

## 51. Excluded volume and the Flory exponent

`RequestProject/Flory.lean`. The Flory free energy `F(R) = aR²/N + vN²/R³`.

* `floryFreeEnergy_min` — a unique, strict global minimiser on `(0,∞)`, proved by the exact
  factorisation `3R⁵ − 5R*²R³ + 2R*⁵ = (R−R*)²(3R³+6R*R²+4R*²R+2R*³)`.
* `floryRadius_pow_five`, `floryRadius_eq` — the minimiser is `R* = (3v/2a)^{1/5}N^{3/5}`:
  the swollen-coil exponent `ν = 3/5`.
* `flory_swelling` — `R*/√N → ∞` like `N^{1/10}`: no ideal chain of any bond length
  reproduces the size of a long self-avoiding chain.
* `flory_theta` — at the theta point the repulsive term disappears; the exponent is a
  property of the solvent, not of the sequence alone.

## 52. Chain dynamics: the Rouse spectrum and the cost of a chain

`RequestProject/Rouse.lean`. Part VIII priced sampling in relaxation times but left the
relaxation time free; connectivity fixes it.

* `rouseMode_eigen`, `rouseMode_boundary_left`, `rouseMode_boundary_right` — the modes
  `cos(pπ(n+1/2)/N)` diagonalise the free-end chain Laplacian exactly, with eigenvalues
  `λ_p = 4sin²(pπ/2N)`.
* `rouse_slowest_eigenvalue_le` — `λ₁ ≤ π²/N²`, the Rouse `τ ∼ N²` law.
* `rouse_run_length` — fed back into the Part VIII bound: a run reporting populations to
  accuracy `eps` from a start `d` away needs at least `(1 − eps/d)N²/π²` steps. Equilibration
  cost grows at least quadratically with the length of the region *before* any barrier, and
  `barrier_cost` then multiplies it by `exp B`.

## 53. The specification, made physical

`physical_realism_design_laws` (`RequestProject/PartNine.lean`) bundles Part IX in seven
clauses: exact worm-like-chain statistics; scattering as a relabelling-invariant pair
statistic with the quantitative Guinier law; the `r^{-6}` bias bounds; charge patterning and
its salt screening; heat capacity as fluctuation with forced cold denaturation; the exact
Rouse spectrum and the quadratic equilibration cost; and the Flory minimum with `ν = 3/5`.

Read with `model_must_be`, `model_cannot_be`, `quantitative_design_laws`,
`physical_design_laws`, `statistical_design_laws`, `precision_design_laws`,
`collective_design_laws` and `temporal_design_laws`, the specification is now physical: the
object to be predicted is a temperature-, salt- and sequence-pattern-dependent conformational
distribution over a chain whose size exponent is set by solvent quality and whose
equilibration time grows with its length; and the data that constrain it are nonlinear
functionals of that distribution, with the exact forms — and the exact biases — proved here.

---

# Part X — The measured solution state

Part IX made the chain physical. Part X makes the *measurement* physical: the two large
classes of solution observable not yet covered (orientational NMR data, and translational
diffusion), and the one piece of structural physics that separates a disordered region from
a featureless random coil — transient, cooperative secondary structure.

## 54. Orientational order: `S²` and residual dipolar couplings

`RequestProject/NMR.lean`. Bond directions are unit vectors `u_k` of three-space carried by
the conformers, with weights `w_k`.

* `orderTensor`, `orderParam` — the second-rank order tensor `M_{ab} = Σ_k w_k u_{ka}u_{kb}`
  and the Lipari–Szabo generalised order parameter `S² = (3/2)Σ_{ab}M_{ab}² − 1/2`.
* `orderParam_eq_pair` — the exact identity `S² = (3/2)Σ_{kl}w_kw_l⟨u_k,u_l⟩² − 1/2`, i.e.
  the formula by which an ensemble predicts `S²`.
* `orderTensor_trace`, `orderParam_nonneg`, `orderParam_le_one` — `0 ≤ S² ≤ 1`. The upper
  bound is Cauchy–Schwarz; the lower bound is the three-dimensionality of space, `(tr M)² ≤
  3 tr M²` with `tr M = 1`.
* `orderParam_eq_one_of_aligned` — a rigid direction saturates the bound.
* `orientational_disorder_of_orderParam_lt_one` — the usable converse: a measured `S² < 1`
  *proves* that two populated conformers carry non-parallel bond vectors. This is the
  observable that certifies disorder; no single structure can carry it.
* `rdc`, `rdc_bounds` — the residual dipolar coupling `D = D_max Σ_k w_k P₂(cos θ_k)` and its
  sharp range `−D_max/2 ≤ D ≤ D_max`.
* `rdc_cancellation` — an explicit ensemble (weight `1/3` parallel, `2/3` perpendicular to the
  alignment axis) with `D = 0` in which every populated member has a coupling of full
  magnitude, `+D_max` and `−D_max/2`. A vanishing RDC is evidence of averaging, not of an
  isotropic conformer; fitting one orientation to `D = 0` returns a structure that is nowhere
  in the ensemble.

## 55. Hydrodynamics: what a diffusion coefficient returns

`RequestProject/Hydrodynamics.lean`.

* `kirkwoodSum`, `hydroRadius` — the Kirkwood formula `1/R_h = (1/N²)Σ_{i≠j}1/r_ij`.
* `kirkwoodSum_translation_invariant`, `kirkwoodSum_perm` — like small-angle scattering,
  hydrodynamics is invariant under rigid motion *and* under relabelling the chain: a
  diffusion coefficient carries no sequence information.
* `kirkwood_cauchy_schwarz`, `hydroRadius_le_pairDist` — `(N²−N)² ≤ (Σ_{i≠j}r_ij)(Σ_{i≠j}1/r_ij)`,
  so the Kirkwood radius is a harmonic-type mean and never exceeds the arithmetic mean of the
  pair distances (with the exact combinatorial factor).
* `stokesEinstein`, `stokesEinstein_antitone`, `stokesEinstein_injective` — `D = k_BT/6πηR_h`
  is strictly decreasing in `R_h`: the measurement is exactly one number about the ensemble.
* `stokesEinstein_ensemble` — in fast exchange the measured coefficient is *exactly* the
  weight average of the conformers' coefficients, so the radius one reports is the harmonic
  mean `appRadius w R = (Σ_k w_k/R_k)⁻¹`.
* `appRadius_le_mean`, `appRadius_lt_mean_two` — Jensen, and an explicit strict instance: the
  reported hydrodynamic radius lies below the mean radius of the members, strictly so as soon
  as two populated conformers differ. Size restraints from diffusion data bias a model
  compact by exactly the heterogeneity that is being modelled.

## 56. Transient secondary structure: the helix–coil chain

`RequestProject/HelixCoil.lean`. The model is built from its configuration sum; the transfer
matrix is *derived*, not assumed.

* `weight`, `Zhead` — Boltzmann weight of a helix/coil configuration (field `h` per residue,
  nearest-neighbour coupling `J`), and the partition function of a chain with a prescribed
  first residue.
* `Zhead_succ` — the transfer-matrix identity `Z_{n+1}(b) = Σ_c e^{hσ_b+Jσ_bσ_c}Z_n(c)`,
  proved from the sum over configurations.
* `helixFraction_pos`, `helixFraction_lt_one` — at every finite propensity and temperature the
  helical population of a residue is strictly inside `(0,1)`. The right answer for a residue
  of an IDR is a number in the open unit interval, i.e. a population, not a structure.
* `helix_cooperativity` — with `J > 0` neighbouring helix/coil variables have strictly
  positive covariance while each has mean zero: the joint distribution does not factorise, so
  independent per-residue propensities are provably insufficient.

## 57. The solution-state design laws

`solution_state_design_laws` (`RequestProject/PartTen.lean`) bundles Part X in three clauses:
orientational observables are quadratic and certify disorder; diffusion is sequence-blind and
returns a harmonic mean; secondary structure is fractional and cooperative.

Read together with `physical_realism_design_laws` (Part IX) and the information-theoretic
laws of Parts I–VIII, the specification is complete in the sense that matters for practice:
the object to be predicted is a temperature-, salt- and sequence-pattern-dependent
conformational distribution, and *every* standard experiment used to constrain it — SAXS,
PRE/NOE, FRET, NMR order parameters and RDCs, translational diffusion, calorimetry — enters
through the exact nonlinear functional, with the exact bias, proved here.

---

# Part XI — Removing the idealisations: continuum, symmetry, well-posedness, learnability

Parts I–X are stated for *finite* conformational ensembles, take the existence of "the
ensemble of the region" for granted, and say nothing about generalisation from one protein
to the next. Those are exactly the three places where a reader is entitled to ask whether
the theory is about proteins or about an abstraction. Part XI closes all three.

## 58. The continuum: the true target is a measure, and the metric is forced

`RequestProject/Continuum.lean`. A finite ensemble *is* a probability measure — the mixture
of Dirac masses at its conformations (`Ens.toMeasure`), whose integrals are exactly the
`expect` used throughout (`Ens.integral_toMeasure`). So the two pictures can be compared
directly, and the comparison is stark.

* `tvDist_toMeasure_eq_one` — **every finite ensemble, of any size and with any weights, is
  at total-variation distance exactly `1` from an atomless target.** Model and truth are
  mutually singular. Since `RequestProject.Metric` identifies uniform observational error
  with total variation, the population-space error of Part III does not survive the
  continuum limit: `no_finite_ensemble_tv_approx`.
* `exists_ens_lipschitz_approx` — and yet, on a compact conformation space, for every
  resolution `eps` there is a finite ensemble reproducing *every* `L`-Lipschitz observable to
  within `L·eps`. The construction is the covering-number one: an `eps`-net of conformation
  space, each conformer carrying the mass of its cell — the rate–distortion count of
  `RequestProject.Quantization`, now derived for a continuous target.
* `continuum_dichotomy` — the two together. At every resolution there is a finite ensemble
  that is weakly `eps`-accurate *and* maximally wrong in total variation. **The choice of
  metric is therefore not a matter of taste**: a model of a disordered region must be built
  and scored against geometry-aware (Lipschitz / transport) functionals, and overlap- or
  likelihood-type scores against a continuous truth are saturated and carry no signal.

## 59. Equivariance is incompatible with single-structure output

`RequestProject/Symmetry.lean`. Every serious structural architecture is equivariant, and it
should be. But if the target ensemble is invariant under a symmetry `s` — degenerate
rotamers, a homo-repeat, a symmetric interface — then:

* `equivariant_point_is_fixed` — an equivariant predictor that returns one structure must
  return a **fixed point of `s`**;
* `no_equivariant_point_predictor` — so if `s` has no fixed point, no such predictor exists
  at all;
* `equivariant_point_prediction_unpopulated` — and in the generic case, where the fixed
  points are unpopulated, the structure returned has probability **zero** in the truth. It is
  not an inaccurate structure; it is a conformation the region never adopts.
* `Rotamer.rotamer_prediction_is_origin`, `Rotamer.rotamer_prediction_unpopulated`,
  `Rotamer.rotamer_prediction_distance` — concretely, on the three-fold rotamer triangle (the
  cube roots of unity, gauche+/trans/gauche− equally populated) the forced answer is the
  centroid, which carries no population and is a full rotamer radius (distance `1`) away from
  every conformation in the ensemble.
* `equivariant_ensemble_predictor_exists` — the obstruction is *not* caused by equivariance.
  An equivariant **ensemble**-valued predictor can be exactly right, because the space of
  ensembles contains the symmetric average and conformation space does not.

## 60. Well-posedness: the ensemble a sequence has, and the accuracy the energy must have

`RequestProject/GibbsMeasure.lean`. The target of prediction is the Boltzmann–Gibbs measure
`dmu ∝ e^{-U} dlam` of the energy `U` (in units of `kT`) that sequence and context determine,
against the flat measure `lam` of the conformational degrees of freedom.

* `Zpart_pos`, `isProbabilityMeasure_gibbs` — for a bounded measurable energy the partition
  function is strictly positive and the Gibbs ensemble is a probability measure. The
  prediction problem has an answer.
* `noAtoms_gibbs` — if the conformational degrees of freedom are continuous, so is the
  ensemble: **the physical target is atomless**, so §58 applies to it and not merely to a
  hypothetical.
* `gibbs_stability`, `gibbs_population_error` — **populations are exponential in the energy
  error**: two force fields that agree everywhere to within `d` in units of `kT` give
  populations within a factor `e^{2d}`, hence within `e^{2d} − 1` in absolute terms, and this
  is the entire accuracy budget of the enterprise. To place a population to a relative factor
  `1 + eps` the energy must be right to about `eps/2 kT`. No amount of sampling or capacity
  repairs it — the bound is on the *target* that the sampler converges to. (The matching
  lower statement, that an energy gap moves a population ratio by exactly its exponential, is
  `Thermo.log_pop_ratio` in Part IX.)

* `gibbs_reparam`, `jacobian_population_error` — **the reference measure is part of the
  model.** Changing the flat measure of the conformational degrees of freedom by a density
  `rho` is *exactly* the same as shifting the energy by `−log rho`: sampling uniformly in
  torsion space and sampling uniformly in Cartesian space are models of two different
  energies, differing by the log-Jacobian, and if that log-Jacobian varies within `d` the
  populations move by up to `e^{2d} − 1`. A Jacobian of order `kT` is a modelling error of the
  same order as the effect being modelled.

## 61. The error floor is not an artefact of discreteness

`RequestProject/ContinuumLoss.lean`. For a probability measure on any conformation space and
any structural coordinate `q` with a finite second moment:

* `coord_pointLoss_eq` — the exact bias–variance identity
  `∫ (q − a)² dmu = Var(q) + (a − ⟨q⟩)²`;
* `coordVar_pos_of_noAtoms` — the variance is **strictly positive** whenever the law of `q`
  has no atoms, which is what disorder means for a real chain;
* `single_structure_floor`, `continuum_error_floor` — hence every single-structure prediction,
  and indeed every constant value reported for the coordinate, incurs at least `Var(q) > 0`.
  The Part I floor is a property of the continuous target, not of the finite idealisation.

## 62. Learnability: covering buys generalisation, switches cost capacity

`RequestProject/Generalisation.lean`. The remaining question is whether the object specified
by Parts I–X can be *learned across proteins*. The answer is a dichotomy in the modulus of
continuity of the map `sequence-and-context ↦ ensemble`.

* `nearest_neighbour_generalisation` — if that map is `K`-Lipschitz on the observables of
  interest and the training set is a `delta`-net of input space, then even nearest-neighbour
  prediction is accurate to `K·delta` at every input, including unseen ones. **Coverage of
  sequence space, not architecture, is what buys generalisation.**
* `switch_forces_capacity`, `no_smooth_model_of_switch` — but disordered regions switch: one
  phosphorylation, one charge substitution, one partner can move the ensemble a long way. If
  two inputs at distance `d` have ensembles differing by `G` on an observable, then every
  predictor whose output varies with Lipschitz constant `L` has error at least `(G − L·d)/2`
  on one of them. Smoothness in sequence space is not free regularisation; at a switch it is
  an error floor, and the model needs modulus of continuity at least `G/d`.

## 63. The Part XI capstone

`continuum_design_laws` (`RequestProject/PartEleven.lean`) bundles §58, §60 and §61 into six
clauses about the honest target `gibbs lam U`: it exists and is a probability measure; it is
atomless; it has a strictly positive single-structure error floor; it is at total-variation
distance `1` from every finite model; it is nonetheless `eps`-approximable on all Lipschitz
observables at every resolution; and its populations are exponentially stable in the energy.
`equivariance_design_law` adds §59 as the seventh, structural, clause.

`FlatChain.flat_chain_design_laws` checks that the hypotheses are satisfiable by an entirely
standard object — the uniform distribution of one continuous conformational coordinate over a
bounded range — so none of the above is an empty implication.

Read with `model_must_be`, `model_cannot_be`, `physical_realism_design_laws` (Part IX) and
`solution_state_design_laws` (Part X), the specification is now free of its idealisations:
the object to be predicted is a *continuous, atomless, temperature-, salt- and
context-dependent* probability measure on conformation space, produced by an energy that must
be known to a fraction of `kT`; it must be output as a distribution and scored in a transport
metric; it cannot be returned as one structure even by a perfectly equivariant model; and it
can be learned across proteins exactly to the extent that sequence space is covered and
switches are given the capacity they demand.

---

# Part XII — Excluded volume, microscopically: the chain that cannot overlap

Part IX treated excluded volume the way a physicist estimates it: Flory's mean-field free
energy, an entropic spring balanced against a repulsive term, giving the swollen-coil exponent
`ν = 3/5` (`RequestProject/Flory.lean`). That is an estimate about a chain, not a theorem
about a chain that really cannot overlap — and "excluded volume makes a disordered region less
floppy than it looks" is the standard reason offered for hoping that a small library of
structures might, after all, suffice. Part XII settles the question exactly, by counting.

## 64. The general lattice chain, and why cutting it matters

`RequestProject/LatticeWalk.lean` develops the theory for *any* lattice: an abelian group `V`
of positions and a finite set `dir : Fin q → V` of bond vectors. A conformation of an `n`-bond
chain is a word `w : Fin n → Fin q`; `sites` are the positions it visits (the partial sums of
its bonds); `IsSAW` says those positions are pairwise distinct — the chain does not overlap
itself.

* `isSAW_append` — **any piece of a self-avoiding chain is self-avoiding.** This is the exact
  content of "excluded volume is inherited by subchains", and, because cutting a chain at a
  residue is injective, it gives `cntOf_submultiplicative` : `cnt (m + n) ≤ cnt m · cnt n`.
  Conformational entropy is subadditive in chain length.
* `directed_isSAW_of_hom` — a chain whose bonds all increase some additive functional cannot
  return to a position it has left, so it is automatically self-avoiding. This gives the
  matching exponential *lower* bound `pow_le_cntOf`.
* `connectiveConstantOf`, `tendsto_connectiveConstantOf` — by Fekete's lemma the
  conformational entropy per residue `μ = lim (log (cnt n))/n` therefore exists, and
  `connectiveConstantOf_le_div` says every single chain length bounds it above:
  `μ ≤ log (cnt k)/k`. A finite computation is enough to prove that excluded volume costs
  entropy.

## 65. What excluded volume does, and does not, buy

`RequestProject/SelfAvoiding.lean` (square lattice) and `RequestProject/CubicLattice.lean`
(cubic lattice — where a polypeptide actually lives) instantiate this.

* `cnt_four : cnt 4 = 100` against `4⁴ = 256`, and `cnt3_two : cnt3 2 = 30` against `6² = 36`
  — verified by decision procedure, not by simulation. With `connectiveConstant_le_div` these
  give `connectiveConstant_lt_log_four` and `connectiveConstant3_lt_log_six`: **the entropy
  per residue of a self-avoiding chain is strictly below the ideal-chain value.** Excluded
  volume removes a fixed fraction of the conformational entropy of *every* residue.
* `two_pow_le_cnt : 2 ^ n ≤ cnt n` and `three_pow_le_cnt3 : 3 ^ n ≤ cnt3 n` (the directed
  walks), hence `log_two_le_connectiveConstant` and `log_three_le_connectiveConstant3`:
  **it does not collapse the ensemble.** The number of conformations still grows
  exponentially in the length of the region.
* `saw_fraction_tendsto_zero`, `saw3_fraction_tendsto_zero` — the two facts together:
  self-avoiding conformations are an *exponentially vanishing fraction* of the freely jointed
  ones (`≤ (25/64)^{n/4}` in two dimensions, `≤ (5/6)^{n/2}` in three). A Gaussian- or
  freely-jointed-chain generator is therefore wrong about the **support** of the ensemble, not
  merely about its weights — while a model that hoped self-avoidance would shrink the answer
  to a handful of structures is wrong by an exponential factor in the other direction.

## 66. Growth dead-ends, and self-avoidance has unbounded memory

Two facts about *generating* conformations residue by residue, both proved on explicit walks.

* `exists_trapped_walk` — the seven-step walk `E N N W W S E` is self-avoiding and **none** of
  its four extensions is. Sequential growth can dead-end, so a generator that appends
  residues must be able to reject and backtrack: correctness cannot be certified one step at
  a time.
* `unbounded_memory` — for every context length `k`, the walks `W^k` and `E^k N W^k` are both
  self-avoiding and have the *same last `k` steps*, yet a south step may be appended to the
  first and not to the second. **Self-avoidance is not a finite-context property of the step
  sequence.** No autoregressive generator with a bounded context window, and no `k`-th order
  Markov chain over torsions, can be right — the microscopic counterpart of the
  receptive-field law of Part III (`RequestProject/Invariance.lean`).

## 67. The Part XII capstone

`excluded_volume_design_laws` (`RequestProject/PartTwelve.lean`) bundles seven clauses. The
capacity clauses join the count to the ensemble theory of Part III: `sawEns n` is the athermal
(uniform) ensemble of the self-avoiding chain, `saw_exact_capacity` shows an exactly correct
model carries at least `cnt n ≥ 2^n` components, and `saw_capacity_lower_bound` (with its
three-dimensional twin `saw3_capacity_lower_bound`) shows that accuracy `eps` in the
population metric still costs `2^n (1 − eps)` — respectively `3^n (1 − eps)` — components.
`saw_folding_gap` adds the thermodynamic clause: to give one conformation half the Boltzmann
population, the force field must supply a gap of `(1/β)·log (cnt n − 1)`, an energy linear in
the length of the region, exactly as for the ideal chain of Part IV.

So excluded volume — the most physical objection one can raise to the idealised chains of
Parts IV and IX — changes the constant in the exponent and nothing else. The target of
prediction for an intrinsically disordered region remains a distribution with exponentially
many populated conformations, it must still be produced generatively rather than enumerated,
and the generator may not be a bounded-context sequential sampler.

# Part XIII — From a microscopic sequence Hamiltonian

## 68. Why another part

Everything up to Part XII takes the target ensemble as given, or as the Gibbs measure of an
abstract energy. The physically interesting question is prior to that: *given a sequence and a
force field, is the region ordered or disordered?* Part XIII answers it inside the formal
development, from an explicit microscopic Hamiltonian, with no mean-field step anywhere.

The Hamiltonian is the standard hydrophobic/polar contact energy on the self-avoiding chains of
Part XII (`RequestProject/HPModel.lean`). A conformation is a self-avoiding walk; residue `i`
sits at `posOf dir w i`, and `posOf_injective` is excluded volume in the form needed here —
distinct residues occupy distinct sites. `contacts dir seq w` collects the ordered pairs of
residues that are hydrophobic (`seq i = true`), occupy neighbouring lattice sites, and are not
bonded neighbours; `hpEnergy dir eps seq w = -eps · |contacts|`. `hpEns` is its Boltzmann
ensemble at inverse temperature `beta`, built by `IDR.Boltz.boltzEns`
(`RequestProject/Boltzmann.lean`).

## 69. Excluded volume caps the energy

`contacts_card_le`: on a lattice with `q` bond vectors, a *self-avoiding* chain of `n` bonds has
at most `(n+1)·q` contacts. The proof is the physical one: a residue has at most `q` neighbouring
sites, and no two residues share a site, so the map sending a contact to (first residue,
displacement) is injective into a set of size `(n+1)·q`. Hence `hpEnergy` lies in
`[−eps·(n+1)·q, 0]` (`hpEnergy_ge`, `hpEnergy_nonpos`): the landscape is extensive, with slope
at most `eps·q` per residue.

## 70. The ordering threshold

Two extensive quantities compete. Ordering one conformation to half the population requires an
energy gap of `(1/beta)·log (cnt n − 1)` (`folded_needs_entropic_gap`, Part IV), and the
conformational entropy of an excluded-volume chain grows at least `log r` per residue whenever
the lattice supports `r^n` conformations (`log_cnt_pred_ge`, from Part XII's directed-walk
bound). The energy can supply at most `eps·q` per residue. Comparing slopes gives
`hp_no_folding_of_growth`:

> if `2·q·beta·eps < log r`, then for every chain of at least three bonds, **every**
> hydrophobic/polar sequence and every conformation, the equilibrium population is below one
> half.

Concretely, `8·beta·eps < log 2` on the square lattice (`square_hp_no_folding`) and
`12·beta·eps < log 3` on the cubic lattice (`cubic_hp_no_folding`, the case of a real
polypeptide backbone). Below the threshold, disorder is not a failure of prediction; it is a
theorem about the Hamiltonian, uniform in the sequence. No amount of sequence design helps,
because the bound never mentions the sequence.

## 71. The criterion is sharp

The converse holds too. `IDR.Boltz.boltz_unique_ground_half` — and its HP form
`hp_folding_sufficient` — shows that a *unique* lowest-energy conformation separated from all
others by a gap with `beta·gap ≥ log (cnt n)` does hold at least half the population. So
`beta·gap ≈ log (number of conformations)` is the exact criterion for order, necessary by
Part IV and sufficient here. What makes a region intrinsically disordered is precisely the
absence of a gap on the scale of `kT` times the conformational entropy.

## 72. Two limiting cases

* **Polar regions.** With no hydrophobic residues the landscape is exactly flat
  (`hpEnergy_polar`), so the equilibrium ensemble *is* the athermal self-avoiding ensemble of
  Part XII at every temperature (`polar_ens_same_saw`), and modelling it exactly costs
  `cnt n ≥ 2^n` (in three dimensions `3^n`) mixture components (`polar_chain_capacity`).
  Cooling never narrows it.
* **Degenerate ground states.** More generally, `IDR.Boltz.ground_state_capacity` prices the
  low-temperature limit: a model within `tol` of equilibrium needs
  `|S| − tol·(|S| + cnt n · e^{−beta·gap})` components, where `S` is the set of ground states.
  A rugged landscape with an exponentially degenerate minimum is as expensive to model as a
  flat one; `T → 0` is not an escape from the capacity law of Part III.

## 73. Hydrophobic content

The threshold can be sharpened by composition, and this is where the model becomes recognisably
biological. Only hydrophobic residues supply contact energy, so `contacts_card_le_hCount`
improves the cap to `h·q`, where `h = hCount seq` counts them. Feeding this into the same
comparison gives `hp_no_folding_of_hydrophobicity` and, in terms of the hydrophobic *fraction*
`f`,

> if `2·q·f·beta·eps < log r` then no conformation of that sequence reaches half the
> population,

i.e. `8·f·beta·eps < log 2` on the square lattice
(`square_low_hydrophobicity_disordered`) and `12·f·beta·eps < log 3` on the cubic lattice
(`cubic_low_hydrophobicity_disordered`). Disorder is forced by *low mean hydrophobicity* — the
microscopic counterpart of the empirical composition rules used to annotate disordered regions
from sequence. The polar chain of §72 is the extreme case `f = 0`.

## 74. The Part XIII capstones

`IDR.sequence_hamiltonian_design_laws` (square lattice) bundles seven clauses — the energy cap,
the necessity of the entropic gap, the sequence-independent threshold stated as a bound on the
population of *every* point of conformation space, its refinement by hydrophobic content, the
sharpness of the criterion, the polar chain, and the ground-state degeneracy law. `IDR.cubic_sequence_hamiltonian_laws` states the
three-dimensional version. Both depend only on Lean's standard axioms.

The design conclusion of Part XIII is the physical justification of the specification: for a
region whose contact energies are small on the scale of `kT` per contact — which is what
"intrinsically disordered" means microscopically — the equilibrium target has no dominant
conformation at any temperature, so a predictor that outputs one structure is provably wrong
about the majority of the population, and the capacity, metric, context and data laws of
Parts I–XII apply to it in full.

# Part XIV — The evaluation: which number may a disorder model be judged by

Parts I–XIII fix what a model of an intrinsically disordered region has to *be*. Part XIV fixes
how it is to be *scored*, which is the other half of the design: a pipeline returns whatever its
score selects, so a score whose minimiser is not the truth guarantees a wrong model however good
the architecture. The development is `RequestProject/Scoring.lean`, bundled in
`RequestProject/PartFourteen.lean`.

The setting is a finite conformation library `X`. A prediction is a population vector
`p : X → ℝ` (`IDR.Scoring.IsProbVec`), the truth is another one `q`, and a score `S p y` is the
penalty when the prediction meets an observed conformation `y`; `expScore S p q = ∑_y q y · S p y`
is its expected value under the truth. `Proper` means no prediction beats the truth,
`StrictlyProper` means the truth is the *unique* minimiser.

## 75. A strictly proper ensemble score exists

The quadratic (Brier) score `brier p y = ∑_x (p x − 1{x=y})²` has the exact excess-risk identity

> `expScore brier p q − expScore brier q q = ∑_x (p x − q x)²`   (`brier_excess`),

so it is strictly proper (`brier_strictly_proper`): the penalty for being wrong is precisely the
squared error of the predicted populations. Ensembles can therefore be fitted and ranked
honestly, and cheaply.

Two consequences. `strictlyProper_rejects_point_prediction`: under *any* strictly proper score, a
single-structure prediction is strictly worse than the truth unless the truth is that single
structure — the evaluation-side counterpart of the Part I error floor. And in the ensemble
language of `RequestProject/EnsembleCore.lean`, `brier_rejects_deterministic_model`: if two
conformations are populated, every deterministic model scores strictly worse than the true
ensemble (via `Ens.popVec`, the population vector of an ensemble, and `Ens.sum_prob`).

## 76. The sample-distance score rewards collapse

The score in everyday use draws a structure from the model and measures its deviation from the
observed one: `sqDistScore c p y = ∑_x p x (c x − c y)²` along a structural coordinate `c`. Its
expected value is exactly

> `var(model) + var(truth) + (mean(model) − mean(truth))²`   (`sqDistScore_expected`).

Three readings, all proved. It sees the prediction only through two numbers, so it cannot
separate models with the same mean and variance (`sqDistScore_depends_on_two_moments`). At fixed
mean it strictly *decreases* as the model becomes less dispersed, whatever the target
(`sqDistScore_rewards_collapse`): a model is penalised precisely for reporting the disorder that
is really there. And it is not strictly proper (`sqDistScore_not_strictlyProper`): on three
equally populated rotamers with coordinate `−1, 0, 1` the truthful ensemble scores `4/3` while
the single structure at the mean scores `2/3` (`sqDist_three_state`) — the truth pays exactly its
own variance.

## 77. The best-of-`N` score rewards hedging

The other score in everyday use reports `N` structures and keeps the closest one:
`bestOfScore d p y` is the minimum of `d x y` over the conformations the model populates. It does
not see the weights at all (`bestOfScore_eq_of_supp_eq`); enlarging the reported set can only
lower it (`bestOfScore_antitone_supp`); and *every* prediction whose support contains the truth's
scores a perfect zero (`bestOfScore_expected_eq_zero`) — the truth, an arbitrarily badly weighted
model on the same conformations, and the model that simply lists the whole library are all tied
at the optimum. Hence `bestOfScore_not_strictlyProper`, witnessed by two conformations with
populations `(1/2, 1/2)` and `(3/4, 1/4)` both scoring `0`.

## 78. What an honest score does to a context-blind model

If one prediction must serve a family of targets `q i` occurring with frequencies `r i` — the
same sequence in several cellular contexts — then under the quadratic score the unique optimum is
their weighted average `mixVec r q` (`brier_optimal_prediction_is_average`, from
`expScore_brier_mix`). An honestly trained context-blind model therefore converges on the
context-averaged ensemble, which is generally the truth in no context at all. This is the
training-time image of the context laws of Part V and `Binding.lean`.

## 79. The Part XIV capstone

`IDR.evaluation_design_laws` bundles the five clauses: (1) a strictly proper ensemble score exists
with excess risk equal to the squared population error; (2) any strictly proper score rejects
single-structure output; (3) the sample-distance score decomposes as variance + variance + bias²,
rewards collapse and is not strictly proper; (4) the best-of-`N` score is population-blind,
hedging-rewarding and not strictly proper; (5) an honest score drives a context-blind model to
the context average. It depends only on Lean's standard axioms.

The design conclusion of Part XIV: the specification of §§1–74 is incomplete without a matching
evaluation. A model of a disordered region must output an ensemble conditioned on context, *and*
be fitted and ranked by a strictly proper score of that ensemble — the two scores in common use
select, respectively, a collapsed structure and an over-hedged list.

## 80. The irreducible floor: raw scores are not comparable across regions

A strictly proper score ranks models correctly against a *fixed* target; it does not measure
difficulty. `RequestProject/ScoreFloor.lean` makes the confound exact. Writing
`gini q = 1 − ∑_x q x²` for the probability that two independent draws from the target differ,

> `expScore brier p q = ∑_x (p x − q x)² + gini q`   (`brier_decomposition`),

so the score is the model's squared population error plus a floor fixed by the target. Even a
perfect model pays the floor (`brier_floor`); the floor is non-negative (`gini_nonneg`), vanishes
exactly on an ordered, single-conformation target (`gini_eq_zero_iff`), and is largest,
`1 − 1/|X|`, on the maximally disordered one (`gini_le_one_sub_inv_card`).

The consequence is `benchmark_confounded`: an *exactly correct* model of three equally populated
rotamers scores `2/3`, while a model that is *wrong* about an ordered region — misplacing a tenth
of its population — scores `1/50`. Raw benchmark numbers therefore rank regions, not models. A
disorder model must be reported by its excess over the floor, which by `brier_decomposition` is
precisely the squared error of its predicted populations. `IDR.benchmarking_design_laws` bundles
the three clauses.

## 81. Crowding: the ensemble in the cell is not the ensemble in the tube

Every published disordered-region ensemble was measured in dilute buffer; the region works at a
few hundred grams per litre of other macromolecules. `RequestProject/Crowding.lean` makes the
displacement exact. A crowded background at osmotic pressure `Pi` costs each conformation the
work `Pi·v(x)` of opening a cavity of its excluded volume, so the in-cell populations are the
exponential tilt `crowded q v beta Pi ∝ q · exp(−beta·Pi·v)` — equivalently, a shift of the
energy landscape by `beta·Pi·v` (`crowded_eq_boltz`), reducing to the measured ensemble at
`Pi = 0` (`crowded_zero`).

The response is a *fluctuation of the dilute ensemble*:

> `d⟨f⟩/dPi = −beta · Cov(f, v)`   (`hasDerivAt_crowdedMean`).

The covariance is controlled by the weighted two-point (Hoeffding) identity
`cov_two_point_of_weights`, giving `cov_nonneg_of_comonotone` and, when two populated
conformations are separated by both observables, `cov_pos_of_comonotone` — the weighted
Chebyshev sum inequality. Hence if larger conformations exclude more volume the mean size is a
**strictly decreasing** function of the crowder pressure, at every pressure
(`crowding_strictly_compacts`), and so the dilute ensemble a model is trained on is never the
in-cell ensemble it is asked about (`in_cell_ne_in_vitro`). A region whose conformations all
exclude the same volume — a folded domain — is untouched (`rigid_region_ignores_crowding`): the
correction is specific to conformational heterogeneity.

Finally the correction is a genuinely new parameter, not a refitted temperature. On a
three-state ladder with energies `0, 1, 2` the Boltzmann populations are geometric at every
temperature (`boltz_ladder_geometric`, from the general test `boltz_geom_test`), while a crowder
that penalises only the most expanded state over-populates the middle state by exactly
`exp(beta·Pi)` (`crowded_ladder_not_geometric`). Therefore no inverse temperature reproduces the
crowded populations (`crowding_is_not_a_temperature`), and a model of a disordered region must
carry the crowder pressure explicitly.

## 82. Force spectroscopy: compliance is fluctuation, and the curve sees one marginal

`RequestProject/Force.lean` treats the pulling experiment as the tilt by `beta·F·x` along an
extension coordinate. The slope of the force–extension curve is exactly `beta` times the
*variance* of the extension in the ensemble at that force (`hasDerivAt_extension`,
`stiffness_eq_beta_var`), so the curve is monotone (`extension_mono`); a single-structure model
is exactly inextensible (`rigid_is_inextensible`) and therefore cannot exhibit entropic
elasticity at all, while two populated conformations of different extension already force a
strictly increasing curve (`extension_strictMono`).

The two-state bond is solved in closed form: `⟨x⟩ = b·tanh(beta·F·b)`
(`two_state_bond_extension`), with zero-force stiffness `beta·b²` (`two_state_stiffness_zero_force`,
obtained from the general variance formula, not postulated) and the contour bound `|⟨x⟩| < b`
(`two_state_extension_lt`).

What the experiment cannot see is equally sharp. Writing `extLaw` for the law of the extension
coordinate, two ensembles — of different sizes, on different conformation spaces — inducing the
same `extLaw` have identical curves at every force and temperature
(`same_extension_law_same_force_curve`). The explicit witness `force_curve_blind_to_structure`
has two libraries with the same extensions and populations but a second structural coordinate
differing by a full unit: identical pulling data, different ensembles. A model fitted to force
data is unconstrained in every direction orthogonal to the pulling axis.

## 83. Single molecules in time: unbiased at equilibrium, blind outside the basin

`RequestProject/Trajectory.lean` formalises the tacit ergodicity assumption of single-molecule
work. With `distAt` the law after `t` steps and `timeAvg` the expected time average over `T`
frames: started from a stationary ensemble the expected time average is *exactly* the ensemble
average, for every observable and every window (`timeAvg_stationary`) — the positive result.

The negative one is that this is all one gets. A trajectory started in a kinetically closed set
of conformations never leaves it (`supportedOn_evolve`, `supportedOn_distAt`), so two kinetic
models whose rows agree on that set produce identical laws at every time
(`distAt_eq_of_agree`) and identical expected time averages for every observable and window
(`timeAvg_eq_of_agree`). The witness `single_molecule_cannot_identify_the_ensemble` exhibits
two three-state models — in one the second basin is a trap, in the other it decays into the
watched basin — with identical single-molecule statistics from any start in the basin and
different equilibrium ensembles (`deltaTwo_stationary_trap`,
`deltaTwo_not_stationary_decay`). And `reducible_stationary_not_unique` shows that when exchange
is absent the equilibrium ensemble is not even a function of the kinetics: two distinct
stationary ensembles for the same model. Between-basin populations must be obtained from
elsewhere, or reported as unidentified.

## 84. The Part XV capstone

`IDR.situated_design_laws` bundles the three clauses — crowding (tilt, strict compaction,
rigidity exemption, not a temperature), force (stiffness as variance, inextensibility of rigid
models, the `tanh` law, blindness to everything but one marginal) and trajectories
(unbiasedness at equilibrium, blindness outside the visited basin). It depends only on Lean's
standard axioms.

## 85. Linkage: the ligand's grip on the ensemble is the ensemble's grip on the ligand

`RequestProject/Linkage.lean` tilts by two couplings at once — a structural field `lam·A` and a
ligand chemical potential `mu·B`. Both partial responses are covariances in the same ensemble
(`hasDerivAt_mean2_mu`, `hasDerivAt_mean2_lam`), and covariance is symmetric (`cov2_comm`), so

> `∂⟨A⟩/∂mu = ∂⟨B⟩/∂lam`   (`linkage_reciprocity`, Wyman's linkage relation).

The coupling vanishes exactly when the partner does not discriminate between conformations
(`no_linkage_of_uniform_affinity`) and is strictly positive as soon as it does, with the
populated conformations ordering both observables alike (`linkage_of_comonotone`, reusing the
weighted Chebyshev inequality of §81). A model therefore cannot fit "how the partner reshapes
the ensemble" and "how the ensemble sets the affinity" as two independent numbers.

The finite form is the thermodynamic box: four states `U, F, UL, FL` with energies `0`, `eF`,
`eL`, `eF + eL + w`. The cross-product of the populations is `exp(−w)` regardless of the
intrinsic stability and affinity (`thermodynamic_box`), so the factor by which the ligand shifts
the folding equilibrium equals the factor by which folding shifts the binding equilibrium
(`folding_stabilization_eq_binding_enhancement`), and with favourable coupling the folded
fraction strictly rises on saturation (`apo_folded_fraction_lt_holo`). There is one coupling
constant, and a model must report it once.

## 86. The tether: the disordered linker is part of the binding site

`RequestProject/Tether.lean` treats the ideal three-dimensional linker of `2m` steps as three
independent one-dimensional walks, so end-to-end contact is the simultaneous return of all three
coordinates and `contactProb m = (C(2m,m)/4^m)³`. From the central binomial identity comes the
exact recursion `ret1_succ`, hence strict decrease at every length (`ret1_strictAnti`,
`contactProb_strictAnti`), and by induction the two elementary bounds
`ret1 m ² · (3m+1) ≤ 1` (`ret1_sq_mul_le_one`) and `1 ≤ ret1 m ² · (4m+1)`
(`one_le_ret1_sq_mul`), i.e. `(4m+1)^{−1/2} ≤ ret1 m ≤ (3m+1)^{−1/2}` (`inv_sqrt_le_ret1`,
`ret1_le_inv_sqrt`) and the two-sided `N^{−3/2}` law `le_contactProb`, `contactProb_le` for the
three-dimensional contact probability — the familiar decay of the effective concentration of an
ideal tether, with the exponent pinned from both sides rather than assumed.

The design consequence is `avidity_not_a_property_of_the_motif`: with effective concentration
`effConc` and apparent constant `apparentK = Kintr · effConc`, two constructs with the *same*
motif and linkers of different lengths have strictly different apparent affinities, the longer
linker always the weaker. An affinity measured in one construct is not a property of the motif
and does not transfer; a model that predicts binding by a disordered region must model the
disordered part.

## 87. The Part XVI capstone

`IDR.coupling_design_laws` bundles the two clauses — linkage (reciprocity, the vanishing and
strict-positivity criteria, the thermodynamic box and its saturation consequence) and the tether
(exact recursion, strict monotonicity, the `N^{−3/2}` law, and the non-transferability of
affinity). It depends only on Lean's standard axioms.

Taken with the earlier parts, the specification is now: a *situated conditional* distribution —
conditioned on sequence, partner, temperature, salt, crowder pressure, tether and preparation —
forward-modelled through the exact nonlinear functionals of each experiment, fitted and ranked by
a strictly proper ensemble score reported against its floor, and accompanied by the
identifiability limits that the data themselves impose.

## 88. Concentration: a reported ensemble belongs to a sample

Disordered regions are weakly self-associating, and NMR or SAXS ensembles are measured far above
the concentrations at which the region often works. `RequestProject/SelfAssociation.lean` solves
the monomer–dimer mass action `c = m + 2·K·m²` in closed form (`mass_action`), giving the
monomer fraction `monoFrac K c = 2/(1 + √(1 + 8Kc))`. It is a genuine fraction
(`monoFrac_pos`, `monoFrac_le_one`), equals one only at infinite dilution (`monoFrac_zero`),
**strictly** decreases with total concentration at every concentration (`monoFrac_strictAnti`)
— there is no regime in which a sample is "dilute enough" — and vanishes as the sample is
concentrated (`monoFrac_tendsto_zero`).

Hence any observable with different monomer and dimer values drifts strictly with concentration
(`measured_observable_depends_on_concentration`, with the direction of drift in
`apparentObs_strictAnti`), and only the infinite-dilution value is a property of the molecule
(`apparentObs_at_zero_eq_monomer`). A model fitted to published ensembles inherits their
concentrations unless the association equilibrium is modelled or extrapolated away.

## 89. Multisite modification: single-site data do not add up

Disordered regions are the main substrates of multisite modification, and the standard
experiment modifies one site at a time. `RequestProject/Multisite.lean` measures the failure of
additivity by the interaction free energy
`coupling = log Z(λ,μ) + log Z(0,0) − log Z(λ,0) − log Z(0,μ)` of the doubly tilted ensemble.

If one modification does not discriminate between conformations, the effects are exactly
additive (`coupling_eq_zero_of_constant`) and single-site data do suffice. Otherwise they do
not: for two equally populated conformations with both modifications favouring the same one,
the coupling is strictly positive (`coupling_pos_of_correlated`), with exact value
`log(2(e²+1)/(e+1)²)`, so the joint effect exceeds the sum of the single effects
(`multisite_effects_not_additive`). For a multiply modifiable region every pairwise coupling is
a separate parameter — and by Part XVI each is simultaneously the reciprocal coupling that
fixes the partner's affinity.

## 90. The Part XVII capstone

`IDR.sample_design_laws` bundles the two clauses: the concentration dependence of any reported
ensemble (mass action, strict decay of the monomer fraction, strict drift of a distinguishing
observable, the infinite-dilution value), and the non-additivity of multisite modification
(exact additivity criterion, strictly positive witness, failure of additivity). It depends only
on Lean's standard axioms.

## 91. Driven: in a living cell the target is not a Gibbs ensemble

Every equilibrium statement so far — Boltzmann populations, the variational free energy,
exponential tilting, linkage — presupposes detailed balance. A cell does not supply it: ATP-driven
kinases, chaperones and translocation maintain conformational cycles that run in one direction.
`RequestProject/Driven.lean` proves what that costs a model, in the smallest system that can show
it: a three-state cycle `cycleP a b` that steps forward with probability `a` and backward with
probability `b`.

The good news first. The driven cycle is a legitimate kinetics (`cycleP_stochastic`) with a
perfectly ordinary stationary distribution, the uniform one (`cycle_stationary_unif`), so the
observation theory of Part XV survives intact: a trajectory started in the steady state still
reports the ensemble average exactly (`Trajectory.timeAvg_stationary`). Nothing about the
*measurement* theory breaks when the region is driven.

The bad news. The steady state carries a probability current `(a − b)/3` around each edge of the
cycle (`cycle_current`): it is stationary without being at rest. And by Kolmogorov's criterion, in
its smallest instance, **no strictly positive distribution whatsoever** satisfies detailed balance
with the driven kernel when `a ≠ b` (`no_detailed_balance_of_driven`; the proof multiplies the
three edge conditions, cancels the populations and concludes `a³ = b³`). The driven steady state
is therefore the reversible equilibrium of no energy function at all — there is no landscape to
fit, however flexible the functional form. `driven_needs_kinetics` bundles this: for a region held
out of equilibrium a model must parameterise a *kinetics* and predict its stationary distribution,
not the equilibrium of a landscape.

## 92. What the drive costs: entropy production

`RequestProject/EntropyProduction.lean` prices the drive. For a kinetics `P` with steady state
`pi`, the entropy production rate is the relative entropy of the forward one-step process against
its time reverse,

    epRate P pi = ∑_i ∑_j  pi_i P_ij · log ( pi_i P_ij / pi_j P_ji ).

Symmetrising the double sum turns it into `½ ∑_{i,j} (x − y)(log x − log y)` with `x = pi_i P_ij`
and `y = pi_j P_ji` (`two_mul_epRate`), and each term of that is nonnegative. So the rate is never
negative (`epRate_nonneg`) — the second law, as an identity plus one elementary inequality — and it
vanishes **exactly** at detailed balance (`epRate_eq_zero_iff_detailedBalance`), because the pair
term is strictly positive as soon as `x ≠ y`. Dissipation is not an extra modelling assumption: it
is precisely the obstruction to describing the region by a landscape, and a landscape model is
committed to predicting zero of it.

For the three-state cycle the rate is exactly `(a − b)·log(a/b)` (`cycle_epRate`), strictly
positive whenever the cycle is driven (`cycle_epRate_pos`) and zero when it is not
(`cycle_epRate_eq_zero`). The design consequence is `dissipation_not_determined_by_populations`:
two kinetics on the same three conformations, both stochastic, with the *same* stationary
populations, one reversible with zero entropy production and one dissipating at a strictly
positive rate. No functional of the reported ensemble recovers the cell's energy budget; it is a
separate parameter of the model. `IDR.nonequilibrium_design_laws` (`PartNineteen.lean`) bundles
the three clauses of Parts XVIII–XIX.

## 93. The integrator: a finite timestep is a silent change of Hamiltonian

Every simulated ensemble of a disordered region came from a discrete integrator run at a finite
timestep. `RequestProject/Integrator.lean` proves that this is not an implementation detail: it
changes the ensemble. Take overdamped Langevin (Euler–Maruyama) dynamics in a harmonic well of
stiffness `k` at inverse temperature `beta` — the normal-mode description of a flexible chain
(`Rouse.lean`). The update acts on the variance by `v ↦ (1 − k·dt)² v + 2 dt/beta`, so the variance
after `n` steps is known in closed form (`varSeq_eq`) and, inside the stability limit `k·dt < 2`,
relaxes geometrically (`varSeq_tendsto`) to

    discVar = 2 / (beta·k·(2 − k·dt)).

That is not the Boltzmann variance `1/(beta·k)`. The sampled ensemble is too broad, by exactly
`dt / (beta(2 − k·dt))` (`bias_eq`, `bias_pos`, `exactVar_lt_discVar`) — a first-order error in the
timestep that vanishes only in the limit (`bias_tendsto_zero_at_zero_timestep`). And the excess is
not noise: the simulation is *exactly* the equilibrium ensemble of a different force field, one
with the strictly smaller stiffness `effK = k(2 − k·dt)/2`
(`integrator_samples_a_softer_force_field`). A finite timestep is a silent change of Hamiltonian,
and a reported radius of gyration carries it.

Two further consequences. It is a bias, not a variance, so the statistics of Part V do not help:
beyond some run length the sampled variance stays at least half the bias away from the target, from
any starting point (`no_sampling_removes_the_timestep_bias`). And past the stability limit
`k·dt > 2` the integrator does not merely mis-sample — the variance diverges
(`unstable_of_large_timestep`).

Finally the positive half, which makes this a design law rather than a complaint. For *any* scheme
whose one-step action on the variance is affine, `v ↦ r v + s` — that is, for any scheme built from
linear drift and additive noise — the stationary variance is `s/(1 − r)`, and sampling the
Boltzmann ensemble exactly is the single algebraic condition `s = (1 − r)/(beta·k)` matching the
noise to the damping (`unbiased_iff_consistency`). The exact Ornstein–Uhlenbeck propagator
satisfies it at *every* timestep (`ou_unbiased`). So the bias belongs to the chosen scheme, not to
discreteness as such. `reported_ensemble_belongs_to_the_integrator` states the law: a reported
ensemble is a statement about an integrator as much as about an energy function, so the integrator
— and whether it meets the consistency condition — is part of the model.

## 94. One relaxation time is not enough

Fluorescence correlation, single-molecule FRET and NMR relaxation of disordered regions report
*hierarchies* of timescales, and a two-state kinetic model is the default interpretation of almost
every such experiment. `RequestProject/Memory.lean` proves the two are incompatible. The two-state
chain `twoP a b` is a perfectly good reversible kinetics — stochastic (`twoP_stochastic`), with
stationary populations `piTwo` (`piTwo_stationary`) satisfying detailed balance
(`twoP_detailedBalance`) — so nothing here is a complaint about its equilibrium. But its
propagator acts on any observable as `mean + (f i − mean)·λ^t` with the single eigenvalue
`λ = 1 − a − b` (`prop_eq`), so the equilibrium autocorrelation of *every* observable is the one
geometric `Var(f)·λ^t` (`autocorr_eq`): a two-state model has exactly one relaxation time, and it
is a property of the model, not of the probe. A genuine two-timescale decay is not a geometric:
matching `c₁λ₁^t + c₂λ₂^t` (positive weights, distinct rates) at `t = 0, 1, 2` already forces
`c₁c₂(λ₁ − λ₂)² = 0` by strict Cauchy–Schwarz (`no_single_geometric`). Hence
`two_state_model_cannot_fit_two_timescales`: no two-state model, with any rates and any
observable, reproduces a two-exponential correlation function. Distinct measured timescales are a
*capacity* statement about the kinetic model, exactly as distinct populated conformations are a
capacity statement about the ensemble (Part I).

# Part XXII — Uncertainty: what a model must promise about its own answer

Part XIV fixed the score. This part fixes the two things a pipeline reports *about* the score —
the reliability diagram and the prediction set — since both are routinely offered as evidence that
a model "knows what it does not know". The files are `RequestProject/Calibration.lean` and
`RequestProject/PredictionSets.lean`, bundled in `RequestProject/PartTwentyTwo.lean`.

## 95. Calibration is not enough: the resolution a model throws away

The setting is the one the earlier parts force. A model does not see its target: it computes a
finite internal code `k = b i` of its input `i` (the sequence, the context, the bin of a confidence
readout) and answers with a population vector `A k` on the conformation library. The benchmark
weights inputs by `mu`, and the truth at input `i` is `T i`. `Calibrated` says what a reliability
diagram checks: inside every code class the predicted populations equal the average true
populations.

* `risk_decomposition` — the exact identity `risk = calError + resolution`. The `mu`-averaged
  squared population error (which is exactly the excess risk of the strictly proper quadratic
  score of Part XIV, `risk_eq_brier_excess`) splits into a calibration term and a *resolution*
  term: the weighted variance of the truth *inside* the code classes, i.e. precisely the
  contextual information the code destroyed.
* `calError_eq_zero_of_calibrated`, `risk_eq_resolution_of_calibrated` — a calibrated model has
  paid the calibration term and nothing else. Everything a reliability diagram can detect has been
  removed; the whole remaining error is the resolution deficit.
* `resolution_pos_of_conflated` — and that term is strictly positive as soon as two weighted
  inputs with different targets share a code. Calibration is therefore blind to exactly the
  failure the earlier parts identify as the central one: conflating contexts.
* `recalibrated_risk_eq_resolution`, `recalibration_improves` — recalibration (answering with the
  class averages) is always an improvement and is optimal for the given code; the floor it reaches
  is the resolution term. Recalibrating a context-blind model cannot make it context-aware.
* `single_bucket_calibrated`, `calibrated_but_wrong_everywhere` — the extreme case: a model that
  ignores its input and always reports the population-averaged ensemble is *perfectly calibrated*
  and has strictly positive error. In the project's ensemble language this is
  `ens_context_blind_calibrated_but_wrong` (`PartTwentyTwo.lean`). Calibration is necessary, never
  sufficient; it must be reported together with resolution.

## 96. Prediction sets: validity is free, informativeness is the disorder

A disorder model is often asked for something weaker than an ensemble: a *set* of conformations
claimed to contain the truth — the top-`N` decoys, the states above a population cut, the
conformers deposited as "the ensemble". `mass p S` is the population assigned to `S`, and `S`
`Covers` the truth `q` at level `alpha` when `1 − alpha ≤ mass q S`.

* `univ_covers`, `validity_is_free` — the whole library covers at every level. Validity by itself
  therefore ranks no model: the content of a prediction set is its *size*.
* `mass_diff_le_half_ell1`, `covers_of_close` — coverage transfers with accuracy: a model whose
  populations are within `eps` of the truth in `l¹` loses at most `eps/2` of coverage. This is the
  precise sense in which the ensemble accuracy of Part III buys honest prediction sets.
* `card_ge_of_covers` — the size law: if no conformation of the target carries more than `pmax`,
  every set covering at level `alpha` has at least `(1 − alpha)/pmax` conformations. The size of an
  honest answer is fixed by the flatness of the *truth*, not by the model.
* `unif_covers_iff` — for `m` equally populated conformations covering is *equivalent* to
  containing at least `(1 − alpha)·m` of them, so the law is sharp (`card_ge_of_covers_unif`), and
  `no_singleton_covers` — a single structure is not a valid answer for any target flatter than
  `1 − alpha`. This is the Part I verdict in the language of coverage.
* `marginal_not_conditional` — the coverage a benchmark reports is a *marginal*. Two contexts, an
  ordered target in each, and a context-blind set: the reported coverage is 90% at level `1/10`
  while the coverage in the second context is exactly zero. Coverage must be reported per context —
  the same context-conditionality the earlier parts force on the model itself.

## 97. The Part XXII capstone

`IDR.uncertainty_design_laws` (`PartTwentyTwo.lean`) bundles six clauses: the
calibration–resolution decomposition; calibration is not sufficient (a calibrated model's whole
error is its resolution deficit, strictly positive under conflation); recalibration is optimal for
a given code and never enough; validity is free; the size law with its sharp uniform case and the
rejection of single-structure answers; and marginal coverage is not conditional coverage. Together
with Part XIV: the score must be strictly proper, the reported error must be its excess over the
target's floor, calibration must be reported with resolution, and coverage must be reported per
context together with set size.

# Part XXIII — Mutations: what a sequence-to-ensemble model must be linear in

Every earlier part concerns one region in one context, but a model is used to predict
*differences*: the effect of a substitution, and how two substitutions combine. The default
analysis sums single-mutant effects on the measured population and reads the deviation
("epistasis") as evidence of an interaction between sites. `RequestProject/Epistasis.lean` shows
that the arithmetic is being done on the wrong quantity, and prices the error. The setting is the
two-state form every population report takes: a region is ordered with probability
`sig(−beta·E)`, `sig` the logistic function, and a double-mutant cycle of a quantity `f` is
`cyc f x a b = f(x+a+b) + f(x) − f(x+a) − f(x+b)`.

## 98. Additivity is the absence of a cycle — of the right quantity

An affine quantity has zero cycle on every background (`cyc_of_affine`), so cycles are the right
diagnostic once the quantity is right. For the two-site energy `h₁s₁ + h₂s₂ + Js₁s₂` the energy
cycle is exactly the coupling `J` (`energy_cycle_eq_coupling`). And the population's log-odds *is*
the energy up to `−beta` (`logit_sig`, `logit_pop`), so a cycle taken in log-odds returns
`−beta·J` exactly (`logodds_cycle_eq_coupling`) and, at non-zero temperature, vanishes if and only
if the two sites are uncoupled (`coupling_iff_logodds_cycle`). That is the positive law: a
sequence-to-ensemble model must be additive in energy, must report energies, and is therefore
non-linear in everything it predicts.

## 99. Populations do not add, even for uncoupled sites

Two strictly favourable, energetically independent substitutions have a strictly *negative*
population cycle (`sig_cyc_neg`; the proof reduces to the factorisation
`(u−1)(v−1)(uv−1) > 0` for `u = e^a, v = e^b > 1`). So measured non-additivity is not evidence of
an interaction: `population_cycle_not_evidence_of_coupling` exhibits two uncoupled substitutions
worth `1 kT` each with zero energy cycle and a strictly negative population cycle — the curvature
of the Boltzmann map, not physics of the sites.

Worse, the *sign* is a property of the background. The cycle is odd under reflection of the
background (`cyc_reflect`, from the point symmetry `sig(−x) = 1 − sig x`), so the same pair of
substitutions is negatively epistatic on one background and positively epistatic on its mirror
image, with no change in any energy (`epistasis_sign_depends_on_background`).

## 100. Saturation, and the Part XXIII capstone

On a sufficiently stabilising background every further stabilisation, however large in energy,
moves the population by less than any prescribed `eps` (`saturation`): population data lose
sensitivity to the very quantity the model must get right, which is why fits and benchmarks must
be run in energy. `IDR.mutational_design_laws` (`PartTwentyThree.lean`) bundles the five clauses.
Read with Part IV.3 (`EnergyModels.lean`): the model must carry an energy function, its parameters
must be fitted and reported in energy, and every quantity it predicts is a non-linear function of
them.

# Part XXIV — The molecular model: continuous space, a real force field, real solvent, and the Gibbs measure

Parts I–XXIII use the smallest carrier of each phenomenon: finite ensembles, lattice chains,
two-state populations. Part XXIV removes those carriers and re-proves the load-bearing statements
in the setting in which a molecular model is actually built — `N` atoms at real coordinates in
`R³`, a class-I molecular-mechanics Hamiltonian, implicit solvent, and a Boltzmann–Gibbs
probability measure on `R^(3N)` that has a genuine density with respect to Lebesgue measure.

## 101. The context is a coordinate of the model

`RequestProject/Context.lean` makes the environment explicit: `BiophysicalContext` carries an
absolute temperature, an ionic strength and a partner concentration, with positivity as part of
the structure, and the SI constants (`kB`, `elementaryCharge`, `vacuumPermittivity`, `avogadro`,
`waterPermittivity`) are the real ones. From these the Debye screening length
`debyeLength = sqrt(εᵣε₀k_BT / (2·N_A·10³·e²·I))` is defined and shown positive
(`debyeLength_pos`), strictly decreasing in ionic strength (`debyeLength_strictAnti_ionicStrength`)
and strictly increasing in temperature (`debyeLength_strictMono_temperature`). A screened Coulomb
interaction is strictly weaker in magnitude than the bare one (`screened_lt_bare`) and its
magnitude strictly decreases with salt (`screened_abs_strictAnti_ionicStrength`); a Boltzmann
population moves with temperature (`boltzmann_strictMono_temperature`), and a bound fraction with
partner concentration (`bound_strictMono_ligand`, `bound_lt_one`).

Each of the three coordinates is load-bearing on its own: `context_coordinates_load_bearing`
exhibits, for each coordinate, two contexts agreeing on the other two and disagreeing on a
predicted observable. Consequently `no_context_free_model`: no constant predictor reproduces the
temperature dependence of a population. An ensemble prediction without a stated context is not a
prediction.

## 102. The Hamiltonian, and stability

`RequestProject/Potentials.lean` defines the pair potentials with no smoothing or truncation: the
Lennard-Jones 12-6 `lj`, Coulomb with a dielectric `coulomb`, the harmonic bonded term `harmonic`
and a periodic `torsion`. The 12-6 well is characterised exactly — its zero is at `σ`
(`lj_eq_zero_iff_sigma`), it is negative precisely beyond `σ` (`lj_neg_iff`, `lj_pos_iff`), it is
bounded below by `−ε` (`lj_ge_neg_eps`) with equality exactly at the minimum, which is attained at
`2^(1/6)σ` with value `−ε` (`lj_min_at`); inside `σ/2` the repulsion already exceeds
`2ε(σ/r)^12` (`lj_ge_half_core`) and the potential is unbounded above (`lj_unbounded_above`).

The key analytic fact is `pair_bddBelow`: for every `ε, σ > 0` and every Coulomb coefficient `c`
(of either sign), `lj ε σ r + c/r` is bounded below uniformly in `r > 0`. The `r^-12` core beats the
`r^-1` attraction, so a point-charge model *with* a repulsive core is stable while one without is
not. Conversely `pair_repulsive_unbounded`: however strong the electrostatic attraction, at small
enough separation the pair energy exceeds any bound.

`RequestProject/Hamiltonian.lean` assembles these on `Conf N = Fin N → EuclideanSpace ℝ (Fin 3)`.
A `ForceField N` carries partial charges, per-atom `ε` and `σ`, bond force constants and
equilibrium lengths, and a dielectric constant, with the positivity conditions as fields; the
Hamiltonian `H = Ebond + Eelec + Evdw` uses Lorentz–Berthelot mixing. Three theorems make it a
well-posed target for a probability measure:

* `ForceField.H_bddBelow` — stability: there is a finite `B` with `−B ≤ H x` for every
  configuration whose atoms are pairwise distinct, and the bound is assembled from the pair bound.
* `ForceField.H_repulsive_core` — excluded volume as a *theorem*: for every pair and every `M`
  there is a separation below which `H > M`. The hard core that the lattice models of Part XII had
  to postulate is a consequence here.
* `ForceField.H_rigid_invariant` — exact E(3) invariance. A `RigidMotion` is a linear isometry of
  `R³` (rotations *and* reflections) followed by a translation; because `H` is a function of the
  interatomic distances only (`IsPairwiseGeometric`, `invariant_of_pairwiseGeometric`), it is
  exactly invariant, with no approximation and no data augmentation.

Finally `H_measurable`, and `coincident_null`: the set of configurations in which some two atoms
are exactly superposed is Lebesgue-null, so the singularity of the `r^-12` and `r^-1` terms is
invisible to the measure.

## 103. Implicit solvent: Generalized Born and surface area

`RequestProject/Solvation.lean` adds the solvent as energy rather than as a fitted correction. The
Still interpolation `fGB Ri Rj r = sqrt(r² + Ri·Rj·exp(−r²/(4RiRj)))` satisfies exactly the
properties the Generalized Born model needs: `fGB R R 0 = R` (`fGB_self`), it dominates both
`sqrt(RiRj)` (`fGB_ge_sqrt`) and the distance `r` (`fGB_ge_dist`), and it approaches `r` from above
with a controlled gap (`fGB_sub_dist_le`), so the polar term interpolates between the Born
self-energy at contact and the screened Coulomb form at long range.

The pair term `gbPair` gives `born_self_neg` — a nonzero charge is strictly stabilised by a solvent
more polarisable than the solute — and `born_self_strictAnti_radius`: the smaller the Born radius,
the larger the desolvation penalty. This is the quantitative form of the reason a charged
disordered region resists burial. The total polar term is bounded below (`gbEnergy_bddBelow`).

The nonpolar term is built from a genuine geometric surface: `accessibleSet` is the set of points
of the atom's expanded sphere not inside any other expanded sphere, and `sasa` is its
two-dimensional Hausdorff measure `μH[2]`. `accessible_of_far` (a distant atom occludes nothing),
`sasa_eq_zero_of_engulfed` (an engulfed atom has zero accessible surface), `sasa_mono` and
`nonpolar_le_of_occlusion` (burial lowers the hydrophobic term) are the statements a collapse
argument uses.

The solvated Hamiltonian `Hsolv = H + gbEnergy + nonpolarEnergy` inherits both properties that
matter: `Hsolv_rigid_invariant` (exact E(3) invariance, including of the surface term, via
`accessibleSet_act` and the isometry invariance of Hausdorff measure) and `Hsolv_bddBelow`
(stability — adding solvent does not destroy the existence of the ensemble).

## 104. The Boltzmann–Gibbs measure and the two identities

`RequestProject/GibbsField.lean` constructs the ensemble. `GibbsData N` carries exactly what is
needed and nothing more: an inverse temperature, a measurable energy, a container `D` of finite
positive volume, and a lower bound for the energy on `D` — the hypothesis supplied by the stability
theorem of §102–§103. Then

* `Zpart_pos` — the partition function is finite and strictly positive;
* `gibbsMeasure` is a probability measure (instance), `gibbs_absolutelyContinuous` and
  `gibbs_rnDeriv` — it has Lebesgue density exactly the Boltzmann density
  `exp(−βU)/Z` on `D`. The output of the model is a density on `R^(3N)`, not a structure;
* `dirac_not_absolutelyContinuous` — a point prediction is not merely a bad density, it has *no*
  density: the Dirac measure at a conformation is not absolutely continuous for `N ≥ 1`.

Two identities make training and sampling well posed without ever evaluating `Z`:

* `density_ratio` — for two conformations in the container, `f(x₁)/f(x₂) = exp(−β(U(x₁) − U(x₂)))`.
  This is the Metropolis acceptance ratio, exactly.
* `score_hasFDerivAt` / `score_eq_neg_beta_grad` — wherever the energy is differentiable,
  `∇ log f = −β ∇U`. The partition function has disappeared; this is why score matching and
  diffusion training target the force field itself.

Invariance transfers to the measure: the diagonal action of a rigid motion preserves Lebesgue
measure on `R^(3N)` (`act_measurePreserving`, via the isometry invariance of the Haar measure and
the product structure), so an invariant energy and container give an invariant ensemble
(`gibbsMeasure_rigid_invariant`, through `map_withDensity_of_invariant`).

## 105. Scoring densities, and the exact price of non-equivariance

`RequestProject/ContinuousScore.lean` redoes the Part XIV scoring theory measure-theoretically.
The continuous quadratic (Hyvärinen–Brier) score has excess exactly the squared `L²` distance
between the model density and the truth (`l2_excess_decomposition`), which vanishes precisely when
the model is right almost everywhere (`l2risk_eq_zero_iff`) — strict propriety in the continuum.

The symmetrisation theorem prices non-equivariance exactly: for a symmetry `g` of the target,

  `risk f = risk (symmetrise f g) + (1/4)·∫ (f − f∘g)²`   (`symmetrization_identity`),

so symmetrising never hurts (`symmetrization_improves`) and the risks are equal exactly when the
model is already equivariant almost everywhere (`equal_risk_iff_equivariant`). A non-equivariant
architecture is *strictly* beaten by its own symmetrisation, by a computable amount: this is the
formal reason to build the symmetry in rather than to train it in.

## 106. The generative operator

`RequestProject/Generative.lean` states the generative programme as an operator identity:
`IsGenerator T μ_Z μ_C` means `T_#μ_Z = μ_C`. Generators compose (`IsGenerator.comp`), and
equivariance transfers: an invariant latent plus an intertwining generator gives an invariant
ensemble (`IsGenerator.invariant`). A noiseless model is excluded — a constant map pushes any
latent to a Dirac measure, which has no density (`not_isGenerator_of_const`).

Existence holds in full generality. Via the quantile function of a real distribution
(`le_cdf_quantile`, `quantile_le_iff`, `map_qgen_unif`: the inverse-transform theorem) and the
Borel isomorphism of the uncountable Polish space `Conf N` with `R` (`conf_uncountable`),
`exists_generator` gives, for every Borel probability measure on conformation space — in particular
for the Gibbs ensemble of the solvated Hamiltonian — a measurable `T` with `T_# Unif(0,1) = μ`. The
programme is not obstructed at the level of representability; what the earlier parts obstruct is
producing the *right* measure and reporting it honestly.

## 107. The Part XXIV capstone

`IDR.molecular_design_laws` (`PartTwentyFour.lean`) bundles nine clauses: the context is
load-bearing; the solvated Hamiltonian is bounded below; excluded volume holds and the overlap set
is null; the solvent terms behave as physics requires; the energy is exactly E(3) invariant; the
ensemble is a density and a point prediction is not; the ratio and score identities hold without
the partition function; the ensemble is invariant and non-equivariance has an exact price; and the
ensemble is representable as a push-forward.

What Part XXIV does *not* claim: the Hamiltonian is a class-I additive force field with implicit
solvent. Electronic polarisability, explicit-solvent many-body potentials of mean force,
protonation equilibria (pH-dependent charge regulation), rigid-constraint (Fixman) corrections and
broken ergodicity (for instance cis/trans proline isomerisation) are each an additional model,
not a missing lemma of this one. What is established is that the standard molecular model is well
posed — the ensemble exists, has a density, is exactly invariant, and its log-gradient is the
force — and that every idealisation still present is named rather than hidden.

# Part XXV — What the class-I model still assumes, and what each assumption costs

Part XXIV's Hamiltonian is additive, has fixed charges, is sampled for a finite time, and is
usually run with rigid constraints on fast degrees of freedom. Part XXV removes each of those four
idealisations and prices it exactly. In every case the answer has the same shape: the idealisation
is not a small error to be absorbed by refitting; it is a representability or identification
failure of a size that can be computed.

## 108. Pairwise additivity cannot represent cooperativity

`RequestProject/ManyBody.lean` works with a three-body potential of mean force `W(a,b,c)` of the
three separations. Additivity, `W a b c = f a + g b + h c`, is exactly the vanishing of the mixed
second difference `mixed W a a' b b' c` (`mixed_eq_zero_of_additive`). The cooperative desolvation
form `−e^{−a}e^{−b}` has a strictly nonzero mixed difference (`mixed_cooperative_neg`), hence
`not_additive_cooperative`: no choice of pair potentials reproduces it. `not_additive_of_mixed_ne`
gives the generic criterion, and `additive_error_lower_bound` prices the best possible additive
surrogate: its worst error on the four exposing configurations is at least a quarter of the mixed
difference. Many-body solvation is a different model, not a better fit of this one.

## 109. Fixed charges are a model of one pH

`RequestProject/Protonation.lean` replaces the fixed partial charge of a titratable group by its
equilibrium occupancy `protonatedFraction pKa pH = 1/(1 + 10^{pH−pKa})`, which is a genuine
probability (`protonatedFraction_mem_Ioo`), equals `1/2` exactly at `pH = pKa`
(`protonatedFraction_half`) and is strictly decreasing in pH (`protonatedFraction_strictAnti_pH`).
The mean charge of a region of independent sites is then strictly decreasing in pH whenever
protonation makes each site more positive (`netCharge_strictAnti_pH`) — charge regulation. Hence
`no_fixed_charge_model`: no constant reproduces the titration curve of even a single acidic site,
so pH is a coordinate of the model in exactly the sense of `Context.lean` (§101).

The coupling to binding is not free either: `linkage_cycle` and `pKa_shift_eq_binding_shift` state
the closed thermodynamic cycle — a predicted pKa shift on binding *is*, with the same number, a
predicted pH dependence of the affinity. A model may not report one without the other.

## 110. A finite run is not the equilibrium ensemble

`RequestProject/BrokenErgodicity.lean` treats the slow degree of freedom that breaks ergodicity in
practice — the classic case is cis/trans isomerisation of a prolyl bond. The two-state master
equation is solved exactly (`popB_hasDerivAt`, `popB_zero`) and converges to the Boltzmann
population, but only as `t → ∞` (`popB_tendsto_equilibrium`). The quantitative statement is
`popB_stuck`: if the total rate times the horizon is at most `δ`, the deviation from equilibrium at
the end of the run is still at least `1 − δ` times its initial value. Averaging over the whole
trajectory does not help: `timeAverage_eq` computes the running average in closed form and
`timeAverage_stuck` gives it the same bound. `finite_run_not_boltzmann` puts it as the
practitioner's statement: for every horizon and every tolerance there is a barrier for which the
run misreports the population by nearly the whole gap. An ensemble claim must therefore be
accompanied by the timescales it is claimed over — the Part XXI point, now for the molecular model.

## 111. Rigid constraints are not the stiff limit: the Fixman factor

`RequestProject/Constraints.lean` treats the standard practice of freezing bond lengths and angles.
With one soft coordinate `q` and one stiff coordinate of stiffness `w(q)/ε`, the marginal of `q` in
the unconstrained ensemble is `exp(−βV(q))·sqrt(2πε/(βw(q)))` (`softMarginal_eq`, by the Gaussian
integral). Therefore `soft_ratio`: the ratio of physical weights at two conformations is the
constrained ratio `exp(−βΔV)` times `sqrt(w(q₂)/w(q₁))`, *for every* `ε` — the discrepancy does not
disappear as the constraint becomes rigid. `soft_eq_rigid_iff` shows the two ensembles agree at a
pair of points exactly when the stiffness is the same there, and `rigid_ne_soft` exhibits a system
where they disagree. A constrained model must either carry the Fixman factor or state that it
samples the constrained ensemble, which is a different distribution.

## 112. The Part XXV capstone

`IDR.residual_assumption_laws` (`PartTwentyFive.lean`) bundles the four statements. Together with
Part XXIV they give the honest position: mathematics cannot deliver a model with *no* assumptions,
but every assumption of the molecular model above is now either removed or explicitly priced —
pairwise additivity, fixed charges, finite-time sampling and rigid constraints included. What
remains outside the development, and is named rather than hidden: electronic polarisability
(the force field is non-polarisable), explicit-solvent structure beyond the two solvation terms and
the three-body obstruction of §108, and quantum-mechanical bond making and breaking.

# Part XXVI — Tractability: an energy function is not yet an ensemble

## 113. Exponentially many conformations, a computable configuration sum

`RequestProject/TransferMatrix.lean` starts from the arithmetic fact that a residue-level model of
`n+1` residues with `k` local states each denotes a distribution over `k^(n+1)` conformations
(`card_chain_states`): a population is a ratio of sums with exponentially many terms, and writing an
energy function down does not by itself produce one number. `Z_eq_vecMul_pow` is the transfer-matrix
theorem for arbitrary `k` (the two-state Ising case is solved by recursion in §69 of
`HelixCoil.lean`): for any nearest-neighbour weight matrix `M` and boundary weights `v`, `u`,

  `∑_x v(x₀) ∏_i M(x_i, x_{i+1}) u(x_n) = v · Mⁿ · u`.

Exponentially many terms, `n` matrix multiplications. With strictly positive weights the sum is
positive (`Z_pos`) and the normalized measure is a genuine ensemble: strictly positive on every
conformation and summing to one (`chainProb_pos`, `chainProb_sum_one`). `Z_boltzmann` states the
same identity for a nearest-neighbour energy at inverse temperature `beta`, which is the form in
which it is used.

## 114. What locality costs: a long-range contact is not nearest-neighbour

Tractability is bought, and `TransferMatrix.lean` prices it. Write `PairFactored P` for the general
nearest-neighbour shape of a three-residue distribution, `P(x) = f(x₀,x₁)·g(x₁,x₂)` — every
nearest-neighbour Gibbs measure has it (`chainProb_pairFactored`), with the boundary conditions
absorbed into the bond weights. Such a model makes the two ends conditionally independent given the
middle residue: the cross products of the four configurations sharing a middle state agree
(`pairFactored_cross`).

`contactDist` is the caricature of a transient tertiary contact: the two ends are pinned to the same
state, the middle residue is free. It is a probability distribution (`contactDist_sum_one`) whose
ends are genuinely correlated — joint population `1/2` against a product of marginals `1/4`
(`contactDist_ends_correlated`) — and it violates the cross-product identity, so it is not
pair-factored (`contactDist_not_pairFactored`). Hence `chainProb_ne_contactDist`: for **every**
transfer matrix and all boundary weights, the nearest-neighbour model differs from it at some
conformation. A local factorization is a restriction on the ensembles a model can express, and the
transient long-range contacts that distinguish a disordered region from a random coil lie outside
it.

## 115. The reweighting repair does not scale: tensorization

`RequestProject/Tensorization.lean` prices the standard alternative — sample a reference model and
reweight onto the target. Part V.1 gives the cost of one such operation, `essFrac = 1/(1+χ²)`. For
product (independent-residue) chain models the second moment of the importance weight factorizes
over sites (`sum_sq_div_prodDist`), so the cost *multiplies*:

  `1 + χ²(P‖Q) = ∏ᵢ (1 + χ²(pᵢ‖qᵢ))`   (`chiSq_prodDist`).

With the same per-residue mismatch `c > 0` at each of `n` residues the effective sample size
fraction is exactly `(1+c)^(−n)` (`essFrac_prodDist_iid`), and retaining `neff` effective frames
requires at least `neff·(1+c)^n` frames of simulation (`frames_needed_prodDist`). Relative entropy
tells the same story additively: a per-residue error of `K` nats is a chain-level error of `n·K`
nats (`kl_prodDist`). A reference ensemble that is wrong by a fixed amount *per residue* cannot be
repaired by reweighting on any realistic budget; it must be refitted. What a factorized model does
buy is cheap local averages: a one-site observable costs a single sum of `k` terms
(`sum_prodDist_local`), not a sum over `k^n` conformations.

## 116. The Part XXVI capstone

`IDR.tractability_laws` (`PartTwentySix.lean`) bundles the four statements: exponential
configuration space with a linear-time configuration sum and a genuine normalized ensemble; the
non-representability of a long-range contact by any nearest-neighbour model; the exponential-in-
length cost of reweighting a mismatched product reference; and the additivity of relative entropy
over residues. The design conclusion is architectural, and two-sided. A model of a disordered
region must be factorized enough to be evaluated and sampled at all — an energy function alone is
not an ensemble — and the factorization must already contain the long-range structure the model is
meant to predict, because neither locality nor post-hoc reweighting will supply it later.

# Part XXVII — Sampling: what it takes to turn an energy function into conformations

## 117. The Metropolis chain, and why the partition function cancels

`RequestProject/Metropolis.lean` builds the sampler in the finite setting. Given a target `p` and a
symmetric proposal `q`, the kernel `mhK` proposes a move and accepts it with probability
`min(1, p(j)/p(i))`, leaving the rejection mass on the diagonal. Three facts justify it: the kernel
is stochastic (`mhK_stochastic`), it satisfies detailed balance with respect to `p`
(`mhK_detailedBalance`, from the elementary identity `a·min(1,b/a) = min(a,b)`), and hence — through
`Kinetics.stationary_of_detailedBalance` of Part VII — the target is stationary (`mhK_stationary`).

The design point is `mhK_smul`: rescaling the target by any positive constant leaves the kernel
*unchanged*, because only ratios enter. So `mhK_of_unnormalized`: the chain built from the
normalized Boltzmann populations `exp(−βE)/Z` is the same chain as the one built from the bare
weights `exp(−βE)`, and `acc_boltzmann` shows the acceptance probability is a function of the energy
*difference* alone. The exponentially long sum of Part XXVI never has to be evaluated: an
unnormalized energy is enough to generate conformations.

## 118. The bottleneck lemma: what sampling still costs

`RequestProject/Mixing.lean` prices the time. For any transition kernel and any region `S` of
conformation space from which the one-step escape probability is at most `eps`, one step raises the
population outside `S` by at most `eps` (`massOut_evolve_le`); by induction, after `t` steps started
inside `S` the population outside is at most `t·eps` (`massOut_iterate_le`), so reaching a
population `m` outside takes at least `m/eps` steps (`steps_needed`).

Applied to the smallest honest caricature of a rugged landscape — three states, two degenerate wells
separated by a barrier of height `B`, nearest-neighbour proposals — the Metropolis escape
probability from the initial well is exactly `e^{−βB}/2` (`barrier_escape_le`), so the population of
everything else after `t` steps is at most `t·e^{−βB}/2` (`barrier_mass_le`) and equilibrating the
two wells takes at least `e^{βB}` steps (`barrier_steps_needed`). This is the discrete Markov-chain
Monte Carlo counterpart of the continuous-time broken ergodicity of §110: a finite run is not the
Boltzmann ensemble, and the gap is exponential in the barrier.

## 119. The Part XXVII capstone

`IDR.sampling_laws` (`PartTwentySeven.lean`) bundles the four statements. With Part XXVI they
complete the computational half of the specification. A model of a disordered region must come with
a sampler; the sampler needs only unnormalized weights, so an energy function *is* usable; but the
ensemble a finite run produces is the ensemble the model may claim — the Boltzmann ensemble of its
energy function only if the run is long compared with the exponential of its barriers. Cheap
evaluation and honest sampling are separate requirements, and both belong in the specification.

# Part XXVIII — Local restraints: what a coupling and an exchange rate can pin down

## 120. A scalar coupling is two moments of the torsion distribution

Parts IX and IX.2 treated the global experiments. The two local, residue-resolved restraints that
carry most of the remaining experimental weight are three-bond scalar couplings and amide hydrogen
exchange, and each constrains the ensemble far less than it is usually taken to.

`RequestProject/Karplus.lean` starts from the Karplus relation `J(θ) = A cos²θ + B cos θ + C` and
the fact that, in fast exchange, the measured number is the population average. Then
`avgJ_eq` : the measurement is exactly `A⟨cos²θ⟩ + B⟨cos θ⟩ + C`. The torsion distribution enters
through *two numbers*, and `avgJ_congr_of_moments` draws the consequence: any two ensembles that
agree on those two moments agree on every Karplus parametrisation simultaneously, so measuring more
couplings of the same torsion — other nuclei, other parametrisations — adds no information at all.
Two further degeneracies follow. `avgJ_reflect`: the coupling is even in the torsion, so it cannot
distinguish `θ` from `−θ`, i.e. α_R from α_L. And `avgJ_eq_karplusOfCos_add_var`: the standard
single-angle inversion is biased by exactly `A·Var(cos θ)`, with `cosVar_pos_of_two` showing the
variance is strictly positive as soon as two populated conformers differ in `cos θ` — the bias is
proportional to precisely the quantity that makes the region disordered.

## 121. Torsion populations are not identifiable — and where they are

The degeneracy is exhibited, not merely asserted. `fiveAngles` is a five-basin torsion support with
cosines `1, √2/2, 0, −√2/2, −1`; `popA` puts `(1/4, 0, 1/2, 0, 1/4)` on it and `popB` puts
`(0, 1/2, 0, 1/2, 0)`. Both have `⟨cos θ⟩ = 0` and `⟨cos²θ⟩ = 1/2`, so
`karplus_two_ensembles_agree`: they predict identical couplings for *every* `A, B, C`, although one
places half its population in the `θ = π/2` basin and the other places none. Because the constraint
is linear in the populations, the whole segment between them is consistent
(`karplus_underdetermined_family`), and `karplus_any_population_consistent` states the conclusion in
the form a modeller needs: for any target population `u ∈ [0, 1/2]` of that basin there is an
admissible ensemble with that population reproducing every measured coupling.

The positive counterpart is `twoBasin_identifiable`, and it is the exact reason the classical
two-state analysis of a `³J` value is legitimate: with only two basins and a coupling that
distinguishes them, one measurement determines the populations uniquely. Two basins are the whole
budget.

## 122. Hydrogen exchange averages rates, not free energies

`RequestProject/HydrogenExchange.lean` treats the standard residue-resolved probe of local
stability. Under EX2 the observed rate is `k_int·⟨p_open⟩` and the protection factor is the
reciprocal of the mean openness (`protectionFactor_eq`). The reported free energy is therefore
`ΔG_app = −RT log⟨p_open⟩`, while the per-conformer stabilities are `ΔG_k = −RT log p_k`, and Jensen
for the concave logarithm gives `deltaGapp_le_mean`: `ΔG_app ≤ ⟨ΔG⟩`, always. An explicit 50:50
ensemble of a fully open state and a state open with probability `1/100` has a strict gap
(`deltaGapp_lt_mean_two`), so this is a bias, not a boundary case. `deltaGapp_le_of_weight` is the
sharp form and the exchange analogue of the `r^{-6}` minority-report bound of §IX.2: a conformer of
weight `w_k` and openness `p_k` caps the apparent free energy at `ΔG_k − RT log w_k`, whatever the
other conformers do — a 1% fully-open state limits the apparent stability to `RT log 100`, about
2.7 kcal/mol at 300 K, however stable the remaining 99% is.

## 123. And exchange is not a functional of the equilibrium ensemble

The Linderstrøm-Lang steady-state rate is `k_ex = k_op k_int/(k_cl + k_int)`. The EX2 reading
`k_int·K_op` is exact only in a limit, and the defect is computed exactly:
`kex_eq_EX2_sub` gives `k_int K_op − k_ex = k_int K_op · k_int/(k_cl + k_int)`. In the opposite
regime the measurement saturates at the opening rate (`kex_le_kop`) and stops reporting on stability
at all. The decisive statement is `equilibrium_does_not_determine_kex`: two amides with the *same*
opening equilibrium constant — hence the same equilibrium populations and the same Boltzmann
ensemble — exchange at different observed rates. No model that outputs only a distribution over
conformations can predict a hydrogen exchange experiment; either the forward model carries the
kinetics, or the data must be restricted to a verified EX2 regime.

## 124. The Part XXVIII capstone

`IDR.local_restraint_laws` (`PartTwentyEight.lean`) bundles the five statements. Local restraints
belong in an ensemble refinement as forward-modelled averages with their exact kernels, accompanied
by the degeneracy they leave behind: a coupling fixes two moments of `cos θ` and nothing more, and
an exchange rate fixes a rate average that is not the mean stability and not, in general, a property
of the equilibrium ensemble at all.

# Part XXIX — What the simulation actually reports: box, cutoff, error bars

## 125. The periodic cell compacts the region

Parts XX and XXVII priced the dynamics of generating an ensemble. Three systematic differences
remain between the ensemble a paper reports and the ensemble its model denotes, and all three are
properties of the protocol rather than of the force field.

`RequestProject/PeriodicBox.lean` proves the first from a single inequality. `sum_antivary_le` is a
weighted Chebyshev inequality: if `f` is antitone in `R` then `(Σ w R f)(Σ w) ≤ (Σ w R)(Σ w f)`,
i.e. `R` and `f` are negatively correlated under every nonnegative weighting. Hence
`wmean_tilt_le`: reweighting any ensemble by a positive factor that decreases with the chain
dimension can only decrease `⟨R⟩`. A periodic image interaction is exactly such a factor — larger
conformations sit closer to their own images and pay more — so `boxMean_le_freeMean`:
`⟨R⟩_box ≤ ⟨R⟩_∞` at every inverse temperature, for any ensemble whatsoever. The bias is strict in
an explicit two-state instance at every `β > 0` (`boxMean_lt_freeMean_two`), and it is monotone in
the box (`boxMean_mono_in_box`): the reported dimension of a disordered region increases with the
cell until the images are gone. A single box size cannot validate a reported dimension; the
box-size dependence has to be shown to be flat.

## 126. A truncated potential has no landscape beyond the cutoff

`RequestProject/Cutoff.lean` makes the second failure exact rather than rhetorical. A conformation
all of whose pair distances exceed `rc` has *zero* truncated interaction energy
(`truncEnergy_eq_zero_of_beyond`), so `cutoff_blind`: any two such conformations receive exactly
equal Boltzmann weight at every temperature. A cutoff model does not merely misestimate the
long-range part of the landscape; beyond `rc` it assigns none. That the true model does separate
them is exhibited in `cutoff_loses_true_ranking` with a Coulomb-like attraction and two separations
beyond the cutoff. Quantitatively, `truncation_error_le`: with a tail bound `|u(r)| ≤ C/r` the total
neglected energy of an `n`-site conformation is at most `n²C/rc` — quadratic in the length of the
region at fixed cutoff, which is why the truncation error of a long disordered polyelectrolyte
cannot be absorbed into a constant and why a lattice-sum treatment is not optional.

## 127. Frames are not samples

`RequestProject/CorrelatedSampling.lean` treats the statistical error, the one most often quoted
wrongly. For a stationary series with exponentially decaying autocorrelation `γ(k) = σ²ρ^k` — the
autocorrelation of the two-state exchange of §VIII.1 and of a Rouse mode sampled at interval `Δt`,
with `ρ = e^{−Δt/τ}` — the variance of the average of `N` frames is computed exactly:
`corrSum_eq` gives `Σ_{i,j<N} ρ^{|i−j|} = N(1+ρ)/(1−ρ) − 2ρ(1−ρ^N)/(1−ρ)²`, so
`varMean = σ²/N²` times that sum. Four readings follow. `varMean_eq_iid`: at `ρ = 0` this is the
textbook `σ²/N`. `varMean_frozen`: at `ρ = 1` it is `σ²` *independently of `N`* — a trajectory
shorter than the relaxation time contains exactly one sample however often it is written to disk.
`varMean_ge_iid`: in between, the true variance is never below the independent-sample formula, so
naive error bars are always optimistic. And `varMean_ge_inflated`: the inflation factor is the
statistical inefficiency `(1+ρ)/(1−ρ) ≈ 2τ/Δt`, up to an `O(1/N)` correction, which inverted
(`frames_needed`) is the number of frames an error bar costs — proportional to the relaxation time,
itself exponential in the barrier by §VIII.1 and §XXVII.

## 128. The Part XXIX capstone

`IDR.protocol_laws` (`PartTwentyNine.lean`) bundles the seven statements. Every capacity and
sample-complexity bound of Parts V–VII is stated in *independent* samples, and §127 is the
conversion factor from frames to samples; §125 and §126 are the two protocol parameters — cell size
and interaction cutoff — in which a reported ensemble must be shown to be stationary before it can
be attributed to the model rather than to the run that produced it.

# Part XXX — The entropy price of ordering

## 129. Conformational selection is an exact decomposition

Part XVI established the reciprocity between conformational populations and affinity, and §X
established that a binding constant is not a function of a mean structure with error bars. What
neither supplies is the accounting: how much free energy a region pays for being disordered when
it binds.

`RequestProject/Selection.lean` works in the standard finite-conformer formalism: free-state
populations `p` over `m` conformers, a binding-competent subset `S`, an interaction `−eps k`
gained on binding, and a bound-state conformational sum that runs over `S` only, so that
`ΔG = −kT log Σ_{k∈S} p_k e^{βeps_k}`. With a uniform interaction `e` over the competent set,
`deltaG_eq_selection` gives

    ΔG = −e + kT log (1/P_S),

the intrinsic interaction plus the free energy of the population restriction. The penalty is
strictly positive whenever the competent conformers are not the whole ensemble
(`selection_penalty_pos`) and is `kT log(1/P_S)` — about 1.4 kcal/mol per decade of population at
300 K. An affinity is therefore not a property of the bound structure: it carries a term that
belongs entirely to the free state.

## 130. The penalty is the conformational entropy

`penalty_uniform_eq_log_card` identifies it: a uniform free ensemble over `m` conformers that
binds through exactly one of them pays `kT log m`, which is `kT` times the Gibbs–Shannon entropy
of the free state (§FreeEnergy). Two models that agree on the predicted bound structure and
disagree on the breadth of the free-state ensemble predict affinities differing by exactly the
entropy they disagree on. This is the quantitative reason a binding model for a disordered region
cannot be validated on bound-state geometry alone.

## 131. Fuzziness pays it back, and averages understate binding

Three consequences. `deltaG_antitone_subset`: if the complex tolerates a larger set of
conformers, the binding free energy can only go down — a fuzzy complex is the thermodynamically
favoured arrangement whenever the interaction survives it, not a defect of the model or of the
experiment. `deltaG_le_neg_mean`: by Jensen for `exp`, the exact binding free energy is at most
the population average of the per-conformer interaction energies, so scoring a designed binder by
a mean interaction energy over an ensemble is systematically conservative. And
`deltaG_le_of_conformer`: one competent conformer of population `p_k` and interaction `eps_k`
guarantees `ΔG ≤ −eps_k + kT log(1/p_k)` whatever the rest of the ensemble does — a 1% conformer
presenting a 10 kcal/mol interface already binds at least as well as −7.3 kcal/mol at 300 K. This
is the binding counterpart of the `r^{-6}` minority-report bound of §IX.2 and the hydrogen
exchange bound of §122: rare conformers dominate exponentially weighted observables.

## 132. The Part XXX capstone

`IDR.selection_laws` (`PartThirty.lean`) bundles the six statements. An affinity prediction for a
disordered region is a difference of two *ensemble* free energies. What a model must report is
therefore the free-state populations of the binding-competent conformers together with their
interaction energies — not a bound pose, and not a mean structure with error bars.

# Part XXXI — The scaling exponent is fitted, not measured

## 133. What a two-point log–log fit returns

The Flory scaling exponent is the single number most often quoted for a disordered region, and it
is never measured directly: it is inferred from a handful of chain lengths, through a relation
that carries corrections to scaling, `R_g² = A N^{2ν}(1 + B/N)`, which are not small at the
lengths of real disordered regions.

`RequestProject/ScalingExponent.lean` computes the standard two-point estimator exactly.
`nuHat_eq`: the fitted exponent is

    ν̂ = ν + (log(1 + B/N₂) − log(1 + B/N₁)) / (2 log(N₂/N₁)),

the true exponent plus a term built entirely from the correction. Three consequences.
`nuHat_lt_of_pos_correction`: a positive correction amplitude biases the apparent exponent
*down*, so the region looks less swollen than it is — an apparent `ν` of 0.54 where the truth is
0.6 is what a positive correction to scaling looks like; `nuHat_gt_of_neg_correction` gives the
mirror statement. `nuHat_bias_le`: the bias is at most `B/(2 N₁ log(N₂/N₁))`, so it shrinks only
with the *shortest* chain in the series and with the lever arm in log length — adding one longer
construct helps far less than removing the shortest one.

## 134. And why the fit quality cannot defend it

`two_point_fit_exact` is the uncomfortable statement: a pure power law with the fitted exponent
and *no* correction term reproduces both measurements exactly. Two lengths therefore contain no
information at all about the correction to scaling, and no goodness-of-fit argument on them can
support the exponent that was read out. A reported exponent is a statement about the lengths used
and the correction model assumed, not a measured property of the sequence.

## 135. The Part XXXI capstone

`IDR.scaling_laws` (`PartThirtyOne.lean`) bundles the four statements. The design consequence is
the one Part IX drew for every other observable: compare the model with the measurements — the
radii at the lengths actually studied, forward-modelled — rather than with a derived quantity
whose bias has a known sign and cannot be detected in the data it was derived from.

## 136. Dynamics: what a relaxation rate can see of the motion (Part XXXII)

Everything so far constrains *which* conformations a model must carry and with what weights.
The other half of what an experiment on a disordered region reports is *how fast* the chain
moves, and that is measured by NMR spin relaxation. The point of `SpinRelaxation.lean` is that
motion enters the measured rates `R₁`, `R₂` and the heteronuclear NOE through exactly one
object, the spectral density

    J(ω) = Σ_k w_k · 2τ_k / (1 + (ω τ_k)²),

a finite mixture of Lorentzians over correlation times `τ_k` with populations `w_k`
(`specDens`). Three exact facts about it decide what the experiment can report.

First, `specDens_zero`: `J(0) = 2⟨τ⟩` is twice the *mean* correlation time. It is therefore an
arithmetic average of timescales, and `specDens_ge_term` says that every component contributes
its full share: a slow minority state cannot be hidden. `minority_slow_state_dominates` makes
that concrete — a population of one per cent that moves a thousand times slower than the bulk
already raises `J(0)` above ten times the bulk value. Transverse relaxation in a disordered
region is a report about its slowest excursions, not about its typical motion.

Second, `lorentz_le_inv_freq`: at frequency `ω` a Lorentzian never exceeds `1/ω`, and it attains
that value exactly when `ωτ = 1`. Relaxation is a band-pass filter centred on the Larmor
frequency, blind on either side. Worse, it is two-to-one: `lorentz_tau_ambiguity` and
`lorentz_eq_iff` prove that `τ` and `1/(ω²τ)` give exactly the same value, so a single rate does
not distinguish a slow motion from the matching fast one. This is the classical `τ_c` ambiguity,
and here it is an identity rather than a rule of thumb.

## 137. And what a one-field data set leaves open

`relaxation_underdetermined` is the negative result. Two motional models over the same four
correlation times, both with strictly positive populations, predict *identical* spectral
densities at three distinct frequencies — as many independent numbers as `R₁`, `R₂` and the NOE
provide at one magnetic field — while their mean correlation times, and hence their values of
`J(0)`, differ by `77/240` of the unit. No fit to a one-field data set can prefer one over the
other, and the difference between them is precisely the quantity such a fit is used to report.

The standard repair is to fit a model with few parameters, and `modelFree_eq_specDens` identifies
what that repair is: the Lipari–Szabo "model-free" spectral density is *exactly* the
two-component case of the same mixture, with populations `S²` and `1 − S²` and correlation times
`τ_m` and the harmonic combination `τ_m τ_e/(τ_m + τ_e)`. An order parameter is a fitted
population of a two-state motional model, and `modelFree_underdetermined` exhibits two pairs
`(S², τ_e)` a measurement cannot tell apart. On the positive side, `twoComponent_identifiable`
shows that two *known* correlation times, provided their product is not `1/ω²`, are identified
from one frequency: identifiability is a property of the model one has assumed, not of the data
one has taken.

## 138. The invisible state, and how invisible it can be

`ChemicalExchange.lean` treats the experiment designed to see minority conformations directly:
CPMG relaxation dispersion. Under the standard fast-exchange (Luz–Meiboom) forward model — an
assumption of the file, stated as such — the effective transverse rate is

    R₂,eff = R₂⁰ + (Φ_ex/k_ex)·(1 − 2 tanh(k_ex t_cp/2)/(k_ex t_cp)),   Φ_ex = p_A p_B Δω².

The shape is pinned down exactly: the excess relaxation is strictly positive and strictly below
the plateau `Φ_ex/k_ex` (`disp_nonneg`, `R2eff_lt_plateau`); at short cycle times the exchange is
refocused and the excess vanishes linearly in `t_cp` (`R2eff_sub_le_short_cycle`, proved from
`tanh y ≤ y` and `y ≤ sinh y`); at long cycle times the profile approaches the plateau within
`2Φ_ex/(k_ex² t_cp)` (`plateau_gap_le`).

What the profile determines is a genuinely positive result. `population_lower_bound`: since the
chemical-shift difference cannot exceed the spectral range, a measured `Φ_ex` forces
`p_B ≥ Φ_ex/Δω_max²`. Exchange broadening is evidence that a minority state exists, and it puts
a floor under its population.

What it does not determine is how much of it there is. The population and the shift enter only
through the product `p_A p_B Δω²` (`R2eff_congr_of_phiEx`), so
`invisible_state_population_unidentifiable` produces, for *every* population in `(0, 1/2]`, a
chemical-shift difference reproducing the whole measured profile at every cycle time. A quoted
`p_B` is a consequence of the assumed `Δω`. And `arbitrarily_populated_invisible_state` closes
the loop: at fixed amplitude the entire profile is squeezed into a band of height `Φ_ex/k_ex`, so
for any population, any shift and any detection threshold there is an exchange rate at which the
state contributes less than the threshold everywhere. Disordered regions, which interconvert
fast, are exactly the regime in which a heavily populated state is invisible. `IDR.nmr_dynamics_laws`
(`PartThirtyTwo.lean`) bundles the eight statements.

## 139. Single-molecule histograms: how much of the width is the molecule? (Part XXXIII)

A broad single-molecule FRET histogram is the most frequently cited direct evidence that a
region is heterogeneous. `PhotonCounting.lean` computes how much of the width belongs to the
molecule, from the photon statistics up: a burst is `N` photons, each detected in the acceptor
channel with the transfer efficiency of the conformation that emitted it, so the acceptor count
is binomial (`binom`). Its normalisation, mean and variance are obtained from Mathlib's
Bernstein-polynomial identities (`binom_sum`, `binom_mean`, `binom_var`), so nothing about the
detector is assumed beyond counting.

The two results are an equality and a warning. `measured_unbiased`: the mean of the measured
efficiencies is exactly the ensemble mean efficiency, for every burst size — the histogram is
centred correctly. `shotnoise_decomposition`: the variance of the measured efficiencies is
exactly

    Var(measured) = varConf + (Σ_k w_k e_k(1 − e_k))/N,

the conformational variance plus a shot-noise term. The second term does not vanish with more
molecules; it vanishes only with more photons per molecule.

`homogeneous_histogram_has_width` draws the consequence: a completely homogeneous ensemble — one
conformation, `0 < e < 1` — produces a histogram of strictly positive width `e(1−e)/N`. And
`width_not_evidence_of_heterogeneity` makes the ambiguity quantitative: a single conformation at
`e = 1/2` observed with 100 photons per burst, and a genuine two-state ensemble at `e = 0.46`
and `0.54` observed with 276, produce histograms of *exactly* the same variance `1/400`, though
their conformational variances are `0` and `1/625`. Only the photon budget separates them.

The positive statements come from the same identity. The shot-noise term never exceeds `1/(4N)`
(`shotNoise_le_quarter`), so any width beyond `1/(4N)` is a lower bound on the conformational
variance (`heterogeneity_detected`), and a budget of `N ≥ 1/(4·varConf)` photons per burst puts
the detector term below the ensemble term (`photons_needed`) — the cost of resolving a
sub-population grows as the inverse square of the efficiency splitting it produces.

Finally `dynamic_averaging`, which cuts the other way. A burst is not instantaneous. If the
chain interconverts faster than the burst lasts, its photons come from several independently
sampled conformations and the burst reports their average: with two independent halves the
histogram width becomes `varConf/2 + shotNoise`, the conformational term halved and the detector
term untouched. A *narrow* histogram is therefore no more evidence of homogeneity than a broad
one is of heterogeneity — it may only mean that the exchange is fast on the burst timescale.
`IDR.single_molecule_laws` (`PartThirtyThree.lean`) bundles the six statements.

## 140. Association kinetics: what fly-casting would have to mean (Part XXXIV)

The last part addresses the kinetic claim usually made for disorder: that an extended,
fluctuating chain binds faster because it presents a larger capture radius. In the
diffusion-limited regime the claim is decidable, by composing Smoluchowski capture
`k = 4π D R_c` with the Stokes–Einstein relation `D = kT/(6πη R_h)` already used in Part X.2.
`smoluchowski_stokes` gives the exact rate

    k = (2kT/3η) · (R_c/R_h).

The two radii enter only through their ratio, and the prefactor contains no property of the
chain at all. Three consequences follow immediately. `rate_scale_invariant`: a chain that swells
at fixed shape binds at exactly the same rate, however large it becomes. `flycasting_iff`: a
speed-up occurs if and only if the capture radius grows strictly faster than the hydrodynamic
radius. `no_flycasting_from_scaling`: if both radii obey the same scaling law `∝ N^ν` — as they
do for ideal and self-avoiding chains alike — the diffusion-limited rate does not depend on chain
length at all. A fly-casting effect must therefore come from a capture radius set by something
other than the chain's own size: electrostatic steering, a large target, or a reaction that can
be initiated anywhere along the contour. A model that predicts on-rates has to say which.

`ensembleRate_eq` records that rates average over the ensemble, and
`rate_not_determined_by_apparent_size` shows what that does to a surrogate structure. For an
equally weighted two-conformer ensemble with radii `1` and `3`, the single structure carrying
the ensemble's *measured* hydrodynamic radius — the harmonic mean `3/2` of Part X.2 — and its
*mean* capture radius `2` binds `4/3` times too fast. The experiment averages the two radii
differently, so no single structure fitted to both reproduces the kinetics; the error has a
determined sign. `selection_scales_rate` and `selection_slows` add the kinetic counterpart of
Part XXX's entropy penalty: conformational selection multiplies the rate by the competent
population. `IDR.association_laws` (`PartThirtyFour.lean`) bundles the six statements.

## 141. Titration curves: the cooperativity is fitted, not measured (Part XXXV)

The `m`-value of a chemical denaturation curve is the last of the derived numbers a disordered
region is routinely characterised by, and it is quoted as a cooperativity. `Denaturant.lean`
computes what the two-state linear-extrapolation fit actually returns when the underlying signal
is a gradual expansion rather than a transition.

The two-state curve is the logistic `frac dG m RT x = sigmoid((m x − ΔG)/RT)`. `frac_midpoint`
puts its half-point at `x = ΔG/m`; `frac_deriv_midpoint` gives its slope there as exactly
`m/(4RT)`; and `m_eq_four_RT_slope` states the consequence in the form a spectroscopist meets it:
the reported `m` is four `RT` times the measured midpoint slope, no more and no less. A fitted
`m`-value repackages one point and one slope of the data.

`fit_matches_any_curve` turns that into a negative result: for every midpoint and every positive
slope there is a two-state model agreeing with the data there in value *and* in slope. A fit that
reproduces the transition midpoint and its steepness is no evidence whatsoever that there are two
states. `sigmoid_tangent_cubic` strengthens it beyond first order: the logistic curve never
departs from its own midpoint tangent by more than `|u|³/48`, where `u = m(x − x½)/RT` — proved
from `tanh y ≤ y` and the cubic lower bound `y − y³/3 ≤ tanh y`, both derived here from the
monotonicity of explicit auxiliary functions. Translating back (`frac_close_to_linear`), a
strictly linear, non-cooperative expansion of a disordered chain is reproduced by a two-state
model with `m = 4RT·s` to within `(4s|x − x₀|)³/48`. The discrepancy a titration must resolve, in
the presence of noise, before "cooperativity" is warranted is a cubic one.

`frac_mem_Ioo` isolates the only qualitative difference: the two-state curve is confined to
`(0,1)` while a gradual expansion is not, so the discriminating information lives at the ends of
the titration — exactly where the pre- and post-transition baselines are fitted, and therefore
exactly where the fit is least constrained. `IDR.titration_laws` (`PartThirtyFive.lean`) bundles
the five statements. The design consequence is the same one Part XXXI drew for the scaling
exponent: compare a model with the titration curve itself, forward-modelled, and not with a
`ΔG` and an `m`-value.

## 142. Circular dichroism: which secondary-structure content is determined (Part XXXVI)

The most frequently reported experiment on a disordered region is a far-UV CD spectrum, and the
most frequently reported *numbers* are the helix, sheet, turn, polyproline II and coil
percentages returned by deconvolving it. `Dichroism.lean` asks what those percentages are a
statement about, in the forward model every deconvolution program uses: the spectrum is the
mixture `mix B f i = ∑ⱼ fⱼ Bⱼᵢ` of fixed reference spectra `B` sampled at the measured
wavelengths, and `f` is nonnegative and sums to one.

The organising notion is a *null direction* (`IsNull`): a change of composition `d` with
`∑ⱼ dⱼ = 0` and `∑ⱼ dⱼ Bⱼᵢ = 0` at every wavelength. Along such a direction the data do not move
at all — `mix_perturb` and `sum_perturb` say that both the predicted spectrum and the
normalisation are *exactly* constant along `f + t·d` — so no amount of data quality distinguishes
the members of that family. The question is then whether null directions exist, and the answer is
that they exist for arithmetic reasons: `exists_null` builds one whenever the number of
wavelengths plus one is smaller than the number of classes, by exhibiting the map
`d ↦ (∑ⱼ dⱼ, mix B d)` as a linear map into a space of too small a dimension. Nothing about the
particular reference spectra enters.

## 143. And why that underdetermination is not a boundary effect

A natural reply is that the fit is constrained by nonnegativity: fractions cannot go below zero,
and perhaps the constraint pins the answer down. `segment_of_null` refutes this in general. Given
any *strictly positive* composition and any nonzero null direction, an explicit step size —
`t = (minⱼ fⱼ)/(1 + ∑ⱼ|dⱼ|)`, small enough to keep every fraction nonnegative — produces a
genuinely different composition with the same total and the same spectrum. Underdetermination
reaches into the interior of the simplex, not only its faces.

`two_wavelength_witness` makes this concrete with a four-class basis of realistic ellipticities
sampled at 222 nm and 208 nm: two nonnegative, normalised compositions whose β-sheet contents
differ by more than 13 percentage points give bit-identical spectra. The witness is deliberately
extreme in its channel count, but the mechanism is the one that operates in real deconvolutions,
where the effective rank of a reference basis is far below the number of wavelengths recorded.

The case that matters most for a disordered region is sharper still. Polyproline II and
statistical coil have nearly identical far-UV reference spectra. `swapDir_isNull` and
`equal_basis_split_free` show that if they were *exactly* identical, the PPII/coil split would be
a completely free parameter: every division of their combined weight fits the data equally.
Because they are only nearly identical, `near_degenerate_tolerance` prices the residual
information: if two reference spectra differ by at most `ε` at every wavelength, transferring
weight `t` between the two classes perturbs the prediction by at most `|t|·ε`, so a data
tolerance `η` admits every split with `|t| ≤ η/ε`. A reported PPII content is a statement about
the noise floor and the regulariser as much as about the protein.

What survives is characterised exactly. `readout_determined` shows that a weighted content
`∑ⱼ wⱼ fⱼ` is the same for *every* fitting composition as soon as the weight vector is an affine
read-out of the reference spectra, `wⱼ = c₀ + ∑ᵢ cᵢ Bⱼᵢ`. Secondary-structure content is
identifiable precisely to the extent that it is a linear functional of the measured spectrum.
`IDR.dichroism_laws` (`PartThirtySix.lean`) bundles the six statements, and the design
consequence repeats the one drawn for the scaling exponent and the `m`-value: a model should
predict the spectrum and be compared with the spectrum.

## 144. Aggregation kinetics: there is no lag phase (Part XXXVII)

Disordered regions are the parts of a proteome that aggregate, and the standard experiment is a
thioflavin curve with its flat lag, steep growth and plateau. `Aggregation.lean` works in the
exactly solvable early-time nucleation–elongation model — fibril mass obeying `M'' = κ²M` with
`M(0) = 0` and `M'(0) = v`, where `v` is set by primary nucleation and `κ` by elongation and
secondary nucleation — whose solution is `mass v κ t = (v/κ)·sinh(κt)`. `mass_hasDerivAt` and
`mass_second_deriv` verify that this is the stated initial value problem, so the two parameters
mean what they are said to mean.

The first result is negative and immediate: `mass_pos` shows the mass is strictly positive at
*every* positive time. There is no lag phase in the model. The flat portion of a measured trace
is the interval on which the mass is below the detection threshold, and the "lag time" is a
property of the assay as much as of the chemistry. What the curve does determine is given by
`mass_invariant`: the model carries the exact constant of the motion `(M')² − κ²M² = v²`, so one
time point with its slope fixes the nucleation flux once the growth rate is known.

## 145. And why a lag time is a logarithm

`mass_lagTime` computes the crossing time of a threshold `M_c` exactly:
`lagTime = arsinh(κM_c/v)/κ`. The nucleation flux enters through an inverse hyperbolic sine —
that is, through a logarithm — and the consequences are the two insensitivity laws. Multiplying
the detection threshold by `r ≥ 1` delays the apparent lag by at most `log r/κ`
(`lag_shift_le`), and multiplying the nucleation rate by `r ≥ 1` shortens it by at most the same
amount (`lag_rate_shift_le`); both follow from the inequality `arsinh(rx) ≤ log r + arsinh x`,
proved here from `√(1 + r²x²) ≤ r√(1 + x²)`. Nor is the dependence any weaker than logarithmic:
`lag_shift_ge` gives the matching lower bound `(log r − log(3/2))/κ` once the threshold is at or
above `v/κ`, from `log(2rx) ≤ arsinh(rx)` and `arsinh x ≤ log(3x)`. A model of the soluble
ensemble that predicts the nucleation rate to within a factor of a hundred therefore predicts the
lag time to within `4.6/κ`; and a lag time measured to ten per cent constrains the nucleation rate
only to within a large multiplicative factor.

The last pair of results says that the lag time alone constrains nothing at all.
`lag_underdetermined` shows that for *every* growth rate `κ` there is a nucleation flux,
`v = κM_c/sinh(κT)`, reproducing an observed lag time `T` exactly, and `two_models_one_lag`
exhibits two such fits — `κ = 1` and `κ = 2` at unit threshold and unit lag — with strictly
different fluxes, the comparison resting on `sinh 2 = 2 sinh 1 cosh 1` and `cosh 1 > 1`. Lag time
and growth rate have to be reported together. `IDR.aggregation_laws` (`PartThirtySeven.lean`)
bundles the seven statements: an ensemble model is neither confirmed nor falsified by a lag time,
and the nucleation rate it predicts enters that observable only through its logarithm.

## 146. Density maps: why a disordered region is missing (Part XXXVIII)

The operational definition of a disordered region, in the structural databases, is negative: it
is the stretch of chain for which the crystallographic or cryo-EM map shows no interpretable
density. `Density.lean` asks what that absence is a statement about, using the standard harmonic
treatment of a smeared atom — occupancy `q`, isotropic displacement parameter `B`,
structure-factor contribution `formFactor q B s = q·exp(−Bs²/4)`, real-space Gaussian peak height
`peak q B = q·(4π/B)^{3/2}`.

In reciprocal space the situation is the familiar one of this development. A single resolution
shell determines nothing: `single_shell_degenerate` exhibits, for *any* assumed occupancy `q'`,
the displacement parameter `B + 4·log(q'/q)/s₀²` that reproduces the measured amplitude exactly.
Two shells do determine both — `two_shells_identify` recovers `q` and `B` from amplitudes at any
two distinct `s²`, by taking logarithms and subtracting — so the model is identifiable in
principle. The interesting result is the quantitative one in between. `formFactor_close` shows
that two models agreeing at a shell `s₀` differ at a higher shell by at most
`F₀·|B − B'|·(s² − s₀²)/4`, where `F₀` is their common amplitude; the proof factors the
contribution as `F₀·exp(−B(s² − s₀²)/4)` and uses `|eˣ − eʸ| ≤ |x − y|` for nonpositive exponents,
itself derived from `1 + x ≤ eˣ`. Separating occupancy from disorder therefore demands data
quality proportional to the *span* of resolution actually measured, which is exactly what a
region whose scattering has already decayed away cannot supply.

## 147. And what "no density" actually bounds

In real space the degeneracy is not merely severe but exact. `peak_scale_invariant` states the
classical occupancy–`B` correlation as an identity: for every `c > 0`,
`peak (c³q) (c²B) = peak q B`. Raising the occupancy by `c³` and the displacement parameter by
`c²` leaves the map unchanged. A side chain modelled at quarter occupancy and a fully occupied
but mobile one are, to this model, the same density.

`peak_antitone` records that disorder costs peak height monotonically, and `peak_hundred` gives
the rate in the units a crystallographer uses: at unchanged occupancy, `B = 100 Å²` yields one
eighth of the peak at `B = 25 Å²`. The operational definition then becomes a threshold statement.
`visibility_bound` shows that a peak reaching a contour level `τ` forces `B³τ² ≤ (4π)³q²` —
visible density is a bound on the combination `q²/B³` and on nothing else — and
`invisible_of_large_B` gives the converse: any displacement parameter with `(4π)³q² < B³τ²`
produces no peak at that contour, whatever the occupancy. "No density" is not the absence of a
residue; it is an inequality relating occupancy, mobility and the contour level chosen.

`IDR.density_laws` (`PartThirtyEight.lean`) bundles the seven statements. The design consequence
is the one that motivates the entire development from the beginning: a model of a disordered
region cannot be validated against a deposited structure, because the deposited structure is
precisely the part of the molecule that is not the disordered region. What a map can be compared
with is a density forward-modelled from the ensemble, at the occupancy and displacement
parameters the ensemble itself implies — and even then the comparison constrains only `q²/B³`
unless the resolution span is wide enough to separate the two.

## 148. The entropy per residue, bracketed above (Part XXXIX)

Part XII established that a self-avoiding chain on the square lattice has a well-defined
conformational entropy per residue — the connective constant `μ = lim (log cₙ)/n`, existing by
Fekete's lemma from the submultiplicativity of the conformation count — and bracketed it between
`log 2` and `log 4`. The upper end of that bracket is the ideal-chain value itself, and it is
worth asking how much excluded volume really costs. `ConnectiveBound.lean` answers with a
sharper bound.

The natural comparison is not `log 4` but `log 3`. The crudest excluded-volume rule — a chain may
not immediately retrace the bond it has just laid down — leaves three continuations of each bond,
so at most `4·3^{n−1}` conformations and an entropy per residue of `log 3` — an elementary count,
quoted as the point of comparison rather than formalised. `cnt_seven` computes
the exact seven-bond conformation count, `c₇ = 2172`, by exhaustive kernel enumeration of all
`4⁷ = 16384` bond sequences (a kernel computation, so no extra axioms). Since `2172 < 2187 = 3⁷`
and every chain length gives an upper bound on `μ`, `connectiveConstant_lt_log_three` concludes
`μ < log 3`. Real excluded volume costs strictly more entropy per residue than the no-retraction
rule alone; the interactions that matter are not confined to the previous bond.

## 149. And bracketed below, by Hammersley's bridges (Part XL)

The lower bound of Part XII, `log 2`, came from counting *directed* chains — those built from two
bond directions that both increase a coordinate, which are self-avoiding for free. That argument
cannot see any conformation that ever steps sideways, and no refinement of it will, because at
most two of the four square-lattice bonds can increase a single additive functional.
`Bridge.lean` removes the limitation with the standard device.

A *bridge* for an additive functional `phi` of position (`IsBridge`) is a self-avoiding chain
every site of which, after the first, has `phi > 0`, and none of which exceeds the value at the
far end. The point of the definition is `isBridge_append`: two bridges concatenate to a bridge.
The first chain lies weakly below its own endpoint in `phi` and the second strictly above its
start, so no site of one can equal a site of the other; the proof splits the site list of the
concatenation as `sites l₁ ++ (translated tail of sites l₂)` and compares `phi` across the
junction. Consequently bridge counts are *super*multiplicative
(`brCntOf_supermultiplicative`) — the exact opposite of the submultiplicativity that gives the
upper bounds — and iterating gives `brCnt N ^ k ≤ brCnt (kN) ≤ cnt (kN)`. Since the entropy per
residue is a limit, `log_brCntOf_div_le_connectiveConstantOf` converts this into a lower bound
from a *single* finite count: `μ ≥ (log brCnt N)/N` for every `N`.

Applied to the square lattice with `phi` the horizontal coordinate, exhaustive enumeration gives
`brCnt_six : brCnt 6 = 101`, and hence `μ ≥ (log 101)/6 = 0.769…`, strictly better than the
directed bound `log 2 = 0.693…` (`log_two_lt_log_bridge_div_six`). Together with Part XXXIX,
`IDR.entropy_per_residue_laws` (`PartForty.lean`) records the tightened bracket

  `(log 101)/6 ≤ μ < log 3`,   i.e.  `0.769 ≤ μ < 1.099`,

both ends strictly interior to the ideal-chain value `log 4 = 1.386`. The design consequence is
quantitative rather than qualitative: the conformational freedom that a generative model of a
disordered region must reproduce is a definite amount, certified from both sides by finite
computations, and a model calibrated on ideal-chain branching over-counts it exponentially in the
length of the region.

## 150. Evolution: the sequence diverges, the descriptor does not (Part XLI)

A model of a disordered region is trained on the regions we have sequenced and asked about the
ones we have not, and the sequences it must generalise across are, in this class of proteins,
barely alignable. `Evolution.lean` asks what that costs, using the mean-field Debye–Hückel
descriptor of `Electrostatics.lean` as the concrete conserved physics.

The reason a descriptor can be conserved while a sequence is not is a symmetry. Every descriptor
assembled from pairwise terms `q_i q_j w(|i−j|)` is unchanged by reading the chain backwards,
because reversal preserves every separation; `pairSum_rev` proves this once and for all, and
`screenedEnergy_rev`, `scd_rev` and `netCharge_rev` specialise it to the screened electrostatic
energy at *every* salt concentration, the sequence charge decoration and the net charge. The
restriction is real: `headCharge_not_rev_invariant` gives a directional descriptor — the net
charge of the N-terminal half — that reversal changes, so the invariance statements say something
specific about the level of description, and the directional physics of a real backbone is
outside their scope by construction.

The consequence is stark. `diverged_sequences_same_physics` shows that the block polyampholyte
`+^m −^m` and its reversal agree at **no** position — sequence identity exactly `0`
(`block_rev_identity_zero`) — and have identical screened energy at every salt concentration,
identical charge decoration, identical net charge. Percent identity is not a weak proxy for the
conserved quantity; the two can be at opposite extremes at once. Nor is the neutral set small:
`composition_class_card` counts the zero-net-charge composition class of a `2m`-residue chain as
`C(2m,m)`, `composition_class_exponential` gives `4^m < m·C(2m,m)`, and
`training_set_covers_vanishing_fraction` turns that into the statement that a training set of `T`
sequences covers at most a fraction `T·m/4^m` of one composition class. Generalisation across
evolution cannot be memorisation.

What follows for the architecture is also a theorem, and it is a positive one.
`asymmetry_forces_error` shows that against a target with a symmetry, a model without it has two
errors summing to at least its own asymmetry, so `asymmetry_error_half` gives an error of at
least half the asymmetry on one of the two sequences; conversely `symmetrised_error_le` and
`symmetrised_error_lt` show that averaging the model over the group never increases the squared
error and strictly decreases it whenever the model is asymmetric. Building the neutral group of
the target into the model is free, and omitting it is not. `IDR.neutral_evolution_laws`
(`PartFortyOne.lean`) bundles the five statements.

## 151. Synthesis: the nascent chain is not the free chain (Part XLII)

Every disordered region exists first as a nascent chain leaving the ribosome; every model of one
is trained against the mature chain at equilibrium. `Cotranslational.lean` prices the two
distinct errors.

The first is a lag behind a moving target. Let `p t` be the actual distribution after `t`
elongation events and `pi t` the equilibrium ensemble of the `t`-residue chain. Assume the
standard spectral-gap input — one elongation step contracts the distance to the *current*
equilibrium by `delta < 1` — and that each elongation moves the equilibrium by at most `d`. Then
`tracking_error_le` gives `dist(p t, pi t) ≤ delta^t·dist(p 0, pi 0) + d(1−delta^t)/(1−delta)`
and `tracking_error_le_steady` the steady form with `d/(1−delta)`. This is not a loose estimate:
`sharp_saturates` constructs a chain satisfying every hypothesis with equality whose lag is
exactly `d(1−delta^t)/(1−delta)`, and `sharp_tracking_error_tendsto` shows it converges to
`d/(1−delta)`. Writing `delta = e^{-lambda·tau}` for a chain relaxing at rate `lambda` in codon
time `tau`, `quasi_static_limit` recovers the quasi-static picture — the bound tends to `d`, a
single elongation step of drift, as synthesis slows — and `lag_bound_ge_of_fast_synthesis` bounds
the same quantity below by `d/(lambda·tau)`, which diverges as synthesis outruns relaxation.

The second error does not go away in that limit. The equilibrium of the emerged fragment is not
the marginal of the mature ensemble, because the partners that have not been synthesised are
absent from the Hamiltonian, not merely improbable. In the two-state caricature `vectorial_gap`
computes the difference in the population of one fragment conformation exactly as
`(e^{beta·eps} − 1)/(2(e^{beta·eps} + 1))`; `vectorial_gap_pos` makes it positive for every
stabilising contact and `vectorial_gap_tendsto_half` shows it saturates at `1/2`, the whole
population. `mature_model_error_on_nascent` reads that as the error carried by a model that is
exact on the mature chain. `IDR.cotranslational_laws` (`PartFortyTwo.lean`) bundles the five
statements: co-translational measurements are a different observable, not a consistency check on
the same one.

## 152. The material state: what condensate rheology excludes (Part XLIII)

The condensates that disordered regions form are modelled as liquid droplets — a viscosity, a
relaxation time, Stokes–Einstein diffusion inside — and measured to be broadly power-law,
subdiffusive and ageing. `Rheology.lean` proves what each of those measurements excludes.

Take the relaxation modulus of a finite Maxwell spectrum, `G(t) = Σ_k g_k e^{-t/tau_k}` — what a
coarse-grained model with finitely many slow variables produces. `maxwell_le_exp` bounds it by
`G(0)·e^{-t/T}` with `T` the slowest mode, and `maxwell_lt_power_law` concludes that for every
power law `C·t^{-alpha}` the model eventually lies strictly below it: the discrepancy is
one-sided, the model always too fast. The mismatch is categorical, not quantitative, at the level
of transport coefficients: `maxwell_integrableOn` and `maxwell_viscosity` show that a finite
spectrum always has a terminal viscosity and that it equals `Σ_k g_k tau_k`, while
`power_law_not_integrableOn` shows that a power law with `alpha ≤ 1` has none. "The condensate
has a viscosity" is therefore a different claim about the material, not an approximation to the
measured one.

For motion inside, `msd_of_uncorrelated` proves that uncorrelated stationary steps give exactly
`MSD(m) = m·σ²`, with no freedom at all. So a sublinear measurement forces the off-diagonal
covariances to sum to a negative number (`subdiffusion_forces_anticorrelation`) and some pair of
steps to be correlated (`subdiffusion_forces_memory`), and for a fitted `MSD(m) = A·m^alpha` with
`alpha < 1` this bites at every `m ≥ 2` (`power_law_msd_forces_memory`). A sampler that draws
independent displacements is excluded by the data, not merely inaccurate on them.

Finally, ageing. `aging_forces_error` prices non-stationarity in the same two-point form used
elsewhere in this development: if the modulus at lag `t` differs between two waiting times, a
single waiting-time-independent predicted curve is wrong by at least half that difference on one
of them. `IDR.material_state_laws` (`PartFortyThree.lean`) bundles the six statements.

## 153. Fitting: how much agreement with the data is evidence (Part XLIV)

An ensemble model of a disordered region is usually built by reweighting a pool of candidate
conformations until the predicted averages of `n` measured observables reproduce the data, and
the resulting agreement is reported as validation. `Fitting.lean` asks how much of that
agreement carries information about the ensemble.

The first answer is Carathéodory's theorem, read as a statement about ensembles.
`fit_with_few_structures` proves that if the data can be fitted at all, they can be fitted
*exactly* by an ensemble supported on at most `n + 1` conformations, no matter how large the
pool is: the observable vector of the fitted ensemble lies in the convex hull of the pool's
observable vectors, and Carathéodory produces an affinely independent, hence at most
`(n+1)`-element, sub-selection reproducing it. Exact agreement with `n` measurements is
therefore arithmetic that `n + 1` structures can always supply; it constrains the ensemble only
through feasibility.

The second answer says what a larger pool buys: not resolution, but freedom. `fit_not_unique`
shows that if the pool has more than `n + 1` members and some fit gives them all positive
weight, then the fit map — the linear map sending a weight vector to its total together with the
`n` predicted averages — has nontrivial kernel by rank–nullity, so there is a nonzero weight
direction that sums to zero and is invisible in every measured observable, and an entire
interval `w + t·u` of exact fits along it. The data choose none of them; the regulariser does.
`fit_blind_to_unmeasured` shows the practical consequence on three conformations and one datum:
two exact fits assign an unmeasured observable the two extreme values it takes on the pool.

The honest positive counterpart is `fit_unique_of_ker_trivial`: the fit is unique exactly when
that kernel is trivial. That is a property of the *experimental design*, checkable before the
fitting begins, and one that a pool larger than `n + 1` conformations never has.
`IDR.ensemble_fitting_laws` (`PartFortyFour.lean`) bundles the four statements.

## 154. Benchmarks: what an incomplete annotation can certify (Part XLV)

Everything quantitative said about disorder *prediction* is said about performance on an
annotated benchmark, and the annotation has a structure the scoring ignores: a residue is
annotated disordered when an experiment has shown it to be, and left unannotated otherwise —
whether it was shown to be ordered or was never examined. The labels are sound but incomplete,
which is exactly positive-unlabelled data. `Benchmark.lean` derives the consequences.

Two are reassuring in direction. `precision_label_le_precision_truth` shows that the reported
precision is a *lower* bound on the true precision: the incomplete annotation can only make a
predictor look worse on that measure. `falsePos_label_eq` decomposes the reported false
positives exactly, as the true false positives plus the predicted disordered residues the
annotation has not yet recorded — a predictor is penalised, residue by residue, precisely for
being ahead of the database. And `accuracy_close` gives the quantitative certificate: if the
labels differ from the truth at `d` of `N` residues, the true accuracy of any predictor is
within `d/N` of the number the benchmark reports. With a good annotation the reported accuracy
means something, and how much is a computation.

The ranking, which is what actually drives model selection, has no such protection.
`benchmark_can_invert_ranking` exhibits a four-residue chain with sound labels, one disordered
residue unannotated, and two predictors: the one that reproduces the truth exactly scores `3/4`
on the benchmark, and the one that reproduces the annotation — and is wrong about the truth —
scores a perfect `1`. The benchmark order is the reverse of the true order. Selecting models by
benchmark rank therefore actively selects against predictors that generalise beyond the
annotation, and the effect does not disappear at high annotation quality; only its frequency
does. `IDR.benchmark_laws` (`PartFortyFive.lean`) bundles the four statements.

## 155. Compensation: why fitted enthalpies and entropies are correlated (Part XLVI)

When a disordered region folds upon binding, the thermodynamics is reported as a pair `(ΔH, ΔS)`
extracted from binding constants measured at several temperatures. Collect such pairs across
variants, ligands or buffer conditions and they line up: large enthalpic gains come with large
entropic penalties, with a correlation so tight that the residual scatter is often smaller than
the error bars on either coordinate. This is the enthalpy–entropy compensation that a large
literature reads as a mechanism — solvent restructuring, conformational entropy released as
contacts form, and so on. `Compensation.lean` asks what the plot would look like if there were
no mechanism at all.

The fit is ordinary least squares of `y i = log K(T i)` on `x i = 1/T i`, with `ΔH = −R·slope`
and `ΔS = R·intercept`. Everything follows from `intercept_eq_mean_sub`, which is just the
normal equation: `intercept = ȳ − slope · x̄`. The entropy is not measured; it is the response
extrapolated from the experimental window out to infinite temperature, along the fitted line.
Two quantities determined by one lever arm cannot vary independently.

`compensation_of_equal_mean` makes that exact. If two data sets have the same mean log-constant
over the window — which is what it means for two variants to bind comparably well in the range
where they were compared — then their fitted intercepts differ by exactly `−x̄` times the
difference of their slopes. The `(ΔH, ΔS)` points lie on a straight line, with no error term,
whatever the molecules are and whether or not anything is compensating.
`compensation_temperature_eq_harmonic_mean` names the slope of that line: it is
`n / Σ_i T_i⁻¹`, the harmonic mean of the temperatures at which the experiment was performed.
The famous observation that fitted compensation temperatures cluster near the experimental range
is therefore not an observation about proteins.

What survives is small and precise. `deviation_from_compensation` drops the equal-mean
hypothesis and computes the residual exactly: the departure from the compensation line is the
difference in mean log-constant over the window, i.e. the difference in mean affinity across the
measured temperatures. That difference — and nothing else in the plot — is thermodynamic
information about the molecules. `compensation_amplifies_error` reads the same identity in the
direction an experimentalist cares about: `|Δintercept| = |x̄| · |Δslope|`, so any error in the
fitted enthalpy reappears in the fitted entropy scaled by the mean inverse temperature, and a
reported `ΔS` is never better determined than the reported `ΔH` divided by the harmonic mean
temperature. Quoting them as independent measurements overstates the evidence twice over.

`compensation_example` supplies a two-point instance in which both differences are nonzero, so
the identity is not being satisfied vacuously. `IDR.compensation_laws` (`PartFortySix.lean`)
bundles the five statements. The practical consequence for a model of disorder is a reporting
rule: a compensation plot is not evidence for a mechanism unless the residual about the
design-determined line, not the correlation itself, is shown to be significant.

## 156. Which moment does the experiment weigh? (Part XLVII)

Two experiments dominate the structural characterisation of a disordered region, and they
routinely disagree about how big it is. Small-angle scattering reports a radius of gyration — a
*second* moment of the distance distribution. Single-molecule FRET reports a mean transfer
efficiency, and the efficiency of a conformation at donor–acceptor distance `r` is
`R₀⁶/(R₀⁶ + r⁶)`, so the ensemble average is dominated by the *sixth* moment, and within it by
the short distances. The disagreement has been the subject of a long methodological argument.
`SaxsFret.lean` shows that a large part of it is not experimental at all: it is a property of
which moment each observable weighs, and it is exactly computable.

The forward model is set up first (`eff_pos`, `eff_le_one`, `eff_antitone`: efficiency lies in
`(0,1]` and decreases with distance), together with the standard inversion — solve
`eff R₀ d = E` for `d` — in a root-free form, `apparentSixth R₀ E = R₀⁶(1−E)/E`, with
`apparentDistance_pow_six` confirming it is the sixth power of the distance actually reported.

`meanEff_ge` is the inequality everything rests on, and it is proved from Cauchy–Schwarz in
Engel form rather than by any asymptotic argument: the measured efficiency of a heterogeneous
ensemble is at least the efficiency of the single conformation whose sixth power of distance
equals the ensemble mean. `apparentSixth_le_meanSixth` turns this into the statement that
matters at the bench: **the FRET-inferred distance never exceeds the sixth-moment mean distance**.
Averaging an `r⁻⁶`-weighted observable and then inverting always reports a chain as more compact
than it is. `apparentSixth_eq_of_homogeneous` shows that the bias is not an offset of the method
— it vanishes identically for a homogeneous ensemble — and `apparentSixth_mem_Icc` bounds it by
the spread: the inferred distance always lies between the smallest and largest distances present.
So the size of the discrepancy is a measure of the width of the ensemble, which is precisely the
quantity a model of disorder is supposed to predict.

How far apart can the two experiments be? `meanSq_cube_le_meanSixth` gives the only general
relation, the power-mean ordering `⟨r²⟩³ ≤ ⟨r⁶⟩`. It is only an inequality, and the two explicit
examples show that no better relation exists. `saxs_blind_to_fret` gives two two-state ensembles
with *identical* mean square distance whose transfer efficiencies are `1/2` and `5/9`:
scattering cannot predict the FRET measurement. `fret_blind_to_saxs` gives two with *identical*
transfer efficiency `1/2` whose mean square distances differ, the second strictly larger: FRET
cannot predict the scattering measurement.

The design consequence is sharp, and it is the same one this development reaches from several
directions. A model of a disordered region must be compared with each experiment through that
experiment's own forward model — predicted distances pushed through `R₀⁶/(R₀⁶+r⁶)` for FRET and
through the second moment for scattering — and never through a "distance" or a "size" extracted
by inverting one of them. Reconciling a FRET distance with a scattering radius is not a
consistency check on the molecule; the two numbers are different functionals of the ensemble, and
`IDR.saxs_fret_laws` (`PartFortySeven.lean`) says by exactly how much they may differ.

## 157. The landscape does not determine the rate (Part XLVIII)

Almost every narrative account of a disordered region is a story about a free-energy landscape:
a coordinate — an end-to-end distance, a helicity, a transfer efficiency — and the equilibrium
weights `p i` along it, with wells, a barrier, and a rate read off the barrier height. The
weights are what a simulation or a reweighted experiment reports. The question this part settles
is how much of the kinetic story they actually carry.

The setting is the smallest honest one: states `0, …, n` along the coordinate, hops `i → i+1` at
rate `kp i` and `i → i−1` at rate `km i`, a reflecting lower end (`km 0 = 0`), and detailed
balance `p i · kp i = p (i+1) · km (i+1)` — which is exactly the statement that the hopping
dynamics has the reported profile as its equilibrium. The mean first-passage time to the far end
is defined, as usual, by first-step analysis (`IsMFPT`), and `FirstPassage.lean` solves that
system completely.

`mfptFun_isMFPT` shows the closed form
`T m = Σ_{m ≤ i < n} (Σ_{j ≤ i} p j) / (p i · kp i)`
is a solution, so the system is satisfiable and nothing below is vacuous; `flux_eq` and
`mfpt_unique` show it is the *only* solution. The proof is a flux argument: detailed balance
makes `p i · kp i · (T i − T (i+1))` increase by exactly `p i` at each step, so it telescopes to
the cumulative weight below `i`. This is the discrete Kramers formula, obtained here without any
continuum limit.

Now read the formula. It depends on the profile `p` **and** on the rates `kp`. Rescaling every
rate by a constant `c > 0` preserves detailed balance with the *same* `p`
(`detailedBalance_smul`), hence preserves every equilibrium quantity — populations, free
energies, all thermodynamic observables — and divides every first-passage time by `c`
(`mfptFormula_smul`). `landscape_does_not_determine_rate` states the consequence: no function of
the landscape alone returns the rate, because two systems with identical landscapes have times
differing by an arbitrary factor. And the freedom is not merely an overall prefactor:
`kinetics_not_determined_by_profile` exhibits two rate profiles on one and the same *flat*
landscape, differing only in the speed of the second step, whose crossing times are `3` and `5`.
A position-dependent diffusion coefficient is an independent input to a kinetic model.

What the landscape does supply is a bracket. `mfptFormula_ge_barrier` gives, for every
intermediate state `b`, the bound `T ≥ p 0 / (p b · kp b)`, and `mfpt_ge_exp_barrier` writes it
for Boltzmann weights as `T ≥ exp(β (F b − F 0)) / kp b`: taking `b` at the top of the barrier,
the crossing time is at least the Arrhenius factor divided by the local rate. `mfptFormula_le`
gives the matching upper bound `Σ_i 1/(p i · kp i)` for a normalised profile. So the barrier
height controls the *exponential* part of the answer and the kinetic profile controls the rest —
which is precisely why a model that reports only populations can predict the temperature
dependence of a rate and not the rate.

`IDR.first_passage_laws` (`PartFortyEight.lean`) bundles the six statements.

## 158. The committor: nor the mechanism (Part XLIX)

If rates are not in the landscape, what about mechanism? The object that defines mechanism in a
hopping model is the **committor**: `q i` is the probability that a chain started at `i` reaches
the product end before returning to the reactant end, and "the transition state" *is*, by
definition, the place where `q = 1/2`. `Committor.lean` treats it with the same machinery.

`committorFun_isCommittor` and `committor_unique` establish the closed form
`q i = (Σ_{k<i} 1/(p k · kp k)) / (Σ_{k<n} 1/(p k · kp k))`
as the unique solution of the harmonic system with `q 0 = 0`, `q n = 1`. The proof is again a
flux argument — `flux_const`: the reactive flux `p i · kp i · (q (i+1) − q i)` is the same across
every bond, detailed balance playing the part of Kirchhoff's law — and the formula is the
electrical one, a ratio of resistances `1/(p k · kp k)`. `committor_strictMono` and
`committor_mem_Icc` confirm the committor rises strictly from `0` to `1`, so there is exactly one
transition region.

The resistances, not the weights, are what the committor sees. Two corollaries follow, and they
are the point of the part. If the forward rate is the same at every step — a constant diffusion
coefficient — then `committor_uniform_rate` gives
`q i = (Σ_{k<i} 1/p k) / (Σ_{k<n} 1/p k)`, a functional of the profile alone, dominated by the
states of *least* equilibrium weight. That is the precise sense in which the familiar statement
"the transition state is the top of the barrier" is a theorem. Drop the assumption and it fails:
`mechanism_not_determined_by_landscape` puts the transition state at `q 1 = 1/2` for the uniform
rate profile and at `q 1 = 1/3` for the profile whose second step is twice as slow — on the same
flat landscape, with the same populations and the same free energy at every state.

Taken with Part XLVIII the conclusion is uniform: neither the rate nor the mechanism of a
conformational transition is a functional of the free-energy landscape. A model of a disordered
region that reports populations, however accurately and along however good a coordinate, has not
yet reported kinetics; it must carry a kinetic profile — position-dependent rates, equivalently a
diffusion profile — as data of its own before any of the usual mechanistic language is licensed.

`IDR.committor_laws` (`PartFortyNine.lean`) bundles the five statements.

## 159. A barrier is neither necessary nor sufficient for slow kinetics (Part L)

Parts XLVIII and XLIX say that the landscape fixes neither rate nor mechanism. `BarrierRate.lean`
turns that into the two statements one actually reasons with at the bench, and isolates the extra
hypothesis under which the familiar Arrhenius reading is correct.

`flatLandscape_arbitrarily_slow`: for every `M` there is a detailed-balanced hopping model whose
equilibrium profile is exactly flat — all three states equally populated, no barrier anywhere —
and whose mean first-passage time exceeds `M`. The construction is transparent: make one step
slow. So an observed slow interconversion in a disordered region is not, by itself, evidence of a
free-energy barrier; it is equally consistent with a barrierless landscape and a small local
diffusion coefficient (internal friction, in the usual language).

`barrier_arbitrarily_fast`: conversely, for every barrier height `B` and every target time
`eps > 0` there is a detailed-balanced model whose profile has a barrier of exactly height `B`,
`p 0 / p 1 = exp B`, and whose mean first-passage time is exactly `eps`. A barrier drawn in a
reported free-energy profile is not, by itself, evidence of slowness.

What survives is `barrier_brackets_rate`: if the kinetic prefactor is bounded,
`kmin ≤ kp i ≤ kmax`, then the crossing time lies between `p 0 / (p b · kmax)` — the Arrhenius
factor of any intermediate state `b`, i.e. `exp(β(F b − F 0))/kmax` for a Boltzmann profile — and
`(Σ_{i<n} 1/p i)/kmin`. Arrhenius reasoning is a theorem about a landscape *plus* a bounded
diffusion profile, and about nothing less.

For a model of a disordered region this closes the loop opened in Part XLVIII: a reported
free-energy landscape is not a kinetic prediction, and a measured relaxation time is not a
measurement of a barrier. `IDR.barrier_rate_laws` (`PartFifty.lean`) bundles the three
statements.

## 160. Electronic polarisability, priced exactly (Part LI)

`RequestProject/Polarisability.lean` removes the first idealisation Part XXV named and did not
remove: the force field of Part XXIV gives every atom a fixed charge. In the standard
point-polarisable model the induced dipoles are not given by a formula but by the
self-consistent system `mu i = a i · (E i + Σ_j T i j · mu j)` (`SelfConsistent`). Under the
standard damping condition `|a i| · Σ_j |T i j| ≤ c < 1` (`Damped`, the exclusion of the
polarisation catastrophe) `selfConsistent_unique` and `selfConsistent_exists` prove there is
exactly one solution for every external field — injectivity of `I − aT`, proved by a maximum
argument, gives surjectivity in finite dimension. Adding polarisability is therefore a well-posed
repair.

The symmetric `m`-site cluster in a uniform field is then solved in closed form
(`clusterDipole_selfConsistent`, `energy_uniform_eq`): `mu = a/(1 − (m−1)at)`, `U = −(m/2)·mu`.
The inclusion–exclusion residue of the three-site cluster, i.e. what is left after the best
possible one- and two-body terms, is computed exactly:

`threeBody_eq`:  `U₃ − 3U₂ + 3U₁ = −3a³t²/((1 − 2at)(1 − at))`.

It is strictly negative in the damped regime whenever the coupling is nonzero
(`threeBody_neg`) — polarisation is cooperative — and it is second order in the coupling, which
is why a pairwise-fitted force field can be numerically decent and still structurally wrong.
`polarisable_not_pairwise_additive` draws the consequence: no assignment whatsoever of one-body
and two-body energies reproduces the cluster energies of a polarisable model. A fixed-charge
model is not a polarisable model with bad parameters; it is a different model, and the difference
is the number above. `IDR.polarisability_laws` (`PartFiftyOne.lean`) bundles the four statements.

## 161. Nuclear quantum effects, and the isotope effect a classical model cannot have (Part LII)

`RequestProject/NuclearQuantum.lean` treats one harmonic mode of energy quantum `w = ħω` at
inverse temperature `b`. `qFree_eq_zpe_add` gives the exact decomposition
`F_q = w/2 + (1/b) log(1 − e^{−bw})`; `clFree_lt_qFree` and `qFree_lt_zpe` give the sandwich
`F_cl < F_q < w/2`, the first by `sinh x > x` — there is no regime in which a classical harmonic
mode has the right free energy. `qEnergy_gt_kT` and `qEnergy_gt_zpe` are the same failure for the
mean energy: the mode holds more than the classical `kT` at every temperature (the elementary
inequality behind it, `sinh x < x cosh x`, is proved from the derivative). The model is not wrong
everywhere: `tendsto_partition_ratio_one` shows the quantum and classical partition functions
agree in the limit of a soft mode, so the collective motions of a disordered chain are safely
classical and the stretches are not.

The sharp statement is the pair `classical_isotope_independent` / `quantum_isotope_effect`. A mass
substitution scales every frequency by one factor `s`; the factor cancels *exactly* from every
classical free-energy difference, so a classical force field predicts identically zero equilibrium
isotope effect — not a small one. The quantum model does not: the low-temperature limits of the
two free-energy differences are the zero-point differences `(wA − wB)/2` and `s(wA − wB)/2`, so
there is a temperature at which they differ. A measured H/D equilibrium effect on a disordered
region is therefore a measurement of something a classical model assigns the value zero; the
repair — a path-integral treatment of the nuclei — is standard, and what these theorems price is
the decision not to make it. `IDR.nuclear_quantum_laws` (`PartFiftyTwo.lean`) bundles five
statements.

## 162. The orientation factor: a FRET efficiency is not a distance (Part LIII)

`RequestProject/Orientation.lean` removes the `κ² = 2/3` substitution that Part XLVII left in
place. `kappaSq_le_four` proves the exact range `0 ≤ κ² ≤ 4` for unit dipoles and a unit
separation direction, from the Gram determinant of the three vectors (`gram_identity`: the
determinant equals the squared triple product, so its nonnegativity is a polynomial identity, not
an assumption). `meanKappaSq_eq` exhibits a finite orientational model — donor and acceptor each
uniform over the six axis directions — whose mean orientation factor is exactly the textbook
`2/3`.

And that is all `2/3` is. `meanEff_ne_effMean`: for the same model, at the distance where the
`κ² = 2/3` formula returns `E = 2/5`, the true mean efficiency is `1/5`, because the efficiency is
not linear in `κ²`. Read as a distance (`apparentSixth_of_meanEff`) the measurement returns
`r⁶ = 8/3` where the truth is `1`. And the residual uncertainty does not shrink with photon
statistics: `apparentSixth_ratio_six` shows that at fixed efficiency the distance inferred with
`κ² = 4` and with `κ² = 2/3` differ by a factor `6` in `r⁶`. For a disordered region — short
linkers, sticky dyes, incomplete rotational averaging — an ensemble is not comparable with a FRET
efficiency unless the model carries the dye orientational distribution, or the analysis reports
the bracket rather than a distance. `IDR.orientation_laws` (`PartFiftyThree.lean`) bundles five
statements.

## 163. Transition-path times: the crossing is not the waiting (Part LIV)

`RequestProject/TransitionPath.lean` brings inside the development the object a single-molecule
experiment actually resolves, and does so with no new dynamical assumption. The committor
`h`-transform is constructed explicitly: `reactP p q i = p i · q i²`,
`reactKp = kp i · q(i+1)/q(i)`, `reactKm = km i · q(i−1)/q(i)`. `reactive_detailedBalance` proves
the conditioned dynamics is *again* a detailed-balanced hopping chain, and `reactKm_zero` that its
lower end is automatically reflecting — a reactive trajectory cannot return. Everything proved in
Part XLVIII therefore applies verbatim: `tpt_eq` gives existence and uniqueness of the mean
transition-path time and its closed form (the discrete Kramers formula for the reweighted profile
`p q²`), and `tptFormula_pos` its positivity.

`slow_reaction_fast_paths` is the law the two times violate together: for every `M` there is a
detailed-balanced three-state model whose mean first-passage time is at least `M` and whose mean
transition-path time is at most `1/M`. The witness is solved in closed form — on the profile
`(1, e, 1)` with unit forward rates the first-passage time is `1 + (1+e)/e` and the
transition-path time is `e/(1+e)` — so raising the barrier makes the reaction slower and the
crossings faster. No function of the one returns the other, and a measured transition-path time is
not a measurement of a rate. `IDR.transition_path_laws` (`PartFiftyFour.lean`) bundles four
statements.

## 164. Heat capacity: the curved van 't Hoff plot (Part LV)

`RequestProject/HeatCapacity.lean` covers the case Part XLVI excluded. With a constant `ΔCp`,
`ΔH(T) = ΔH₀ + ΔCp(T − T₀)`, `ΔS(T) = ΔS₀ + ΔCp log(T/T₀)`, and everything is an identity.
`vantHoff_eq_enthalpy_at`: the two-point van 't Hoff enthalpy of a window `[T₁, T₂]` is the *true*
enthalpy at one interior temperature,

`ΔH_vH = ΔH(M)`,  `M = T₁T₂ log(T₁/T₂)/(T₁ − T₂)`,

and `logMeanRecip_mem_Ioo` places `M` strictly between the endpoints (by `log x < x − 1` applied
in both directions). Hence `vantHoff_ne_endpoints`: once `ΔCp ≠ 0` the fitted number is the
enthalpy at neither temperature of the window. `secondDiff_eq` computes the curvature exactly —
sampling `log K` at three temperatures equally spaced in `1/T` gives
`(ΔCp/R) log((T₁+T₂)²/(4T₁T₂))` — which vanishes identically when `ΔCp = 0` and is nonzero with
the sign of `ΔCp` otherwise, by strict arithmetic–geometric mean. `ΔCp` is thus identifiable from
`log K` alone and invisible to the linear fit whose algebra Part XLVI analysed. For a disordered
region, where burial of apolar surface on binding makes `|ΔCp|` large, this is the ordinary case.
`IDR.heat_capacity_laws` (`PartFiftyFive.lean`) bundles four statements.

## 165. Aggregation after the early-time regime (Part LVI)

`RequestProject/Depletion.lean` treats the saturated kinetics Part XXXVII excluded:
`M' = κM(1 − M/m₀)`, solved by `M(t) = m₀/(1 + e^{−κ(t − t½)})`. `fibrilMass_hasDerivAt` verifies
the solution, and `fibrilMass_pos`, `fibrilMass_lt_total`, `fibrilMass_strictMono`,
`tendsto_fibrilMass_total` give positivity, the bound by the total protein, monotonicity and the
plateau. `fibrilMass_le_exp` shows depletion only ever slows growth — the early-time treatment is
an upper bound of known sign, not an approximation of unknown sign.

`crossing_eq` is the threshold identity: the fraction `f` is reached at
`t½ + log(f/(1−f))/κ`; `crossing_threshold_shift` shows two different detection thresholds give
genuinely different crossing times, so a quoted lag is a statement about the instrument as much as
about the sample. `tangent_slope` and `tangentLag_eq` give the standard construction — maximal
slope `κm₀/4` at `t½`, tangent meeting the baseline at `t½ − 2/κ` — and
`lag_does_not_determine_rate` shows that lag time fixes neither `κ` nor `t½`: for any lag and any
two distinct growth rates there are half-times realising it, with curves that differ. Together
with Part XXXVII: in the early-time regime the lag is a logarithm of the nucleation rate, in the
saturated regime a two-parameter shadow of the whole curve, and in neither is it a rate.
`IDR.depletion_laws` (`PartFiftySix.lean`) bundles seven statements.

## 166. Anisotropic displacement and discrete conformers in a map (Part LVII)

`RequestProject/Anisotropy.lean` removes the two idealisations Part XXXVIII named: one occupancy
and one *isotropic* displacement parameter per atom. For a disordered region those are the whole
phenomenon — the atom is in several places, and the smear is in general a tensor.

The two-conformer density `twoSite s d` (sites at `±d`, each Gaussian of width `s`) factorises
exactly, `twoSite s d x = gauss s x · e^{-d²/2s²} · cosh(xd/s²)`, and from that factorisation
`twoSite_ne_gauss` proves the sharp statement: for **every** single-site width `u > 0` the two
densities differ somewhere. The refinement program's one-parameter family does not contain the
two-site density at all; inflating `B` is not an approximation to splitting a site, it is a
different model. The error even has a known sign where one looks: under the standard
second-moment prescription `u² = s² + d²`, `twoSite_center_lt_matched` shows the fit is strictly
*too high* at the midpoint (the proof is `1 + t < e^t` in disguise). And once the conformers are
resolved, `2s² ≤ d²`, `twoSite_bimodal` gives a dip at the midpoint — a shape, not a width, and
one no Gaussian has anywhere.

For the tensor, `anisoDensity_inj` is the positive result: an anisotropic density determines its
principal widths, so nothing is lost by refining the tensor. `equivB_not_determining` and
`aniso_ratio_unbounded` are the price of not doing so: two tensors with the same isotropic
equivalent have different densities, and at fixed isotropic `B` the ratio of principal widths is
unbounded — a quoted `B` constrains the mean square displacement and says nothing about its
shape. `occupancy_anisotropy_degenerate` carries the Part XXXVIII occupancy trade-off into tensor
form. `IDR.anisotropy_laws` (`PartFiftySeven.lean`) bundles five statements.

## 167. Residual dipolar couplings: the orientational observable (Part LVIII)

`RequestProject/Rdc.lean` adds the experiment the development did not have: an orientational one.
Every other observable here is a distance, a rate, a population or a coupling constant; the
residual dipolar coupling of a weakly aligned sample is the measurement that constrains the
*direction* of a bond, and it is used on disordered regions precisely because it survives
conformational averaging. The model is the standard one, `rdc A u = (3uᵀAu − tr A)/2` with `A`
symmetric and traceless.

What one measurement cannot do: `rdc_neg` — a bond vector and its reverse are indistinguishable,
in every medium; `rdc_level_set_circle` — for an axially symmetric tensor the coupling is
*constant* on a whole circle of unit directions, so the ambiguity left by one medium is
continuous, not discrete; `rdc_axes_mean_zero` — averaged over three orthogonal directions the
coupling vanishes for every alignment tensor, so a small RDC is what symmetry gives and not
evidence of a particular geometry; `order_population_degenerate` — an explicit fully ordered
ensemble at an intermediate angle and an explicit half-ordered ensemble at the pole give exactly
the same coupling, so population and order parameter enter only through their product.

What any number of measurements can do: `meanRdc_eq_quad_secondMoment` shows the ensemble average
is a linear functional of the second-moment matrix `⟨u_i u_j⟩`, hence
`meanRdc_eq_of_secondMoment_eq` — two ensembles with the same second moment are indistinguishable
by RDCs *in every medium simultaneously* — and `secondMoment_not_injective` exhibits two
two-conformer ensembles with **no conformer in common** that share it. What *is* determined is
five numbers: `alignment_decomposition` gives the explicit five-tensor decomposition of a
symmetric traceless matrix and `rdc_five_directions_determine` recovers it from five bond
directions, which is the linear algebra behind "five independent alignment media", proved rather
than asserted. `IDR.rdc_laws` (`PartFiftyEight.lean`) bundles six statements.

## 168. Benchmarks whose annotations are wrong in both directions (Part LIX)

`RequestProject/LabelNoise.lean` drops the assumption Part XLV made explicit: that every annotated
residue really is disordered. A disorder benchmark cannot make it. A residue is called disordered
because it is missing from a map, and Part XXXVIII enumerated what else produces a missing
residue; a residue is called ordered because a structure was solved under conditions that need
not be the cellular ones. Errors run both ways, and nothing here assumes otherwise.

`errors_le_add_noise` and its converse bracket the measured error count by the true one to within
`noise T L`, the number of mislabelled residues — that is the entire positive content of an
unsound benchmark. `perfect_predictor_penalised` is the sharpest form of the problem: a model that
reproduces the truth residue for residue is recorded as making exactly `noise` mistakes, and with
false positives present those mistakes cannot be argued away as missing coverage.
`ranking_certified` is the usable statement: a margin of more than twice the noise certifies the
ranking, so quoting a margin next to an annotation error rate is either a certificate or it is
nothing. `ranking_inversion_false_positive` and `ranking_inversion_single_label` show the failure
is real — a single spurious label suffices to invert a ranking — and `ranking_certificate_sharp`
shows the factor two cannot be improved: at a margin of exactly twice the noise the benchmark
records a tie. `IDR.label_noise_laws` (`PartFiftyNine.lean`) bundles five statements.

## 169. Explicit solvent: the potential of mean force is not a force field (Part LX)

`RequestProject/Pmf.lean` supplies the item Part XXV named and left outside: the structure of the
solvent. The construction is exact — a finite solvent state space, an arbitrary solute–solvent
interaction, and `pmf b Uint x = -(1/b) log ∑_s e^{-b·Uint x s}`. `marginal_eq` proves that the
solute marginal of the joint Boltzmann measure is *identically* the Boltzmann measure of
`Usol + pmf`, for every interaction and every temperature. Integrating out the water is not an
approximation. The question is only what the resulting object is.

It is not a force field. In the smallest model that can show it — one water molecule with a bound
and a free state, binding stabilised additively by each of three solutes in contact, so the
*interaction* is exactly pairwise — the free energy is not, because a logarithm of a sum is not a
sum. `solventThreeBody_eq` computes the inclusion–exclusion residue in closed form at the coupling
`b·eps = log 2`:

`W₃ − 3W₂ + 3W₁ − W₀ = (1/b)·log(250/243)`,

and `solventThreeBody_pos` gives its sign: strictly positive, so the solvent-mediated three-body
force is *anti*-cooperative — the third solute gains less than the second. It is not even a
potential: `pmf_temperature_dependent` shows the solvation contribution of a single solute takes
different values at two temperatures, so it is a free energy carrying an entropy and cannot be
tabulated once and transferred. For a region that is, by construction, mostly surface, this is
where the collapse transition, the temperature dependence of the radius of gyration, and the
cooperativity of hydrophobic contacts live. `IDR.pmf_laws` (`PartSixty.lean`) bundles four
statements.

## 170. What a single-molecule FRET number is worth: photophysics and linkers (Part LXI)

`RequestProject/Photophysics.lean` removes the last two idealisations in the treatment of
single-molecule FRET. The first is photophysical. What a detector delivers is photon counts, and
the quantity formed from them is the proximity ratio `nA/(nA + nD)`; converting it into a transfer
efficiency needs the factor `g = (η_A φ_A)/(η_D φ_D)`, the ratio of detection efficiencies times
quantum yields. `proximityRatio_eq_self_iff` shows the raw ratio equals the efficiency exactly
when `g = 1` and at no interior efficiency otherwise. What survives an unknown `g` is stated
positively by `proximityRatio_strictMono`: the observed ratio is strictly increasing in the true
efficiency, so the ordering of conformers and the direction of any change — the two things a
titration or a mutant series actually uses — are correct whatever `g` is. What does not survive is
the magnitude. Since `E/(1−E) = (R₀/r)⁶`, `sixthPower_off_by_gamma` gives
`(1−P)/P = (1/g)·(1−E)/E`: the inferred sixth power of the distance is the true one divided by
`g`, exactly, and `gamma_unidentifiable` shows `g` cannot be recovered from the observed ratio,
because every ratio in `(0,1)` arises from every `g > 0` at a suitable efficiency. Background adds
a bias with a direction: `apparentEff_pos_of_background` shows that with any acceptor background,
a state with zero transfer is measured at strictly positive efficiency — and in a disordered
region the zero-transfer states are precisely the expanded ones, so the reported distance
distribution is pulled towards compaction.

The second idealisation is geometric. The dye is not the residue; it sits at the end of a linker.
`linker_bound` proves that if each dye lies within `L` of its attachment point then
`|dist b₁ b₂ − dist a₁ a₂| ≤ 2L`, in any metric space, from the triangle inequality alone. That is
the entire positive content: a dye–dye distance brackets the residue–residue distance to `±2L`.
And `linker_bound_sharp` shows the bracket is attained at both ends — two configurations with the
same attachment distance whose dye distances differ by `4L` — so no function of the attachment
distance corrects it. For a typical linker and a typical `R₀` the bracket is comparable with the
difference between a compact and an expanded state of a short region. Together with the factor `6`
in `r⁶` from `κ²` (Part LIII), the three corrections a single-molecule distance carries are
multiplicative, independent, and unaffected by collecting more photons. `IDR.photophysics_laws`
(`PartSixtyOne.lean`) bundles six statements.

## 171. The Markov state model that is actually estimated (Part LXII)

Part IV.6 settles the qualitative question about clustering microstates into macrostates: under
Dynkin's condition the coarse variable is Markov, and generically it is not. In practice nobody
verifies the condition. What is done instead is to run a long equilibrium trajectory, count the
transitions between clusters at some lag time, and normalise — which in the infinite-sampling
limit produces the stationary-weighted lumping

`macroW a b = (Σ_{i ∈ a} π i · Σ_{j ∈ b} P i j) / (Σ_{i ∈ a} π i)`,

the conditional probability of being in cluster `b` one lag later given equilibrium *within*
cluster `a`. `RequestProject/MarkovStateModel.lean` asks what that object is a statement about
when the clustering is not lumpable, and the answer separates into three cleanly different
regimes.

It is a consistent estimator. `macroW_eq_of_lumpable`: when the clustering does happen to satisfy
Dynkin's condition, the estimated matrix is the true reduced matrix.

Its thermodynamics is unconditionally right. With no assumption on the clustering at all,
`macroW_nonneg` and `macroW_stochastic` make the estimate a stochastic matrix,
`macroW_stationary` makes the coarse-grained equilibrium populations stationary for it, and
`macroW_detailedBalance` inherits detailed balance from the microscopic chain. This is a warning
disguised as a positive result: a Markov state model reproduces the equilibrium populations of its
own clustering by construction, so the usual observation that the model "gets the populations
right" is no evidence whatever that the clustering is a good one.

Its kinetics is one-sided, and provably so. Write `form`, `nrm` and `mean` for the equilibrium
quadratic form `⟨f, Pf⟩`, mean square and mean of an observable — the ingredients of the Rayleigh
quotient whose maximum over mean-zero observables is the second eigenvalue of a reversible chain.
Then `mean_lump_eq`, `nrm_lump_eq` and `form_lump_eq` say that for a macro observable `g` all
three coarse quantities are *exactly* the microscopic quantities of `g ∘ phi`. The coarse model's
Rayleigh quotients are therefore a subset of the fine model's, and `macro_gap_bound` transfers any
microscopic bound to the coarse model. Coarse-graining cannot invent a slow mode. Every relaxation
rate a Markov state model exhibits is one the underlying dynamics already had, so a reported
implied timescale is a lower bound on the truth and never an overestimate — clustering hides slow
motion, it does not manufacture it.

And the number reported depends on the lag. `chapman_kolmogorov_of_lumpable` shows that under
lumpability the matrix estimated at lag two is the square of the matrix estimated at lag one, so
the Chapman–Kolmogorov ("implied timescale") test is a genuine test of the modelling assumption
rather than an identity. `chapman_kolmogorov_fails` shows the test is generically failed, on an
example with nothing wrong with it: the three-state nearest-neighbour chain with hopping
probability `1/2`, which `msmP_stochastic` and `msmP_detailedBalance` certify is stochastic and
reversible with uniform equilibrium — no absorbing state, no vanishing population — clustered as
`{0,1} | {2}`. The lag-two estimate of the escape probability is `1/4`; the square of the lag-one
estimate is `5/16`. The two differ by a quarter of the smaller.

For a disordered region the criterion is the sharpest available statement of the difficulty:
lumpability requires the clustering to respect exit probabilities exactly, which a partition of a
continuum of interconverting conformations into "compact" and "extended" does not do. The design
consequence is a three-part rule for reporting: populations from a state-decomposition kinetic
model may be quoted without qualification, timescales must be quoted as lower bounds, and the
model is not defined at all until the lag time is quoted with it. `IDR.markov_state_model_laws`
(`PartSixtyTwo.lean`) bundles five statements.

## 172. Pulling out of equilibrium: work, dissipation and the free energy (Part LXIII)

Part XV.2 treats a pulling experiment at equilibrium, where the ensemble at each force is the
tilted Boltzmann one. A real optical-tweezer or AFM experiment on a disordered region is not
performed at equilibrium: the trap moves at a finite speed, the chain is dragged along, and what
is recorded is the work of each pull. `RequestProject/WorkTheorem.lean` takes the single physical
hypothesis that governs such an experiment — microscopic reversibility, Crooks' relation
`p γ = q(rev γ)·exp(b·(W γ − dF))`, relating the probability of a forward trajectory to that of
its time reverse under the reversed protocol — and derives everything the work distribution
determines. `crooks_satisfiable` first shows the hypothesis is not vacuous: for any reverse
protocol and any work function there is a forward protocol obeying it, with `dF` forced to be the
logarithm of the exponential average.

Three results are positive. `jarzynski`: the exponential average of the work is exactly
`exp(−b·dF)`, however fast and however dissipative the pull — a nonequilibrium experiment does
determine an equilibrium free energy, which is not obvious and is the reason the technique exists.
`dissipation_eq_relEntropy`: the dissipated work is *exactly* a relative entropy,
`b·(⟨W⟩ − dF) = Σ p·log(p / q∘rev)`. Irreversibility is not a vague notion here; it is the
statistical distinguishability of the forward pull from its own time reverse, in the same
information-theoretic currency as everything else in this development. `crooks_histogram` and
`crooks_crossing`: the forward and reverse work histograms satisfy
`P_F(w) = exp(b·(w − dF))·P_R(−w)`, so they cross exactly at `dF` — a free energy read off two
measured histograms with no model in between.

Two are negative, and they bound what a one-directional experiment can claim. `second_law` gives
`dF ≤ ⟨W⟩` — Gibbs' inequality applied to the relative entropy above — and
`no_dissipation_iff_deterministic` says the bound is attained exactly when the work is the same on
every trajectory, which by Crooks' relation is exactly when the forward process coincides with the
reversed one. Any spread in the measured work is dissipation. This matters more for a disordered
region than for a folded domain: there is no cooperative folded state to keep the work
distribution narrow, so the gap between `⟨W⟩` and `dF` is generically large, and a model fitted to
a mean work is fitted to the dissipation of the instrument as much as to the region.

The exponential average that would correct this is the second negative result.
`rare_trajectories_dominate` exhibits, for every `M > 0`, a two-branch work distribution in which
the rare branch has probability at most `exp(−M)` yet contributes more than the entire rest to
`⟨exp(−b·W)⟩`, and in which the true free energy is `−M` or below while the estimate obtained from
the typical branch alone is `0`. The error of a finite-sample Jarzynski estimate is therefore
unbounded. It is also signed: `omitting_low_work_overestimates` proves that discarding any set of
trajectories whose work is below all the retained ones can only increase the estimated free
energy, so an undersampled estimate drifts from `dF` towards `⟨W⟩` and never past it.

The design reading is a three-part report. A pulling experiment on a disordered region yields an
upper bound `⟨W⟩` whose gap is a relative entropy; an exponential average that is unbiased in
principle and sample-limited in a quantifiable, one-directional way in practice; and — only if the
reverse protocol is measured as well — an unbiased crossing point. `IDR.work_theorem_laws`
(`PartSixtyThree.lean`) bundles six statements.

## 173. Structure-based coarse-graining: the inverse problem (Part LXIV)

Part LX treats the forward direction of coarse-graining: integrating out the solvent is exact,
and what it leaves is not a pairwise, transferable potential. `RequestProject/InversePotential.lean`
treats the inverse direction, which is how coarse-grained models of disordered regions are
actually built — fit a potential so that the model reproduces a measured structural statistic (a
radial distribution function, a set of contact frequencies, a distance histogram), then use that
potential elsewhere. Iterative Boltzmann inversion, force matching and relative-entropy
coarse-graining are all instances of it.

The first result is positive, and it is the one that makes the practice legitimate.
`gibbs_unique_of_meanFeature_eq` is Henderson's uniqueness theorem in the discrete setting: two
parameter vectors whose Boltzmann distributions have the same mean features have the same
Boltzmann distribution. The proof is short and worth stating, because it is the same tool used
throughout this development: the symmetrised relative entropy of the two models is exactly `b`
times the inner product of the parameter difference with the difference of the feature means, so
matching the features makes `KL(p‖p') + KL(p'‖p)` vanish, and Gibbs' inequality then forces the
two distributions to coincide. A fitted coarse-grained potential is therefore not an arbitrary
selection among many potentials that fit the data equally well; the structure determines the
model.

The second result says what that model is worth. It determines the model — and the model is blind
to everything beyond the statistics it was fitted to, in the strongest possible sense. Take three
spins and the parity ensemble, uniform on the four configurations with `s₁s₂s₃ = +1`. All three of
its pair correlations are zero, exactly as for the uniform ensemble, while its triple correlation
is `1`. By the uniqueness theorem, every pair-potential model whose pair correlations match those
of the parity ensemble *is* the uniform ensemble, whose triple correlation is `0`
(`pair_potentials_blind_to_three_body`); and `zero_matches_pair_structure` shows such a model
exists, so the statement is not vacuous. Fitting the pair structure exactly therefore gets the
three-body structure maximally wrong, and no choice of pair potential repairs it. For a disordered
region this is not a corner case: collapse, hydrophobic cooperativity and condensation are exactly
the many-body part of the problem.

The third says the fitted potential is a free energy, not a force field.
`inverse_potential_temperature_dependent` works out the smallest example: the potential that
reproduces the feature value `1/3` at inverse temperature `1` is `log 2`, the potential that
reproduces the same value at inverse temperature `2` is `(log 2)/2`, and the first, transferred to
the second temperature, does not reproduce the target. A structure-based potential is a statement
about a state point, and disordered regions are studied precisely by changing state points — salt,
temperature, crowding, partner concentration.

Taken with Part LX, the two halves bracket the coarse-graining operation. The forward map leaves
an object that is exact but neither pairwise nor transferable; the inverse map returns an object
that is unique but equally non-transferable and blind by construction beyond its fitted
statistics. A coarse-grained model of a disordered region must therefore either carry the state
point as an explicit input, or be refitted at every state point it is used at — and must report
which correlations it was fitted to, because those are exactly the correlations it can be trusted
for. `IDR.inverse_potential_laws` (`PartSixtyFour.lean`) bundles four statements.

## 174. Replica exchange: unbiased, and not a cure (Part LXV)

Nothing in the earlier parts said how the conformations were generated. In practice they are
generated by replica exchange: several copies of the system are run at different temperatures and
neighbouring copies periodically attempt to swap configurations. Two claims are routinely made for
it — that it is unbiased, and that it fixes the sampling problem that motivated it.
`ReplicaExchange.lean` proves the first exactly and shows the second is false as stated.

The first is a theorem, not an approximation. Write the state of the exchanging pair as `(i, j)`,
the configuration held at the hot temperature and the one held at the cold temperature, with
extended weight `prodW p1 p2 (i,j) = p1 i · p2 j`. The swap attempt is exactly the Metropolis move
for that extended weight under the symmetric proposal "exchange the two configurations", so
`swapK_detailedBalance` gives reversibility, `swapK_stationary` stationarity, and
`prodW_marginal_fst` / `prodW_marginal_snd` say the marginal at each temperature is that
temperature's Boltzmann ensemble. `reK_stationary` extends this to the full chain of
within-temperature moves plus swaps. `accSwap_boltz` records why the move is implementable at all:
the acceptance probability is `min 1 exp((b2−b1)(E j − E i))`, and no partition function appears.

What the acceptance rate measures is settled exactly. `meanAcc_eq_overlap` and
`meanAcc_eq_one_sub_tv` prove that the mean acceptance is `1 − TV(pi, pi∘swap)`: it is the overlap
of the extended weight with its swap, and it is nothing else. Two consequences follow immediately,
in opposite directions. `meanAcc_le_exp_neg_gap` shows that if the two energy histograms are
separated by a gap `g`, the acceptance is at most `exp(−(b2−b1) g)` — attained exactly, by
`meanAcc_two_state`, in a two-state instance, so the bound is sharp — and therefore
`deltaBeta_le_of_meanAcc` caps the temperature step at `log(1/al)/g` if acceptance `al` is to be
held, and `replicas_needed` turns that into a ladder length: spanning `[b 0, b K]` needs at least
`(b K − b 0) g / log(1/al)` rungs. The energy gap between two temperatures grows with the number
of residues, so the number of replicas needed for a disordered region grows with its size; this is
the exact form of the folklore that tempering a large IDR is expensive. In the other direction,
`meanAcc_flat_eq_one` shows the acceptance rate is worthless as a diagnostic on its own: at equal
temperatures every swap is accepted while the two replicas sample the same ensemble and the
exchange conveys nothing at all.

The negative result is the one worth stating carefully, because it is the assumption most often
left implicit. Suppose the within-temperature move set never crosses between a set `A` of
conformations and its complement — at *any* temperature. This is not exotic: a topological trap,
a chirality, a bond that no move breaks, a hard constraint in the integrator all have this form.
Then `reK_conserves_count` shows that the number of replicas holding a configuration inside `A` is
a conserved quantity of the whole extended chain: the swap move permutes the configurations
currently held, and the within-temperature moves cannot cross the boundary, so nothing can change
the count. `re_iterate_support` propagates this through any number of sweeps, and
`replica_exchange_cannot_repair_ergodicity` states the conclusion: a run started with every replica
inside `A` never visits the complement, at any temperature and after any number of sweeps, even
though the Boltzmann ensemble puts positive weight there. Raising the temperature lowers barriers;
it does not connect what the move set disconnects. `IDR.replica_exchange_laws`
(`PartSixtyFive.lean`) bundles the five statements.

## 175. Umbrella sampling: overlap is exactly the condition (Part LXVI)

The other half of the protocol is the free-energy profile, and profiles are assembled from biased
windows. `Umbrella.lean` asks which features of the assembled profile are theorems about the data
and which are artefacts of the assembly.

Inside a window everything is exact. `winDist_unbias` shows that reweighting the biased histogram
by `exp(V)` returns precisely the conditional distribution of the target on the window support —
the bias is removed exactly, with no assumption at all. What a window cannot do is fix its own
weight relative to another window: `winDist_smul` shows the data are invariant under rescaling the
target, which is exactly why WHAM and MBAR return free-energy offsets only up to a global constant
and must be tied together through overlaps.

`free_energy_unidentifiable` prices the failure of that tying, and the price is total. Suppose the
conformation space splits into `A` and its complement in such a way that every window support lies
entirely inside one side. Then for *every* positive ratio `r` there is a target with exactly that
ratio of populations which reproduces every window histogram exactly. The relative free energy of
the two regions is not poorly determined; it is completely undetermined, and no amount of further
sampling inside those windows changes that. A profile whose two basins were sampled by disjoint
sets of windows reports, for the difference between them, whatever the recombination procedure put
in.

The converse is equally sharp. `umbrella_identifiable` proves that if consecutive window supports
meet and the windows cover the space, then two targets with the same window data are proportional;
`population_determined` converts proportionality into equality of every population and every
free-energy difference. Overlap of the window supports is therefore exactly the right condition —
necessary by the previous theorem, sufficient by this one — and the folklore rule "make sure
neighbouring windows overlap" is, stated that way, a theorem.

The last statement is the one that bites in practice. An overlap that exists in the design need not
exist in the data. If the shared region carries probability `m` under the window distribution, then
`overlap_missed_ge` bounds the probability that an `N`-frame run misses it entirely below by
`1 − N m`, and `half_of_runs_blind` says that for `N ≤ 1/(2m)` at least half of all runs record
nothing there. The recorded supports are then disjoint, and the unidentifiability theorem applies to
what was actually recorded. `thin_overlap_example` makes this concrete on three states: two windows
`{0,1}` and `{1,2}` genuinely overlap, but if no frame visits state `1` the recorded supports are
`{0}` and `{2}`, and the free energy of `{0,1}` against `{2}` is free to take any value.
`IDR.umbrella_laws` (`PartSixtySix.lean`) bundles the five statements.

## 176. How much sequence a pairwise theory can carry (Part LXVII)

The whole point of a model of a disordered region is that it is sequence-resolved: change the
charge pattern and the predicted ensemble changes. `SequenceDegeneracy.lean` asks how much sequence
information the analytic theories in use can carry, and the answer is exactly computable.

Sequence charge decoration, the random-phase free energies and every preaveraged Debye–Hückel
treatment share one structural feature: the sequence enters through pairwise terms whose strength
depends on the pair only through its separation `d = j − i` along the chain. `pairSum_eq_shell`
proves that any such energy is identically `Σ_d k d · shell d`, where `shell d = Σ_i q i · q (i+d)`
is the charge autocorrelation. This is an identity, valid for every kernel `k`, hence at every
screening length, every salt concentration and every temperature. The consequence,
`confEnergy_eq_of_shell_eq` and `sequence_blind`, is stated in the strongest available form: two
sequences with equal autocorrelation give the *same conformational energy function*, so every
functional of the model agrees — the partition function, the Boltzmann ensemble at every
temperature, the radius of gyration, every observable average, every derived free energy. There is
no measurement, within such a model, that separates them. Sequence charge decoration itself is one
such functional (`scd_congr_of_shell_eq`), and is blind for exactly the same reason.

That would be harmless if the autocorrelation determined the sequence. It does not.
`shell_seqA_eq_seqB` exhibits two 12-residue charge patterns,

    seqA = + + + + − + − − + + − −      seqB = + + − + + − + + + − − −

with the same net charge, the same composition (seven positive, five negative) and the same
autocorrelation at every separation; `seqA_ne_seqB` records that they are genuinely different
sequences — distinct, not each other's reverse, and not each other's charge inversion. These are
not pathological objects; they are plain patterns of twelve residues, of the kind a designed
variant or a natural paralogue differs by.

What does separate them is third order. `triple_seqA` and `triple_seqB` compute the
nearest-neighbour three-body correlations as `2` and `−6`. So the degeneracy is not a deep
obstruction, it is a statement about the order of the theory: a model of a disordered region is
sequence-resolved only in so far as its sequence dependence is not pairwise-in-separation, and the
first term that lifts the degeneracy is cubic in the charge sequence. Read with Part LXIV — where a
pair-potential model fitted to pair structure gets the three-body structure maximally wrong — the
two parts say the same thing from the two ends of the modelling pipeline: the pairwise level is
closed under a degeneracy that the physics does not respect. `IDR.sequence_resolution_laws`
(`PartSixtySeven.lean`) bundles the four statements.


## 177. The full ladder: `K` replicas, and the invariant that survives (Part LXVIII)

Section 174 treats one exchanging pair, which is what a single attempt is; a production run is a
ladder of `K` replicas. The obvious question is whether either conclusion — exactness, and the
impossibility of repairing a disconnected move set — was an artefact of having only two replicas.
Neither was, and `ReplicaLadder.lean` proves both in the general case by isolating the mechanism.

The mechanism is that an exchange is a Metropolis move *along an involution*. Whatever the state
space, whatever the positive weight, and whatever involution `s` of the states one moves along,
`invK_stochastic` gives a transition kernel, `invK_detailedBalance` reversibility,
`invK_stationary` stationarity, and `invAcc_mean_eq_overlap` the identification of the mean
acceptance with the overlap of the weight and its image. Swapping the configurations held by
replicas `a` and `c` is an involution of the ladder's state space, so all four statements apply
verbatim, for any `K` and any pair.

On the ladder the arithmetic collapses to the pair involved. `ladderW_ratio` shows the ratio of
extended weights is `(p a (x c) · p c (x a))/(p a (x a) · p c (x c))` — the other `K − 2` factors
cancel — and `ladder_acc_boltz` turns this into `min 1 exp((b c − b a)(E (x c) − E (x a)))`, again
with no partition function. `ladder_unbiased` is the exactness statement in the form a practitioner
needs: for every replica `k` and every observable `f`, the extended-ensemble average of `f` applied
to replica `k`'s configuration is exactly `∑_i p k i · f i`, the Boltzmann average at that
replica's own temperature, whatever the other replicas are doing; `ladder_swap_stationary` adds
that the exchange preserves the extended ensemble.

The conserved count survives too, and this is the statement to act on.
`ladder_swap_conserves_count` says an exchange merely permutes the configurations held, so it
cannot change how many of them lie in `A`; `ladder_within_conserves_count` says a within-temperature
update that never crosses the boundary of `A` cannot change the count either; and
`ladder_iterate_support` and `ladder_cannot_repair_ergodicity` propagate this through any number of
sweeps: a run started with all `K` replicas inside `A` never visits a configuration outside `A`, at
any temperature. Adding rungs to the ladder does not help, and neither does raising the top
temperature, because the obstruction is a conserved quantity of the move set and the temperatures
do not appear in it. What has to be argued before a tempering run is believed is that the moves
connect the conformations — a statement about the move set, not about the thermostat.
`IDR.replica_ladder_laws` (`PartSixtyEight.lean`) bundles the four statements.

## 178. How many experiments an ensemble costs (Part LXIX)

An ensemble model of a disordered region is, concretely, a population vector over a library of `m`
conformations, and it is fitted against `k` experimental averages — SAXS intensities at a list of
angles, PRE rates, RDCs, chemical shifts, FRET efficiencies. Each of them is a known linear
functional of the population vector. The question nobody can avoid, and that `Restraints.lean`
answers exactly, is how large `k` must be relative to `m` for the data to pin the ensemble down at
all.

The counting is elementary and the consequence is not. `exists_null_direction`: the `k` observables
together with normalisation are `k + 1` linear functionals on an `m`-dimensional space, so as soon
as `k + 1 < m` they annihilate a nonzero signed population direction — a redistribution of weight
that every measurement is blind to, simultaneously. What makes this a physical statement rather
than a dimension count is `exists_perturbation`: if the target is *interior*, populating every
library conformation with weight at least `d` — which is exactly what disorder means, since nothing
is excluded — then the blind direction can be followed a finite distance without leaving the
simplex. The scale of the move is set by the distance of the target from the boundary, not by the
direction.

Putting the two together, `restraints_insufficient` states the design law: below the threshold
there is a *bona fide* ensemble, nonnegative and normalised, reproducing **every** measured average
exactly, at population distance at least `2 d` from the truth. On a uniform target that is `2/m`
(`uniform_restraints_insufficient`) — the weight of two whole conformations, not a perturbative
wobble. Read in the contrapositive (`experiments_needed`), an experiment set that determines an
interior target to better than `2 d` must contain at least `m - 1` restraints: one per
conformation. And since a library realising conformational entropy `H` has `exp H` members,
`restraints_exp_entropy` says the restraint count needed grows exponentially in the conformational
entropy. Tens of restraints against a library of thousands is not under-determination at the
margin; it is under-determination in almost every direction.

Two corollaries close the argument. `cross_validation_cannot_certify`: splitting the experiments
into a fitting set and a held-out validation set only redistributes them, so if the *total* count
is below threshold, a single ensemble far from the truth matches the fitting data and the held-out
data alike — agreement with held-out restraints is not evidence the counting law forbids.
`indicator_restraints_determine`: the threshold is sharp, since reading off all but one population
does determine the ensemble, so `m - 1` is the exact answer and not an artefact of the argument.
`feasible_convex` explains why there is no lucky choice of observables below the threshold: the
feasible set is a convex slice of the simplex, and the deficient directions form a subspace.
`IDR.restraint_counting_laws` (`PartSixtyNine.lean`) bundles six statements.

## 179. What is verifiable in a reported ensemble (Part LXX)

Section 178 says the data almost never determine the ensemble. That makes the next question the
operative one: of the numbers actually extracted from a fitted ensemble — the population of a
bound-like substate, a helical content, a contact frequency, a mean radius of gyration — which are
consequences of the experiment, and which are consequences of the reference ensemble?
`Identifiability.lean` answers it completely for the linear functionals, which is what a reported
number is.

The answer is a subspace. Call `f` *determined* if every ensemble matching the data reports the
same average of `f` as the truth. `determined_of_mem` proves that every `f` in the span of the
constant function together with the measured observables is determined — the constant appears
because normalisation is itself a restraint, always available and always used.
`not_determined_of_notMem` proves the converse, and the proof is the dual of the perturbation
argument: an `f` outside the span is separated from it by a linear functional; a linear functional
on population space *is* a signed population direction (`exists_vector_of_dual`); being null on the
span it is invisible to every measurement; and the interior target can be moved along it. So
`determined_iff_mem`: against an interior target, `f` is verifiable **iff**
`f ∈ span {1, g 1, …, g k}`. Identifiability is not a matter of degree, of regularisation strength,
of prior width, or of how good the fit looks. It is membership in a subspace fixed by the
experiment list alone.

That subspace has a dimension, and `identifiable_dim` bounds it: at most `k + 1`. An experiment set
of size `k` supports at most `k + 1` independent verifiable numbers, whatever the library, whatever
the sampling protocol, whatever the fitting method. `exists_population_not_determined` makes it
concrete: if `k + 1 < m`, at least one single-conformation population is not determined at all.

This is a protocol, and it is checkable. Publish `m` and `k`; publish the observable list; and
quote as a result only a functional exhibited as a combination of the constant and the measured
observables, with the combination given. A number so exhibited is a theorem about the experiment.
A number not so exhibited is a property of the reference ensemble — which the maximum-entropy
analysis of Part III already identified as where such numbers come from, and which Section 178
quantifies. `IDR.identifiability_laws` (`PartSeventy.lean`) bundles five statements.

## 180. The precision floor of an ensemble measurement (Part LXXI)

Sections 178 and 179 assume the restraints are matched exactly. They never are: a restraint is
matched to a tolerance — an error bar, a chi-squared target, the statistical uncertainty of a
finite run. `Tolerance.lean` prices the tolerance, and the price does not fall when the restraint
list grows.

The construction is the simplest one available. `exists_pair_perturbation`: move population `c`
from conformation `b` to conformation `a`. The ensemble moves by at least `2 c` in population
distance, and the `j`-th measured average changes by exactly `c (g j a - g j b)`. Only the
*contrast* of the observable between the two conformations is visible to the data. Two immediate
consequences follow.

The first is qualitative and is the conformational counterpart of the sequence degeneracy of
Section 176. `pair_degeneracy`: if two conformations give the same value of *every* measured
observable, their relative population is not determined at all — an ensemble `2 d` away in
population distance matches every measurement exactly, for every restraint count `k`. A pair of
conformers the observables cannot separate is a pair whose populations are reported by the prior.

The second is quantitative and is the point of this part. `precision_floor`: with observables
bounded by `G` — every back-calculated observable on a finite library is bounded — and the data
matched to tolerance `eps`, there is a genuine ensemble consistent with all the data at population
distance `min (2 d) (eps / G)` from the truth. The bound contains no `k`. Adding experiments cannot
push it down; only reducing `eps` — more frames, more photons, better calibration — can. And the
scaling is right rather than an artefact of the construction: `tolerance_ceiling` gives the
matching upper bound, `2 (m - 1) eps`, when the populations themselves are known to tolerance
`eps`. The resolution of an ensemble measurement is bracketed between `eps / G` and
`2 (m - 1) eps`, linear in the tolerance from both sides.

Parts LXIX–LXXI together give the full accounting for a reported ensemble. The restraint count
fixes how many independent numbers can be verified (at most `k + 1`); membership in the measured
span fixes which ones; and the tolerance fixes to what precision, at a floor of order `eps / G`
that no further experiment removes. An ensemble model of a disordered region is therefore a
measurement with a stated resolution — and stating that resolution is part of stating the model.
`IDR.precision_laws` (`PartSeventyOne.lean`) bundles four statements.

## 181. The design and reporting theorem: what to measure, and how to report it (Part LXXII)

Sections 178–180 are negative, and taken alone they would counsel despair: the restraints are too
few, the verifiable functionals too rare, the resolution floored. They point, however, at a
constructive procedure, and `ReportDesign.lean` proves that procedure optimal.

Start from the end of the pipeline. A study reports `r` numbers — the population of a bound-like
conformer, a helical content, a contact frequency, a mean radius of gyration — that is, `r` linear
functionals of the ensemble. Two questions follow: what must be measured to make those numbers
verifiable, and what is the fitted ensemble contributing to them?

The second has the more striking answer. `mem_measured_iff_combo`: a functional is identifiable
exactly when it carries a certificate — an explicit representation `f = c0 + Σ_j c_j g_j` in terms
of normalisation and the measured observables. `report_value_from_data`: the certificate *computes
the answer from the data*. Every ensemble consistent with the measured averages `D_1, …, D_k`
reports the value `c0 + Σ_j c_j D_j` for that functional. So the identifiable content of an
ensemble fit is a linear readout of the deposited data; the fitted ensemble contributes nothing to
it, and two consistent ensembles necessarily agree (`consistent_ensembles_agree`). This is the
strongest form verifiability can take: a reader recomputes the reported number from the data
without downloading the model. Conversely — by Section 179 — anything that *cannot* be recomputed
that way was not measured.

The first question has the reassuring answer. `report_sufficient`: measuring the functionals one
intends to report determines all of them, and every combination of them with normalisation, so `r`
experiments buy `r + 1` verifiable dimensions. And this is optimal: `report_min_experiments` shows
that any experiment set whatsoever making all `r` reports verifiable against an interior target
satisfies `finrank (span {1, f_1, …, f_r}) ≤ k + 1`, so `report_needs_r` gives `r ≤ k` whenever the
reports and normalisation are linearly independent. The minimum number of experiments needed to
verify `r` independent reported numbers is exactly `r` — and, notably, it does not depend on the
size of the conformational library at all.

That contrast is the resolution of the whole development. Determining the *ensemble* of a
disordered region costs `m - 1` experiments, exponentially many in the conformational entropy
(Section 178), and is hopeless. Determining `r` *reported numbers* costs exactly `r`, and is
routine — provided the experiment list is chosen to span what will be reported, and provided
nothing outside that span is reported. A model of an intrinsically disordered region can therefore
be made fully verifiable at a stated resolution. Not by pinning down the ensemble, which no
feasible experiment does, but by matching every claim to the measurements through the criterion of
Section 179, quoting it with its certificate, and stating the resolution of Section 180 alongside.
`IDR.report_design_laws` (`PartSeventyTwo.lean`) bundles the four statements.

## 182. How much of a sequence a pairwise charge model can carry (Part LXXIII)

The predictive claims made for disordered regions almost always run through a *patterning
parameter*: a single number computed from the charge sequence — `SCD`, `kappa`, a screened-Coulomb
chain energy — which is then correlated with compaction or with condensation. Every one of these
has the same shape: a sum over pairs, `E(q) = sum_{i<j} w(j-i) q_i q_j`, with a kernel `w` that
encodes the physics of screening and chain connectivity. `ChargePatterning.lean` asks what such a
model can see.

The answer is exact. Regrouping the pair sum by separation gives
`pairEnergy_eq_sum_autocorr`: `E(q) = sum_{d=1}^{N-1} w d * C(d)`, where `C(d)` is the charge
autocorrelation at lag `d`. Every pairwise, separation-dependent charge model — for every kernel,
every salt concentration, every temperature — is a *linear functional of `N - 1` numbers*. And the
converse holds: because each single autocorrelation coordinate is itself realised by a Kronecker
kernel (`pairEnergy_delta_kernel`), two sequences give equal energies under every kernel exactly
when their autocorrelations agree (`autocorr_eq_iff_pairEnergy_eq`). The autocorrelation vector is
therefore precisely the sufficient statistic of the whole model class: nothing coarser, nothing
finer.

That would be a formality if the fibres of the map were trivial. They are not.
`homometric_autocorr` exhibits two length-nine charge sequences, `+ + - + + - - - +` and
`- + + - + + + - -`, with the same composition and the same autocorrelation at every lag. They are
genuinely distinct — not equal, not each other's reversal, not each other's negation, not the
reversed negation — yet `homometric_blind` shows they have the same energy under *every* kernel,
`homometric_scd` the same `SCD`, and `homometric_debye` the same Debye-screened energy at every
salt concentration. No amount of experimental precision, and no refitting of parameters, can make
a pairwise charge theory distinguish them.

What does distinguish them is a three-body descriptor: `homometric_triple_ne` computes
`sum_i q_i q_{i+1} q_{i+3}` as `+2` for one and `-2` for the other. This locates the missing
information exactly. A model of a disordered region that intends to be complete on the sequence
side needs either many-body sequence terms or an explicit conformational ensemble; a patterning
parameter, of any kernel whatsoever, is a projection onto `N - 1` coordinates and should be
reported as such. `IDR.charge_patterning_laws` (`PartSeventyThree.lean`) bundles the statements.

## 183. Multivalent binding: what a Hill slope can be (Part LXXIV)

Disordered regions bind through multiple short motifs, and their binding curves are routinely
summarised by a Hill coefficient. `BindingPolynomial.lean` asks what that number is a statement
about, treating the binding polynomial `Z(x) = sum_k a_k x^k` as the analytic object it is.

Two derivative identities do the work. `hasDerivAt_meanOcc` proves that the mean occupancy is
`d ln Z / d ln x` and that *its* derivative is the occupancy variance. Two consequences are
immediate and assumption-free: occupancy is nondecreasing in ligand activity
(`meanOcc_nondecreasing_deriv`), whatever the site energies, and the Hill slope *is* a variance —
so a steep binding curve is a statement about the width of the occupancy distribution, not about
any particular mechanism.

The bounds follow. `hill_le_valence`: the slope never exceeds the valence. `hill_eq_valence_iff`
and `hill_allOrNone`: equality holds exactly for the all-or-none polynomial, so a fitted Hill
coefficient equal to the valence asserts that no partially bound state is populated at any ligand
concentration — a strong structural claim usually made inadvertently. At the other end,
`hillInd_le_one` proves by Cauchy–Schwarz that independent sites can never exceed slope one, with
equality exactly for identical sites (`hillInd_identical`).

A measured Hill slope is therefore bracketed by two model-free numbers, and its position in that
bracket measures one thing only: the deviation from independence of the sites. It does not measure
conformational change, allostery or induced fit; those are interpretations added afterwards.
`IDR.binding_polynomial_laws` (`PartSeventyFour.lean`) bundles the statements.

## 184. The exact critical point of a chain-length demixing model (Part LXXV)

Condensate formation by disordered regions is modelled, nine times out of ten, by the
Flory–Huggins free energy of a chain of `N` segments. `FloryHuggins.lean` computes its critical
point instead of quoting it.

The curvature of the free energy has an exact factorisation (`curvature_identity`) whose minimum
over the composition interval is explicit. That yields the critical composition
`phiC N = 1/(1 + sqrt N)` and the critical coupling `chiC N = (1 + sqrt N)^2/(2N)`, and — the part
usually left implicit — a two-sided sharpness statement: `no_demixing_below_chiC` proves the free
energy is convex, hence has no demixing at all, at or below `chiC`, while `fh_demixes_above_chiC`
constructs an explicit demixing witness above it. The threshold is not an estimate.

Three corollaries are what a model builder actually uses. `chiC_strictAnti`: longer chains demix
at strictly weaker coupling, so chain length is itself a driver of condensation.
`chiC_gt_half` with `chiC_tendsto_half`: the critical coupling is always above `1/2` and tends to
it, so `1/2` is a floor no chain length can beat. `phiC_tendsto_zero`: the critical composition of
a long chain is dilute, which is why condensates of long disordered regions form at low
concentration. `IDR.flory_huggins_laws` (`PartSeventyFive.lean`) bundles the statements.

## 185. From sequence to phase diagram, and the blind spot it inherits (Part LXXVI)

Sections 182 and 184 are the two halves of what a predictive condensation model claims: sequence
in, phase diagram out. `SequencePhase.lean` joins them through the affine sequence-to-coupling law
`chiEff chi0 lam P = chi0 + lam P`, which is the shape of every published correlation between a
patterning parameter and condensation.

The join produces an exact and testable threshold. `demixes_iff_gt_threshold`: with positive slope,
the model predicts a condensate precisely when the patterning parameter exceeds
`(chiC N - chi0)/lam`, and predicts stability at every composition otherwise. It also produces a
trade-off: `threshold_strictAnti` shows that threshold decreases strictly with chain length, so
within the model, blockiness and length buy the same thing and can be exchanged.

The link is not vacuous. `blocky_demixes_alternating_stable` puts two length-four sequences of
*identical composition* — blocked and alternating — on opposite sides of the threshold at one chain
length and one chemistry: composition does not decide condensation, patterning does.

And the link inherits exactly the blindness of Section 182. `homometric_same_phase_diagram`: for
every kernel, every sequence-to-coupling law and every chain length, the homometric pair has
literally the same free-energy density, hence the same phase diagram and the same verdict
(`homometric_same_verdict`). A pairwise-electrostatics theory of condensation cannot explain any
measured difference between those two sequences — and if a difference is measured, the whole model
class is refuted, not merely its fitted parameters. The honest way to report such a model is with
the patterning coordinate it uses, the chain length it was fitted at, and the sequences it cannot
distinguish. `IDR.sequence_phase_laws` (`PartSeventySix.lean`) bundles the statements.

## 186. What a radius of gyration reports, and what it cannot (Part LXXVII)

The most-quoted number about a disordered region is its radius of gyration.
`ChainGeometry.lean` states exactly what that number is, in any real inner-product space:
`rg2_eq_pairSum` proves `Rg^2 = (1/2n^2) sum_{i,j} |x_i - x_j|^2`. The radius of gyration is one
fixed *linear* functional of the matrix of squared interbead distances, with no reference to the
coordinates. That is a licensing statement in the sense of Section 181 — it is exactly the kind of
linear, recomputable report those sections permit — and simultaneously a limitation:
`rg2_congr_of_dist_eq` shows any two conformations with the same distance matrix have the same
size, so `n(n-1)/2` numbers are compressed into one before any experimental error enters.

For the architecture biology actually presents — a folded domain with a disordered partner — the
relevant statement is the two-module parallel-axis law `rg2_union`:
`n Rg^2 = n_A Rg_A^2 + n_B Rg_B^2 + (n_A n_B/n) d^2`, with `d` the distance between the module
centroids. It is one equation in three unknowns, and it cuts both ways. Constructively,
`inter_module_dist_sq_le` turns it into a usable experimental inequality: because block radii are
nonnegative, a measured global `Rg` is a hard ceiling on the module separation,
`d^2 <= (n^2/(n_A n_B)) Rg^2`. Cautionarily, `rg2_tradeoff` exhibits two explicit four-bead chains
with exactly the same global `Rg^2 = 2`, one with compact modules held `2` apart and one with a
fourfold expanded module and coincident centroids. Tail expansion and module separation are
exchangeable at fixed size; a single global number reports neither.

Finally, chain connectivity gives a computable ceiling. If consecutive beads are within `b`, then
distances grow at most linearly along the chain (`norm_sub_le_bond`) and `rg2_le_rod` gives
`Rg^2 <= b^2 (n^2 - 1)/12`; `rod_rg2` shows the straight rod attains it exactly, so the bound is
the true supremum. A reported radius of gyration should be read against that ceiling, against the
degeneracy above it, and with the distance-matrix identity that makes it verifiable.
`IDR.chain_geometry_laws` (`PartSeventySeven.lean`) bundles the statements.

## 187. Modification scanning: the experiment that breaks the sequence blind spot (Part LXXVIII)

Section 182 is a negative result about one kind of experiment — comparing sequences. Biology
routinely performs a different kind on disordered regions: it modifies them. `Phosphorylation.lean`
solves that perturbation exactly inside the pairwise charge model, and the result is that
modification is strictly the more informative probe.

Removing a charge `z` at one site is a rank-one perturbation. `pairEnergy_phos` gives the
single-site law `E(phos z k q) = E(q) - z h_k`, where `h_k` is the *local field* at the modified
site: the kernel-weighted sum of all the other charges. `phos_comm` adds that modifications of
distinct sites commute, so the model class forbids any dependence of the outcome on the order in
which a kinase writes multiple marks — an immediately falsifiable prediction, given the known
hierarchies of multisite phosphorylation.

The central result concerns two sites. `phospho_epistasis`: the non-additivity of a double
phosphorylation, `Delta_kl - Delta_k - Delta_l`, is exactly `z^2 w(d)`, where `d` is the spacing of
the two sites. It does not depend on the sequence, on the composition, or on where the pair sits
along the chain (`phospho_epistasis_indep_of_sequence`). Two consequences follow. First,
`phospho_no_three_body`: the third-order interaction of three phosphosites vanishes identically, so
a measured three-body effect is evidence of many-body physics rather than of a mis-fitted kernel.
Second, and this is the point of the part, `kernel_of_epistasis` and
`pairEnergy_eq_of_epistasis_eq`: scanning the double-phosphorylation epistasis over spacings
determines the kernel `w` itself, and hence the model's prediction for *every* sequence.

Set that against Section 182. Sequence comparison sees only `N - 1` autocorrelation coordinates and
is provably blind on the homometric pair; a modification scan identifies the interaction kernel
outright. The reason is structural: a modification acts at a single site, so its second-order
response is the kernel evaluated at a single spacing, whereas sequence variation only ever enters
through the autocorrelation. The design consequence is concrete — a pairwise model whose kernel is
to be determined should be fitted to modification data.

`phospho_dissolves_condensate` closes the part with the phenotype the theory has to reproduce. A
four-residue polycationic patch under a contact kernel sits above the demixing threshold of
Section 184 at chain length one; a single phosphorylation with the physiological charge `z = 2`
takes it below. The model predicts a condensate that one kinase event dissolves — the regulatory
logic that makes disordered regions interesting, derived rather than assumed.
`IDR.phosphorylation_laws` (`PartSeventyEight.lean`) bundles six statements.

## 188. Salt: what screening keeps, and why condensation can be reentrant (Part LXXIX)

Salt is the cheapest knob in the laboratory and the least carefully modelled in the theory. Inside
the pairwise charge model of Section 182, adding salt means screening the kernel; writing the Debye
factor in the fugacity variable `x = exp(-kappa b)` compresses the entire salt axis into one real
parameter, with kernel `screen x d = x^d / d`. `Screening.lean` then shows the patterning energy is
a power series in that parameter whose coefficients are exactly the charge autocorrelations:
`energy N q x = sum_{d=1}^{N-1} (x^d/d) C(d)`. Salt reweights the `N - 1` coordinates a pairwise
model can see; it never adds a new one.

The high-salt limit is then a theorem rather than a slogan. `energy_high_salt` bounds the
difference between the energy and `C(1) x` by `x^2 sum_{d>=2} |C(d)|/d` for all `0 <= x <= 1`, so
as screening increases the surviving sequence information collapses onto the nearest-neighbour
charge correlation, at an explicit rate. `energy_neg_of_small` is the corollary a modeller needs: a
sequence whose neighbouring charges anticorrelate cannot be condensation-favouring at sufficiently
high salt, whatever it does at low salt.

The interesting behaviour is in between. Because the coefficients `C(d)` of a real sequence can
alternate in sign, the series need not be monotone in `x`, and `Screening.lean` exhibits a
twelve-residue charge sequence for which it is not: the patterning energy is *negative* at zero
salt, *positive* at intermediate salt, and *negative* again at high salt
(`energy_reent_one`, `energy_reent_four_fifths`, `energy_reent_half`). By the intermediate value
theorem there is then a zero on each side of the favourable window (`reentrant_window`): the
condensation-favouring region of salt space is an interval bounded away from both limits.
`reentrant_demixing` states the phase consequence against the critical coupling of Section 184 —
the model predicts a condensate at intermediate salt and stability at every composition at zero and
at high salt.

The claim worth extracting is a negative one about attribution. Reentrant salt dependence of
condensation is often read as evidence for ion-specific binding, charge condensation or hydration
effects. None of that is present in this model; the reentrance follows from a sign pattern in the
charge autocorrelation and the shape of the screened kernel alone. It is therefore a mechanism that
must be excluded before a more elaborate explanation is invoked — which is exactly the kind of
statement a model of a disordered region should be able to make about itself.
`IDR.screening_laws` (`PartSeventyNine.lean`) bundles four statements.

## 189. Crosslinking mass spectrometry: the yield is a population, not a distance (Part LXXX)

A chemical crosslinker joins two residues only in those conformations in which the two side chains
are within the reach of its spacer arm, and the mass spectrometer reports how much of the sample
was joined. The measured number is therefore not a distance: it is `freq E d r`, the *fraction of
the ensemble* whose site–site distance `d` is at most the reach `r` — one value of the cumulative
distribution of that distance (`Crosslink.lean`). `freq_nonneg`, `freq_le_one` and `freq_mono` say
exactly that: a yield lies in `[0,1]` and is nondecreasing in the spacer length. Being an ensemble
average, it is also a linear restraint of precisely the kind counted in Sections 178–179, so `k`
crosslinks are `k` restraints and support at most `k+1` verifiable numbers.

The first consequence is that the standard experiment already proves disorder without any
structural modelling. A single structure predicts every yield to be `0` or `1` (`freq_dirac`);
hence `not_deterministic_of_fractional`: a single sub-stoichiometric yield `0 < f < 1` refutes
every single-structure model of the region. No geometry, no second crosslink, no violated-restraint
count is needed — the fractional yield *is* the evidence.

The second is a test on the data rather than on the model. Call two crosslinks *mutually exclusive*
when no conformation satisfies both (`Exclusive`); `sum_freq_le_one` shows the yields of mutually
exclusive crosslinks sum to at most one, so a data set exceeding one comes from no ensemble
whatsoever and must be a measurement or assignment error.

The third upgrades the familiar qualitative reading. `card_ge_of_exclusive` and
`card_ge_of_exclusive_model` show that `k` mutually exclusive crosslinks that are all observed
force at least `k` distinct conformations — in the target and in every model reproducing the data.
"The crosslinks are incompatible with a single structure" thereby becomes a quantitative lower
bound on ensemble size. The geometric input is `exclusive_of_separated` (anchors further apart than
twice the reach give exclusive links), and the bound is realised by an explicit instance: a tail tip
found at each of three anchors `0, 10, 20` a third of the time, crosslinked with a spacer reaching
`3`, gives three exclusive yields of `1/3` (`triad_freq`), so every model of that region needs at
least three states (`triad_card_ge`).

Finally, the positive design statement. Because the yields of one site pair are values of one
cumulative distribution, a *series* of crosslinkers with spacer reaches `0, h, 2h, …` recovers the
mean site–site distance by a layer-cake sum, with error at most the spacing:
`mean_distance_from_series` gives `|E.expect d − series E d h n| ≤ h` whenever the distance stays in
`[0, n·h]`. One crosslinker reports one number about the ensemble; a spacer-length series reports a
distribution. That is the design recommendation this part contributes — and it is the same lesson
as the rest of the development, in the one experiment whose output is most often mistaken for a
distance restraint on a single structure.
`IDR.crosslink_laws` (`PartEighty.lean`) bundles five statements.

## 190. Ensemble reweighting as convex duality: the certificate and the diverging multiplier (Part LXXXI)

Section 43 established *what* maximum-entropy ensemble refinement computes — the exponentially
tilted ensemble is the unique minimum-relative-entropy ensemble matching the data. It says nothing
about how the multipliers are found, or what happens when the restraints cannot be matched at all.
Both are questions about the dual objective `dual q f d lam = log Z(lam) − ⟨lam, d⟩` (`Dual.lean`),
and both have clean answers.

*The problem is convex.* `logPartition_convex` proves the log-partition function convex in the
multipliers, by applying the weighted arithmetic–geometric mean inequality termwise over the pool,
and `dual_convex` carries this to the objective, the data term being linear. Refinement therefore
has no spurious local minima: a descent method that stops has stopped at the global answer, whatever
the pool and whatever the restraints.

*The optimality gap is a relative entropy.* `dual_gap` is the substantive identity: if the tilt at
`lam` reproduces the data, then for *every* multiplier vector `mu`,
`dual mu − dual lam = KL(tilt lam ‖ tilt mu)`. The duality gap is not merely nonnegative; it is
exactly the information distance between the two reweighted ensembles. That makes the shortfall of
a candidate fit a reportable certificate, in the same units as the refinement objective itself.
Two corollaries follow immediately: `dual_min_of_matches` (a matching multiplier vector is a global
minimiser) and `tilt_eq_of_both_match` — two matching multiplier vectors produce the *identical*
ensemble. Multipliers are routinely non-identifiable, because experimental restraints are usually
linearly dependent on a large pool; the fitted ensemble is not.

*Feasibility is exactly boundedness.* `dual_ge_of_feasible` shows that if any ensemble on the pool
reproduces the data, the objective never falls below `log c`, where `c` is the smallest prior
weight — the optimisation cannot run away. Conversely `dual_le_of_separated` and
`dual_unbounded_of_separated` show that if some direction `u` in restraint space separates the data
from every conformation of the pool by a margin `eps`, then `dual(t·u) ≤ −t·eps`, so the objective
is unbounded below and the multipliers diverge.

The practical reading is the last one. A refinement run whose Lagrange multipliers blow up is
commonly treated as a numerical problem, to be damped by a regulariser or a cap on the multipliers.
The theorem says the divergence is informative: up to the separation condition it is a *proof* that
the restraints are inconsistent with the conformational pool, and the correct response is to enlarge
the pool or re-examine the measurements — not to suppress the symptom. Together with the counting
laws of Sections 178–179, this completes the account of ensemble refinement as an inference
procedure: convex, certified by a relative entropy, unique in the ensemble it returns, and
self-diagnosing when the pool is wrong.
`IDR.reweighting_duality_laws` (`PartEightyOne.lean`) bundles five statements.

## 191. Heterogeneous kinetics: what a single rate constant reports (Part LXXXII)

Everything above treats the structure of a disordered region as a distribution. Its *kinetics*
deserve the same treatment, and the consequences are just as sharp. Proteolysis, degradation,
chemical modification and labelling all proceed from whichever conformations expose the relevant
site, so in the slow-exchange limit — interconversion slower than the chemistry, which is the regime
in which substates are resolvable at all — the decaying object is a mixture: weights `w j` over
substates, a first-order rate `k j` for each, and a surviving fraction
`surv w k t = Σ_j w_j exp(−k_j t)` (`Heterokinetics.lean`).

Three facts follow, none of which requires any special structure in the rate distribution.

*Heterogeneity looks like stability.* `surv_ge_exp_mean` proves `exp(−⟨k⟩t) ≤ surv t`: a mixture
always survives at least as well as a homogeneous population decaying at its own mean rate. A broad
rate distribution therefore mimics protection with no protective mechanism present at all — the
apparent resistance is a statement about the width of the ensemble, not about any shielding.

*Only the initial slope reports the average.* `apparentRate_zero` shows the apparent rate at `t = 0`
is exactly the population mean rate; `apparentRate_antitone` shows that it then falls,
monotonically, at all times, for every mixture. The proof is a Chebyshev-type symmetrisation of a
double sum rather than a differentiation, so the statement is exact rather than asymptotic. The
fragile conformations are consumed first and the survivors are progressively enriched in slow ones;
`apparentRate_ge_min` and `apparentRate_le_max` bracket the drift between the extreme substate rates
and `surv_ge_slowest` identifies the long-time tail with the slowest substate. `surv_log_convex`
records the same fact as convexity of `log surv`.

*The effect is large at realistic contrast.* `twoW`/`twoK` put half the population at rate `1` and
half at rate `1/100`. The initial apparent rate is `101/200` (`twoState_rate_zero`), and by `t = 10`
the apparent rate is already below `1/10` (`twoState_rate_ten`) — a fivefold change in the fitted
"rate constant" of the very same sample, obtained purely by measuring later.

The conclusion for model design mirrors the structural one exactly. A single fitted rate constant is
not a property of a disordered region: it is a property of the ensemble *and* of the measurement
window. A model that predicts one rate is underspecified in the same way as a model that predicts
one structure; what must be predicted, and reported, is a rate distribution.
`IDR.heterogeneous_kinetics_laws` (`PartEightyTwo.lean`) bundles five statements.

## 192. Ion mobility: the arrival-time distribution is the datum (Part LXXXIII)

Ion mobility is the one experiment on a disordered region that delivers a size and a width in the
same trace, and it is therefore the sharpest test of the thesis of this development. Each conformer
family drifts at its own time `t j`, proportional to its collision cross section, and is broadened
to an instrumental width `sig j`; the recorded arrival-time distribution is the weighted mixture of
those peaks, and what a fit reports is its first two moments (`IonMobility.lean`).

*The width splits, exactly.* `width2_eq_instr_add_spread` is the law of total variance written in
the instrument's units: measured squared width = mean instrumental variance + conformational spread
of the drift times. Two consequences follow immediately. First, the spread is identifiable
(`spread_eq_width2_sub_instr`) — but only as the excess of the measured width over the instrument
function, so an ion-mobility experiment reports ensemble heterogeneity only to the extent that it
characterises its own broadening. Second, the measured width can never fall below the instrument
function (`width2_ge_instr`): a trace narrower than the instrument is a calibration error, not a
rigid conformer.

*A single structure predicts the instrument and nothing else.* `width2_single` computes the width a
one-conformation model predicts: the instrumental variance, exactly. So `excess_width_refutes_single`
— any excess width whatsoever, however small, refutes every single-structure model of the region.
This is the ion-mobility twin of the sub-stoichiometric crosslink yield of Part LXXX: a fractional
observable that no deterministic model can produce.

*The centroid selects nothing, and neither does the centroid with the width.* The measured mean
arrival time is bracketed by the conformer drift times (`min_le_meanTime`, `meanTime_le_max`), and
conversely `mixture_hits_intermediate` shows that *every* value strictly between a compact and an
extended conformer is reproduced exactly by a two-state mixture with both weights strictly positive:
a reported "measured cross section" lying between two candidate structures is evidence for both and
for neither. Adding the width does not repair this. `moment_ambiguity` exhibits two ensembles on the
drift times `0,1,2,3,4` — half at `1` and half at `3`, versus `1/8, 3/4, 1/8` at `0, 2, 4` — with the
same centroid `2` and the same spread `1`. Mean and width together still do not determine the
ensemble; only the distribution does.

*And resolving conformers has an exact price.* With peak widths set by the resolving power,
`sig = t/R`, and a separation criterion `k(sig_1+sig_2) <= |t_1-t_2|`, `resolved_iff` shows that two
families are separated precisely when `k(t_1+t_2) <= R|t_1-t_2|` — the requirement scales as the
inverse of the *fractional* cross-section difference. `resolved_two_percent` is the concrete
instance: two conformers differing by two percent are separated (at `k = 1`) exactly when `R >= 101`.
Below that resolving power the two families are reported as one peak of excess width, which is
exactly the observable the previous paragraphs analyse. When peaks *are* resolved they bound model
complexity from below: `card_ge_of_distinct_peaks` shows `k` distinct peaks force at least `k`
conformers in any model reproducing them, the ion-mobility counterpart of the crosslink counting
bound.

The design conclusion is the standard one of this development, restated in a new instrument. An
ion-mobility measurement of a disordered region is a distribution; the honest report is a centroid
together with an excess width and, where the resolving power allows, a peak count. A model of the
region must predict all three. The single collision cross section that a single structure would
predict is the one quantity the experiment does not deliver.
`IDR.ion_mobility_laws` (`PartEightyThree.lean`) bundles five statements.

## 193. The moment problem: what a finite list of averages can and cannot fix (Part LXXXIV)

Every experiment analysed above returns a moment of some conformational observable: a mean distance,
a mean square, a sixth power, an arrival-time width. A refinement protocol receives `k` such numbers
and is asked for the distribution behind them. This part settles what that list determines
(`MomentProblem.lean`).

*Without prior structural information, nothing.* `moment_indeterminacy` produces, for every `k`, two
ensembles over the same `k+2` conformations with observable values `0,1,…,k+1`: one supported on the
even-numbered conformations with weights `C(k+1,i)/2^k`, the other on the odd-numbered ones. Both are
genuine probability vectors, they share no conformation whatsoever, and their moments agree at every
order `p ≤ k`. The mechanism is the vanishing of the alternating binomial sum
`Σ_i (-1)^i C(k+1,i) i^p` for `p < k+1` — the `(k+1)`-st forward difference of a polynomial of degree
`p` — proved as `alt_choose_pow_sum`. The consequence is uncomfortable and exact: a model that shares
no structure at all with the truth can reproduce `k` measured averages to machine precision. Small
residuals are therefore not evidence that an ensemble has been measured. `two_state_instance` is the
familiar minimal case: half a population at `0` and half at `2` versus all of it at `1` — same mean,
no shared conformation, told apart only by the variance.

*With a support, everything.* `weights_eq_of_moments_eq` shows that once the candidate conformations
are fixed and distinct, the moments of orders `0,…,k` determine the populations of `k+1` of them
uniquely. The proof is Lagrange interpolation: moments of orders up to `k` determine the ensemble
average of every polynomial of degree at most `k`, and the Lagrange basis polynomial at a node
returns that node's weight. So the structural prior — the pool, the candidate set, the assumed
support — is not a convenience of the protocol; it is the ingredient that converts averages into an
ensemble. A reported ensemble is a statement about the data *and* about the pool, and the pool must
be reported with it.

*And with no prior at all, a bound.* What a moment certifies outright is a population bound.
`markov_bound`: at most `⟨x⟩/a` of the ensemble can have `x ≥ a`. `chebyshev_bound`: at most `Var/a²`
of it can deviate from the mean by `a` or more. These survive the indeterminacy above — they hold for
*every* ensemble consistent with the measurement — and they are therefore the honest form in which a
fit to an average should be reported: not "the distance is `⟨r⟩`", but "at most this fraction of the
ensemble is beyond this distance". And they cannot be improved: `markov_sharp` exhibits, for every
mean in range, a two-conformation ensemble attaining the Markov bound exactly.

This closes the loop with the rest of the development. Parts I onwards argue that a model of a
disordered region must predict a distribution; this part shows what the experiments, taken as
finitely many averages, can pin that distribution down to — nothing at all on their own, a unique
answer given a declared support, and certified population bounds in every case.
`IDR.moment_problem_laws` (`PartEightyFour.lean`) bundles five statements.

# Part LXXXV — The capacity threshold, made exact and made testable

Everything the development says about capacity up to here is a prohibition: with fewer components
than the target has populated states, the model is wrong, and no amount of data or training repairs
it. A prohibition tells you where not to build. This part makes the same quantity two-sided, and
then commits it to an experiment that can fail.

*The exact law.* Let the target populate `m` states with populations `w_0 >= w_1 >= ... >= w_{m-1}`,
the form in which an NMR, single-molecule or ensemble-deposition study reports them, and let
`tail P k` be the population outside the `k` most populated states. Then the smallest `l1` error
attainable by any `k`-component model is exactly `2 tail P k`: `ell1_ge_two_tail` proves no model
does better, and `ell1_trunc_le_two_tail` exhibits one that does exactly that, by keeping the top
`k` states and renormalising. That equality — `minErr_eq` — is the whole content of the part.
Everything else is a reading of it.

*It predicts a curve, not a verdict.* Adding the `(k+1)`-st component improves the attainable error
by exactly `2 w_k` (`minErr_step`), so the entire error-versus-capacity curve is fixed in advance by
the measured populations. The curve is strictly decreasing below `k = m` (`minErr_strictMono_below`)
and exactly zero from `k = m` on (`minErr_eq_zero_iff`): a kink at the measured state count, with no
improvement afterwards. That is a different prediction from the smooth diminishing returns one would
expect on general grounds, and the difference is measurable — the location of the last significant
drop is a statistic with a value predicted before the model is run.

*And it sharpens as the ensemble broadens.* On a target with `m` equally populated states the
attainable error at capacity `k` is exactly `2(m-k)/m` (`minErr_uniform`), so for any fixed
component count there are targets on which the best that count can do approaches the maximal error
`2` (`fixed_capacity_degrades`). The cost of a fixed architecture is not a constant; it grows with
the disorder, by a stated amount.

*It says what to build.* `optimalK` returns the least component count achieving a requested
accuracy; it achieves it (`optimalK_spec`), nothing smaller does (`optimalK_min`), and it never
exceeds the state count (`optimalK_le`). This is the part of the development that points at a
design rather than away from one.

*It survives the trip to real measurements.* Models emit structures, not labels; the comparison is
made after assignment to reference states. Assignment is a push-forward, a push-forward cannot
increase the number of mixture components, and so the floor still applies to the assigned model
(`stateLevel_floor`). Reported populations carry error bars; a profile perturbation of `eta` in `l1`
moves the floor by at most `2 eta` (`minErr_perturb`), and once the tolerance sits `2 eta` below the
reported floor the predicted failure holds against every profile within the bars
(`baseline_must_fail_robust`). And the failure is observable rather than metric:
`missed_states_of_under_capacity` names a set of conformations to which the under-capacity model
assigns population zero and which really carries at least `tail P k` of the population.

*What the proof buys, and what it does not.* `baseline_must_fail` proves that on an admissible
record no model with the practitioner's fixed component count reaches the pre-registered tolerance.
That half of any experiment is therefore not at risk: if it fails, something in the bridge is wrong
— the state count, the populations, the read-out, or the claim that the baseline really has that
many components — and the protocol in `PREREGISTRATION.md` says which to inspect. The other half is
at risk, and provably: `at_capacity_not_sufficient` shows an `m`-component model can be maximally
wrong, so "at or above threshold" never entails a good fit, and `confirms_refutable` exhibits
outcomes consistent with every theorem here on which the pre-registered criterion is false. The
formal statement of the split is `confirms_iff_threshold_fits`: given admissibility, the criterion
holds exactly when the threshold-respecting model meets its tolerance. All of the scientific content
is in that clause, and none of it can be supplied by a proof.

*Status.* The worked records in `Falsification.lean` use stipulated populations, not measurements,
and every numerical claim about them is checked by computation. No data has been analysed in this
development, and no claim about any real system or published tool is made here. What is delivered is
the threshold, the design rule, the read-out bridge, the error-bar robustness, and a pre-registered
protocol — stated before the fact, refutable, and with the entailed and the empirical halves kept
apart. `IDR.capacity_threshold_laws` and `IDR.prereg_test_laws` (`PartEightyFive.lean`) bundle them.

# Part LXXXVI — A cryo-EM map of a disordered region is an occupancy, not a structure

Single-particle reconstruction is the one experiment that appears to hand back coordinates directly,
and it is the experiment on which disordered regions most conspicuously fail to appear. Both facts
have the same explanation, and this part proves it: a reconstruction averages over the particles it
was built from, so the map of an atom is its *occupancy distribution over voxels* — the population of
particles putting that atom in each voxel — and nothing else. `RequestProject/CryoEM.lean` develops
that statement on a finite voxel grid.

*The map is a probability distribution.* `dens_nonneg`, `sum_dens_eq_one` and `dens_le_one`: the map
of atom `a` is nonnegative, sums to one over the grid, and never exceeds the value `1` that a single
rigid structure would place at its own voxel. So the units of a map are populations, and a low value
is a low population, not a blurred coordinate.

*Peak height is a conformation count.* If no voxel of an atom's map rises above `p`, then the atom
occupies at least `1/p` distinct voxels (`one_le_card_support_mul_peak`, `card_support_ge_inv_peak`).
The corresponding statement for what is displayed is `card_above_le_inv_threshold`: at most `1/t`
voxels per atom can survive a contour level `t`. Weak density is therefore quantitative evidence of
heterogeneity, with a number attached, and the number can be read off the deposited map.

*Disappearance is a theorem.* `uniform_spread_invisible`: an atom spread uniformly over `m` voxels
with `1/m < t` has *no* voxel above the contour level, so the displayed map contains none of it —
while `invisible_mass` records that the sub-threshold voxels carry the entire population. "No
density for residues 1–60" is a statement about populations, not about absence, and the map that
shows nothing is the map that has correctly reported a broad ensemble.

*What the map does determine.* `expect_from_map`: every single-atom average is an exact linear
read-out of the map, `Σ_j w_j f(pos_j a) = Σ_x f(x) dens(a,x)`. Mean position and positional variance
are therefore recoverable exactly, and `expect_eq_of_dens_eq` shows two ensembles with the same map
agree on all of them. This is the positive deliverable: a per-atom occupancy together with the
averages computed from it, quoted with the conformation count its peak height forces.

*What it does not.* `map_blind_to_correlation` exhibits two two-residue ensembles with *identical*
maps for both residues, one of which has the residues always in the same voxel and the other never.
Particle averaging destroys the joint distribution, so no reconstruction, at any resolution or
signal-to-noise, can referee between ensembles that agree residue by residue. Joint claims — contacts,
compaction, coupled motions — need an experiment that is not an average over particles.

*Classification does not repair it by itself.* For **any** partition of the particles into classes,
the class maps sum back to the total map (`sum_classDens`), so agreement with the consensus map is no
evidence that a classification is correct. And a `K`-class reconstruction remains a `K`-point model:
if the particles realise `n > K` distinct positions of an atom, every assignment of one structure per
class leaves a strictly positive mean squared error, however the classes are chosen
(`resid_pos_of_injective`) — the same capacity floor as elsewhere in this development, in the
reconstruction's own currency.

`IDR.cryoem_occupancy_laws` and `IDR.cryoem_blindness_laws` (`PartEightySix.lean`) bundle the two
halves.

# Part LXXXVII — How much of the ensemble has the sampling seen?

Every ensemble model of a disordered region is built from a finite sample: a trajectory, a generated
pool, a set of deposited conformers. The population the sample never visited is invisible to every
diagnostic computed from the sample — cross-validation, convergence plots, block averages all live
inside what was drawn — so the modeller needs a bound on it. `RequestProject/Coverage.lean` proves
what can be said, with the sample space `Fin N → Fin m` of `N` independent draws from an ensemble of
`m` states with populations `w`.

*The exact expectation.* The sample weights `sw w s = ∏_i w(s_i)` form a probability distribution
(`sum_sw_eq_one`); the probability a given state is never drawn is exactly `(1 − w_x)^N`
(`prob_unseen`); and therefore the expected unseen population is exactly the *missing mass*
`Σ_x w_x (1 − w_x)^N` (`expected_unseenMass`). This is an equality, not a bound.

*It never vanishes.* `missingMass_pos`: if some state carries population strictly between `0` and
`1` — that is, if the ensemble is not a single conformation — the expected unseen population is
strictly positive at every sample size. No finite sample ever certifies that it has seen the whole
of a disordered ensemble; the honest report is a number, not a claim of convergence.

*The cost of coverage.* On a uniform ensemble the missing mass is `(1 − 1/m)^N`
(`missingMass_uniform`), and Bernoulli's inequality bounds it below by `1 − N/m`
(`missingMass_uniform_ge_one_sub`). Hence `sample_size_needed`: leaving at most `eps` of the
population unseen requires at least `(1 − eps)·m` draws. The sampling cost is linear in the number
of populated states — and the number of populated states of a disordered region is, elsewhere in
this development, exponential in its length. That is the quantitative form of the sampling problem.

*And the positive half: the missing mass is estimable from the sample itself.* `good_turing`: the
expected number of states drawn exactly once in a sample of size `N + 1` equals `(N + 1)` times the
missing mass at size `N`. So the fraction of a conformational sample consisting of states seen
exactly once is an unbiased estimate of the population the sample is missing — computed with no
knowledge of how many states there are or what their populations are, and available from the sample
a modeller already has. Together with the capacity threshold of Part LXXXV, this is the pair of
numbers an ensemble model of a disordered region should carry: how many components it needs, and how
much population its sampling has not yet seen.

`IDR.coverage_laws` (`PartEightySeven.lean`) bundles six statements.

# Part LXXXVIII — Conditioning: what a regularised ensemble fit actually reports

Fitting an ensemble to data is an inverse problem, and every practical fit is regularised — by a
maximum-entropy prior, a Tikhonov penalty, an early stop, or a conformational pool chosen in
advance. Regularisation is usually presented as a technical necessity. It is an exact trade, and
`RequestProject/Conditioning.lean` computes it.

Diagonalise the measurement: a conformational mode of true amplitude `c` is seen with sensitivity
`s`, the datum is `d = s·c + n`, and the regularised reconstruction at penalty `lam` is
`recon = s·d/(s² + lam)`.

*What the fit returns.* `recon_eq_filt_add_noise`: exactly `filt·c + gain·n`, with filter factor
`filt = s²/(s² + lam)` and noise gain `gain = s/(s² + lam)`. Two numbers, both known before the data
arrive.

*Shrinkage is unavoidable.* `filt_lt_one`, `filt_mono_sens`, `filt_anti_pen`: the filter factor is
below one for every mode, rising with sensitivity and falling with the penalty. A regularised fit
never reports the amplitude it inferred; it reports a fraction of it, and the fraction is computable.

*Stability is bought, and priced.* `gain_le_inv_two_sqrt`: the penalty caps the noise gain at
`1/(2√lam)` uniformly in the sensitivity, and `gain_eq_at_sqrt` shows the cap attained at `s = √lam`,
so the constant is the right one. `gain_unbounded_of_no_penalty`: without a penalty the gain is `1/s`
and exceeds every bound as sensitivity falls — the ill-posedness the penalty exists to cure. And
`stability_bias_identity` gives the exchange rate exactly: `s·(1 − filt) = lam·gain`. Stability and
bias are one quantity read in two directions.

*The resolution boundary is known in advance.* `filt_ge_half_iff`: the datum outweighs the prior
exactly when `s² ≥ lam`. Below the boundary the reported amplitude is the prior's
(`prior_dominates_of_insensitive`), and a mode with `s² ≤ eps·lam` survives only to the fraction
`eps` (`recon_le_of_insensitive`). `error_le` collects the two contributions:
`lam|c|/(s² + lam) + |n|/(2√lam)`, bias plus noise.

*Across a spectrum.* `resolvedModes` is the set of modes with `s² ≥ lam`: the features the data
control. It can only shrink as the penalty rises (`resolvedModes_anti`), and on everything outside
it the fit reports the prior whatever the data say (`unresolved_report_prior`). That count — not the
number of fitted parameters, not the size of the conformational pool — is the number of numbers a
regularised ensemble fit has actually measured.

Design consequence: the penalty is part of the claim. A regularised ensemble model of a disordered
region should quote `lam`, the sensitivities of the features it reports, and hence which of those
features are data-driven and which are the prior seen through the fit.

`IDR.conditioning_laws` and `IDR.resolved_modes_laws` (`PartEightyEight.lean`) bundle the two groups.


## 194. From a threshold to a study design: power, visibility and the four numbers (Part LXXXIX)

The capacity law says what an under-capacity model gets wrong and by exactly how much. It does not
say how many molecules to watch, with what probe, or through how rich a read-out — and without
those, the law is a consistency statement rather than something a run could lose against. This part
supplies them.

*How many observations.* An under-capacity model gives population zero to a set of conformations
carrying true population `tau`. The probability that `n` independent observations all avoid that set
is exactly `(1 - tau)^n` (`Power.missProb_eq`) — an equality, not a bound. One observation inside it
makes the model's likelihood exactly zero (`Power.likelihood_zero_of_hit`), so the outcome is a
refutation rather than a worse fit statistic. Hence the sample size `ceil(log(1/alpha)/tau)`
(`Power.samplesFor_spec`), computable from the independently measured populations before any model
exists. And the anti-spin clause: with `n` draws the chance of seeing nothing is still at least
`1 - n·tau` (`Power.missProb_ge_one_sub`), so an under-powered null result is not support for the
baseline.

*Why the failure is usually invisible, and what to measure instead.* An observable with values in
`[a, b]` differs between two ensembles by at most `(b-a)/2` times their population `l1` distance
(`Visible.expect_gap_le_range`). A model exactly on the capacity floor therefore moves it by at most
`(b-a)·tau`: a fixed three-component mixture that is provably far away in population space can have
an excellent chi-squared against a low-contrast global observable. That is a mechanism, and a
theorem. The probe that can see the failure at precision `sigma` must have dynamic range at least
`sigma/tau` (`Visible.contrast_requirement`), and the extremal probe is the indicator of the omitted
states — a contact, a distance window, a labelled pair — which reports the entire discrepancy
(`Visible.optimal_reporter`). Independently, unless the measurement suite carries at least `m-1`
independent observables, some population the test is scored against is not determined by the data
(`Visible.observables_needed`).

*The instrument.* `Instrument.Study` is the record that must be frozen before modelling, and
`Instrument.report` computes from it the four numbers — components, observations, observables,
contrast — in exact rational arithmetic, each with a soundness theorem. On the stipulated
illustrative five-state record it returns five components, fourteen observations, four observables,
contrast `1/10`, baseline floor `2/5`, baseline excluded.

*The panel.* Freeze `n` records at level `alpha/n`, take that level's sample size on each, and every
system refutes its under-capacity baseline with probability at least `1 - alpha`
(`panel_design_laws`): multiplicity paid in planned observations rather than post-hoc corrections.

*Where to spend.* `budget_laws` puts the two routes to accuracy side by side on the same target.
On `m` equally populated states, accuracy `eps` costs exactly `ceil(m(1 - eps/2))` components, and
that price is payable by an explicit model with no data at all once the populated states are
known; a support-honest learner — any reweighting or weighted-frames model — needs `m(1 - eps)`
samples of a state space exponential in the length of the region. On broad ensembles, accuracy is
bought with capacity and state identification, not with pool size.

*The limits, kept explicit.* Being at capacity never entails fitting well, and a poor read-out
cannot certify the scored populations (`design_limits`). The empirical clause remains empty: no
number in this part is a measurement, and no claim is made about any real protein or any published
tool. `STUDY_DESIGN.md` is the protocol that would fill it.

## Part XC — the same test on a real instrument

Part LXXXIX prices the refutation under an idealisation: that an observation reveals the
conformation, so one molecule in the omitted set drives an under-capacity model's likelihood to
zero. Part XC removes it. With a binary reporter of sensitivity `se` and specificity `sp`, the
signal is `τ·(se+sp−1)` rather than `τ` (`IDR.Noisy.readRate_sub`); the exact mean and variance
of the read-out law give a Chebyshev bound and hence a test whose two error probabilities are at
most `α` after `⌈1/(α·(τJ)²)⌉` molecules (`IDR.PartXC.noisy_capacity_power`) — quadratic where
the idealised design paid a logarithm, `693` molecules against `14` on the illustrative record
(`IDR.NoisyInstrument.demoNoisy_report`). A probe with conformation-dependent response must be
scored with its population-weighted mean sensitivity, and states it cannot see subtract from the
contrast and can invert it (`IDR.Reporter.contrast_eq`, `contrast_neg_of_dark`). Combining probes
with an OR rule caps the index at the product of the specificities, so panels of many imperfect
probes lose contrast exponentially (`IDR.Panel.probe_panel_laws`). Two limits are proved rather
than assumed: a real population is exactly indistinguishable from a calibration error of size
`τ·J`, at every sample size (`IDR.PartXC.calibration_is_the_binding_constraint`), and no analysis
of any kind attains both errors `α` below `(1−2α)/Δ` molecules
(`IDR.Lower.molecules_lower_bound`), with repeated reads of one molecule contributing nothing
(`IDR.Lower.replicate_no_help`). `REALISM.md` is the prose account; `IDR.PartXC.realistic_design_verdict`
is the single statement.

## Part XCI — the discovery list, when the whole proteome is screened

Part XC prices the refutation on one system read by one realistic reporter. No one tests one
disordered region: a claim about disorder in general is made by screening thousands of candidate
regions and reporting those that fire, and a *list* has an error rate of its own. The Bonferroni
split of Part LXXXIX.4 controls the probability of any false entry, but over twenty thousand
candidates it demands a per-candidate level of `2.5·10⁻⁶`, which is not an experiment anyone runs.

Part XCI proves the alternative and prices it. `IDR.FDR.bh_fdr_control` is the
Benjamini–Hochberg theorem in the finite product experiment: independent candidates, superuniform
nulls, and the expected fraction of the discovery list at level `q` that is null is at most
`|H₀|·q/N` — at most `q` — whatever the non-null candidates do. The proof turns on
`IDR.FDR.bhR_eq_iff_pZero`, which says that on the event that a candidate's p-value is below the
`k`-th threshold, the step-up index is unchanged by setting that p-value to zero; the event
`{BH stops at k}` then belongs to the other candidates, and independence factorises the term. And
the list is never smaller than Bonferroni's (`IDR.FDR.bh_dominates_bonferroni`), so the weaker
guarantee is bought with strictly more discoveries, not fewer.

The p-values are built from the read-out law of Part XC and nothing else.
`IDR.Screen.chebP_superuniform` shows that the two-moment (Chebyshev) statistic is a valid
p-value at every level, using only the exact mean and variance — no normal approximation, no
exact binomial tail. `IDR.Chernoff.sum_recProb_exp` computes the read-out law's moment generating
function exactly, `(1 - q + q·e^λ)ⁿ`, and Markov's inequality on it gives
`P(count ≥ n·q + a) ≤ exp(-a²/4n)` at every finite `n` (`IDR.Chernoff.chernoff_tail`), hence a
second valid p-value (`IDR.Chernoff.expP_superuniform`). Both drive the same screen, because BH
asks only for superuniformity.

What they cost differs by orders of magnitude. `IDR.Screen.screen_budget_quadratic`: a two-moment
screen costs at least `N²/(q·Δ²)` molecules — quadratic in the number of candidates, because a
candidate must reach `q/N` and Chebyshev buys tail probability only quadratically.
`IDR.Chernoff.screen_exp_cheaper`: the exponential-tail screen costs
`⌈(16·log(N/q) + 1/α)/Δ²⌉` per candidate, logarithmic in the size of the screen. On the worked
record of twenty thousand regions at `q = 0.05` with reporter contrast `Δ = 0.1`, that is
`40 000 000` molecules per candidate against at most `22 800`
(`IDR.PartXCI.worked_cost_ratio`). The physics is identical in the two designs; the factor of a
thousand is the inequality the analysis is prepared to prove.

Two limits are stated rather than hidden. Independence across candidates is assumed here, and real
screens couple candidates through shared reagents and calibration; under arbitrary dependence BH
needs the harmonic correction, which is proved in Part XCIV
(`IDR.DepBH.benjamini_yekutieli_level`) and shown there to be unimprovable
(`IDR.DepBHSharp.bh_fdr_eq_harmonic`). And superuniformity requires each candidate's baseline
rate to be known to better than its contrast — the calibration floor of Part XC, which is per
candidate, so multiplicity control does not soften it. `IDR.PartXCI.screen_laws` is the single
statement.

## Part XCII — a run that may be watched

Every sample size in Parts XC and XCI is a *fixed* one: the number of molecules is chosen before
the run, and the data may be scored once, at the end. No laboratory works that way. Molecules
arrive one at a time, the counter is visible, and the run stops when the answer looks clear or
when the material runs out. Under the fixed-`n` theory that behaviour has no proved error rate at
all, and the inflation is not a technicality: two looks, each at level `1/2`, give a combined
false-refutation probability of `3/4` (`IDR.Seq.peeking_inflates_error`, restated as
`IDR.PartXCII.watching_costs_nothing_only_for_the_wealth_test`).

Part XCII rebuilds the capacity test so that stopping is free, and proves both of its halves.

**What is watched.** Not the count of positive reads but the *wealth* of a bet on the alternative
read-out rate: the running likelihood ratio (`IDR.Seq.wealth`), which `IDR.Seq.wealth_eq_ratio`
identifies with `probL q₁ / probL q₀` on the molecules scored so far.

**The level, at every horizon.** `IDR.Seq.ville` is Ville's inequality, proved from scratch by
induction over the read-out word — no measure theory, no martingale library: under the null the
probability that the wealth *ever* reaches `c`, at any point of a run of any length, is at most
`1/c`. Hence `IDR.Seq.anytime_valid`: stopping the run the moment the likelihood ratio exceeds
`1/α`, and calling that a refutation, has type-I error at most `α` — whatever the run length,
however often the counter is inspected, and whatever rule is used to decide to stop
(`IDR.Seq.anytime_valid_all_horizons` states the uniformity in `n` explicitly).

**The power, and hence a molecule count.** A test that never stops has error rate zero and no
value, so the design is finished only with the other half. The expected log wealth after `n`
molecules is exactly `n·KL(q₁‖q₀)` (`IDR.Seq.expected_logWealth`) and its variance is exactly
`n·klVar` (`IDR.Seq.variance_logWealth`); both rates are strictly positive whenever the
alternative read-out rate differs from the null one (`IDR.Seq.kl2_pos`, `IDR.Seq.klVar_pos`, the
latter through the closed form `IDR.Seq.klVar_eq`). Chebyshev on those two exact moments gives a
finite-`n` type-II bound `n·klVar/(n·KL − log c)²` (`IDR.Seq.sequential_power`), and
`IDR.Seq.powerSamples_spec` turns it into an explicit number of molecules delivering power
`1 − β` at level `α`.

**Every stopping rule, not just the crossing rule.** An experimenter does not stop at the
crossing; they stop when the reagent runs out, when the shift ends, or when the number on the
screen looks convincing, and only then look at the wealth. `IDR.Seq.Declares` models a run under
an arbitrary rule — any function of the reads seen so far, with no restriction of any kind — and
`IDR.Seq.stopping_rule_valid` proves that under the null every such run declares a refutation
with probability at most `α`; `IDR.Seq.no_rule_beats_the_bound` states the uniformity over rules
and horizons, so the level cannot be evaded by choosing the rule after seeing the data.
`IDR.PartXCII.capacity_test_under_any_stopping_rule` is the same statement for the capacity test.

**What the freedom costs.** Nothing, at small `α`. Pinsker applied to the reporter's two rates
(`IDR.Seq.kl2_ge_two_sq`) puts the sequential horizon at most `⌈log(1/α)/(2Δ²)⌉` molecules
(`IDR.Seq.sequentialHorizon_le_of_contrast`), where `Δ = τ·J` is the same contrast — omitted
population times Youden index — that Part XC's fixed design pays for. The two designs share the
`Δ²`; they differ in the confidence level, `log(1/α)` against `1/α`, and
`IDR.Seq.sequential_cheaper` is the explicit inequality showing the sequential run is the shorter
one once `√α ≤ KL/(4Δ²)`.

**An estimate, not only a verdict.** Running the same wealth test against every candidate value
of the omitted population and reporting the candidates not yet excluded gives a *confidence
sequence*. It is nested — `IDR.Seq.excluded_of_prefix`, from `IDR.Seq.crossed_append`: a candidate
excluded on a prefix stays excluded, so the reported set only shrinks as molecules accumulate —
and it covers: `IDR.Seq.confSeq_coverage` and `IDR.Seq.confSeq_coverage_all_horizons` bound by `α`
the probability that the true value is ever dropped, at every horizon at once, with
`IDR.Seq.population_confSeq_coverage` and `IDR.PartXCII.sequential_population_estimate` stating it
in the units of the problem. The plot of the interval against the molecule count may therefore be
watched, and the run stopped on it, without correction.

**On the capacity test itself.** `IDR.PartXCII.sequential_capacity_laws` instantiates all of it
where it belongs: the null is the under-capacity model's own prediction, that the probe fires at
its false-positive rate `1 − sp`, and the alternative is the truth, `1 − sp + τ·J`. The theorem
states the signal (`τ·J`), the level at every horizon, the strictly positive evidence rate and
variance, and the power at `powerSamples` molecules — level and power together, for an experiment
whose length is not fixed in advance. `IDR.PartXCII.sequential_horizon_from_contrast` gives the
run length in the reporter's own currency and `IDR.PartXCII.worked_record_admissible` checks the
hypotheses are satisfiable on a stipulated record.

Two limits stay where Part XC left them. The reads are independent, and the null rate `1 − sp`
must be known: sequential analysis does nothing about the calibration confound, and a reporter
miscalibrated by `τ·J` defeats the wealth test exactly as it defeats the counting test, because
the wealth is then a bet on an instrument artefact. The power bound uses two moments only, so it
is conservative, and `klVar` is a property of the reporter rather than of the model under test.

## Part CXXXVIII — Pressure, the axis that tests whether the solvent is in the model

Every perturbation used so far — temperature, salt, denaturant, a binding partner — leaves one
thermodynamic axis untouched. Hydrostatic pressure couples to the partial molar volume of a
conformer, that is, to solvent exclusion, packing voids and the compressed hydration shell, so it
is the sharpest available test of whether a model of a disordered region represents the solvent at
all. `RequestProject/PressureEnsemble.lean` builds the axis: a finite conformer ensemble with a
reference free energy `G i` and a volume `V i`, Boltzmann weight `exp(-(G i + p·V i))`.

Three exact laws hold on it. `hasDerivAt_mean` is the fluctuation–response identity
`d⟨f⟩/dp = −Cov(V, f)`, whose special case `mean_volume_deriv` says that the mean volume falls at a
rate equal to the volume variance — the microscopic compressibility, nonnegative by `var_nonneg`.
`mean_antitone_of_comonotone` is Le Chatelier's principle without calculus: any observable
comonotone with volume has an average that never increases with pressure, strictly so by
`mean_volume_strict_anti` once two conformers differ in volume. `population_tendsto_zero` is the
high-pressure limit: the ensemble collapses onto the conformers of least volume whatever their
free energies, which is what pressure denaturation is.

The two-state section makes the titration usable — `pop_half_iff` (the midpoint pressure is exactly
`−ΔG₀/ΔV`), `pop_strictMono`, `pop_tendsto_one`, `volume_change_recovery` and, for kinetics,
`activation_volume_recovery`. It closes with the identifiability boundary. Expanding to second
order, `ΔG(p) = ΔG₀ + ΔV·p − (Δβ/2)p²`, `pressure_curve_two_point_underdetermined` shows that data
at two pressures determine the compressibility change *not at all*: for every value of `Δβ` there
is an exactly fitting `(ΔG₀, ΔV)`, and the fits differ at every other pressure.
`pressure_curve_three_point_unique` shows three pressures pin all three coefficients.
`pressure_design_law` bundles the clauses.

## Part CXXXIX — Prolines: the substates a model may not average over

Disordered regions are proline-rich and the peptidyl–prolyl bond isomerises on a timescale of tens
of seconds, six to nine orders of magnitude slower than the backbone dynamics an ensemble is built
to describe. `RequestProject/ProlineIsomer.lean` makes the consequence quantitative. With `n`
prolines and independent cis fractions, `sum_isoWeight_eq_one` and `populated_card` say that all
`2ⁿ` isomeric substates are populated, `isoWeight_le_pow` that none exceeds `(1 − δ)ⁿ` of the
population, and `cover_card_ge` that any set of substates carrying a fraction `m` of the molecules
contains at least `m·(1 − δ)^(−n)` of them: enumerating isomers is exponentially expensive, and
ignoring them means reporting a mixture as a structure. `average_not_attained` exhibits a mixture
whose average is a value no member takes, and `hidden_substate_observable` is the exact
non-identifiability — per-isomer values can be moved arbitrarily far apart without changing the
measured average at all. The kinetic half, `no_switch_prob_ge` and `switch_needs_long_window`, says
the label does not move on the experimental window: at the textbook rate `10⁻² s⁻¹`, a one-second
experiment finds `99%` of the molecules in the isomer they started in
(`demo_frozen_on_second_timescale`). `proline_substate_law` collects the clauses: the isomeric
state has to be an explicit slow discrete label, not one more fast degree of freedom.

## Part CXL — What a pressure cell can buy

`RequestProject/PressureDesign.lean` puts the pressure axis through the same design questions as
every other probe in this project. **Resolution:** a volume difference enters only as `p·ΔV`, so
over a range `[0, P]` read to `eps` it is invisible below `eps/P` (`volume_difference_invisible`)
and detectable at the top pressure at or above it (`volume_difference_detectable`) — an exact floor
in both directions. In laboratory units one megapascal times one millilitre is one joule, so
`demo_volume_floor` shows a `200 MPa` cell read to `10⁻² kT` cannot see `0.1 mL/mol`, of the order
of one buried water per chain. **Blindness:** `pressure_blind_of_cov_zero` and, exactly,
`mean_const_of_volumes_equal` — an ensemble whose conformers share a volume has no pressure
response whatsoever, so a pressure series is a complement to a size probe, never a substitute.
**Extrapolation:** `two_point_extrapolation_arbitrary` is the sharpest form of the Part CXXXVIII
boundary — from two pressures, the free energy at any third pressure can be made *any* value by a
second-order model fitting both points exactly, so two-point data support no extrapolation at all,
including the usual one back to ambient pressure. `pressure_design_report` bundles the clauses.

## Part CXLI — Where the volume comes from, and a model that cannot be pressure denatured

The partial molar volume is not a primitive: `V i = v_vdW + v_void·voids i − v_hyd·surf i`, and only
the last two terms depend on the conformation. `RequestProject/PartialVolume.lean` turns that into a
falsification test. `residue_additive_blind` is the negative result: a volume obtained by summing
residue contributions — no voids, no conformation-dependent surface — assigns the same volume to
every conformer and therefore predicts, exactly, no pressure response of any observable at any
pressure. Such a model has no compressibility and cannot be pressure denatured; it fails the
experiment structurally, not numerically. `voids_antitone` and `voids_strict_anti` are the positive
result — pressure squeezes voids out — and `cov_volume_left` with `hasDerivAt_mean_voids` give the
response exactly, `d⟨voids⟩/dp = −v_void·Var(voids) + v_hyd·Cov(surf, voids)`, with
`volume_variance_of_voids` identifying the compressibility as `v_void²·Var(voids)`.
`partial_volume_law` states the dichotomy: a model that is to have a pressure axis at all must
carry a conformation-dependent void or hydration term.

## Part CXLII — The closed phase diagram

With temperature and pressure varied together, the stability of a structured element inside a
disordered region — a transient helix, a folding-upon-binding motif — is to second order a
quadratic form in `(t, p) = (T − T₀, P − P₀)`. `RequestProject/StabilityDiagram.lean` proves what
follows when the heat-capacity and compressibility changes make that form negative definite
(`quadratic_nonpos_of_neg_disc`, `negDefinite_bound`, with the explicit modulus `stabModulus`).
`stability_region_bounded`: the set of conditions at which the element is stable lies inside an
explicit disc — the phase diagram is closed, so stability is not monotone in either axis and no fit
that assumes it is can be extrapolated. `cold_and_heat_denaturation`: if the element is stable at
the reference condition and the heat-capacity change is positive, the temperature axis crosses the
boundary on *both* sides, so cold denaturation is a theorem rather than an extra assumption; and
`pressure_denaturation_exists` says the same on the pressure axis. `elliptic_stability_law` bundles
the three: heat, cold and pressure denaturation are three crossings of one closed curve.

## Part CXLIII — The membrane interface: excluded area and the transfer scale

Many disordered regions work at a lipid surface, and two features of that experiment have no
analogue in solution binding. `RequestProject/Interface.lean` formalises both. **Excluded area:** a
bound peptide covers `n` lipids, so the isotherm is Stankowski's `x = A(1 − n x)^n`, not Langmuir's.
`bindFun_strictMono` and `exists_unique_coverage` make the coverage well defined — exactly one
solution in `[0, 1/n]` for each condition — `coverage_lt_saturation` and `coverage_near_saturation`
show the surface saturates at one peptide per `n` lipids and approaches that limit, and
`langmuir_underestimates_affinity` is the design clause: reading the same measured coverage with
the Langmuir isotherm returns a strictly smaller affinity for every `n ≥ 2` and every nonzero
coverage, a one-signed bias that better statistics do not remove. **The transfer scale:**
interfacial binding energies are additive over residues, so a binding curve measures one
composition sum per sequence. `transfer_scale_unidentifiable` shows that if every measured sequence
carries equal counts of two residue types, their individual transfer energies can be moved
arbitrarily far apart with no change to any prediction, and `transfer_scale_separated` shows the
same perturbation is visible on an unbalanced sequence: the remedy is compositional design, not
more measurement. `interface_design_law` bundles the clauses.

## Part CXLIV — Chaperones: an ATP cycle is not a stronger binder

A disordered region inside a cell is not left alone. Hsp70, Hsp90, trigger factor and the
small heat-shock proteins bind it, hold it, and release it while hydrolysing ATP, and the
question a model has to answer is what the hydrolysis buys that binding alone cannot.

`RequestProject/ChaperoneCycle.lean` answers it on the smallest scheme in which the question
can be posed: three states — free disordered client `U`, chaperone-bound client `C`, folded
client `F` — and six positive rate constants running round the triangle. The stationary
state is solved exactly by the Kirchhoff matrix-tree weights, and the net probability flux is
shown to be the *same* number across all three edges,

    J = (f0 f1 f2 − b0 b1 b2) / Z,

so that the Kolmogorov cycle condition `f0 f1 f2 = b0 b1 b2` is precisely the condition for
equilibrium.

The consequence is a no-free-lunch theorem. **If the cycle condition holds, the stationary
folded : disordered ratio is `b0 / f2`** — the ratio set by the spontaneous folding and
unfolding rates alone. Every chaperone rate has cancelled: how tightly the chaperone binds,
how fast it captures, how fast it releases, none of it appears. Two cycles that share the
spontaneous rates and both satisfy the cycle condition have exactly the same folded fraction.
An equilibrium chaperone, however good a binder, cannot move the client's folded fraction.

Away from the cycle condition the ceiling disappears in both directions. Raising the capture
rate `f0` alone pushes the stationary folded : disordered ratio past any target `R`; raising
the rate `b2` at which the chaperone recaptures the folded client alone pushes it below any
target `eps` — the driven machine can unfold a client that would fold on its own. The price is
an entropy production `J · log A` that is proved nonnegative, and zero exactly at equilibrium.

The design law is `chaperone_design_law`. Its content for a model of a disordered proteome is
sharp: a chaperone represented by *any* term added to a free-energy function is provably
unable to reproduce chaperone action, because every such representation satisfies the cycle
condition and therefore leaves the folded fraction at `b0 / f2`. Chaperones have to enter as
rates, and the cycle has to be driven.

## Part CXLV — Disulfide topology and the loop entropy of a disordered region

Cysteines are the one chemistry that can staple a disordered region to itself, and every
staple closes a loop. `RequestProject/Disulfide.lean` takes the standard loop-closure free
energy `ΔG(L) = c0 + ν log L` and asks what it delivers.

One disulfide delivers nothing about the exponent: for *any* exponent one cares to assume
there is an offset that reproduces the measured closure free energy exactly, so a model
calibrated on a single loop has not measured a scaling law. Two loops of different length
identify the offset and the exponent uniquely, and the exponent comes out in closed form as
the slope in `log L`. Three loops turn the power law into a refutable claim: the three points
must be colinear in `(log L, ΔG)`, and if they are not, then no offset and exponent whatsoever
fit the data.

The topology results are the sharper ones. For four cysteines with consecutive gaps `a, b, c`
there are three pairings. The crossed pairing `(02)(13)` is always beaten by the
beads-on-a-string pairing `(01)(23)` — that much is universal. But the comparison with the
nested pairing `(03)(12)` is decided by an explicit inequality, `a·c` against `b·(a+b+c)`, and
both outcomes actually occur: with equal gaps the sequential topology is cheaper, while with a
short middle gap and long flanks (gaps 10, 1, 10) the nested topology is cheaper. A model that
hard-codes a topology preference for disulfide-bonded disordered regions is therefore wrong on
an explicitly exhibited set of cysteine spacings; the preference is a function of the spacings,
and it flips.

## Part CXLVI — Turnover: a steady-state level is one number about two rates

Disordered regions in a cell are short-lived: they carry degrons, they are recognised by the
proteasome without needing to be unfolded, and their abundance is a balance between synthesis
and degradation rather than a property of a folding funnel. `RequestProject/Turnover.lean`
states what a model of that balance actually contains.

The abundance solves `m' = s − d·m`, and the solution `s/d + (m0 − s/d)·e^{−dt}` is proved to
solve it and to relax to `s/d`. The steady state is then shown to be invariant under
multiplying *both* rates by the same factor — and that rescaling is proved to be exactly a
rescaling of time, the same curve run at a different speed. So an abundance measurement,
however precise, fixes only the ratio: for every degradation rate there is a synthesis rate
reproducing the observed level, and a protein made fast and destroyed fast is indistinguishable
at steady state from a slow one.

A time course breaks the degeneracy completely. If two production–degradation models generate
the same trajectory from the same starting abundance, and that abundance is not already the
steady state, the models are proved equal — both rates, not just their ratio. The handle is the
relaxation half-time `log 2 / d`, which is set by degradation alone, and a degron of strength
`k` is proved to divide the steady state by exactly `k`.

The design consequence, `turnover_design_law`: a model of a disordered proteome fitted to
abundances is fitting one number per protein, and can neither predict nor be tested against
either rate on its own. A pulse–chase is not a refinement of the abundance measurement — it is
the measurement that makes the model identifiable.

## Part CXLVII — Temperature: why an enthalpic model can never produce an LCST

Part LXXV fixed the exact critical coupling `chiC N = (1+√N)²/(2N)` of a disordered chain of
`N` segments, but left open where `chi` comes from. In a real solution it comes from
temperature, and how it depends on temperature decides the entire topology of the phase
diagram. `RequestProject/SolventQuality.lean` writes `chi(T) = A + B/T + C·T` — an offset, an
enthalpic term, and the entropic (hydrophobic) term — and proves the consequences against the
honest demixing predicate of Part LXXV, the existence of a genuine two-phase splitting of the
Flory–Huggins density.

**A purely enthalpic model has no lower critical solution temperature.** With `C = 0` and
`B ≥ 0`, if the solution is homogeneous at one temperature it is homogeneous at every higher
temperature, and if it demixes at one temperature it demixes at every lower one. Demixing on
heating — the behaviour of elastin-like regions and of many condensate-forming disordered
regions — is not a matter of fitting `A` and `B` badly; it is outside the model's reach.

With a positive entropic coefficient the LCST appears: the solution is proved to demix at every
sufficiently high temperature. And the two mechanisms together give both critical temperatures
at once: with `A = 0`, `B = 10`, `C = 1/10` a solution of unit chains is proved to demix at
`T = 1`, to be homogeneous at `T = 10`, and to demix again at `T = 100`.

The identifiability half is the usual one, and it is sharp here. A phase boundary measured at
one temperature leaves the three contributions unseparated — every pair of enthalpic and
entropic coefficients admits an offset that fits. Three distinct temperatures determine all
three uniquely; the proof is that a quadratic with three distinct roots vanishes.

## Part CXLVIII — Proofreading: how a low-affinity motif can be read accurately

Recognition by disordered regions is recognition by short linear motifs, and short linear
motifs bind weakly: the equilibrium discrimination between a cognate and a near-cognate
partner is one ratio of dissociation constants, often no better than tenfold. Signalling
discriminates far better. `RequestProject/Proofreading.lean` proves both halves of Hopfield's
resolution.

The equilibrium half is a no-go theorem, and it is the same lesson as the chaperone theorem of
Part CXLIV in a different arena: if the intermediate steps are ordinary equilibria and the
intermediates themselves do not distinguish the two partners, the overall population ratio is
the *first* equilibrium constant ratio, however many steps are inserted. Adding conformational
states to an equilibrium model of motif recognition cannot add specificity.

The kinetic half is a gain theorem. In a driven cascade of `n` stages, at each of which the
complex either advances or falls apart, the discrimination is `((kf+koffW)/(kf+koffR))^n` — the
equilibrium factor raised to the number of stages — strictly increasing in `n` and unbounded.

The price is exact. The yield of the *correct* partner falls geometrically to zero, and the
exchange rate between the two is a constant of the chemistry: the ratio of log accuracy to log
yield loss is proved to be the same for every `n`. No cascade design escapes it, and a model
that reports an accuracy without reporting the accompanying loss of flux has reported half the
answer.

## Part CXLIX — Facilitated diffusion: what a disordered tail buys in a target search

Many DNA-binding proteins find their site faster than three-dimensional diffusion allows by
alternating three-dimensional hops with one-dimensional sliding, and the sliding is very often
mediated by a positively charged disordered tail keeping a weak nonspecific grip on the
backbone. `RequestProject/FacilitatedSearch.lean` says exactly what a model of such a tail is
claiming.

With `tau1` the mean sliding time, `tau3` the mean hop time, `D1` the sliding diffusion
coefficient and `L` the number of sites, the mean search time `L(tau1+tau3)/(2√(D1 tau1))` is
proved to be bounded below by `L√(tau3/D1)`, with the bound attained exactly at `tau1 = tau3`
and strictly missed at every other partition. **The optimal search spends equal time sliding
and hopping** — a statement about the tail's grip, not about the folded domain.

The gain is proved to be exactly the sliding length: at the optimum the search beats a pure
three-dimensional search by the factor `√(D1 tau3)`, the number of sites scanned per excursion.

And the caveat is proved too: the optimal time depends on `D1` and `tau3` only through their
ratio, so multiplying both by the same factor leaves every measured association rate unchanged.
A rate measurement alone cannot say how fast the tail slides — only how far it slides per hop.

## Part CL — Feasibility geometry: which distance panels an ensemble can produce at all

Everything measured about a disordered region that has spatial content is a pairwise distance
averaged over the ensemble: a PRE, an NOE, a FRET efficiency, a crosslink. Before asking whether
a model reproduces such a panel, one can ask which panels are the mean-distance panel of *some*
ensemble. `RequestProject/DistanceRealizability.lean`, `RequestProject/DistanceCutEnsembles.lean`,
`RequestProject/ContactBudget.lean` and `RequestProject/ContactPacking.lean` answer that, and
`RequestProject/DistancePanelVerdict.lean` collects the answer into one law.

Two constraints survive averaging, and both hold for the average itself, so both are directly
comparable with data: the triangle inequality (`meanDist_triangle`) and the contour bound
(`meanDist_le_chain`, `⟨d(i,j)⟩ ≤ b·|i−j|` for a chain of bond length at most `b`). With error
bars they become model-free refutations: a measured triangle defect above `3e` is compatible with
**no** ensemble in any metric space (`no_ensemble_of_triangle_defect`), so an observed defect `D`
becomes a falsification exactly when the error bar is pushed below `D/3`
(`triangle_precision_design_rule`).

Passing those tests buys much less than it appears to. `cutEnsemble_meanDist` shows that
ensembles of legitimate chains realise the entire cut cone up to the contour scale, and the
smallest witness is a two-state exchange: four labelled sites, half the time with the terminal
pairs collapsed and half the time with the central pair collapsed, has mean panel `1` around the
cycle and `2` across both diagonals. That panel is a metric, respects the contour bound — and is
the distance panel of no four points of any Euclidean space, because sites `1` and `3` would both
have to be the midpoint of `0` and `2` (`no_single_structure_of_fourCycle`). The failure is not a
knife-edge: every configuration misses some entry by at least `1/5`
(`single_structure_fit_error_lower_bound`). **Fitting a single structure to averaged distance
restraints is not an approximation to the ensemble; the target of the fit need not exist.**

Finally, populations are budgeted by excluded volume. A short measured mean distance forces
population (`contactPop_ge`, a Markov bound: a fraction at least `1 − M/D` of the ensemble has the
pair inside the contact radius `D`), and populations compete: if no conformation realises more
than `N` of the reported contacts, they sum to at most `N` (`sum_contactPop_le_multiplicity`).
The multiplicity `N` is supplied by physics — `packing_card_le`, a grid-pigeonhole hard-sphere
packing bound, gives at most `(2⌈2D/σ⌉+1)³` partners inside a contact radius `D` at hard-core
separation `σ` — so every ensemble of chains satisfies `∑ₖ (1 − Mₖ/D) ≤ (2⌈2D/σ⌉+1)³`
(`hard_core_contact_budget`), and a panel demanding more is refuted outright
(`hard_core_panel_falsified`). The same counting fixes the size of the model: `r` mutually
exclusive populated contacts require at least `r` distinct conformations (`min_ensemble_size`).
The prose account is `FEASIBILITY_CONTEXT.md`.

The strength of the triangle test is then settled exactly at the smallest interesting size.
`RequestProject/DistanceThreePoint.lean` realises any three measured mean distances obeying the
triangle inequality by an explicit three-state exchange, each state collapsing two of the three
labelled sites (`threePointEnsemble_meanDist`), so a three-site panel is an ensemble average if
and only if it satisfies the triangle inequality (`three_point_feasible_iff`). At three sites the
geometric test is therefore complete — refuting a model there needs distributional data, not more
geometry — and the "structure" three distances appear to determine is a three-state exchange, an
illusion that breaks at four sites.

## Part CLI — Topology: the clause of a model that reweighting cannot supply

Every clause so far has been metric: distances, radii, contacts, populations, histograms. A
disordered region in a cell also carries constraints of a different kind. It threads through
pores and rings, wraps partner helices and nucleic acid, and entangles with its neighbours in a
condensate — and how much it is wrapped cannot be changed by any motion that keeps the chain
intact and keeps it out of the object it is wrapped around. `RequestProject/Winding.lean` proves
this exactly, in the plane transverse to a straight axis. The chord bound
`2R|sin(θ/2)| ≤ b` (`two_mul_abs_sin_half_le`) says a bond of length `b` taken at distance at
least `R` from the axis sweeps at most `πb/(2R)` radians (`abs_turn_le`), so a chain of `n` bonds
obeys the **wrapping budget** `|winding| ≤ n b/(4R)` (`abs_winding_le`) — and, with no excluded
volume at all, still `|winding| ≤ n/2` (`abs_winding_le_half`). Read backwards this is a
demand on the model: `k` turns of observed threading require at least `4kR/b` residues of
disordered chain (`length_demand_of_winding`). For a closed loop the winding number is an integer
(`winding_int_of_closed`), and along any continuous deformation that keeps residues outside `R`
and bonds below `2R` it never changes at all (`winding_invariant_of_deformation`): **topological
protection**, proved from continuity of the total turn plus the intermediate value theorem. Its
contrapositive is the statement a simulation must answer for — a trajectory that changes topology
has passed a chain through an object, or through itself (`unthreading_requires_violation`).

`RequestProject/Threading.lean` carries this to ensembles and data. The budget survives averaging
and, through a Markov step, caps a *population*: no admissible ensemble puts more than
`n b/(4Rk)` of its weight in the sector wound `k` times or more
(`threadedFraction_le_budget`), so a reported threaded fraction above that ceiling is refuted by
no ensemble whatsoever (`no_ensemble_of_excess_threading`), and any nonzero threaded population
forces `n ≥ 4kR/b` residues (`length_demand_of_threaded_fraction`). The sharpest modelling
statement is `reweighting_cannot_create_threading`: if the conformations a model samples are all
unthreaded, then for **every** weight vector its threaded population is exactly zero and its
discrepancy against a measured population `phi` is exactly `phi` (`unthreaded_model_gap`).
Maximum-entropy correction, Bayesian ensemble refinement and force-field rescaling all change
weights, never support; and by protection, admissible dynamics cannot move the support either
(`sector_population_invariant`). A model has to be *built* in the right topological sector — a
statement about how the ensemble is generated, not about how it is scored. The design law is not
merely a bound: an explicit regular wrap of `n` equal bonds on a circle of radius `R` realises
exactly `k` turns whenever `n ≥ 2πkR/b` (`wrap_admissible`), bracketing the true residue demand
between `4kR/b` and `2πkR/b` — tight to within a factor `π/2` (`wrap_threshold_window`). And the
familiar warning recurs: any mean threading level between `0` and `k` is reproduced by a
two-state mixture of one wrapped and one unwrapped conformation (`meanWinding_realises`), so a
measured mean is a population, not a structure. The capstone is `entanglement_design_law`; the
prose account is `ENTANGLEMENT_CONTEXT.md`.

Two extensions complete the axis. First, none of it is a planar caricature: `Winding3D.lean`
transfers the budget and the protection to a chain in `ℝ³` wrapping a straight axis
(`abs_winding3_le`, `length_demand3`, `winding3_invariant_of_deformation`), the point being that
projection onto the transverse plane cannot lengthen a bond and that `‖transverse u‖` is exactly
the distance from a residue to the axis (`dist_axis_attained`). Second, and more useful,
`ThreadingExtension.lean` makes the topological state visible to ordinary metric experiments. A
bond that turns the chain at the axis spends transverse length, and transverse length is
subtracted in quadrature from the reach available along the axis; summing with Cauchy–Schwarz
gives the trade-off `|z(n) − z(0)| ≤ n b − 8R²w²/(b n)` (`axial_extension_le`) — the contour limit
less the reach forfeited to wrapping. Read backwards, an observed extension `D` caps the winding
at `w² ≤ b n (n b − D)/(8R²)` (`winding_bound_of_extension`), and a chain seen beyond
`n b − 8R²/(b n)` is not wrapped even once (`unwrapped_of_large_extension`). Threading, which
looked like an invariant no experiment reports, is bounded by a FRET or force-extension number.

At ensemble level the trade-off is sharper than the kinematic ceiling: a measured mean extension
`D` caps the mean squared winding at `b n (n b − D)/(8R²)` (`meanSqWinding_le_of_extension`), and
Chebyshev converts that into a threaded-population ceiling `b n (n b − D)/(8R²k²)`
(`threadedFraction_le_of_extension`). For a well-extended disordered region, an ordinary FRET or
force-extension mean therefore bounds how entangled the ensemble can be, with no topological assay
at all.

## Part CM. Building the model in pieces: the price of modularity

No one builds a model of a long disordered region in one piece. The region is cut into
fragments, each fragment is sampled or measured separately, and the pieces are joined through
the stretch of chain they share. `RequestProject/ModularGluing.lean` and its companions ask what
that operation costs.

Cut the chain into (segment, seam, segment), so a conformation is a triple `(x, y, z)` and a
fragment experiment sees only `(x, y)` or `(y, z)`. The modular model is
`glue p (x,y,z) = p(x,y)·p(y,z)/p(y)`. Three facts fix its status. It reproduces **both**
fragment panels exactly (`glue_margXY`, `glue_margYZ`), so no observable confined to one fragment
can ever show it to be wrong — fragment-level agreement is a tautology, not evidence. It is
conditionally independent across the seam, and it is exactly right precisely for ensembles that
already are (`condIndep_glue`, `glue_eq_self_iff_condIndep`). And its error has a closed form:
`klG_glue_eq_cmi` — the relative entropy of the truth from its glued model **equals** the
conditional mutual information across the seam, which is nonnegative and vanishes only in the
conditionally independent case (`cmi_nonneg`, `cmi_eq_zero_iff_condIndep`). Through Pinsker this
becomes the design rule `seam_design_rule`: cut where the information across the cut is below
`eps²/2` and the modular model is within `eps` in population.

The cost is not an artefact of that particular joining rule.
`no_modular_model_beats_seam_information` shows that every strictly positive ensemble which is
conditionally independent across the seam — every ensemble any modular pipeline can produce,
however fitted — is at relative entropy at least `cmi p` from the truth, a floor the glued model
attains. The proof is self-contained: the log-sum inequality (`log_sum_inequality`) and the
resulting data-processing bound (`klXY_ge_klY`).

On the safe side, `l1_le_of_conditional_defect` gives a direct bound with no entropy in between:
segments decoupled to within `delta` given the seam put the glued model within `delta·|X|·|Z|` in
population. On the unsafe side, a minimal and physically standard counterexample: two termini
held together by a long-range contact, with the seam in a single state (`longRange`). Each
fragment panel is exactly that of an uncorrelated ensemble, so no fragment measurement can detect
the coupling; the glued model is the uniform product, at the maximal population distance `1`
from the truth (`longRange_l1`); it reports the end-to-end contact probability as `1/2` when the
truth is `1` (`longRange_contact`); and its seam information is `log 2`, so every modular model
whatsoever is at least one bit away (`longRange_no_modular_model_is_close`).

Finally, the way such models are usually checked is biased in their favour.
`cmi_coarse_le` — coarse-graining a segment can only lower the apparent seam information, so a
coarse validation returns a lower bound on the modular error and never an upper bound — and
`coarse_validation_can_hide_everything`: a blind readout of one terminus makes the price of the
long-range contact vanish entirely while the truth is a full bit. The capstone is
`modular_design_law` with its non-vacuity witness `modular_design_law_is_not_vacuous`; the prose
account is `MODULAR_CONTEXT.md`.

`fragment_panels_never_falsify_modularity` states the epistemic consequence: for any truth
whatsoever the glued ensemble matches both fragment panels and has seam information zero, so the
fragment data are always consistent with a perfectly modular truth. The modular assumption is
testable only by an observable spanning the cut.

## Part XCIV — the screen that is not a product, and where its price is paid

Part XCI controls the false discovery rate of a proteome-scale screen assuming the candidates are
independent; Part XCIII removes that assumption but only for screens that report e-values. A real
disorder screen reports p-values and couples its candidates through shared reagents, a shared
calibration run and batch effects, and the statement covering that case — the harmonic correction
— was recorded as unproved. It is now proved.

`IDR.DepBH.benjamini_yekutieli_level`: under an *arbitrary* joint law, with validity of the null
p-values as the only assumption, the Benjamini–Hochberg list run at the deflated level `α/H_m`
has expected false discovery proportion at most `α`. The general form
`IDR.DepBH.selfConsistent_fdr_le_harmonic` covers every step-up rule at once, through the
telescoping identity `IDR.DepBH.wgt_decomp`: the weight `1/|R|` of a discovery is a combination of
the *nested* events `{i ∈ R, |R| ≤ j}`, and self-consistency contains each of them in the
single-candidate event `{p_i ≤ j·q/m}`, so nothing about the joint law is ever needed. The price
is logarithmic — `log(m+1) ≤ H_m ≤ 1 + log m` — against Bonferroni's factor `m`, and the corrected
list still contains the Bonferroni list (`IDR.DepBH.by_dominates_bonferroni`).

The factor is not an artefact of the proof. `IDR.DepBHSharp.bh_fdr_eq_harmonic` constructs, for
every `m` and every level, a joint law in which all `m` hypotheses are null, all p-values are
valid, and uncorrected BH has false discovery rate *exactly* `q·H_m`; from two candidates on this
exceeds the nominal level (`IDR.DepBHSharp.uncorrected_bh_exceeds_level`), and the corrected
procedure sits exactly at its own bound on the same instance
(`IDR.DepBHSharp.by_level_is_attained`). Under arbitrary dependence the deflation is necessity,
not conservatism.

Since the factor depends only on how many hypotheses the screen states, it is a design parameter.
Disorder is a property of a region, so a screen may test regions instead of residues.
`IDR.RegionScreen.fdp_lift`: reading a region-level report at residue level multiplies the number
of false discoveries and the length of the list by the same region length, so the false discovery
proportion is identical. Hence `IDR.RegionScreen.region_screen_fdr_le` — the region-level
procedure at level `α/H_M` controls the residue-level rate at `α` — while
`IDR.RegionScreen.harmonic_gain_log` shows the saving is `H_n − H_M ≥ log b − 1`: aggregating
residues into regions of length `b` buys back a `log b` of the correction and loses nothing.
`DEPENDENT_BH_CONTEXT.md` gives the prose account.
