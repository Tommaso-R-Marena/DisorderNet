# Structural error in ångström: a transport geometry for ensembles of a disordered region

This round adds one coherent development to the project: a *metric geometry of structural
error* for ensemble models of intrinsically disordered regions, and the experimental
certificates that geometry supports. Six new self-contained files, every theorem proved
with no `sorry` and only the standard axioms (`propext`, `Classical.choice`, `Quot.sound`);
the whole project builds.

The starting point is the observation already recorded in `RequestProject/Transport.lean`:
the population-space `ℓ¹` loss is blind to structural similarity, so the objective a
disorder model is judged by must be a transport (earth-mover) cost against a structural
metric. That file defined the cost. It did not establish that the cost *behaves* like a
distance, what it controls, or what an experiment proves about it. Those are the questions
answered here.

## 1. The cost is a genuine distance (`RequestProject/TransportGeometry.lean`)

* `exists_optimal_coupling` — the infimum over transport plans is **attained**: the set of
  plans is a compact polytope and the cost is continuous on it, so there is a cheapest
  explicit matching of the model's structures to the true ones. Everything downstream uses
  the optimal plan rather than an approximating sequence.
* `transportCost_self`, `transportCost_comm` — vanishing on the diagonal, and symmetry.
* `glue`, `planCost_glue_le`, `transportCost_triangle` — the **gluing lemma**: two plans
  compose into a plan for the composite move (mass arriving at an intermediate conformation
  is split in proportion to where it must go next), so structural errors are subadditive
  along a chain of ensembles. This is what makes "the model is 1.5 Å from the truth"
  a statement one may chain and combine.
* `matchPlan`, `transportCost_eq_zero_iff_same` — the cost vanishes **exactly** on
  observationally identical ensembles, so it is a metric on ensembles modulo experiment,
  not merely a pseudometric. `transportCost_pseudoMetric` packages the axioms.

## 2. What the distance certifies (`RequestProject/TransportDuality.lean`)

* `expect_diff_le_of_lipschitz` — **accuracy transfer** (the easy half of Kantorovich
  duality): an observable that is `L`-Lipschitz against the structural metric is predicted
  to `L·ε` by any ensemble within `ε`. One geometric error budget controls every Lipschitz
  observable simultaneously.
* `transportCost_ge_of_observable` — the same statement read backwards is a **falsification
  certificate**: a measured discrepancy `Δ` in the average of an `L`-Lipschitz observable
  proves the candidate ensemble is at least `Δ/L` away from the truth, in structural units,
  with no modelling assumption of any kind.
* `transportCost_pairEns` — an exactly solvable case: for two ensembles supported on the
  same compact/expanded pair, the transport distance is exactly `|p − q|` times the
  structural separation of the two states. `certificate_is_sharp` uses it to show the
  certificate above is *attained* — the experiment leaves no slack.

## 3. Which observables are strong probes (`RequestProject/TransportRadiusGyration.lean`)

Conformations are Cartesian: `Struct m = EuclideanSpace ℝ (Fin m × Fin 3)`, three
coordinates per residue, with the fixed-frame RMSD as structural metric (`rmsd_isMetric`).

* `gyr_lipschitz` — the **radius of gyration is 1-Lipschitz in RMSD**. The proof is that
  centring is a contraction (`norm_centerVec_le`, from the fact that the mean minimises the
  sum of squared deviations). `gyr_shift` shows `Rg` is translation invariant, so the bound
  survives optimising RMSD over translations: it is about shape, not frame.
* `resDist_lipschitz` — a **single inter-residue distance is only `√(2m)`-Lipschitz**, and
  `resDist_lipschitz_sharp` exhibits two single-structure ensembles attaining that constant
  exactly. So for a 100-residue region the same ensemble accuracy is worth `√200 ≈ 14`
  times less on a FRET pair than on the global size. Local probes are intrinsically weaker
  than global ones, and this is a theorem, not a rule of thumb.
* `rg_gap_le_transportCost` and `worked_saxs_certificate` — the certificate in the form an
  experimentalist uses it: a model predicting a mean `Rg` of 22 Å against a measured 27 Å
  is at least 5 Å from the truth in transport distance.
* `fret_bound_from_transport` — and the same budget spent on the FRET observable, with the
  `√(2m)` penalty explicit.

## 4. Coarse-graining and assembly (`RequestProject/TransportProcessing.lean`)

* `transportCost_map_le` — the **data-processing inequality**: comparing ensembles through
  an `L`-Lipschitz descriptor (a contact map, a Cα trace, a disorder score) gives at most
  `L` times the structural error. No descriptor can make a model look worse than it is;
  `transportCost_ge_of_map` is the contrapositive certificate, and
  `transportCost_map_eq_of_isometry` says an isometric relabelling loses nothing.
* `blockPlan`, `transportCost_mix_le` — **joint convexity**: mixing sub-ensembles (states of
  an equilibrium, clusters of a trajectory, replicas) never amplifies structural error, so a
  per-state error budget is a valid budget for the assembled model.

## 5. The `ℓ¹` theory is the crudest special case (`RequestProject/TransportTotalVariation.lean`)

* `transportCost_unitDist_eq_half_ell1` — **strong duality in the discrete geometry**: when
  every pair of distinct conformations is counted one unit apart, the transport cost is
  exactly half the `ℓ¹` population distance, i.e. the total-variation distance. The upper
  bound is the explicit maximal-overlap plan (`tvPlan`: keep the shared population in place
  and redistribute the rest as a product), the matching lower bound is the single dual test
  function `1_{p > q}`, and the two meet.
* Consequence: the project's earlier capacity and entropy laws, all phrased in `ℓ¹`, are
  statements about transport distance in the geometry that denies any resemblance between
  distinct structures; the transport cost against RMSD is the same law computed in a
  geometry that knows which structures resemble each other.
* Of independent use: `Ens.canon` (the canonical one-component-per-conformation form) and
  `transportCost_congr_same` (the cost depends on ensembles only through what experiment
  can see).

## 6. One statement (`RequestProject/TransportVerdict.lean`)

`transport_design_law` bundles the five properties for the RMSD geometry on `m`-residue
chains: metric (with attained optimal matching) modulo experiment, jointly convex under
assembly, a one-for-one budget for `Rg` with its experimental certificate, a `√(2m)`
budget for one FRET pair with the penalty attained, and the `ℓ¹` capacity theory as the
discrete-geometry case. They are all properties of a single quantity — which is the point.

## What this changes about "how to model a disordered region"

The earlier parts of the project argue that the output must be an ensemble, bound how much
capacity that ensemble needs, and say what an experiment identifies about it. This part
supplies the missing measurement: *how far a proposed ensemble is from the truth, in
ångström*, with a distance that composes, that is not fooled by relabelling structures,
that no coarse-graining can inflate, that assembly cannot amplify, and whose value is
bounded below by numbers a real experiment returns. It also says which experiments are
worth doing for that purpose: global size observables buy the full budget, single labelled
pairs buy it divided by `√(2m)`.
