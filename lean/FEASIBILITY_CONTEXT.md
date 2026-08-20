# Feasibility geometry: which distance panels an ensemble can produce at all

*Companion prose for `RequestProject/DistanceRealizability.lean`,
`RequestProject/DistanceCutEnsembles.lean`, `RequestProject/ContactBudget.lean`,
`RequestProject/ContactPacking.lean` and the capstone
`RequestProject/DistancePanelVerdict.lean`.*

## 1. The question

Almost everything measured about a disordered region that has spatial content is a *pairwise
distance*, averaged over the ensemble: a PRE profile, an NOE, a FRET efficiency, a crosslink
yield. A model is fitted to a panel of such numbers, and the panel is the first thing the model
is checked against.

Before asking whether a particular model reproduces a panel, one can ask a prior question that
needs no model at all:

> Which panels of numbers are the mean-distance panel of *some* ensemble of chains?

The answer has three parts: constraints that always hold (so a panel violating them is
self-refuting), a demonstration that those constraints are very far from sufficient (so passing
them buys much less than it appears to), and a budget that excluded volume imposes on how much
contact population a panel may demand.

## 2. What survives averaging

Averaging destroys most structural constraints. Two survive, and both are directly comparable
with data because they hold for the average itself, not for the individual conformations.

* **The triangle inequality.** `meanDist_triangle`: for any weights and any conformations,
  `⟨d(i,k)⟩ ≤ ⟨d(i,j)⟩ + ⟨d(j,k)⟩`. The proof is one line of mathematics — average the triangle
  inequality — but the consequence is not vacuous: a measured panel is a metric or it is wrong.
* **The contour bound.** `meanDist_le_chain`: if consecutive residues are at most `b` apart in
  every conformation, then `⟨d(i,j)⟩ ≤ b·|i−j|`, again for the average.

With error bars these become tests with thresholds. `no_ensemble_of_triangle_defect`: if the
measured numbers satisfy `M(i,k) > M(i,j) + M(j,k) + 3e`, then **no** ensemble, in any metric
space, reproduces the three numbers within `±e`. `triangle_precision_design_rule` reads the same
statement as an instrument specification: an observed defect `D` becomes a refutation exactly
when the error bar is pushed below `D/3`. `panel_contour_defect_falsifies` does the same job for
the contour bound. Neither test mentions a model; a panel can be refuted before any modelling
begins.

## 3. Passing those tests buys almost nothing

It is tempting to read "the averaged panel is a metric" as "the averaged panel describes a
structure". The standard practice of fitting a single structure to averaged distance restraints
presupposes exactly that. It is false, and quantifiably so.

`cutEnsemble_meanDist` gives the general construction. Take any family of two-block splits of
the residues, any nonnegative weights summing to at most one, and any scale `t`: there is an
ensemble of legitimate chain conformations — every bond at most `t` — whose mean-distance panel
is exactly the weighted sum of the corresponding cut pseudometrics. Ensembles therefore realise
the entire cut cone, truncated at the contour scale. Most of that cone consists of panels that
are not distances between points of space at all.

The smallest witness is a two-state exchange of the kind a disordered region actually shows.
Four labelled sites; in one state the N-terminal pair is collapsed and the C-terminal pair is
collapsed; in the other the central pair is collapsed. Half and half, the mean panel is

```
d(0,1) = d(1,2) = d(2,3) = d(0,3) = 1,     d(0,2) = d(1,3) = 2
```

— the four-cycle metric. It obeys the triangle inequality, it obeys the contour bound, it is a
perfectly respectable metric, and `no_single_structure_of_fourCycle` shows that **no four points
of any Euclidean space have those distances**: sites `1` and `3` would both have to be the exact
midpoint of `0` and `2`, so they would coincide, yet they are two units apart. The proof is the
equality case of the triangle inequality, obtained from the parallelogram law.

The failure is not a knife-edge. `single_structure_fit_error_lower_bound`: *every* configuration
of four points misses some entry of that panel by at least `1/5` — a fifth of the bond scale,
uniformly over all structures, however the fit is performed. Fitting one structure to averaged
distances is not an approximation to the ensemble; the target of the fit need not exist.

The design consequence is the same one the transport part of this project reaches from the
statistical side, now from geometry: a model of a disordered region must *denote an ensemble*
and predict *averages of the measured observable*.

## 4. What excluded volume costs

A short measured mean distance is not evidence of a rare close encounter: it forces population.
`contactPop_ge` is the Markov bound — if the measured mean distance of a labelled pair is `M`,
then a fraction at least `1 − M/D` of the ensemble has the pair inside the contact radius `D`.

Populations compete. `sum_contactPop_le_multiplicity`: if no single conformation can realise more
than `N` of the reported contacts, the reported populations sum to at most `N`. Combining the
two gives the **contact budget** `∑ₖ (1 − Mₖ/D) ≤ N` (`contact_budget`), and hence a
falsification (`contact_panel_falsified`) of any panel that demands more.

Where does `N` come from? From the one piece of physics no chain escapes. `packing_card_le` is a
hard-sphere packing bound in three-dimensional space, proved by a grid pigeonhole: at most
`(2⌈2D/σ⌉+1)³` points that are pairwise at least `σ` apart fit inside a ball of radius `D`.
Hence `contact_multiplicity_le` — a residue has at most that many partners inside its contact
radius in any conformation — and `hard_core_contact_budget`:

```
∑ₖ (1 − Mₖ/D) ≤ (2⌈2D/σ⌉ + 1)³
```

for any ensemble of chains with hard-core separation `σ`, with `hard_core_panel_falsified` the
corresponding refutation. Excluded volume alone, with no force field and no sampling, converts a
contact panel into an inequality the data may already violate.

The same counting has a positive reading. `min_ensemble_size`: if `r` of the reported contacts
are mutually exclusive and each is forced to be populated, then any ensemble reproducing the
panel contains at least `r` distinct conformations. How many structures a model needs is set by
the data, not chosen for convenience.

## 6. How strong is the triangle test? Exactly right at three sites

`RequestProject/DistanceThreePoint.lean` settles the strength of the necessary condition at the
smallest interesting size. Given three measured mean distances `p = ⟨d(0,1)⟩`, `q = ⟨d(0,2)⟩`,
`r = ⟨d(1,2)⟩` obeying the three triangle inequalities, the three-state exchange with equal
weights and cut amplitudes `3(p+q−r)/2`, `3(p+r−q)/2`, `3(q+r−p)/2` — each state a collapse in
which one labelled site is displaced and the other two coincide — reproduces the panel exactly
(`threePointEnsemble_meanDist`). Hence `three_point_feasible_iff`: a three-site panel is an
ensemble average **if and only if** it satisfies the triangle inequality, which by itself already
forces the three numbers to be nonnegative.

Two consequences. The triangle test is complete at three sites, so a three-site panel that passes
it cannot be refuted by any further geometric argument: refutation there requires distributional
data, which is what the transport part of this project supplies. And the "structure" that three
distances appear to determine is an illusion of small numbers — the realising object is a
three-state exchange, and by §3 the illusion breaks at four sites.

## 7. What is not claimed

No real data have been touched. The four-cycle panel is a stipulated illustration of a two-state
exchange, not a measurement, and no claim is made about any particular protein. The packing cap
`(2⌈2D/σ⌉+1)³` is a rigorous bound, not the optimal sphere-packing constant; a sharper geometric
input would sharpen the budget and every downstream test with it. The results are stated for
finite ensembles of finitely many labelled sites, which is what a panel of measurements reports.
