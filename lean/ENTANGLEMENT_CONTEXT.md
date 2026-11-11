# Topology: the clause of a disorder model that reweighting cannot supply

## Why this axis

Everything this project has proved so far about a disordered region is *metric*: distances,
radii, contacts, populations, transport distances between histograms. Metric data are what
PRE, NOE, FRET, SAXS and crosslinking report, and metric constraints are what a force field
adjusts. But a disordered region in a cell is also subject to constraints of a completely
different character. An intrinsically disordered region threads through pores and rings, wraps
around partner helices, nucleic acid and filaments, and in a condensate it entangles with its
neighbours. How much a chain is wrapped around an object cannot be changed by any motion that
keeps the chain intact and keeps it out of the object. It is not a soft coordinate that a
better potential will relax into place; it is fixed, and it is fixed *by geometry*.

That has a practical consequence which is the point of this axis. Ensemble modelling of
disordered regions is overwhelmingly done by generating a pool of conformations and then
reweighting it against data — maximum entropy, Bayesian ensemble refinement, ensemble
reweighting of an MD trajectory. Reweighting changes weights, never support. If the pool was
generated in the wrong topological sector, no weight vector reaches the right one, and the
discrepancy against the data is exactly the population the data demand. And the usual escape
hatch — run longer, or with a softer potential — is worse than useless: a soft-core potential
that permits chain crossing lets the simulation equilibrate a quantity that in the real system
is frozen.

## What is proved

The setting is deliberately the simplest one in which the constraint is already sharp: the chain
projected onto the plane transverse to a straight axis (a partner helix, a pore, a filament) with
the axis at the origin, residues at `p 0, p 1, …`, an exclusion radius `R` that no residue
enters, and a maximum bond length `b`.

**`RequestProject/Winding.lean` — the geometry.**

* `turn`, `totalTurn`, `winding`: the angle a bond sweeps at the axis (the argument of the ratio
  of consecutive residue positions), its sum, and the winding number in turns.
* `two_mul_abs_sin_half_le`: the chord bound `2 R |sin(θ/2)| ≤ b` — the exact statement that a
  short bond taken far from the axis cannot wrap much. Everything else follows from it.
* `abs_turn_le`: one bond sweeps at most `π b / (2 R)` radians (chord bound plus Jordan's
  inequality).
* `abs_winding_le`: **the wrapping budget** `|winding| ≤ n b / (4 R)`.
* `abs_winding_le_half`: with no excluded volume at all, still `|winding| ≤ n / 2`: a bond cannot
  sweep half a turn about a point it misses.
* `length_demand_of_winding`: the budget read backwards — `k` turns of observed threading require
  at least `4 k R / b` residues of disordered chain.
* `winding_int_of_closed`: a closed loop winds a whole number of times (proved through
  `Real.Angle`, the argument of the telescoping product being zero).
* `winding_invariant_of_deformation`: **topological protection.** Along any continuous
  deformation that keeps every residue outside `R` and every bond below `2 R`, the winding number
  of a closed chain never changes. The proof is the honest one: the total turn is continuous
  because no bond can reach the branch cut of the argument, it is integer valued, and a
  continuous integer-valued function on an interval is constant.
* `unthreading_requires_violation`: the contrapositive, stated for simulations — a trajectory
  that changes topology has passed a chain through an object, or through itself.

**`RequestProject/Threading.lean` — ensembles and data.**

* `meanAbsWinding`, `threadedFraction`: the mean amount of wrapping, and the population of the
  threaded sector at level `k`.
* `meanAbsWinding_le_budget`, `threadedFraction_le_budget`: the budget survives averaging and,
  through a Markov step, caps the *population*: no admissible ensemble puts more than
  `n b / (4 R k)` of its weight in the sector wound `k` times or more.
* `no_ensemble_of_excess_threading`: with error bars, a reported threaded population above that
  ceiling is refuted by no ensemble at all — a model-free falsification in the same style as the
  distance-panel tests proved elsewhere in this project.
* `length_demand_of_threaded_fraction`: any nonzero threaded population forces `n ≥ 4 k R / b`
  residues. Threading data are a hard lower bound on the length of the region involved.
* `reweighting_cannot_create_threading` and `unthreaded_model_gap`: the modelling no-go. If the
  sampled conformations are unthreaded, then for *every* weight vector the threaded population is
  exactly `0` and the discrepancy against a measured `phi` is exactly `phi`.
* `sector_population_invariant`: the ensemble form of protection — sector populations are
  constant under admissible dynamics.
* `wrap`, `wrap_admissible`, `wrap_threshold_window`: an explicit regular wrap of `n` equal bonds
  on a circle of radius `R` realises exactly `k` turns whenever `n ≥ 2π k R / b`. With the budget
  this **brackets** the residue demand for `k` turns between `4 k R / b` and `2π k R / b`: the
  design law is tight to within a factor `π / 2`, not merely a bound.
* `two_state_meanWinding`, `meanWinding_realises`: and the warning this project keeps repeating —
  a measured *mean* amount of threading between `0` and `k` is reproduced exactly by a two-state
  mixture of one fully wrapped and one unwrapped conformation. The mean reports a population, not
  a structure.

**`RequestProject/Winding3D.lean` — the same geometry in space.**  A real disordered region moves
along the axis as well as around it, and nothing is lost: projecting onto the transverse plane
cannot lengthen a bond (`norm_transverse_sub_le`) and `‖transverse u‖` is exactly the distance
from the residue to the axis (`norm_transverse_le_dist_axis`, `dist_axis_attained`).  Hence
`abs_winding3_le` (the budget `n b / (4 R)` for a chain in `ℝ³` wrapping a rod, pore or
filament), `length_demand3`, and `winding3_invariant_of_deformation` (protection in space).

**`RequestProject/ThreadingExtension.lean` — threading costs extension.**  This is what makes the
topological state accessible to ordinary experiments.  A bond that turns the chain at the axis
must spend transverse length to do it, and transverse length is subtracted in quadrature from
the length available along the axis (`abs_axial_step_le`, from Pythagoras and the chord bound).
Summing, with Cauchy–Schwarz, gives `axial_extension_le`:

    |z(n) - z(0)|  ≤  n b  -  8 R² w² / (b n).

The first term is the contour limit; the second is the reach a chain forfeits by being wrapped
`w` times.  Read backwards (`winding_bound_of_extension`), an observed extension `D` caps the
winding at `w² ≤ b n (n b - D) / (8 R²)`, and a chain observed beyond `n b - 8 R² / (b n)` is not
wrapped even once (`unwrapped_of_large_extension`).  A single-molecule FRET or force-extension
measurement therefore bounds the topological sector — and a model that places a disordered region
in a threaded state predicts a shorter accessible extension, by an amount this inequality
quantifies.

At the ensemble level the same inequality is sharper than anything the kinematic budget gives.
Averaging the trade-off, a measured mean extension `D` caps the mean squared winding at
`b n (n b - D) / (8 R²)` (`meanSqWinding_le_of_extension`), and Chebyshev turns that into a
population ceiling: the fraction of the ensemble wound `k` times or more is at most
`b n (n b - D) / (8 R² k²)` (`threadedFraction_le_of_extension`).  For a well-extended disordered
region this is a strong statement obtained from purely metric data, with no topological assay.

**`RequestProject/EntanglementVerdict.lean`** bundles all of it into one statement,
`entanglement_design_law`.

## How to use it on real data

Concretely, with a bond length `b ≈ 3.8 Å` (Cα–Cα) and an exclusion radius `R ≈ 10 Å` for a
partner helix with its solvation shell, one full turn of wrapping demands at least
`4 · 1 · 10 / 3.8 ≈ 11` residues, and `2π · 10 / 3.8 ≈ 17` residues suffice. A construct with
eight disordered residues between two anchored points cannot wrap that helix once, whatever any
model says; a construct with twenty can. Threading assays that report a *population* — the
fraction of molecules topologically linked — are then read against
`threadedFraction_le_budget`: at level `k = 1` the ceiling is `n b / (4 R)`, and a measured
fraction above it is refuted before any model is proposed.

## What is not claimed

The winding number used here is that of the chain about a straight axis, computed in the
transverse plane; it is the honest invariant for threading through a pore or wrapping a rod, and
it is not a knot invariant. Genuine self-knotting of a chain, and the Gauss linking number of two
closed chains, are not formalised here. The constants are exact for what they state — the budget
`n b / (4 R)` is proved, and the wrap achieving `k` turns with `2π k R / b` bonds is exhibited —
but the true minimal residue demand is only bracketed between them, not determined. No
experimental data were touched: the numbers in the paragraph above are illustrative arithmetic
with the proved inequalities.  The extension trade-off is an inequality, not an equality: it
bounds the reach a wrapped chain can have and hence bounds the winding compatible with a measured
extension, but it does not predict the extension of a given wrapped conformation.
