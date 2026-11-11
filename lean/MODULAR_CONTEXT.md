# The price of modularity: building a full-length ensemble out of fragment models

*Companion prose for `RequestProject/ModularGluing.lean`,
`RequestProject/ModularOptimality.lean`, `RequestProject/ModularCoupling.lean`,
`RequestProject/ModularCoarse.lean` and the capstone
`RequestProject/ModularDesignLaw.lean`.*

## 1. The question

Nobody models a 300-residue disordered region in one piece. The region is cut into fragments;
each fragment is sampled, measured, or predicted separately; and the pieces are joined through
the stretch of chain they share. Every practical pipeline for intrinsically disordered proteins
— fragment libraries, per-segment simulation, block-wise generative sampling, "sample the linker
and splice" — is an instance of this.

The question this part answers is not whether such a construction is convenient, but what it
*costs*, exactly, and where one should cut so that the cost is small.

The setting is deliberately minimal. A conformation of the chain is a triple `(x, y, z)`: the
state `x` of the first segment, the state `y` of the **seam** (the stretch both fragments
contain), the state `z` of the second segment. A measurement or simulation of the first fragment
sees only the pair `(x, y)`; one of the second fragment only `(y, z)`. Those two panels,
`margXY` and `margYZ`, are all the data a modular construction has. The construction itself is

```
glue p (x, y, z) = p(x, y) · p(y, z) / p(y)
```

— the two panels joined through the seam.

## 2. The modular model always fits the data it was built from

`glue_margXY` and `glue_margYZ`: the glued ensemble reproduces both fragment panels *exactly*.
Consequently no observable confined to one fragment — no per-segment radius of gyration, no
intra-fragment distance, no fragment secondary-structure content — can ever reveal that the
glued model is wrong. Fragment-level agreement is not evidence; it is a tautology.

`fragment_panels_never_falsify_modularity` sharpens this into an epistemic statement:
whatever the truth is, the glued ensemble reproduces both fragment panels exactly *and* has
seam information zero. So the fragment data are always consistent with a perfectly modular
truth: the conditional independence a modular pipeline assumes cannot be tested by any
measurement confined to a fragment, only by an observable that spans the cut.

`condIndep_glue`: every glued ensemble is conditionally independent across the seam, and
`glue_eq_self_iff_condIndep`: gluing is exact precisely for ensembles that already have that
property. So the modular family is exactly the conditionally independent family, and the
question becomes how far the truth is from it.

## 3. The price is the seam information

The natural quantity is `cmi p`, the conditional mutual information between the two segments
given the seam state. The central identity is `klG_glue_eq_cmi`:

> the relative entropy of the true ensemble from its glued model **equals** the conditional
> mutual information across the seam.

`cmi_nonneg` and `cmi_eq_zero_iff_condIndep` make this a genuine cost: it is nonnegative, and it
vanishes exactly when the modular construction is not merely cheap but correct.

Relative entropy is not the currency a structural biologist is paid in, so `modular_l1_le_sqrt_cmi`
converts it through Pinsker's inequality (proved from scratch elsewhere in this development, in
`RequestProject/Pinsker.lean`) into the population `ℓ¹` distance, and `seam_design_rule` states
the operational rule:

> **cut the chain where the conditional mutual information across the cut is below `eps²/2`, and
> the modular model is within `eps` in population.**

That is the design statement. It says what "a good place to cut" means, and it is checkable in
principle from a global simulation or from an ensemble one already trusts: it is a property of
the cut, not of the joining procedure.

## 4. No cleverer joining rule helps

One might hope the cost is an artefact of the particular rule `p(x,y)·p(y,z)/p(y)`. It is not.
`no_modular_model_beats_seam_information`: for **every** strictly positive ensemble `q` that is
conditionally independent across the seam — that is, for every ensemble any modular pipeline can
produce, however it was fitted, by maximum likelihood or by force field or by hand — the relative
entropy `KL(p‖q)` is at least `cmi p`. Since the glued model attains that value, gluing is
optimal and the seam information is a property of the cut alone.

The proof is self-contained: the log-sum inequality (`log_sum_inequality`), from which relative
entropy is shown to decrease under marginalisation (`klXY_ge_klY`), plus Gibbs' inequality for
the second fragment's panel.

## 5. When modularity is safe, quantitatively

`l1_le_of_conditional_defect`: if, conditionally on the seam, the two segments are decoupled to
within `delta` — the precise version of "the linker is long and the ends do not talk" — then the
glued model is within `delta·|X|·|Z|` in the population metric. No entropy and no Pinsker in
between: a direct bound in the units the model is judged in.

## 6. When it is not: a long-range contact

The counterexample is minimal and physically standard. The two termini of a disordered region
are held together by a long-range contact — electrostatic or hydrophobic — so each terminus is
either released or engaged and the two are always in the same state, while the intervening seam
sits in a single state. This is `longRange`, a stipulated illustration; no data were used.

* `longRange_margXY`, `longRange_margYZ`: each fragment panel is exactly what an uncorrelated
  ensemble gives, so **no fragment measurement can detect the coupling**;
* `longRange_glue`: the modular model is the uniform product;
* `longRange_l1`: its population error is `1`, the maximum possible between two distributions;
* `longRange_contact`: it reports the end-to-end contact probability as `1/2` when the truth
  is `1` — a factor of two on precisely the observable the experiment is about;
* `longRange_cmi`: the seam information is `log 2`, so by clause 4 every modular model whatsoever
  is at least one bit from the truth (`longRange_no_modular_model_is_close`).

## 7. Coarse validation understates the price

Modular models are usually validated on coarse readouts. `ModularCoarse.lean` shows that this is
systematically optimistic. Coarse-graining commutes with gluing (`glue_coarse`), and
`cmi_coarse_le`: the seam information seen through a coarse description of a segment is at most
the true seam information. A coarse test therefore returns a *lower* bound on the modular error,
never an upper bound.

The gap can be total: `coarse_validation_can_hide_everything` — under a blind readout of one
terminus, the long-range contact ensemble has seam information zero, so the modular model looks
exactly right, while the true seam information is a full bit.

## 8. What this adds to the design specification

Read together with the rest of the development, this part converts "build it modularly" from a
practice into a specification with a checkable condition:

1. a modular pipeline is a *conditional independence assumption* across each cut, nothing more
   and nothing less;
2. its error, in relative entropy, is exactly the conditional mutual information across the cut,
   and no fitting procedure can reduce it;
3. cuts must therefore be placed where the two sides are conditionally independent given the
   overlap — long-range contacts, electrostatic clamps and transient tertiary structure are
   exactly the features that forbid a cut, and widening the overlap until the seam screens the
   coupling is the remedy;
4. fragment-level and coarse-grained agreement are not evidence of correctness: the first is
   automatic, the second is biased in the model's favour;
5. what must be reported for a modular ensemble is the cut positions and an estimate of the
   information across each cut.

## 9. Caveats, stated plainly

* The conformational state spaces are finite. This is the standard discretisation into rotameric
  or clustered states; no continuum statement is claimed here.
* `longRange` and the numbers attached to it are a stipulated illustration, not a fit to any
  measurement.
* The optimality theorem assumes the competing modular model is strictly positive — it may not
  exclude any conformation outright. This is the same absolute-continuity condition that the
  maximum-likelihood results elsewhere in the project require, and it cannot be dropped.
* The design rule uses Pinsker's inequality, which is not tight; the bound is conservative,
  which is the direction one wants in a specification.
