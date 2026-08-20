# Fragment pipelines: the price of building a disordered region out of pieces

This note gives the context for four new Lean files, all sorry-free and using only the
standard axioms (`propext`, `Classical.choice`, `Quot.sound`):

| file | content |
|---|---|
| `RequestProject/ChainPipeline.lean` | the pipeline model of a chain of many blocks; the additive design law |
| `RequestProject/ChainPipelineLimits.lean` | optimality, the operational budget, extensivity, unfalsifiability |
| `RequestProject/SeamPhysics.lean` | which chain models are free to fragment, and what a contact costs |
| `RequestProject/SeamCoarsening.lean` | longer fragments are never worse |
| `RequestProject/SeamPlacement.lean` | where the cut goes matters at first order |
| `RequestProject/PipelineStability.lean` | imperfect fragments, and the bias–variance split |

## The setting

No one builds a model of a long disordered region in one piece.  The chain is coarse-grained
into blocks, cut into overlapping fragments, each fragment is measured or simulated, and
consecutive fragments are joined through the block they share.  Earlier work in this project
priced **one** such cut: the relative entropy of the truth from the glued two-fragment model
is the conditional mutual information across the seam.

Here the chain has arbitrarily many blocks and arbitrarily many seams.  A conformation is an
element of `Blocks A n` (a chain of `n+1` blocks, each in the finite state type `A`);
`mproj p` is the model the pipeline returns when every fragment is fitted exactly to the
truth `p`, and `seamInfoTotal p` is the sum of the conditional mutual informations at the
seams.

## What is proved

**1. The additive design law** (`klG_mproj_eq_seamInfoTotal`).  The relative entropy of the
truth from the pipeline model equals *exactly* the sum of the seam informations.  Every cut is
paid for once, at the price set by the information the two sides of that cut share once the
shared block is known.  There are no cross terms: seams neither interfere nor cancel.

**2. Nothing better exists** (`no_pipeline_beats_seamInfoTotal`).  Every strictly positive
ensemble that is Markov along the chain — every ensemble any fragment pipeline can express,
however its pieces were fitted or joined — is at relative entropy at least `seamInfoTotal p`
from the truth.  The price belongs to the choice of cuts, not to the joining algorithm.  When
the total is positive, no pipeline captures the ensemble at all
(`no_pipeline_captures_chain`): it differs from the truth somewhere.

**3. The budget** (`pipeline_l1_le_sqrt_seamInfoTotal`, `pipeline_design_rule`,
`pipeline_seam_budget`).  Through Pinsker's inequality the price becomes an operational `ℓ¹`
error: to build a full-length ensemble to accuracy `eps`, place the cuts so that the seam
informations *sum* to at most `eps²/2`; with `k` seams a uniform per-seam budget of
`eps²/(2k)` suffices.

**4. The error is extensive, and that caps the length** (`seamInfoTotal_ge_of_seamFloor`,
`pipeline_length_limit`).  If every seam carries at least `c` of information, the price grows
linearly in the number of seams, so a pipeline model accurate to `eps` in relative entropy can
span at most `eps/c` seams.  Fragment-based descriptions of a disordered region are not merely
approximate; they have a maximum length.

**5. Fragment data cannot test the assumption**
(`fragment_panels_never_falsify_pipeline`).  The pipeline model reproduces the panel of the
leading fragment exactly, marginalises to the pipeline model of the truncated chain — hence
matches every fragment panel along the chain — and carries zero seam information of its own.
There is always an exactly Markov ensemble consistent with all the fragment data, so the
assumption is testable only by an observable that spans a cut.

**6. Which chains are free** (`isPipeline_of_isLocalChain`, `seamInfoTotal_localChain`).  A
chain whose weight is a product of single-block and neighbouring-block factors — the
transfer-matrix form of nearest-neighbour models of disorder — is exactly Markov along its
blocks.  For such models a pipeline is not an approximation but exact.  Everything a pipeline
loses therefore comes from couplings reaching beyond the neighbouring block.

**7. What a contact costs** (`sticker_cmi`, `sticker_cmi_ge`, `sticker_not_condIndep`).  For
three two-state blocks with a contact of strength `t` between the two blocks a cut separates —
the minimal sticker–spacer motif — the seam information is exactly

    I = ½[(1+t)·log(1+t) + (1−t)·log(1−t)]  ≥  t²/2 .

Combined with 4, `sticker_chain_length_limit`: a chain carrying such a contact at each of `n`
seams admits no pipeline model of relative-entropy accuracy `eps` once `n > 2·eps/t²`.

**8. Longer fragments are never worse** (`merged_seam_le_sum`, `coarse_pipeline_le_fine`).
Merging two adjacent seams into one longer overlap never increases the price, so the total
seam information is monotone in how finely the chain is cut: there is no free refinement, and
the shortest fragment a pipeline uses sets the floor on its error.  The proof is variational —
an explicit two-seam model that is conditionally independent in the coarse view, priced by the
optimality theorem for a single seam — and needs no chain rule for conditional mutual
information.

**9. Cut placement matters at first order** (`cut_placement_strict`).  On one and the same
chain — the sticker motif with a spare block attached — the pipeline that cuts between the two
contacting blocks pays at least `t²/2`, while the pipeline that keeps them in one fragment and
cuts one block later pays nothing and is exact.  Same number of fragments, one block of
overlap moved: an unmodellable chain becomes exactly modellable.  This also shows the
coarsening law of point 8 is strict.

**10. Imperfect fragments, and the bias–variance split** (`mproj_l1_stability`,
`fitted_pipeline_error`).  If the fragment panels are only accurate to `δ` in `ℓ¹`, the
assembled model moves by at most `(number of fragments)·δ`: panel errors are felt once each and
never amplified along the chain.  Hence

    total error  ≤  √(2 · total seam information)  +  (number of fragments) · (panel error).

The first term is the price of cutting the chain, paid even with perfect data; the second is
the price of imperfect data.  Making the cuts denser raises the first (point 8) and shrinks the
fragments on which the second is measured, which is why an optimal fragment length exists.

## Reading the results together

For a disordered region, points 6 and 7 delimit the physics: local chain statistics are free
to fragment; contacts that span a cut are not, and cost at least the square of their strength
per seam.  Points 1, 2 and 4 then say that those costs simply add, cannot be evaded by a
cleverer joining rule, and accumulate along the chain until the pipeline is useless.  Point 8
and point 9 say the only levers available are longer overlaps and better-placed cuts, and
point 5 warns that none of this is
visible in fragment data: a pipeline always looks perfect on the panels it was built from.
