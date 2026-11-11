# The pairwise protocol: capacity, and what block-correlated label noise costs it

Files: `RequestProject/PairwiseProtocol.lean` (general theory),
`RequestProject/PairwiseChain.lean` (the chain instance).
Everything below is proved in Lean, sorry-free, with the standard axioms only.

## The problem this answers

The earlier capacity results (`RequestProject/LabelNoise.lean`,
`RequestProject/BenchmarkCapacity.lean`, `RequestProject/BenchmarkCapacityLabels.lean`) price a
benchmark that scores a method **residue by residue** against an annotation: a comparison is
certified only when the margin exceeds twice the annotation error, so a benchmark whose labels
are a fraction `ν_label` wrong can place at most about `1/(2·ν_label)` methods in a certified
order, however many targets it collects and however many methods enter.

That is a statement about a *protocol*, not about nature. These files change the protocol and
recompute the ceiling.

## The protocol

Fix a grouping `grp` of the residues (a protein, a domain, a window). The **scored pairs** are
the ordered pairs `(i, j)` of distinct residues in the same group (`scoredPairs`). A labelling
`S` returns on such a pair the **verdict** "`i` is disordered and `j` is not" (`Verdict`). A
method is scored on how many of the reference verdicts it reproduces.

The annotation's own error rate in this protocol is the **pair discordance**

    ν_pair = pairDiscord T L grp / #scoredPairs,

the fraction of scored pairs on which the annotation `L` and the truth `T` return different
verdicts (`pairDiscord`).

## Theorem 1 — `pairwise_capacity`

    a certified family of methods has at most  ⌈1 / (2 ν_pair)⌉  members.

The pairwise protocol obeys the same capacity law as the residue protocol, with its own noise
rate substituted. Formally, `pairLabel T grp` is the reference answer key induced by a residue
labelling, `noise_pairLabel` identifies its noise with the pair discordance, and the bound is the
existing capacity machinery instantiated at the type of scored pairs.

So a change of protocol buys exactly what it buys in `ν_pair`, and nothing else. Which raises the
question the second theorem answers: when is `ν_pair` smaller than `ν_label`?

## Theorem 2 — `correlated_noise_reduces_pair_discordance`

Annotation errors in this domain are not independent per residue: a region is annotated
disordered or ordered as a whole, and when the annotation is wrong it is wrong about the region.
Model that as a **block structure**: a partition of the residues into blocks of `b`, a truth `T`
that is constant on blocks (`BlockConstant`), and an error set `T Δ L` that is also constant on
blocks — the annotation errs by flipping whole blocks.

The structural fact (`discordPairs_subset`) is that under those hypotheses a scored pair can be
corrupted **only if it straddles a block boundary and has an endpoint in a flipped block**. Both
endpoints of a within-block pair carry the same label under the truth and the same label under
the annotation, so its verdict is negative under both, whether or not the block was flipped. All
the exposure is at the boundary.

Counting that exposure gives the theorem. With every block carrying at most `D` boundary-crossing
scored pairs, writing the boundary mass as `D · #blocks`:

    ν_pair ≤ (boundary mass / total scored pairs) · ν_label.

The integer form, `card_mul_pairDiscord_le`, is

    #residues · (corrupted pairs) ≤ D · #blocks · (mislabelled residues),

and it is deterministic: no probability model of which blocks flip, no measurement, no appeal to
averages. The factor `ν_label` appears because a block-constant error set of size `|T Δ L|`
consists of exactly `|T Δ L| / b` flipped blocks (`card_blockConstant`), each exposing at most
its own boundary.

## The order of magnitude — `chain_pairwise_noise_bound`

The chain instance fixes the constant. Residues `0, …, n−1` in adjacent comparison groups
`{0,1}, {2,3}, …`; noise blocks of `b` consecutive residues. A block is cut only at its two ends,
so it carries at most four boundary-crossing ordered pairs (`chain_boundaryDeg_le_four`); there
are `n/b` blocks and at least `n` scored pairs (`chain_card_scoredPairs_ge`). Hence

    ν_pair ≤ (4 / b) · ν_label,

i.e. the boundary fraction is `O(1/|block|)`. Equivalently (`chain_capacity_ceiling_gain`) the
capacity ceiling of the pairwise leaderboard satisfies

    1 / (2 ν_pair) ≥ (b/4) · 1 / (2 ν_label),

so the number of methods a challenge can honestly order grows linearly in the length of the
correlated block. In the extreme where the blocks are unions of comparison groups — even block
length here — the protocol is exactly noise-free: `ν_pair = 0`
(`chain_aligned_no_discordance`), however many blocks the annotation flips.

`chain_worked_instance` checks that none of this is vacuous: six residues, blocks of three, the
annotation flipping the whole second block. Half of the residues are mislabelled
(`ν_label = 1/2`) and exactly one of the six scored pairs is corrupted (`ν_pair = 1/6`) — the one
pair cut by the block boundary.

## What is and is not claimed

* The bounds are on the *rates*, hence on the *ceilings* the capacity theorem imposes. A larger
  ceiling is permission to order more methods, not a guarantee that a particular leaderboard
  does; the sharpness results for the capacity bound itself live in
  `RequestProject/BenchmarkCapacity.lean`.
* `D` is a per-block bound on boundary mass, so `D · #blocks` is an upper bound for the total
  boundary mass, tight when the boundary mass is spread evenly over the blocks — as it is on a
  homogeneous chain.
* The block hypotheses are exactly two: the truth is constant on blocks, and the error set is a
  union of blocks. Nothing is assumed about which blocks flip, or how many.
* The mechanism is not specific to disorder, or to biology. Whenever the noise is constant on the
  cells of a partition and the statistic is a within-cell comparison, only the boundary is
  exposed, and the exposed fraction is the boundary mass of the partition.
