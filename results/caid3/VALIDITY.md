# Does the extra resolution carry signal? The objection, tested

## The objection

The pairwise protocol certifies 72 orderings where the residue-level protocol
certifies 8. A referee's first move is: *more resolution is not automatically
more signal. A protocol could resolve more pairs and resolve them wrongly.*

That is testable. **57 methods entered both CAID2 and CAID3.** Rank them on
CAID3 under each protocol, and ask which CAID3 ranking reproduces their CAID2
ranking — a different round, different targets, one protein shared. If the
pairwise ordering were noise, it would not transfer.

## The result

Spearman between the CAID3 ranking and the held-out CAID2 ranking, over methods
scorable in both rounds:

| benchmark | methods | **pairwise → pairwise** | **pooled → pooled** | pairwise → pooled | pooled → pairwise |
|---|---:|---:|---:|---:|---:|
| Disorder-PDB | 24 | **0.997** | 0.976 | 0.932 | 0.962 |
| Disorder-NOX | 24 | 0.927 | 0.950 | 0.864 | 0.961 |
| Binding | 29 | 0.828 | 0.906 | 0.754 | 0.721 |
| Linker | 47 | 0.843 | 0.822 | 0.787 | 0.774 |

**The pairwise ranking transfers across rounds as well as the pooled one.**
Self-consistency is 0.83–0.997 for pairwise against 0.82–0.98 for pooled —
pairwise higher on two benchmarks, lower on two, and highest of any figure in
the table at 0.997 on the largest.

On Disorder-PDB the pairwise ranking predicts the held-out pairwise ordering
significantly better than the pooled ranking does (+0.036, 95% CI
[+0.010, +0.122], p = 0.0062). On the other three the difference is not
significant, and on Disorder-NOX it is nominally negative.

## What this establishes, and what it does not

**It answers the objection.** The extra resolution is not noise: a ranking that
reproduces itself at ρ = 0.997 across two independent rounds and 24 methods is
measuring something stable. The pairwise protocol certifies nine times as many
orderings **at no cost in reproducibility.**

**It does not establish that the pairwise ordering is *more* valid**, and this
document does not claim that. Three of four benchmarks show no significant
difference, and one favours the pooled protocol. The case for the protocol rests
on capacity — what the labels can support — not on a validity advantage that the
data do not show.

The two claims are independent and both are needed:

| | source | status |
|---|---|---|
| the pairwise protocol certifies 72 orderings, the residue one 8 | `card_le_benchCapacity` + measured `ε_pair` | **proved and measured** |
| the extra orderings are reproducible on a held-out round | this analysis, 57 shared entrants | **measured** |
| the extra orderings are *more* valid than the pooled ones | — | **not shown** |

That third line is stated rather than quietly omitted, because a protocol
proposal that claims more than it demonstrates is the failure mode this whole
project has been built to avoid.

## Why the honest version is still the strong version

The argument does not need a validity advantage. It is:

1. At the measured annotation error rate, the residue-level protocol can order
   **8** of 117 entrants, and `unresolvable_pair` makes the rest a hard
   impossibility rather than a limitation of any analysis.
2. The pairwise protocol can order **72**, because discordance is a second-order
   event — `discordant = 2·d·u` exactly, so `ε_pair ≤ 2·ε_label²`.
3. Its rankings reproduce across rounds **as well as** the ones the challenge
   reports.

A protocol that resolves nine times as much of the field, with the same
cross-round stability, from the same data, is worth adopting whether or not it
also happens to be more accurate.
