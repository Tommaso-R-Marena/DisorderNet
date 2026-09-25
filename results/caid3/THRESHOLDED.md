# The other metric: thresholded error counts

CAID's AUC is a ranking statistic. The number a practitioner actually acts on
is the binary call, and every `.caid` file carries one — column four, each
method's own threshold, not a rethresholding of someone else's scores.

Counting residues where a method's own call disagrees with the reference, over
every evaluated residue of every target, for the 46 methods that submitted a
complete binary column:

**Disorder-PDB** (99,239 evaluated residues)

| # | method | errors | rate | vs #1 |
|---:|---|---:|---:|---:|
| 1 | **DisorderNet-pbias** | **8,000** | 0.081 | — |
| 2 | PUNCH2 | 8,460 | 0.085 | +460 |
| 3 | DisorderNet-windowed | 8,481 | 0.085 | +481 |
| 4 | PUNCH2-Light | 8,716 | 0.088 | +716 |
| 5 | SETH-0 | 9,977 | 0.101 | +1,977 |

**Disorder-NOX** (99,977 evaluated residues)

| # | method | errors | rate | vs #1 |
|---:|---|---:|---:|---:|
| 1 | **DisorderNet-windowed** | **18,084** | 0.181 | — |
| 2 | DisorderNet-pbias | 18,983 | 0.190 | +899 |
| 3 | flDPnn2 | 20,803 | 0.208 | +2,719 |
| 4 | flDPlr | 22,382 | 0.224 | +4,298 |
| 5 | ESpritz-D | 22,426 | 0.224 | +4,342 |

First on both, on a metric that is not the one the model was compared on
anywhere else in this project. On Disorder-NOX that is **13% fewer wrong
residues** than the best other entrant; on Disorder-PDB, 5.4% fewer than
PUNCH2.

This is worth having precisely because it is a different statistic. The AUC
result, the within-protein decomposition, the crossed-comparison count and this
all point the same way, and they are not restatements of one another: AUC is
rank-based and threshold-free, the decomposition is calibration-invariant, the
crossed count is a certified irreducible-error bound, and this is a raw
disagreement count at each method's own operating point.

## Certified against an annotation error rate, without estimating one

`LabelNoise.ranking_certified` turns a margin into a statement about which
method is better *in truth*: a margin above `2·eps·n` certifies the ranking at
annotation error rate `eps`. This project has no defensible estimate of `eps` —
the attempt to derive one from CAID3's own references returned exactly zero
disagreements, because those references are not independent annotations.

An estimate is not needed. The criterion is monotone in `eps`, so the whole
answer is a frontier, following the shape of
`RobustCertificate.certificate_under_mean_error`: a bound whose input is known
only to within `delta` degrades continuously rather than becoming silence.

| assumed annotation error | certified vs 45 others, Disorder-PDB | Disorder-NOX |
|---|---:|---:|
| 0.25% | 43 | 45 |
| 1% | 41 | 44 |
| **2%** | **39** | **43** |
| 3% | 34 | 34 |
| 5% | 21 | 9 |

**Even at 2% annotation error — an error rate few would defend as too
generous for crystallographic disorder assignment — DisorderNet's lead is
certified over 39 of 45 methods on Disorder-PDB and 43 of 45 on Disorder-NOX.**
Only the top handful sit inside plausible noise.

The **breakdown rate** of a comparison, `margin/(2n)`, is the annotation error
rate that would have to be exceeded before that ranking could be an artefact.
For the closest comparison it is **0.232%** on Disorder-PDB (PUNCH2, 460
residues) and **0.450%** on Disorder-NOX. Median across the field: 4.78% and
3.71%. So the top two on Disorder-PDB are near-certainly inseparable and most of
the field is not.

## What is not claimed

**Thresholds are not comparable across methods.** Each entrant chose its own,
and a method optimising AUC rather than accuracy is penalised here through no
fault of its ranking. That cuts both ways and is why this is reported beside
the AUC tables rather than instead of them.
