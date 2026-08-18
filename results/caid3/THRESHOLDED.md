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

## What is not claimed

**Not certified.** `LabelNoise.ranking_certified` would turn a margin into a
statement about which method is better *in truth*, given an estimate of the
annotation error rate. This project does not have a defensible one — see
`label_noise_certificate.py`, where the attempt to derive it from CAID3's own
references returned exactly zero disagreements because those references are not
independent annotations. A margin of 460 residues on Disorder-PDB is 0.46% of
the evaluated set, and any plausible annotation error rate is larger than that,
so the top two are near-certainly inside the noise however it is eventually
measured. The NOX margin of 2,719 (2.7%) is a better candidate for surviving
one.

**Thresholds are not comparable across methods.** Each entrant chose its own,
and a method optimising AUC rather than accuracy is penalised here through no
fault of its ranking. That cuts both ways and is why this is reported beside
the AUC tables rather than instead of them.
