# Does `AUC_within` predict usefulness better than pooled AUC? No.

The obvious prescriptive claim from this project's decomposition is that CAID
should report `AUC_within` — the calibration-invariant component — alongside or
instead of pooled AUC. This tested that claim and **it does not hold.**

## The test

For every full-coverage entrant on each reference, correlate three CAID
statistics against the **operating cost**: the fraction of each protein a method
must flag to guarantee a miss rate of 0.10. Spearman throughout, since all are
used as rankings. The difference between two correlations is bootstrapped on
the *same* method resamples, because two correlations over one method set are
dependent and comparing their intervals separately would be wrong.

## The result

Against the **calibration-invariant** (per-protein quantile) cost:

| benchmark | methods | pooled | `AUC_within` | within − pooled | p |
|---|---:|---:|---:|---:|---:|
| Disorder-PDB | 70 | 0.947 | 0.956 | +0.008 | 0.614 |
| **Disorder-NOX** | 70 | **0.907** | 0.787 | **−0.121** | **0.0072** |
| Binding | 94 | 0.629 | 0.583 | −0.045 | 0.186 |
| Binding-IDR | 104 | 0.776 | 0.799 | +0.024 | 0.375 |

**On Disorder-NOX pooled AUC predicts the calibration-invariant operating cost
significantly better than `AUC_within` does.** Elsewhere the difference is
indistinguishable from zero. The proposed replacement statistic is not better,
and the paper will not claim it is.

Why: pooled AUC and `AUC_between` are nearly the same number — 0.947 against
0.947 on Disorder-PDB, to three decimals — which is exactly what the
decomposition predicts when `w_between ≈ 1`. Whatever the operating cost
depends on, pooled AUC already summarises it about as well as anything else
available.

## What survives

**The operating cost is not a restatement of AUC on the binding tasks.** No
statistic predicts it above 0.65 on Binding, and pooled AUC predicts the
*global-threshold* cost at only 0.522 on Binding-IDR. There the cost carries
information the leaderboard does not, and reporting it is worth doing.

On the disorder tasks the cost is largely predictable from AUC (ρ ≈ 0.95). That
is the honest reading of our own first place there: it is **consistent with**
our AUC standing rather than an artefact of a metric chosen to flatter us. Had
the correlation been weak while we happened to lead, the ranking would deserve
the suspicion it would attract.

## Why this is here

It refutes a claim this project would otherwise have made. The decomposition
identifies `AUC_within` as the calibration-invariant part, and it is tempting to
go from *"this component is theoretically privileged"* to *"the field should
report it"*. The first is a theorem; the second is an empirical question, and
the answer on 70–104 methods is no.

Reported because the alternative — proposing a new summary statistic, never
testing it against the incumbent, and citing the theorem as though it settled
the matter — is how a methods paper becomes advocacy.
