# Pre-registration 9 — optimise the part no recalibration can change

Committed **before** the run. Methodology `METHODOLOGY.md`. Floors from
`PREREGISTRATION_6.md`. Control is `mt_control`, which shares this run's
training union and validation holdout.

## The argument, from the decomposition rather than from intuition

CAID's metric splits exactly:

    AUC_pooled = w_within · AUC_within + w_between · AUC_between

and `auc_within_strictMono_invariant` identifies `AUC_within` as the part **no
per-protein strictly monotone recalibration can change** — the irreducible
skill, which calibration can neither buy nor lose.

**Nothing in the standard training recipe optimises it.** Binary cross-entropy
fits each residue's mean. `distribution_matching_loss` (PREREGISTRATION_8) fits
the protein's distribution, and is deliberately permutation-invariant within a
chain. Neither touches ordering *inside* a chain, which is exactly what
`AUC_within` is.

`AUC_within` is the pair-weighted mean of per-protein Mann–Whitney statistics,
so its smooth surrogate is the logistic loss on score differences of
positive–negative pairs drawn **inside one protein**:

    L = mean over (p, n) in the same chain of  softplus(s_n − s_p)

Verified: descending on this loss alone drives measured `AUC_within` from 0.4975
to 1.0000 on synthetic data, and it is invariant to per-protein shifts while BCE
is not — both asserted in tests, the second because if BCE were shift-invariant
too the term would be redundant.

The three losses then partition the objective the way the metric partitions:

| term | fits | metric component |
|---|---|---|
| BCE | per-residue mean | neither, directly |
| W₁ | protein-level distribution | between-protein |
| **ranking** | **within-protein ordering** | **within-protein** |

## Why this is the right lever now

The field-wide operating-cost analysis measured what protein-level calibration
is worth to each method at a guaranteed miss rate — the gap between a global
threshold and a calibration-invariant per-protein quantile. On CAID3
Disorder-PDB at risk ≤ 0.05:

| method | calibration credit |
|---|---:|
| DisorderNet-pbias | +5.8% |
| DisorderNet-windowed | +8.2% |
| PUNCH2 | +13.1% |
| PredIDR2-Seq-Art | +14.0% |

The leaders lean on calibration roughly twice as hard as we do, and on the
**calibration-invariant** cost DisorderNet already leads by ~10 points on
Disorder-NOX (72.6% against PUNCH2's 82.8%). That is the axis where our margin
is largest and where the field is weakest, and it is the axis this loss targets
directly.

## The run

`mt_rank` — `mt_control` in every respect plus `--ranking-weight 0.2`. Same five
references, same fixed validation holdout, same windowed training, wide trunk,
no chirality channel, no private trunk, no distributional term. One flag, so
the comparison is clean.

Weight 0.2 is chosen, not tuned. It is **not** swept: sweeping on CAID3 would
fit a hyperparameter on the test set, and the validation holdout exists so a
future sweep can be legitimate.

## Primary endpoints

On **Disorder-NOX**, all 204 targets, coverage 1.00, paired protein-clustered
bootstrap (10,000 resamples), Holm across these two:

- **P1** — `mt_rank` beats `mt_control` on **within-protein** AUC. This is the
  mechanism, tested directly.
- **P2** — `mt_rank` within-protein AUC ≥ 0.8564, matching Metapredict-v3,
  which leads that axis while we sit 5th at 0.8346.

## Floors

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9585 |
| Disorder-NOX | 0.8810 |
| Linker | 0.8998 |
| Binding | 0.7824 |

## Falsifiable prediction

The loss is invariant to per-protein shifts, so it **cannot** improve
between-protein calibration. If pooled AUC rises and the gain is in the
between-protein component, something other than this term produced it and the
explanation is wrong however good the number looks.

## The risk, stated in advance

`w_within` is 0.5% of the metric on Disorder-PDB. Optimising it can trade
against the other 99.5%: a model free to reorder within chains at the cost of
cross-chain comparability will lose pooled AUC while gaining exactly what this
term asks for. **That outcome is a finding, not a failure** — it would be a
direct demonstration that CAID's metric penalises improving the part of it that
calibration cannot fake, which is the sharpest possible form of this paper's
critique. The floors are what stop it being reported as a win.

## Stopping rule

Scored once under the floors above. No weight sweep on CAID3 under any outcome.
If P1 fails, the negative is reported: the within-protein axis is not reachable
by a ranking surrogate in this architecture.
