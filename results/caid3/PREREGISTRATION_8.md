# Pre-registration 8 — fit the distribution, not the average

Committed **before** the run. Methodology `METHODOLOGY.md`. Floors from
`PREREGISTRATION_6.md`. Control is `mt_control`, which shares this run's
training union and validation holdout.

## The argument

Binary cross-entropy fits each residue's **mean**. The Lean development states
what that can and cannot do (`DistributionVerdict.distributional_design_law`):

1. **averages never suffice** — for any finite panel of observables and any
   error budget, two ensembles reproduce every average exactly and remain
   further apart than the budget;
2. along a measured coordinate the transport distance **equals** the L1
   distance between the cumulative distributions
   (`transportCost_line_eq_cdfL1`) — a distribution determines the error rather
   than bounding it;
3. and it **dominates** the gap between the means (`mean_gap_le_cdfL1`), so
   nothing is given up by using it.

For two empirical distributions with the same number of atoms the 1-D transport
distance has a closed form — sort both, average the absolute differences — so
the term is one line and is exactly the theorem's quantity. Verified numerically
against a direct CDF integration to 1e-6, and the domination inequality checked
on every trial.

## Why this is the right term for this benchmark

With binary labels the loss asks the `k` highest predicted probabilities to be
1 and the rest 0, where `k` is the true count. It says nothing about **which**
residues those are.

That is precisely the split the CAID decomposition exposes, and it is not a
coincidence: BCE shapes the **within-protein** ordering, and this shapes the
**protein-level distribution** — the axis that carries 97–99.5% of the pooled
metric's pair weight, and the axis on which the certified analysis shows a
0.0218 within-protein deficit being overturned by a 0.0833 between-protein
advantage.

Every earlier attempt on that axis added *capacity* — a protein-level bias
term, a private trunk. This adds none. It changes what the existing parameters
are asked to be right about.

Computed per protein and averaged over the batch. A distribution pooled across
proteins is a different object: the one CAID already reports.

## The run

`mt_wass` — `mt_control` in every respect, plus `--distribution-weight 0.1`.
Same five references, same fixed validation holdout, same windowed training,
wide trunk, no chirality channel, no private trunk. One flag.

Weight 0.1 is chosen, not tuned: it puts the term at roughly a tenth of BCE's
scale at initialisation. It is **not** swept, because sweeping it on CAID3
would be fitting a hyperparameter on the test set, and the holdout exists
precisely so that a future sweep can be done legitimately.

## Primary endpoints

On **Disorder-NOX**, all 204 targets, coverage 1.00, paired protein-clustered
bootstrap (10,000 resamples), Holm across these two:

- **P1** — `mt_wass` beats `mt_control` on pooled Disorder-NOX.
- **P2** — `mt_wass` beats `mt_control` on **between-protein** AUC for
  Disorder-NOX. This is the mechanism, tested directly rather than inferred
  from the pooled number.

## Floors

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9585 |
| Disorder-NOX | 0.8810 |
| Linker | 0.8998 |
| Binding | 0.7824 |

## Falsifiable prediction

The term is permutation-invariant within a protein — asserted in the tests — so
it **cannot** improve the within-protein ordering. If pooled AUC rises and the
gain is in the within-protein component, something other than this term
produced it and the explanation is wrong.

## The risk, stated in advance

The loss is minimised by saturating probabilities at 0 and 1, and **ties hurt a
ranking metric**. If AUC falls while the term's own value improves, that is the
mechanism, and the honest conclusion is that the distributional objective and
the ranking metric are in tension — itself worth reporting, because it would
mean CAID's metric penalises the model for getting the distribution right.

## Stopping rule

Scored once under the floors above. No weight sweep on CAID3 under any outcome.
If P1 fails, the negative is reported and any future sweep runs against the
validation holdout, never the benchmark.
