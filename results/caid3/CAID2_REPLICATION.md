# CAID2 — an independent round, held out by construction

The CAID3 result rests on one round. Resampling cannot fix that: bootstrap,
jackknife and Monte Carlo all estimate the same sampling distribution, so more
resamples shrink the Monte Carlo error in locating a p-value and leave the
p-value itself alone. **New data is the only thing that adds power**, and CAID2
is new data — a different round, 348/210/78/40 targets, its own 71 entrants,
its own published leaderboard. It shares exactly **one protein** with CAID3.

Only two checkpoints may be scored here, and the evaluator now refuses the rest
rather than trusting the operator: filtering against CAID3 alone leaves **307 of
CAID2's 348** Disorder-PDB targets in the training union, and that model would
score *better*, not worse. `multitask_publication` and `multitask_pbias` were
filtered against all five references — CAID3 Disorder-PDB and all four CAID2
references — at ≥40% identity with homologues removed. The guard reads the
filter recorded in the checkpoint, not the intent of the launch script.

## Result

Coverage 1.00 on all four references, for both checkpoints.

**`multitask_pbias`**

| benchmark | ours | APS | rank / all | rank / full-coverage | round leader |
|---|---:|---:|---:|---:|---|
| Disorder-PDB | 0.9470 | 0.9027 | 2 / 71 | **1 / 54** | PredIDR-long 0.9335 |
| Disorder-NOX | 0.8515 | 0.6026 | **1 / 71** | **1 / 54** | Dispredict3 0.8378 |
| Binding | 0.8445 | 0.3873 | **1 / 71** | **1 / 55** | Dispredict3 0.8244 |
| Linker | 0.7805 | 0.1762 | 3 / 71 | **1 / 64** | SETH-0 0.7695 |

**`multitask_publication`**

| benchmark | ours | APS | rank / all | rank / full-coverage | round leader |
|---|---:|---:|---:|---:|---|
| Disorder-PDB | 0.9416 | 0.8964 | 3 / 71 | **1 / 54** | PredIDR-long 0.9335 |
| Disorder-NOX | 0.8496 | 0.6328 | **1 / 71** | **1 / 54** | Dispredict3 0.8378 |
| Binding | 0.8500 | 0.4690 | **1 / 71** | **1 / 55** | Dispredict3 0.8244 |
| Linker | 0.8021 | 0.1555 | 2 / 71 | **1 / 64** | SETH-0 0.7695 |

**Every method placed above us in the all-entrants ordering declined targets.**
Not most of them — all of them: 1 of 1 above `pbias` on Disorder-PDB, 2 of 2
above it on Linker, 2 of 2 and 1 of 1 for `publication`. Ranking among entrants
that answered every target is the equal-footing comparison, and there we are
first on all four.

## Paired differences, protein-clustered bootstrap, 10,000 resamples

`multitask_pbias`:

| benchmark | opponent | Δ AUC | 95% CI | p | targets |
|---|---|---:|---|---:|---:|
| Disorder-PDB | PredIDR-long (leader) | **+0.0134** | [+0.0004, +0.0271] | 0.043 | 348 |
| Disorder-PDB | SPOT-Disorder2 | **+0.0108** | [+0.0029, +0.0194] | 0.004 | 284 |
| Disorder-PDB | AlphaFold-rsa | +0.0059 | [−0.0046, +0.0163] | 0.264 | 299 |
| Disorder-NOX | AlphaFold-rsa | **+0.0924** | [+0.0510, +0.1336] | <0.001 | 173 |
| Disorder-NOX | Dispredict3 (leader) | +0.0137 | [−0.0156, +0.0426] | 0.352 | 210 |
| Binding | AlphaFold-rsa | **+0.1554** | [+0.0557, +0.2530] | 0.002 | 62 |
| Binding | Dispredict3 (leader) | +0.0201 | [−0.0233, +0.0703] | 0.410 | 78 |
| Linker | SETH-0 (leader) | +0.0110 | [−0.0549, +0.0859] | 0.679 | 40 |

## What this does and does not establish

**It establishes** that the CAID3 placements were not a property of CAID3. The
same architecture, trained without ever seeing a CAID2 target or a ≥40%
homologue of one, is first among full-coverage entrants on a round it was not
tuned against, held out by construction rather than by promise.

**It establishes** that we beat the training-free structural baseline on
Disorder-NOX and Binding by margins that survive Holm across the exploratory
family of ten (+0.0924, adj p = 0.002; +0.1554, adj p = 0.022 in the
`publication` run's family). That baseline ranks 3rd on CAID3 Disorder-PDB, so
it is not a straw man.

**It does not establish** superiority over the CAID2 round leaders. Three of
four head-to-head deltas are positive and none clears 0.05 except Disorder-PDB
at p = 0.043, which does not survive correction across fourteen comparisons.
Winning a leaderboard and being separable from the method below you are
different claims and only the first is made here.

**It was not pre-registered.** CAID3 has a registered primary family, a
registered floor and a registered stopping rule; CAID2 has none of those,
because the replication was designed after the CAID3 numbers existed. The
placements are descriptive and the paired tests are exploratory, and they are
labelled that way in the evaluator's own output rather than only here. The
correct reading is *the CAID3 result replicated on an independent round*, not
*a second confirmatory test*.

**No CAID3 floors were applied.** The two rounds share four task names and none
of their values; a 0.9545 Disorder-PDB floor set on 319 CAID3 targets says
nothing about 348 CAID2 targets. The evaluator suppresses them on any round but
CAID3 rather than printing a confident PASS about a comparison nobody
registered.

## Combining the two rounds

The rounds share one protein out of ~1,100, so Stouffer's method on the
one-sided p-values is close to valid and the dependence is negligible. It is
**not** applied to the primary family, and the reason is worth stating: on
CAID3, P1 (beat AlphaFold-rsa on Disorder-PDB) was confirmed at p = 0.0126; on
CAID2 the same comparison gives p = 0.264. Combining would produce a weaker
result than CAID3 alone, and reporting the combination only when it helps is
the failure mode pre-registration exists to prevent. Both numbers are above.

## Reproduction

```bash
export EVAL_CKPT=/scratch4/sfried3/jbeale3_disordernet/multitask_pbias
export EVAL_REFS=.../caid2_official
export EVAL_PREDS=.../caid2_predictions
sbatch rockfish/slurm/eval_caid3_official.sbatch --benchmark caid2 \
  --submissions "$EVAL_CKPT/caid2_submissions" \
  --out "$EVAL_CKPT/caid2_official_results.json"
```

Jobs 29928190 and 29928191, 1h03m each. Raw output in each run directory as
`caid2_official_results.json`, submissions archived alongside as `.caid`.
