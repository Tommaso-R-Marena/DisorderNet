# Corrected results — the architecture the checkpoints were trained with

Every earlier CAID3 figure in this project was produced by an evaluator that
built the head with default dilations, (1,2,4,8), a 61-residue receptive field,
while both checkpoints were trained with `--wide-receptive-field`, (1,4,16,32),
213 residues. Dilation changes the receptive field and not a single weight
shape, so `load_state_dict(strict=True)` accepted the mismatch without a word
and then computed a different function.

The models were being evaluated at under a third of their trained context.

## Corrected standing

Both checkpoints, all five official references, every target predicted, coverage
1.00 throughout, zero exact ties, ranks recomputed independently by
`verify_ranks.py`.

**`mt_windowed`** — the model of record:

| benchmark | ours | rank / all | rank / full-coverage | leader (its coverage) |
|---|---:|---:|---:|---|
| **Disorder-PDB** | **0.9595** | **1 / 115** | **1 / 58** | PUNCH2 0.9552 (1.00) |
| **Disorder-NOX** | **0.8928** | **1 / 115** | **1 / 58** | ESMDisPred-2PDB 0.8855 (0.89) |
| **Linker** | **0.9243** | **1 / 115** | **1 / 91** | IPA-AF2-Linker 0.8985 (0.87) |
| Binding | 0.7934 | 2 / 115 | 2 / 70 | DisoFLAG-PB 0.7760 (0.98) |
| Binding-IDR | 0.5007 | 30 / 115 | 17 / 70 | bindEmbed21IDR 0.6407 (1.00) |

**`mt_full`** — the earlier checkpoint, no long proteins in training:

| benchmark | ours | rank / all | rank / full-coverage |
|---|---:|---:|---:|
| Disorder-PDB | **0.9636** | **1 / 115** | **1 / 58** |
| Linker | 0.9048 | **1 / 115** | **1 / 91** |
| Binding | 0.7775 | 3 / 115 | 3 / 70 |
| Disorder-NOX | 0.8479 | 11 / 115 | 3 / 58 |
| Binding-IDR | 0.5221 | 22 / 115 | 14 / 70 |

## The pre-registered test, properly executed

| hypothesis | Δ AUC | p | p adjusted | outcome |
|---|---:|---:|---:|---|
| **P1 — beat AlphaFold-rsa** | **+0.0097** | **0.0126** | **0.0252** | **CONFIRMED** |
| P2 — beat PUNCH2 | +0.0043 | 0.2022 | 0.2022 | not significant |

Non-inferiority: 0.9595 against a floor of 0.9553 — **PASS**.

P1 was registered as the minimum bar: AlphaFold-rsa is training-free, ranks 3rd
on Disorder-PDB, and a trained model that cannot beat it has demonstrated
nothing. It now clears, after Holm across the registered family of two.

P2 was registered as a stretch expected to fail, so that a success could not be
claimed as though planned and a failure could not be quietly dropped. It failed.
**We top the Disorder-PDB table and superiority over PUNCH2 remains unproven.**

## Why this is a correction and not a second bite

The earlier evaluation returned adjusted p = 0.0512 for P1 — a failure by
0.0012 — and the corrected one returns 0.0252. That pattern deserves suspicion,
so here is what makes it checkable:

- The bug was found while wiring an unrelated flag, and **fixed and committed
  (`cc77ee7`) before any corrected number existed**.
- It is a bug on its face: the model ran at a 61-residue receptive field having
  been trained at 213.
- The fix was applied to **both** checkpoints and **improved every benchmark in
  both** — five of five in each case, not a selective rescue.
- The analysis re-run is the identical registered one: same family of two, same
  Holm correction, same floor, same 10,000 resamples.

What changed was the measurement, not the hypothesis or the test.

## Effect of the correction

| benchmark | mt_windowed before | after | mt_full before | after |
|---|---:|---:|---:|---:|
| Disorder-PDB | 0.9590 | 0.9595 | 0.9603 | 0.9636 |
| Disorder-NOX | 0.8900 | 0.8928 | 0.8422 | 0.8479 |
| Linker | 0.9154 | 0.9243 | 0.8885 | 0.9048 |
| Binding | 0.7924 | 0.7934 | 0.7649 | 0.7775 |
| Binding-IDR | 0.4945 | 0.5007 | 0.5180 | 0.5221 |

Ten of ten improved. The wide receptive field was doing real work and the
evaluator was discarding it.

## Which checkpoint is the model of record

`mt_windowed`, on three first places against `mt_full`'s two, and because
`mt_full` sits 11th on Disorder-NOX where `mt_windowed` is 1st. `mt_full` is
better on Disorder-PDB (0.9636 against 0.9595) and Binding-IDR, so a
head-to-head between the two on Disorder-PDB is worth registering separately
rather than settling by preference.
