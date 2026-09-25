# Confirmatory result — windowed training

> **Superseded numbers below the fold.** The first evaluation of this run built
> the head with default dilations while the checkpoint was trained with
> `--wide-receptive-field`, so it ran at a 61-residue receptive field instead of
> 213. Dilation changes no weight shape, so `load_state_dict(strict=True)`
> accepted it silently. The corrected figures are in `CORRECTED.md`; everything
> below is kept as the record of what was reported and why it was wrong.


Job 29833016 (training) and 29833018 (evaluation), analysed exactly as fixed in
`PREREGISTRATION.md` before the run started, under `METHODOLOGY.md`.

## Standing

All five benchmarks, every target predicted, coverage 1.00 throughout, zero
exact ties. Ranks recomputed independently by `verify_ranks.py`.

| benchmark | ours | rank / all | rank / full-coverage | leader (its coverage) |
|---|---:|---:|---:|---|
| **Disorder-PDB** | **0.9590** | **1 / 115** | **1 / 58** | PUNCH2 0.9552 (1.00) |
| **Disorder-NOX** | **0.8900** | **1 / 115** | **1 / 58** | ESMDisPred-2PDB 0.8855 (0.89) |
| **Linker** | **0.9154** | **1 / 115** | **1 / 91** | IPA-AF2-Linker 0.8985 (0.87) |
| Binding | 0.7924 | 2 / 115 | 2 / 70 | DisoFLAG-PB 0.7760 (0.98) |
| Binding-IDR | 0.4945 | 34 / 115 | 18 / 70 | bindEmbed21IDR 0.6407 (1.00) |

First place on three of the five, and second on a fourth — where our 0.7924 is
also above the published leader's 0.7760, with only ESpritz-D (0.7972) ahead.

## The pre-registered test failed

Both primary hypotheses, Disorder-PDB, unfused, all 319 targets, Holm across
exactly these two:

| hypothesis | Δ AUC | p | p adjusted | outcome |
|---|---:|---:|---:|---|
| P1 — beat AlphaFold-rsa | +0.0092 | 0.0256 | **0.0512** | **not significant** |
| P2 — beat PUNCH2 | +0.0038 | 0.3070 | 0.3070 | not significant |

P1 misses by 0.0012. It is reported as a failure because that is what was
registered, and the whole point of registering it in advance was to remove the
option of deciding afterwards that 0.0512 is close enough.

So: **we top three tables, and the confirmatory significance test did not
clear.** Both statements are true and neither replaces the other. A rank is a
point estimate; superiority over AlphaFold-rsa on Disorder-PDB remains
unproven at the level this benchmark can resolve.

**Non-inferiority: PASS.** 0.9590 against a floor of 0.9553, so the windowed
variant is accepted. It is now the model of record.

## Secondary, exploratory, Holm within a family of 13

| comparison | p | adjusted | |
|---|---:|---:|---|
| Linker − AlphaFold-rsa | 0.0004 | 0.0048 | significant |
| Disorder-NOX − AlphaFold-rsa | 0.0010 | 0.0110 | significant |
| Binding − AlphaFold-rsa | 0.0108 | 0.1080 | — |
| Binding-IDR − bindEmbed21IDR | 0.1064 | 0.7447 | — |

## Why it worked

One variable changed: proteins longer than 1022 residues kept as overlapping
windows instead of dropped, recovering 2,440 of them. Training set 20,053 →
27,412 rows before the leak filter.

The gain is monotone in length, which is what the hypothesis predicted:

| Disorder-NOX band | targets | mt_full | windowed | Δ |
|---|---:|---:|---:|---:|
| 0–400 | 112 | 0.8918 | 0.8990 | +0.007 |
| 400–700 | 51 | 0.8881 | 0.8812 | −0.007 |
| 700–1022 | 19 | 0.8175 | 0.8465 | +0.029 |
| 1022–1500 | 15 | 0.7542 | 0.8395 | **+0.085** |
| 1500+ | 7 | 0.7357 | 0.9329 | **+0.197** |

Short proteins barely move; the longest improve by 0.197. Nothing else in the
run changed, so this is the effect of training on the proteins that were being
discarded — and those are exactly the targets competing methods decline.

## What got worse

Disorder-PDB slipped 0.9603 → 0.9590, inside the pre-registered margin, and
Binding-IDR fell 0.5180 → 0.4945.

Binding-IDR is the outstanding failure and more data will not fix it. Our
binding head scores 0.7924 on Binding and 0.4810 on Binding-IDR, which is the
same labels restricted to disordered residues. Conditioning on disorder erases
the signal entirely: the model has learned to find disorder, not binding. That
needs a different training signal, not more of this one.

## Leak filter

27,412 proteins before, 765 removed: 283 exact id/sequence matches against the
CAID3 targets, 482 homologs at identity ≥ 0.40. Reference file md5
`6feaff35263e7fd4a3f03640c23786fe`, the official one. The union of all five
references is the same 319 proteins, all covered.
