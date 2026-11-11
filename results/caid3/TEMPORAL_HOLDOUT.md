# A benchmark the calendar held out

Every other leak control in this project is a filter that has to be *correct*.
One of them was not: the validation holdout's homology pass silently removed
nothing for a while because a set of tuples was tested for membership of a
string, and the log printed a plausible number throughout.

A **temporal** holdout needs no filter to be right. The training caches are
dated 2026-08-08 (DisProt) and 2026-08-10 (MobiDB). A structure first released
on 2026-08-11 or later was not in them, whatever any code does.

This is also the nearest thing to a CASP-style assessment available for
disorder. **CASP retired its disorder category after CASP10** and CAID replaced
it, so there is no CASP disorder track to enter. But Disorder-PDB's labels *are*
"present in SEQRES, absent from the coordinates", which is exactly what a newly
deposited structure reports. Scoring on structures nobody had when the model was
trained is the experiment CASP runs, on the quantity CAID scores.

## Construction

1,916 protein polymer entities released on or after 2026-08-11 (X-ray and
cryo-EM; NMR excluded, since every SEQRES residue has coordinates in every
model). Labels are RCSB's own `UNOBSERVED_RESIDUE_XYZ` annotation, not a
re-derivation.

| stage | chains |
|---|---:|
| fetched | 1,916 |
| after length, alphabet, both-classes, sequence-dedup | **645** |
| minus exact training sequences | −175 |
| minus BLAST homologues at ≥40% identity | −284 |
| **scored** | **186** |

**71% of the surviving chains were removed as already-represented.** That is
the finding to take away from the construction: most "new" PDB entries are of
proteins the training data already contains, and a temporal cutoff *alone* is
not a leak control. The calendar and the homology filter are both necessary.

267,655 residues before filtering, 26.8% unobserved — close to CAID3
Disorder-PDB's 31.6% prevalence, which is evidence the label definition matches.

## Result, all 186 chains

| model | pooled | within-protein | between-protein |
|---|---:|---:|---:|
| **`mt_windowed`** | **0.8933** | **0.8502** | 0.8936 |
| `mt_pbias` | 0.8859 | 0.8171 | 0.8865 |
| `mt_publication` | 0.8857 | 0.8102 | 0.8863 |

No published entrant can be scored here — there are no submissions for
structures that did not exist — so the comparison is against training-free
structural baselines, on the chains where they can be computed.

## Result, matched on the 61 chains with an AlphaFold model

Ours restricted to the same subset, because comparing a 186-chain score with a
61-chain score is comparing two benchmarks:

| model | pooled | within-protein |
|---|---:|---:|
| `mt_publication` | **0.9113** | 0.7895 |
| `mt_windowed` | 0.9095 | **0.8023** |
| `mt_pbias` | 0.9024 | 0.7890 |
| AlphaFold-pLDDT | 0.8737 | 0.7816 |
| AlphaFold-rsa | 0.8229 | 0.7383 |

Paired against the best baseline, protein-clustered bootstrap, 2,000 resamples:

| comparison | Δ | 95% CI | p | Holm-adjusted |
|---|---:|---|---:|---:|
| `mt_publication` − AlphaFold-pLDDT | +0.0376 | [+0.0050, +0.0802] | 0.0240 | 0.072 |
| `mt_windowed` − AlphaFold-pLDDT | +0.0358 | [+0.0020, +0.0738] | 0.0370 | 0.074 |
| `mt_pbias` − AlphaFold-pLDDT | +0.0287 | [−0.0072, +0.0717] | 0.1209 | 0.121 |

**Two of three clear 0.05 nominally and neither survives Holm across the three.**
This was not pre-registered — it is an exploratory analysis of a set built after
the CAID3 numbers existed — and 61 chains is a small sample. The honest reading
is *the model beats the strongest training-free structural baseline on
structures it could not have seen, by a margin whose interval excludes zero but
whose p-value does not survive correction across three comparisons.*

## The finding that matters most, and it is not flattering

**`mt_pbias` is our best CAID3 model (0.9635 on Disorder-PDB) and the worst of
the three here** — and the only one that fails to separate from a training-free
baseline. Its within-protein AUC on the full 186 is 0.8171 against
`mt_windowed`'s 0.8502, a gap of 0.033, larger than the gap between them on
CAID3.

The protein-level bias term buys CAID3 points and **generalises worse to
proteins nobody had seen**. That is exactly what a temporal holdout is for, it
is the kind of result a homology filter cannot produce, and it means the
checkpoint this project has been treating as its best is not the one to ship.

`mt_windowed` — no protein bias, no private trunk — is the model of record for
generalisation. Under `PREREGISTRATION_6`'s floors the CAID3 selection would
still pick `mt_pbias`; the temporal set says that selection is optimistic.

## Caveats

- **Not pre-registered.** Built after the CAID3 result. Descriptive.
- **61 chains** for every baseline comparison, so the intervals are wide.
- **One week of PDB.** The cutoff is set by the training caches, not chosen;
  extending it means waiting, not re-selecting, and re-running on a later
  cutoff is the correct way to add power.
- **Chain-level, not domain-level.** A construct with a disordered purification
  tag counts its tag. This is the same convention Disorder-PDB uses, so it is
  comparable, but it is not a curated set and it is harder for that reason: our
  0.8933 here against 0.9595 on CAID3 is mostly that.
- AlphaFold models were fetched for these accessions after the fact. 593 of 634
  accessions have one; 61 chains survive the homology filter *and* have a model
  whose sequence matches exactly.

## Reproduction

```bash
python rockfish/build_temporal_holdout.py --cutoff 2026-08-11 \
  --out .../temporal/disorder_pdb.fasta --raw .../temporal/raw_records.json
sbatch rockfish/slurm/eval_temporal.sbatch
```

Jobs 30034161 (build, 1 min), 30034251 (AlphaFold prefetch, 15 s), 30034521
(evaluation, 2m47s). Raw output `temporal/results.json`.
