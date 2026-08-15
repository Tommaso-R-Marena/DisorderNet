# Pre-registration 2 — disorder-conditioned binding

Written and committed **before** the run starts and before any of its results
exist. Git history is the timestamp. Methodology is `METHODOLOGY.md`, unchanged.

## The problem this addresses

Binding-IDR is the Binding labels restricted to disordered residues — verified
identical on all 7,741 evaluated residues against CAID's own files. Our head
scores 0.7934 on Binding and 0.5007 on Binding-IDR. Conditioning on disorder
erases the signal, which means what it learned was disorder: across a protein,
binding sites sit in IDRs and IDRs are the disordered part, so "is this
disordered" answers Binding well and Binding-IDR not at all.

More data will not fix a model answering a different question. So the model is
given the disorder answer instead of being made to rediscover one.

## What changes, and only this

Each conditioned task gets a second read-out over `[trunk, p(disorder)]` — 36
parameters — added to its existing logit. Everything else is identical to
`mt_windowed`: same five tasks, frozen ESM-2 650M over layers 21–32,
`structure_dim=24`, wide receptive field, 5 folds, 8 epochs, identity 0.40,
long proteins kept as overlapping windows.

`mt_windowed` is therefore the exact unconditioned control: same configuration,
`condition_binding=False`. One variable separates them.

The conditioning signal is detached, so the path is strictly additive and no
gradient from a binding task reaches the disorder read-out. That is asserted in
`tests/test_disorder_conditioned_binding.py`, not merely intended.

## Primary endpoints

Two, on **Binding-IDR**, unfused, all 52 official targets, coverage 1.00
required, paired protein-clustered bootstrap (10,000 resamples, two-sided),
Holm–Bonferroni across **these two alone**:

- **P1 — conditioned > `mt_windowed` unconditioned.**
  The ablation. Both models predict the same 52 targets, so the comparison is
  paired on every residue. This is the test of whether conditioning works.
- **P2 — conditioned > bindEmbed21IDR (leader, 0.6407).**
  A stretch. We are at 0.5007 and the benchmark's own resolution needs about
  +0.009; closing 0.14 in one architectural change would be extraordinary.
  Registered so that a success cannot be claimed as though planned, and a
  failure cannot be quietly dropped.

Family size is 2. Nothing is added to it afterwards.

## Non-inferiority constraints — all four current placements

The run must not buy Binding-IDR with the benchmarks already won. Each floor is
the corrected `mt_windowed` figure less a 0.005 margin:

| benchmark | current | floor | current rank |
|---|---:|---:|---|
| Disorder-PDB | 0.9595 | **0.9545** | 1 / 115 |
| Disorder-NOX | 0.8928 | **0.8878** | 1 / 115 |
| Linker | 0.9243 | **0.9193** | 1 / 115 |
| Binding | 0.7934 | **0.7884** | 2 / 115 |

**If any floor is breached, the conditioned variant is rejected outright**,
whatever it does on Binding-IDR, and `mt_windowed` remains the model of record.
Four floors rather than one, because there are now four placements to protect
and the change touches a task that shares their trunk.

## Secondary — exploratory, labelled as such

The ensemble of all available checkpoints, and per-length-band breakdowns.
Corrected within their own family and reported whether or not they improve.

## Stopping rule

One run, one evaluation, reported once. No re-running with different seeds,
resample counts or fusion partners until something crosses a threshold.

The single exception, stated in advance: if the evaluation is found to have been
executed with a misconfigured architecture — as happened once, when the head was
built at a 61-residue receptive field for a checkpoint trained at 213 — the
measurement is void and is repeated with the identical analysis. That is a
correction, not a second attempt, and the distinction is only honest if the
condition is written down before it arises.

## Reported regardless of outcome

Every primary and secondary number, raw and adjusted; coverage on all five
benchmarks; each of the four non-inferiority checks; and which claims are
confirmatory and which exploratory.
