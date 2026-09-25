# Pre-registration 4 — private trunk capacity for the binding tasks

Committed **before** the run starts and before its results exist. Methodology is
`METHODOLOGY.md`, unchanged.

## What the last run showed

The protein-level bias moved Binding-IDR from 0.5007 to **0.6062**, rank 30 to
**5 of 115** — the largest gain on that benchmark in this project, and exactly
on the axis the AUC decomposition identified. It was rejected because three
floors broke: Disorder-NOX −0.0118, Linker −0.0285, Binding −0.0010.

The cause is precise. The bias path was detached, so no gradient from a binding
task reached a disorder read-out, and tests proved it. But the binding *losses*
still shaped the **shared trunk**, and the disorder tasks read from that trunk.
Protecting a read-out is not protecting a representation.

## What changes

The binding tasks read a **detached** shared trunk through two private residual
blocks of their own. Their loss therefore contributes exactly zero gradient to
`proj` and `blocks`, and the disorder tasks train as though the binding tasks
were absent. This is asserted at `atol=0, rtol=0` on a single model in eval
mode, alongside a companion test that the private stack does learn, so the
isolation is not achieved by the path being dead.

`mt_pbias` is the control: same protein bias, same conditioning, same windowed
training, no private trunk.

## Primary endpoints

On **Binding-IDR**, all 52 targets, coverage 1.00 required, paired
protein-clustered bootstrap (10,000 resamples), Holm across these two:

- **P1** — retains the pbias gain: Binding-IDR ≥ 0.5900, i.e. within 0.0162 of
  `mt_pbias`'s 0.6062. The private trunk removes binding's influence over the
  shared representation, which could cost it; P1 asks whether the gain survives.
- **P2** — beats bindEmbed21IDR (0.6407). Registered as a stretch, expected to
  fail from 0.6062.

## The constraint this run exists to satisfy

All four floors, unchanged, and now with a mechanism that should make them
automatic rather than hoped for:

| benchmark | floor |
|---|---:|
| Disorder-PDB | 0.9545 |
| Disorder-NOX | 0.8878 |
| Linker | 0.9193 |
| Binding | 0.7884 |

**Prediction that makes this falsifiable**: because binding contributes no
gradient to the shared trunk, Disorder-PDB, Disorder-NOX and Linker should move
by *no more than seed noise* from a run without binding tasks. Measured
seed-to-seed variation on CV is 0.0002. If those three move materially, the
isolation is not doing what the tests say and the explanation is wrong even if
the floors pass.

Binding is not protected by the isolation — it is one of the isolated tasks —
so its floor remains a genuine risk.

## Stopping rule

One run, one evaluation, reported once. The single exception, as before: an
evaluation executed with a misconfigured architecture is void and repeated with
the identical analysis.
