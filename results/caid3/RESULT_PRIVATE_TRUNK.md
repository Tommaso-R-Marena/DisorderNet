# Private trunk — REJECTED, and the reason is the point

Registered in `PREREGISTRATION_4.md` before the run. Job 29922832, 18h24m.
Evaluated job 30020129, all five official CAID3 references, coverage 1.00
throughout.

## Outcome

| endpoint | required | measured | outcome |
|---|---:|---:|---|
| **P1** Binding-IDR retains the pbias gain | ≥ 0.5900 | **0.5800** | **FAIL** |
| **P2** beats bindEmbed21IDR | ≥ 0.6407 | 0.5800 | FAIL (registered as a stretch) |
| Disorder-PDB floor | ≥ 0.9545 | 0.9570 | PASS |
| Disorder-NOX floor | ≥ 0.8878 | 0.8700 | **FAIL** |
| Linker floor | ≥ 0.9193 | 0.9048 | **FAIL** |
| Binding floor | ≥ 0.7884 | 0.7319 | **FAIL** |

**Variant rejected.** Three floors and the primary endpoint.

Placements, for the record: Disorder-PDB 1/115, Linker 1/115, Disorder-NOX
4/115 (1/58 full-coverage), Binding-IDR 8/115, Binding 38/115.

## The registered prediction was falsified, and that is the finding

`PREREGISTRATION_4.md` said, before the run:

> because binding contributes no gradient to the shared trunk, Disorder-PDB,
> Disorder-NOX and Linker should move by *no more than seed noise* from a run
> without binding tasks. Measured seed-to-seed variation on CV is 0.0002. If
> those three move materially, the isolation is not doing what the tests say
> and the explanation is wrong even if the floors pass.

Against `mt_windowed` they moved by −0.0228 on Disorder-NOX and −0.0195 on
Linker, a hundred times seed noise. But `mt_windowed` was filtered against one
CAID reference and `mt_private` against five, so that comparison spans two
training regimes and is not the test the prediction intended.

Against **`mt_pbias`**, the regime-matched control — same five references, same
protein bias, same conditioning, same windowed training, no private trunk:

| benchmark | mt_pbias | mt_private | Δ |
|---|---:|---:|---:|
| Disorder-PDB | 0.9635 | 0.9570 | −0.0065 |
| Disorder-NOX | 0.8760 | 0.8700 | −0.0060 |
| Linker | 0.8908 | 0.9048 | **+0.0140** |
| Binding | 0.7874 | 0.7319 | **−0.0555** |
| Binding-IDR | 0.6062 | 0.5800 | −0.0262 |

Now the shape is legible. The three disorder tasks moved by −0.0065, −0.0060
and +0.0140 — small, and not the 0.0002 predicted, but the right order of
magnitude for "roughly untouched". The two **isolated** tasks are the ones that
collapsed: Binding −0.0555 and Binding-IDR −0.0262.

## What this actually establishes

**The isolation worked and the isolation was the wrong thing to want.**

The gradient guarantee holds — it is asserted at `atol=0, rtol=0` and the tests
pass. Binding contributes exactly nothing to the shared trunk. And that is
precisely what cost it: binding has 891 training proteins against disorder's
21,386, and the entire argument for a shared trunk in this project was that it
"carries disorder's data into tasks with a twentieth of it". Detaching the
trunk stops binding's loss from reshaping the representation *and* stops
binding from being trained on a representation shaped by disorder.

So the trade is now measured rather than assumed:

- what the disorder tasks lose to binding's gradient: **~0.006 AUC**
- what the binding tasks lose without disorder's representation: **~0.03–0.06**

Sharing is net positive by roughly an order of magnitude. The multi-task
premise this project was built on is confirmed by the run designed to remove
it.

## The floors were compared across training regimes, and that is a defect

`NON_INFERIORITY_FLOORS` were set from `mt_windowed`, a one-reference run.
Three of the last four variants were dual-filtered against CAID3 and all four
CAID2 references, which removes more training data. Judging a five-reference
run against a one-reference floor charges it for data it was required to give
up in order to be evaluable on CAID2 at all.

`mt_private` fails regime-matched floors too — derived from the best
five-reference run per task, less the same 0.005 margin — so the rejection
stands either way:

| benchmark | regime-matched floor | mt_private | outcome |
|---|---:|---:|---|
| Disorder-PDB | 0.9585 | 0.9570 | FAIL |
| Disorder-NOX | 0.8810 | 0.8700 | FAIL |
| Linker | 0.8858 | 0.9048 | PASS |
| Binding | 0.7824 | 0.7319 | FAIL |

That the conclusion survives both floor sets is the only reason this is being
reported rather than re-run. The floors are corrected going forward
(`PREREGISTRATION_6.md`), not retroactively, and the run judged under both.

## What follows

Not another isolation variant. The next question is not *how do we stop binding
from touching the trunk* but *how do we give binding a differently shaped view
of the same trunk* — the shared gradient kept, the read-out narrowed to the
scale a binding site actually occupies. On CAID3 Binding-IDR the method leading
both the pooled and the within-protein axis is LIPNet, a linear-interacting-
peptide predictor, and a short linear motif is five to fifteen residues against
this trunk's global dependency span.
