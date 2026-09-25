# Three registered endpoints, closed — all three fail

Job 30098857, `results/caid3/registered_endpoints.py`, raw output
`registered_endpoints.json`.

`PREREGISTRATION_7`, `_8` and `_9` each state their primary endpoint on a
statistic the CAID3 evaluator does not write — pooled Binding-IDR against a
regime-matched control, the **between**-protein component of Disorder-NOX, and
the **within**-protein component of Disorder-NOX. Their pooled scores had been
recorded and each was informally filed as rejected, but the endpoint they were
registered on had never been computed. Calling a variant rejected without
evaluating its own endpoint is asserting an outcome rather than measuring one.

Control throughout is `mt_control`, which shares each run's training union and
validation holdout.

## The verdicts

| registration | variant | task | statistic | variant | control | Δ | P1 | P2 |
|---|---|---|---|---:|---:|---:|---|---|
| PREREG_7 | `mt_motif` | Binding-IDR | pooled | 0.5281 | 0.5422 | −0.0141 | **FAIL** | FAIL (bar 0.7620) |
| PREREG_8 | `mt_wass` | Disorder-NOX | between-protein | 0.8564 | 0.8600 | −0.0036 | **FAIL** | — |
| PREREG_9 | `mt_rank` | Disorder-NOX | within-protein | 0.8500 | 0.8511 | −0.0011 | **FAIL** | FAIL (bar 0.8564) |

**All three fail on the statistic they were registered on**, and **not one of
the fifteen paired per-target comparisons survives Holm** (smallest adjusted
p = 0.345).

The two mechanism predictions are the sharper part. `PREREGISTRATION_8` said the
distribution-matching term should move the **between**-protein component,
because the term is permutation-invariant within a protein and so cannot touch
the within-protein ordering; it moved between-protein by **−0.0036**.
`PREREGISTRATION_9` said the ranking surrogate should move the **within**-protein
component, tested directly rather than inferred; it moved it by **−0.0011**.
Each registration named the quantity its own mechanism had to move, and neither
moved it.

## The finding that was not registered and is worth more

On Binding-IDR, restricting to the 42 targets that carry both classes among
evaluated residues **reverses the sign of every one of the three comparisons**:

| variant vs control, Binding-IDR | all 52 targets | 42 two-class targets |
|---|---:|---:|
| `mt_motif` | −0.0141 | **+0.0480** |
| `mt_wass` | −0.0308 | **+0.0535** |
| `mt_rank` | −0.0050 | **+0.0117** |

Ten targets carry one class. A single-class target contributes **no
within-protein pair at all** — every pair it enters is between-protein — so the
whole of each variant's apparent regression on Binding-IDR lives in targets that
carry no residue-level question. On the 42 targets that do ask one, all three
variants beat the control.

This is the paper's thesis appearing where nobody put it: the endpoint was
registered on the pooled number, the pooled number is dominated by
between-protein pairs, and on Binding-IDR a fifth of the targets contribute
nothing but between-protein pairs. The registered endpoints still fail — they
were registered on the pooled statistic and that is the statistic that decides
them — but the reason they fail is not that the variants are worse at the
residue-level question.

## Full decomposition, Disorder-NOX

| | pooled | within (pair-wtd) | within (unwtd) | between |
|---|---:|---:|---:|---:|
| `mt_control` | 0.8557 | 0.7914 | 0.8511 | 0.8600 |
| `mt_motif` | 0.8684 | 0.8039 | 0.8518 | 0.8681 |
| `mt_rank` | 0.8575 | 0.7812 | 0.8500 | 0.8613 |
| `mt_wass` | 0.8515 | 0.7824 | 0.8478 | 0.8564 |

`mt_motif` is the only variant that improves anything on Disorder-NOX, and it
was registered on Binding-IDR.

## Status

Confirmatory for the three registered endpoints, which are reported as
**failed**. The two-class-subset reversal is exploratory: it was found while
computing the endpoints and was not registered.
