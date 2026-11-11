# Evolutionary constraint as a binding signal — negative result

The hypothesis: short linear motifs bind, are therefore under selection, and are
therefore more conserved than the disordered sequence around them. ESM-2's
masked-token probability for the residue actually present is a conservation
proxy needing no alignment, so binding residues inside an IDR should carry
higher local constraint. Binding-IDR was the target because the field is close
to helpless there — leader 0.641, top-eight entrants correlating 0.336 with one
pair at −0.209.

Computed on all 319 unique CAID3 targets, training-free, 16 masked passes per
chain.

## Per-residue — the prediction fails

| benchmark | raw PLL | local | inverted local | leader |
|---|---:|---:|---:|---:|
| Disorder-PDB | 0.3526 | 0.4604 | 0.5396 | 0.955 |
| Disorder-NOX | 0.4074 | 0.4814 | 0.5186 | 0.885 |
| Binding | 0.4640 | 0.5049 | 0.4951 | 0.776 |
| **Binding-IDR** | **0.5475** | **0.5258** | 0.4742 | 0.641 |
| Linker | 0.3653 | 0.4583 | 0.5417 | 0.897 |

On Binding-IDR, the benchmark this was built for, raw pseudo-likelihood reaches
0.5475 and the baseline-corrected version 0.5258. Both are barely off chance and
nowhere near the 0.641 leader. **The per-residue prediction is not supported.**

The local correction, which was supposed to isolate a motif from its composition
background, made it *worse* — 0.5258 against 0.5475. Whatever weak signal the
raw quantity carries is regional, not a local excess, which is the opposite of
the motif picture that motivated it.

Disorder-PDB at 0.3526 is the informative failure: strongly *anti*-correlated,
as it should be. Constrained residues are structured ones. The measurement is
working; the binding hypothesis is what fails.

## Per-protein — weak, and not significant

| benchmark | targets | Spearman | p | AUC_between |
|---|---:|---:|---:|---:|
| Disorder-PDB | 319 | −0.287 | **0.0000** | 0.3766 |
| Disorder-NOX | 204 | −0.114 | 0.105 | 0.5067 |
| Binding | 52 | −0.087 | 0.539 | 0.4585 |
| **Binding-IDR** | 52 | +0.186 | **0.186** | **0.5765** |
| Linker | 31 | −0.030 | 0.872 | 0.5114 |

Binding-IDR is the only benchmark where the sign matches the hypothesis, and its
AUC_between of 0.5765 does exceed our model's 0.4982 on that axis. But p=0.186
over 52 proteins is not evidence, and 0.5765 remains far below the leader's
0.6400.

The one significant result is Disorder-PDB's −0.287 at p<0.0001, which says
proteins whose disordered regions are more constrained have less disorder. That
is expected biology, not a finding.

## Verdict

**The hypothesis is not supported and the idea is dropped.** A protein-level
constraint feature might contribute something to Binding-IDR — 0.5765 against
our 0.4982 on the between-protein axis is the one number pointing the right way
— but on 52 proteins at p=0.186 it would be building on noise, and this project
has already retracted one claim built exactly that way.

Recorded because a negative result on a well-motivated hypothesis is worth as
much as the positive one it was hoping for, and because the machinery is tested
and reusable if a better-powered dataset appears.
