# Backbone handedness — REJECTED, with a mechanism

Registered in `PREREGISTRATION_5.md` (the gate) and `PREREGISTRATION_6.md` (the
run) before either number existed. Jobs 30022469 (`mt_chiral`) and 30022470
(`mt_control`), 11h08m and 11h20m, identical in every respect except two extra
structural channels.

## The gate opened, by a factor of twenty-two

Training-free on CAID3, direction fitted on 800 MobiDB proteins with every
CAID3 target and sequence excluded:

| benchmark | handedness (within-protein) | achiral control `\|sin τ\|` | difference |
|---|---:|---:|---:|
| Disorder-PDB | 0.6435 | 0.4180 | **+0.2256** |
| Disorder-NOX | 0.6494 | 0.3966 | **+0.2529** |

The gate required ≥0.010. The achiral control sits *below chance*, so on the
face of it the entire signal was in the sign. The fitted direction was the one
polymer physics predicts — disordered residues less right-handed (mean 0.074)
than ordered (0.211) — and on the real cache 61.3% of torsions are
right-handed, which is what L-amino-acid proteins should give.

## The A/B says no, on every task

The fixed validation holdout — 2,991 rows, the same chains for both runs, the
first time in this project two architectures have been measured on identical
held-out data:

| task | `mt_chiral` | `mt_control` | Δ |
|---|---:|---:|---:|
| Disorder-NOX | 0.8409 | **0.8623** | **−0.0214** |
| Linker | 0.9150 | **0.9308** | −0.0158 |
| Binding | 0.8301 | **0.8386** | −0.0085 |
| Binding-IDR | 0.6235 | **0.6924** | **−0.0689** |
| Disorder-PDB | 0.9119 | **0.9128** | −0.0009 |

**Worse on all five.** P1 (beat the control on Disorder-NOX) fails.

Cross-validation had said the opposite on the primary task — chiral +0.0042 on
Disorder-NOX — which is exactly why the holdout was built. CV is computed on
each run's own folds; the holdout is the same chains for both.

## Why: the informative part of handedness is not the chiral part

Both facts can be true — a real training-free signal, and a channel that hurts —
and the reason is measurable.

| | Disorder-PDB | Disorder-NOX |
|---|---:|---:|
| corr(handedness, rsa) | −0.219 | −0.314 |
| corr(handedness, pLDDT) | +0.216 | +0.288 |
| **R² of handedness on (rsa, pLDDT, contacts)** | **0.057** | **0.108** |
| AUC of handedness alone | 0.6116 | 0.5924 |
| **AUC of its residual** | **0.4930** | **0.4951** |
| AUC of rsa | 0.9264 | 0.8182 |

Handedness is **not** redundant with the existing channels — they explain only
6–11% of its variance. But **the 90% they do not explain carries no label
information at all**: the residual scores 0.493 and 0.495, chance to three
decimals.

So every bit of handedness's predictive power lives in the small component it
shares with solvent accessibility, and the large genuinely-new component is
noise with respect to disorder. A channel like that can only cost capacity, and
the holdout measures the cost at −0.001 to −0.069.

## The claim this corrects

Earlier in this project the framing was: *every structural channel the model
reads is mirror-invariant, so handedness is information the model cannot see.*
The first half is true and was verified — reflect a protein and rsa, contacts
and pLDDT are unchanged while the torsion changes sign.

The second half does not follow, and the measurement says it is wrong. The
information the model cannot see is information it does not need. The part of
handedness that predicts disorder is not the chiral part; it is the part
correlated with burial, because helices are both buried and right-handed. That
is also why the achiral control fell below chance: `|sin τ|` destroys the sign
and with it the correlation with rsa, leaving something anti-informative.

## What is established

- Backbone handedness from a single predicted structure is computable, cheap
  (it comes out of the same mmCIF parse), and genuinely absent from every
  channel the model reads.
- It carries real training-free signal: within-protein AUC 0.61–0.65.
- **All of that signal is already available through solvent accessibility.**
  Conditioned on rsa, pLDDT and contacts, handedness is uninformative about
  disorder — 0.493 AUC.
- Adding it as a model input is therefore harmful, and the cost is measured
  rather than assumed.

This is a negative result with a mechanism, and it was registered as reportable
before the run: *"handedness is real, absent from every existing channel,
measurable training-free, and not usable by this architecture — which is a
finding about the architecture worth having."* The mechanism turns out to be
about the biology rather than the architecture, which is better.

## Still to come

The CAID3 evaluation of both checkpoints (jobs 30050407, 30050408) reports P1,
P2 and the four `PREREGISTRATION_6` floors on the official references. The
holdout already decides the comparison; the CAID3 numbers are the registered
endpoint and will be reported whatever they say.
