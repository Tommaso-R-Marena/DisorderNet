# DisorderNet

**A benchmark that ranks methods on imperfect labels has a capacity — the number
it can place in a certified order. Most are far over it. This is the theory, the
proofs, and the protocol that resolves more.**

[![CI](https://github.com/Tommaso-R-Marena/DisorderNet/actions/workflows/ci.yml/badge.svg)](https://github.com/Tommaso-R-Marena/DisorderNet/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/disordernet.svg)](https://pypi.org/project/disordernet/)
[![Python](https://img.shields.io/pypi/pyversions/disordernet.svg)](https://pypi.org/project/disordernet/)
[![Lean 4](https://img.shields.io/badge/Lean%204-sorry--free-0F766E)](lean/)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Try%20it-0D9488)](https://huggingface.co/spaces/The-Philosopher/DisorderNet-Capacity)
[![License](https://img.shields.io/badge/license-MIT-informational)](LICENSE)

---

## The result in one table

Substituting published, human-validated label-error rates into a bound that is
machine-checked in Lean 4:

| benchmark | items | ε | **methods it can order** |
|---|---:|---:|---:|
| QuickDraw | 50,426,266 | 10.12% | **5** |
| CAID3 Disorder-PDB | 99,239 | 8.01% | **7** |
| CIFAR-100 | 10,000 | 5.85% | **9** |
| **ImageNet** | 50,000 | 5.83% | **9** |
| CIFAR-10 | 10,000 | 0.54% | 93 |
| MNIST | 10,000 | 0.15% | 334 |

ImageNet's leaderboard ranks on the order of a thousand entries. Rates are lower
bounds ([Northcutt, Athalye & Mueller, NeurIPS 2021](https://arxiv.org/abs/2103.14749)),
so these capacities are upper bounds.

**This does not say a published ranking is wrong.** It says the data cannot
establish it: past the capacity, for two close methods there exist two ground
truths consistent with every observation the benchmark recorded, one favouring
each. No bootstrap, permutation test or cross-validation recovers an ordering
that is not a function of the data.

## And what to do about it

Whether a benchmark can do better is decided by the statistic it chose. An
average of per-item losses admits no escape. A **ranking statistic over pairs
within a group** is corrupted only when *both* members of a pair are
mis-annotated, in opposite directions — a structural constraint, not a
distributional assumption, which makes the noise second-order.

Measured on ImageNet's own adjudicated labels, re-scored on per-class
one-vs-rest AUC:

| | capacity, top-1 accuracy | capacity, per-class ranking |
|---|---:|---:|
| ImageNet | **10** | **387,425** |
| CIFAR-100 | 12 | 27,337 |

Same items, same labels, same errors.

---

## Install

```bash
pip install disordernet
```

Python 3.9–3.13, Linux/macOS/Windows. Core install is `numpy` + `scipy` — no
GPU, no model weights, no network.

```bash
pip install "disordernet[web]"      # Gradio app
pip install "disordernet[predict]"  # the disorder predictor (torch, ESM-2)
pip install "disordernet[all]"
```

## Use it

### Assess a benchmark

```bash
disordernet capacity --methods 117 --eps 0.0801
```

```
117 methods, annotation error rate 0.0801
  capacity, as scored                   7
  capacity, scored on pairs            51
  the benchmark is over capacity by 16.7x
  comparisons it cannot decide        920 of 6,786
```

```bash
disordernet table                                       # the table above
disordernet capacity --methods 1060 --eps 0.0583 \
                     --range 0.55 0.92                  # ImageNet
```

### Measure your own error rate

Where an item has been annotated twice — two structures of one protein, two
assessors on one query, an original label and an adjudicated correction — both
rates the theory needs can be measured rather than assumed.

```python
from disordernet import rates

r = rates(annotation_a, annotation_b)     # two 0/1 arrays over the same items
print(r)
# eps_label 0.0651   eps_pair 9.96e-03   ratio 7x   kappa 1.186
```

The counts are exact, not estimates: `discordant = 2·d·u`
(`discordant_eq_flip_product`) and `comparable = 2·(a·e + d·u)`
(`card_comparablePairs`).

### Score a field under the pairwise protocol

```python
from disordernet import rank

board = rank(reference, predictions, eps_pair=0.00996)
print(board)
```

Six steps, each in `disordernet/protocol.py`: a coverage gate (declining targets
makes a method ineligible, not merely penalised), the Mann–Whitney AUC per
target, the **unweighted** mean over targets, Holm-corrected Wilcoxon
separation, and a capacity beyond which no rank is printed.

### In a browser

[**Hugging Face Space**](https://huggingface.co/spaces/The-Philosopher/DisorderNet-Capacity) — paste
a size and an error rate, get the verdict.

### In Docker

```bash
docker compose run --rm cli capacity --methods 117 --eps 0.0801
docker compose up web        # http://localhost:7860
```

---

## The proofs

`lean/` — 424 Lean 4 files, `sorry`-free, on the standard axioms (`propext`,
`Classical.choice`, `Quot.sound`). Toolchain pinned to
`leanprover/lean4:v4.28.0`, mathlib4 to `rev = v4.28.0`, because the axiom claim
is only meaningful against a specific toolchain.

```bash
cd lean && lake update && lake build
```

```lean
import RequestProject.DiscordantImbalance
#print axioms IDR.nuPair_le_imbalanced
```

Every function in the Python package names the theorem it implements, and CI
fails if a `sorry` appears or the pin drifts. `lean/README.md` maps each
statement the paper cites to its file.

| result | theorem |
|---|---|
| capacity `k ≤ ⌈1/2ε⌉`, three score models, attained | `card_le_benchCapacity`, `benchCapacity_attained` |
| the ordering is not a function of the data | `unresolvable_pair` |
| `discordant = 2·d·u`, exactly | `discordant_eq_flip_product` |
| `ν_pair ≤ κ·ε²/(1−ε)²`, no balance hypothesis | `nuPair_le_imbalanced` |
| Benjamini–Yekutieli under arbitrary dependence, and attained | `selfConsistent_fdr_le_harmonic`, `bh_fdr_eq_harmonic` |
| conformal p-values are superuniform, and compose | `conformalP_superuniform`, `conformal_screen_fdr_le` |
| ≥798 of 6,786 CAID3 comparisons undecidable | `unresolvable_count_117` |
| NP-hardness of the recalibration optimum | `Complexity/biasThreshold_hard` |

---

## The predictor

DisorderNet is also a per-residue disorder predictor: a frozen ESM-2 650M
backbone, a learned mixture over its layers, a four-block dilated CNN trunk with
an AlphaFold structure-channel block, and one linear read-out per CAID task
(~2M trainable parameters).

| | CAID3 | rank / entered | rank / full-coverage |
|---|---:|---:|---:|
| Disorder-PDB | 0.9595 | **1 / 115** | **1 / 58** |
| Disorder-NOX | 0.8928 | **1 / 115** | **1 / 58** |
| Linker | 0.9243 | **1 / 115** | **1 / 91** |
| Binding | 0.7934 | 2 / 115 | 2 / 70 |
| Binding-IDR | 0.5007 | 30 / 115 | 17 / 70 |

First among full-coverage entrants on all four CAID2 references as well, from
training that never saw a CAID2 target or a 40%-identical homologue.

**It is not state of the art on the metric CAID reports**: +0.0043 over PUNCH2
at *p* = 0.20, and we do not claim otherwise. What it does separate on is the
calibration-invariant axis, on all five benchmarks after Holm correction over
thirty comparisons — a separation the benchmark's own metric provably cannot
make. That is the paper's point rather than a footnote to it.

Weights are not redistributed here; `results/caid3/` carries every score with
the job that produced it.

---

## Reproducing the paper

```bash
python paper/make_figures.py         # six figures, each from its job's JSON
python paper/make_supplementary.py   # Supplementary Data, S1–S18
cd paper/latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

No figure restates a number: each panel reads the file its compute job wrote,
and the two panels that quote a committed table parse it. `paper/latex/README.md`
carries the submission checklist; [`docs/REVIEWER.md`](docs/REVIEWER.md) is the
ten-minute path.

### What we got wrong

Supplementary Note S5 catalogues it, with the measurement that caught each —
including two numbers this project published and then corrected: a pairwise
capacity of 72 that formalising the denominator turned into 51, and a bound
called tight whose hypothesis fails on our own data. A benchmark culture that
reports only the number cannot catch any of them, which is rather the point.

---

## Repository

| path | what |
|---|---|
| `disordernet/` | the library and CLI |
| `lean/` | the formal development, 424 files |
| `paper/` | manuscript, figures, supplementary, LaTeX submission package |
| `results/caid3/` | every analysis, each with its job identifier |
| `colab/`, `rockfish/` | model, training and cluster evaluation |
| `hf_space/` | the Gradio app |
| `tests/` | including the tests that pin the paper's numbers |

## Citing

See [`CITATION.cff`](CITATION.cff).

> Marena, T. R. *Every benchmark has a capacity: how many methods imperfect
> labels can place in order.* (2026).

## Authorship

[`AUTHORSHIP.md`](AUTHORSHIP.md) records the division of labour, including the
use of Aristotle (Harmonic) for the proofs and of coding assistants for parts of
the analysis and drafting.

## Licence

MIT — see [`LICENSE`](LICENSE).
