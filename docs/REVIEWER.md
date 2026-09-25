# For a reviewer, in about ten minutes

## Check a theorem the paper cites

```bash
cd lean && lake update && lake build      # pinned: lean4 v4.28.0, mathlib v4.28.0
```

```lean
import RequestProject.DiscordantImbalance
#print axioms IDR.nuPair_le_imbalanced
-- 'IDR.nuPair_le_imbalanced' depends on axioms:
-- [propext, Classical.choice, Quot.sound]
```

`lean/README.md` maps every statement in the manuscript to its file, and
`lean/PROOF_GAPS_CLOSED.md` maps each statement the paper once used ahead of its
proof to the theorem that closes it.

## Check a number the paper reports

```bash
pip install disordernet
disordernet table                                  # Table S17
disordernet capacity --methods 117 --eps 0.0801    # CAID3: capacity 7
disordernet capacity --methods 1060 --eps 0.0583 --range 0.55 0.92   # ImageNet
```

## Re-derive the figures

```bash
python paper/make_figures.py         # six PDFs, each from the JSON its job wrote
python paper/make_supplementary.py   # the Supplementary Data CSVs
```

No figure restates a number: each panel reads the file its compute job produced,
and the two panels that quote a committed table parse it rather than repeating
it. `results/caid3/*.md` carry the job identifiers.

## The things we got wrong

Supplementary Note S5 catalogues them with the measurement that caught each,
including two numbers this manuscript published and then corrected: a pairwise
capacity of 72 that formalising the denominator turned into 51, and a bound
called tight whose hypothesis fails on our own data.
