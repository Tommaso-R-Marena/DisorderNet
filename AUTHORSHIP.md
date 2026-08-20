# Who did what

This repository accompanies a manuscript, so the division of labour should be on
the record rather than inferred from a commit log.

## Tommaso R. Marena

Conceived the project and directed it throughout. Every question the work
answers was posed here, and every decision about what to pursue, what to
abandon, and what counted as an answer was made here.

**The entire Lean 4 development is his work.** That is the spine of the paper:
the capacity theorem in three score models with attainment instances, the
constructive converse (`unresolvable_pair`), the AUC decomposition and its
invariance, the discordance identity and the exact comparable-pair count, the
imbalance-corrected bound, NP-hardness of the recalibration optimum proved from
a verifier definition of NP through a verified Tseitin transformation, the
arbitrary-dependence Benjamini–Yekutieli theorem with its matching sharpness
instance, conformal p-value validity and its composition with that screen, and
the counting converse. Roughly forty files, `sorry`-free, on `propext`,
`Classical.choice` and `Quot.sound`.

Several of the paper's corrections originate there rather than in any analysis:
formalising `card_comparablePairs` is what exposed a wrong denominator that had
produced a published capacity of 72 (the correct figure is 51), and formalising
the hypotheses of `nuPair_le_two_eps_sq` is what showed the bound this project
had called tight does not apply to its own data.

## Assistants

Parts of the analysis code, the evaluation pipeline, the figures and the
manuscript drafting were produced with AI coding assistants — Claude (Anthropic)
and Cursor — under direction. Their contributions are visible in the history:
commits carry `Co-Authored-By` trailers where applicable, and some are authored
directly by an agent identity (`Cursor Agent`, `cursor[bot]`, `Claude`).

Nothing has been removed from that record. It is left intact because the paper's
central claim is that a result should be checkable against its source, and a
history edited to look tidier than the work was would contradict it. Nature
Portfolio also requires AI assistance to be disclosed, and a repository cited in
a Code Availability statement is part of that disclosure.

## What that division means for the paper

The theorems are the contribution, and they are not assistant output. The
assistants did what assistants are good at: fetching and verifying public data,
running jobs on a cluster, computing statistics, drawing figures, and drafting
prose against a specification. Each of those is checkable, and much of it was
checked and found wrong at least once — the errors and the measurements that
caught them are catalogued in Supplementary Note S5 rather than quietly fixed.

## Reproducing the claims

- Lean: `RequestProject/`. `lake build`, then `#print axioms` on any cited name.
- Analysis: `results/caid3/*.py`, each writing the JSON that a figure reads.
- Figures and tables: `paper/make_figures.py`, `paper/make_supplementary.py`.
- Manuscript: `paper/latex/`, `pdflatex` on a stock TeX Live.

No figure restates a number; each panel reads the file its job wrote, and the
two panels that quote a committed table parse it.
