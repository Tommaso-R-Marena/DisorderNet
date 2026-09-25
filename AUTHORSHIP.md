# Who did what

This repository accompanies a manuscript, so the division of labour should be on
the record rather than inferred from a commit log.

## Tommaso R. Marena

Conceived the project and directed it throughout. Every question the work
answers was posed here, and every decision about what to pursue, what to
abandon, and what counted as an answer was made here.

He specified every theorem in the formal development and directed its
construction: which statement was worth proving, what its hypotheses should be,
which of them were load-bearing, and when a proved statement did not say what
the paper needed it to say.

## The formal development

The Lean 4 proofs were produced with **Aristotle** (Harmonic) under that
direction. They are the spine of the paper: the capacity theorem in three score
models with attainment instances, the constructive converse
(`unresolvable_pair`), the AUC decomposition and its invariance, the discordance
identity and the exact comparable-pair count, the imbalance-corrected bound,
NP-hardness of the recalibration optimum proved from a verifier definition of NP
through a verified Tseitin transformation, the arbitrary-dependence
Benjamini–Yekutieli theorem with its matching sharpness instance, conformal
p-value validity and its composition with that screen, and the counting
converse. Roughly forty files, `sorry`-free, on `propext`, `Classical.choice`
and `Quot.sound`.

**Provenance does not bear on whether these are true.** A `sorry`-free Lean
proof on the standard axioms is checked by the kernel, so it is correct
independently of who or what wrote it — which is the reason to state the tooling
plainly rather than to hedge about it. What provenance does bear on is whether
the right things were proved, and that is the part directed here: several of the
paper's corrections originate in the formalisation rather than in any analysis.
Formalising `card_comparablePairs` exposed a wrong denominator that had produced
a published capacity of 72 (the correct figure is 51), and formalising the
hypotheses of `nuPair_le_two_eps_sq` showed the bound this project had called
tight does not apply to its own data. Neither would have surfaced without
someone deciding those were the statements to formalise.

The development is in `lean/` — 403 files, with the toolchain and mathlib
revision pinned so the axiom claim can be checked rather than taken on trust.
Commits before 2026-08-20 do not touch it, which is why no `Co-Authored-By`
trailer sits on a proof.

## Assistants

Parts of the analysis code, the evaluation pipeline, the figures and the
manuscript drafting were produced with AI coding assistants — Claude (Anthropic)
and Cursor — under direction. Their contributions are visible in the history:
commits from 2026-07-30 onward carry `Co-Authored-By` trailers, and some are
authored directly by an agent identity (`Cursor Agent`, `cursor[bot]`,
`Claude`). Those trailers cover analysis and manuscript work only; none of them
sits on a proof, because the proofs are not kept here.

Nothing has been removed from that record. It is left intact because the paper's
central claim is that a result should be checkable against its source, and a
history edited to look tidier than the work was would contradict it. Nature
Portfolio also requires AI assistance to be disclosed, and a repository cited in
a Code Availability statement is part of that disclosure.

## What that division means for the paper

The theorems are the contribution. They were machine-produced and are
machine-checked, which makes their correctness independent of their authorship;
what was human was the choice of what to prove. The coding assistants did what
coding assistants are good at: fetching and verifying public data,
running jobs on a cluster, computing statistics, drawing figures, and drafting
prose against a specification. Each of those is checkable, and much of it was
checked and found wrong at least once — the errors and the measurements that
caught them are catalogued in Supplementary Note S5 rather than quietly fixed.

## Reproducing the claims

- Lean: `lean/`. `lake update && lake build`, then `#print axioms` on any cited
  name. `lean/README.md` maps every claim in the paper to its file.
- Analysis: `results/caid3/*.py`, each writing the JSON that a figure reads.
- Figures and tables: `paper/make_figures.py`, `paper/make_supplementary.py`.
- Manuscript: `paper/latex/`, `pdflatex` on a stock TeX Live.

No figure restates a number; each panel reads the file its job wrote, and the
two panels that quote a committed table parse it.
