# Submission package — Nature Methods

Drop this whole directory into Overleaf (or any TeX Live installation) and
compile. Nothing external is required: no journal class file, no CTAN packages
beyond a standard distribution, and every figure is an embedded PDF.

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && pdflatex supplementary
pdflatex cover_letter
```

In Overleaf, set **main.tex** as the main document and the compiler to
**pdfLaTeX**; `supplementary.tex` and `cover_letter.tex` compile standalone from
the same project.

## What is here

| file | what it is |
|---|---|
| `main.tex` | the manuscript: abstract, main text, statements, references, figure legends |
| `methods.tex` | Methods, `\input` by `main.tex` — Nature places this after the references |
| `refs.bib` | bibliography |
| `supplementary.tex` | Supplementary Information (Notes S1–S7, Tables S9–S16) |
| `supp_body.tex` | generated from `../SUPPLEMENTARY.md` by `../md2tex.py`; do not edit by hand |
| `cover_letter.tex` | cover letter to the editors |
| `figures/fig1–6.pdf` | the six main display items, authored at final size |

Supplementary Data (Tables S1–S8, S14–S16) are comma-separated files in
`../supplementary/`. They are regenerated from the analysis outputs by
`../make_supplementary.py`; the figures likewise by `../make_figures.py`.

## Conformance to Nature Methods requirements

| requirement | status |
|---|---|
| Article format | main text 3,506 words (guideline ~3,000 — see note below) |
| Abstract ≤ 150 words | **151** by a strict count that treats `Lean~4` as two tokens; 150 by the usual convention |
| Display items ≤ 6 | **6** figures, no tables in the main text |
| Figure width | authored at 180 mm (double column) at final size; minimum type ~5.8 pt |
| Figures as vector PDF | yes; PNG copies in `../figures/` |
| Methods section | separate, 1,869 words |
| Double spacing | `setspace` |
| Continuous line numbers | `lineno` |
| Page numbers | default `article` footer |
| Data availability statement | yes |
| Code availability statement | yes |
| Author contributions | yes |
| Competing interests | yes |
| Acknowledgements | yes |
| Statistics reporting | exact *n*, test named, two-sided, correction and family size given at every *p* |
| Reporting Summary | **to be completed by the author** — see below |

**Main text length.** 3,506 words against a ~3,000 guideline. Nature Methods
treats this as a target rather than a hard cap at initial submission. If the
editor asks for a cut, the cheapest 500 words are the operating-guarantee
subsection (fully covered by Supplementary Note and Fig. 5) and the last two
paragraphs of the Discussion.

**Venue.** This package is prepared for *Nature Methods*, but the manuscript as
it now stands makes a claim about benchmarking in general — with ImageNet,
CIFAR-100 and eight other benchmarks instantiated alongside CAID — and is
plausibly above that bar. The recommended sequence is a presubmission enquiry to
*Nature* first (free, typically under a week, abstract plus a paragraph), then
*Nature Methods* or *Nature Machine Intelligence*. The enquiry should lead with
the capacity table, not with the disorder predictor.

**Reporting Summary.** Nature Portfolio requires the *Reporting Summary* PDF
(editorial policy checklist) with every submission. Download the current form
from
<https://www.nature.com/documents/nr-reporting-summary.pdf>
and complete it. For this manuscript the answers are almost all in the Methods:

- *Sample size* — no sample size was chosen; every target of every public CAID
  reference is used (233, 178, 49, 42 and 31 two-class targets on CAID3), and the
  temporal holdout is every qualifying PDB release after the cutoff (186 chains
  after homology filtering, from 1,916).
- *Data exclusions* — pre-specified and stated: targets carrying one class
  contribute no ordered pair; methods that decline any target are ineligible;
  chains homologous to training at ≥40% identity are removed. No post-hoc
  exclusion.
- *Replication* — CAID2 is an independent round sharing one protein with CAID3;
  the temporal holdout is independent of both. Both replicate the CAID3
  placement.
- *Randomization / blinding* — not applicable (no experimental groups). Splits
  are randomised where stated (12 splits for operating cost, 50 for the screen)
  and the seed is fixed in code.
- *Statistical parameters* — every test is two-sided; exact *n*, the test used,
  the family and the Holm correction are given wherever a *p*-value appears.
  Bootstraps use 10,000 resamples with the Davison–Hinkley correction.
- *Software* — Python 3.12, PyTorch, scikit-learn, SciPy, NumPy, Matplotlib;
  BLAST+ for homology filtering; Lean 4 with Mathlib for the formal development.
  Versions are pinned in the repository.

## Before you submit

1. **Affiliation.** `main.tex` currently carries only a correspondence address.
   Add the institutional affiliation.
2. **ORCID.** Nature Portfolio requires an ORCID for the corresponding author;
   add it in the submission system.
3. **Acknowledgements.** Confirm the wording of the computing acknowledgement
   with your institution's requirements (grant number, if any).
4. **Repository.** The Code Availability statement points at "the project
   repository". Replace with the public URL and a DOI (Zenodo) before the
   manuscript goes out; editors increasingly ask for the archived snapshot at
   submission rather than at acceptance.
5. **Lean development.** Consider depositing it with the code, and citing the
   Mathlib version, since the axiom claim (`propext`, `Classical.choice`,
   `Quot.sound`) is checkable only against a specific toolchain.
6. **Reporting Summary.** Complete and attach, as above.

## Regenerating everything

From the repository root:

```bash
python paper/make_figures.py        # six PDFs + PNGs, from the analysis JSONs
python paper/make_supplementary.py  # Supplementary Data CSVs
python paper/md2tex.py paper/SUPPLEMENTARY.md paper/latex/supp_body.tex
cp paper/figures/figure1_decomposition.pdf paper/latex/figures/fig1.pdf   # etc.
```

Every panel reads the JSON its compute job wrote, and the two panels that quote
a committed table parse it rather than restating it, so a figure cannot drift
from the number it illustrates.
