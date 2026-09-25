#!/usr/bin/env python3
"""Render the manuscript as one self-contained page, figures embedded.

Two things this does that a plain markdown render would not.

**The proof rail.** Every section of the Results rests on a different kind of
evidence — a machine-checked theorem, a measurement, or a pre-registered test —
and the paper's whole argument depends on the reader keeping those apart. The
rail in the left gutter names, for each section, the theorem behind it and
whether that theorem is proved or pending. It encodes the distinction the text
keeps making rather than decorating the margin.

**Figures inline.** The PNGs are embedded as data URIs because the artifact CSP
blocks every external host, and a paper without its figures is not reviewable.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
FIGS = os.path.join(PAPER, "figures")

#: section heading -> (rail label, rail detail, status)
#: status drives the mark: "proved" a machine-checked theorem, "measured" an
#: empirical result with no theorem behind it, "pending" a theorem in progress.
RAIL = {
    "CAID's headline statistic is 97–99.5% a between-protein question, and the split is not a metaphor":
        ("auc_within_strictMono_invariant", "invariance under per-protein recalibration", "proved"),
    "The declared winner is 21st–25th at the residue-level question, and each inversion is certified":
        ("inversion_requires_between_gap", "each inversion is a certificate", "proved"),
    "A benchmark can order at most ⌈1/2ε⌉ methods, and CAID3 admits seventeen times that":
        ("card_le_benchCapacity", "+ benchCapacity_attained, unresolvable_pair", "proved"),
    "Pairwise scoring makes the noise second-order, and the capacity rises sixfold":
        ("discordant_eq_flip_product", "+ card_comparablePairs, both exact", "proved"),
    "A protocol, and the field re-scored under it":
        ("auc_target_strictMono_invariant", "per-target form, formalisation in progress", "pending"),
    "The extra resolution is not noise":
        ("57 shared entrants", "CAID3 → held-out CAID2", "measured"),
    "An operating guarantee, and the price of it":
        ("PredictionSets.validity_is_free", "+ Calibration.risk_decomposition", "proved"),
    "DisorderNet":
        ("PREREGISTRATION.md", "primary family fixed before the run", "measured"),
    "The recalibration decomposition is NP-hard to compute":
        ("Complexity/biasThreshold_hard", "NP → circuit SAT → Tseitin → IS → LOP", "proved"),
    "Reporting a screen: the unit of testing is a design choice too":
        ("selfConsistent_fdr_le_harmonic", "+ bh_fdr_eq_harmonic, fdp_lift", "proved"),
}

FIGURE_FOR = {
    "Figure 1": "figure1_decomposition.png",
    "Figure 2": "figure2_capacity.png",
    "Figure 3": "figure3_rescored.png",
    "Figure 4": "figure4_model.png",
    "Figure 5": "figure5_operating.png",
    "Figure 6": "figure6_screen.png",
}


def data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        return f"data:{mime};base64,{base64.b64encode(fh.read()).decode()}"


def render():
    md = open(os.path.join(PAPER, "MANUSCRIPT.md")).read()

    # The figure legends live in a trailing section; lift each one to sit under
    # its own figure rather than making the reader scroll to find it.
    body, legends = md.split("## Figures\n", 1)
    legend = {}
    for block in re.split(r"\n\n(?=\*\*Figure )", legends.strip()):
        m = re.match(r"\*\*(Figure \d+) — (.+?)\.\*\*\s*(.*)", block, re.S)
        if m:
            legend[m.group(1)] = (m.group(2), m.group(3).strip())

    html = markdown.markdown(
        body, extensions=["tables", "attr_list", "sane_lists"])
    # the masthead carries the title and byline; drop the markdown's own copy
    html = re.sub(r"^.*?<h2>Abstract</h2>", "<h2>Abstract</h2>", html,
                  flags=re.S)

    # Insert each figure where the text first calls it.
    for num, fname in FIGURE_FOR.items():
        path = os.path.join(FIGS, fname)
        if not os.path.isfile(path):
            continue
        title, caption = legend.get(num, ("", ""))
        caption = markdown.markdown(caption).replace("<p>", "").replace("</p>", "")
        block = (
            f'<figure class="fig" id="{num.lower().replace(" ", "")}">'
            f'<div class="fig-frame"><img src="{data_uri(path)}" '
            f'alt="{num}. {title}"></div>'
            f'<figcaption><span class="fig-num">{num}</span>'
            f'<span class="fig-title">{title}.</span> {caption}</figcaption>'
            f"</figure>")
        # place after the paragraph containing the first bold reference
        key = f"<strong>Fig. {num.split()[1]}"
        i = html.find(key)
        if i == -1:
            html += block
            continue
        j = html.find("</p>", i)
        j = html.find("</table>", i) if j == -1 else j
        cut = html.find(">", j) + 1
        html = html[:cut] + block + html[cut:]

    # Wrap each h3 in a section carrying its rail.
    def rail_for(match):
        text = re.sub(r"<[^>]+>", "", match.group(1))
        entry = RAIL.get(text)
        if not entry:
            return match.group(0)
        name, detail, status = entry
        mark = {"proved": "proved", "pending": "in progress",
                "measured": "measured"}[status]
        return (f'<div class="railed"><aside class="rail rail-{status}">'
                f'<span class="rail-mark">{mark}</span>'
                f'<code class="rail-name">{name}</code>'
                f'<span class="rail-detail">{detail}</span></aside>'
                f"<h3>{match.group(1)}</h3></div>")

    html = re.sub(r"<h3>(.*?)</h3>", rail_for, html, flags=re.S)
    # The abstract is the one block that gets its own surface: it is the only
    # part most readers will read, and it is not a section of the argument.
    a = html.find("<h2>Abstract</h2>")
    b = html.find("<h2>Introduction</h2>")
    if a != -1 and b != -1:
        html = (html[:a] + '<section class="abstract">' + html[a:b]
                + "</section>" + html[b:])
    html = html.replace("<table>", '<div class="tbl"><table>')
    html = html.replace("</table>", "</table></div>")
    return html


PAGE = """<title>The Capacity of a Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
<style>
:root {
  --paper:      #FBFCFC;
  --surface:    #F1F5F4;
  --ink:        #12181A;
  --ink-soft:   #4A5A5C;
  --ink-faint:  #7E8E8F;
  --rule:       #DCE5E4;
  --accent:     #0E6F68;
  --accent-dim: #D5E9E6;
  --warn:       #A44E08;
  --shadow:     0 1px 2px rgba(18,24,26,.05), 0 8px 24px rgba(18,24,26,.05);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:      #0E1315;
    --surface:    #161D1F;
    --ink:        #E2E9E8;
    --ink-soft:   #9BADAD;
    --ink-faint:  #6C7E7E;
    --rule:       #24302F;
    --accent:     #58CFC2;
    --accent-dim: #16332F;
    --warn:       #E5A445;
    --shadow:     0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --paper:      #0E1315;
  --surface:    #161D1F;
  --ink:        #E2E9E8;
  --ink-soft:   #9BADAD;
  --ink-faint:  #6C7E7E;
  --rule:       #24302F;
  --accent:     #58CFC2;
  --accent-dim: #16332F;
  --warn:       #E5A445;
  --shadow:     0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 400 17px/1.62 "Source Serif 4", Georgia, "Times New Roman", serif;
  -webkit-font-smoothing: antialiased;
}
.wrap {
  max-width: 1240px;
  margin: 0 auto;
  padding: 0 24px 96px;
  display: grid;
  grid-template-columns: 1fr;
}
main { max-width: 40rem; margin: 0 auto; }

/* ── masthead ─────────────────────────────────────────────────────────── */
header.mast {
  max-width: 46rem; margin: 0 auto; padding: 72px 0 40px;
  border-bottom: 1px solid var(--rule);
}
.kicker {
  font: 500 11px/1 "IBM Plex Sans", system-ui, sans-serif;
  letter-spacing: .16em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 20px;
}
h1 {
  font: 600 clamp(28px, 4.2vw, 42px)/1.15 "IBM Plex Sans", system-ui, sans-serif;
  letter-spacing: -.018em; text-wrap: balance; margin: 0 0 22px;
}
.byline {
  font: 400 15px/1.5 "IBM Plex Sans", system-ui, sans-serif;
  color: var(--ink-soft);
}
.byline b { font-weight: 600; color: var(--ink); }

/* ── headings ─────────────────────────────────────────────────────────── */
h2 {
  font: 600 13px/1 "IBM Plex Sans", system-ui, sans-serif;
  letter-spacing: .17em; text-transform: uppercase;
  color: var(--ink-faint);
  margin: 76px 0 28px; padding-bottom: 10px;
  border-bottom: 1px solid var(--rule);
}
h3 {
  font: 600 22px/1.3 "IBM Plex Sans", system-ui, sans-serif;
  letter-spacing: -.01em; text-wrap: balance;
  margin: 0 0 18px; color: var(--ink);
}
.railed { margin: 52px 0 0; position: relative; }
.railed h3 { margin-top: 0; }

/* ── the proof rail ───────────────────────────────────────────────────── */
.rail {
  display: flex; flex-direction: column; gap: 3px;
  margin: 0 0 14px; padding-left: 13px;
  border-left: 2px solid var(--accent);
}
.rail-warn, .rail-pending { border-left-color: var(--warn); }
.rail-measured { border-left-color: var(--ink-faint); }
.rail-mark {
  font: 500 10px/1 "IBM Plex Sans", system-ui, sans-serif;
  letter-spacing: .14em; text-transform: uppercase; color: var(--accent);
}
.rail-pending .rail-mark { color: var(--warn); }
.rail-measured .rail-mark { color: var(--ink-faint); }
.rail-name {
  font: 400 12.5px/1.4 "IBM Plex Mono", ui-monospace, monospace;
  color: var(--ink); word-break: break-word;
}
.rail-detail {
  font: 400 12px/1.4 "IBM Plex Sans", system-ui, sans-serif;
  color: var(--ink-faint);
}
@media (min-width: 1200px) {
  .railed { display: grid; grid-template-columns: 13rem 1fr; gap: 32px;
            align-items: start; margin-left: -15rem; }
  .railed > .rail { margin: 6px 0 0; text-align: right;
                    border-left: 0; border-right: 2px solid var(--accent);
                    padding-left: 0; padding-right: 13px; align-items: flex-end; }
  .railed > .rail-pending { border-right-color: var(--warn); }
  .railed > .rail-measured { border-right-color: var(--ink-faint); }
}

/* ── body ─────────────────────────────────────────────────────────────── */
p { margin: 0 0 1.05em; }
strong { font-weight: 600; }
code {
  font: 400 .875em/1.5 "IBM Plex Mono", ui-monospace, monospace;
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 3px; padding: .08em .34em;
}
pre {
  background: var(--surface); border: 1px solid var(--rule);
  border-left: 2px solid var(--accent);
  border-radius: 4px; padding: 16px 18px; overflow-x: auto;
  font: 400 13.5px/1.6 "IBM Plex Mono", ui-monospace, monospace;
  margin: 0 0 1.3em;
}
pre code { background: none; border: 0; padding: 0; font-size: inherit; }
hr { border: 0; border-top: 1px solid var(--rule); margin: 56px 0; }
ul, ol { margin: 0 0 1.05em; padding-left: 1.3em; }
li { margin-bottom: .4em; }
li::marker { color: var(--ink-faint); }
a { color: var(--accent); text-underline-offset: 2px; }

/* ── tables ───────────────────────────────────────────────────────────── */
.tbl { overflow-x: auto; margin: 0 0 1.6em; }
table {
  border-collapse: collapse; width: 100%;
  font: 400 14px/1.45 "IBM Plex Sans", system-ui, sans-serif;
  font-variant-numeric: tabular-nums;
}
th {
  text-align: left; font-weight: 600; color: var(--ink-faint);
  font-size: 11px; letter-spacing: .08em; text-transform: uppercase;
  padding: 0 14px 8px 0; border-bottom: 1px solid var(--ink-faint);
  white-space: nowrap;
}
td { padding: 8px 14px 8px 0; border-bottom: 1px solid var(--rule);
     vertical-align: top; }
th[align="right"], td[align="right"] { text-align: right; }
tbody tr:hover td { background: var(--surface); }
td strong { color: var(--accent); font-weight: 600; }

/* ── figures ──────────────────────────────────────────────────────────── */
.fig { margin: 40px 0 44px; }
@media (min-width: 1000px) { .fig { margin-left: -5.5rem; margin-right: -5.5rem; } }
.fig-frame {
  background: #fff; border: 1px solid var(--rule); border-radius: 5px;
  padding: 14px; box-shadow: var(--shadow); overflow-x: auto;
}
.fig-frame img { display: block; width: 100%; height: auto; min-width: 640px; }
figcaption {
  margin-top: 12px;
  font: 400 13.5px/1.55 "IBM Plex Sans", system-ui, sans-serif;
  color: var(--ink-soft);
}
.fig-num {
  font-weight: 600; color: var(--accent); letter-spacing: .04em;
  margin-right: .5em; text-transform: uppercase; font-size: 11.5px;
}
.fig-title { font-weight: 600; color: var(--ink); }

/* ── abstract ─────────────────────────────────────────────────────────── */
.abstract {
  background: var(--surface); border: 1px solid var(--rule);
  border-radius: 5px; padding: 26px 28px; margin: 34px 0 0;
}
.abstract h2 { margin-top: 0; }

:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
<header class="mast">
  <div class="kicker">Benchmark theory · intrinsic disorder · machine-checked proof</div>
  <h1>A benchmark cannot order more methods than its labels can distinguish</h1>
  <p class="byline"><b>Tommaso R. Marena</b> — capacity limits in CAID, and the
  protocol that escapes them. Every figure is regenerated from the analysis
  outputs by <code>paper/make_figures.py</code>; the Supplementary Information
  and its fourteen tables are in the repository.</p>
</header>
<main>
__BODY__
</main>
</div>
"""


def main():
    out = PAGE.replace("__BODY__", render())
    path = os.path.join(PAPER, "manuscript_web.html")
    with open(path, "w") as fh:
        fh.write(out)
    print(f"wrote {path}  {len(out) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
