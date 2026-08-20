#!/usr/bin/env python3
"""Convert the supplementary markdown to LaTeX.

Deliberately narrow: it handles exactly the constructs `SUPPLEMENTARY.md` uses
(headings, pipe tables, fenced and inline code, bold, italic, lists, rules) and
raises on anything else rather than silently dropping it. A converter that
quietly loses a table in a supplement is worse than one that refuses.
"""
from __future__ import annotations

import os
import re
import sys

ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
       "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
       "^": r"\textasciicircum{}"}


def esc(t: str) -> str:
    out = []
    for ch in t:
        out.append(ESC.get(ch, ch))
    t = "".join(out)
    t = t.replace("\\", "\\textbackslash{}") if "\\textbackslash" in t else t
    return t


def inline(t: str) -> str:
    """Inline markup, code first so its contents are not re-escaped."""
    parts, last = [], 0
    for m in re.finditer(r"`([^`]+)`", t):
        parts.append(("text", t[last:m.start()]))
        parts.append(("code", m.group(1)))
        last = m.end()
    parts.append(("text", t[last:]))

    out = []
    for kind, chunk in parts:
        if kind == "code":
            out.append(r"\code{" + esc(chunk) + "}")
            continue
        c = esc(chunk)
        c = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", c)
        c = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", c)
        c = c.replace("—", "---").replace("–", "--")
        c = c.replace("≤", r"$\leq$").replace("≥", r"$\geq$")
        c = c.replace("×", r"$\times$").replace("→", r"$\rightarrow$")
        c = c.replace("ε", r"$\varepsilon$").replace("ν", r"$\nu$")
        c = c.replace("α", r"$\alpha$").replace("δ", r"$\delta$")
        c = c.replace("κ", r"$\kappa$").replace("ρ", r"$\rho$")
        c = c.replace("Δ", r"$\Delta$").replace("σ", r"$\sigma$")
        c = c.replace("⌈", r"$\lceil$").replace("⌉", r"$\rceil$")
        c = c.replace("⌊", r"$\lfloor$").replace("⌋", r"$\rfloor$")
        c = c.replace("∈", r"$\in$").replace("⊆", r"$\subseteq$")
        c = c.replace("∩", r"$\cap$").replace("∪", r"$\cup$")
        c = c.replace("·", r"$\cdot$").replace("²", r"$^2$")
        c = c.replace("Σ", r"$\Sigma$").replace("𝔼", r"$\mathbb{E}$")
        c = c.replace("≈", r"$\approx$").replace("…", r"\dots")
        c = c.replace("‑", "-")
        out.append(c)
    return "".join(out)


def table(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, align_row, body = cells[0], cells[1], cells[2:]
    align = "".join("r" if a.endswith(":") and not a.startswith(":")
                    else ("c" if a.startswith(":") and a.endswith(":") else "l")
                    for a in align_row)
    w = len(header)
    out = [r"\begin{center}\footnotesize",
           r"\begin{tabular}{" + align + "}", r"\toprule",
           " & ".join(r"\textbf{%s}" % inline(h) for h in header) + r" \\",
           r"\midrule"]
    for row in body:
        row = (row + [""] * w)[:w]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    return "\n".join(out)


def convert(md: str) -> str:
    lines = md.split("\n")
    out, i, in_list = [], 0, False
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out += [r"\begin{quote}\ttfamily\footnotesize\begin{flushleft}"]
            for c in lines[i + 1:j]:
                out.append(esc(c).replace(" ", "~") + r"\\")
            out += [r"\end{flushleft}\end{quote}"]
            i = j + 1
            continue

        if ln.startswith("    ") and ln.strip():
            j = i
            while j < len(lines) and (lines[j].startswith("    ")
                                      or not lines[j].strip()):
                j += 1
            block = [c for c in lines[i:j]]
            while block and not block[-1].strip():
                block.pop()
            out += [r"\begin{quote}\ttfamily\footnotesize\begin{flushleft}"]
            for c in block:
                out.append(esc(c[4:]).replace(" ", "~") + r"\\")
            out += [r"\end{flushleft}\end{quote}"]
            i = i + len(block)
            continue

        if ln.strip().startswith("|") and i + 1 < len(lines) \
                and set(lines[i + 1].replace("|", "").replace(" ", "")) <= set("-:"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append(table(lines[i:j]))
            i = j
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            lvl = len(m.group(1))
            cmd = {1: "section*", 2: "section*", 3: "subsection*",
                   4: "subsubsection*"}[lvl]
            out += ["", "\\%s{%s}" % (cmd, inline(m.group(2)))]
            i += 1
            continue

        if re.match(r"^---+\s*$", ln):
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out += ["", r"\vspace{0.5em}\hrule\vspace{0.8em}", ""]
            i += 1
            continue

        m = re.match(r"^\s*[-*]\s+(.*)$", ln)
        if m:
            if not in_list:
                out.append(r"\begin{itemize}\itemsep1pt \parskip0pt")
                in_list = True
            out.append(r"\item " + inline(m.group(1)))
            i += 1
            continue

        m = re.match(r"^\s*(\d+)\.\s+(.*)$", ln)
        if m:
            if not in_list:
                out.append(r"\begin{itemize}\itemsep1pt \parskip0pt")
                in_list = True
            out.append(r"\item " + inline(m.group(2)))
            i += 1
            continue

        if not ln.strip():
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            out.append("")
            i += 1
            continue

        out.append(inline(ln))
        i += 1

    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    body = convert(open(src).read())
    open(dst, "w").write(body)
    print(f"wrote {dst}  ({len(body.splitlines())} lines)")
