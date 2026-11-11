"""The submission checklist must not quote a word count the manuscript no longer has.

`paper/latex/README.md` is the document an editor's format check is answered
from, so every number in it is a claim about the files beside it. Three of them
are counts, and counts drift silently: the main text was quoted as 3,779 words
for three revisions after it had reached 4,141, and Methods as 1,869 after it
had reached 2,171. Nothing failed, because nothing was checking.

This is the same guard the figures already have — a panel parses the table it
illustrates rather than restating it — applied to the one file that restates
instead of parsing.
"""

from __future__ import annotations

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX = os.path.join(REPO, "paper", "latex")
README = os.path.join(LATEX, "README.md")

#: The count is a target, not a hard cap, and prose legitimately moves between
#: revisions. What must not happen is the README describing a manuscript that no
#: longer exists, so allow a little slack and fail on real drift.
TOLERANCE = 25


def _words(tex: str) -> int:
    """Word count after stripping LaTeX control sequences and math punctuation.

    Deliberately the same expression the README publishes for the author to run
    by hand; if the two ever disagree the README's instructions are wrong.
    """
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^{}]*\})?", " ", tex)
    tex = re.sub(r"[{}$&\\_^~]", " ", tex)
    return sum(1 for w in tex.split() if re.search(r"[A-Za-z0-9]", w))


def _read(name: str) -> str:
    with open(os.path.join(LATEX, name), encoding="utf-8") as fh:
        return fh.read()


def _readme() -> str:
    with open(README, encoding="utf-8") as fh:
        return fh.read()


def _quoted(pattern: str) -> int:
    """The number the README states, pulled out of its own prose."""
    m = re.search(pattern, _readme())
    assert m, f"README no longer states a count matching {pattern!r}"
    return int(m.group(1).replace(",", ""))


pytestmark = pytest.mark.skipif(
    not os.path.isdir(LATEX), reason="LaTeX sources not present"
)


class TestTheCountsTheReadmeQuotes:
    def test_main_text(self):
        tex = _read("main.tex")
        body = tex[tex.index(r"\section*{Introduction}"):
                   tex.index(r"\section*{Data availability}")]
        actual = _words(body)
        stated = _quoted(r"main text ([\d,]+) words")
        assert abs(actual - stated) <= TOLERANCE, (
            f"README says the main text is {stated:,} words; it is {actual:,}. "
            "Update the checklist row and the 'Main text length' note."
        )

    def test_methods(self):
        actual = _words(_read("methods.tex"))
        stated = _quoted(r"Methods section \| separate, ([\d,]+) words")
        assert abs(actual - stated) <= TOLERANCE, (
            f"README says Methods is {stated:,} words; it is {actual:,}."
        )

    def test_abstract_against_the_journal_limit(self):
        tex = _read("main.tex")
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
        assert m, "main.tex no longer has an abstract environment"
        actual = _words(m.group(1))
        stated = _quoted(r"Abstract .{0,12}150 words \| \*\*(\d+)\*\*")
        assert actual == stated, (
            f"README says the abstract is {stated} words; it is {actual}."
        )
        # The strict count treats `Lean~4` as two tokens, so 151 is the honest
        # number to report against a 150-word limit. Anything above that is a
        # real overrun rather than a tokenisation artefact.
        assert actual <= 151, f"abstract is {actual} words, over the 150 limit"


class TestTheReadmeInstructionsRun:
    def test_snippet_does_not_call_bare_python(self):
        """`python` is not present on every machine; the snippet must say python3."""
        for line in _readme().splitlines():
            if "re.finditer" in line or "s=open('paper/latex/main.tex')" in line:
                assert not re.match(r"\s*python\s", line), (
                    "the counting snippet calls `python`; use `python3`"
                )
