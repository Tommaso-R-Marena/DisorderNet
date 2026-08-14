"""Every CAID3 figure in the README must be one we actually measured.

The README carried `ESMDisPred = 0.895` as "CAID3 SOTA" in five places. It is
not SOTA, and 0.895 is not even its Disorder-PDB score — it is its Disorder-NOX
score. The same cross-wiring ran through the rest of the block:

    AF2-pLDDT   cited 0.770  — that is its Disorder-NOX score; on
                              Disorder-PDB it scores 0.9342
    AF3-pLDDT   cited 0.747  — Disorder-PDB is 0.9324
    flDPnn3a    cited 0.871  — Disorder-PDB is 0.9006, and it ranks 44th

Figures taken from a paper's table for one benchmark had been pasted in as
though they described another, and a reader comparing our numbers against them
would have drawn a conclusion off by more than 0.15 AUC. Since every entrant's
raw predictions are now available, no CAID3 figure needs transcribing — and none
should be trusted that has not been recomputed.
"""

from __future__ import annotations

import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(REPO, "README.md")

CACHE = os.environ.get("CAID3_OFFICIAL_DIR", "")
PREDS = os.environ.get("CAID3_PREDICTIONS_DIR", "")
has_data = bool(CACHE) and bool(PREDS) and os.path.isdir(CACHE) and os.path.isdir(PREDS)

#: (method, task, auc, rank) as measured. The README must agree with these.
DOCUMENTED = [
    ("PUNCH2", "disorder_pdb", 0.9552, 1),
    ("AlphaFold-rsa", "disorder_pdb", 0.9498, 3),
    ("ESMDisPred-2PDB", "disorder_pdb", 0.9366, 8),
    ("AlphaFold-pLDDT", "disorder_pdb", 0.9342, 13),
    ("AlphaFold3-pLDDT", "disorder_pdb", 0.9324, 14),
    ("flDPnn3a", "disorder_pdb", 0.9006, 44),
]

#: Figures that were wrong and must never reappear as CAID3 Disorder-PDB claims.
RETIRED = {
    "0.895": "ESMDisPred's Disorder-NOX score, cited as its Disorder-PDB score",
    "0.770": "AlphaFold-pLDDT's Disorder-NOX score (Disorder-PDB is 0.9342)",
    "0.747": "AF3-pLDDT, matching no measured CAID3 Disorder-PDB figure",
}


def readme():
    return open(README).read()


class TestRetiredFiguresStayRetired:
    @pytest.mark.parametrize("figure,why", sorted(RETIRED.items()))
    def test_no_retired_figure_is_presented_as_a_caid3_result(self, figure, why):
        """The corrections section may discuss them; a results table may not."""
        for line in readme().splitlines():
            body = line.strip()
            if figure not in body or not body.startswith("|"):
                continue
            assert "CAID3" not in body or "was wrong" in body.lower(), (
                f"README table row cites {figure} alongside CAID3: {body!r}\n"
                f"  {figure} is {why}."
            )

    def test_the_correction_is_still_documented(self):
        """Deleting the retired numbers entirely would lose the record of the
        mistake, which is worth more than the tidiness."""
        text = readme()
        assert "0.895" in text, "the correction narrative should survive"
        assert "Disorder-NOX" in text


@pytest.mark.skipif(not has_data,
                    reason="set CAID3_OFFICIAL_DIR and CAID3_PREDICTIONS_DIR")
class TestDocumentedFiguresMatchMeasurement:
    @pytest.mark.parametrize("method,task,auc,rank", DOCUMENTED)
    def test_the_figure_is_what_we_measure(self, method, task, auc, rank):
        from colab.caid3_official import official_leaderboard

        board = official_leaderboard(task, CACHE, PREDS)
        row = next((r for r in board if r["method"] == method), None)
        assert row is not None, f"{method} not among the entrants"
        assert row["auc"] == pytest.approx(auc, abs=0.0005), (
            f"{method} on {task}: measured {row['auc']:.4f}, documented {auc}")
        assert row["rank"] == rank, (
            f"{method} on {task}: measured rank {row['rank']}, documented {rank}")

    @pytest.mark.parametrize("method,task,auc,rank", DOCUMENTED)
    def test_the_readme_states_that_figure(self, method, task, auc, rank):
        text = readme()
        if method not in text:
            pytest.skip(f"{method} is not cited in the README")
        assert f"{auc:.4f}" in text or f"{auc:.3f}" in text, (
            f"{method} appears in the README but not with its measured "
            f"{auc:.4f} — check for a figure from another benchmark.")


class TestClaimsCarryTheirStatus:
    """A rank and a p-value are different kinds of claim and must read that way."""

    def test_the_headline_result_is_present(self):
        assert "0.9603" in readme()

    def test_the_exploratory_status_is_stated(self):
        text = readme()
        assert "exploratory" in text.lower()
        assert "PREREGISTRATION" in text

    def test_the_unadjusted_significance_is_not_claimed_bare(self):
        """Holm-adjusted values must appear wherever the raw ones do."""
        text = readme()
        assert "0.143" in text and "0.084" in text, (
            "the adjusted p-values must accompany the raw ones")

    def test_the_fusion_caveat_is_stated(self):
        text = readme()
        assert "0.938" in text, (
            "the README must record that our own rsa scores 0.938 where CAID's "
            "scores 0.950, so the fused row is not self-contained")
