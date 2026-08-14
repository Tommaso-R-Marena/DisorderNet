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


class TestDataProvenanceIsPinned:
    """The inputs are not in the repo; their identity has to be."""

    @staticmethod
    def _doc():
        p = os.path.join(REPO, "results", "caid3", "DATA_PROVENANCE.md")
        assert os.path.isfile(p), "DATA_PROVENANCE.md must exist"
        return open(p).read()

    def test_every_reference_has_a_checksum(self):
        text = self._doc()
        for task in ("disorder_pdb", "disorder_nox", "binding", "binding_idr",
                     "linker"):
            assert f"`{task}.fasta`" in text, task
        # five md5s, one per reference
        assert len(re.findall(r"`[0-9a-f]{32}`", text)) >= 5

    def test_the_disorder_pdb_checksum_matches_the_one_we_verified(self):
        assert "6feaff35263e7fd4a3f03640c23786fe" in self._doc()

    def test_the_compositions_match_the_module(self):
        from colab.caid3_official import EXPECTED

        text = self._doc()
        for task, (n, pos, neg, und) in EXPECTED.items():
            for value in (n, pos, neg):
                assert f"{value:,}" in text or str(value) in text, (
                    f"{task}: {value} missing from DATA_PROVENANCE.md")

    def test_the_wrong_dataset_is_called_out(self):
        """The API's plain "CAID3" is a different 185-protein set that does not
        reproduce the leaderboard. Scoring against it answers another question
        silently, so the document has to warn about it."""
        text = self._doc()
        assert "CAID3 v3" in text and "185" in text

    def test_the_leader_reproduction_table_is_present(self):
        text = self._doc()
        for value in ("0.9552", "0.8855", "0.7760", "0.6407", "0.8985"):
            assert value in text, value


class TestMethodologyIsFixed:
    """The methodology is only fixed if the code implements what it states."""

    @staticmethod
    def _doc():
        p = os.path.join(REPO, "results", "caid3", "METHODOLOGY.md")
        assert os.path.isfile(p), "METHODOLOGY.md must exist"
        return open(p).read()

    def test_both_ranking_fields_are_required(self):
        text = self._doc()
        assert "Report both fields, never one" in text
        assert "full-coverage" in text

    def test_the_evaluator_computes_both_ranking_fields(self):
        src = open(os.path.join(REPO, "rockfish",
                                "eval_caid3_official.py")).read()
        assert "rank_full_coverage" in src
        assert "n_full_coverage_entrants" in src

    def test_the_structural_baseline_is_mandatory(self):
        from rockfish.eval_caid3_official import (PRIMARY_OPPONENTS,
                                                  STRUCTURAL_BASELINE)

        assert STRUCTURAL_BASELINE == "AlphaFold-rsa"
        assert STRUCTURAL_BASELINE in PRIMARY_OPPONENTS
        assert "AlphaFold-rsa" in self._doc()

    def test_full_coverage_is_required_of_us(self):
        src = open(os.path.join(REPO, "rockfish",
                                "eval_caid3_official.py")).read()
        assert "Partial coverage inflates the score" in src

    def test_the_paired_test_resamples_proteins(self):
        from colab.caid3_official import paired_bootstrap

        doc = paired_bootstrap.__doc__ or ""
        assert "resampling proteins rather than residues" in doc

    def test_the_documented_ranks_match_the_committed_results(self):
        """METHODOLOGY.md's summary table must agree with the run it describes."""
        import json

        path = os.path.join(REPO, "results", "caid3",
                            "mt_full_caid3_official.json")
        if not os.path.isfile(path):
            pytest.skip("mt_full results not committed")
        data = json.load(open(path))
        text = self._doc()
        for task, expected_rank in (("disorder_pdb", 1), ("linker", 2),
                                    ("disorder_nox", 13), ("binding", 11),
                                    ("binding_idr", 24)):
            row = data.get(task)
            if not row or "rank" not in row:
                continue
            assert row["rank"] == expected_rank, (
                f"{task}: results say rank {row['rank']}, "
                f"METHODOLOGY.md says {expected_rank}")
            assert f"{row['ours']['auc']:.4f}" in text, task
