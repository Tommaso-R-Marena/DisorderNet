"""Tests for colab/af_plddt.py (network mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from colab.af_plddt import (
    align_plddt_to_sequence,
    fetch_plddt_for_uniprot,
    parse_plddt_from_pdb,
    plddt_to_disorder_score,
)

SAMPLE_PDB = """\
ATOM      1  N   MET A   1      12.000  15.000  20.000  1.00  0.00           N
ATOM      2  CA  MET A   1      12.500  15.500  20.500  1.00 42.00           C
ATOM      3  C   MET A   1      13.000  16.000  21.000  1.00  0.00           C
ATOM      4  N   ALA A   2      14.000  17.000  22.000  1.00  0.00           N
ATOM      5  CA  ALA A   2      14.500  17.500  22.500  1.00 85.00           C
ATOM      6  N   GLY A   3      15.000  18.000  23.000  1.00  0.00           N
ATOM      7  CA  GLY A   3      15.500  18.500  23.500  1.00 30.00           C
"""


class TestParsePdb:
    def test_extracts_ca_plddt(self):
        plddt, seq = parse_plddt_from_pdb(SAMPLE_PDB)
        assert seq == "MAG"
        assert list(plddt) == pytest.approx([42.0, 85.0, 30.0])


class TestAlign:
    def test_exact_match(self):
        plddt = np.array([40.0, 80.0, 35.0], dtype=np.float32)
        out = align_plddt_to_sequence(plddt, "MAG", "MAG")
        assert list(out) == pytest.approx([40.0, 80.0, 35.0])

    def test_substring(self):
        plddt = np.array([10.0, 40.0, 80.0, 35.0, 90.0], dtype=np.float32)
        out = align_plddt_to_sequence(plddt, "XMAGY", "MAG")
        assert list(out) == pytest.approx([40.0, 80.0, 35.0])

    def test_fail_returns_none(self):
        plddt = np.array([40.0, 80.0], dtype=np.float32)
        assert align_plddt_to_sequence(plddt, "AC", "XY") is None


class TestDisorderScore:
    def test_inverse_plddt(self):
        scores = plddt_to_disorder_score(np.array([100.0, 50.0, 0.0]))
        assert list(scores) == pytest.approx([0.0, 0.5, 1.0])


class TestFetchMocked:
    def test_fetch_caches_and_aligns(self, tmp_path):
        meta = {
            "pdbUrl": "http://example.com/model.pdb",
            "uniprotSequence": "MAG",
        }
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = SAMPLE_PDB
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        with patch("colab.af_plddt.fetch_afdb_metadata", return_value=meta):
            plddt = fetch_plddt_for_uniprot(
                "P12345", "MAG", cache_dir=str(tmp_path), session=mock_session,
            )
        assert plddt is not None
        assert len(plddt) == 3
        assert (tmp_path / "P12345.json").exists()

        # second call uses cache
        with patch("colab.af_plddt.fetch_afdb_metadata") as mock_meta:
            plddt2 = fetch_plddt_for_uniprot(
                "P12345", "MAG", cache_dir=str(tmp_path), session=mock_session,
            )
            mock_meta.assert_not_called()
        assert list(plddt2) == list(plddt)
