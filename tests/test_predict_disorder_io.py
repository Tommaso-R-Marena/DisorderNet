"""Tests for predict_disorder FASTA parsing (pure I/O, no ESM needed)."""
from __future__ import annotations

from predict_disorder import read_fasta


def test_read_fasta_multi(tmp_path):
    f = tmp_path / "p.fasta"
    f.write_text(">prot1 description here\nMDVFMK\nGLSKA\n>prot2\nKVFGRCEL\n")
    recs = read_fasta(str(f))
    assert recs == [("prot1", "MDVFMKGLSKA"), ("prot2", "KVFGRCEL")]


def test_read_fasta_single_no_trailing_newline(tmp_path):
    f = tmp_path / "p.fasta"
    f.write_text(">only\nACDEFG")
    assert read_fasta(str(f)) == [("only", "ACDEFG")]
