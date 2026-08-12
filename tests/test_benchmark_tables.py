"""Tests for colab/benchmark_tables.py."""

from __future__ import annotations

from colab.benchmark_tables import (
    build_our_disprot_row,
    get_literature_table,
    get_our_disprot_table,
    print_matched_benchmark_report,
    rank_against_literature,
)


def test_literature_not_comparable():
    """Table A holds published CAID3 figures; Table B holds our DisProt CV.
    No Table A row is comparable to Table B — including the CAID3 Disorder-PDB
    rows, whose AUCs are on a different benchmark entirely. Marking them
    comparable would reintroduce the CV-vs-CAID3 category error."""
    rows = get_literature_table()
    assert all(not r["comparable_to_ours"] for r in rows)
    assert any("ESMDisPred" in r["method"] for r in rows)


def test_literature_carries_the_real_caid3_leader():
    """The table previously topped out at 'ESMDisPred 0.895, rank 1, CAID3
    SOTA'. ESMDisPred scores 0.937 on Disorder-PDB and PUNCH2 leads at 0.955,
    so that row set the bar ~0.06 AUC too low."""
    rows = {r["method"]: r for r in get_literature_table()}
    assert rows["PUNCH2"]["auc"] == 0.955
    assert rows["ESMDisPred-2PDB"]["auc"] == 0.937
    assert max(r["auc"] for r in rows.values()) == 0.955


def test_our_table_cpu_only():
    rows = get_our_disprot_table()
    assert len(rows) == 2
    assert rows[0]["status"] == "verified"
    assert rows[1]["status"] == "pending_full_run"


def test_our_table_with_gpu():
    rows = get_our_disprot_table(gpu_auc=0.87, gpu_ap=0.55)
    assert rows[1]["auc"] == 0.87
    assert rows[1]["status"] == "verified"


def test_rank_contextual():
    r = rank_against_literature(0.85)
    assert r["comparable"] is False
    assert "note" in r
    assert r["literature_rank_if_comparable"] >= 1


def test_matched_report_returns_dict(capsys):
    report = print_matched_benchmark_report(gpu_auc=0.84, gpu_ap=0.5, gpu_f1_max=0.55, gpu_mcc=0.4)
    assert "disclaimer" in report
    assert len(report["literature_reference"]) > 0
    out = capsys.readouterr().out
    assert "TABLE A" in out
    assert "TABLE B" in out
    assert "NOT head-to-head" in out


def test_build_gpu_row():
    row = build_our_disprot_row(0.88, ap=0.6, f1_max=0.57, mcc=0.45)
    assert row["auc"] == 0.88
    assert row["comparable_to_ours"] is True
