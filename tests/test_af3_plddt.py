"""Tests for colab/af3_plddt.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from colab.af3_plddt import (
    find_af3_output_pair,
    load_af3_plddt_for_protein,
    load_af3_plddt_from_files,
    parse_plddt_from_confidences_json,
    parse_plddt_from_mmcif,
)

SAMPLE_MMCIF = """\
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N MET A 1 1.0 2.0 3.0 1.00 10.00
ATOM 2 C CA MET A 1 1.1 2.1 3.1 1.00 42.00
ATOM 3 N N ALA A 2 2.0 3.0 4.0 1.00 10.00
ATOM 4 C CA ALA A 2 2.1 3.1 4.1 1.00 85.00
ATOM 5 N N GLY A 3 3.0 4.0 5.0 1.00 10.00
ATOM 6 C CA GLY A 3 3.1 4.1 5.1 1.00 30.00
"""

SAMPLE_CONFIDENCES = {
    "atom_plddts": [10.0, 42.0, 10.0, 85.0, 10.0, 30.0],
}


class TestParseMmcif:
    def test_ca_plddt_and_sequence(self):
        plddt, seq = parse_plddt_from_mmcif(SAMPLE_MMCIF)
        assert seq == "MAG"
        assert list(plddt) == pytest.approx([42.0, 85.0, 30.0])

    def test_confidences_json_path(self):
        plddt, seq = parse_plddt_from_confidences_json(
            SAMPLE_CONFIDENCES, SAMPLE_MMCIF,
        )
        assert seq == "MAG"
        assert list(plddt) == pytest.approx([42.0, 85.0, 30.0])


class TestFindOutputs:
    def test_finds_job_folder(self, tmp_path):
        job = tmp_path / "P12345"
        job.mkdir()
        conf = job / "P12345_confidences.json"
        cif = job / "P12345_model.cif"
        conf.write_text(json.dumps(SAMPLE_CONFIDENCES))
        cif.write_text(SAMPLE_MMCIF)

        pair = find_af3_output_pair(str(tmp_path), "prot1", "P12345")
        assert pair is not None
        assert pair[0] == str(conf)
        assert pair[1] == str(cif)


class TestLoadForProtein:
    def test_cache_and_align(self, tmp_path):
        job = tmp_path / "outputs" / "P12345"
        job.mkdir(parents=True)
        (job / "P12345_confidences.json").write_text(json.dumps(SAMPLE_CONFIDENCES))
        (job / "P12345_model.cif").write_text(SAMPLE_MMCIF)

        cache = tmp_path / "cache"
        plddt = load_af3_plddt_for_protein(
            "prot1", "MAG", str(tmp_path / "outputs"), uniprot_acc="P12345",
            cache_dir=str(cache),
        )
        assert plddt is not None
        assert len(plddt) == 3
        assert (cache / "prot1.json").exists()

        plddt2 = load_af3_plddt_for_protein(
            "prot1", "MAG", str(tmp_path / "outputs"), uniprot_acc="P12345",
            cache_dir=str(cache),
        )
        assert list(plddt2) == list(plddt)


class TestLoadFromFiles:
    def test_direct_load(self, tmp_path):
        conf = tmp_path / "x_confidences.json"
        cif = tmp_path / "x_model.cif"
        conf.write_text(json.dumps(SAMPLE_CONFIDENCES))
        cif.write_text(SAMPLE_MMCIF)
        plddt, seq = load_af3_plddt_from_files(str(conf), str(cif))
        assert seq == "MAG"
        assert len(plddt) == 3
