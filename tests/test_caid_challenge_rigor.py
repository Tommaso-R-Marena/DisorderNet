"""Tests for CAID leakage audit, challenge suite hooks, and OOM escalation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from colab.caid_leakage import (  # noqa: E402
    audit_train_vs_caid,
    filter_train_proteins,
)
from colab.caid_challenge import (  # noqa: E402
    parse_caid_targets_fasta,
    write_timings_csv,
)
from rockfish.publish_campaign import (  # noqa: E402
    JobState,
    OOM_ESCALATION,
    advance_campaign,
    apply_oom_escalation,
    new_campaign,
    phase_failed_oom,
)


class TestCaidLeakage:
    def test_id_overlap_flagged(self):
        train = [{"id": "DP0001", "sequence": "ACDE" * 20}]
        caid = [{"id": "DP0001", "sequence": "ACDE" * 20}]
        audit = audit_train_vs_caid(train, caid, min_identity=0.4)
        assert audit["n_id_overlap"] == 1
        assert audit["leak_free"] is False
        kept, meta = filter_train_proteins(train, audit)
        assert meta["n_removed"] == 1
        assert kept == []

    def test_homology_hit(self):
        base = "ACDEFGHIKLMNPQRSTVWY" * 5
        train = [{"id": "T1", "sequence": base}]
        # Nearly identical sequence, different id
        caid = [{"id": "C1", "sequence": base[:-1] + "A"}]
        audit = audit_train_vs_caid(train, caid, min_identity=0.9)
        assert audit["n_homology_hits"] >= 1
        assert "T1" in audit["flagged_train_ids"]

    def test_clean_when_distant(self):
        train = [{"id": "T1", "sequence": "A" * 80}]
        caid = [{"id": "C1", "sequence": "C" * 80}]
        audit = audit_train_vs_caid(train, caid, min_identity=0.4)
        assert audit["leak_free"] is True


class TestCaid4BlindParse:
    def test_sequence_only_fasta(self, tmp_path):
        p = tmp_path / "t.fasta"
        p.write_text(">P1\nACDEACDE\n>P2\nGGGGAAAA\n")
        prots = parse_caid_targets_fasta(str(p))
        assert len(prots) == 2
        assert prots[0]["id"] == "P1"
        assert prots[0]["labels"] == []

    def test_timings_csv(self, tmp_path):
        out = tmp_path / "timings.csv"
        write_timings_csv(str(out), [("P1", 12.5), ("P2", 8.0)], header_note="test")
        text = out.read_text()
        assert "sequence,milliseconds" in text
        assert "P1,12.500" in text


class TestOomEscalation:
    def test_escalation_ladder(self, monkeypatch):
        camp = new_campaign(results_root=Path("/tmp/oom_test"), stamp="O1")
        phase = camp["phases"][0]
        step = apply_oom_escalation(camp, phase)
        assert step["batch_scale"] == 0.5
        assert phase["oom_level"] == 1
        step2 = apply_oom_escalation(camp, phase)
        assert step2["partition"] == "ica100"
        assert camp.get("partition_650m") == "ica100"

    def test_phase_failed_oom_state(self):
        phase = {"job_ids": {"ultra_650M": "9"}}
        states = {"9": JobState("9", "OUT_OF_MEMORY", "0:125")}
        assert phase_failed_oom(states, phase) is True

    def test_advance_escalates_on_oom(self, tmp_path, monkeypatch):
        calls = []

        def fake_submit(campaign, phase, dry_run=False):
            calls.append(
                {
                    "kind": phase["kind"],
                    "oom_level": phase.get("oom_level", 0),
                    "batch_scale": __import__("os").environ.get("DISORDERNET_BATCH_SCALE"),
                    "partition": phase.get("force_partition"),
                }
            )
            phase["job_ids"] = {"ultra_650M": "100", "package": "101"}
            phase["status"] = "running"
            Path(phase["root"]).mkdir(parents=True, exist_ok=True)
            return phase

        monkeypatch.setattr("rockfish.publish_campaign._submit_kind", fake_submit)
        camp = new_campaign(results_root=tmp_path, stamp="O2")
        # First submit
        camp = advance_campaign(camp, dry_run=True)
        assert calls[0]["oom_level"] == 0
        # Simulate OOM then resubmit
        def job_query(ids):
            return {str(ids[0]): JobState(str(ids[0]), "OUT_OF_MEMORY", "0:125")}

        camp = advance_campaign(camp, dry_run=True, job_query=job_query)
        assert calls[-1]["oom_level"] >= 1
        assert float(calls[-1]["batch_scale"]) <= 0.5

    def test_escalation_table_nonempty(self):
        assert len(OOM_ESCALATION) >= 4
