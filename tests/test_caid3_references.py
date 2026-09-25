"""A reconstructed benchmark is comparable only if its composition matches.

Only disorder_pdb.fasta ships in the CAID demo-data, so the other four CAID3
benchmarks have to be reconstructed or they cannot be measured at all. The risk
is obvious: a reconstruction that is easier than the published benchmark
produces a flattering number that looks head-to-head and is not.

Measured against the published composition, from the 319 Disorder-PDB targets:

    Linker        31 targets / 1,379 positives   EXACT
    Binding       49 / 4,673  vs  52 / 2,991     approximate
    Disorder-NOX 319 / 31,518 vs 204 / 26,367    superset

So only Linker may be quoted beside its published leader.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.caid3_references import (
    PUBLISHED,
    build_reference,
    composition,
    validate,
    write_caid_fasta,
)
from colab.caid_tasks import MASK

DISORDER = "IDPO:0000002"
LINKER = "IDPO:0000033"
BINDING = "GO:0005515"


def entry(pid, seq, regions):
    return {"disprot_id": pid, "acc": pid, "sequence": seq, "length": len(seq),
            "regions": [{"term_id": t, "start": s, "end": e} for t, s, e in regions]}


def targets(pairs):
    return [{"id": p, "sequence": s} for p, s in pairs]


class TestHeadToHeadGate:
    def test_exact_composition_is_head_to_head(self):
        v = validate("linker", [{
            "labels": np.array([1] * 1379 + [0] * 10),
            "eval_mask": np.ones(1389, dtype=bool),
        }] + [{"labels": np.zeros(1), "eval_mask": np.ones(1, dtype=bool)}] * 30)
        assert v["reconstructed"]["targets"] == 31
        assert v["reconstructed"]["positives"] == 1379
        assert v["head_to_head"] is True
        assert "EXACT" in v["verdict"]

    def test_wrong_positive_count_is_not_head_to_head(self):
        """Binding reconstructs 4,673 positives against a published 2,991 —
        a different benchmark, however similar the target count."""
        ref = [{"labels": np.ones(4673), "eval_mask": np.ones(4673, dtype=bool)}]
        ref += [{"labels": np.zeros(1), "eval_mask": np.ones(1, dtype=bool)}] * 51
        v = validate("binding", ref)
        assert v["head_to_head"] is False
        assert "not comparable" in v["verdict"].lower()

    def test_superset_is_not_head_to_head(self):
        """Disorder-NOX keeps 204 of the targets; every target with a disorder
        annotation is 319, which is an easier benchmark."""
        ref = [{"labels": np.ones(1), "eval_mask": np.ones(1, dtype=bool)}] * 319
        assert validate("disorder_nox", ref)["head_to_head"] is False

    def test_verdict_names_the_published_leader(self):
        v = validate("linker", [])
        assert v["published"]["leader"] == "IPA-AF2-Linker"
        assert v["published"]["auc"] == 0.897

    @pytest.mark.parametrize("task", sorted(PUBLISHED))
    def test_every_benchmark_has_a_published_reference_point(self, task):
        for key in ("targets", "positives", "leader", "auc", "aps"):
            assert PUBLISHED[task][key] is not None


class TestBuild:
    def test_target_universe_is_fixed_by_the_official_list(self):
        """A reconstruction must not evaluate proteins the challenge did not."""
        tg = targets([("DP1", "A" * 30)])
        by_id = {
            "DP1": entry("DP1", "A" * 30, [(LINKER, 1, 10)]),
            "DP2": entry("DP2", "A" * 30, [(LINKER, 1, 10)]),
        }
        ref = build_reference(tg, by_id, "linker")
        assert [r["id"] for r in ref] == ["DP1"]

    def test_sequence_length_disagreement_is_skipped_not_aligned(self):
        """A different sequence means a different protein; guessing an
        alignment is how this project lost a benchmark once already."""
        tg = targets([("DP1", "A" * 30)])
        by_id = {"DP1": entry("DP1", "A" * 25, [(LINKER, 1, 10)])}
        assert build_reference(tg, by_id, "linker") == []

    def test_invalid_regions_mask_binding_positives(self):
        """CAID3's manual curation is why our binding count is over-inclusive."""
        tg = targets([("DP1", "A" * 40)])
        by_id = {"DP1": entry("DP1", "A" * 40, [(BINDING, 1, 20)])}
        plain = composition(build_reference(tg, by_id, "binding"))
        curated = composition(build_reference(
            tg, by_id, "binding", invalid_regions={"DP1": [(1, 10)]}))
        assert curated["positives"] < plain["positives"]

    def test_targets_with_no_annotation_are_dropped(self):
        tg = targets([("DP1", "A" * 30)])
        by_id = {"DP1": entry("DP1", "A" * 30, [(DISORDER, 1, 10)])}
        assert build_reference(tg, by_id, "linker") == []


class TestFastaRoundTrip:
    def test_written_reference_reads_back_identically(self, tmp_path):
        """The written file must parse under the same reader used for the
        official reference, or the reconstruction is not interchangeable."""
        from colab.caid3_eval import parse_caid_reference_fasta

        labels = np.array([1, 1, MASK, 0, 0, MASK, 1], dtype=np.int8)
        ref = [{"id": "DP1", "sequence": "ACDEFGH", "labels": labels,
                "eval_mask": labels != MASK, "task": "linker"}]
        path = write_caid_fasta(ref, str(tmp_path / "r.fasta"))
        back = parse_caid_reference_fasta(path)[0]
        assert back["sequence"] == "ACDEFGH"
        assert list(np.asarray(back["eval_mask"])) == [True, True, False, True,
                                                       True, False, True]
        got = np.asarray(back["labels"])[np.asarray(back["eval_mask"])]
        assert list(got) == [1, 1, 0, 0, 1]

    def test_masked_positions_are_written_as_dashes(self, tmp_path):
        labels = np.array([MASK, 1, 0], dtype=np.int8)
        ref = [{"id": "D", "sequence": "ACD", "labels": labels,
                "eval_mask": labels != MASK, "task": "binding_idr"}]
        text = open(write_caid_fasta(ref, str(tmp_path / "r.fasta"))).read()
        assert text.splitlines()[2] == "-10"
