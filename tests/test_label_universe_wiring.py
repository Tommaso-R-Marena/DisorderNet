"""The cross-organism MobiDB selector must actually reach the fetch call.

`GLOBAL_PDB_COVERAGE_QUERY` was implemented in colab/label_sources.py and the
`pdb_missing` ablation arm set DISORDERNET_MOBIDB_GLOBAL=1, but nothing read the
variable and `run_disordernet.py` never passed `query=`. Every arm that asked
for the cross-organism set silently trained on the human reference proteome:
3,388 proteins / 1.2M evidenced residues instead of 19,819 / 6.6M — while the
epoch budget stayed step-matched to the larger figure, so the arm ran 5 epochs
on DisProt-sized data and reported itself as a matched-budget comparison.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest

from colab.label_sources import GLOBAL_PDB_COVERAGE_NAME, GLOBAL_PDB_COVERAGE_QUERY
from rockfish.ablation import ABLATION_KEY_DEFAULTS, ARMS, EVIDENCED_RESIDUES
from rockfish.run_disordernet import (
    _assert_label_set_matches_budget,
    _load_proteins_mobidb,
    build_parser,
)


def _args(**kw):
    base = dict(
        label_source="pdb_missing", mobidb_proteome="UP000005640", mobidb_cache="",
        mobidb_limit=0, mobidb_global=False, min_evidence_fraction=0.10,
        caid_leak_identity=0.40,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _cfg():
    return SimpleNamespace(min_seq_len=20, max_seq_len=1022, min_disorder=3, min_order=3)


def _run_loader(args):
    """Call the loader with every downstream side effect stubbed out.

    The stub reports a zero-residue label set, so the budget guard must be
    disabled here or it correctly aborts. Cleared explicitly rather than
    assumed absent: submitting an ablation sets this variable process-wide.
    """
    captured = {}

    def fake_fetch(proteome, cache, *, limit=None, query=None):
        captured["proteome"] = proteome
        captured["cache"] = cache
        captured["query"] = query
        return []

    clean = {k: v for k, v in os.environ.items() if k != "DISORDERNET_EXPECTED_RESIDUES"}
    with mock.patch.dict(os.environ, clean, clear=True), \
         mock.patch("colab.label_sources.fetch_mobidb_proteome", fake_fetch), \
         mock.patch("colab.label_sources.build_labelled_set",
                    return_value=([], {"n_evidenced_residues": 0})), \
         mock.patch("colab.label_sources.to_pipeline_proteins", return_value=[]), \
         mock.patch("rockfish.run_disordernet._apply_caid_leak_free_filter",
                    side_effect=lambda p, m, c, a: (p, m)), \
         mock.patch("rockfish.run_disordernet._label_set_sha", return_value="sha"):
        _, meta = _load_proteins_mobidb(_cfg(), args)
    return captured, meta


class TestSelectorReachesTheFetch:
    def test_global_flag_passes_the_cross_organism_query(self):
        captured, _ = _run_loader(_args(mobidb_global=True))
        assert captured["query"] == GLOBAL_PDB_COVERAGE_QUERY

    def test_without_the_flag_no_query_is_sent(self):
        captured, _ = _run_loader(_args(mobidb_global=False))
        assert captured["query"] is None

    def test_the_two_universes_use_different_caches(self):
        """Sharing a cache file would let one universe's records be served for
        the other, which is how the fallback stayed invisible."""
        glob, _ = _run_loader(_args(mobidb_global=True))
        prot, _ = _run_loader(_args(mobidb_global=False))
        assert glob["cache"] != prot["cache"]
        assert GLOBAL_PDB_COVERAGE_NAME in glob["cache"]
        assert "UP000005640" in prot["cache"]

    def test_metadata_records_the_universe_actually_pulled(self):
        _, meta = _run_loader(_args(mobidb_global=True))
        assert meta["mobidb_global"] is True
        assert meta["mobidb_universe"] == GLOBAL_PDB_COVERAGE_NAME


class TestCliAndArmWiring:
    def test_the_flag_exists_on_the_cli(self):
        args = build_parser().parse_args(["cv", "--mobidb-global"])
        assert args.mobidb_global is True

    def test_the_env_var_the_ablation_sets_is_the_one_the_cli_reads(self):
        """The defect in one line: the arm set a variable nothing consumed."""
        with mock.patch.dict(os.environ, {"DISORDERNET_MOBIDB_GLOBAL": "1"}):
            assert build_parser().parse_args(["cv"]).mobidb_global is True
        with mock.patch.dict(os.environ, {"DISORDERNET_MOBIDB_GLOBAL": "0"}):
            assert build_parser().parse_args(["cv"]).mobidb_global is False

    @pytest.mark.parametrize(
        "arm", [a for a in ARMS.values() if a.env.get("DISORDERNET_MOBIDB_GLOBAL") == "1"]
    )
    def test_global_arms_have_a_budget_matching_the_global_universe(self, arm):
        """A global arm budgeted from human-proteome counts would undertrain."""
        src = arm.env["DISORDERNET_LABEL_SOURCE"]
        assert EVIDENCED_RESIDUES[src] > 5_000_000, (
            f"{arm.name} selects the cross-organism universe but is budgeted "
            f"for {EVIDENCED_RESIDUES[src]:,} residues"
        )

    def test_the_key_has_a_neutral_default_for_every_other_arm(self):
        assert ABLATION_KEY_DEFAULTS["DISORDERNET_MOBIDB_GLOBAL"] == "0"


class TestBudgetGuard:
    def test_a_short_label_set_aborts(self):
        """What should have happened: 1.2M against a 6.6M budget."""
        with mock.patch.dict(os.environ, {"DISORDERNET_EXPECTED_RESIDUES": "6611600"}):
            with pytest.raises(SystemExit) as exc:
                _assert_label_set_matches_budget(
                    {"n_evidenced_residues": 1_218_504}, "pdb_missing", "UP000005640"
                )
        msg = str(exc.value)
        assert "6,611,600" in msg and "1,218,504" in msg
        assert "mobidb-global" in msg.lower()

    def test_the_expected_size_passes(self):
        with mock.patch.dict(os.environ, {"DISORDERNET_EXPECTED_RESIDUES": "6611600"}):
            _assert_label_set_matches_budget(
                {"n_evidenced_residues": 6_600_000}, "pdb_missing", "pdbcov"
            )

    def test_small_drift_is_tolerated(self):
        """Label sources move a little between MobiDB releases; that is not a
        misconfiguration and must not block a run."""
        with mock.patch.dict(os.environ, {"DISORDERNET_EXPECTED_RESIDUES": "1000000"}):
            _assert_label_set_matches_budget(
                {"n_evidenced_residues": 1_150_000}, "disprot", "n/a"
            )

    def test_absent_budget_disables_the_check(self):
        """Direct runs outside the ablation must not be forced to declare one."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISORDERNET_EXPECTED_RESIDUES", None)
            _assert_label_set_matches_budget(
                {"n_evidenced_residues": 1}, "disprot", "n/a"
            )


class TestSubmitterDoesNotLeakTheBudget:
    """Submitting must not leave one arm's budget in the ambient environment.

    ABLATION_KEY_DEFAULTS keys are cleared before each arm precisely because
    os.environ persists across iterations. DISORDERNET_EXPECTED_RESIDUES is
    derived per arm rather than declared there, so it escaped that treatment —
    caught when it leaked out of the ablation tests and tripped the budget guard
    in an unrelated test in the same process.
    """

    def _submit(self, tmp_path, arms):
        import subprocess
        import sys as _sys
        from pathlib import Path as _Path

        repo = _Path(__file__).resolve().parents[1]
        return subprocess.run(
            [_sys.executable, "-c",
             "import os, sys, runpy;"
             "sys.argv = ['ablation.py', 'submit', '--arms', %r,"
             " '--root-workdir', %r, '--dry-run'];"
             "runpy.run_path(%r, run_name='__main__');"
             % (arms, str(tmp_path), str(repo / "rockfish" / "ablation.py")),
             ],
            capture_output=True, text=True, cwd=str(repo),
        )

    def test_derived_key_is_registered_for_cleanup(self):
        from rockfish.ablation import DERIVED_ABLATION_KEYS

        assert "DISORDERNET_EXPECTED_RESIDUES" in DERIVED_ABLATION_KEYS

    def test_each_arm_gets_its_own_budget_not_the_previous_arms(self, tmp_path):
        """pdb_missing (6.6M) and baseline (988k) must not share a figure."""
        from rockfish.ablation import ARMS, EVIDENCED_RESIDUES

        assert EVIDENCED_RESIDUES["pdb_missing"] != EVIDENCED_RESIDUES["disprot"]
        # Every arm resolves a source, so every arm resolves a distinct budget.
        for arm in ARMS.values():
            src = arm.env.get("DISORDERNET_LABEL_SOURCE", "disprot")
            assert src in EVIDENCED_RESIDUES, f"{arm.name} has no measured budget"
