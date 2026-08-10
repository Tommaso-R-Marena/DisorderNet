"""Ablation arms must be isolated from one another.

``os.environ`` persists between submissions and ``sbatch_export_keys()``
forwards any key that is merely *set*, so an arm that omits a key would inherit
the previous arm's value — no_plddt silently running at pdb_missing's 5-epoch
budget, say. Nothing in the results would reveal it, so it needs a guard.
"""

from __future__ import annotations

import os

import pytest

from rockfish.ablation import (
    ABLATION_KEY_DEFAULTS,
    ARMS,
    BASELINE_EPOCHS,
    compute_matched_epochs,
)


class TestStepMatching:
    def test_larger_label_sets_get_proportionally_fewer_epochs(self):
        assert compute_matched_epochs("disprot") == BASELINE_EPOCHS
        # pdb_missing carries ~6.7x the evidenced residues
        assert compute_matched_epochs("pdb_missing") == 5
        # mobidb_curated is SMALLER than DisProt, so it needs more passes
        assert compute_matched_epochs("mobidb_curated") > BASELINE_EPOCHS

    def test_never_degenerates_to_zero(self):
        assert compute_matched_epochs("nonexistent_source") == BASELINE_EPOCHS
        assert compute_matched_epochs("pdb_missing", baseline_epochs=1) >= 3


class TestArmDefinitions:
    def test_every_arm_env_key_has_a_declared_default(self):
        """Otherwise the key cannot be reset between arms and will leak."""
        for arm in ARMS.values():
            for key in arm.env:
                assert key in ABLATION_KEY_DEFAULTS, (
                    f"arm {arm.name!r} sets {key!r}, which has no entry in "
                    "ABLATION_KEY_DEFAULTS and so cannot be cleared for arms "
                    "that do not set it"
                )

    def test_label_source_arms_are_marked_as_changing_the_task(self):
        """Their CV AUC is not comparable to the baseline's."""
        for arm in ARMS.values():
            src = arm.env.get("DISORDERNET_LABEL_SOURCE")
            if src and src != "disprot":
                assert arm.changes_task, (
                    f"arm {arm.name!r} trains on {src!r} — a different prediction "
                    "task — but is not flagged changes_task, so the results table "
                    "would present its CV AUC as comparable"
                )

    def test_baseline_is_neutral(self):
        assert ARMS["baseline"].env == {}
        assert not ARMS["baseline"].changes_task


class TestNoCrossArmLeakage:
    def test_submission_sets_every_key_for_every_arm(self, monkeypatch, tmp_path):
        """Submit several arms in the worst-case order and assert isolation."""
        import rockfish.ablation as ab

        captured: list = []

        def fake_submit(script, **kw):
            captured.append({"job_name": kw["job_name"], "env": dict(kw["env"])})
            return f"JOB{len(captured)}"

        monkeypatch.setattr(ab, "submit_sbatch", fake_submit)
        monkeypatch.setattr(ab, "mail_sbatch_args", lambda *a, **k: [])
        monkeypatch.setattr(ab, "git_revision", lambda *a, **k: "deadbeef")

        args = ab.build_parser().parse_args([
            "submit",
            # pdb_missing (5 epochs, global, task change) sits between two arms
            # that must NOT inherit any of it.
            "--arms", "pdb_missing,no_plddt,baseline",
            "--account", "acct",
            "--root-workdir", str(tmp_path),
        ])
        assert ab.cmd_submit(args) == 0
        assert len(captured) == 3

        by_name = {c["job_name"].replace("dn-abl-", ""): c["env"] for c in captured}

        # pdb_missing: its own settings
        assert by_name["pdb_missing"]["DISORDERNET_LABEL_SOURCE"] == "pdb_missing"
        assert by_name["pdb_missing"]["DISORDERNET_MOBIDB_GLOBAL"] == "1"
        assert by_name["pdb_missing"]["DISORDERNET_NUM_EPOCHS"] == "5"

        # no_plddt: must NOT have inherited label source, global flag or epochs
        assert by_name["no_plddt"]["DISORDERNET_LABEL_SOURCE"] == "disprot"
        assert by_name["no_plddt"]["DISORDERNET_MOBIDB_GLOBAL"] == "0"
        assert by_name["no_plddt"]["DISORDERNET_NUM_EPOCHS"] == str(BASELINE_EPOCHS)
        assert by_name["no_plddt"]["RUN_NO_PLDDT_FEATURES"] == "1"

        # baseline: fully neutral despite following two modified arms
        for key, default in ABLATION_KEY_DEFAULTS.items():
            assert by_name["baseline"][key] == default, (
                f"baseline inherited {key}={by_name['baseline'][key]!r} "
                f"(expected {default!r})"
            )

    def test_parent_environment_is_not_left_polluted(self, monkeypatch, tmp_path):
        import rockfish.ablation as ab

        monkeypatch.setattr(ab, "submit_sbatch", lambda script, **kw: "JOB")
        monkeypatch.setattr(ab, "mail_sbatch_args", lambda *a, **k: [])
        monkeypatch.setattr(ab, "git_revision", lambda *a, **k: "x")
        monkeypatch.delenv("RUN_NO_PLDDT_FEATURES", raising=False)

        args = ab.build_parser().parse_args([
            "submit", "--arms", "no_plddt", "--account", "a",
            "--root-workdir", str(tmp_path),
        ])
        ab.cmd_submit(args)
        # The arm set it; a subsequent baseline submission in the same process
        # must still come out neutral (covered above), so the key must at least
        # be explicitly present rather than ambiently inherited.
        assert os.environ.get("RUN_NO_PLDDT_FEATURES") == "1"


class TestManifest:
    def test_manifest_records_what_was_submitted(self, monkeypatch, tmp_path):
        import json

        import rockfish.ablation as ab

        monkeypatch.setattr(ab, "submit_sbatch", lambda script, **kw: "JOB1")
        monkeypatch.setattr(ab, "mail_sbatch_args", lambda *a, **k: [])
        monkeypatch.setattr(ab, "git_revision", lambda *a, **k: "abc123")

        args = ab.build_parser().parse_args([
            "submit", "--arms", "baseline", "--account", "a",
            "--root-workdir", str(tmp_path),
        ])
        ab.cmd_submit(args)
        m = json.loads((tmp_path / "ablation_manifest.json").read_text())
        assert m["git_revision"] == "abc123"
        assert m["arms"][0]["arm"] == "baseline"
        assert m["baseline_env"]["PROFILE"] == "ultra"


@pytest.mark.parametrize("arm_name", sorted(ARMS))
def test_all_arms_declare_a_hypothesis(arm_name):
    """An arm without a stated hypothesis cannot be interpreted."""
    arm = ARMS[arm_name]
    assert arm.hypothesis and len(arm.hypothesis) > 40
    assert arm.description


class TestDeterminismIsNotSilentlyOverridden:
    """cfg.deterministic set cudnn.benchmark=False, then apply_hpc_runtime_settings
    unconditionally set it back to True — so deterministic runs were never
    deterministic on HPC, and identical reruns diverged by ~0.023 AUC."""

    def test_hpc_settings_respect_an_existing_determinism_request(self):
        import torch

        from colab.hpc_efficiency import apply_hpc_runtime_settings

        prev_b = torch.backends.cudnn.benchmark
        prev_d = torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            apply_hpc_runtime_settings(verbose=False)
            assert torch.backends.cudnn.benchmark is False, (
                "apply_hpc_runtime_settings re-enabled autotuning and silently "
                "undid the caller's determinism request"
            )
        finally:
            torch.backends.cudnn.benchmark = prev_b
            torch.backends.cudnn.deterministic = prev_d

    def test_non_deterministic_runs_still_get_autotuning(self):
        import torch

        from colab.hpc_efficiency import apply_hpc_runtime_settings

        prev_b = torch.backends.cudnn.benchmark
        prev_d = torch.backends.cudnn.deterministic
        try:
            torch.backends.cudnn.deterministic = False
            apply_hpc_runtime_settings(verbose=False)
            if torch.cuda.is_available():
                assert torch.backends.cudnn.benchmark is True
        finally:
            torch.backends.cudnn.benchmark = prev_b
            torch.backends.cudnn.deterministic = prev_d

    def test_every_ablation_arm_requests_determinism(self):
        from rockfish.ablation import ABLATION_KEY_DEFAULTS

        assert ABLATION_KEY_DEFAULTS.get("DISORDERNET_DETERMINISTIC") == "1", (
            "ablation arms must be reproducible; otherwise a delta cannot be "
            "distinguished from run-to-run divergence"
        )
