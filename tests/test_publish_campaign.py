"""Tests for forced confidence mode and publish campaign resume logic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from confidence import (  # noqa: E402
    annotate_confidence,
    conformal_quantile,
    conformal_report,
    forced_prediction,
)
from rockfish.publish_campaign import (  # noqa: E402
    JobState,
    advance_campaign,
    estimate_eta_seconds,
    mail_sbatch_args,
    needs_resubmit,
    new_campaign,
    package_ready,
    parse_sacct,
    phase_progress,
    pipeline_artifacts_ready,
    run_watchdog,
    save_campaign,
)


class TestForcedConfidence:
    def test_forced_prediction_always_labels(self):
        p = np.array([0.1, 0.49, 0.5, 0.9])
        out = forced_prediction(p)
        assert out["y_hat"].tolist() == [0, 0, 1, 1]
        assert out["confidence_pct"].shape == (4,)
        np.testing.assert_allclose(out["confidence_pct"][0], 90.0, atol=1e-4)
        np.testing.assert_allclose(out["confidence_pct"][2], 50.0, atol=1e-4)

    def test_annotate_keeps_abstain_and_forced(self):
        rng = np.random.RandomState(0)
        p = rng.uniform(0, 1, 5000)
        y = (rng.uniform(0, 1, 5000) < p).astype(int)
        q = conformal_quantile(p[:2500], y[:2500], alpha=0.1, class_conditional=True)
        ann = annotate_confidence(p[2500:], q)
        assert "decision" in ann and "confidence_pct" in ann and "y_hat_forced" in ann
        assert len(ann["decision"]) == 2500
        # forced never abstains
        assert set(np.unique(ann["y_hat_forced"])).issubset({0, 1})

    def test_report_includes_forced_accuracy(self):
        rng = np.random.RandomState(1)
        p = rng.uniform(0, 1, 8000)
        y = (rng.uniform(0, 1, 8000) < p).astype(int)
        q = conformal_quantile(p[:4000], y[:4000], alpha=0.1)
        rep = conformal_report(p[4000:], y[4000:], q)
        assert "forced_accuracy" in rep
        assert 0.5 <= rep["forced_accuracy"] <= 1.0
        assert "mean_confidence_pct" in rep


class TestCampaignHelpers:
    def test_mail_args_default(self, monkeypatch):
        monkeypatch.delenv("DISORDERNET_MAIL_USER", raising=False)
        args = mail_sbatch_args()
        assert any(a.startswith("--mail-user=marenatommaso@gmail.com") for a in args)
        assert any("TIME_LIMIT" in a for a in args)

    def test_mail_args_off(self):
        assert mail_sbatch_args("none") == []

    def test_parse_sacct_skips_steps(self):
        text = "111|TIMEOUT|0:1\n111.batch|TIMEOUT|0:1\n222|COMPLETED|0:0\n"
        st = parse_sacct(text)
        assert set(st) == {"111", "222"}
        assert st["111"].state == "TIMEOUT"
        assert st["111"].bad is True
        assert st["222"].ok is True

    def test_pipeline_artifacts_ready(self, tmp_path):
        ckpt = tmp_path / "checkpoints"
        ckpt.mkdir()
        assert pipeline_artifacts_ready(ckpt) is False
        (ckpt / "cv_summary.json").write_text("{}")
        (ckpt / "cv_progress.json").write_text('{"fold_results":[{},{},{},{},{}]}')
        (ckpt / "sota_postprocess_report.json").write_text("{}")
        (ckpt / "structure_distrust_benchmark.json").write_text("{}")
        assert pipeline_artifacts_ready(ckpt) is True

    def test_phase_progress_fraction(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DISORDERNET_RESULTS", str(tmp_path))
        root = tmp_path / "publish_650m_X"
        for label, sub, n in (
            ("ultra_650M", "checkpoints", 2),
            ("ultra_clean_650M", "checkpoints_ultra_clean", 5),
        ):
            ckpt = root / label / sub
            ckpt.mkdir(parents=True)
            (ckpt / "cv_progress.json").write_text(
                json.dumps({"fold_results": [{}] * n})
            )
        prog = phase_progress(root, "650m", include_clean=True)
        assert prog["folds_done"] == 7
        assert prog["folds_total"] == 10
        assert prog["fraction"] == pytest.approx(0.7)

    def test_new_campaign_paths(self, tmp_path):
        camp = new_campaign(results_root=tmp_path, stamp="T0", gpu_account="sfried3_gpu")
        assert camp["phases"][0]["kind"] == "650m"
        assert camp["phases"][1]["kind"] == "3b"
        assert "publish_650m_T0" in camp["phases"][0]["root"]
        assert camp["mail_user"] == "marenatommaso@gmail.com"

    def test_needs_resubmit_on_timeout(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        phase = {"root": str(root), "job_ids": {"ultra_650M": "9"}}
        states = {"9": JobState("9", "TIMEOUT", "0:1")}
        assert needs_resubmit(phase, states) is True

    def test_needs_resubmit_false_when_running(self, tmp_path):
        root = tmp_path / "r"
        root.mkdir()
        phase = {"root": str(root), "job_ids": {"ultra_650M": "9"}}
        states = {"9": JobState("9", "RUNNING", "0:0")}
        assert needs_resubmit(phase, states) is False

    def test_advance_marks_done_when_packages_exist(self, tmp_path):
        camp = new_campaign(results_root=tmp_path, stamp="T1")
        for phase in camp["phases"]:
            pkg = Path(phase["root"]) / "publish_package"
            pkg.mkdir(parents=True)
            (pkg / "PACKAGE_README.md").write_text("ok\n")
            phase["status"] = "running"
            phase["job_ids"] = {"package": "1"}
        camp = advance_campaign(camp, dry_run=True)
        assert camp["status"] == "done"
        assert all(p["status"] == "done" for p in camp["phases"])

    def test_advance_submits_650m_first(self, tmp_path, monkeypatch):
        calls = []

        def fake_submit(campaign, phase, dry_run=False):
            calls.append(phase["kind"])
            phase["job_ids"] = {"ultra_650M": "DRY1", "package": "DRY2"}
            phase["status"] = "running"
            # pretend summary file
            Path(phase["root"]).mkdir(parents=True, exist_ok=True)
            return phase

        monkeypatch.setattr(
            "rockfish.publish_campaign._submit_kind", fake_submit
        )
        camp = new_campaign(results_root=tmp_path, stamp="T2")
        camp = advance_campaign(camp, dry_run=True)
        assert calls == ["650m"]
        assert camp["phases"][0]["status"] == "running"
        assert camp["phases"][1]["status"] == "pending"

    def test_advance_moves_to_3b_after_650m_package(self, tmp_path, monkeypatch):
        calls = []

        def fake_submit(campaign, phase, dry_run=False):
            calls.append(phase["kind"])
            phase["job_ids"] = {"main": "1", "package": "2"}
            phase["status"] = "running"
            Path(phase["root"]).mkdir(parents=True, exist_ok=True)
            return phase

        monkeypatch.setattr(
            "rockfish.publish_campaign._submit_kind", fake_submit
        )
        camp = new_campaign(results_root=tmp_path, stamp="T3")
        pkg = Path(camp["phases"][0]["root"]) / "publish_package"
        pkg.mkdir(parents=True)
        (pkg / "PACKAGE_README.md").write_text("ok\n")
        camp["phases"][0]["status"] = "running"
        camp = advance_campaign(camp, dry_run=True)
        assert camp["phases"][0]["status"] == "done"
        assert calls == ["3b"]

    def test_watchdog_completes(self, tmp_path, monkeypatch):
        camp = new_campaign(results_root=tmp_path, stamp="T4")
        for phase in camp["phases"]:
            pkg = Path(phase["root"]) / "publish_package"
            pkg.mkdir(parents=True)
            (pkg / "PACKAGE_README.md").write_text("ok\n")
        path = tmp_path / "camp.json"
        save_campaign(path, camp)
        sleeps = []
        rc = run_watchdog(
            path,
            poll_seconds=1,
            max_cycles=3,
            sleep_fn=lambda s: sleeps.append(s),
            job_query=lambda ids: {},
            active_jobs_fn=lambda: False,
        )
        assert rc == 0
        assert load_status(path) == "done"

    def test_estimate_eta(self):
        assert estimate_eta_seconds({"fraction": 0.0}, 1000) is None
        eta = estimate_eta_seconds({"fraction": 0.5}, 3600)
        assert eta is not None and eta > 0


def load_status(path: Path) -> str:
    return json.loads(path.read_text())["status"]


class TestPredictorForcedFields:
    def test_predict_from_embeddings_has_confidence_pct(self):
        from predictor import fit_bundle, predict_from_embeddings, phys_features

        rng = np.random.RandomState(0)
        phys_list, esm_list, lab_list = [], [], []
        for _ in range(12):
            L = 40
            seq = "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), L))
            phys_list.append(phys_features(seq))
            esm_list.append(rng.randn(L, 480).astype(np.float32))
            lab_list.append(rng.randint(0, 2, L).astype(np.float32))
        bundle = fit_bundle(phys_list, esm_list, lab_list, alpha=0.2, n_jobs=1)
        out = predict_from_embeddings(bundle, phys_list[0], esm_list[0])
        assert "confidence_pct" in out and "y_hat_forced" in out
        assert out["confidence_pct"].shape[0] == phys_list[0].shape[0]
