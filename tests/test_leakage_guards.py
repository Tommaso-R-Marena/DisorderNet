"""Regression tests for the four leakage paths that inflated the reported AUC.

Each test fails against the pre-fix behaviour:

1. ``resolve_cv_splits`` — post-training consumers used to re-derive folds with
   the default ``split_method="protein"`` while ultra/ultra3b train with
   ``"homology"``. That put a fold model's own training proteins into its
   "held-out" evaluation set (fold soup) and let the v6 stream train on
   homologues of the proteins it scored.
2. ``apply_meta_stacker`` — the logistic stacker was fitted on every OOF residue
   and then scored those same residues, so ``final_pooled`` was a
   resubstitution score.
3. ``calibrate_fold_results(method="isotonic")`` — isotonic regression was fitted
   on the residues it transformed. Unlike temperature scaling it is not a strictly
   monotone map, so this genuinely inflates AUC/AP rather than leaving them
   invariant. ``temperature_then_isotonic`` is the pipeline default.
4. ``save_cv_progress`` — recorded ``fold_val_ids`` under the default split
   method regardless of the method the run actually trained with.
"""

from __future__ import annotations

import numpy as np
import pytest

from colab.calibration import calibrate_fold_results
from colab.cv_splits import resolve_cv_splits, splits_from_val_ids


class _Cfg:
    """Minimal stand-in for TrainConfig."""

    def __init__(self, split_method="protein", homology_min_identity=0.40):
        self.split_method = split_method
        self.homology_min_identity = homology_min_identity


_AA = "ACDEFGHIKLMNPQRSTVWY"


def _make_proteins(n=20, seed=0):
    """Mutually dissimilar proteins (each its own homology cluster)."""
    rng = np.random.default_rng(seed)
    proteins = []
    for i in range(n):
        length = 120 + 7 * i
        seq = "".join(rng.choice(list(_AA), size=length))
        proteins.append(
            {
                "id": f"DP{i:03d}",
                "sequence": seq,
                "length": length,
                "labels": [(j + i) % 2 for j in range(length)],
            }
        )
    return proteins


def _make_protein_families(n_families=5, per_family=4, length=300, seed=0):
    """Proteins in near-identical families, so homology and protein splits differ.

    Lengths deliberately straddle 200 residues: below that difflib's autojunk
    heuristic stays off and the old code appeared to work.
    """
    rng = np.random.default_rng(seed)
    proteins = []
    for f in range(n_families):
        base = list(rng.choice(list(_AA), size=length))
        for k in range(per_family):
            seq = list(base)
            for pos in rng.choice(length, size=length // 20, replace=False):
                seq[pos] = rng.choice(list(_AA))
            s = "".join(seq)
            proteins.append(
                {
                    "id": f"F{f:02d}M{k:02d}",
                    "sequence": s,
                    "length": len(s),
                    "labels": [(j + f) % 2 for j in range(len(s))],
                }
            )
    return proteins


class TestResolveCvSplits:
    def test_prefers_recorded_val_ids_over_rederivation(self):
        """Recorded val_ids are ground truth even when they disagree with any
        re-derived partition."""
        proteins = _make_proteins(10)
        # A grouping no split function would produce (reverse-ordered blocks).
        grouping = [
            ["DP009", "DP008"],
            ["DP007", "DP006"],
            ["DP005", "DP004"],
            ["DP003", "DP002"],
            ["DP001", "DP000"],
        ]
        fold_results = [{"val_ids": g} for g in grouping]
        splits = resolve_cv_splits(proteins, 5, fold_results=fold_results)

        for fold_idx, ids in enumerate(grouping):
            _, val_idx = splits[fold_idx]
            assert sorted(proteins[i]["id"] for i in val_idx) == sorted(ids)

    def test_honours_cfg_split_method_when_val_ids_absent(self):
        """Without recorded val_ids the run's own split_method must be used."""
        proteins = _make_protein_families(n_families=5, per_family=4, length=300)
        protein_splits = resolve_cv_splits(proteins, 5, cfg=_Cfg("protein"))
        homology_splits = resolve_cv_splits(proteins, 5, cfg=_Cfg("homology"))

        def as_id_sets(splits):
            return [frozenset(proteins[i]["id"] for i in val) for _, val in splits]

        # The two methods must not silently coincide, otherwise this test cannot
        # detect the bug it guards.
        assert as_id_sets(protein_splits) != as_id_sets(homology_splits)

    def test_train_and_val_are_disjoint_and_exhaustive(self):
        proteins = _make_proteins(12)
        fold_results = [
            {"val_ids": [p["id"] for p in proteins[i::4]]} for i in range(4)
        ]
        splits = resolve_cv_splits(proteins, 4, fold_results=fold_results)

        for train_idx, val_idx in splits:
            assert set(train_idx).isdisjoint(set(val_idx))
            assert set(train_idx) | set(val_idx) == set(range(len(proteins)))

    def test_partial_val_ids_fall_back_to_rederivation(self):
        """A half-written progress file must not silently produce empty folds."""
        proteins = _make_proteins(10)
        fold_results = [{"val_ids": ["DP000", "DP001"]}, {}]
        splits = resolve_cv_splits(proteins, 5, cfg=_Cfg("protein"), fold_results=fold_results)
        assert len(splits) == 5
        covered = sorted(i for _, val in splits for i in val)
        assert covered == list(range(len(proteins)))

    def test_unknown_ids_are_dropped_not_crashed(self):
        proteins = _make_proteins(6)
        fold_results = [{"val_ids": ["DP000", "GHOST"]}, {"val_ids": ["DP001"]}]
        splits = splits_from_val_ids(proteins, [f["val_ids"] for f in fold_results])
        _, val0 = splits[0]
        assert [proteins[i]["id"] for i in val0] == ["DP000"]


class TestMetaStackerIsOutOfFold:
    def _streams_and_folds(self, n_proteins=20, rng_seed=0):
        rng = np.random.default_rng(rng_seed)
        proteins = _make_proteins(n_proteins)
        n_folds = 5
        grouping = [[p["id"] for p in proteins[i::n_folds]] for i in range(n_folds)]
        by_id = {p["id"]: p for p in proteins}

        fold_results = []
        gpu, noise = {}, {}
        for ids in grouping:
            probs, labels = [], []
            for pid in ids:
                p = by_id[pid]
                lab = np.asarray(p["labels"], dtype=np.float32)
                # Weakly informative real stream.
                gpu[pid] = np.clip(0.5 + 0.15 * (lab - 0.5) + rng.normal(0, 0.05, len(lab)), 0, 1
                                   ).astype(np.float32)
                # Pure noise: a stacker must not be able to earn AUC from it.
                noise[pid] = rng.random(len(lab)).astype(np.float32)
                probs.append(gpu[pid])
                labels.append(lab)
            fold_results.append(
                {
                    "val_ids": list(ids),
                    "val_probs": np.concatenate(probs),
                    "val_labels": np.concatenate(labels),
                }
            )
        return proteins, fold_results, {"gpu": gpu, "noise": noise}

    def test_reports_out_of_fold_stacking(self):
        from colab.meta_ensemble import apply_meta_stacker

        proteins, fold_results, streams = self._streams_and_folds()
        report, _ = apply_meta_stacker(proteins, fold_results, streams, n_folds=5)
        assert not report.get("skipped"), report
        assert report["fit"]["stacking"] == "out_of_fold"
        assert report["fit"]["n_meta_folds"] == 5

    def test_oof_score_is_not_the_in_sample_score(self):
        """The pre-fix code reported the resubstitution score as `after`."""
        from colab.meta_ensemble import apply_meta_stacker

        proteins, fold_results, streams = self._streams_and_folds()
        report, _ = apply_meta_stacker(proteins, fold_results, streams, n_folds=5)
        fit = report["fit"]
        assert "oof_auc" in fit and "train_auc" in fit
        # In-sample fit can only be optimistic relative to out-of-fold scoring.
        assert fit["oof_auc"] <= fit["train_auc"] + 1e-9


class TestIsotonicCalibrationIsOutOfFold:
    def _fold_results(self, n_folds=5, per_fold=400, seed=0):
        rng = np.random.default_rng(seed)
        folds = []
        for _ in range(n_folds):
            labels = rng.integers(0, 2, per_fold).astype(np.float32)
            # Deliberately uninformative probabilities: honest AUC is ~0.5.
            probs = rng.random(per_fold).astype(np.float32)
            folds.append({"val_probs": probs, "val_labels": labels})
        return folds

    def test_reports_leave_one_fold_out_fit(self):
        folds = self._fold_results()
        _, report = calibrate_fold_results(folds, method="isotonic")
        assert report["fit"] == "leave_one_fold_out"

    def test_does_not_manufacture_signal_from_noise(self):
        """Fitting isotonic on the residues it transforms used to push a pure-noise
        AUC well above 0.5; leave-one-fold-out keeps it honest."""
        from sklearn.metrics import roc_auc_score

        folds = self._fold_results(per_fold=600)
        updated, report = calibrate_fold_results(folds, method="isotonic")
        labels = np.concatenate([f["val_labels"] for f in folds])
        probs = np.concatenate([f["val_probs"] for f in updated])
        auc = roc_auc_score(labels, probs)
        assert auc == pytest.approx(0.5, abs=0.06), f"noise AUC inflated to {auc:.4f}"
        assert report["auc_after"] == pytest.approx(auc, abs=1e-6)

    def test_temperature_scaling_leaves_ranking_invariant(self):
        """Sanity anchor: temperature scaling is strictly monotone, so unlike
        isotonic it cannot change AUC at all."""
        from sklearn.metrics import roc_auc_score

        folds = self._fold_results(per_fold=300, seed=3)
        updated, _ = calibrate_fold_results(folds, method="temperature")
        labels = np.concatenate([f["val_labels"] for f in folds])
        before = roc_auc_score(labels, np.concatenate([f["val_probs"] for f in folds]))
        after = roc_auc_score(labels, np.concatenate([f["val_probs"] for f in updated]))
        assert after == pytest.approx(before, abs=1e-9)

    def test_single_fold_degrades_gracefully(self):
        folds = self._fold_results(n_folds=1, per_fold=300)
        updated, report = calibrate_fold_results(folds, method="isotonic")
        assert report["fit"] == "in_sample_single_fold"
        assert len(updated) == 1


class TestCvProgressRecordsSplitMethod:
    def test_progress_payload_carries_the_runs_split_method(self):
        """cv_progress.json is the reproducibility artifact; recording fold
        membership under a method the run never used misdescribes the design."""
        from colab.cv_splits import get_fold_val_protein_ids

        proteins = _make_protein_families(n_families=5, per_family=4, length=300)
        protein_ids = get_fold_val_protein_ids(proteins, 5, split_method="protein")
        homology_ids = get_fold_val_protein_ids(
            proteins, 5, split_method="homology", homology_min_identity=0.40
        )
        assert [sorted(f) for f in protein_ids] != [sorted(f) for f in homology_ids]


class TestHomologyClusteringActuallyClusters:
    """The homology split silently degenerated into a protein split.

    ``difflib.SequenceMatcher`` enables ``autojunk``, which for inputs >= 200
    elements treats anything occurring in >1% of positions as junk. Every amino
    acid clears 1%, so for proteins longer than 199 residues the similarity of
    two 95%-identical sequences collapsed to ~0.01, nothing reached the 0.40
    threshold, and every protein became its own cluster.
    """

    def test_identity_survives_the_200_residue_autojunk_cliff(self):
        from colab.homology_splits import sequence_identity

        rng = np.random.default_rng(5)
        for length in (150, 200, 400, 1000):
            base = list(rng.choice(list(_AA), size=length))
            mutant = list(base)
            for pos in rng.choice(length, size=length // 20, replace=False):
                mutant[pos] = rng.choice(list(_AA))
            ident = sequence_identity("".join(base), "".join(mutant))
            assert ident > 0.80, f"length={length} identity collapsed to {ident:.3f}"

    def test_long_protein_families_still_cluster(self):
        from colab.homology_splits import cluster_proteins_by_homology

        proteins = _make_protein_families(n_families=6, per_family=4, length=450)
        cluster_ids, meta = cluster_proteins_by_homology(proteins, min_identity=0.40)

        assert meta["n_clusters"] == 6, meta
        assert not meta["degenerate"]
        # Family members share a cluster; different families do not.
        for f in range(6):
            members = {
                int(cluster_ids[i]) for i, p in enumerate(proteins)
                if p["id"].startswith(f"F{f:02d}")
            }
            assert len(members) == 1

    def test_length_bin_boundary_pair_is_not_missed(self):
        """Fixed 50-residue bins never compared 249 vs 251."""
        from colab.homology_splits import cluster_proteins_by_homology

        rng = np.random.default_rng(9)
        base = "".join(rng.choice(list(_AA), size=249))
        proteins = [
            {"id": "A", "sequence": base, "length": 249},
            {"id": "B", "sequence": base + "AA", "length": 251},
        ]
        cluster_ids, meta = cluster_proteins_by_homology(proteins, min_identity=0.40)
        assert cluster_ids[0] == cluster_ids[1], meta

    def test_fewer_clusters_than_folds_falls_back_instead_of_raising(self):
        from colab.homology_splits import get_homology_cv_splits

        rng = np.random.default_rng(11)
        base = "".join(rng.choice(list(_AA), size=300))
        proteins = [{"id": f"X{i}", "sequence": base, "length": 300} for i in range(8)]
        splits, meta = get_homology_cv_splits(proteins, 5, min_identity=0.40)
        assert len(splits) == 5
        assert meta["fallback"] == "per_protein_groups"

    def test_caid_leakage_audit_detects_a_near_duplicate(self):
        """caid_leakage_audit.json is a required go/no-go artifact; with autojunk
        on it certified 'no leakage' for any protein long enough to matter."""
        from colab.caid_leakage import _seq_identity

        rng = np.random.default_rng(13)
        train = list(rng.choice(list(_AA), size=400))
        caid = list(train)
        for pos in rng.choice(400, size=20, replace=False):
            caid[pos] = rng.choice(list(_AA))
        assert _seq_identity("".join(train), "".join(caid)) > 0.80


class TestWatchdogDetectsStalledPhases:
    """The campaign watchdog reported ``status=running folds=0/10`` for 30 hours
    after the GPU job had already failed 4 seconds in.

    ``ACTIVE`` includes ``PENDING``, and Slurm keeps the dependent clean/package
    jobs queued as ``DependencyNeverSatisfied`` once their dependency fails. The
    ``any(st.active)`` short-circuit therefore read a dead phase as progressing
    and never resubmitted.
    """

    @staticmethod
    def _phase():
        return {
            "kind": "650m",
            "root": "/nonexistent/publish_650m_test",
            "status": "running",
            "job_ids": {"ultra": "1001", "clean": "1002", "package": "1003"},
        }

    def test_failed_main_with_pending_dependents_is_stalled(self):
        from colab.cv_splits import resolve_cv_splits  # noqa: F401  (import sanity)
        from rockfish.publish_campaign import JobState, phase_is_stalled

        states = {
            "1001": JobState("1001", "FAILED", "1:0"),
            "1002": JobState("1002", "PENDING"),
            "1003": JobState("1003", "PENDING"),
        }
        reasons = {
            "1002": "DependencyNeverSatisfied",
            "1003": "DependencyNeverSatisfied",
        }
        assert phase_is_stalled(self._phase(), states, reasons) is True

    def test_all_dependents_dead_pending_is_stalled(self):
        from rockfish.publish_campaign import JobState, phase_is_stalled

        states = {
            "1001": JobState("1001", "PENDING"),
            "1002": JobState("1002", "PENDING"),
            "1003": JobState("1003", "PENDING"),
        }
        reasons = dict.fromkeys(["1001", "1002", "1003"], "DependencyNeverSatisfied")
        assert phase_is_stalled(self._phase(), states, reasons) is True

    def test_genuinely_running_phase_is_not_stalled(self):
        from rockfish.publish_campaign import JobState, phase_is_stalled

        states = {
            "1001": JobState("1001", "RUNNING"),
            "1002": JobState("1002", "PENDING"),
            "1003": JobState("1003", "PENDING"),
        }
        reasons = {"1002": "Dependency", "1003": "Dependency"}
        assert phase_is_stalled(self._phase(), states, reasons) is False

    def test_normal_queue_wait_is_not_stalled(self):
        from rockfish.publish_campaign import JobState, phase_is_stalled

        states = {"1001": JobState("1001", "PENDING")}
        reasons = {"1001": "Resources"}
        assert phase_is_stalled(self._phase(), states, reasons) is False

    def test_advance_campaign_resubmits_a_stalled_phase(self, monkeypatch):
        """End-to-end: the stalled phase must trigger a resubmit, not be reported
        as running."""
        import rockfish.publish_campaign as pc

        phase = self._phase()
        campaign = {
            "status": "running",
            "resubmit_count": 0,
            "max_resubmits": 8,
            "phases": [phase],
        }
        submitted: list = []
        monkeypatch.setattr(pc, "package_ready", lambda root: False)
        monkeypatch.setattr(pc, "cancel_jobs", lambda ids, **kw: None)
        monkeypatch.setattr(
            pc, "_submit_kind",
            lambda camp, ph, dry_run=False: submitted.append(ph["kind"]) or ph,
        )

        states = {
            "1001": pc.JobState("1001", "FAILED", "1:0"),
            "1002": pc.JobState("1002", "PENDING"),
            "1003": pc.JobState("1003", "PENDING"),
        }
        reasons = dict.fromkeys(["1002", "1003"], "DependencyNeverSatisfied")

        pc.advance_campaign(
            campaign,
            job_query=lambda ids: states,
            reason_query=lambda ids: reasons,
        )
        assert submitted == ["650m"], "stalled phase was not resubmitted"
        assert campaign["resubmit_count"] == 1


class TestNoInferenceModeInTrainingLoop:
    """fair-esm's RotaryEmbedding memoises cos/sin tables keyed by sequence length.

    A table first created inside ``torch.inference_mode()`` is permanently an
    inference tensor, so the next *training* batch at that same length dies with
    "Inference tensors cannot be saved for backward". The cache is length-keyed
    and the ESM backbone is shared across folds, so the crash surfaces at an
    arbitrary later fold — it killed the 650M publish run at fold 3 after 4.9
    hours of GPU time, having passed folds 1 and 2 cleanly.
    """

    def test_eval_and_inference_paths_use_no_grad(self):
        import pathlib

        offenders = []
        for path in [
            pathlib.Path("colab/disordernet_gpu.py"),
            pathlib.Path("colab/fold_model_soup.py"),
            pathlib.Path("colab/predict_batch.py"),
            pathlib.Path("colab/inference_tta.py"),
        ]:
            if not path.exists():
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "inference_mode" in stripped:
                    offenders.append(f"{path}:{lineno}: {stripped}")
        assert not offenders, (
            "torch.inference_mode() poisons fair-esm's rotary cache and breaks a "
            "later training fold. Use torch.no_grad().\n  " + "\n  ".join(offenders)
        )

    def test_rotary_cache_survives_eval_then_train(self):
        """Reproduce the mechanism directly on a rotary-style length-keyed cache."""
        import torch

        class LengthCachedTable(torch.nn.Module):
            """Mirrors fair-esm RotaryEmbedding's memoisation."""

            def __init__(self):
                super().__init__()
                self._cached_len = None
                self._cached = None
                self.w = torch.nn.Parameter(torch.ones(4))

            def forward(self, x):
                n = x.shape[0]
                if self._cached_len != n:
                    self._cached_len = n
                    self._cached = torch.ones(n, 4)
                # Order matters: the cached tensor must multiply a grad-requiring
                # value so autograd has to SAVE it for backward. That save is what
                # rejects an inference tensor, and it is what rotary cos/sin does.
                return (x * self.w) * self._cached

        # Populate the cache under no_grad (the fix), then train at that length.
        m = LengthCachedTable()
        with torch.no_grad():
            m(torch.randn(7, 4))
        loss = m(torch.randn(7, 4)).sum()
        loss.backward()  # must not raise
        assert m.w.grad is not None

        # And confirm inference_mode genuinely breaks it, so this test is not vacuous.
        m2 = LengthCachedTable()
        with torch.inference_mode():
            m2(torch.randn(7, 4))
        with pytest.raises(RuntimeError, match="[Ii]nference tensor"):
            m2(torch.randn(7, 4)).sum().backward()
