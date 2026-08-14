"""Long proteins belong in training, and their windows belong in one fold.

2,440 proteins were dropped as too_long from the pdb_missing source alone — 11%
of the data, and the hard part of it. The model was then asked to predict
proteins of exactly that kind at test time and scored 0.5183 on CAID3
Disorder-NOX targets above 1500 residues, against 0.8967 for a training-free
structural feature.

Keeping them as overlapping windows introduces a leak that is easy to miss:
two windows of one protein are near-duplicates, so if they land in different
folds the model trains on a sequence it is later evaluated on. That is the exact
failure homology clustering exists to prevent, reintroduced underneath it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rockfish.train_multitask import chunk_long_rows

TASKS = ("disorder_nox", "disorder_pdb")


def row(pid, n, with_structure=True):
    rng = np.random.default_rng(abs(hash(pid)) % 2**31)
    r = {
        "id": pid, "sequence": "".join(rng.choice(list("ACDEFGHIKLM"), n)),
        "length": n, "uniprot_acc": f"P{pid}",
        "task_labels": {t: rng.integers(0, 2, n).astype(np.int8) for t in TASKS},
        "task_evidence": {t: rng.random(n) > 0.2 for t in TASKS},
    }
    if with_structure:
        for k in ("rsa", "plddt", "contacts", "structure_available"):
            r[k] = rng.random(n).astype(np.float32)
    return r


class TestChunking:
    def test_short_proteins_pass_through_untouched(self):
        r = row("A", 300)
        out, stats = chunk_long_rows([r], max_len=1022)
        assert len(out) == 1
        assert out[0]["id"] == "A"
        assert stats["proteins_chunked"] == 0

    def test_a_long_protein_becomes_several_windows(self):
        out, stats = chunk_long_rows([row("L", 3000)], max_len=1022)
        assert stats["proteins_chunked"] == 1
        assert len(out) >= 4
        assert all(o["length"] <= 1022 for o in out)

    def test_every_residue_appears_in_some_window(self):
        n = 2600
        out, _ = chunk_long_rows([row("L", n)], max_len=1022)
        seen = np.zeros(n, dtype=int)
        for o in out:
            a = o["window_offset"]
            seen[a:a + o["length"]] += 1
        assert (seen > 0).all()

    def test_labels_and_evidence_track_the_window(self):
        src = row("L", 2500)
        out, _ = chunk_long_rows([src], max_len=1022)
        for o in out:
            a, n = o["window_offset"], o["length"]
            for t in TASKS:
                assert np.array_equal(o["task_labels"][t],
                                      src["task_labels"][t][a:a + n])
                assert np.array_equal(o["task_evidence"][t],
                                      src["task_evidence"][t][a:a + n])

    def test_structure_is_sliced_to_the_same_window(self):
        """Otherwise the head sees residue i's embedding beside residue
        a+i's accessibility — a silent misalignment, not an error."""
        src = row("L", 2500)
        out, _ = chunk_long_rows([src], max_len=1022)
        for o in out:
            a, n = o["window_offset"], o["length"]
            for k in ("rsa", "plddt", "contacts", "structure_available"):
                assert np.array_equal(o[k], src[k][a:a + n]), k

    def test_sequence_is_sliced_to_the_same_window(self):
        src = row("L", 2500)
        out, _ = chunk_long_rows([src], max_len=1022)
        for o in out:
            a, n = o["window_offset"], o["length"]
            assert o["sequence"] == src["sequence"][a:a + n]

    def test_windows_carry_their_parent(self):
        out, _ = chunk_long_rows([row("L", 2500)], max_len=1022)
        assert {o["parent"] for o in out} == {"L"}
        assert len({o["id"] for o in out}) == len(out), "window ids must differ"

    def test_short_rows_also_carry_a_parent(self):
        """Fold assignment groups on parent, so it must exist on every row."""
        out, _ = chunk_long_rows([row("A", 200)], max_len=1022)
        assert out[0]["parent"] == "A"

    @pytest.mark.parametrize("n", [1023, 1500, 2048, 3088, 5000])
    def test_no_window_exceeds_the_model_limit(self, n):
        out, _ = chunk_long_rows([row("L", n)], max_len=1022)
        assert all(o["length"] <= 1022 for o in out)


class TestWindowsShareAFold:
    """The leak this guards: two windows of one protein on opposite sides of a
    train/validation split are a near-duplicate straddling it — exactly what
    homology clustering exists to prevent, reintroduced underneath it.

    The clustering itself is stubbed. Real BLAST on a dozen synthetic sequences
    trips the degenerate-split guard, and what needs testing is our grouping,
    not theirs: that clustering is asked about parents and the answer is applied
    to every window of that parent.
    """

    @staticmethod
    def _stub(monkeypatch, seen):
        import colab.homology_splits as hs

        def fake(proteins, min_identity=0.4, **kw):
            seen.extend(p["id"] for p in proteins)
            # every protein its own cluster: the hardest case for grouping,
            # since nothing merges windows for us
            return list(range(len(proteins))), {"degenerate": False}

        monkeypatch.setattr(hs, "cluster_proteins_by_homology_cached", fake)

    def test_windows_of_one_protein_land_in_one_fold(self, monkeypatch):
        from rockfish.train_multitask import homology_folds

        seen: list[str] = []
        self._stub(monkeypatch, seen)
        rows = [row(f"S{i}", 400) for i in range(9)]
        rows.extend(chunk_long_rows([row("LONG", 3000)], max_len=1022)[0])
        rows.extend(chunk_long_rows([row("LONG2", 2600)], max_len=1022)[0])

        folds = homology_folds(rows, n_folds=3, min_identity=0.4, seed=0)

        where: dict[str, set] = {}
        for fi, members in enumerate(folds):
            for idx in members:
                where.setdefault(rows[idx].get("parent", rows[idx]["id"]),
                                 set()).add(fi)
        for parent, fs in where.items():
            assert len(fs) == 1, f"{parent} split across folds {sorted(fs)}"

    def test_clustering_is_asked_about_parents_not_windows(self, monkeypatch):
        """Clustering every window would burn BLAST time rediscovering that a
        protein resembles itself."""
        from rockfish.train_multitask import homology_folds

        seen: list[str] = []
        self._stub(monkeypatch, seen)
        rows = [row(f"S{i}", 400) for i in range(9)]
        long_windows = chunk_long_rows([row("LONG", 3000)], max_len=1022)[0]
        rows.extend(long_windows)

        homology_folds(rows, n_folds=3, min_identity=0.4, seed=0)

        assert "LONG" in seen
        assert not any("#w" in s for s in seen), seen
        assert len(seen) == 10, seen

    def test_every_row_is_assigned_exactly_once(self, monkeypatch):
        from rockfish.train_multitask import homology_folds

        self._stub(monkeypatch, [])
        rows = [row(f"S{i}", 400) for i in range(9)]
        rows.extend(chunk_long_rows([row("LONG", 3000)], max_len=1022)[0])
        folds = homology_folds(rows, n_folds=3, min_identity=0.4, seed=0)
        flat = [i for f in folds for i in f]
        assert sorted(flat) == list(range(len(rows)))
