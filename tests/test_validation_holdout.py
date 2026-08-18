"""The fixed validation set exists to make two runs comparable.

Cross-validation as this trainer computes it is anti-correlated with CAID3
across our own checkpoints — Spearman -0.60 on Disorder-PDB, -0.43 on Linker,
+0.03 on Binding-IDR — and choosing by it would have cost 0.1117 AUC on
Binding-IDR. The cause is not that validation is useless but that those numbers
were never comparable: windowing changes the units from proteins to windows,
and a wider leak filter changes which proteins remain to score.

So the holdout has exactly one job, and these tests check that job: the same
chains, in every run, whatever the run does. Everything else about it is
secondary to that.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from rockfish.train_multitask import (  # noqa: E402
    HOLDOUT_FRACTION,
    HOLDOUT_SALT,
    chunk_long_rows,
    in_validation_holdout,
    reserve_validation_holdout,
)


def row(rid, seq, tasks=("disorder_pdb",)):
    n = len(seq)
    return {
        "id": rid, "sequence": seq, "length": n,
        "task_labels": {t: np.zeros(n, np.int8) for t in tasks},
        "task_evidence": {t: np.ones(n, bool) for t in tasks},
    }


def seqs(n, length=60, seed=0):
    rng = np.random.default_rng(seed)
    aa = np.frombuffer(b"ACDEFGHIKLMNPQRSTVWY", dtype=np.uint8)
    return ["".join(chr(c) for c in rng.choice(aa, size=length))
            for _ in range(n)]


class TestMembershipIsAPropertyOfTheSequence:
    def test_the_same_sequence_always_lands_the_same_way(self):
        s = "MKVLAAGIVGWTSQ" * 4
        assert all(in_validation_holdout(s) == in_validation_holdout(s)
                   for _ in range(50))

    def test_membership_does_not_depend_on_the_rest_of_the_union(self):
        """A cluster id, a row index or a per-run RNG would all shift when the
        input set changes — and every checkpoint here was filtered against a
        different set of benchmark references."""
        pool = seqs(400, seed=1)
        first = {s: in_validation_holdout(s) for s in pool}
        for subset in (pool[:50], pool[100:], pool[::3]):
            for s in subset:
                assert in_validation_holdout(s) == first[s]

    def test_roughly_the_requested_fraction_is_selected(self):
        pool = seqs(4000, seed=2)
        got = sum(in_validation_holdout(s, 0.05) for s in pool) / len(pool)
        assert 0.03 < got < 0.07, got

    def test_a_larger_fraction_is_a_superset_of_a_smaller_one(self):
        """Nesting means a run that reserves 10% still holds out everything a
        5% run did, so the two remain partially comparable rather than
        disjointly incomparable."""
        pool = seqs(2000, seed=3)
        small = {s for s in pool if in_validation_holdout(s, 0.05)}
        big = {s for s in pool if in_validation_holdout(s, 0.10)}
        assert small <= big
        assert len(big) > len(small)

    def test_a_zero_fraction_selects_nothing(self):
        assert not any(in_validation_holdout(s, 0.0) for s in seqs(500, seed=4))

    def test_the_salt_is_fixed_in_the_source(self):
        """A tunable salt reintroduces exactly the problem the hash solves."""
        assert HOLDOUT_SALT == b"disordernet-validation-holdout-v1"
        src = open(os.path.join(REPO, "rockfish",
                                "train_multitask.py")).read()
        assert 'HOLDOUT_SALT = b"disordernet-validation-holdout-v1"' in src
        assert "--holdout-salt" not in src, "the salt must not be an argument"


class TestWindowsMoveWithTheirProtein:
    def test_every_window_of_a_protein_lands_on_the_same_side(self):
        """Windows of one chain are near-identical. Splitting them would put
        the validation protein's own sequence in the training set."""
        long_seq = "".join(seqs(1, length=3000, seed=5))
        rows, _ = chunk_long_rows([row("P1", long_seq)], max_len=1022,
                                  stride=511)
        assert len(rows) > 1, "the fixture must actually produce windows"
        keys = {r.get("parent_sequence") for r in rows}
        assert keys == {long_seq}
        sides = {in_validation_holdout(r["parent_sequence"]) for r in rows}
        assert len(sides) == 1

    def test_a_short_protein_also_carries_its_own_sequence_as_parent(self):
        rows, _ = chunk_long_rows([row("P2", "MKVL" * 20)], max_len=1022)
        assert rows[0]["parent_sequence"] == rows[0]["sequence"]


class TestHomologuesAreReservedToo:
    """The stub here returns what BLAST actually returns.

    An earlier version of these tests stubbed it as a set of protein ids. The
    real function returns ``(query_index, subject_index, identity)`` triples,
    so the production code's ``set(hits)`` was a set of tuples and its
    membership test never matched — not one homologue left training, while the
    count printed in the log looked correct and every test passed. A fake that
    does not match the interface it fakes proves nothing about the code that
    calls the real one.
    """

    @staticmethod
    def _split(rows, homologous_ids):
        """Run the reservation with a homology filter stubbed at its real
        contract: index triples, not ids."""
        import rockfish.train_multitask as tm

        mod = sys.modules.setdefault("colab.homology_splits",
                                     type(sys)("colab.homology_splits"))
        old = getattr(mod, "blast_cross_identity_hits", None)

        def fake(query, subject, min_identity):
            if homologous_ids is None:
                return None
            return [(i, 0, 0.91) for i, q in enumerate(query)
                    if q["id"] in homologous_ids]

        mod.blast_cross_identity_hits = fake
        try:
            return tm.reserve_validation_holdout(rows, 0.05, 0.4)
        finally:
            if old is None:
                delattr(mod, "blast_cross_identity_hits")
            else:
                mod.blast_cross_identity_hits = old

    def _rows_with_a_seed(self):
        pool = seqs(600, seed=6)
        seed = next(s for s in pool if in_validation_holdout(s))
        others = [s for s in pool if not in_validation_holdout(s)][:20]
        rows = [row("SEED", seed)] + [row(f"O{i}", s)
                                      for i, s in enumerate(others)]
        return rows, seed

    def test_a_homologue_of_a_held_out_protein_leaves_training(self):
        rows, _ = self._rows_with_a_seed()
        train, holdout, stats = self._split(rows, {"O3"})
        assert {r["id"] for r in holdout} >= {"SEED", "O3"}
        assert "O3" not in {r["id"] for r in train}
        assert stats["n_homologous"] == 1

    def test_an_exact_duplicate_sequence_leaves_training(self):
        """A duplicate under a different id hashes identically, so it is
        selected by the hash itself and never needs the exact-match pass."""
        rows, seed = self._rows_with_a_seed()
        rows.append(row("DUP", seed))
        train, holdout, _ = self._split(rows, set())
        assert "DUP" not in {r["id"] for r in train}
        assert "DUP" in {r["id"] for r in holdout}

    def test_a_window_matching_a_held_out_sequence_leaves_training(self):
        """The one case the hash alone misses: a row whose parent is a
        different protein but whose own sequence is a held-out chain — a window
        of a longer protein that contains it. The exact-match pass exists for
        this, and without it the validation sequence would be in training."""
        rows, seed = self._rows_with_a_seed()
        embedded = dict(row("WINDOW", seed))
        embedded["parent_sequence"] = "M" + seed + "M"   # hashes elsewhere
        assert not in_validation_holdout(embedded["parent_sequence"])
        rows.append(embedded)

        train, holdout, stats = self._split(rows, set())
        assert "WINDOW" not in {r["id"] for r in train}
        assert "WINDOW" in {r["id"] for r in holdout}
        assert stats["n_exact_id_or_sequence"] == 1

    def test_nothing_is_in_both_halves(self):
        rows, _ = self._rows_with_a_seed()
        train, holdout, _ = self._split(rows, {"O1", "O7"})
        assert not ({r["id"] for r in train} & {r["id"] for r in holdout})
        assert len(train) + len(holdout) == len(rows)

    def test_missing_blast_refuses_rather_than_holding_out_nothing(self):
        """Holding out a protein while training on its paralogue holds out
        nothing, and a validation number computed that way is worse than
        none — it looks like a measurement."""
        rows, _ = self._rows_with_a_seed()
        with pytest.raises(SystemExit) as exc:
            self._split(rows, None)
        assert "holds out nothing" in str(exc.value)

    def test_missing_blast_may_be_overridden_knowingly(self):
        import rockfish.train_multitask as tm

        rows, _ = self._rows_with_a_seed()
        mod = sys.modules.setdefault("colab.homology_splits",
                                     type(sys)("colab.homology_splits"))
        mod.blast_cross_identity_hits = lambda *a, **k: None
        try:
            train, holdout, stats = tm.reserve_validation_holdout(
                rows, 0.05, 0.4, allow_missing_homology=True)
        finally:
            del mod.blast_cross_identity_hits
        assert holdout and train
        assert stats["homology_filter"] == "skipped"

    def test_disabling_the_holdout_returns_everything_for_training(self):
        import rockfish.train_multitask as tm

        rows, _ = self._rows_with_a_seed()
        train, holdout, stats = tm.reserve_validation_holdout(rows, 0.0, 0.4)
        assert holdout == []
        assert len(train) == len(rows)
        assert stats["n_holdout"] == 0


class TestTheTrainerUsesIt:
    @staticmethod
    def _src():
        return open(os.path.join(REPO, "rockfish", "train_multitask.py")).read()

    def test_the_split_happens_before_the_folds_are_built(self):
        src = self._src()
        assert src.index("rows, holdout_rows, holdout_stats = "
                         "reserve_validation_holdout(") < \
            src.index("folds = homology_folds(")

    def test_the_split_happens_after_the_benchmark_leak_filter(self):
        """Reserving before the CAID filter would let a benchmark target sit in
        the validation set, where its score would be memorisation."""
        src = self._src()
        assert src.index("rows, leak = drop_caid_targets(") < \
            src.index("rows, holdout_rows, holdout_stats = ")

    def test_the_default_fraction_is_the_module_constant(self):
        src = self._src()
        assert '"--holdout-fraction", type=float, default=HOLDOUT_FRACTION' in src
        assert HOLDOUT_FRACTION == 0.05

    def test_the_holdout_score_is_written_into_the_checkpoint(self):
        """A comparison between two checkpoints has to be able to verify they
        held out the same chains, not assume it."""
        src = self._src()
        block = src[src.index("atomic_torch_save({"):]
        assert '"validation_holdout"' in block[:2000]

    def test_the_final_model_is_scored_on_it(self):
        src = self._src()
        assert "scoring the fixed validation holdout" in src
        assert src.index("fitting final head on all") < \
            src.index("scoring the fixed validation holdout")


class TestTheHomologyContractIsPinned:
    """The bug was a mismatch between a stub and the function it stood for.

    So the contract itself is asserted, against the real function's source and
    against the other caller that already had it right, rather than left to be
    re-learned.
    """

    @staticmethod
    def _src(name):
        return open(os.path.join(REPO, *name.split("/"))).read()

    def test_the_filter_documents_index_triples(self):
        src = self._src("colab/homology_splits.py")
        block = src[src.index("def blast_cross_identity_hits"):]
        assert "(query_index, subject_index, identity)" in block[:1200]

    def test_both_callers_map_indices_back_to_ids(self):
        src = self._src("rockfish/train_multitask.py")
        assert src.count('homologous = {kept[q]["id"] for q, _s, _i in hits}') == 2, (
            "both the CAID filter and the validation holdout must map BLAST's "
            "index triples back to ids")
        assert "homologous = set(hits)" not in src

    def test_a_set_of_ids_would_not_have_worked(self):
        """The counterfactual, so the failure mode stays legible: membership of
        an id in a set of tuples is always False."""
        hits = [(0, 3, 0.9), (7, 1, 0.95)]
        assert "O3" not in set(hits)
        assert {("id",)} != {"id"}
