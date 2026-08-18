"""The SASA memo must be a speed change and nothing else.

Shrake-Rupley cost 5h04m of an 18h training run, recomputing accessibilities
that depend on nothing that changed between runs. Caching them is worth an hour
of wall clock per run and is exactly the kind of optimisation that quietly
changes a number: a stale entry, a dropped channel, a shortened array, and the
model trains on different inputs than the run it is compared against.

So these tests do not check that the cache is fast. They check that it returns
the same arrays as the computation it replaces, that it notices when the
structure underneath it changes, and that it refuses to interpret the older
cache layout — which stored rsa/plddt/seq and predates the contacts channel,
and would therefore have silently removed a model input.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colab import structure_rsa as sr  # noqa: E402


@pytest.fixture
def fake_structure(tmp_path, monkeypatch):
    """A .cif whose parse is a counted stub, so cache hits are observable."""
    cif = tmp_path / "P12345.cif"
    cif.write_text("data_P12345\n# not a real mmCIF; the parse is stubbed\n")

    seq = "MKVLAAGIVGWTS"
    n = len(seq)
    truth = {
        "rsa": np.linspace(0.1, 0.9, n).astype(np.float32),
        "seq": seq,
        "plddt": np.linspace(30.0, 95.0, n).astype(np.float32),
        "contacts": np.arange(n, dtype=np.float32),
        # Signed on purpose: the channel exists to be odd under reflection.
        "ca_torsion": np.linspace(-170.0, 170.0, n).astype(np.float32),
    }
    calls = {"n": 0}

    def stub(path):
        calls["n"] += 1
        return (truth["rsa"], truth["seq"], truth["plddt"], truth["contacts"],
                truth["ca_torsion"])

    monkeypatch.setattr(sr, "rsa_from_structure", stub)
    return {"cif": str(cif), "cache": str(tmp_path / "cache"),
            "truth": truth, "calls": calls, "dir": str(tmp_path)}


class TestTheCacheReturnsWhatItReplaces:
    def test_first_call_computes_and_second_reads(self, fake_structure):
        f = fake_structure
        a = sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 1
        b = sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 1, "the second call recomputed"

        for i, key in enumerate(("rsa", "seq", "plddt", "contacts",
                                 "ca_torsion")):
            if key == "seq":
                assert a[i] == b[i] == f["truth"]["seq"]
            else:
                np.testing.assert_array_equal(a[i], b[i])
                np.testing.assert_array_equal(a[i], f["truth"][key])

    def test_the_cached_values_are_bit_identical_not_merely_close(
            self, fake_structure):
        """float32 through npz must round-trip exactly. 'Close' would let a
        cached run and an uncached run disagree in the last bit and make an
        ablation irreproducible."""
        f = fake_structure
        fresh = sr.rsa_from_structure_cached(f["cif"], None)
        cached = sr.rsa_from_structure_cached(f["cif"], f["cache"])
        cached = sr.rsa_from_structure_cached(f["cif"], f["cache"])
        for i in (0, 2, 3, 4):
            assert cached[i].dtype == fresh[i].dtype
            np.testing.assert_array_equal(cached[i], fresh[i])

    def test_passing_no_cache_never_writes_anything(self, fake_structure):
        f = fake_structure
        sr.rsa_from_structure_cached(f["cif"], None)
        sr.rsa_from_structure_cached(f["cif"], None)
        assert f["calls"]["n"] == 2
        assert not os.path.exists(f["cache"])

    def test_no_part_files_are_left_behind(self, fake_structure):
        """np.savez_compressed appends '.npz' to a path lacking it, so a naive
        temp name would leave a file beside the one we rename and the rename
        would fail every time."""
        f = fake_structure
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        left = os.listdir(f["cache"])
        assert left == ["P12345.npz"], left


class TestTheCacheInvalidatesItself:
    def test_a_changed_structure_is_recomputed(self, fake_structure):
        f = fake_structure
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 1

        # AlphaFold DB moved v4 -> v6 during this project. An accession-keyed
        # cache would have served the old model's features forever.
        with open(f["cif"], "a") as fh:
            fh.write("# refreshed from AFDB v6\n")
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 2, "a changed mmCIF was served from cache"

    def test_the_digest_is_content_not_mtime(self, fake_structure):
        f = fake_structure
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        os.utime(f["cif"], (0, 0))          # rsync rewrites mtime routinely
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 1, "a touch alone invalidated the cache"

    def test_a_truncated_cache_file_is_recomputed_not_raised(
            self, fake_structure):
        f = fake_structure
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        with open(os.path.join(f["cache"], "P12345.npz"), "r+b") as fh:
            fh.truncate(12)
        out = sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 2
        np.testing.assert_array_equal(out[0], f["truth"]["rsa"])


class TestTheOlderLayoutIsNotMisread:
    def test_a_v1_file_without_contacts_is_ignored(self, fake_structure):
        """scratch/rsa_features holds 2,529 files with rsa/plddt/seq and no
        contacts. Reading one as current would drop a model input and nothing
        would look wrong."""
        f = fake_structure
        os.makedirs(f["cache"], exist_ok=True)
        np.savez_compressed(
            os.path.join(f["cache"], "P12345.npz"),
            rsa=np.zeros(13, np.float32), plddt=np.zeros(13, np.float32),
            seq="MKVLAAGIVGWTS")

        out = sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 1, "the v1 layout was read as current"
        np.testing.assert_array_equal(out[3], f["truth"]["contacts"])

    def test_a_file_with_a_wrong_version_is_ignored(self, fake_structure):
        f = fake_structure
        os.makedirs(f["cache"], exist_ok=True)
        with open(os.path.join(f["cache"], "P12345.npz"), "wb") as fh:
            np.savez_compressed(
                fh, version=sr.RSA_CACHE_VERSION + 1,
                cif_digest=sr.cif_digest(f["cif"]),
                rsa=np.zeros(13, np.float32), seq="MKVLAAGIVGWTS",
                plddt=np.zeros(13, np.float32),
                contacts=np.zeros(13, np.float32))
        sr.rsa_from_structure_cached(f["cif"], f["cache"])
        assert f["calls"]["n"] == 1


class TestStructureFeaturesIsUnchangedByCaching:
    def test_cached_and_uncached_features_are_identical(self, fake_structure):
        f = fake_structure
        seq = f["truth"]["seq"]
        plain = sr.structure_features("P12345", seq, f["dir"],
                                      feature_cache=None)
        warm = sr.structure_features("P12345", seq, f["dir"])
        again = sr.structure_features("P12345", seq, f["dir"])
        assert plain is not None and warm is not None and again is not None
        for key in ("rsa", "rsa_raw", "plddt", "contacts", "ca_torsion",
                    "handedness"):
            np.testing.assert_array_equal(plain[key], warm[key])
            np.testing.assert_array_equal(plain[key], again[key])
        assert plain["window"] == again["window"] == sr.RSA_SMOOTH_WINDOW

    def test_the_default_cache_lives_under_the_structure_directory(
            self, fake_structure):
        f = fake_structure
        sr.structure_features("P12345", f["truth"]["seq"], f["dir"])
        assert os.path.isfile(os.path.join(f["dir"], sr.RSA_CACHE_DIRNAME,
                                           "P12345.npz"))

    def test_an_explicit_none_disables_the_cache(self, fake_structure):
        f = fake_structure
        sr.structure_features("P12345", f["truth"]["seq"], f["dir"],
                              feature_cache=None)
        assert not os.path.isdir(os.path.join(f["dir"], sr.RSA_CACHE_DIRNAME))

    def test_a_sequence_mismatch_still_returns_none_when_cached(
            self, fake_structure):
        """The alignment check is the one guard that must survive caching: a
        cached structure for the wrong isoform is exactly the error that made
        this project's CAID3 numbers wrong for its whole history."""
        f = fake_structure
        assert sr.structure_features("P12345", f["truth"]["seq"], f["dir"])
        assert sr.structure_features("P12345", "MKVLAAGIVGWTSQQQ",
                                     f["dir"]) is None

    def test_the_smoothing_window_is_still_a_free_parameter(
            self, fake_structure):
        """Raw rsa is cached, not smoothed rsa. Caching the smoothed values
        would freeze a window that was chosen on the benchmark."""
        f = fake_structure
        seq = f["truth"]["seq"]
        w5 = sr.structure_features("P12345", seq, f["dir"], window=5)
        w21 = sr.structure_features("P12345", seq, f["dir"], window=21)
        assert not np.array_equal(w5["rsa"], w21["rsa"])
        np.testing.assert_array_equal(w5["rsa_raw"], w21["rsa_raw"])


class TestThePrefillScript:
    def test_it_reports_a_failure_rate_rather_than_dying_on_one_structure(self):
        import rockfish.build_rsa_cache as b

        acc, n, err = b._one(("/nonexistent/ZZZZZZ.cif", None))
        assert acc == "ZZZZZZ"
        assert n == 0
        assert err, "a failed parse must be reported, not swallowed"

    def test_a_wholesale_parse_failure_is_a_nonzero_exit(self, tmp_path):
        """An empty cache would let the next training run pay the full 5h
        again while the prefill job reported success."""
        import rockfish.build_rsa_cache as b

        for i in range(5):
            (tmp_path / f"X{i}.cif").write_text("not an mmCIF")
        rc = b.main(["--structures", str(tmp_path), "--workers", "1"])
        assert rc == 1


class TestAgainstRealStructures:
    """The stubbed tests above never run Shrake-Rupley.

    This project has already been bitten by an audit that checked the shape of
    the code instead of the behaviour of it — the seventh evidence-sentinel leak
    passed a syntactic audit and was only caught by an integration test through
    the real path. So the cache is also checked against real mmCIF files, where
    they exist.
    """

    @staticmethod
    def _structures():
        return os.environ.get("DISORDERNET_AF_STRUCTURES", "")

    @pytest.mark.skipif(
        not os.environ.get("DISORDERNET_AF_STRUCTURES"),
        reason="set DISORDERNET_AF_STRUCTURES to a directory of AlphaFold .cif "
               "files (Biopython is not installed off the cluster)")
    def test_real_parses_round_trip_through_the_cache_exactly(self, tmp_path):
        d = self._structures()
        cifs = sorted(f for f in os.listdir(d) if f.endswith(".cif"))[:25]
        assert cifs, f"no .cif files under {d}"

        cache = str(tmp_path / "c")
        n_checked = 0
        for name in cifs:
            path = os.path.join(d, name)
            fresh = sr.rsa_from_structure_cached(path, None)
            sr.rsa_from_structure_cached(path, cache)     # populate
            warm = sr.rsa_from_structure_cached(path, cache)  # read back
            assert warm[1] == fresh[1], name
            for i in (0, 2, 3, 4):
                assert warm[i].dtype == fresh[i].dtype, name
                np.testing.assert_array_equal(warm[i], fresh[i], err_msg=name,
                                              strict=False)
            n_checked += 1
        assert n_checked >= 1
        # Every structure produced a file, so nothing was silently skipped.
        assert len(os.listdir(cache)) == n_checked

    @pytest.mark.skipif(
        not os.environ.get("DISORDERNET_AF_STRUCTURES"),
        reason="set DISORDERNET_AF_STRUCTURES to a directory of AlphaFold .cif "
               "files (Biopython is not installed off the cluster)")
    def test_the_cache_is_faster_than_the_parse_it_replaces(self, tmp_path):
        """The whole point is wall clock. If the memo were not faster it would
        be pure risk, so this is worth asserting rather than assuming."""
        import time

        d = self._structures()
        cifs = sorted(f for f in os.listdir(d) if f.endswith(".cif"))[:10]
        assert cifs
        cache = str(tmp_path / "c")

        t0 = time.time()
        for name in cifs:
            sr.rsa_from_structure_cached(os.path.join(d, name), cache)
        cold = time.time() - t0

        t0 = time.time()
        for name in cifs:
            sr.rsa_from_structure_cached(os.path.join(d, name), cache)
        warm = time.time() - t0

        assert warm < cold / 3.0, (
            f"cache read {warm:.2f}s vs parse {cold:.2f}s — not worth the risk")
