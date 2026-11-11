"""Fail before the GPU-hours, not after them.

Three lite_frozen replicates died 1.2-1.5 hours in, at fold boundaries, on two
different nodes, exit code 1, with no traceback in any log. The home directory
had hit its 50 GB quota: the checkpoint write failed, and so did the write of
the traceback reporting it. A full disk is the one failure that also destroys
its own evidence — so it has to be caught up front.

df cannot detect this. It reported 13 TB free on the filesystem while the
per-user quota was exhausted, and quota(1) reported no quota at all. Only an
actual write tests the condition that matters.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from rockfish.run_disordernet import preflight_writable_space


class TestPreflight:
    def test_passes_on_a_writable_directory(self, tmp_path):
        preflight_writable_space(str(tmp_path), probe_mb=1)   # raises if not
        assert tmp_path.is_dir()
        assert not list(tmp_path.glob("*probe*")), "probe file left behind"

    def test_creates_the_directory_if_absent(self, tmp_path):
        target = tmp_path / "checkpoints"
        preflight_writable_space(str(target), probe_mb=1)
        assert target.is_dir()

    def test_leaves_no_probe_file_behind(self, tmp_path):
        preflight_writable_space(str(tmp_path), probe_mb=1)
        assert list(tmp_path.iterdir()) == []

    def test_a_quota_error_aborts_the_run(self, tmp_path):
        """errno 122 (EDQUOT) is exactly what rsync reported on the cluster."""
        real_open = open

        def fail_on_probe(path, *a, **kw):
            if ".preflight_" in str(path):
                raise OSError(122, "Disk quota exceeded")
            return real_open(path, *a, **kw)

        with mock.patch("builtins.open", fail_on_probe):
            with pytest.raises(SystemExit) as exc:
                preflight_writable_space(str(tmp_path), probe_mb=1)
        assert "Disk quota exceeded" in str(exc.value)

    def test_the_error_explains_the_silent_failure_mode(self, tmp_path):
        with mock.patch("builtins.open", side_effect=OSError(28, "No space left")):
            with pytest.raises(SystemExit) as exc:
                preflight_writable_space(str(tmp_path), probe_mb=1)
        msg = str(exc.value)
        assert "DISORDERNET_RESULTS" in msg, "must say where to put runs instead"
        assert "du -sh" in msg, "df does not show the quota; say what does"

    def test_probe_cleaned_up_even_when_the_write_fails(self, tmp_path):
        """A partial probe left behind consumes the space it was testing for."""
        real_open = open
        state = {}

        def partial_write(path, *a, **kw):
            if ".preflight_" in str(path):
                fh = real_open(path, *a, **kw)
                state["path"] = path
                orig = fh.write

                def w(data):
                    orig(data)
                    raise OSError(122, "Disk quota exceeded")
                fh.write = w
                return fh
            return real_open(path, *a, **kw)

        with mock.patch("builtins.open", partial_write):
            with pytest.raises(SystemExit):
                preflight_writable_space(str(tmp_path), probe_mb=4)
        assert not os.path.exists(state["path"])
        assert list(tmp_path.iterdir()) == []


class TestCli:
    def test_preflight_is_on_by_default(self):
        from rockfish.run_disordernet import build_parser

        assert build_parser().parse_args(["cv"]).skip_space_preflight is False

    def test_probe_size_is_configurable(self):
        from rockfish.run_disordernet import build_parser

        args = build_parser().parse_args(["cv", "--preflight-mb", "512"])
        assert args.preflight_mb == 512

    def test_default_probe_is_the_size_of_a_real_checkpoint(self):
        """A 1 MB probe would pass on a disk too full for a 15 MB fold."""
        from rockfish.run_disordernet import build_parser

        assert build_parser().parse_args(["cv"]).preflight_mb >= 64
