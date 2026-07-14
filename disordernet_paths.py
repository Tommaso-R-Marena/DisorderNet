"""Portable path configuration for the CPU pipeline.

Historically the root scripts (``fetch_disprot.py``, ``extract_esm_embeddings.py``,
``run_v6_mem.py`` etc.) hardcoded absolute paths under
``/home/user/workspace/disorder_model/``. That only worked inside one sandbox and
broke on a normal ``git clone``. This module centralizes those locations and makes
them **repo-local by default** while allowing overrides via environment variables
(mirroring the ``DISORDERNET_*`` convention already used in ``rockfish/utils.py``).

Environment variables (all optional):

* ``DISORDERNET_HOME`` -- master base dir for data + results. Set this to
  reproduce the historical layout, e.g.
  ``DISORDERNET_HOME=/home/user/workspace/disorder_model``. Defaults to the repo root.
* ``DISORDERNET_DATA_DIR`` -- processed DisProt JSON + ESM embeddings live here.
  Defaults to ``<home>/data``.
* ``DISORDERNET_EMB_DIR`` -- ESM embedding ``.npy`` cache. Defaults to
  ``<data>/embeddings``.
* ``DISORDERNET_RESULTS_ROOT`` -- parent dir for the per-version results folders
  (``results``, ``results_v5``, ``results_v6``). Defaults to ``<home>``.

Usage::

    from disordernet_paths import DISPROT_JSON, EMB_DIR, DATA_DIR, results_dir
    RESULTS_DIR = results_dir("results_v6", create=True)
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _env_path(name):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


HOME_DIR = _env_path("DISORDERNET_HOME") or REPO_ROOT

DATA_DIR = _env_path("DISORDERNET_DATA_DIR") or (HOME_DIR / "data")
EMB_DIR = _env_path("DISORDERNET_EMB_DIR") or (DATA_DIR / "embeddings")
DISPROT_JSON = DATA_DIR / "disprot_processed.json"

RESULTS_ROOT = _env_path("DISORDERNET_RESULTS_ROOT") or HOME_DIR


def results_dir(name="results_v6", create=False):
    """Return the results directory ``<results_root>/<name>``.

    When ``create`` is True the directory (and parents) is created if missing.
    """
    path = RESULTS_ROOT / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
