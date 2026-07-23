# AGENTS.md

## Cursor Cloud specific instructions

DisorderNet is a Python ML research pipeline (no web server, DB, or long-running
services) that predicts intrinsically disordered regions (IDRs) in proteins. Work
happens through Python scripts, a pytest suite, and Colab/Slurm notebooks (the
GPU/Colab/Rockfish paths are not runnable in this CPU-only cloud environment).

### Python environment
- Dependencies are installed into a virtualenv at `.venv` (Python 3.12). The
  startup update script creates it and installs `requirements-dev.txt` plus the
  CPU-pipeline deps in `requirements-cpu.txt` (`lightgbm`, `xgboost`, `fair-esm`, …).
- Activate it before running anything: `source .venv/bin/activate`.

### Tests (fast, self-contained)
- Run: `pytest tests/ -v` (config in `pytest.ini`; CI in `.github/workflows/test.yml`).
- The suite mocks ESM (`tests/conftest.py`) and uses synthetic fixtures — no
  network, GPU, or model downloads needed. `gpu`-marked tests are skipped on CPU.
- The full suite imports `lightgbm`/`xgboost`/`fair-esm`, so install BOTH
  `requirements-dev.txt` and `requirements-cpu.txt` (the update script/CI do this).
- Coverage: `pytest tests/ --cov=. --cov-report=term-missing`.

### Linting / CI
- `ruff` is configured in `ruff.toml` (critical-error rules only: syntax + undefined
  names; notebooks excluded). Run `ruff check .`.
- CI (`.github/workflows/test.yml`) has three jobs: ruff lint, import-smoke, and a
  pytest+coverage matrix on Python 3.11/3.12.

### Running the CPU pipeline (the "application")
The end-to-end CPU model lives in the top-level scripts. Paths are centralized in
`disordernet_paths.py` and default to **repo-local** dirs (`./data`, `./data/embeddings`,
`./results_v6`), so no `mkdir`/symlink workaround is needed. Override the location with
`DISORDERNET_HOME` (or finer-grained `DISORDERNET_DATA_DIR` / `DISORDERNET_RESULTS_ROOT`).
Run in order (see README "Option 2" / "Path A"):
```
python fetch_disprot.py            # downloads DisProt -> data/disprot_processed.json (needs network)
python extract_esm_embeddings.py   # ESM-2 embeddings -> data/embeddings/*.npy (downloads weights; ~9 min CPU for all 3333 proteins)
python run_v6_mem.py               # 5-fold CV train+eval -> results_v6/metrics.json (~6 min)
python generate_figures_v6.py      # ROC/PR + benchmark figures -> results_v6/*.png
```
Notes:
- `extract_esm_embeddings.py` tries the ESM-2 35M model first and only falls back
  to 8M on a `RuntimeError`/`MemoryError`; on CPU it just runs (slower) with 35M.
- `run_v6_mem.py` only uses proteins that already have an embedding `.npy`, so it
  works even if extraction is partial. Expect pooled AUC ≈ 0.83–0.84.
- CPU-pipeline deps live in `requirements-cpu.txt` (not `requirements-dev.txt`,
  which is test-only); the startup update script installs both.
- Generated `data/`/`results*/` files are gitignored.

### Rockfish publish path (HPC)
- From scratch: see root `README.md` § **From scratch** (Path C) and
  `rockfish/README.md` § **From scratch on Rockfish**.
- Use the two submitters (not the retired all-in-one):
  `bash rockfish/slurm/submit_publish_650m.sh` and/or
  `bash rockfish/slurm/submit_publish_3b.sh`.
- Prefer `python rockfish/publish_submit.py submit-650m|submit-3b|package --kind …`.
- Packaging is **strict by default** (`PACKAGE_STRICT=1` / `--strict`): missing
  go/no-go artifacts fail the job. Use `--no-strict` / `--no-strict-package` only
  when debugging.
- Canonical docs: `rockfish/README.md` § Publish path; checklist:
  `docs/METHODS_CHECKLIST.md`. `submit_publish_all.sh` exits with an error redirect.
