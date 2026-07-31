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

### Feature computation
All sliding-window statistics go through `window_stats.py` (`moving_average`,
`moving_variance`, `SymbolWindows`, `build_index_table`/`encode_sequence`). The
three featurisers — `features.py` (204-dim), `features_fast.py` (162-dim) and
`run_v6_mem.phys` (118-dim, re-exported as `wavg`/`wvar` for `predictor.py`,
`run_v7.py` and `experiments/`) — all build on it, so do not reintroduce local
cumsum helpers. Two invariants matter: prefix sums accumulate in **float64**
(a float32 cumsum over the bulkiness/MW scales loses the significant digits
that `E[x^2]-E[x]^2` depends on) and moving variances are **clipped at 0**.

### Measuring CPU accuracy (before claiming a change helps)
`cpu_accuracy_bench.py` runs the same 5-fold protein-grouped CV as `run_v6_mem.py`
so two variants can be compared under identical splits and seeds:
```
python cpu_accuracy_bench.py --data real                 # DisProt, the number that counts
python cpu_accuracy_bench.py --data synthetic            # no-download fallback
python cpu_accuracy_bench.py --variant smoothed --out ab.json
```
Rules of thumb:
- `--data real` needs `fetch_disprot.py` + `extract_esm_embeddings.py` to have run.
  Both need network (`disprot.org`, `dl.fbaipublicfiles.com`); some sandboxes
  block them, in which case only `--data synthetic` is available.
- **Synthetic AUC is not comparable to the DisProt number.** It is calibrated to
  the same 0.80–0.87 band and is only meaningful as an A/B between variants on
  the same corpus. It has a genuine Bayes ceiling (part of the label is driven by
  a field the model never sees), so it does not saturate.
- Re-run with only `model_seed` changed to get this benchmark's **noise floor**;
  a claimed improvement has to beat it before it means anything.
- `run_v6_mem.evaluate` picks its decision threshold with Youden's J **on the
  data being scored**, which inflates f1/mcc/precision/recall (AUC/AP are
  unaffected). Pass an explicit `threshold=` for an unbiased number — the bench
  derives each fold's threshold from the other folds and reports both.

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
- Operator ops guide (finish signals, timelines, stuck QOS recovery): root
  `README.md` § **Path C**. Also `rockfish/README.md` § From scratch,
  `rockfish/V8_MULTISCALE.md` for the cheaper v8 path,
  `rockfish/PUBLISH_FULL.md` for **650M→3B auto-resume campaign**.
- **Accounts:** GPU/`a100` jobs need `-A <gpu_account> --qos=qos_gpu`
  (usually `sfried3_gpu`); CPU/`shared` stays `-A sfried3` with no qos.
  Prefer `bash rockfish/slurm/submit_v8.sh` (never submits an empty `--qos`).
  Discover GPU account via `sacctmgr … | awk … /qos_gpu/`.
- **Walltime:** Rockfish a100 max is **72 h** (not 48 h); shared ≈ 36 h.
  Fold resume via `cv_progress.json`; campaign watchdog:
  `bash rockfish/slurm/submit_publish_full.sh`.
- Use the two publish submitters (not the retired all-in-one):
  `bash rockfish/slurm/submit_publish_650m.sh --account sfried3_gpu --qos qos_gpu`
  (and/or `submit_publish_3b.sh` the same way).
  Prefer full campaign: `submit_publish_full.sh` (mail default
  `marenatommaso@gmail.com`).
- Prefer `python rockfish/publish_submit.py submit-650m|submit-3b|package --kind …`.
- Packaging is **strict by default** (`PACKAGE_STRICT=1` / `--strict`): missing
  go/no-go artifacts fail the job. Use `--no-strict` / `--no-strict-package` only
  when debugging.
- Canonical docs: `rockfish/README.md` § Publish path; checklist:
  `docs/METHODS_CHECKLIST.md`. `submit_publish_all.sh` exits with an error redirect.
