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
  a claimed improvement has to beat it before it means anything. Measured over
  5 corpus seeds: noise floor ±0.0011 AUC; the float32→float64 featurizer
  rewrite is −0.00017 (p=0.77, indistinguishable from noise); per-protein
  smoothing is +0.00142 (p=0.0003, 5/5 seeds).
- One seed is not enough. A single-seed featurizer comparison read −0.0016 on
  5/5 folds and disappeared once the noise floor was measured.
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

### Fold alignment — the leakage rule that matters most
Every post-training consumer (fold soup, v6/v6-pro OOF, fusion, stacking, stats,
function head) must partition proteins **exactly the way training did**. Use
`colab.cv_splits.resolve_cv_splits`, which prefers the `val_ids` recorded on each
fold result and otherwise honours the run's own `split_method`.

Calling `get_cv_splits(proteins, n_folds)` re-derives with the default
`"protein"` method. The `ultra` / `ultra3b` / `screen_plus` profiles train with
`split_method="homology"`, so that call silently disagrees with training and
puts a fold model's own training proteins into its "held-out" evaluation set.
`tests/test_leakage_guards.py` covers this; do not bypass it.

Related invariants, all regression-tested:
- The meta-stacker must be fitted **out-of-fold** (grouped by protein), never on
  the residues it scores.
- Isotonic calibration must be fitted **leave-one-fold-out**. Temperature scaling
  is strictly monotone so it cannot move AUC/AP; isotonic is not and does.
- Threshold-dependent metrics (f1/mcc) must take the cut-point from other folds.

### Homology clustering (`colab/homology_splits.py`)
- **`difflib.SequenceMatcher` must be constructed with `autojunk=False`.** With
  the default, difflib junks every amino acid for inputs of 200+ residues, and
  two 95%-identical proteins score ~0.01. This made homology splits and the CAID
  leakage audit silent no-ops for 78.5% of DisProt. There is a preflight guard
  in `rockfish/slurm/_smoke_checks.py`.
- BLAST+ is the preferred backend and is ~1000x faster than the Python path
  (16 s vs "does not finish" on 2663 proteins). `_common.sh` loads `blast-plus`
  and warns if `blastp` is missing. `meta["backend"]` records which ran.
- Never size worker pools from `os.cpu_count()` — that is the whole node, not
  the allocation. Use `homology_splits.available_cpus()`.

### Rockfish publish path (HPC)
- Operator ops guide (finish signals, timelines, stuck QOS recovery): root
  `README.md` § **Path C**. Also `rockfish/README.md` § From scratch,
  `rockfish/V8_MULTISCALE.md` for the cheaper v8 path,
  `rockfish/PUBLISH_FULL.md` for **650M→3B auto-resume campaign**.
- **Accounts:** GPU/`a100` jobs need `-A <gpu_account> --qos=qos_gpu`
  (usually `sfried3_gpu`); CPU/`shared` stays `-A sfried3` with no qos.
  Prefer `bash rockfish/slurm/submit_v8.sh` (never submits an empty `--qos`).
  Discover GPU account via `sacctmgr … | awk … /qos_gpu/`.
- **Sizing:** a100 / ica100 / shared / express enforce `MaxMemPerCPU=4000`.
  Slurm silently raises `AllocCPUS` to `ceil(mem_MB / 4000)` when a request
  exceeds that ratio, so `--mem=180G` grabbed **47 of 48 CPUs** on a 4-GPU node
  to run one GPU — blocking three A100s and quadrupling the billing. Always size
  as `cpus-per-task * 4000M`. Measured peak RSS for this pipeline is ~12 GB.
  Also note `--mem=192G` exceeds a100 node RealMemory (187.5 GiB).
- **`_common.sh` sourcing:** Slurm copies the batch script into a per-job spool
  directory, so `${BASH_SOURCE[0]}` does **not** resolve to the repo. Resolve
  against `PROJECT_DIR` first (every sbatch already does; keep it that way).
- **QOS caps:** `express_queue` is 4 CPUs/job; `shared` is 32; `qos_gpu` allows
  10 GPUs per user. GPU jobs need `-A sfried3_gpu --qos=qos_gpu`; CPU jobs use
  `-A sfried3` with no qos.
- **Never run compute on a login node.** Even the pytest suite goes to `shared`
  (it takes ~2.5 min there).
- **Sync one directory per `rsync`.** `rsync -az tests/ colab/ host:dn_rigor/`
  copies the *contents of both* into the destination root — it does not create
  `tests/` and `colab/` there. That scattered 127 modules and test files across
  the repo root, where `pytest` then collected each test twice under two module
  names, and a stale duplicate of an edited module sat one `sys.path` entry away
  from shadowing the real one. Write the destination explicitly, one source at a
  time: `rsync -az colab/ host:dn_rigor/colab/`. Check for the damage with
  `ls *.py | wc -l` (should be 21 in `~/dn_rigor`).
- **Always export `TORCH_HOME` to scratch. One omission killed four jobs.**
  `fair-esm` downloads into `$TORCH_HOME` (default `~/.cache/torch`), and
  ESM-2 3B is **5.7 GB**. A `lite_3b` submission that forgot the variable took
  home from 45 GB to 51 GB, past the 50 GB quota, and killed every other job
  running at the time — `lite_pdb_missing` at fold 5, `lite_pdb_long` at fold 2,
  and an `ultra` recovery at 1h16 — with empty `.err` files, because the quota
  also blocks writing the traceback. The 650M model is 2.6 GB and 3B is 5.7 GB,
  so two backbones alone exceed a fifth of the quota:
  `export TORCH_HOME=/scratch4/<PI>/<user>_disordernet/torch_home`.
- **"No GPU detected" has two distinct causes here; check the banner.** Both
  present identically — the job dies in `setup_environment` about seven seconds
  in while `sacct` reports `gres/gpu:a100=1` allocated. Compare the `ENV_DIR=`
  and node lines of a failing run against a working one:
  - wrong venv (see next bullet) — `ENV_DIR` differs, any node;
  - a bad node — `ENV_DIR` matches a working job and the node repeats. `icgpu04`
    failed two `lite_3b` submissions this way while `icgpu03` ran the sibling
    arm from the same `sbatch` call. Resubmit with `--exclude=<node>`.
  Do not settle on one explanation before checking the other: both were live at
  the same time, and each looked like the other.
- **Do not set `DISORDERNET_VENV=~/venvs/disordernet_rigor` for GPU jobs.** That
  venv's PyTorch (2.5.1+cu121) cannot see the GPU on the ica100 nodes, and the
  job dies in `setup_environment` with "No GPU detected" about seven seconds in
  — while `sacct` cheerfully reports `gres/gpu:a100=1` allocated, which makes it
  look like node flakiness. It is not: `~/venvs/disordernet` (2.7.1+cu118)
  works on the same nodes, and the ablation submitter succeeds precisely because
  it never sets the variable. Six jobs were lost to this before the `ENV_DIR=`
  line in the two banners was compared side by side. `disordernet_rigor` is fine
  for pytest and CPU analysis.
- **GPU jobs submitted with `--wrap` need `--gres=gpu:1` spelled out.** The
  `#SBATCH` lines in `rockfish/slurm/*.sbatch` do not apply to a wrapped command,
  and `--partition=ica100 --qos=qos_gpu` alone allocates no GPU: the job starts,
  loads config, and dies in `setup_environment` with "No GPU detected" ten
  seconds in. Prefer the real sbatch scripts over `--wrap`.
- **Walltime:** Rockfish a100 max is **72 h** (not 48 h); shared ≈ 36 h; l40s 24 h.
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
