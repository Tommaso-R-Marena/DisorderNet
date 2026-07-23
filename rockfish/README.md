# DisorderNet on JHU Rockfish (Slurm)

Run the full SOTA pipeline on Rockfish instead of Colab: longer wall times (72 h), more RAM, scratch storage, and no Drive quota issues.

## Related documentation

| Document | Role |
|----------|------|
| **This file** | Canonical Rockfish setup, publish path, artifacts, go/no-go, env vars |
| [`docs/ROCKFISH_PUBLISH_RUNBOOK.md`](../docs/ROCKFISH_PUBLISH_RUNBOOK.md) | Short publish-path pointer |
| [`docs/METHODS_CHECKLIST.md`](../docs/METHODS_CHECKLIST.md) | Preprint freeze checklist |
| [`docs/STRUCTURE_DISTRUST_ATLAS.md`](../docs/STRUCTURE_DISTRUST_ATLAS.md) | Structure-distrust claim + eval artifacts |
| [`docs/IDR_BIOLOGY_LAYER.md`](../docs/IDR_BIOLOGY_LAYER.md) | IDR biology layer thesis + stages |
| [`docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md`](../docs/PAPER_OUTLINE_STRUCTURE_DISTRUST.md) | Paper outline / evidence stack |
| [`docs/HOMOLOGY_HOLDOUT.md`](../docs/HOMOLOGY_HOLDOUT.md) | Homology CV wording |
| [Root `README.md`](../README.md) | Project overview, Colab paths, full documentation index |
| [`AGENTS.md`](../AGENTS.md) | Contributor / agent notes (pytest, CPU pipeline, Rockfish conventions) |

## From scratch on Rockfish (start here)

End-to-end on a **login node**, with nothing set up yet.  
**Finish signals, timelines, artifacts, and stuck-job recovery** are documented in the root README  
**[Path C — Rockfish ops guide](../README.md#path-c--rockfish-ops-guide-what-you-know-what-to-run-when-youre-done)**.

```bash
# 1) Clone / update
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet 2>/dev/null || true
cd ~/DisorderNet && git checkout master && git pull

# 2) One-time Python env + logs
bash rockfish/setup_env.sh
source ~/venvs/disordernet/bin/activate
mkdir -p logs

# 3) CPU account + GPU account (a100 requires qos_gpu on the GPU account)
export DISORDERNET_ACCOUNT=sfried3
export DISORDERNET_GPU_ACCOUNT=$(sacctmgr -nP show assoc user=$USER format=account,qos \
  | awk -F'|' '/qos_gpu/{print $1; exit}')
export DISORDERNET_GPU_QOS=qos_gpu
export DISORDERNET_BOLTZ_ROOT=${DISORDERNET_BOLTZ_ROOT:-$HOME/scr4_sfried3/boltz}
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache
echo "GPU account: $DISORDERNET_GPU_ACCOUNT"   # expect sfried3_gpu (or similar)

# 4) Prefetch on LOGIN (internet) — never load ESM models on login
#    python rockfish/prefetch_esm.py

# 5) Optional Boltz warm-up (GPU account + qos_gpu)
sbatch -A "$DISORDERNET_GPU_ACCOUNT" --qos="$DISORDERNET_GPU_QOS" \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch

# 6) Submit publish bundles (GPU chain → CPU package strips _gpu from account)
bash rockfish/slurm/submit_publish_650m.sh \
  --account "$DISORDERNET_GPU_ACCOUNT" --qos "$DISORDERNET_GPU_QOS"
# and/or:
bash rockfish/slurm/submit_publish_3b.sh \
  --account "$DISORDERNET_GPU_ACCOUNT" --qos "$DISORDERNET_GPU_QOS"   # add --partition ica100 if OOM

# 7) Monitor — done when queue empty AND sacct shows COMPLETED|0:0
squeue -u $USER
sacct -j <JOBID> --format=JobID,State,ExitCode,Elapsed -P

# 8) When done: package README + checklist + go/no-go
less ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
# Tick docs/METHODS_CHECKLIST.md from the package artifacts
```

For the cheaper **v8 CPU ensemble** path first (~hours not days), use
[`V8_MULTISCALE.md`](V8_MULTISCALE.md).

Project-wide paths (CPU / Colab / Rockfish): **[root README — From scratch](../README.md#from-scratch-start-here)**.  
Exact publish flags and layouts: **[Publish path (exact usage)](#publish-path-exact-usage)** below.

## Slurm script convention

All `rockfish/slurm/*.sbatch` files follow the lab Rockfish style:

```bash
#!/bin/bash -ue
#SBATCH --job-name="dn-…"
#SBATCH --partition=a100          # or shared for CPU jobs
# (no --qos: Slurm uses the partition/account default. Add --qos=<name> only if
#  your cluster rejects the default with "Invalid qos specification".)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=…
#SBATCH --gres=gpu:1              # GPU jobs only
#SBATCH --mem=…
#SBATCH --time=DD-HH:MM:SS        # e.g. 03-00:00:00 = 72 h
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/…_%j.out
#SBATCH --error=logs/…_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs}"
WK_DIR="${DISORDERNET_WORKDIR:-}"
```

Shared runtime logic lives in `rockfish/slurm/_common.sh` (`ml` modules + venv activate + run + mirror). Uncomment the `mail-*` lines and set your JH address when you want email on completion/failure.

## Prerequisites

1. **GPU allocation** — Rockfish `a100` requires **`--qos=qos_gpu`**, and that QOS is attached to a **GPU account** (usually `<group>_gpu`, e.g. `sfried3_gpu`), not the CPU account `sfried3`. Discover it with:
   ```bash
   sacctmgr -nP show assoc user=$USER format=account,qos | awk -F'|' '/qos_gpu/{print $1; exit}'
   ```
   Submit GPU jobs with `-A $DISORDERNET_GPU_ACCOUNT --qos=qos_gpu`. CPU/`shared` jobs stay on `-A sfried3` with no `--qos`. Request access via [ARCH support](https://docs.arch.jhu.edu/) if no GPU account appears. See root README **Path C**.
2. **Clone the repo** on Rockfish login node:
   ```bash
   git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet
   cd ~/DisorderNet
   git checkout master
   ```
3. **One-time env setup**:
   ```bash
   bash rockfish/setup_env.sh
   source ~/venvs/disordernet/bin/activate
   mkdir -p logs
   export DISORDERNET_ACCOUNT=sfried3   # Rockfish account (default: sfried3)
   export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
   export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache
   ```

## Publish path (exact usage)

Two separate submitters (recommended). Each fits Rockfish’s **72 h** GPU wall
time by chaining Slurm jobs, then writing an organized `publish_package/`.

Frozen checklist: [`docs/METHODS_CHECKLIST.md`](../docs/METHODS_CHECKLIST.md).

### Shared one-time setup (do this once)

```bash
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet
cd ~/DisorderNet
git checkout master

bash rockfish/setup_env.sh
source ~/venvs/disordernet/bin/activate
mkdir -p logs

export DISORDERNET_ACCOUNT=sfried3          # Rockfish account (default: sfried3)
export DISORDERNET_BOLTZ_ROOT=$HOME/boltz       # optional but recommended
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache
```

Optional Boltz warm-up (if structures are not ready yet):

```bash
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch
```

---

### Script 1 — ESM-2 650M publish bundle

Runs: **ultra 650M → ultra_clean 650M → package**.

**Bash (exact):**
```bash
cd ~/DisorderNet
source ~/venvs/disordernet/bin/activate
export DISORDERNET_ACCOUNT=sfried3

bash rockfish/slurm/submit_publish_650m.sh
```

**CLI (exact equivalents — pick one):**
```bash
python rockfish/publish_submit.py submit-650m --account "$DISORDERNET_ACCOUNT"

# or via the HPC runner:
python rockfish/run_disordernet.py publish-650m --account "$DISORDERNET_ACCOUNT"
```

**Useful flags (bash passes them through to the CLI):**
```bash
bash rockfish/slurm/submit_publish_650m.sh --dry-run
bash rockfish/slurm/submit_publish_650m.sh --no-clean
bash rockfish/slurm/submit_publish_650m.sh --root-workdir $HOME/disordernet_runs/my_650m
bash rockfish/slurm/submit_publish_650m.sh --partition a100
```

**What gets submitted**

| Job | Profile | Approx. GPU time |
|-----|---------|------------------|
| `ultra_650M` | `ultra` / 650M | ~24–48 h |
| `ultra_clean_650M` | `ultra_clean` / 650M (omit with `--no-clean`) | ~24–48 h after main |
| `publish_package/` | CPU | minutes |

**Output layout**
```text
~/disordernet_runs/publish_650m_<stamp>/
  ultra_650M/checkpoints/…
  ultra_clean_650M/checkpoints_ultra_clean/…
  submit_summary.json
  publish_package/
    PACKAGE_README.md
    MANIFEST.json
    comparison.json
    ultra_650M/
    ultra_clean_650M/
```

---

### Script 2 — ESM-2 3B publish bundle

Runs: **ultra3b → ultra_clean 3B → package**.

**Bash (exact):**
```bash
cd ~/DisorderNet
source ~/venvs/disordernet/bin/activate
export DISORDERNET_ACCOUNT=sfried3

bash rockfish/slurm/submit_publish_3b.sh
```

**If OOM on 40GB A100:**
```bash
bash rockfish/slurm/submit_publish_3b.sh --partition ica100
```

**CLI (exact equivalents — pick one):**
```bash
python rockfish/publish_submit.py submit-3b --account "$DISORDERNET_ACCOUNT"

python rockfish/run_disordernet.py publish-3b --account "$DISORDERNET_ACCOUNT" --partition ica100
```

**What gets submitted**

| Job | Profile | Approx. GPU time |
|-----|---------|------------------|
| `ultra3b` | `ultra3b` / 3B | ~30–40 h |
| `ultra_clean_3B` | `ultra3b` + clean flags (omit with `--no-clean`) | ~30–40 h after main |
| `publish_package/` | CPU | minutes |

**Output layout**
```text
~/disordernet_runs/publish_3b_<stamp>/
  ultra3b/checkpoints/…
  ultra_clean_3B/checkpoints_ultra_clean_3b/…
  submit_summary.json
  publish_package/
    PACKAGE_README.md
    MANIFEST.json
    comparison.json
    ultra3b/
    ultra_clean_3B/
```

---

### After jobs finish

```bash
# Monitor
squeue -u $USER

# Open the package (path also printed at submit time / in submit_summary.json)
less ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
# or for 3B:
less ~/disordernet_runs/publish_3b_*/publish_package/PACKAGE_README.md
```

**Re-package an existing bundle** (strict by default — fails if go/no-go artifacts are missing):

```bash
python rockfish/publish_submit.py package \
  --root-workdir ~/disordernet_runs/publish_650m_<stamp> \
  --kind 650m \
  --strict

# Allow incomplete packages only when debugging:
python rockfish/publish_submit.py package \
  --root-workdir ~/disordernet_runs/publish_650m_<stamp> \
  --kind 650m \
  --no-strict
```

Prefer `publish_submit.py package --kind …` over the legacy combined layout in
`package_publish_results.py` (requires `--legacy-combined` without `--kind`).
Package jobs set `PACKAGE_STRICT=1` by default; `submit_summary.json` records
`git_revision` for provenance.

```bash
# Via the HPC runner (strict by default):
python rockfish/run_disordernet.py package-publish \
  --publish-root ~/disordernet_runs/publish_650m_<stamp> \
  --bundle-kind 650m
```

Re-package without re-training (if GPU jobs already finished):
```bash
python rockfish/publish_submit.py package \
  --kind 650m \
  --root-workdir ~/disordernet_runs/publish_650m_YYYYMMDDTHHMMSSZ

python rockfish/run_disordernet.py package-publish \
  --publish-root ~/disordernet_runs/publish_3b_YYYYMMDDTHHMMSSZ \
  --bundle-kind 3b
```

Then fill [`docs/METHODS_CHECKLIST.md`](../docs/METHODS_CHECKLIST.md) and apply go/no-go below.

```text
setup → submit_publish_650m.sh and/or submit_publish_3b.sh
  → open publish_package/ → METHODS_CHECKLIST → go/no-go
```

### Manual one-offs (advanced; not required)

Use these only if you need a single stage without packaging.

### 1. Optional Boltz warm-up

```bash
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch
```

### 2. Single `pipeline_ultra` job (no packaging)

```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_main
export RUN_CAID3=1
export BOLTZ_MODE=ingest
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,RUN_CAID3,BOLTZ_MODE \
  rockfish/slurm/pipeline_ultra.sbatch
```

### 3. Single clean companion

```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_clean
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
  rockfish/slurm/pipeline_ultra_clean.sbatch
```

```bash
python rockfish/run_disordernet.py pipeline \
  --profile ultra_clean \
  --no-hallucination-weighting \
  --no-plddt-features \
  --run-caid3-eval \
  --checkpoint-dir checkpoints_ultra_clean \
  --workdir "$DISORDERNET_WORKDIR"
```

### 4. Verify package / mirrored artifacts

Confirm these exist under each run’s checkpoint dir (or inside `publish_package/`):

| Artifact | Purpose |
|----------|---------|
| `structure_distrust_benchmark.json` | Labeled rescue + matched baselines + CAID floor + contamination |
| `structure_distrust_atlas_report.json` | Proteome summary |
| `structure_distrust_atlas.jsonl` / `.tsv` | Per-protein rows |
| `distrust_figures/**` | Paper figures (if matplotlib available) |
| `caid3_eval_report.json` | Credibility floor source |
| `af_rescue_manifest.json` | Pipeline triage |
| `statistical_validation_report.json` | Paired / fold stats |
| `idr_biology_layer_*` | Supporting layer exports |
| `function_prediction_report.json` | If function head enabled |
| `run_manifest.json` / `cv_summary.json` | Reproducibility |

```bash
python -c "
import json
b=json.load(open('structure_distrust_benchmark.json'))
print(b.get('caid3_credibility_floor'))
print(b.get('training_contamination',{}).get('risk_tier'))
"
```

Expect `caid3_credibility_floor.available == true` on the main run after CAID3.
Fill [`docs/METHODS_CHECKLIST.md`](../docs/METHODS_CHECKLIST.md) from these files
and build a **main vs clean** comparison table (ΔAUC vs inv-pLDDT, labeled
rescue, mask utility, contamination `risk_tier`).

### 5. Publish go / no-go

All must hold on the **main** run, with the **clean** run confirming the claim
is not entirely an artifact of structure training:

1. **Matched distrust** — DN beats inv-pLDDT on matched residues
   (`matched_baselines.delta_auc_dn_minus_plddt` > 0) with supportive paired
   stats in `statistical_validation_report.json`.
2. **Labeled rescue** — meaningful `rescue_rate` under
   `definition: labeled_independent` (not proxy).
3. **Mask utility** — DN distrust mask ≥ size-matched pLDDT mask.
4. **CAID floor** — attached (`available: true`) and competitive enough that
   disorder credibility is not undermined.
5. **Contamination honesty** — `training_contamination` present; main vs clean
   table exists; if clean collapses the ΔAUC claim, do **not** frame rescue as
   a breakthrough.

If criteria fail: do **not** publish rescue-as-breakthrough. Fall back to a
disorder + atlas triage / methods paper, or iterate training — not more feature
scaffolding.

## Quick start (training ladder)

```bash
cd ~/DisorderNet
mkdir -p logs
export DISORDERNET_ACCOUNT=sfried3   # Rockfish account (default: sfried3)

# Step 1: go/no-go screen (~2–3 h; SCREEN_MODE=standard → screen_plus mini-ultra)
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT \
  rockfish/slurm/quick_screen.sbatch

# Step 2: full ultra on 650M (~18–24 h)
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT \
  rockfish/slurm/train_ultra.sbatch

# Step 3: 3B paradigm screen (~8–12 h)
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,SCREEN_MODE=paradigm,BACKBONE=3B,PROFILE=ultra3b \
  rockfish/slurm/quick_screen.sbatch

# Step 4: full ultra3b production (~30–40 h)
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT \
  rockfish/slurm/train_ultra3b.sbatch

# Publish bundles (see Publish path above)
export DISORDERNET_ACCOUNT=sfried3
bash rockfish/slurm/submit_publish_650m.sh
bash rockfish/slurm/submit_publish_3b.sh

# Multi-seed for +0.005–0.015 AUC (3 parallel jobs)
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_ms
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
  rockfish/slurm/multi_seed.sbatch
```

Monitor: `squeue -u $USER` · logs in `logs/dn-*.out` · results mirrored to `~/disordernet_runs/`.

## Rockfish vs Colab

| | Colab Pro | Rockfish A100 |
|--|-----------|---------------|
| Max runtime | ~12–24 h (disconnect risk) | **72 h** per job |
| 3B ESM-2 | Tight on T4; A100 only | **a100** or **ica100** (80 GB) |
| Storage | Drive 96% issues | Scratch + `~/disordernet_runs` |
| DataLoader workers | 0 (Colab hang workaround) | **4–8** (faster) |
| Resume | `cv_progress.json` | Same — re-submit with `STAGE=cv` |

## Partitions

| Partition | GPU | Use for |
|-----------|-----|---------|
| `a100` | 4× A100 40 GB | Default — ultra, ultra3b |
| `ica100` | 4× A100 80 GB | ultra3b if OOM on 40 GB |
| `l40s` | 8× L40s 48 GB | Alternative if a100 queue is long |

Override partition:
```bash
sbatch --partition=ica100 --export=ALL,... rockfish/slurm/train_ultra3b.sbatch
```

## Pipeline stages (CLI)

`rockfish/run_disordernet.py` mirrors the Colab Pro notebook:

| Stage | Colab cells | Description |
|-------|-------------|-------------|
| `screen` | Quick Screen | Stratified subset + mode-aware verdict (`standard`→`screen_plus`) |
| `cv` | 6–7 | 5-fold GPU training |
| `stack` | 7b–7c | GPU+v6 + meta ensemble |
| `postprocess` | 7d | Fold soup + calibration |
| `full` | all training | cv → stack → postprocess |
| `boltz` | structure | Boltz-2 pLDDT (pinned, auto-download) |
| `af3` | structure | Optional AlphaFold 3 ingest/run |
| `idr-layer` | product | Post-structure IDR biology layer export |
| `structure-distrust-atlas` | paper | Labeled/proxy distrust atlas from preds + pLDDT cache |
| `eval` | 8–11 | CAID, AF/Boltz rescue, structure distrust, Phase 3 |
| `pipeline` | all + eval | full → eval [→ CAID3 with `RUN_CAID3=1`; CAID floor patched into distrust benchmark] |
| `publish-650m` | submit | Submit 650M ultra + clean → package (see Publish path) |
| `publish-3b` | submit | Submit ultra3b + clean → package (see Publish path) |
| `package-publish` | package | Assemble `publish_package/` from an existing bundle root |
| `predict` | deploy | FASTA batch inference + `.caid` export |
| `multi-seed-blend` | 7e | Average OOF from multiple seed dirs |

Standalone atlas from existing preds + pLDDT cache:

```bash
python rockfish/run_disordernet.py structure-distrust-atlas \
  --atlas-preds-dir path/to/preds \
  --atlas-plddt-dir path/to/plddt_cache
```

## Training profiles

| Profile | Use | Structure-training flags |
|---------|-----|--------------------------|
| `ultra` | Default production (650M) | `use_plddt_features=True`, hallucination wt on |
| `ultra3b` | ESM-2 3B on A100 40GB+ | same as ultra |
| `ultra_clean` | Publish ablation companion | **both structure flags off** |
| `ultra_fun` | ultra + disorder→function head | same as ultra + function head |
| `sota` / `max` / `balanced` | Intermediate capacity | see `TrainConfig.from_profile` |
| `screen` | `flash` mode only (toy CNN) | hallucination wt off |
| `screen_plus` | `standard`/`paradigm` mini-ultra screen | hallucination wt off |

Clean override without changing profile:

```bash
python rockfish/run_disordernet.py pipeline \
  --profile ultra \
  --no-hallucination-weighting \
  --no-plddt-features
```

## Boltz-2 structure backend (default)

Boltz-2 is the **default** structure pLDDT source (`--structure-backend boltz`).
Pinned package: `boltz[cuda]==2.2.1` — first run **auto-downloads** weights into `$BOLTZ_CACHE`.

```bash
export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache

# Run / auto-download Boltz-2 for pending proteins
sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch

# Or Slurm array:
export BOLTZ_SHARD_COUNT=8
sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=run,BOLTZ_SHARD_COUNT=8 \
  rockfish/slurm/boltz_array.sbatch

# CV prefers Boltz pLDDT when available (training jobs default BOLTZ_MODE=ingest)
python rockfish/run_disordernet.py cv --profile ultra --boltz-mode ingest
```

AF3 remains available as a secondary licensed backend (`--af3-mode ingest|run`, `--structure-backend af3`).

## AlphaFold 3 (optional licensed)

AF3 **code** is open (Apache 2.0); **weights** (`af3.bin`) require a DeepMind license and must **never** go on GitHub.

```bash
export DISORDERNET_AF3_ROOT=$HOME/af3
mkdir -p $DISORDERNET_AF3_ROOT/{outputs,inputs}
# Place licensed af3.bin → $DISORDERNET_AF3_ROOT/af3.bin
git clone --depth 1 https://github.com/google-deepmind/alphafold3.git ~/software/alphafold3
export DISORDERNET_AF3_REPO=$HOME/software/alphafold3

# Run missing proteins (MSA-free by default — no 630 GB DBs)
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=run \
  rockfish/slurm/af3_batch.sbatch

# Then train/eval with AF3 preferred over AF2:
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=ingest,RUN_CAID3=1 \
  rockfish/slurm/pipeline_ultra.sbatch
```

Stages: `af3` · eval with `--af3-mode ingest|run` · CV uses AF3 pLDDT for structure channel when available.

## Efficiency (zero accuracy loss)

| Optimization | Effect |
|--------------|--------|
| Disk token cache | Skip re-tokenization across resumes / multi-seed |
| `persistent_workers` + prefetch | Faster epochs on multi-core nodes |
| TF32 / high matmul precision | Faster Ampere GPUs (bf16 path unchanged) |
| TORCH_HOME on scratch | Faster ESM weight load from local SSD |
| Grad-checkpoint off at infer | ~10–30% faster fold soup / FASTA predict |
| Parallel AF2 pLDDT fetch | Faster structure prefetch |
| **Concurrent DisProt download** | Parallel REST pages at cold start |
| **ESM ‖ AF2 pLDDT overlap** | Prefetch structure while loading ESM |
| **AF3 Slurm array** | Parallel MSA-free AF3 across GPUs |
| **Parallel result mirror** | Threaded copy of reports to `$DISORDERNET_RESULTS` |
| AF3 MSA-free mode | Hours→minutes per protein without 630 GB DBs |

These do **not** change CV folds, losses, or logit values used for AUC.

### AF3 parallel array

```bash
export DISORDERNET_AF3_ROOT=$HOME/af3
export AF3_SHARD_COUNT=8
sbatch --account=$DISORDERNET_ACCOUNT --array=0-7 \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=run,AF3_SHARD_COUNT=8 \
  rockfish/slurm/af3_array.sbatch
```

## vs ESMDisPred / sequence-only SOTA


| Capability | ESMDisPred | DisorderNet (Rockfish) |
|------------|------------|------------------------|
| Sequence-only disorder | ✓ | ✓ (ESM-2 650M–3B + LoRA) |
| Homology-safe CV | CAID protocol | ✓ `split_method=homology` in ultra |
| CAID3 benchmark eval | 0.895 ref | ✓ `caid3_eval_report.json` |
| AF / Boltz hallucination rescue | ✗ | ✓ Boltz-2 default (+ optional AF3) |
| Train-time pLDDT channel | ✗ | ✓ `use_plddt_features` in ultra |
| Disorder → function (IDR roles) | ✗ | ✓ `ultra_fun` / `--function-head` |
| Proteome FASTA deploy | limited | ✓ `predict` stage |

Interactive debug (salloc GPU node):
```bash
salloc --partition=a100 --account=$DISORDERNET_ACCOUNT \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00
source ~/venvs/disordernet/bin/activate
cd ~/DisorderNet
python rockfish/run_disordernet.py screen --screen-mode flash --backbone 650M

# Disorder → function multi-label training
python rockfish/run_disordernet.py cv --profile ultra_fun
# or: --profile ultra --function-head
```

## Fault tolerance

CV checkpoints write to `checkpoints/cv_progress.json` after each fold. If a job is preempted:

```bash
# Re-submit — auto-resumes from last completed fold
sbatch --export=ALL,DISORDERNET_ACCOUNT,STAGE=cv rockfish/slurm/train_ultra.sbatch
```

Set a fixed workdir so resume finds checkpoints:
```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/production_ultra3b
sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR rockfish/slurm/train_ultra3b.sbatch
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DISORDERNET_ACCOUNT` | `sfried3` | Slurm account |
| `DISORDERNET_REPO` | `~/DisorderNet` | Repo path |
| `DISORDERNET_VENV` | `~/venvs/disordernet` | Python venv |
| `DISORDERNET_WORKDIR` | Slurm scratch | Run directory |
| `DISORDERNET_RESULTS` | `~/disordernet_runs` | Durable result copies |
| `CHECKPOINT_SUBDIR` | `checkpoints` | Relative checkpoint dir under workdir |
| `PROFILE` | `ultra` / `ultra3b` / `ultra_clean` | Training profile |
| `BACKBONE` | `650M` / `3B` | ESM-2 size |
| `STAGE` | `full` | Pipeline stage |
| `SEED` | `42` | Random seed |
| `RUN_CAID3` | `0` | Set `1` to append CAID3 + finalize distrust floor |
| `RUN_NO_HALLUC_WEIGHT` | `0` | Set `1` → `--no-hallucination-weighting` |
| `RUN_NO_PLDDT_FEATURES` | `0` | Set `1` → `--no-plddt-features` |
| `PREFETCH_AF` | `0` | Set `1` → `--prefetch-af-plddt` |
| `STRUCTURE_BACKEND` | `boltz` | Prefer boltz / af3 / af2 pLDDT |
| `BOLTZ_MODE` | `ingest` (train) / `auto` (boltz jobs) | Boltz-2 ingest/run/auto |
| `DISORDERNET_BOLTZ_ROOT` | `~/boltz` | Boltz inputs/outputs/cache root |
| `BOLTZ_CACHE` | `$DISORDERNET_BOLTZ_ROOT/cache` | Auto-downloaded Boltz weights |
| `DISORDERNET_PUBLISH_ROOT` | (set by submitter) | Publish bundle parent |
| `DISORDERNET_PACKAGE_DIR` | `$PUBLISH_ROOT/publish_package` | Organized package output |
| `BUNDLE_KIND` | `650m` / `3b` | Kind-aware packaging |
| `PACKAGE_STRICT` | `1` | Fail package job if go/no-go artifacts missing |
| `INCLUDE_CLEAN` | `1` | Include contamination-clean companion in package |
| `MIRROR_REQUIRE_MIN_FILES` | `1` | Fail-loud empty GPU result mirrors |

## Copy DisProt cache from Colab (optional)

If you already downloaded DisProt on Colab, copy `disprot_raw.json` to Rockfish to skip the REST download:

```bash
# From laptop
scp disprot_raw.json rockfish:/path/to/workdir/disprot_raw.json
```

Or place it in `$DISORDERNET_WORKDIR/disprot_raw.json`.

## Expected performance

Same targets as Colab — Rockfish mainly gives **reliability and scale**, not a different algorithm:

| Run | Target pooled AUC |
|-----|-------------------|
| ultra + stack + 7d | 0.88–0.92 |
| ultra3b + full stack | 0.90–0.93 |

After each job, check `~/disordernet_runs/<run_tag>/sota_postprocess_report.json` for final AUC.
