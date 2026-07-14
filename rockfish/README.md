# DisorderNet on JHU Rockfish (Slurm)

Run the full SOTA pipeline on Rockfish instead of Colab: longer wall times (72 h), more RAM, scratch storage, and no Drive quota issues.

## Prerequisites

1. **GPU allocation** — your PI must have a Rockfish `_gpu` account (e.g. `jsmith123_gpu`) and `qos_gpu`. Request via [ARCH support](https://docs.arch.jhu.edu/) if needed.
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
   export DISORDERNET_ACCOUNT=your_pi_gpu   # replace with your _gpu account
   export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
   export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache
   ```

## Publish path (main + clean companion)

Frozen operator path for the structure-distrust paper. After these jobs finish,
remaining work is **judge numbers → go/no-go** — not more feature code.
Checklist boxes: [`docs/METHODS_CHECKLIST.md`](../docs/METHODS_CHECKLIST.md).

```text
checkout master → setup_env → sbatch pipeline_ultra → sbatch pipeline_ultra_clean
  → verify mirrored artifacts → METHODS_CHECKLIST → go/no-go on numbers
```

### 1. Optional Boltz warm-up

```bash
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch
```

### 2. Main run (production ultra + CAID3)

Boltz ingest + full pipeline + CAID3. Eval writes the labeled distrust benchmark;
CAID3 then **patches** `structure_distrust_benchmark.json` in place via
`finalize_distrust_benchmark_with_caid3`.

```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_main
export RUN_CAID3=1
export BOLTZ_MODE=ingest   # or auto if structures not yet available
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,RUN_CAID3,BOLTZ_MODE \
  rockfish/slurm/pipeline_ultra.sbatch
```

### 3. Clean companion (required for publish honesty)

Same capacity as ultra, but `use_hallucination_weighting=False` and
`use_plddt_features=False`. **Must use a separate workdir** so it cannot
clobber the main run. Script sets `PROFILE=ultra_clean`,
`CHECKPOINT_SUBDIR=checkpoints_ultra_clean`, and
`RUN_NO_HALLUC_WEIGHT=1` / `RUN_NO_PLDDT_FEATURES=1`.

```bash
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_clean
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
  rockfish/slurm/pipeline_ultra_clean.sbatch
```

Equivalent CLI (interactive GPU node):

```bash
python rockfish/run_disordernet.py pipeline \
  --profile ultra_clean \
  --no-hallucination-weighting \
  --no-plddt-features \
  --run-caid3-eval \
  --checkpoint-dir checkpoints_ultra_clean \
  --workdir "$DISORDERNET_WORKDIR"
```

### 4. Verify mirrored artifacts

After both jobs finish, confirm these exist under the workdir checkpoints (or
`$DISORDERNET_RESULTS/<run_tag>/`):

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
export DISORDERNET_ACCOUNT=your_pi_gpu   # replace with your _gpu account

# Step 1: go/no-go screen (~2–3 h)
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

# Recommended publish pair (see Publish path above)
export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_main
export RUN_CAID3=1
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,RUN_CAID3=1 \
  rockfish/slurm/pipeline_ultra.sbatch

export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_clean
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
  rockfish/slurm/pipeline_ultra_clean.sbatch

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
| `screen` | Quick Screen | Stratified subset + verdict |
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
| `screen` / `screen_plus` | Quick screen | hallucination wt off |

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
salloc --partition=a100 --qos=qos_gpu --account=$DISORDERNET_ACCOUNT \
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
| `DISORDERNET_ACCOUNT` | (required) | Slurm `_gpu` account |
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
