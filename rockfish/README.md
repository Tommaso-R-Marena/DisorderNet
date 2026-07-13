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
   ```

## Quick start

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

# Recommended: complete pipeline + eval + CAID3 (~24–48 h ultra)
export RUN_CAID3=1
sbatch --account=$DISORDERNET_ACCOUNT \
  --export=ALL,DISORDERNET_ACCOUNT,RUN_CAID3=1 \
  rockfish/slurm/pipeline_ultra.sbatch

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
| `eval` | 8–11 | CAID, AF rescue, structure calibration, Phase 3 |
| `pipeline` | all + eval | full → eval [→ CAID3 with `RUN_CAID3=1`] |
| `predict` | deploy | FASTA batch inference + `.caid` export |
| `multi-seed-blend` | 7e | Average OOF from multiple seed dirs |

## AlphaFold 3 on Rockfish

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
| AF3 MSA-free mode | Hours→minutes per protein without 630 GB DBs |

These do **not** change CV folds, losses, or logit values used for AUC.

## vs ESMDisPred / sequence-only SOTA


| Capability | ESMDisPred | DisorderNet (Rockfish) |
|------------|------------|------------------------|
| Sequence-only disorder | ✓ | ✓ (ESM-2 650M–3B + LoRA) |
| Homology-safe CV | CAID protocol | ✓ `split_method=homology` in ultra |
| CAID3 benchmark eval | 0.895 ref | ✓ `caid3_eval_report.json` |
| AF hallucination rescue | ✗ | ✓ unique differentiator |
| Train-time pLDDT channel | ✗ | ✓ `use_plddt_features` in ultra |
| Proteome FASTA deploy | limited | ✓ `predict` stage |

Interactive debug (salloc GPU node):
```bash
salloc --partition=a100 --qos=qos_gpu --account=$DISORDERNET_ACCOUNT \
  --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00
source ~/venvs/disordernet/bin/activate
cd ~/DisorderNet
python rockfish/run_disordernet.py screen --screen-mode flash --backbone 650M
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
| `PROFILE` | `ultra` / `ultra3b` | Training profile |
| `BACKBONE` | `650M` / `3B` | ESM-2 size |
| `STAGE` | `full` | Pipeline stage |
| `SEED` | `42` | Random seed |

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
