# DisorderNet v8 on Rockfish (JHU) — exact runbook

This runs the **v8 multi-scale ensemble** (ESM-2 35M + 150M + 650M → ~0.857 AUC,
calibrated + conformal) on JHU Rockfish. GPU is used only for the fast embedding
extraction; the GBDT training runs on CPU (`shared` partition) so we never hold a
GPU idle.

Two Slurm scripts, matched to the lab's conventions: **GPU jobs** need the GPU
account + `qos_gpu` (e.g. `-A sfried3_gpu --qos=qos_gpu`); **CPU jobs** stay on
`-A sfried3` / `shared`. The `#SBATCH --account=` line inside the scripts is only a
default — always override on the `sbatch` command line.

| Script | Partition | Account / QOS | Purpose | Typical wall |
|--------|-----------|---------------|---------|--------------|
| `rockfish/slurm/v8_extract_embeddings.sbatch` | `a100` (1 GPU) | **GPU account + `qos_gpu`** | extract 35M/150M/650M embeddings | ~1–2 h once started |
| `rockfish/slurm/v8_pipeline.sbatch` | `shared` (8 CPU) | **CPU account `sfried3`** (no qos) | v7 CV ×3 (+ homology) → v8 ensemble | ~6–12 h once started |

> **Courteous allocation notes.** These request the minimum that fits: one A100 for a
> few hours (GPU is scarce), and a single `shared` CPU slot for the trees. Please keep
> `--mail-type` off unless you want mail, don't submit many copies, and let the two
> jobs run once. Everything writes under a **clearly-labelled** project area
> (`~/scr4_sfried3/disordernet_v8` and `~/DisorderNet`); nothing else is touched.

> ⚠️ **Rotate the SSH key you pasted earlier** — it is now exposed. On your laptop:
> `ssh-keygen -t ed25519 -f ~/.ssh/rockfish_id_new`, then
> `ssh-copy-id -i ~/.ssh/rockfish_id_new.pub rockfish`, update `~/.ssh/config` to use
> the new key, and remove the old public key from `~/.ssh/authorized_keys` on Rockfish.

---

## 0 · Connect (on your Ubuntu laptop)

```bash
ssh rockfish            # uses your ~/.ssh/config alias -> jbeale3@login.rockfish.jhu.edu
```

Everything in sections 1–4 runs **on the Rockfish login node** (submitting jobs is
fine there; the actual compute runs on the allocated nodes — never run training on
the login node).

## 1 · One-time setup (on Rockfish)

```bash
# Clone the repo into your home (or `git pull` if already present)
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet
cd ~/DisorderNet && git checkout master

# Build the Python venv (creates ~/venvs/disordernet and installs deps)
bash rockfish/setup_env.sh
source ~/venvs/disordernet/bin/activate
mkdir -p logs

# Also install the CPU pipeline deps (lightgbm/xgboost/fair-esm) if not already:
pip install -r requirements-cpu.txt
```

The working data/results live under scratch (plenty of space in `scr4_sfried3`):

```bash
export DISORDERNET_V8_DIR=$HOME/scr4_sfried3/disordernet_v8
```

## 2 · Submit the jobs (on Rockfish)

```bash
cd ~/DisorderNet && git pull && source ~/venvs/disordernet/bin/activate
export DISORDERNET_V8_DIR=$HOME/scr4_sfried3/disordernet_v8

# Preferred — discovers sfried3_gpu and always passes --qos=qos_gpu
# (avoids "Invalid qos specification" from an empty --qos="$DISORDERNET_GPU_QOS")
bash rockfish/slurm/submit_v8.sh
```

Manual equivalent:

```bash
# If a previous pair is stuck Pending (QOS / Dependency), cancel first:
#   scancel <EMBED_ID> <PIPELINE_ID>

EMBED_ID=$(sbatch --parsable \
  -A sfried3_gpu --qos=qos_gpu \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_extract_embeddings.sbatch)
echo "embedding job: $EMBED_ID"

sbatch -A sfried3 \
  --dependency=afterok:$EMBED_ID \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_pipeline.sbatch
```

Want just one backbone (faster) or to skip the homology pass?

```bash
# only 650M extract (still needs GPU account + literal qos_gpu)
sbatch -A sfried3_gpu --qos=qos_gpu \
  --export=ALL,DISORDERNET_V8_DIR,DISORDERNET_BACKBONES=esm2_t33_650M_UR50D \
  rockfish/slurm/v8_extract_embeddings.sbatch
# then pipeline with RUN_HOMOLOGY=0 if desired
```

## 3 · Monitor — how you know it’s finished

```bash
squeue -u $USER
# PD = waiting, R = running. Empty queue (job gone) = Slurm released it.
sacct -j $EMBED_ID --format=JobID,State,ExitCode,Elapsed -P
# Success looks like: COMPLETED|0:0

tail -f logs/dn-v8-embed_*.out    # extraction progress (~1–2 h once R)
tail -f logs/dn-v8-cv_*.out       # CV + ensemble (~6–12 h once R)
```

| State / symptom | What it means | What to do |
|-----------------|---------------|------------|
| Embed `R`, pipeline `PD (Dependency)` | Normal — wait for embed | `tail -f` embed log |
| Embed `PD` reason mentions **QOS** | Wrong account/qos | `scancel` both; resubmit with GPU account + `qos_gpu` |
| Embed `COMPLETED`, pipeline `R` | Normal | `tail -f` cv log |
| Both gone + `metrics.json` exists | **Done** | Open results (§4) |
| Pipeline `PD (Dependency)` forever | Embed never succeeded | `sacct` embed; fix; resubmit both |

## 4 · Results (what you get + how to see them)

**Done when** these files exist and `sacct` shows the pipeline job `COMPLETED|0:0`:

```bash
cat $DISORDERNET_V8_DIR/ensemble/results_v8/metrics.json        # expect AUC ~0.85–0.86
cat $DISORDERNET_V8_DIR/ensemble_hom/results_v8/metrics.json    # homology; slightly lower
ls  $DISORDERNET_V8_DIR/{35m,150m,650m}/results_v7/             # per-backbone metrics + OOF
```

Pull them to your laptop (run this **on your Ubuntu laptop**, not Rockfish):

```bash
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble ./dn_v8_ensemble
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble_hom ./dn_v8_ensemble_hom
```

**Then** (only if you want the LoRA / paper package track) run the publish scripts in
§ “Full publishable run” below — do **not** start those until v8 results look sane.

## GPU LoRA path (targets ≥0.88)

The fine-tuned SOTA path is separate and already has full Rockfish support — see
[`rockfish/README.md`](README.md) (`submit_publish_650m.sh` / `submit_publish_3b.sh`).
Its `run_cross_validation` now auto-attaches the same calibrated + conformal
confidence report to the CV summary.

---

## Full publishable run — exact order (tests + v8 + 650M + 3B)

Copy-paste, top to bottom, **on the Rockfish login node** (steps run on compute
nodes via `srun`/`sbatch`). Budget: 650M (ultra+clean) ≈ 50–100 GPU-h, 3B ≈ 60–120
GPU-h, v8 extract ≈ 2 GPU-h — comfortably within a 15k GPU-h / 150k CPU-h allocation.

```bash
# ── 0 · One-time setup ────────────────────────────────────────────────────
git clone https://github.com/Tommaso-R-Marena/DisorderNet.git ~/DisorderNet 2>/dev/null || true
cd ~/DisorderNet && git checkout master && git pull
bash rockfish/setup_env.sh
source ~/venvs/disordernet/bin/activate
pip install -r requirements-dev.txt          # adds pytest/ruff (torch already installed)
mkdir -p logs
export DISORDERNET_ACCOUNT=sfried3            # CPU account (shared partition)
export DISORDERNET_V8_DIR=$HOME/scr4_sfried3/disordernet_v8
export DISORDERNET_BOLTZ_ROOT=$HOME/scr4_sfried3/boltz
export BOLTZ_CACHE=$DISORDERNET_BOLTZ_ROOT/cache

# The a100 GPU partition requires a GPU QOS (qos_gpu) which lives on a GPU account
# (usually <group>_gpu, e.g. sfried3_gpu). Auto-discover the account that has qos_gpu:
export DISORDERNET_GPU_ACCOUNT=$(sacctmgr -nP show assoc user=$USER format=account,qos | awk -F'|' '/qos_gpu/{print $1; exit}')
export DISORDERNET_GPU_QOS=qos_gpu
echo "GPU account: ${DISORDERNET_GPU_ACCOUNT:-<none found — ask your PI for GPU access>}  QOS: $DISORDERNET_GPU_QOS"

# ── 1 · Tests (compute node, ~2 min) ──────────────────────────────────────
srun -A sfried3 -p shared -c 4 --mem=8G -t 00:20:00 \
  bash -lc 'cd ~/DisorderNet && source ~/venvs/disordernet/bin/activate && ruff check . && pytest tests/ -q'

# ── 1b · Prefetch DisProt + ESM weights on the LOGIN node (has internet) ───
# Downloads once into the shared home cache so the compute-node jobs never need
# internet (and never re-download). ~2.5 GB for 650M; a few minutes.
# NB: use prefetch_esm.py (streams the .pt files to disk) — do NOT instantiate the
# models on the login node, that gets OOM-killed.
DISORDERNET_HOME=$DISORDERNET_V8_DIR python fetch_disprot.py
python rockfish/prefetch_esm.py

# ── 2 · v8 multi-scale ensemble (honest CPU numbers + calibration/conformal) ─
# Prefer the helper (never submits an empty --qos):
bash rockfish/slurm/submit_v8.sh
# Manual: GPU extract needs -A sfried3_gpu --qos=qos_gpu; CPU pipeline stays on sfried3/shared.
# EMBED=$(sbatch --parsable -A sfried3_gpu --qos=qos_gpu rockfish/slurm/v8_extract_embeddings.sbatch)
# sbatch -A sfried3 --dependency=afterok:$EMBED rockfish/slurm/v8_pipeline.sbatch

# ── 3 · 650M publishable bundle (ultra -> ultra_clean -> package) ──────────
# GPU jobs use the GPU account + qos_gpu; the CPU package job auto-uses sfried3.
sbatch -A "$DISORDERNET_GPU_ACCOUNT" --qos=qos_gpu \
  --export=ALL,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
  rockfish/slurm/boltz_batch.sbatch          # optional but recommended (go/no-go artifact)
bash rockfish/slurm/submit_publish_650m.sh --account "$DISORDERNET_GPU_ACCOUNT" --qos qos_gpu

# ── 4 · 3B publishable bundle (optional) ──────────────────────────────────
bash rockfish/slurm/submit_publish_3b.sh --account "$DISORDERNET_GPU_ACCOUNT" --qos qos_gpu
#   add: --partition ica100  if OOM on 40GB A100

# ── 5 · Monitor ───────────────────────────────────────────────────────────
squeue -u $USER

# ── 6 · Inspect (after jobs finish) ───────────────────────────────────────
cat  $DISORDERNET_V8_DIR/ensemble/results_v8/metrics.json
cat  $DISORDERNET_V8_DIR/ensemble_hom/results_v8/metrics.json
less ~/disordernet_runs/publish_650m_*/publish_package/PACKAGE_README.md
cat  ~/disordernet_runs/publish_650m_*/publish_package/comparison.json
```

```bash
# ── 7 · Pull results to your laptop (run in a LOCAL terminal, not Rockfish) ─
scp -r rockfish:disordernet_runs/publish_650m_*           ./publish_650m
scp -r rockfish:disordernet_runs/publish_3b_*             ./publish_3b
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble      ./dn_v8_ensemble
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble_hom  ./dn_v8_ensemble_hom
```

To halve GPU cost, add `--no-clean` to the publish scripts (drops the
contamination-clean companion — but that ablation is part of the strict go/no-go).

