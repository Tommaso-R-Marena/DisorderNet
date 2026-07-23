# DisorderNet v8 on Rockfish (JHU) — exact runbook

This runs the **v8 multi-scale ensemble** (ESM-2 35M + 150M + 650M → ~0.857 AUC,
calibrated + conformal) on JHU Rockfish. GPU is used only for the fast embedding
extraction; the GBDT training runs on CPU (`shared` partition) so we never hold a
GPU idle.

Two Slurm scripts, matched to the lab's conventions (`account=sfried3`, GPU on
`a100` + `qos_gpu`, CPU on `shared`) and modeled on the group's APBS array job:

| Script | Partition | Purpose | Typical time |
|--------|-----------|---------|--------------|
| `rockfish/slurm/v8_extract_embeddings.sbatch` | `a100` (1 GPU) | extract 35M/150M/650M embeddings | ~1–2 h |
| `rockfish/slurm/v8_pipeline.sbatch` | `shared` (8 CPU) | v7 CV ×3 backbones (+ homology) → v8 ensemble → predictor | ~6–12 h |

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
cd ~/DisorderNet

# (a) GPU embedding extraction
EMBED_ID=$(sbatch --parsable --account=sfried3 \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_extract_embeddings.sbatch)
echo "embedding job: $EMBED_ID"

# (b) CPU pipeline — starts automatically after (a) succeeds
sbatch --account=sfried3 \
  --dependency=afterok:$EMBED_ID \
  --export=ALL,DISORDERNET_V8_DIR \
  rockfish/slurm/v8_pipeline.sbatch
```

Want just one backbone (faster) or to skip the homology pass?

```bash
# only 650M, random split only
sbatch --account=sfried3 \
  --export=ALL,DISORDERNET_V8_DIR,DISORDERNET_BACKBONES=esm2_t33_650M_UR50D,RUN_HOMOLOGY=0 \
  rockfish/slurm/v8_extract_embeddings.sbatch
```

## 3 · Monitor (on Rockfish)

```bash
squeue -u $USER
tail -f logs/dn-v8-embed_*.out    # extraction progress
tail -f logs/dn-v8-cv_*.out       # CV + ensemble progress
```

## 4 · Results

```bash
cat $DISORDERNET_V8_DIR/ensemble/results_v8/metrics.json        # random-split ensemble
cat $DISORDERNET_V8_DIR/ensemble_hom/results_v8/metrics.json    # homology-split ensemble
ls  $DISORDERNET_V8_DIR/35m/results_v7/                         # per-backbone metrics + OOF
```

Pull them to your laptop (run this **on your Ubuntu laptop**, not Rockfish):

```bash
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble ./dn_v8_ensemble
scp -r rockfish:scr4_sfried3/disordernet_v8/ensemble_hom ./dn_v8_ensemble_hom
```

## GPU LoRA path (targets ≥0.88)

The fine-tuned SOTA path is separate and already has full Rockfish support — see
[`rockfish/README.md`](README.md) (`submit_publish_650m.sh` / `submit_publish_3b.sh`).
Its `run_cross_validation` now auto-attaches the same calibrated + conformal
confidence report to the CV summary.
