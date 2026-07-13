#!/bin/bash
# Shared Slurm preamble for DisorderNet on JHU Rockfish.
# Sourced by individual .sbatch scripts — do not submit this file directly.

set -euo pipefail

# ── User overrides (export before sbatch) ─────────────────────────────────────
: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${DISORDERNET_PARTITION:=a100}"
: "${DISORDERNET_QOS:=qos_gpu}"
: "${DISORDERNET_VENV:=$HOME/venvs/disordernet}"
: "${DISORDERNET_REPO:=$HOME/DisorderNet}"
: "${DISORDERNET_WORKDIR:=}"

: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${SEED:=42}"
: "${STAGE:=full}"
: "${SCREEN_MODE:=standard}"
: "${NUM_WORKERS:=4}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Stage=$STAGE  profile=$PROFILE  backbone=$BACKBONE  seed=$SEED"
date

module purge 2>/dev/null || true
if command -v module &>/dev/null; then
  if module is-avail gcc/11.4.0 2>/dev/null; then module load gcc/11.4.0; fi
  if module is-avail cuda/11.8.0 2>/dev/null; then module load cuda/11.8.0; fi
fi

# shellcheck disable=SC1091
source "$DISORDERNET_VENV/bin/activate"
cd "$DISORDERNET_REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

: "${DISORDERNET_RESULTS:=$HOME/disordernet_runs}"

WORKDIR_ARG=()
if [[ -n "$DISORDERNET_WORKDIR" ]]; then
  WORKDIR_ARG=(--workdir "$DISORDERNET_WORKDIR")
fi

EXTRA_ARGS=()
if [[ "$STAGE" == "screen" ]]; then
  EXTRA_ARGS+=(--screen-mode "$SCREEN_MODE")
fi
if [[ "${PREFETCH_AF:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--prefetch-af-plddt)
fi
if [[ "${RUN_CAID3:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--run-caid3-eval)
fi
if [[ -n "${SEED_DIRS:-}" ]]; then
  EXTRA_ARGS+=(--seed-dirs "$SEED_DIRS")
fi
if [[ -n "${FASTA_PATH:-}" ]]; then
  EXTRA_ARGS+=(--fasta "$FASTA_PATH")
fi

CHECKPOINT_ARG=(--checkpoint-dir "${CHECKPOINT_SUBDIR:-checkpoints}")

python rockfish/run_disordernet.py "$STAGE" \
  --profile "$PROFILE" \
  --backbone "$BACKBONE" \
  --seed "$SEED" \
  --num-workers "$NUM_WORKERS" \
  "${WORKDIR_ARG[@]}" \
  "${CHECKPOINT_ARG[@]}" \
  "${EXTRA_ARGS[@]}"

RUN_TAG="${STAGE}_${PROFILE}_${BACKBONE}_s${SEED}_j${SLURM_JOB_ID:-local}"
DEST="$DISORDERNET_RESULTS/$RUN_TAG"
mkdir -p "$DEST"

for f in checkpoints/cv_progress.json checkpoints/cv_summary.json \
         checkpoints/run_manifest.json checkpoints/sota_postprocess_report.json \
         checkpoints/gpu_v6_ensemble_report.json checkpoints/sota_stack_report.json \
         checkpoints/quick_screen_*.json checkpoints/eval_summary.json \
         checkpoints/caid3_eval_report.json checkpoints/phase3_integrated_report.json \
         checkpoints/statistical_validation_report.json checkpoints/af_rescue_report.json \
         checkpoints/inference_fusion_report.json checkpoints/af_rescue_manifest.json \
         checkpoints/multi_seed_blend_report.json; do
  [[ -f "$f" ]] && cp -a "$f" "$DEST/"
done

if [[ -d checkpoints ]]; then
  find checkpoints -maxdepth 1 -name 'fold_*_compact.pt' -exec cp -a {} "$DEST/" \; 2>/dev/null || true
fi

echo "Results mirrored to $DEST"
date
