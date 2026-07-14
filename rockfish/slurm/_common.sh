#!/bin/bash
# Shared Slurm helpers for DisorderNet on JHU Rockfish.
# Sourced by individual .sbatch scripts — do not submit this file directly.
#
# Provides:
#   disordernet_slurm_setup   — modules + venv + cd repo
#   disordernet_slurm_run     — run rockfish/run_disordernet.py with env flags
#   disordernet_slurm_mirror  — mirror reports to $DISORDERNET_RESULTS/<tag>
#
# Existing sbatch files keep working: sourcing this file autoruns setup+run+mirror
# unless DISORDERNET_COMMON_AUTORUN=0.

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

disordernet_slurm_setup() {
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
}

disordernet_slurm_run() {
  local WORKDIR_ARG=()
  if [[ -n "${DISORDERNET_WORKDIR:-}" ]]; then
    WORKDIR_ARG=(--workdir "$DISORDERNET_WORKDIR")
  fi

  local EXTRA_ARGS=()
  if [[ "$STAGE" == "screen" ]]; then
    EXTRA_ARGS+=(--screen-mode "$SCREEN_MODE")
  fi
  if [[ "${PREFETCH_AF:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--prefetch-af-plddt)
  fi
  if [[ "${RUN_CAID3:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--run-caid3-eval)
  fi
  if [[ "${RUN_NO_HALLUC_WEIGHT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-hallucination-weighting)
  fi
  if [[ "${RUN_NO_PLDDT_FEATURES:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-plddt-features)
  fi
  if [[ -n "${SEED_DIRS:-}" ]]; then
    EXTRA_ARGS+=(--seed-dirs "$SEED_DIRS")
  fi
  if [[ -n "${FASTA_PATH:-}" ]]; then
    EXTRA_ARGS+=(--fasta "$FASTA_PATH")
  fi
  if [[ "${AF3_MODE:-off}" != "off" ]]; then
    EXTRA_ARGS+=(--af3-mode "$AF3_MODE")
  fi
  if [[ -n "${DISORDERNET_AF3_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--af3-root "$DISORDERNET_AF3_ROOT")
  fi
  if [[ -n "${AF3_MAX_PROTEINS:-}" ]]; then
    EXTRA_ARGS+=(--af3-max-proteins "$AF3_MAX_PROTEINS")
  fi
  if [[ -n "${AF3_SHARD_INDEX:-}" && -n "${AF3_SHARD_COUNT:-}" ]]; then
    EXTRA_ARGS+=(--af3-shard-index "$AF3_SHARD_INDEX" --af3-shard-count "$AF3_SHARD_COUNT")
  fi

  # Boltz-2 (default structure backend). Training jobs default to ingest-only;
  # use BOLTZ_MODE=auto or rockfish/slurm/boltz_batch.sbatch to run predictions
  # (pinned boltz auto-downloads weights on first use).
  : "${BOLTZ_MODE:=ingest}"
  : "${STRUCTURE_BACKEND:=boltz}"
  EXTRA_ARGS+=(--structure-backend "$STRUCTURE_BACKEND")
  EXTRA_ARGS+=(--boltz-mode "$BOLTZ_MODE")
  if [[ -n "${DISORDERNET_BOLTZ_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--boltz-root "$DISORDERNET_BOLTZ_ROOT")
  fi
  if [[ -n "${BOLTZ_MAX_PROTEINS:-}" ]]; then
    EXTRA_ARGS+=(--boltz-max-proteins "$BOLTZ_MAX_PROTEINS")
  fi
  if [[ -n "${BOLTZ_SHARD_INDEX:-}" && -n "${BOLTZ_SHARD_COUNT:-}" ]]; then
    EXTRA_ARGS+=(--boltz-shard-index "$BOLTZ_SHARD_INDEX" --boltz-shard-count "$BOLTZ_SHARD_COUNT")
  fi
  if [[ "${BOLTZ_USE_MSA_SERVER:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--boltz-use-msa-server)
  fi

  local CHECKPOINT_ARG=(--checkpoint-dir "${CHECKPOINT_SUBDIR:-checkpoints}")

  echo "▶ Running: stage=$STAGE profile=$PROFILE backbone=$BACKBONE ckpt=${CHECKPOINT_SUBDIR:-checkpoints}"
  python rockfish/run_disordernet.py "$STAGE" \
    --profile "$PROFILE" \
    --backbone "$BACKBONE" \
    --seed "$SEED" \
    --num-workers "$NUM_WORKERS" \
    "${WORKDIR_ARG[@]}" \
    "${CHECKPOINT_ARG[@]}" \
    "${EXTRA_ARGS[@]}"
}

disordernet_slurm_mirror() {
  local RUN_TAG="${STAGE}_${PROFILE}_${BACKBONE}_s${SEED}_j${SLURM_JOB_ID:-local}"
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    RUN_TAG="${RUN_TAG}_a${SLURM_ARRAY_TASK_ID}"
  fi
  if [[ -n "${MIRROR_TAG_SUFFIX:-}" ]]; then
    RUN_TAG="${RUN_TAG}_${MIRROR_TAG_SUFFIX}"
  fi
  local DEST="${DISORDERNET_RESULTS:-$HOME/disordernet_runs}/$RUN_TAG"
  mkdir -p "$DEST"

  local CWD="."
  if [[ -n "${DISORDERNET_WORKDIR:-}" ]]; then
    CWD="$DISORDERNET_WORKDIR"
  fi

  # Parallel mirror (threads) — faster than serial cp for many reports
  python rockfish/mirror_results.py --dest "$DEST" --workers "${MIRROR_WORKERS:-8}" --cwd "$CWD"

  echo "Results mirrored to $DEST"
  # Stash last mirror dest for packaging wrappers
  export DISORDERNET_LAST_MIRROR_DEST="$DEST"
  date
}

# Default: behave like the historical _common.sh (autorun on source)
if [[ "${DISORDERNET_COMMON_AUTORUN:-1}" == "1" ]]; then
  disordernet_slurm_setup
  disordernet_slurm_run
  if [[ "${SKIP_MIRROR:-0}" != "1" ]]; then
    disordernet_slurm_mirror
  fi
fi
