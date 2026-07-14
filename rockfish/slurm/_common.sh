#!/bin/bash
# Shared Slurm helpers for DisorderNet on JHU Rockfish.
# Sourced by individual .sbatch scripts — do not submit this file directly.
#
# Provides:
#   disordernet_slurm_setup   — modules + venv + cd repo (fail-loud if missing)
#   disordernet_slurm_run     — run rockfish/run_disordernet.py with env flags
#   disordernet_slurm_mirror  — mirror reports to $RESULTS_DIR/<tag>
#
# Sbatch entrypoints use: #!/bin/bash -ue  and  #SBATCH --export=ALL
# Parent scripts should set PROJECT_DIR / ENV_DIR / WK_DIR before sourcing,
# or rely on the defaults below.
#
# Autorun on source unless DISORDERNET_COMMON_AUTORUN=0.
#
# Shell contract: callers use `bash -ue`. We re-assert pipefail so pipelines
# fail on any command status, not only the last.
# shellcheck shell=bash

set -euo pipefail

# ── Paths (lab convention; override via env or parent script) ─────────────────
PROJECT_DIR="${DISORDERNET_REPO:-${PROJECT_DIR:-${HOME}/DisorderNet}}"
ENV_DIR="${DISORDERNET_VENV:-${ENV_DIR:-${HOME}/venvs/disordernet}}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${RESULTS_DIR:-${HOME}/disordernet_runs}}"
WK_DIR="${DISORDERNET_WORKDIR:-${WK_DIR:-}}"

# Keep env aliases in sync for CLI / Python
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
[[ -n "${WK_DIR}" ]] && export DISORDERNET_WORKDIR="${WK_DIR}"

: "${DISORDERNET_ACCOUNT:=sfried3}"
: "${DISORDERNET_PARTITION:=a100}"
: "${DISORDERNET_QOS:=qos_gpu}"

: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${SEED:=42}"
: "${STAGE:=full}"
: "${SCREEN_MODE:=standard}"
: "${NUM_WORKERS:=4}"

disordernet_slurm_setup() {
  echo "============================================================"
  echo "DisorderNet Rockfish"
  echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
  echo "Stage=${STAGE}  profile=${PROFILE}  backbone=${BACKBONE}  seed=${SEED}"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  echo "WK_DIR=${WK_DIR:-"(repo cwd / default scratch)"}"
  echo "ENV_DIR=${ENV_DIR}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  date -Is
  echo "============================================================"

  # Prefer Rockfish `ml` shorthand; fall back to `module`
  if command -v ml &>/dev/null; then
    ml purge 2>/dev/null || true
    ml gcc/11.4.0 2>/dev/null || true
    ml cuda/11.8.0 2>/dev/null || true
  elif command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    if module is-avail gcc/11.4.0 2>/dev/null; then module load gcc/11.4.0; fi
    if module is-avail cuda/11.8.0 2>/dev/null; then module load cuda/11.8.0; fi
  fi

  if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
    echo "ERROR: Python venv missing: ${ENV_DIR}/bin/activate" >&2
    echo "Run: bash rockfish/setup_env.sh" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "${ENV_DIR}/bin/activate"

  if [[ ! -f "${PROJECT_DIR}/rockfish/run_disordernet.py" ]]; then
    echo "ERROR: Cannot find rockfish/run_disordernet.py under PROJECT_DIR=${PROJECT_DIR}" >&2
    exit 1
  fi

  cd "${PROJECT_DIR}" || exit 1
  mkdir -p "${PROJECT_DIR}/logs" "${RESULTS_DIR}"
  if [[ -n "${WK_DIR}" ]]; then
    mkdir -p "${WK_DIR}" || exit 1
  fi

  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"
  export TOKENIZERS_PARALLELISM=false
  export TMPDIR="${TMPDIR:-/tmp}"
  export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${PROJECT_DIR}/.cache/torch}"
  mkdir -p "${HF_HOME}" "${TORCH_HOME}"
}

disordernet_slurm_run() {
  local WORKDIR_ARG=()
  if [[ -n "${DISORDERNET_WORKDIR:-}" ]]; then
    WORKDIR_ARG=(--workdir "${DISORDERNET_WORKDIR}")
  fi

  local EXTRA_ARGS=()
  if [[ "${STAGE}" == "screen" ]]; then
    EXTRA_ARGS+=(--screen-mode "${SCREEN_MODE}")
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
    EXTRA_ARGS+=(--seed-dirs "${SEED_DIRS}")
  fi
  if [[ -n "${FASTA_PATH:-}" ]]; then
    EXTRA_ARGS+=(--fasta "${FASTA_PATH}")
  fi
  if [[ "${AF3_MODE:-off}" != "off" ]]; then
    EXTRA_ARGS+=(--af3-mode "${AF3_MODE}")
  fi
  if [[ -n "${DISORDERNET_AF3_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--af3-root "${DISORDERNET_AF3_ROOT}")
  fi
  if [[ -n "${AF3_MAX_PROTEINS:-}" ]]; then
    EXTRA_ARGS+=(--af3-max-proteins "${AF3_MAX_PROTEINS}")
  fi
  if [[ -n "${AF3_SHARD_INDEX:-}" && -n "${AF3_SHARD_COUNT:-}" ]]; then
    EXTRA_ARGS+=(--af3-shard-index "${AF3_SHARD_INDEX}" --af3-shard-count "${AF3_SHARD_COUNT}")
  fi

  : "${BOLTZ_MODE:=ingest}"
  : "${STRUCTURE_BACKEND:=boltz}"
  EXTRA_ARGS+=(--structure-backend "${STRUCTURE_BACKEND}")
  EXTRA_ARGS+=(--boltz-mode "${BOLTZ_MODE}")
  if [[ -n "${DISORDERNET_BOLTZ_ROOT:-}" ]]; then
    EXTRA_ARGS+=(--boltz-root "${DISORDERNET_BOLTZ_ROOT}")
  fi
  if [[ -n "${BOLTZ_MAX_PROTEINS:-}" ]]; then
    EXTRA_ARGS+=(--boltz-max-proteins "${BOLTZ_MAX_PROTEINS}")
  fi
  if [[ -n "${BOLTZ_SHARD_INDEX:-}" && -n "${BOLTZ_SHARD_COUNT:-}" ]]; then
    EXTRA_ARGS+=(--boltz-shard-index "${BOLTZ_SHARD_INDEX}" --boltz-shard-count "${BOLTZ_SHARD_COUNT}")
  fi
  if [[ "${BOLTZ_USE_MSA_SERVER:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--boltz-use-msa-server)
  fi

  local CHECKPOINT_ARG=(--checkpoint-dir "${CHECKPOINT_SUBDIR:-checkpoints}")

  echo "Running: stage=${STAGE} profile=${PROFILE} backbone=${BACKBONE} ckpt=${CHECKPOINT_SUBDIR:-checkpoints}"
  python rockfish/run_disordernet.py "${STAGE}" \
    --profile "${PROFILE}" \
    --backbone "${BACKBONE}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
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
  local DEST="${RESULTS_DIR}/${RUN_TAG}"
  mkdir -p "${DEST}"

  local CWD="."
  if [[ -n "${DISORDERNET_WORKDIR:-}" ]]; then
    CWD="${DISORDERNET_WORKDIR}"
  fi

  # Fail loud on empty mirrors when MIRROR_REQUIRE_MIN_FILES is set (default 1
  # for GPU pipeline jobs — an empty mirror after training is almost always a bug).
  local min_files="${MIRROR_REQUIRE_MIN_FILES:-1}"
  python rockfish/mirror_results.py \
    --dest "${DEST}" \
    --workers "${MIRROR_WORKERS:-8}" \
    --cwd "${CWD}" \
    --require-min-files "${min_files}"

  echo "Results mirrored to ${DEST}"
  export DISORDERNET_LAST_MIRROR_DEST="${DEST}"
  date -Is
}

if [[ "${DISORDERNET_COMMON_AUTORUN:-1}" == "1" ]]; then
  disordernet_slurm_setup
  disordernet_slurm_run
  if [[ "${SKIP_MIRROR:-0}" != "1" ]]; then
    disordernet_slurm_mirror
  fi
fi
