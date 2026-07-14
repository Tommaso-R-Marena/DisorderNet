#!/bin/bash -ue
#SBATCH --job-name="dn-boltz-arr"
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02-00:00:00
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --export=ALL
#SBATCH --array=0-7
#SBATCH --output=logs/dn-boltz-arr_%A_%a.out
#SBATCH --error=logs/dn-boltz-arr_%A_%a.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Parallel Boltz-2 over pending proteins (Slurm array).
#
#   export BOLTZ_SHARD_COUNT=8
#   sbatch --account="${DISORDERNET_ACCOUNT}" --array=0-7 \
#     --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=run,BOLTZ_SHARD_COUNT=8 \
#     rockfish/slurm/boltz_array.sbatch

: "${BOLTZ_MODE:=run}"
: "${STAGE:=boltz}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STRUCTURE_BACKEND:=boltz}"
: "${BOLTZ_SHARD_COUNT:=8}"
export STAGE PROFILE BACKBONE STRUCTURE_BACKEND BOLTZ_MODE
export DISORDERNET_BOLTZ_ROOT="${DISORDERNET_BOLTZ_ROOT:-${HOME}/boltz}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-${DISORDERNET_BOLTZ_ROOT}/cache}"
export BOLTZ_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export BOLTZ_SHARD_COUNT

# ── Paths (override via env before sbatch) ───────────────────────────────────
PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs}"
WK_DIR="${DISORDERNET_WORKDIR:-}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
[[ -n "${WK_DIR}" ]] && export DISORDERNET_WORKDIR="${WK_DIR}"
mkdir -p "${PROJECT_DIR}/logs"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
