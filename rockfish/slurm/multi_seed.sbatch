#!/bin/bash -ue
#SBATCH --job-name="dn-seed"
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=03-00:00:00
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --export=ALL
#SBATCH --array=0-2
#SBATCH --output=logs/dn-seed_%A_%a.out
#SBATCH --error=logs/dn-seed_%A_%a.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT

# Multi-seed CV array (seeds 42, 43, 44) for +0.005–0.015 AUC blend.
#
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   export DISORDERNET_WORKDIR="${HOME}/disordernet_runs/ultra_multiseed"
#   sbatch --account="${DISORDERNET_ACCOUNT}" \\
#     --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,PROFILE=ultra,BACKBONE=650M \\
#     rockfish/slurm/multi_seed.sbatch
#
# After array finishes, blend:
#   export SEED_DIRS="${DISORDERNET_WORKDIR}/seed_42/checkpoints_s42,..."
#   sbatch --export=ALL,STAGE=multi-seed-blend,SEED_DIRS rockfish/slurm/blend_seeds.sbatch

: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=full}"
SEEDS=(42 43 44)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$IDX]}"
: "${DISORDERNET_WORKDIR:=${HOME}/disordernet_runs/ultra_multiseed}"
export STAGE PROFILE BACKBONE SEED
export DISORDERNET_WORKDIR="${DISORDERNET_WORKDIR}/seed_${SEED}"
export CHECKPOINT_SUBDIR="checkpoints_s${SEED}"
export PREFETCH_AF=1

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
