#!/bin/bash -ue
#SBATCH --job-name="dn-blend"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00-02:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --export=ALL
#SBATCH --output=logs/dn-blend_%j.out
#SBATCH --error=logs/dn-blend_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Blend multi-seed OOF predictions (CPU-only).
#
#   export SEED_DIRS="path/checkpoints_s42,path/checkpoints_s43,path/checkpoints_s44"
#   sbatch --account="${DISORDERNET_ACCOUNT}" rockfish/slurm/blend_seeds.sbatch

: "${STAGE:=multi-seed-blend}"
: "${SEED_DIRS:?Set SEED_DIRS to comma-separated checkpoint directories}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
export STAGE SEED_DIRS PROFILE BACKBONE

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
