#!/bin/bash -ue
#SBATCH --job-name="dn-ultra"
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=03-00:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-ultra_%j.out
#SBATCH --error=logs/dn-ultra_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Full ultra pipeline on ESM-2 650M: CV → stack → postprocess (~18–24 h).
#
#   sbatch --account="${DISORDERNET_ACCOUNT}" rockfish/slurm/train_ultra.sbatch

: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=full}"
export STAGE PROFILE BACKBONE
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
