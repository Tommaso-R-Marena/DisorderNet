#!/bin/bash -ue
#SBATCH --job-name="dn-stack"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00-08:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --export=ALL
#SBATCH --output=logs/dn-stack_%j.out
#SBATCH --error=logs/dn-stack_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Post-CV ensemble on CPU (STAGE=stack). Postprocess/fold-soup still needs GPU.
#
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,STAGE=stack rockfish/slurm/postprocess_cpu.sbatch

: "${STAGE:=stack}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
export STAGE PROFILE BACKBONE

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
