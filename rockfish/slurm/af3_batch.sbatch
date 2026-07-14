#!/bin/bash -ue
#SBATCH --job-name="dn-af3"
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03-00:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-af3_%j.out
#SBATCH --error=logs/dn-af3_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Run / ingest AlphaFold 3 (MSA-free by default; licensed af3.bin required).
#
#   export DISORDERNET_AF3_ROOT="${HOME}/af3"
#   sbatch --account="${DISORDERNET_ACCOUNT}" \
#     --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=run \
#     rockfish/slurm/af3_batch.sbatch

: "${AF3_MODE:=run}"
: "${STAGE:=af3}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
export STAGE PROFILE BACKBONE AF3_MODE
export DISORDERNET_AF3_ROOT="${DISORDERNET_AF3_ROOT:-${HOME}/af3}"

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
