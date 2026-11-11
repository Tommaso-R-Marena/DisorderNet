#!/bin/bash -ue
#SBATCH --job-name="dn-pipeline-clean"
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=48000M
#SBATCH --time=03-00:00:00
#SBATCH --account=sfried3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --export=ALL
#SBATCH --output=logs/dn-pipeline-clean_%j.out
#SBATCH --error=logs/dn-pipeline-clean_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Contamination-clean companion (ultra capacity, structure-training flags off).
#
# Submit:
#   export DISORDERNET_ACCOUNT=sfried3
#   export DISORDERNET_WORKDIR="${HOME}/disordernet_runs/ultra_clean"
#   sbatch --account="${DISORDERNET_ACCOUNT}" rockfish/slurm/pipeline_ultra_clean.sbatch

: "${PROFILE:=ultra_clean}"
: "${BACKBONE:=650M}"
: "${STAGE:=pipeline}"
: "${DISORDERNET_WORKDIR:=${HOME}/disordernet_runs/ultra_clean}"
: "${CHECKPOINT_SUBDIR:=checkpoints_ultra_clean}"

export STAGE PROFILE BACKBONE
export DISORDERNET_WORKDIR CHECKPOINT_SUBDIR
export PREFETCH_AF=1
export RUN_CAID3=1
export RUN_NO_HALLUC_WEIGHT=1
export RUN_NO_PLDDT_FEATURES=1

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


# Locate _common.sh. Slurm copies the batch script into a per-job spool dir, so
# ${BASH_SOURCE[0]} does NOT resolve to the repo checkout at runtime — resolving
# relative to it makes every Slurm job die instantly with "No such file or
# directory". Resolve against the repo first; fall back to the script dir for
# local (non-Slurm) invocation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_DIR}/rockfish/slurm/_common.sh" ]]; then
  COMMON_SH="${PROJECT_DIR}/rockfish/slurm/_common.sh"
elif [[ -f "${SCRIPT_DIR}/_common.sh" ]]; then
  COMMON_SH="${SCRIPT_DIR}/_common.sh"
else
  echo "ERROR: cannot locate rockfish/slurm/_common.sh" >&2
  echo "  PROJECT_DIR=${PROJECT_DIR}  SCRIPT_DIR=${SCRIPT_DIR}" >&2
  echo "  Set DISORDERNET_REPO to your DisorderNet checkout and resubmit." >&2
  exit 1
fi
# shellcheck source=_common.sh
source "${COMMON_SH}"
