#!/bin/bash -ue
#SBATCH --job-name="dn-smoke"
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16000M
#SBATCH --time=00-00:20:00
#SBATCH --account=sfried3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --export=ALL
#SBATCH --output=logs/dn-smoke_%j.out
#SBATCH --error=logs/dn-smoke_%j.err
#
# Cheap preflight for the publish campaign. Proves, on a real GPU node, that:
#   1. _common.sh resolves even though Slurm runs the script from a spool dir
#      (this is exactly what killed every job in the previous campaign),
#   2. the venv activates and CUDA is actually visible,
#   3. the repo imports and the autojunk homology guard is in force.
# Twenty minutes of walltime, 1/12th of a node — never commit 72 h without it.

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/dn_rigor}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet_rigor}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs_rigor}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${RESULTS_DIR}"

# Source the shared helpers but do NOT autorun the training pipeline.
export DISORDERNET_COMMON_AUTORUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_DIR}/rockfish/slurm/_common.sh" ]]; then
  COMMON_SH="${PROJECT_DIR}/rockfish/slurm/_common.sh"
elif [[ -f "${SCRIPT_DIR}/_common.sh" ]]; then
  COMMON_SH="${SCRIPT_DIR}/_common.sh"
else
  echo "FAIL: cannot locate _common.sh" >&2
  echo "  PROJECT_DIR=${PROJECT_DIR}  SCRIPT_DIR=${SCRIPT_DIR}" >&2
  exit 1
fi
echo "OK: resolved _common.sh -> ${COMMON_SH}"
echo "    Slurm ran this script from: ${SCRIPT_DIR}"

# shellcheck source=_common.sh
source "${COMMON_SH}"
disordernet_slurm_setup

echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "=== allocation actually granted ==="
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}MB  node=$(hostname)"

echo "=== torch / cuda ==="
python "${PROJECT_DIR}/rockfish/slurm/_smoke_checks.py"

echo "SMOKE_OK"
