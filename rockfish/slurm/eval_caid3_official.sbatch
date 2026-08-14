#!/bin/bash -ue
#SBATCH --job-name="dn-official"
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=00-04:00:00
#SBATCH --account=sfried3_gpu
#SBATCH --export=ALL
#SBATCH --output=logs/dn-official_%j.out
#SBATCH --error=logs/dn-official_%j.err
# Score a checkpoint on the five official CAID3 references, paired against all
# 117 published entrants.
#
#   export EVAL_CKPT=/scratch4/sfried3/jbeale3_disordernet/multitask_bio
#   sbatch rockfish/slurm/eval_caid3_official.sbatch

: "${STAGE:=eval-caid3-official}"
: "${EVAL_CKPT:?Set EVAL_CKPT to a directory holding multitask_head.pt}"
: "${EVAL_REFS:=/scratch4/sfried3/jbeale3_disordernet/caid3_official}"
: "${EVAL_PREDS:=/scratch4/sfried3/jbeale3_disordernet/caid3_predictions}"
: "${EVAL_STRUCTURES:=/scratch4/sfried3/jbeale3_disordernet/af_structures}"
: "${EVAL_NBOOT:=2000}"
export STAGE EVAL_CKPT EVAL_REFS EVAL_PREDS EVAL_STRUCTURES EVAL_NBOOT

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_DIR}/rockfish/slurm/_common.sh" ]]; then
  COMMON_SH="${PROJECT_DIR}/rockfish/slurm/_common.sh"
elif [[ -f "${SCRIPT_DIR}/_common.sh" ]]; then
  COMMON_SH="${SCRIPT_DIR}/_common.sh"
else
  echo "ERROR: cannot locate rockfish/slurm/_common.sh" >&2
  exit 1
fi
# shellcheck source=_common.sh
source "${COMMON_SH}"

disordernet_slurm_setup

DISPROT="${EVAL_DISPROT:-${EVAL_CKPT}/disprot_raw.json}"

cd "${PROJECT_DIR}"
python rockfish/eval_caid3_official.py \
  --checkpoint "${EVAL_CKPT}" \
  --refs "${EVAL_REFS}" \
  --predictions "${EVAL_PREDS}" \
  --structure-cache "${EVAL_STRUCTURES}" \
  --disprot "${DISPROT}" \
  --n-boot "${EVAL_NBOOT}" \
  "${@}"
