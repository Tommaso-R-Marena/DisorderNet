#!/bin/bash -ue
#SBATCH --job-name="dn-temporal-eval"
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icgpu04
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=00-04:00:00
#SBATCH --account=sfried3_gpu
#SBATCH --export=ALL
#SBATCH --output=logs/dn-temporal_%j.out
#SBATCH --error=logs/dn-temporal_%j.err
# Score checkpoints on PDB entries released after the training data was built.
#
#   export TEMPORAL_REF=/scratch4/.../temporal/disorder_pdb.fasta
#   export TEMPORAL_CKPTS="/scratch4/.../multitask_pbias /scratch4/.../multitask_windowed"
#   sbatch rockfish/slurm/eval_temporal.sbatch

: "${STAGE:=eval-temporal}"
: "${DISORDERNET_REPO:?Set DISORDERNET_REPO to the checkout to run from}"
: "${TEMPORAL_REF:?Set TEMPORAL_REF to the temporal reference fasta}"
: "${TEMPORAL_CKPTS:?Set TEMPORAL_CKPTS to a space-separated list of run dirs}"
: "${TEMPORAL_RAW:=}"
: "${TEMPORAL_OUT:=}"
export STAGE

PROJECT_DIR="${DISORDERNET_REPO}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
export DISORDERNET_REPO="${PROJECT_DIR}" DISORDERNET_VENV="${ENV_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
export DISORDERNET_COMMON_AUTORUN=0
# shellcheck source=_common.sh
source "${PROJECT_DIR}/rockfish/slurm/_common.sh"
disordernet_slurm_setup

if ! python -c "
import sys, torch
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    sys.exit(1)
print('GPU ok:', torch.cuda.get_device_name(0))
"; then
  echo "ERROR: no usable CUDA device on $(hostname); resubmit excluding it" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
CKPT_ARGS=""
for c in ${TEMPORAL_CKPTS}; do CKPT_ARGS="${CKPT_ARGS} --checkpoint ${c}"; done
# shellcheck disable=SC2086
python rockfish/eval_temporal.py \
  --reference "${TEMPORAL_REF}" \
  ${TEMPORAL_RAW:+--raw-records "${TEMPORAL_RAW}"} \
  ${TEMPORAL_OUT:+--out "${TEMPORAL_OUT}"} \
  ${CKPT_ARGS} "${@}"
