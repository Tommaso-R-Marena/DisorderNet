#!/bin/bash -ue
#SBATCH --job-name="dn-official"
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
# icgpu04 reports idle and accepts jobs, then fails them in ~12s with
# "CUDA initialization: CUDA unknown error" and zero visible devices.
# It has done this twice in this project, most recently killing both
# re-evaluation jobs at once because Slurm placed them there together.
# Override on the command line if it is ever repaired.
#SBATCH --exclude=icgpu04
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

# No default. DISORDERNET_REPO silently defaulting to ~/DisorderNet sent this
# job into a different checkout, whose older _common.sh dispatches the pipeline
# at source time — so it ran run_disordernet.py with a stage that does not exist
# instead of the script named below, and created directories in a repo it was
# never meant to touch. A wrong checkout is not a condition to guess at.
: "${DISORDERNET_REPO:?Set DISORDERNET_REPO to the checkout to run from}"
PROJECT_DIR="${DISORDERNET_REPO}"
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
# _common.sh runs the whole publish pipeline at the end of sourcing unless this
# is set. Without it, sourcing dispatches run_disordernet.py with STAGE — which
# is not one of its stages — and this script's own python is never reached.
export DISORDERNET_COMMON_AUTORUN=0
# shellcheck source=_common.sh
source "${COMMON_SH}"

disordernet_slurm_setup

DISPROT="${EVAL_DISPROT:-${EVAL_CKPT}/disprot_raw.json}"

# Confirm the GPU is actually usable before anything else runs. A node can be
# allocated, report idle, and still initialise CUDA with zero visible devices —
# icgpu04 does exactly that. Failing here names the node and the cause in one
# line, instead of a torch traceback forty lines into the job.
if ! python -c "
import sys, torch
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    sys.exit(1)
print('GPU ok:', torch.cuda.get_device_name(0))
"; then
  echo "ERROR: no usable CUDA device on $(hostname). The node was allocated" >&2
  echo "  but CUDA reports zero devices. Resubmit excluding it:" >&2
  echo "    sbatch --exclude=$(hostname -s) ..." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
python rockfish/eval_caid3_official.py \
  --checkpoint "${EVAL_CKPT}" \
  --refs "${EVAL_REFS}" \
  --predictions "${EVAL_PREDS}" \
  --structure-cache "${EVAL_STRUCTURES}" \
  --disprot "${DISPROT}" \
  --n-boot "${EVAL_NBOOT}" \
  "${@}"
