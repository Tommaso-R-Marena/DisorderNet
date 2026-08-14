#!/bin/bash -ue
#SBATCH --job-name="dn-mt"
#SBATCH --partition=ica100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48000M
#SBATCH --time=01-00:00:00
#SBATCH --account=sfried3_gpu
#SBATCH --export=ALL
#SBATCH --output=logs/dn-mt_%j.out
#SBATCH --error=logs/dn-mt_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Train the multi-task head across all five CAID3 challenges.
#
# This exists because the equivalent run was launched from an ad-hoc `sbatch
# <<EOF` heredoc. That job could not be reproduced (Slurm keeps no Command for
# it), could not be inspected, and — because it never sourced _common.sh — ran
# with stdout block-buffered. Nine hours in, its log still showed hour four:
# monitoring was impossible, and a hard kill would have taken the rest of the
# buffer with it. Sourcing _common.sh sets PYTHONUNBUFFERED=1 along with the
# allocator, thread and cache settings the other jobs already rely on.
#
#   export MT_WORKDIR=/scratch4/sfried3/jbeale3_disordernet/multitask_full
#   sbatch --account="${DISORDERNET_ACCOUNT}" rockfish/slurm/train_multitask.sbatch
#
# Memory: MaxMemPerCPU is 4000M on ica100, so --mem must stay at or below
# cpus-per-task x 4000M or the job is rejected at submit time, not at run time.

: "${STAGE:=train-multitask}"
: "${MT_WORKDIR:?Set MT_WORKDIR to the output directory (use /scratch4, not \$HOME)}"
: "${MT_BACKBONE:=650M}"
: "${MT_FOLDS:=5}"
: "${MT_EPOCHS:=8}"
: "${MT_STRUCTURE_DIM:=24}"
: "${MT_TASKS:=disorder_nox,linker,binding,binding_idr,disorder_pdb}"
export STAGE MT_WORKDIR MT_BACKBONE MT_FOLDS MT_EPOCHS MT_STRUCTURE_DIM MT_TASKS

# ── Paths (override via env before sbatch) ───────────────────────────────────
PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${MT_WORKDIR}"

# Slurm copies the batch script into a per-job spool dir, so ${BASH_SOURCE[0]}
# does not resolve to the repo checkout at runtime. Resolve against the repo
# first; fall back to the script dir for local invocation.
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

disordernet_slurm_setup

# The 50GB home quota is invisible to df and quota, and a job that crosses it
# fails on its next write — including the write of its own log, which is how
# three earlier runs disappeared without leaving a diagnosis. Everything heavy
# belongs on /scratch4.
case "${MT_WORKDIR}" in
  "${HOME}"/*) echo "ERROR: MT_WORKDIR is under \$HOME (${MT_WORKDIR})." >&2
               echo "  Checkpoints there will hit the 50GB quota. Use /scratch4." >&2
               exit 1 ;;
esac

cd "${PROJECT_DIR}"
python rockfish/train_multitask.py \
  --workdir "${MT_WORKDIR}" \
  --backbone "${MT_BACKBONE}" \
  --n-folds "${MT_FOLDS}" \
  --epochs "${MT_EPOCHS}" \
  --structure-dim "${MT_STRUCTURE_DIM}" \
  --tasks "${MT_TASKS}" \
  "${@}"
