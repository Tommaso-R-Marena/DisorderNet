#!/bin/bash -ue
#SBATCH --job-name="dn-mt"
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
# DisProt tasks only. disorder_pdb is appended by the trainer from the MobiDB
# missing-residue source, which is CAID3's own label definition. Listing it here
# too produced six tasks with a duplicate name, and the DisProt-derived version
# is degenerate: its rules mark only annotated disorder, so it arrives at 100%
# prevalence with no negatives at all.
: "${MT_TASKS:=disorder_nox,linker,binding,binding_idr}"

# Inputs live in one shared cache, not in the workdir. Defaulting them to the
# workdir meant a run into a fresh directory died on a FileNotFoundError after
# the queue wait, for want of a file that was already on disk elsewhere.
: "${MT_DISPROT:=${HOME}/.cache/disordernet/disprot_raw.json}"
: "${MT_PDB_MISSING:=${HOME}/.cache/disordernet/mobidb_pdbcov.ndjson}"
: "${MT_CAID_REF:=/scratch4/sfried3/jbeale3_disordernet/caid3_official/disorder_pdb.fasta}"
# Additional benchmark references to exclude, space separated. Filtering CAID3
# alone leaves 307 of CAID2's 348 targets in training — the rounds share one
# protein — so a model meant to be evaluated on CAID2 must name it here.
: "${MT_CAID_REF_EXTRA:=}"
export STAGE MT_WORKDIR MT_BACKBONE MT_FOLDS MT_EPOCHS MT_STRUCTURE_DIM MT_TASKS
export MT_DISPROT MT_PDB_MISSING MT_CAID_REF MT_CAID_REF_EXTRA

# ── Paths (override via env before sbatch) ───────────────────────────────────
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
# _common.sh runs the whole publish pipeline at the end of sourcing unless this
# is set. Without it, sourcing dispatches run_disordernet.py with STAGE — which
# is not one of its stages — and this script's own python is never reached.
export DISORDERNET_COMMON_AUTORUN=0
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

# Check every input before the GPU is touched. Each of these has cost a queue
# wait to discover, and the message from a missing file three frames into
# argparse names neither the file's purpose nor where it should have come from.
missing=0
for f in "${MT_DISPROT}" "${MT_PDB_MISSING}" "${MT_CAID_REF}"; do
  if [[ ! -s "${f}" ]]; then
    echo "ERROR: required input missing or empty: ${f}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "  Set MT_DISPROT / MT_PDB_MISSING / MT_CAID_REF to override." >&2
  exit 1
fi
echo "inputs: disprot=${MT_DISPROT}"
echo "        pdb-missing=${MT_PDB_MISSING}"
echo "        caid-reference=${MT_CAID_REF}"

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
python rockfish/train_multitask.py \
  --workdir "${MT_WORKDIR}" \
  --disprot "${MT_DISPROT}" \
  --pdb-missing-cache "${MT_PDB_MISSING}" \
  --caid-reference "${MT_CAID_REF}" \
  ${MT_CAID_REF_EXTRA:+$(for r in ${MT_CAID_REF_EXTRA}; do printf -- '--caid-reference %s ' "$r"; done)} \
  --backbone "${MT_BACKBONE}" \
  --n-folds "${MT_FOLDS}" \
  --epochs "${MT_EPOCHS}" \
  --structure-dim "${MT_STRUCTURE_DIM}" \
  --tasks "${MT_TASKS}" \
  "${@}"
