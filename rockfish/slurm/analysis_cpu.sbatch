#!/bin/bash -ue
#SBATCH --job-name="dn-analysis"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00-06:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-analysis_%j.out
#SBATCH --error=logs/dn-analysis_%j.err
# Run a CPU-only analysis script from the repo on a compute node.
#
# Scoring 117 entrants and bootstrapping ten paired comparisons at 10,000
# resamples is real compute — minutes to an hour — and it was being run on a
# login node, where it is both slow and inconsiderate to everyone else sharing
# it. Login nodes are for editing, submitting and inspecting; anything that
# occupies a core for minutes belongs here.
#
#   export ANALYSIS_SCRIPT=results/caid3/ensemble_probe.py
#   export ANALYSIS_ENV="ENSEMBLE_OUT=/scratch4/.../ensemble_submissions"
#   sbatch rockfish/slurm/analysis_cpu.sbatch

: "${STAGE:=analysis}"
: "${ANALYSIS_SCRIPT:?Set ANALYSIS_SCRIPT to a repo-relative python file}"
: "${DISORDERNET_REPO:?Set DISORDERNET_REPO to the checkout to run from}"
: "${CAID3_OFFICIAL_DIR:=/scratch4/sfried3/jbeale3_disordernet/caid3_official}"
: "${CAID3_PREDICTIONS_DIR:=/scratch4/sfried3/jbeale3_disordernet/caid3_predictions}"
export STAGE CAID3_OFFICIAL_DIR CAID3_PREDICTIONS_DIR

PROJECT_DIR="${DISORDERNET_REPO}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export REPO="${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

if [[ ! -f "${PROJECT_DIR}/${ANALYSIS_SCRIPT}" ]]; then
  echo "ERROR: ${PROJECT_DIR}/${ANALYSIS_SCRIPT} does not exist" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${PROJECT_DIR}/rockfish/slurm/_common.sh" ]]; then
  COMMON_SH="${PROJECT_DIR}/rockfish/slurm/_common.sh"
else
  COMMON_SH="${SCRIPT_DIR}/_common.sh"
fi
# _common.sh dispatches the publish pipeline at the end of sourcing unless this
# is set, which would run run_disordernet.py instead of the script named above.
export DISORDERNET_COMMON_AUTORUN=0
# shellcheck source=_common.sh
source "${COMMON_SH}"

disordernet_slurm_setup

cd "${PROJECT_DIR}"
# ANALYSIS_ENV carries "KEY=VALUE KEY=VALUE" for the script's own settings.
# shellcheck disable=SC2086
env ${ANALYSIS_ENV:-} python "${ANALYSIS_SCRIPT}" "${@}"
