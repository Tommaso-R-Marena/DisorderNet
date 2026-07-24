#!/bin/bash -ue
#SBATCH --job-name="dn-phase"
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --time=03-00:00:00
#SBATCH --account=sfried3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --export=ALL
#SBATCH --output=logs/dn-phase_%j.out
#SBATCH --error=logs/dn-phase_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Generic GPU pipeline phase. Configure via env (PROFILE, BACKBONE, STAGE,
# DISORDERNET_WORKDIR, CHECKPOINT_SUBDIR, RUN_CAID3, clean flags, …).
# Prefer: bash rockfish/slurm/submit_publish_650m.sh | submit_publish_3b.sh
#
# NOTE: --mem=180G (not 192G) leaves headroom on 192GB a100 nodes; requesting the
# full node RAM often fails at start with Run time 00:00:00. CLI --account/--qos
# from publish_submit override the defaults above when needed.

: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=pipeline}"
export STAGE PROFILE BACKBONE
export PREFETCH_AF="${PREFETCH_AF:-1}"
export RUN_CAID3="${RUN_CAID3:-1}"

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
