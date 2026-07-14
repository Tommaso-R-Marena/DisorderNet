#!/bin/bash -ue
#SBATCH --job-name="dn-screen"
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00-04:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-screen_%j.out
#SBATCH --error=logs/dn-screen_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Quick paradigm screen (~2–3 h on A100-40GB; often faster on 80GB).
# SCREEN_MODE=standard|paradigm → train_profile=screen_plus (mini-ultra).
# SCREEN_MODE=flash → train_profile=screen (toy; not alone for go/no-go).
#
#   sbatch --account="${DISORDERNET_ACCOUNT}" rockfish/slurm/quick_screen.sbatch

: "${PROFILE:=screen_plus}"
: "${BACKBONE:=650M}"
: "${STAGE:=screen}"
: "${SCREEN_MODE:=standard}"
export STAGE PROFILE BACKBONE SCREEN_MODE

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
