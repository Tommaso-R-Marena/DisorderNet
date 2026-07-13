#!/bin/bash -l
#SBATCH --job-name=dn-boltz
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-boltz_%j.out
#SBATCH --error=logs/dn-boltz_%j.err

# Run / ingest Boltz-2 on Rockfish (pinned boltz==2.2.1, weights auto-download).
#
# Setup (once):
#   export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
#   mkdir -p $DISORDERNET_BOLTZ_ROOT
#
# Auto (install + run pending + ingest):
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=auto \
#     rockfish/slurm/boltz_batch.sbatch
#
# Ingest only:
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=ingest \
#     rockfish/slurm/boltz_batch.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${BOLTZ_MODE:=auto}"
: "${STAGE:=boltz}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STRUCTURE_BACKEND:=boltz}"

export STAGE PROFILE BACKBONE STRUCTURE_BACKEND
export BOLTZ_MODE
export DISORDERNET_BOLTZ_ROOT="${DISORDERNET_BOLTZ_ROOT:-$HOME/boltz}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$DISORDERNET_BOLTZ_ROOT/cache}"

mkdir -p logs
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
