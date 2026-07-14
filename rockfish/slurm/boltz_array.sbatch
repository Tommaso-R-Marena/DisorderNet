#!/bin/bash -l
#SBATCH --job-name=dn-boltz-arr
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/dn-boltz-arr_%A_%a.out
#SBATCH --error=logs/dn-boltz-arr_%A_%a.err

# Parallel Boltz-2 over pending proteins (Slurm array).
#
#   export DISORDERNET_BOLTZ_ROOT=$HOME/boltz
#   export BOLTZ_SHARD_COUNT=8
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_BOLTZ_ROOT,BOLTZ_MODE=run,BOLTZ_SHARD_COUNT=8 \
#     rockfish/slurm/boltz_array.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${BOLTZ_MODE:=run}"
: "${STAGE:=boltz}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STRUCTURE_BACKEND:=boltz}"
: "${BOLTZ_SHARD_COUNT:=8}"

export STAGE PROFILE BACKBONE STRUCTURE_BACKEND BOLTZ_MODE
export DISORDERNET_BOLTZ_ROOT="${DISORDERNET_BOLTZ_ROOT:-$HOME/boltz}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$DISORDERNET_BOLTZ_ROOT/cache}"
export BOLTZ_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export BOLTZ_SHARD_COUNT

mkdir -p logs
# shellcheck disable=SC1091
source "$(dirname "$0")/_common.sh"
