#!/bin/bash -l
#SBATCH --job-name=dn-seed
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-seed_%A_%a.out
#SBATCH --error=logs/dn-seed_%A_%a.err

# Multi-seed CV array (seeds 42, 43, 44) for +0.005–0.015 AUC blend.
#
# Submit:
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_multiseed
#   sbatch --account=$DISORDERNET_ACCOUNT \
#     --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR,PROFILE=ultra,BACKBONE=650M \
#     rockfish/slurm/multi_seed.sbatch
#
# After all array tasks finish, blend:
#   export SEED_DIRS="$DISORDERNET_WORKDIR/checkpoints_s42,$DISORDERNET_WORKDIR/checkpoints_s43,..."
#   sbatch --export=ALL,STAGE=multi-seed-blend,SEED_DIRS rockfish/slurm/blend_seeds.sbatch

set -euo pipefail

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]:-42}

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=full}"
: "${DISORDERNET_WORKDIR:=$HOME/disordernet_runs/multiseed}"

export STAGE PROFILE BACKBONE SEED
export DISORDERNET_WORKDIR="${DISORDERNET_WORKDIR}/seed_${SEED}"
export CHECKPOINT_SUBDIR="checkpoints_s${SEED}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
