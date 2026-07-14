#!/bin/bash -l
#SBATCH --job-name=dn-af3-arr
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dn-af3_%A_%a.out
#SBATCH --error=logs/dn-af3_%A_%a.err

# Parallel AF3 generation via Slurm array (MSA-free).
# Each array task processes pending proteins where i % N == task_id.
#
# Setup once:
#   export DISORDERNET_AF3_ROOT=$HOME/af3
#   # place af3.bin, clone alphafold3
#
# Submit (8 parallel GPUs by default — adjust --array):
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   export AF3_SHARD_COUNT=8
#   sbatch --account=$DISORDERNET_ACCOUNT --array=0-7 \
#     --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=run,AF3_SHARD_COUNT=8 \
#     rockfish/slurm/af3_array.sbatch
#
# After array finishes, ingest into training:
#   sbatch --export=ALL,...,AF3_MODE=ingest rockfish/slurm/pipeline_ultra.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${AF3_MODE:=run}"
: "${AF3_SHARD_COUNT:=8}"
: "${STAGE:=af3}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"

export STAGE PROFILE BACKBONE AF3_MODE
export DISORDERNET_AF3_ROOT="${DISORDERNET_AF3_ROOT:-$HOME/af3}"
export AF3_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
export AF3_SHARD_COUNT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
