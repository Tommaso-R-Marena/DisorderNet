#!/bin/bash -l
#SBATCH --job-name=dn-ultra3b
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-ultra3b_%j.out
#SBATCH --error=logs/dn-ultra3b_%j.err

# Full ultra3b pipeline on ESM-2 3B (~30–40 h on A100 40GB).
#
# Requires: A100 40GB+ partition (a100 or ica100 for 80GB).
#
# Submit:
#   mkdir -p logs
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   sbatch --export=ALL,DISORDERNET_ACCOUNT rockfish/slurm/train_ultra3b.sbatch
#
# Use 80GB nodes if batch OOM on 40GB:
#   sbatch --partition=ica100 --export=ALL,DISORDERNET_ACCOUNT rockfish/slurm/train_ultra3b.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${PROFILE:=ultra3b}"
: "${BACKBONE:=3B}"
: "${STAGE:=full}"

export STAGE PROFILE BACKBONE
export PREFETCH_AF=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
