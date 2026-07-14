#!/bin/bash -l
#SBATCH --job-name=dn-pipeline
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-pipeline_%j.out
#SBATCH --error=logs/dn-pipeline_%j.err

# Complete production run: CV → stack → postprocess → eval → CAID3 (~24–48 h ultra).
#
# Submit:
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   sbatch --account=$DISORDERNET_ACCOUNT rockfish/slurm/pipeline_ultra.sbatch
#
# ultra3b:
#   sbatch --account=$DISORDERNET_ACCOUNT \
#     --export=ALL,DISORDERNET_ACCOUNT,PROFILE=ultra3b,BACKBONE=3B \
#     rockfish/slurm/pipeline_ultra.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=pipeline}"

export STAGE PROFILE BACKBONE
export PREFETCH_AF=1
export RUN_CAID3=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
