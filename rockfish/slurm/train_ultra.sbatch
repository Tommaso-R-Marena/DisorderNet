#!/bin/bash -l
#SBATCH --job-name=dn-ultra
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-ultra_%j.out
#SBATCH --error=logs/dn-ultra_%j.err

# Full ultra pipeline on ESM-2 650M: CV → stack → postprocess (~18–24 h).
#
# Submit:
#   mkdir -p logs
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   sbatch --export=ALL,DISORDERNET_ACCOUNT rockfish/slurm/train_ultra.sbatch
#
# CV only (resume-friendly; re-submit after preemption):
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,STAGE=cv rockfish/slurm/train_ultra.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=full}"

export STAGE PROFILE BACKBONE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
