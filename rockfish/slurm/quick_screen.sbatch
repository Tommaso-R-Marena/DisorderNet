#!/bin/bash -l
#SBATCH --job-name=dn-screen
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dn-screen_%j.out
#SBATCH --error=logs/dn-screen_%j.err

# Quick paradigm screen (~2–3 h on A100).
#
# Submit (edit #SBATCH --account or export DISORDERNET_ACCOUNT):
#   mkdir -p logs
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   sbatch --account=$DISORDERNET_ACCOUNT \
#     --export=ALL,DISORDERNET_ACCOUNT,SCREEN_MODE=standard,BACKBONE=650M \
#     rockfish/slurm/quick_screen.sbatch
#
# 3B paradigm test:
#   sbatch --account=$DISORDERNET_ACCOUNT \
#     --export=ALL,DISORDERNET_ACCOUNT,SCREEN_MODE=paradigm,BACKBONE=3B,PROFILE=ultra3b \
#     rockfish/slurm/quick_screen.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${SCREEN_MODE:=standard}"
: "${BACKBONE:=650M}"
: "${PROFILE:=ultra}"
: "${STAGE:=screen}"

export STAGE PROFILE BACKBONE SCREEN_MODE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
