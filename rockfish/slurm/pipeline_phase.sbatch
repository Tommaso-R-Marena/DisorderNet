#!/bin/bash -l
#SBATCH --job-name=dn-phase
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-phase_%j.out
#SBATCH --error=logs/dn-phase_%j.err

# Generic GPU pipeline phase. Configure via env before sbatch:
#   PROFILE, BACKBONE, STAGE, DISORDERNET_WORKDIR, CHECKPOINT_SUBDIR,
#   RUN_CAID3, RUN_NO_HALLUC_WEIGHT, RUN_NO_PLDDT_FEATURES, …
#
# Prefer rockfish/slurm/submit_publish_all.sh over submitting this directly.

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"
: "${STAGE:=pipeline}"

export STAGE PROFILE BACKBONE
export PREFETCH_AF="${PREFETCH_AF:-1}"
export RUN_CAID3="${RUN_CAID3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
