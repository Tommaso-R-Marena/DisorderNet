#!/bin/bash -l
#SBATCH --job-name=dn-pipeline-clean
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-pipeline-clean_%j.out
#SBATCH --error=logs/dn-pipeline-clean_%j.err

# Contamination-clean companion: ultra capacity WITHOUT structure-training signals
# (use_hallucination_weighting=False, use_plddt_features=False).
#
# Uses a distinct workdir / checkpoint subdir so it cannot clobber the main run.
#
# Submit:
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   export DISORDERNET_WORKDIR=$HOME/disordernet_runs/ultra_clean
#   sbatch --account=$DISORDERNET_ACCOUNT \
#     --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_WORKDIR \
#     rockfish/slurm/pipeline_ultra_clean.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${PROFILE:=ultra_clean}"
: "${BACKBONE:=650M}"
: "${STAGE:=pipeline}"
: "${DISORDERNET_WORKDIR:=$HOME/disordernet_runs/ultra_clean}"
: "${CHECKPOINT_SUBDIR:=checkpoints_ultra_clean}"

export STAGE PROFILE BACKBONE
export DISORDERNET_WORKDIR CHECKPOINT_SUBDIR
export PREFETCH_AF=1
export RUN_CAID3=1
# Defense in depth even if profile is overridden
export RUN_NO_HALLUC_WEIGHT=1
export RUN_NO_PLDDT_FEATURES=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
