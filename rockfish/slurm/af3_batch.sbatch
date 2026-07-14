#!/bin/bash -l
#SBATCH --job-name=dn-af3
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dn-af3_%j.out
#SBATCH --error=logs/dn-af3_%j.err

# Run / ingest AlphaFold 3 on Rockfish (MSA-free by default).
#
# One-time setup:
#   export DISORDERNET_AF3_ROOT=$HOME/af3
#   mkdir -p $DISORDERNET_AF3_ROOT
#   # Place licensed af3.bin from DeepMind into $DISORDERNET_AF3_ROOT/af3.bin
#   git clone --depth 1 https://github.com/google-deepmind/alphafold3.git ~/software/alphafold3
#
# Ingest existing outputs only:
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=ingest \
#     rockfish/slurm/af3_batch.sbatch
#
# Run missing proteins (MSA-free ≈ no 630 GB DBs):
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,DISORDERNET_AF3_ROOT,AF3_MODE=run \
#     rockfish/slurm/af3_batch.sbatch

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${AF3_MODE:=run}"
: "${STAGE:=af3}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"

export STAGE PROFILE BACKBONE
export AF3_MODE
export DISORDERNET_AF3_ROOT="${DISORDERNET_AF3_ROOT:-$HOME/af3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
