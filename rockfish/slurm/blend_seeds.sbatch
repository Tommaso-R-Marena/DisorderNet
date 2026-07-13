#!/bin/bash -l
#SBATCH --job-name=dn-blend
#SBATCH --partition=shared
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dn-blend_%j.out
#SBATCH --error=logs/dn-blend_%j.err

# Blend multi-seed OOF predictions (CPU-only, fast).
#
# Prerequisite: SEED_DIRS env var with comma-separated checkpoint directories.
#   export SEED_DIRS="path/checkpoints_s42,path/checkpoints_s43,path/checkpoints_s44"

set -euo pipefail

: "${STAGE:=multi-seed-blend}"
: "${SEED_DIRS:?Set SEED_DIRS to comma-separated checkpoint directories}"

export STAGE SEED_DIRS

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"
