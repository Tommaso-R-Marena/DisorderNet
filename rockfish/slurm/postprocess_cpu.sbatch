#!/bin/bash -l
#SBATCH --job-name=dn-stack
#SBATCH --partition=shared
#SBATCH --account=CHANGE_ME
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dn-stack_%j.out
#SBATCH --error=logs/dn-stack_%j.err

# Post-CV ensemble + postprocess on CPU (no GPU needed).
# Run after train_ultra.sbatch if you split stages for fault tolerance.
#
# Prerequisite: checkpoints/cv_progress.json from a completed CV job.
# Copy checkpoints into $DISORDERNET_REPO/checkpoints/ or set DISORDERNET_WORKDIR.
#
# Submit:
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,STAGE=stack rockfish/slurm/postprocess_cpu.sbatch
#   sbatch --export=ALL,DISORDERNET_ACCOUNT,STAGE=postprocess rockfish/slurm/postprocess_cpu.sbatch

set -euo pipefail

: "${STAGE:=stack}"
: "${PROFILE:=ultra}"
: "${BACKBONE:=650M}"

export STAGE PROFILE BACKBONE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module purge
# shellcheck disable=SC1091
source "${DISORDERNET_VENV:-$HOME/venvs/disordernet}/bin/activate"
cd "${DISORDERNET_REPO:-$HOME/DisorderNet}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# stack/postprocess need ESM loaded for soup — use a GPU node if stack fails on CPU-only
# For stack-only (v6 OOF, no ESM forward): could run on shared, but postprocess needs GPU.
echo "WARNING: postprocess stage needs GPU for fold soup. Use train_*.sbatch for full pipeline."
echo "This job is intended for STAGE=stack (v6 ensemble) only."

python rockfish/run_disordernet.py "$STAGE" \
  --profile "$PROFILE" \
  --backbone "$BACKBONE" \
  --seed "${SEED:-42}" \
  --num-workers 8
