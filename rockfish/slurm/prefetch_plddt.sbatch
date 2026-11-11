#!/bin/bash -ue
#SBATCH --job-name="dn-plddt"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00-04:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-plddt_%j.out
#SBATCH --error=logs/dn-plddt_%j.err
#
# Populate the shared AlphaFold pLDDT cache on CPU, before any GPU job needs it.
#
# The structure channel (use_plddt_features), the hallucination weighting, and
# structure_distrust_benchmark.json — a required strict-package artifact — all
# depend on this cache. Fetching it inside the GPU job means thousands of HTTP
# round-trips billed as A100 time, repeated for every run in the bundle.
#
# Point runs at the result with:
#   export DISORDERNET_PLDDT_CACHE=$HOME/.cache/disordernet/af_plddt

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/dn_rigor}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet_rigor}"

module purge 2>/dev/null || true
module load gcc/11.4.0 python/3.11.6 2>/dev/null || true
source "${ENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export DISORDERNET_PLDDT_CACHE="${DISORDERNET_PLDDT_CACHE:-${HOME}/.cache/disordernet/af_plddt}"
export DISORDERNET_DISPROT_CACHE="${DISORDERNET_DISPROT_CACHE:-${HOME}/.cache/disordernet/disprot_raw.json}"
mkdir -p "${DISORDERNET_PLDDT_CACHE}"

python rockfish/slurm/_prefetch_plddt.py
