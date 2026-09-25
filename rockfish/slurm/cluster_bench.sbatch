#!/bin/bash -ue
#SBATCH --job-name="dn-cluster"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=00-01:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-cluster_%j.out
#SBATCH --error=logs/dn-cluster_%j.err
# Time homology clustering on the real DisProt release before committing GPU
# hours to it. CPU-only: no reason to hold an A100 to measure string alignment.

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/dn_rigor}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet_rigor}"

module purge 2>/dev/null || true
module load gcc/11.4.0 python/3.11.6 2>/dev/null || true
module load blast-plus/2.14.1 2>/dev/null || module load blast/2.13.0 2>/dev/null || true
source "${ENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
echo "blastp: $(command -v blastp || echo MISSING)"
echo "cpus:   ${SLURM_CPUS_PER_TASK}"

python rockfish/slurm/_cluster_bench.py
