#!/bin/bash -ue
#SBATCH --job-name="dn-rsa-cache"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48000M
#SBATCH --time=00-08:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-rsa-cache_%j.out
#SBATCH --error=logs/dn-rsa-cache_%j.err
# Precompute Shrake-Rupley accessibilities for every cached AlphaFold structure.
#
# attach_structure recomputed these on every run: 5h04m of the private-trunk
# job's 18h, for a quantity that depends on nothing that changed between runs.
# Filling the cache once turns that into a disk read.
#
#   sbatch rockfish/slurm/build_rsa_cache.sbatch
#
# CPU partition on purpose: this never touches the GPU, and running it inside a
# training job holds an A100 idle for five hours.

: "${STAGE:=build-rsa-cache}"
: "${DISORDERNET_REPO:?Set DISORDERNET_REPO to the checkout to run from}"
: "${RSA_STRUCTURES:=/scratch4/sfried3/jbeale3_disordernet/af_structures}"
export STAGE RSA_STRUCTURES

PROJECT_DIR="${DISORDERNET_REPO}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# _common.sh dispatches the publish pipeline at the end of sourcing unless this
# is set, which would run run_disordernet.py instead of the script named below.
export DISORDERNET_COMMON_AUTORUN=0
# shellcheck source=_common.sh
source "${PROJECT_DIR}/rockfish/slurm/_common.sh"

disordernet_slurm_setup

cd "${PROJECT_DIR}"

# The cache is only worth having if it returns exactly what it replaces, and
# the off-cluster suite cannot check that: Biopython is not installed there, so
# every real-structure test skips. Here it can run, so it runs first — building
# 21,879 cache entries and *then* discovering the round-trip is lossy would
# poison every run that reads them.
echo "=== verifying the cache against real structures before filling it ==="
DISORDERNET_AF_STRUCTURES="${RSA_STRUCTURES}" \
  python -m pytest tests/test_rsa_feature_cache.py -q

echo
echo "=== filling ==="
python rockfish/build_rsa_cache.py \
  --structures "${RSA_STRUCTURES}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  "${@}"
