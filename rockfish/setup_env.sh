#!/bin/bash
# One-time environment setup for JHU Rockfish (or similar Slurm + CUDA cluster).
#
# Usage:
#   cd /path/to/DisorderNet
#   bash rockfish/setup_env.sh
#   source ~/venvs/disordernet/bin/activate
#
# Then submit jobs from repo root:
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   sbatch rockfish/slurm/quick_screen.sbatch

set -euo pipefail

VENV_DIR="${DISORDERNET_VENV:-$HOME/venvs/disordernet}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== DisorderNet HPC setup ==="
echo "Repo: $REPO_ROOT"
echo "Venv: $VENV_DIR"

module purge 2>/dev/null || true

# Rockfish: load a CUDA stack if available (adjust versions via `module spider cuda`)
if command -v module &>/dev/null; then
  if module is-avail gcc/11.4.0 2>/dev/null; then
    module load gcc/11.4.0
  elif module is-avail gcc/9.3.0 2>/dev/null; then
    module load gcc/9.3.0
  fi
  if module is-avail cuda/11.8.0 2>/dev/null; then
    module load cuda/11.8.0
  elif module is-avail cuda/12.1.1 2>/dev/null; then
    module load cuda/12.1.1
  fi
fi

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools

# PyTorch with CUDA — cu118 is widely compatible on Rockfish A100 nodes
pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install -r "$REPO_ROOT/rockfish/requirements-hpc.txt"

python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Set your Slurm GPU account (PI allocation ending in _gpu):"
echo "  export DISORDERNET_ACCOUNT=your_pi_gpu"
echo ""
echo "Submit quick screen:"
echo "  cd $REPO_ROOT && sbatch rockfish/slurm/quick_screen.sbatch"
