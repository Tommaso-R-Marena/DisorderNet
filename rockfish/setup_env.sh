#!/bin/bash
# One-time environment setup for JHU Rockfish (or similar Slurm + CUDA cluster).
#
# Usage:
#   cd /path/to/DisorderNet
#   bash rockfish/setup_env.sh
#   source ~/venvs/disordernet/bin/activate
#
# Then submit jobs from repo root:
#   export DISORDERNET_ACCOUNT=sfried3
#   sbatch rockfish/slurm/quick_screen.sbatch

set -euo pipefail

VENV_DIR="${DISORDERNET_VENV:-$HOME/venvs/disordernet}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== DisorderNet HPC setup ==="
echo "Repo: $REPO_ROOT"
echo "Venv: $VENV_DIR"

module purge 2>/dev/null || true

# IMPORTANT: the Rockfish login node's default `python3` is ancient (e.g. 3.6, pip 9),
# which cannot install modern torch ("No matching distribution found for torch").
# Load a modern Python (Anaconda) module first so the venv uses Python 3.10+.
PYTHON_BIN="python3"
if command -v module &>/dev/null; then
  for m in anaconda3/2024.02-1 anaconda3/2023.09 anaconda3 python/3.11.6 python/3.10.7; do
    if module is-avail "$m" 2>/dev/null; then
      module load "$m" && break
    fi
  done
  # CUDA / GCC toolchain (torch wheels bundle their own CUDA, so exact version is lenient)
  if module is-avail gcc/11.4.0 2>/dev/null; then module load gcc/11.4.0
  elif module is-avail gcc/9.3.0 2>/dev/null; then module load gcc/9.3.0; fi
  if module is-avail cuda/12.1.1 2>/dev/null; then module load cuda/12.1.1
  elif module is-avail cuda/11.8.0 2>/dev/null; then module load cuda/11.8.0; fi
fi
command -v python3 >/dev/null 2>&1 && PYTHON_BIN="python3"

PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "?")"
echo "Using Python ${PY_VER} at $(command -v "$PYTHON_BIN")"
case "${PY_VER}" in
  3.9|3.10|3.11|3.12|3.13) : ;;
  *) echo "ERROR: Python ${PY_VER} is too old for modern torch. Run 'module avail anaconda3'"
     echo "       and load a Python 3.10+ module (e.g. 'module load anaconda3/2024.02-1'), then re-run."
     exit 1 ;;
esac

# Recreate the venv cleanly (in case a previous run made a stale/old-Python venv)
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools

# PyTorch with CUDA — cu118 is widely compatible on Rockfish A100 nodes.
# Fall back to the default index if the pinned CUDA wheel is unavailable.
pip install torch --index-url https://download.pytorch.org/whl/cu118 || pip install torch

pip install -r "$REPO_ROOT/rockfish/requirements-hpc.txt"

python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Set your Slurm GPU account (PI allocation ending in _gpu):"
echo "  export DISORDERNET_ACCOUNT=sfried3"
echo ""
echo "Submit quick screen:"
echo "  cd $REPO_ROOT && sbatch rockfish/slurm/quick_screen.sbatch"
