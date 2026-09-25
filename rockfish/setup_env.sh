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
#
# Environment overrides:
#   DISORDERNET_VENV       venv location      (default ~/venvs/disordernet)
#   DISORDERNET_TORCH_CUDA torch wheel index  (default cu121)
#   DISORDERNET_MIN_PY     minimum Python     (default 3.11, matching CI)

set -euo pipefail

VENV_DIR="${DISORDERNET_VENV:-$HOME/venvs/disordernet}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_CUDA="${DISORDERNET_TORCH_CUDA:-cu121}"
MIN_PY="${DISORDERNET_MIN_PY:-3.11}"

echo "=== DisorderNet HPC setup ==="
echo "Repo: $REPO_ROOT"
echo "Venv: $VENV_DIR"

module purge 2>/dev/null || true

# Rockfish's bare `python3` is 3.6.8 and cannot install modern torch. Modern
# Pythons live under the gcc hierarchy, so the compiler module must be loaded
# first — `module is-avail python/3.11.6` returns false until it is, which is
# why the previous probe loop fell through to a bare `anaconda3` load and
# silently produced a Python 3.9.12 venv.
PY_LOADED=""
if command -v module &>/dev/null; then
  for gcc_mod in gcc/11.4.0 gcc/9.3.0; do
    module purge 2>/dev/null || true
    module load "$gcc_mod" 2>/dev/null || continue
    for py_mod in python/3.12.2 python/3.11.9 python/3.11.8 python/3.11.6; do
      if module load "$py_mod" 2>/dev/null; then
        PY_LOADED="$gcc_mod + $py_mod"
        break 2
      fi
    done
  done
  # Last resort: a versioned anaconda new enough to clear MIN_PY. Bare
  # `anaconda3` resolves to the site default (Python 3.9.12) — never use it.
  if [[ -z "$PY_LOADED" ]]; then
    module purge 2>/dev/null || true
    for m in anaconda3/2024.02-1 anaconda3/2023.09-0; do
      if module load "$m" 2>/dev/null; then PY_LOADED="$m"; break; fi
    done
  fi
  # CUDA/GCC toolchain. torch wheels bundle their own CUDA runtime, so this only
  # matters for anything that compiles against the system toolchain.
  module load cuda/12.1.0 2>/dev/null || module load cuda/11.8.0 2>/dev/null || true
fi

PYTHON_BIN="$(command -v python3)"
PY_VER="$("$PYTHON_BIN" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
echo "Python module: ${PY_LOADED:-<none, using system python3>}"
echo "Using Python ${PY_VER} at ${PYTHON_BIN}"

# Fail loudly rather than building an env the test suite never runs against.
if ! "$PYTHON_BIN" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= tuple(int(x) for x in '${MIN_PY}'.split('.')) else 1)"; then
  echo "ERROR: Python ${PY_VER} is below the required ${MIN_PY}." >&2
  echo "       CI runs 3.11/3.12; older interpreters are not supported." >&2
  echo "       Try:  module load gcc/11.4.0 python/3.11.6" >&2
  echo "       Then: bash rockfish/setup_env.sh" >&2
  exit 1
fi

# Large wheels (torch is ~2.5GB unpacked) must not unpack into a small tmpfs —
# a truncated shared library installs "successfully" and then dies at import
# with SIGBUS. Stage in a disk-backed directory and clean up afterwards.
BUILD_TMP="$(mktemp -d "${TMPDIR:-/tmp}/dn_pip_XXXXXX")"
trap 'rm -rf "$BUILD_TMP"' EXIT
export TMPDIR="$BUILD_TMP"

rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --no-cache-dir --upgrade pip wheel setuptools

echo "Installing torch (${TORCH_CUDA}) …"
pip install --no-cache-dir torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
  || { echo "WARN: ${TORCH_CUDA} wheel unavailable; falling back to default index" >&2; \
       pip install --no-cache-dir torch; }

pip install --no-cache-dir -r "$REPO_ROOT/rockfish/requirements-hpc.txt"

# Import every native extension together. Truncated wheels and duplicated
# OpenMP runtimes only surface when the libraries are loaded in one process.
python - <<'PYEOF'
import sys
mods = ["numpy", "scipy", "sklearn", "torch", "lightgbm", "xgboost", "esm", "pandas", "matplotlib"]
import importlib
for m in mods:
    importlib.import_module(m)
import torch
print(f"Python {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__}  cuda_built={torch.version.cuda}  cuda_available={torch.cuda.is_available()}")
print("All native imports OK")
PYEOF

echo ""
echo "NOTE: cuda_available is expected to be False on a login node (no GPU)."
echo "      It is verified for real by the GPU smoke test."
echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Slurm accounts (CPU vs GPU differ on Rockfish):"
echo "  export DISORDERNET_ACCOUNT=sfried3"
echo "  export DISORDERNET_GPU_ACCOUNT=sfried3_gpu"
echo "  export DISORDERNET_GPU_QOS=qos_gpu"
echo ""
echo "Submit quick screen:"
echo "  cd $REPO_ROOT && sbatch rockfish/slurm/quick_screen.sbatch"
