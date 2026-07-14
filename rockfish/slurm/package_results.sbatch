#!/bin/bash -l
#SBATCH --job-name=dn-package
#SBATCH --partition=shared
#SBATCH --qos=normal
#SBATCH --account=CHANGE_ME_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dn-package_%j.out
#SBATCH --error=logs/dn-package_%j.err

# Assemble organized publish package from ultra_650M / clean / ultra3b workdirs.
# Usually submitted by rockfish/slurm/submit_publish_all.sh with dependencies.
#
# If shared/CPU account differs from GPU account, submit with:
#   --account=$DISORDERNET_CPU_ACCOUNT


set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${DISORDERNET_VENV:=$HOME/venvs/disordernet}"
: "${DISORDERNET_REPO:=$HOME/DisorderNet}"
: "${DISORDERNET_PUBLISH_ROOT:=$HOME/disordernet_runs/publish_bundle}"
: "${DISORDERNET_PACKAGE_DIR:=$DISORDERNET_PUBLISH_ROOT/publish_package}"

echo "Packaging publish results from $DISORDERNET_PUBLISH_ROOT → $DISORDERNET_PACKAGE_DIR"
date

# shellcheck disable=SC1091
source "$DISORDERNET_VENV/bin/activate"
cd "$DISORDERNET_REPO"

PKG_ARGS=(--root-workdir "$DISORDERNET_PUBLISH_ROOT" --package-dir "$DISORDERNET_PACKAGE_DIR")
if [[ "${INCLUDE_CLEAN:-1}" != "1" ]]; then
  PKG_ARGS+=(--no-clean)
fi
if [[ "${INCLUDE_3B:-1}" != "1" ]]; then
  PKG_ARGS+=(--no-3b)
fi
if [[ -n "${PACKAGE_ID:-}" ]]; then
  PKG_ARGS+=(--package-id "$PACKAGE_ID")
fi

python rockfish/package_publish_results.py "${PKG_ARGS[@]}"

echo "Done. Open $DISORDERNET_PACKAGE_DIR/PACKAGE_README.md"
date
