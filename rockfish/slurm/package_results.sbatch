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

# Assemble organized publish package from a bundle root.
# Prefer: python rockfish/publish_submit.py package --kind 650m|3b --root-workdir ...
# This sbatch is used by the submit-* CLI with Slurm dependencies.

set -euo pipefail

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
: "${DISORDERNET_VENV:=$HOME/venvs/disordernet}"
: "${DISORDERNET_REPO:=$HOME/DisorderNet}"
: "${DISORDERNET_PUBLISH_ROOT:=$HOME/disordernet_runs/publish_bundle}"
: "${DISORDERNET_PACKAGE_DIR:=$DISORDERNET_PUBLISH_ROOT/publish_package}"
: "${BUNDLE_KIND:=650m}"

echo "Packaging publish results ($BUNDLE_KIND) from $DISORDERNET_PUBLISH_ROOT → $DISORDERNET_PACKAGE_DIR"
date

# shellcheck disable=SC1091
source "$DISORDERNET_VENV/bin/activate"
cd "$DISORDERNET_REPO"

PKG_ARGS=(
  package
  --root-workdir "$DISORDERNET_PUBLISH_ROOT"
  --kind "$BUNDLE_KIND"
  --package-dir "$DISORDERNET_PACKAGE_DIR"
)
if [[ "${INCLUDE_CLEAN:-1}" != "1" ]]; then
  PKG_ARGS+=(--no-clean)
fi
if [[ -n "${PACKAGE_ID:-}" ]]; then
  PKG_ARGS+=(--package-id "$PACKAGE_ID")
fi

python rockfish/publish_submit.py "${PKG_ARGS[@]}"

echo "Done. Open $DISORDERNET_PACKAGE_DIR/PACKAGE_README.md"
date
