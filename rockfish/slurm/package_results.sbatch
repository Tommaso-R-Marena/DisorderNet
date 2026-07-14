#!/bin/bash -ue
#SBATCH --job-name="dn-package"
#SBATCH --partition=shared
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00-02:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-package_%j.out
#SBATCH --error=logs/dn-package_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
# Assemble organized publish package from a bundle root.
# Prefer CLI: python rockfish/publish_submit.py package --kind 650m|3b --strict
# Used by submit_publish_650m.sh / submit_publish_3b.sh with dependencies.
#
# Env:
#   DISORDERNET_PUBLISH_ROOT  — bundle parent (required for kind-aware packaging)
#   DISORDERNET_PACKAGE_DIR   — output dir (default: $PUBLISH_ROOT/publish_package)
#   BUNDLE_KIND               — 650m | 3b (default: 650m)
#   PACKAGE_STRICT            — 1 (default) fail if required artifacts missing
#   INCLUDE_CLEAN             — 1 (default) include clean companion in package
#   PACKAGE_ID                — optional stable package id

# ── Paths (override via env before sbatch) ───────────────────────────────────
PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs}"
WK_DIR="${DISORDERNET_WORKDIR:-}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
[[ -n "${WK_DIR}" ]] && export DISORDERNET_WORKDIR="${WK_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
: "${DISORDERNET_PUBLISH_ROOT:=${RESULTS_DIR}/publish_bundle}"
: "${DISORDERNET_PACKAGE_DIR:=${DISORDERNET_PUBLISH_ROOT}/publish_package}"
: "${BUNDLE_KIND:=650m}"
: "${PACKAGE_STRICT:=1}"

echo "Packaging publish results (${BUNDLE_KIND}) from ${DISORDERNET_PUBLISH_ROOT} → ${DISORDERNET_PACKAGE_DIR}"
echo "PACKAGE_STRICT=${PACKAGE_STRICT}"
date -Is

# Prefer Rockfish `ml`; venv activate is enough for packaging
if command -v ml &>/dev/null; then
  ml purge 2>/dev/null || true
elif command -v module &>/dev/null; then
  module purge 2>/dev/null || true
fi

if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
  echo "ERROR: Python venv missing: ${ENV_DIR}/bin/activate" >&2
  echo "Run: bash rockfish/setup_env.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
cd "${PROJECT_DIR}" || exit 1

PKG_ARGS=(
  package
  --root-workdir "${DISORDERNET_PUBLISH_ROOT}"
  --kind "${BUNDLE_KIND}"
  --package-dir "${DISORDERNET_PACKAGE_DIR}"
)
if [[ "${INCLUDE_CLEAN:-1}" != "1" ]]; then
  PKG_ARGS+=(--no-clean)
fi
if [[ -n "${PACKAGE_ID:-}" ]]; then
  PKG_ARGS+=(--package-id "${PACKAGE_ID}")
fi
case "${PACKAGE_STRICT}" in
  0|false|FALSE|no|NO) PKG_ARGS+=(--no-strict) ;;
  *) PKG_ARGS+=(--strict) ;;
esac

python rockfish/publish_submit.py "${PKG_ARGS[@]}"

echo "Done. Open ${DISORDERNET_PACKAGE_DIR}/PACKAGE_README.md"
date -Is
