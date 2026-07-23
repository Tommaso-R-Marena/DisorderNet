#!/bin/bash -ue
#SBATCH --job-name="dn-package"
#SBATCH --partition=shared
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

# Paths are resolved inside _common.sh (fail-loud venv + repo checks).
PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
RESULTS_DIR="${DISORDERNET_RESULTS:-${HOME}/disordernet_runs}"
: "${DISORDERNET_PUBLISH_ROOT:=${RESULTS_DIR}/publish_bundle}"
: "${DISORDERNET_PACKAGE_DIR:=${DISORDERNET_PUBLISH_ROOT}/publish_package}"
: "${BUNDLE_KIND:=650m}"
: "${PACKAGE_STRICT:=1}"
export DISORDERNET_REPO="${PROJECT_DIR}"
export DISORDERNET_VENV="${ENV_DIR}"
export DISORDERNET_RESULTS="${RESULTS_DIR}"
export DISORDERNET_PUBLISH_ROOT DISORDERNET_PACKAGE_DIR BUNDLE_KIND PACKAGE_STRICT

# Package job: setup + package only (no GPU pipeline run / mirror).
export DISORDERNET_COMMON_AUTORUN=1
export DISORDERNET_COMMON_PACKAGE_ONLY=1
export SKIP_MIRROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "${SCRIPT_DIR}/_common.sh"
