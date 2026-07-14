#!/usr/bin/env bash
# One-command Rockfish publish bundle:
#   650M ultra → (optional) ultra_clean → ultra3b → organized package
#
# Uses Slurm job dependencies so each GPU phase fits in a 72 h wall clock.
# Submit once; jobs chain automatically.
#
# Usage:
#   export DISORDERNET_ACCOUNT=your_pi_gpu
#   bash rockfish/slurm/submit_publish_all.sh
#
# Optional env:
#   DISORDERNET_PUBLISH_ROOT  — parent workdir (default: ~/disordernet_runs/publish_bundle_<stamp>)
#   INCLUDE_CLEAN=0           — skip contamination-clean companion
#   INCLUDE_3B=0              — skip ESM-2 3B phase
#   BOLTZ_MODE=ingest|auto    — structure backend mode (default ingest)
#   DISORDERNET_PARTITION     — GPU partition (default a100; try ica100 for 3B OOM)
#
# Estimated GPU time:
#   ultra 650M:   ~24–48 h
#   ultra_clean:  ~24–48 h  (after 650M)
#   ultra3b:      ~30–40 h  (after 650M; parallel with clean)
#   package:      minutes   (CPU; after GPU phases)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

: "${DISORDERNET_ACCOUNT:=CHANGE_ME_gpu}"
if [[ "$DISORDERNET_ACCOUNT" == "CHANGE_ME_gpu" ]]; then
  echo "ERROR: set DISORDERNET_ACCOUNT to your Rockfish _gpu account" >&2
  exit 1
fi

: "${DISORDERNET_REPO:=$REPO_ROOT}"
: "${DISORDERNET_VENV:=$HOME/venvs/disordernet}"
: "${DISORDERNET_RESULTS:=$HOME/disordernet_runs}"
: "${DISORDERNET_PARTITION:=a100}"
: "${DISORDERNET_CPU_ACCOUNT:=${DISORDERNET_ACCOUNT}}"
: "${INCLUDE_CLEAN:=1}"
: "${INCLUDE_3B:=1}"
: "${RUN_CAID3:=1}"
: "${PREFETCH_AF:=1}"
: "${BOLTZ_MODE:=ingest}"
: "${STRUCTURE_BACKEND:=boltz}"
: "${STAGE:=pipeline}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
: "${DISORDERNET_PUBLISH_ROOT:=$HOME/disordernet_runs/publish_bundle_${STAMP}}"
: "${DISORDERNET_PACKAGE_DIR:=$DISORDERNET_PUBLISH_ROOT/publish_package}"
: "${PACKAGE_ID:=publish_${STAMP}}"

mkdir -p \
  "$DISORDERNET_PUBLISH_ROOT/ultra_650M" \
  "$DISORDERNET_PUBLISH_ROOT/ultra_clean_650M" \
  "$DISORDERNET_PUBLISH_ROOT/ultra3b" \
  "$DISORDERNET_PACKAGE_DIR"

echo "════════════════════════════════════════════════════════"
echo " DisorderNet publish-all bundle"
echo " Root:    $DISORDERNET_PUBLISH_ROOT"
echo " Package: $DISORDERNET_PACKAGE_DIR"
echo " Clean:   INCLUDE_CLEAN=$INCLUDE_CLEAN"
echo " 3B:      INCLUDE_3B=$INCLUDE_3B"
echo "════════════════════════════════════════════════════════"

_export_base() {
  local extras=("$@")
  local out="ALL,DISORDERNET_ACCOUNT,DISORDERNET_REPO,DISORDERNET_VENV,DISORDERNET_RESULTS"
  out+=",DISORDERNET_WORKDIR,PROFILE,BACKBONE,CHECKPOINT_SUBDIR,STAGE"
  out+=",RUN_CAID3,PREFETCH_AF,BOLTZ_MODE,STRUCTURE_BACKEND"
  out+=",INCLUDE_CLEAN,INCLUDE_3B,DISORDERNET_PUBLISH_ROOT,DISORDERNET_PACKAGE_DIR,PACKAGE_ID"
  out+=",RUN_NO_HALLUC_WEIGHT,RUN_NO_PLDDT_FEATURES"
  if [[ -n "${DISORDERNET_BOLTZ_ROOT:-}" ]]; then
    out+=",DISORDERNET_BOLTZ_ROOT"
  fi
  if [[ -n "${BOLTZ_CACHE:-}" ]]; then
    out+=",BOLTZ_CACHE"
  fi
  local e
  for e in "${extras[@]}"; do
    out+=",$e"
  done
  printf '%s' "$out"
}

submit_phase() {
  # Usage: submit_phase job_name profile backbone workdir ckpt_subdir dependency
  # Prints ONLY the job id on stdout; status lines go to stderr.
  local job_name="$1"
  local profile="$2"
  local backbone="$3"
  local workdir="$4"
  local ckpt_subdir="$5"
  local dep="${6:-}"

  export PROFILE="$profile"
  export BACKBONE="$backbone"
  export DISORDERNET_WORKDIR="$workdir"
  export CHECKPOINT_SUBDIR="$ckpt_subdir"
  export STAGE=pipeline
  export RUN_CAID3 PREFETCH_AF BOLTZ_MODE STRUCTURE_BACKEND
  export DISORDERNET_ACCOUNT DISORDERNET_REPO DISORDERNET_VENV DISORDERNET_RESULTS
  export INCLUDE_CLEAN INCLUDE_3B DISORDERNET_PUBLISH_ROOT DISORDERNET_PACKAGE_DIR PACKAGE_ID
  : "${RUN_NO_HALLUC_WEIGHT:=0}"
  : "${RUN_NO_PLDDT_FEATURES:=0}"
  export RUN_NO_HALLUC_WEIGHT RUN_NO_PLDDT_FEATURES

  local export_list
  export_list="$(_export_base)"

  local dep_args=()
  if [[ -n "$dep" ]]; then
    dep_args=(--dependency="$dep")
  fi

  local jid
  jid=$(sbatch --parsable \
    --account="$DISORDERNET_ACCOUNT" \
    --partition="$DISORDERNET_PARTITION" \
    --job-name="$job_name" \
    "${dep_args[@]}" \
    --export="$export_list" \
    "$SCRIPT_DIR/pipeline_phase.sbatch")
  echo "Submitted $job_name → job $jid  (profile=$profile backbone=$backbone)" >&2
  printf '%s\n' "$jid"
}

# ── Phase 1: ultra 650M (always) ─────────────────────────────────────────────
export RUN_NO_HALLUC_WEIGHT=0 RUN_NO_PLDDT_FEATURES=0
J_ULTRA=$(submit_phase \
  dn-pub-650M ultra 650M \
  "$DISORDERNET_PUBLISH_ROOT/ultra_650M" \
  checkpoints \
  "")

AFTER_GPU="afterok:${J_ULTRA}"
J_CLEAN=""
J_3B=""

# ── Phase 2a: clean companion (after 650M) ───────────────────────────────────
if [[ "$INCLUDE_CLEAN" == "1" ]]; then
  export RUN_NO_HALLUC_WEIGHT=1 RUN_NO_PLDDT_FEATURES=1
  J_CLEAN=$(submit_phase \
    dn-pub-clean ultra_clean 650M \
    "$DISORDERNET_PUBLISH_ROOT/ultra_clean_650M" \
    checkpoints_ultra_clean \
    "afterok:${J_ULTRA}")
fi

# ── Phase 2b: ultra3b (after 650M; parallel with clean) ───────────────────────
if [[ "$INCLUDE_3B" == "1" ]]; then
  export RUN_NO_HALLUC_WEIGHT=0 RUN_NO_PLDDT_FEATURES=0
  J_3B=$(submit_phase \
    dn-pub-3B ultra3b 3B \
    "$DISORDERNET_PUBLISH_ROOT/ultra3b" \
    checkpoints \
    "afterok:${J_ULTRA}")
fi

# Dependency for package: wait for every GPU phase that was submitted
AFTER_GPU="afterok:${J_ULTRA}"
[[ -n "$J_CLEAN" ]] && AFTER_GPU="${AFTER_GPU}:${J_CLEAN}"
[[ -n "$J_3B" ]] && AFTER_GPU="${AFTER_GPU}:${J_3B}"
# ── Phase 3: package (CPU) ───────────────────────────────────────────────────
export DISORDERNET_PUBLISH_ROOT DISORDERNET_PACKAGE_DIR PACKAGE_ID INCLUDE_CLEAN INCLUDE_3B
PKG_EXPORT="$(_export_base)"
J_PKG=$(sbatch --parsable \
  --account="$DISORDERNET_CPU_ACCOUNT" \
  --job-name=dn-pub-pkg \
  --dependency="$AFTER_GPU" \
  --export="$PKG_EXPORT" \
  "$SCRIPT_DIR/package_results.sbatch")

echo ""
echo "All jobs submitted:"
echo "  ultra_650M:  $J_ULTRA"
[[ -n "$J_CLEAN" ]] && echo "  ultra_clean: $J_CLEAN"
[[ -n "$J_3B" ]] && echo "  ultra3b:     $J_3B"
echo "  package:     $J_PKG  (after GPU phases)"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Package:  $DISORDERNET_PACKAGE_DIR"
echo "Bundle:   $DISORDERNET_PUBLISH_ROOT"
