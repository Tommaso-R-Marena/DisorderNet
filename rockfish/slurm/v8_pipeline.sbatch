#!/bin/bash -ue
#SBATCH --job-name="dn-v8-cv"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01-00:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-v8-cv_%j.out
#SBATCH --error=logs/dn-v8-cv_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
#
# CPU pipeline for the v8 multi-scale ensemble. The GBDT models (LightGBM/XGBoost/
# HistGBM) are CPU-only, so this runs on the `shared` partition — do NOT hold a GPU
# for it. Run v8_extract_embeddings.sbatch first (it fills the per-backbone homes).
#
# Submit after extraction (optionally chain with --dependency):
#   sbatch --account=sfried3 rockfish/slurm/v8_pipeline.sbatch
#   # or: sbatch --dependency=afterok:<embed_jobid> rockfish/slurm/v8_pipeline.sbatch
#
# Override via env: DISORDERNET_REPO, DISORDERNET_VENV, DISORDERNET_V8_DIR,
#                   DISORDERNET_BACKBONES, RUN_HOMOLOGY=1

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
WK_DIR="${DISORDERNET_V8_DIR:-${HOME}/scr4_sfried3/disordernet_v8}"
BACKBONES="${DISORDERNET_BACKBONES:-esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D}"
RUN_HOMOLOGY="${RUN_HOMOLOGY:-1}"

cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
mkdir -p "${PROJECT_DIR}/logs"

labels=()
for MODEL in ${BACKBONES}; do
	case "${MODEL}" in
		*_8M_*)   LABEL="8m" ;;
		*_35M_*)  LABEL="35m" ;;
		*_150M_*) LABEL="150m" ;;
		*_650M_*) LABEL="650m" ;;
		*_3B_*)   LABEL="3b" ;;
		*)        LABEL="${MODEL}" ;;
	esac
	labels+=("${LABEL}")
	HOME_BB="${WK_DIR}/${LABEL}"
	echo "[$(date +%T)] v7 random-split CV: ${LABEL}"
	DISORDERNET_HOME="${HOME_BB}" python run_v7.py
	if [[ "${RUN_HOMOLOGY}" == "1" ]]; then
		echo "[$(date +%T)] v7 homology-split CV: ${LABEL}"
		DISORDERNET_HOME="${HOME_BB}" DISORDERNET_SPLIT=homology \
			DISORDERNET_RESULTS_ROOT="${HOME_BB}/hom" python run_v7.py
	fi
done

# Multi-scale ensemble (random split)
rand_dirs=(); for L in "${labels[@]}"; do rand_dirs+=("${WK_DIR}/${L}/results_v7"); done
echo "[$(date +%T)] v8 ensemble (random split)"
DISORDERNET_HOME="${WK_DIR}/${labels[0]}" \
	DISORDERNET_RESULTS_ROOT="${WK_DIR}/ensemble" \
	python run_v8_multiscale.py --backbones "${rand_dirs[@]}"

if [[ "${RUN_HOMOLOGY}" == "1" ]]; then
	hom_dirs=(); for L in "${labels[@]}"; do hom_dirs+=("${WK_DIR}/${L}/hom/results_v7"); done
	echo "[$(date +%T)] v8 ensemble (homology split)"
	DISORDERNET_HOME="${WK_DIR}/${labels[0]}" \
		DISORDERNET_RESULTS_ROOT="${WK_DIR}/ensemble_hom" \
		python run_v8_multiscale.py --backbones "${hom_dirs[@]}"
fi

# Deployable predictor bundle (ESM-2 35M home for fast inference)
if [[ -d "${WK_DIR}/35m/data/embeddings" ]]; then
	echo "[$(date +%T)] training deployable predictor bundle"
	DISORDERNET_HOME="${WK_DIR}/35m" python train_predictor.py
fi

echo "[$(date +%T)] v8 pipeline complete."
echo "  ensemble metrics : ${WK_DIR}/ensemble/results_v8/metrics.json"
[[ "${RUN_HOMOLOGY}" == "1" ]] && echo "  homology metrics : ${WK_DIR}/ensemble_hom/results_v8/metrics.json"
