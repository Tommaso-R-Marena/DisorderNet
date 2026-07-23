#!/bin/bash -ue
#SBATCH --job-name="dn-v8-embed"
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00-08:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-v8-embed_%j.out
#SBATCH --error=logs/dn-v8-embed_%j.err
##SBATCH --mail-user=you@jh.edu
##SBATCH --mail-type=END,FAIL,INVALID_DEPEND,TIME_LIMIT
#
# GPU-accelerated ESM-2 embedding extraction for the v8 multi-scale ensemble.
# Extracts 35M/150M/650M into per-backbone homes under a scratch working dir, then
# the CPU pipeline job (v8_pipeline.sbatch) trains v7 per backbone and ensembles.
#
# Submit (from the repo root, after rockfish/setup_env.sh):
#   sbatch --account=sfried3 rockfish/slurm/v8_extract_embeddings.sbatch
#
# Override via env before sbatch (see also --export):
#   DISORDERNET_REPO, DISORDERNET_VENV, DISORDERNET_V8_DIR, DISORDERNET_BACKBONES

PROJECT_DIR="${DISORDERNET_REPO:-${HOME}/DisorderNet}"
ENV_DIR="${DISORDERNET_VENV:-${HOME}/venvs/disordernet}"
WK_DIR="${DISORDERNET_V8_DIR:-${HOME}/scr4_sfried3/disordernet_v8}"
BACKBONES="${DISORDERNET_BACKBONES:-esm2_t12_35M_UR50D esm2_t30_150M_UR50D esm2_t33_650M_UR50D}"

cd "${PROJECT_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
mkdir -p "${PROJECT_DIR}/logs" "${WK_DIR}/data"

export DISORDERNET_HOME="${WK_DIR}"
DISPROT="${WK_DIR}/data/disprot_processed.json"
[[ -f "${DISPROT}" ]] || python fetch_disprot.py

for MODEL in ${BACKBONES}; do
	case "${MODEL}" in
		*_8M_*)   LABEL="8m" ;;
		*_35M_*)  LABEL="35m" ;;
		*_150M_*) LABEL="150m" ;;
		*_650M_*) LABEL="650m" ;;
		*_3B_*)   LABEL="3b" ;;
		*)        LABEL="${MODEL}" ;;
	esac
	HOME_BB="${WK_DIR}/${LABEL}"
	mkdir -p "${HOME_BB}/data"
	ln -sf "${DISPROT}" "${HOME_BB}/data/disprot_processed.json"
	echo "[$(date +%T)] extracting ${MODEL} -> ${HOME_BB}/data/embeddings"
	python extract_embeddings.py \
		--model "${MODEL}" \
		--data "${DISPROT}" \
		--out "${HOME_BB}/data/embeddings" \
		--batch-tokens 8000
done

echo "[$(date +%T)] embedding extraction complete under ${WK_DIR}"
