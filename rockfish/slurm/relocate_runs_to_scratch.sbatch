#!/bin/bash -ue
#SBATCH --job-name="dn-relocate"
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=00-03:00:00
#SBATCH --account=sfried3
#SBATCH --export=ALL
#SBATCH --output=logs/dn-relocate_%j.out
#SBATCH --error=logs/dn-relocate_%j.err
# Relocate a project run directory from $HOME to /scratch4, leaving a symlink.
#
# The home quota is 50GB and invisible to both df and quota. A job that crosses
# it fails on its next write — including the write of its own log, which is how
# three earlier runs disappeared without leaving any diagnosis. Run outputs are
# the only part of this project large enough to matter, and they belong on
# /scratch4 anyway.
#
# Nothing is removed until the copy is verified file-for-file and byte-for-byte.
# On any mismatch the source is left exactly as it was.
#
#   export RELOC_SRC="${HOME}/disordernet_runs_rigor"
#   export RELOC_DST=/scratch4/sfried3/jbeale3_disordernet/runs_rigor
#   sbatch rockfish/slurm/relocate_runs_to_scratch.sbatch

: "${RELOC_SRC:?Set RELOC_SRC to the directory to relocate}"
: "${RELOC_DST:?Set RELOC_DST to the destination under /scratch4}"

case "${RELOC_DST}" in
  /scratch4/*) ;;
  *) echo "ERROR: RELOC_DST must be under /scratch4 (got ${RELOC_DST})" >&2
     exit 1 ;;
esac
if [[ -L "${RELOC_SRC}" ]]; then
  echo "${RELOC_SRC} is already a symlink; nothing to do."
  exit 0
fi
if [[ ! -d "${RELOC_SRC}" ]]; then
  echo "ERROR: ${RELOC_SRC} is not a directory" >&2
  exit 1
fi

echo "source      : ${RELOC_SRC}"
echo "destination : ${RELOC_DST}"
mkdir -p "${RELOC_DST}"

echo "copying…"
rsync -a --no-compress "${RELOC_SRC}/" "${RELOC_DST}/"

# Verify before removing anything. Count regular files and total bytes on both
# sides; a copy that lost or truncated a file must not be followed by a delete.
src_files=$(find "${RELOC_SRC}" -type f | wc -l)
dst_files=$(find "${RELOC_DST}" -type f | wc -l)
src_bytes=$(find "${RELOC_SRC}" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
dst_bytes=$(find "${RELOC_DST}" -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')

echo "files : ${src_files} -> ${dst_files}"
echo "bytes : ${src_bytes} -> ${dst_bytes}"

if [[ "${src_files}" -ne "${dst_files}" || "${src_bytes}" -ne "${dst_bytes}" ]]; then
  echo "ERROR: copy does not match the source. Source left untouched." >&2
  exit 1
fi

# A second pass with checksums, since matching sizes do not prove matching
# contents. --dry-run so this only reports; any output means a difference.
diffs=$(rsync -rc --dry-run --out-format='%n' "${RELOC_SRC}/" "${RELOC_DST}/" | wc -l)
if [[ "${diffs}" -ne 0 ]]; then
  echo "ERROR: ${diffs} file(s) differ by checksum. Source left untouched." >&2
  exit 1
fi
echo "checksum pass: contents identical"

# Only now does the source come out, and it is replaced by a symlink so every
# existing path keeps resolving.
trash="${RELOC_SRC}.relocated.$$"
mv "${RELOC_SRC}" "${trash}"
ln -s "${RELOC_DST}" "${RELOC_SRC}"
rm -rf "${trash}"

echo "done: ${RELOC_SRC} -> $(readlink "${RELOC_SRC}")"
du -sh "${RELOC_DST}"
