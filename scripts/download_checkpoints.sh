#!/usr/bin/env bash
# Fetch the two model checkpoints. Neither can be redistributed in git.
#
#   ./scripts/download_checkpoints.sh          # both
#   ./scripts/download_checkpoints.sh fgc      # FGC-GraspNet only (13 MB)
#   ./scripts/download_checkpoints.sh uoais    # UOAIS only (427 MB)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"

fgc () {
  local dest="$ROOT/FreeGrasp_code/logs/checkpoint_fgc.tar"
  [ -f "$dest" ] && { echo "FGC checkpoint already present"; return; }
  $PY -c "import gdown" 2>/dev/null || $PY -m pip install -q gdown
  echo "downloading checkpoint_fgc.tar (13 MB)"
  tmp=$(mktemp -d)
  $PY -m gdown --folder \
    "https://drive.google.com/drive/folders/1w5cZAfY9h0O9908y9YvL88KIPL_7J7EF" -O "$tmp"
  mkdir -p "$(dirname "$dest")"; mv "$tmp/checkpoint_fgc.tar" "$dest"; rm -rf "$tmp"
  echo "-> FreeGrasp_code/logs/checkpoint_fgc.tar"
}

uoais () {
  local dest="$ROOT/uoais-ft/output/R50_rgbdconcat_mlc_occatmask_hom_concat/model_final.pth"
  [ -f "$dest" ] && { echo "UOAIS weights already present"; return; }
  cat <<MSG
UOAIS weights (427 MB) need the upstream terms accepted. Download
R50_rgbdconcat_mlc_occatmask_hom_concat from https://github.com/gist-ailab/uoais
and place model_final.pth at:
  uoais-ft/output/R50_rgbdconcat_mlc_occatmask_hom_concat/model_final.pth
MSG
}

case "${1:-all}" in
  fgc) fgc ;; uoais) uoais ;; all) fgc; uoais ;;
  *) echo "usage: $0 [fgc|uoais|all]" >&2; exit 1 ;;
esac
