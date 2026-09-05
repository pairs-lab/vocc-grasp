#!/usr/bin/env bash
# Create the environment and install everything the repo needs.
#
#   ./scripts/install.sh            # env + torch + requirements + FGC extensions
#   ./scripts/install.sh uoais      # additionally: detectron2 + AdelaiDet (for UOAIS)
#
# One environment runs the whole repo. Tested on an RTX 5060 Ti (sm_120) with
# torch 2.7.0+cu128 and CUDA 12.8.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-vocc}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

info () { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

if [ ! -x "$PY" ]; then
  info "creating conda env '$ENV_NAME' (python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi

info "installing torch (cu128) and requirements"
"$PY" -m pip install -q torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -r "$ROOT/requirements.txt"

info "installing the CUDA toolchain into the env"
conda install -y -n "$ENV_NAME" -c nvidia \
    cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-driver-dev=12.8 cuda-cccl=12.8

info "building FGC-GraspNet CUDA extensions"
CUDA_HOME="$CONDA_ROOT/envs/$ENV_NAME" PY="$PY" "$ROOT/scripts/build_fgc_extensions.sh"

if [ "${1:-}" = "uoais" ]; then
  info "UOAIS needs detectron2 + AdelaiDet, built against the torch just installed"
  "$PY" -m pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
  "$PY" -m pip install -q 'git+https://github.com/aim-uofa/AdelaiDet.git'
  cat <<'MSG'
If either build fails, follow https://github.com/gist-ailab/uoais -- they pin the
versions that work together. uoais-ft/ here is that repo with a local C++/CUDA fix.
MSG
fi

info "done. activate with:  conda activate $ENV_NAME"
