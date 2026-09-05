#!/usr/bin/env bash
# Build FGC-GraspNet's pointnet2 and knn CUDA extensions against the installed torch.
#
#   CUDA_HOME=$CONDA_PREFIX ./scripts/build_fgc_extensions.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
: "${CUDA_HOME:?set CUDA_HOME to a CUDA 12.x toolkit (nvcc must be in \$CUDA_HOME/bin)}"

# torch's cpp_extension searches only $CUDA_HOME/include, while conda and pip scatter the
# CUDA headers elsewhere. Without this the build dies on a missing cusparse.h.
for f in "$CUDA_HOME"/targets/x86_64-linux/include/*; do
  [ -e "$f" ] && ln -sfn "$f" "$CUDA_HOME/include/$(basename "$f")"
done
for d in cusparse cublas cusolver cufft curand nvjitlink cuda_runtime; do
  for f in "$CUDA_HOME"/lib/python3.*/site-packages/nvidia/$d/include/*; do
    [ -e "$f" ] && ln -sfn "$f" "$CUDA_HOME/include/$(basename "$f")"
  done
done

export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS=8
FGC="$ROOT/FreeGrasp_code/models/FGC_graspnet"
( cd "$FGC/pointnet2" && $PY setup.py build_ext )
( cd "$FGC/knn"       && $PY setup.py build_ext )

SP=$($PY -c 'import site;print(site.getsitepackages()[0])')
mkdir -p "$SP/pointnet2" "$SP/knn_pytorch"
cp "$FGC"/pointnet2/build/lib.*/pointnet2/_ext*.so "$SP/pointnet2/"
cp "$FGC"/knn/build/lib.*/knn_pytorch/knn_pytorch*.so "$SP/knn_pytorch/"
: > "$SP/pointnet2/__init__.py"; : > "$SP/knn_pytorch/__init__.py"
( cd /tmp && $PY -c "import torch, pointnet2._ext, knn_pytorch; print('FGC extensions ok')" )
