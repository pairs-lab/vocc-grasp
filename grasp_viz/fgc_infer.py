"""Backend "fgc": the real FGC-GraspNet forward pass.

This is the only module that touches FreeGrasp_code -- it loads the published
network and checkpoint.  Everything downstream (mask filtering, collision,
top-down selection, rendering) is reimplemented in this package.

pred_decode returns one (N,17) row per grasp:
    [score, width, height, depth, R(9, row-major), translation(3), object_id]
"""
import os
import sys

import numpy as np

from . import paths

# grasp array column slices
C_SCORE, C_WIDTH, C_HEIGHT, C_DEPTH = 0, 1, 2, 3
C_ROT = slice(4, 13)
C_TRANS = slice(13, 16)

_NET = None


# FGC's two CUDA extensions, each built by its own setup.py
_EXT_PACKAGES = ("pointnet2", "knn")


def _ext_build_dirs():
    """Where the compiled ``pointnet2._ext`` / ``knn_pytorch`` modules live.

    ``setup.py build_ext --inplace`` cannot drop the .so beside the sources
    (each source directory *is* the package, so there is no nested package dir
    to copy into), which leaves the only build product under
    ``build/lib.<platform>-cpython-XY/``.  Only directories matching the running
    interpreter are returned: a .so built for another Python or torch ABI fails
    with an undefined-symbol error rather than a clean ImportError.
    """
    suffix = "-cpython-%d%d" % sys.version_info[:2]
    dirs = []
    for pkg in _EXT_PACKAGES:
        build = os.path.join(paths.FGC_ROOT, pkg, "build")
        if not os.path.isdir(build):
            continue
        dirs.extend(os.path.join(build, name)
                    for name in sorted(os.listdir(build), reverse=True)
                    if name.startswith("lib.") and name.endswith(suffix))
    return dirs


def _ensure_path():
    fg_root = os.path.join(paths.ROOT, "FreeGrasp_code")
    for p in (fg_root, os.path.join(paths.FGC_ROOT, "pointnet2")):
        if p not in sys.path:
            sys.path.insert(0, p)
    for p in _ext_build_dirs():
        if p not in sys.path:
            sys.path.insert(0, p)


def load_net(device="cuda", num_view=300, checkpoint=None):
    """Load FGC-GraspNet once and cache it."""
    global _NET
    if _NET is not None:
        return _NET
    _ensure_path()
    import torch
    from models.FGC_graspnet.model.FGC_graspnet import FGC_graspnet

    ckpt_path = checkpoint or paths.FGC_CHECKPOINT
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            "FGC-GraspNet checkpoint missing: %s\n"
            "Download checkpoint_fgc.tar from the FreeGrasp Google Drive." % ckpt_path)

    net = FGC_graspnet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02, hmax=0.02,
                       is_training=False, is_demo=True)
    net.to(device)
    # torch>=2.6 defaults to weights_only=True, which cannot read this archive
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    net._fgc_epoch = ckpt.get("epoch")
    _NET = net
    return net


def predict(cloud_points, device="cuda", scene=None, instance_id=None, **kwargs):
    """(N,3) metres -> (M,17) grasp array, sorted by score descending.

    ``scene``/``instance_id`` are accepted so the two backends share a signature;
    the network only needs the point cloud."""
    _ensure_path()
    import torch
    from models.FGC_graspnet.model.decode import pred_decode

    net = load_net(device=device, **kwargs)
    pts = torch.from_numpy(np.asarray(cloud_points, dtype=np.float32)[None]).to(device)
    with torch.no_grad():
        end_points = net({"point_clouds": pts})
        preds = pred_decode(end_points)
    gg = preds[0].detach().cpu().numpy()
    if len(gg):
        gg = gg[np.argsort(-gg[:, C_SCORE])]
    return gg
