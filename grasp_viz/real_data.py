"""Scene loading for the MetaGraspNetV2 real captures.

The benchmark ships 1200x1200 RGB (``images/``) and instance maps
(``masks_npy_real_crop/``), but no depth and no intrinsics.  Both still exist in
the raw captures under ``data_ifl_*/mnt/data1/data_ifl_real/scene*``, at the
original 1944x1200 resolution; the benchmark files are the centred 1200-wide
crop of those, i.e. ``raw[:, 372:1572]`` -- verified byte-identical for both RGB
and the instance map.

So a real scene is stitched from three places:

    RGB       images/image_%06d.png                      (already cropped)
    instances masks_npy_real_crop/image_%06d.npy         (already cropped)
    depth     <scene>/<view>.npz['depth']                (raw, crop it here)
    K         <scene>/<view>_camera_params.json          (raw, shift cx here)

``image_id -> sceneN_viewM`` comes from ``real_world_mapping_fixed.json``.
Depth is in centimetres like the synthetic set, and ~22% of it is NaN (real
sensor dropouts), which is zeroed so ``scene.sample_cloud``'s ``depth_m > 0``
filter drops those pixels instead of leaking NaN points into the grasp model.
"""
import functools
import glob
import json
import os
import re

import numpy as np
from PIL import Image

from . import paths
from . import scene as S

CROP_X0 = 372              # (1944 - 1200) / 2
CROP_W = 1200
SCENE_VIEW_RE = re.compile(r"(scene\d+)_view(\d+)")


@functools.lru_cache(maxsize=1)
def _scene_index():
    """{'scene0': '/abs/path/.../scene0'} across every data_ifl_* root."""
    index = {}
    for root in sorted(glob.glob(paths.DATA_IFL_GLOB)):
        for name in os.listdir(root):
            index.setdefault(name, os.path.join(root, name))
    return index


@functools.lru_cache(maxsize=1)
def _mapping():
    with open(paths.MAPPING_JSON) as fh:
        return json.load(fh)


def resolve(image_id):
    """image_id -> (scene directory, view number as a string)."""
    key = "%06d" % int(image_id)
    mapping = _mapping()
    if key not in mapping:
        raise KeyError("image_id %s is not in %s" % (key, paths.MAPPING_JSON))
    m = SCENE_VIEW_RE.match(mapping[key])
    if not m:
        raise ValueError("unexpected mapping value %r" % mapping[key])
    scene_name, view = m.group(1), m.group(2)
    index = _scene_index()
    if scene_name not in index:
        raise FileNotFoundError("no capture directory for %s" % scene_name)
    return index[scene_name], view


def camera(image_id):
    """Pinhole K of the *cropped* frame: only cx moves with the crop."""
    scene_dir, view = resolve(image_id)
    with open(os.path.join(scene_dir, view + "_camera_params.json")) as fh:
        p = json.load(fh)
    return S.Camera(width=CROP_W, height=int(p["height"]),
                    fx=float(p["fx"]), fy=float(p["fy"]),
                    cx=float(p["cx"]) - CROP_X0, cy=float(p["cy"]))


def load_scene(image_id):
    """Same Scene contract as ``scene.load_scene``, real data behind it."""
    scene_dir, view = resolve(image_id)
    rgb = np.asarray(Image.open(paths.image_path_real(image_id)).convert("RGB"))
    instances = np.load(paths.annot_path_real(image_id)).astype(np.int32)

    raw = np.load(os.path.join(scene_dir, view + ".npz"))["depth"]
    depth_m = np.nan_to_num(raw[:, CROP_X0:CROP_X0 + CROP_W],
                            nan=0.0, posinf=0.0, neginf=0.0)
    depth_m = depth_m.astype(np.float64) / S.DEPTH_SCALE_CM

    cam = camera(image_id)
    return S.Scene(image_id=image_id, rgb=rgb, depth_m=depth_m,
                   instances=instances, camera=cam,
                   cloud=cam.backproject(depth_m))
