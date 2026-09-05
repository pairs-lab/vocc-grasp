"""RGB-D scene handling for UnoBench.

UnoBench ships no intrinsics, so a pinhole K is synthesised from an assumed
vertical FOV -- the same free parameter the repo already uses elsewhere.
Depth is stored in **centimetres**, so the metric scale is 100, not 1000;
getting that wrong puts the scene at 10 cm and makes a 2-10 cm gripper
meaningless.
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import paths

DEFAULT_FOV_DEG = 60.0
DEPTH_SCALE_CM = 100.0     # depth[.npy] is in cm -> metres
CROP_KERNEL = 0.2          # bbox margin ratio, as in FreeGrasp's crop
CROP_PAD_PX = 50
NUM_POINT = 20000


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(cls, height, width, fov_deg=DEFAULT_FOV_DEG):
        f = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
        return cls(width=width, height=height, fx=f, fy=f,
                   cx=width / 2.0, cy=height / 2.0)

    def project(self, points):
        """(N,3) camera-frame metres -> (N,2) int pixels."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        z = np.where(np.abs(pts[:, 2]) < 1e-9, 1e-9, pts[:, 2])
        u = pts[:, 0] * self.fx / z + self.cx
        v = pts[:, 1] * self.fy / z + self.cy
        return np.stack([u, v], axis=-1)

    def backproject(self, depth_m):
        """(H,W) metres -> organised (H,W,3) cloud in metres."""
        xmap, ymap = np.meshgrid(np.arange(self.width), np.arange(self.height))
        z = depth_m
        x = (xmap - self.cx) * z / self.fx
        y = (ymap - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=-1)


@dataclass
class Scene:
    image_id: int
    rgb: np.ndarray            # (H,W,3) uint8
    depth_m: np.ndarray        # (H,W) float64, metres
    instances: np.ndarray      # (H,W) int, 0 = background
    camera: Camera
    cloud: np.ndarray          # (H,W,3) float64, metres

    @property
    def shape(self):
        return self.depth_m.shape


def load_scene(image_id, fov_deg=DEFAULT_FOV_DEG):
    rgb = np.asarray(Image.open(paths.image_path(image_id)).convert("RGB"))
    depth_m = np.load(paths.depth_path(image_id)).astype(np.float64) / DEPTH_SCALE_CM
    instances = np.load(paths.annot_path(image_id)).astype(np.int32)
    h, w = depth_m.shape
    cam = Camera.from_fov(h, w, fov_deg)
    return Scene(image_id=image_id, rgb=rgb, depth_m=depth_m, instances=instances,
                 camera=cam, cloud=cam.backproject(depth_m))


def instance_mask(scene, instance_id):
    return scene.instances == int(instance_id)


def crop_region(scene, mask):
    """Region of interest around a target mask: its bbox widened by
    CROP_KERNEL on each side plus CROP_PAD_PX, clamped to the image."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.ones(scene.shape, dtype=bool)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    mx = int((x2 - x1) * CROP_KERNEL) + CROP_PAD_PX
    my = int((y2 - y1) * CROP_KERNEL) + CROP_PAD_PX
    h, w = scene.shape
    region = np.zeros(scene.shape, dtype=bool)
    region[max(y1 - my, 0):min(y2 + my, h), max(x1 - mx, 0):min(x2 + mx, w)] = True
    return region


def sample_cloud(scene, region, num_point=NUM_POINT, seed=0):
    """Points + colours inside ``region`` with valid depth, resampled to a
    fixed count so the network always sees the same tensor shape."""
    keep = region & (scene.depth_m > 0)
    pts = scene.cloud[keep]
    cols = scene.rgb[keep].astype(np.float64) / 255.0
    if len(pts) == 0:
        return pts, cols
    rng = np.random.default_rng(seed)
    if len(pts) >= num_point:
        idx = rng.choice(len(pts), num_point, replace=False)
    else:
        idx = np.concatenate([np.arange(len(pts)),
                              rng.choice(len(pts), num_point - len(pts), replace=True)])
    return pts[idx], cols[idx]
