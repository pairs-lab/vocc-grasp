"""Orbiting point-cloud GIFs, in the style of FreeGrasp's demo animations.

Renders the scene's coloured point cloud plus the gripper mesh from a camera
circling the grasp, as an animated GIF.  Uses a small software splat
rasteriser -- no OpenGL context, so it works headless.

    python -m grasp_viz.make_orbit_gif --all
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

from . import cases as C
from . import paths, render
from . import scene as S

BG = (243, 241, 236)
GRIPPER_COLOR = (30, 200, 60)
FRAMES = 72
DURATION_MS = 45
CANVAS = 460
CLOUD_POINTS = 35000
GRIPPER_POINTS = 9000
ELEVATION_DEG = 26.0
SPLAT_RADIUS = 1          # px; a 3x3 splat keeps the cloud looking solid
FILL = 0.78               # fraction of the canvas the scene should span
SCENE_RADIUS_FACTOR = 1.7  # how much cloud to keep around the gripper
MIN_SCENE_RADIUS_M = 0.22


def read_obj(path):
    """Vertices and triangles of the meshes this package writes."""
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def sample_mesh(verts, faces, n, seed=0):
    """Uniform surface samples, so the gripper renders as a solid body."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        return verts
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), n, p=area / total)
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    over = (u + v) > 1
    u[over], v[over] = 1 - u[over], 1 - v[over]
    return a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])


def view_matrix(eye, center, up=(0.0, -1.0, 0.0)):
    """Rows are [right, down, forward] -- forward is +z, matching the
    projection used everywhere else in this package."""
    fwd = np.asarray(center, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, float))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=0)


def splat(points, colors, size=CANVAS, focal=None, radius=SPLAT_RADIUS):
    """Painter-correct point splat: for every pixel keep the nearest sample."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = BG
    z = points[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return Image.fromarray(img)
    p, c = points[ok], colors[ok]
    focal = focal or size
    u = np.rint(p[:, 0] * focal / p[:, 2] + size / 2).astype(np.int64)
    v = np.rint(p[:, 1] * focal / p[:, 2] + size / 2).astype(np.int64)

    offs = [(dx, dy) for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)]
    uu = np.concatenate([u + dx for dx, _ in offs])
    vv = np.concatenate([v + dy for _, dy in offs])
    zz = np.tile(p[:, 2], len(offs))
    cc = np.tile(c, (len(offs), 1))

    inside = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
    uu, vv, zz, cc = uu[inside], vv[inside], zz[inside], cc[inside]
    if len(uu) == 0:
        return Image.fromarray(img)

    flat = vv * size + uu
    order = np.lexsort((zz, flat))            # by pixel, then nearest first
    flat_s = flat[order]
    first = np.ones(len(flat_s), dtype=bool)
    first[1:] = flat_s[1:] != flat_s[:-1]
    sel = order[first]
    img.reshape(-1, 3)[flat[sel]] = cc[sel]
    return Image.fromarray(img)


def _eye(center, dist, az, el):
    return center + dist * np.array([
        math.cos(el) * math.sin(az), -math.sin(el), -math.cos(el) * math.cos(az)])


def orbit_frames(points, colors, center, extent, frames=FRAMES, size=CANVAS,
                 elevation_deg=ELEVATION_DEG):
    """Circle the grasp once, keeping it centred and filling the frame.

    The zoom is fitted from the widest *projected* silhouette over the whole
    orbit -- sizing it from the 3D radius instead would leave the scene small,
    because most of its spread is in depth, which costs no image area."""
    dist = max(extent * 2.6, 0.25)
    el = math.radians(elevation_deg)
    azimuths = [2 * math.pi * i / frames for i in range(frames)]

    spans = []
    for az in azimuths[::max(frames // 12, 1)]:
        eye = _eye(center, dist, az, el)
        cam = (points - eye) @ view_matrix(eye, center).T
        z = np.maximum(cam[:, 2], 1e-6)
        spans.append(float(np.percentile(
            np.maximum(np.abs(cam[:, 0]), np.abs(cam[:, 1])) / z, 97)))
    focal = FILL * (size / 2) / max(float(np.median(spans)), 1e-6)

    radius = SPLAT_RADIUS if len(points) > 25000 else SPLAT_RADIUS + 1
    out = []
    for az in azimuths:
        eye = _eye(center, dist, az, el)
        cam = (points - eye) @ view_matrix(eye, center).T
        out.append(splat(cam, colors, size=size, focal=focal, radius=radius))
    return out


def build_gif(case_dir, out_path, scene_points=CLOUD_POINTS):
    ply = os.path.join(case_dir, "cloud.ply")
    obj = os.path.join(case_dir, "grasp.obj")
    if not (os.path.exists(ply) and os.path.exists(obj)):
        return False

    pts, cols = [], []
    with open(ply) as fh:
        for line in fh:
            if line.startswith("end_header"):
                break
        for line in fh:
            f = line.split()
            if len(f) < 6:
                continue
            pts.append(f[:3])
            cols.append(f[3:6])
    cloud = np.asarray(pts, dtype=np.float64)
    cloud_c = np.asarray(cols, dtype=np.uint8)

    verts, faces = read_obj(obj)
    grip = sample_mesh(verts, faces, GRIPPER_POINTS)
    grip_c = np.tile(np.array(GRIPPER_COLOR, dtype=np.uint8), (len(grip), 1))

    # frame the grasp, not the whole crop: the cropped cloud carries a lot of
    # background plane that would otherwise drag the centre around as it spins
    center = grip.mean(axis=0)
    grip_radius = float(np.linalg.norm(grip - center, axis=1).max())
    keep_radius = max(grip_radius * SCENE_RADIUS_FACTOR, MIN_SCENE_RADIUS_M)
    near = np.linalg.norm(cloud - center, axis=1) <= keep_radius
    cloud, cloud_c = cloud[near], cloud_c[near]

    points = np.concatenate([cloud, grip], axis=0)
    colors = np.concatenate([cloud_c, grip_c], axis=0)
    extent = float(np.percentile(np.linalg.norm(points - center, axis=1), 98))

    frames = orbit_frames(points, colors, center, extent)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=DURATION_MS, loop=0, optimize=True)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=["fgc"], default="fgc")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--dataset", choices=[paths.SYNTHETIC, paths.REAL],
                    default=paths.SYNTHETIC, help="which capture set to render")
    ap.add_argument("--n-medium", type=int, default=7, help="real dataset only")
    ap.add_argument("--n-hard", type=int, default=8, help="real dataset only")
    args = ap.parse_args(argv)

    if not args.methods:
        args.methods = list(paths.method_logs(args.dataset))

    if args.dataset == paths.REAL:
        selected = C.load_cases_real(methods=args.methods, keys=args.cases,
                                     picks=(("Medium", args.n_medium),
                                            ("Hard", args.n_hard)))
    else:
        selected = C.load_cases(methods=args.methods)
        if args.cases:
            wanted = set(args.cases)
            selected = [c for c in selected if c.key in wanted]

    roots = paths.out_dirs(args.backend, args.dataset)
    made = skipped = 0
    for case in selected:
        for method in args.methods:
            d = os.path.join(roots[method], case.key)
            out = os.path.join(d, "05_orbit.gif")
            if build_gif(d, out):
                made += 1
                print("  %s" % out)
            else:
                skipped += 1
    print("\n%d gifs written, %d skipped (no grasp)" % (made, skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
