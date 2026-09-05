"""2D overlays, mesh/cloud export, and the side-by-side comparison figure."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import gripper

METHOD_COLOR = {
    "gemini_uoais_ref": (0, 210, 255),
    "gemini_noref": (255, 165, 0),
    "unograsp": (255, 70, 210),
}
METHOD_TITLE = {
    "gemini_uoais_ref": "Gemini + UOAIS ref",
    "gemini_noref": "Gemini (no ref)",
    "unograsp": "UnoGrasp",
}
CORRECT_COLOR = (40, 200, 90)
WRONG_COLOR = (235, 60, 60)
GT_COLOR = (255, 255, 255)

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def font(size):
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def tint(rgb, mask, color, alpha=0.42):
    """Blend a flat colour into the masked pixels."""
    out = rgb.astype(np.float64).copy()
    col = np.asarray(color, dtype=np.float64)
    out[mask] = out[mask] * (1 - alpha) + col * alpha
    return out.astype(np.uint8)


def mask_outline(mask, thickness=4):
    """Boolean ring just inside the mask border."""
    m = mask.astype(np.uint8)
    eroded = m.copy()
    for _ in range(thickness):
        e = eroded.copy()
        e[1:, :] &= eroded[:-1, :]
        e[:-1, :] &= eroded[1:, :]
        e[:, 1:] &= eroded[:, :-1]
        e[:, :-1] &= eroded[:, 1:]
        eroded = e
    return mask & ~eroded.astype(bool)


def _dashed(draw, p0, p1, color, width, dash=18, gap=14, halo=False):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    if length < 1e-6:
        return
    step = (p1 - p0) / length
    pos = 0.0
    while pos < length:
        a = p0 + step * pos
        b = p0 + step * min(pos + dash, length)
        if halo:
            draw.line([tuple(a), tuple(b)], fill=HALO, width=width + 4)
        draw.line([tuple(a), tuple(b)], fill=color, width=width)
        pos += dash + gap


def draw_bbox_dashed(draw, mask, color, width=4, window=None, label=None):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    box = np.array([[xs.min(), ys.min()], [xs.max(), ys.max()]], dtype=np.float64)
    if window is not None:
        box = _shift(box, window)
    (x1, y1), (x2, y2) = box
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        _dashed(draw, corners[i], corners[(i + 1) % 4], color, width, halo=True)
    if label:
        _label(draw, (x1 + 8, y1 + 6), label, color, size=22, pad=4)


def fit_font_size(draw, text, max_width, start=30, floor=17, pad=8):
    """Largest font size whose widest line still fits ``max_width``."""
    for size in range(start, floor - 1, -1):
        box = draw.multiline_textbbox((0, 0), text, font=font(size))
        if box[2] - box[0] + 2 * pad <= max_width:
            return size
    return floor


def _label(draw, xy, text, fill, size=30, pad=8):
    f = font(size)
    x, y = xy
    box = draw.textbbox((x, y), text, font=f)
    draw.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
                   fill=(0, 0, 0, 230))
    draw.text((x, y), text, font=f, fill=fill)



PAD_COLOR = (26, 26, 30)
HALO = (12, 12, 14)


def compute_window(image_size, masks, extra_pts=(), margin_frac=0.14, min_size=420):
    """Square crop covering every given mask plus any extra points (the
    projected gripper).  A GraspGen wrist can land outside the image, so the
    window is allowed to overhang and is padded when cropped."""
    w, h = image_size
    xs, ys = [], []
    for m in masks:
        if m is None or not m.any():
            continue
        ry, rx = np.nonzero(m)
        xs += [rx.min(), rx.max()]
        ys += [ry.min(), ry.max()]
    for p in np.atleast_2d(np.asarray(extra_pts, dtype=np.float64)) if len(extra_pts) else []:
        xs.append(p[0])
        ys.append(p[1])
    if not xs:
        return (0, 0, w, h)

    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    side = max(x1 - x0, y1 - y0, min_size) * (1 + 2 * margin_frac)
    # centred on the content and never clamped back into the image: a grasp
    # whose wrist sits outside the frame must still be shown whole
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = side / 2
    return (int(round(cx - half)), int(round(cy - half)),
            int(round(cx + half)), int(round(cy + half)))


def crop_window(img, window):
    """Crop, padding with a neutral colour where the window leaves the image."""
    x0, y0, x1, y1 = window
    out = Image.new("RGB", (x1 - x0, y1 - y0), PAD_COLOR)
    out.paste(Image.fromarray(img) if isinstance(img, np.ndarray) else img, (-x0, -y0))
    return out


def _shift(pts, window):
    return np.atleast_2d(np.asarray(pts, dtype=np.float64)) - np.array([window[0], window[1]])


def _haloed(draw, xy, color, width, kind="line"):
    """Draw a dark underlay first so the stroke reads on any background."""
    if kind == "line":
        draw.line(xy, fill=HALO, width=width + 6)
        draw.line(xy, fill=color, width=width)
    else:
        draw.polygon(xy, fill=color, outline=HALO)


def _corners(img_size, box_w, box_h, margin=24):
    w, h = img_size
    return [(margin, margin), (w - box_w - margin, margin),
            (margin, h - box_h - margin), (w - box_w - margin, h - box_h - margin)]


def _pick_corner(img_size, box_w, box_h, avoid_pts, taken=(), margin=24):
    """Corner whose box overlaps the fewest of ``avoid_pts`` (the projected
    gripper), breaking ties by distance from those points."""
    pts = np.atleast_2d(np.asarray(avoid_pts, dtype=np.float64)) if len(avoid_pts) else np.zeros((0, 2))
    best, best_key = None, None
    for c in _corners(img_size, box_w, box_h, margin):
        if any(abs(c[0] - t[0]) < box_w and abs(c[1] - t[1]) < box_h for t in taken):
            continue
        if len(pts):
            inside = np.sum((pts[:, 0] >= c[0]) & (pts[:, 0] <= c[0] + box_w)
                            & (pts[:, 1] >= c[1]) & (pts[:, 1] <= c[1] + box_h))
            centre = np.array([c[0] + box_w / 2, c[1] + box_h / 2])
            key = (int(inside), -float(np.linalg.norm(pts - centre, axis=1).min()))
        else:
            key = (0, 0.0)
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best if best is not None else _corners(img_size, box_w, box_h, margin)[0]


def draw_gripper_2d(draw, camera, geom, color, width=9, window_box=(0, 0, 0, 0)):
    """Project a gripper's wireframe, finger tips and approach arrow.

    ``geom`` comes from the backend (``gripper.draw_geometry`` for FGC,
    ``graspgen_pose.draw_geometry`` for GraspGen) so this stays agnostic to
    which grasp model, gripper and frame convention produced the pose."""
    R, t = geom["R"], geom["t"]
    for a, b in geom["segments"]:
        pa, pb = _shift(camera.project(gripper.to_world([a, b], R, t)), window_box)
        _haloed(draw, [tuple(pa), tuple(pb)], color, width)

    # a dot on each finger tip so the closing axis is unambiguous
    for p in _shift(camera.project(gripper.to_world(geom["tips"], R, t)), window_box):
        draw.ellipse([p[0] - 11, p[1] - 11, p[0] + 11, p[1] + 11], fill=color, outline=HALO, width=3)

    # approach arrow: from behind the wrist to the grasp centre
    pa, pb = _shift(camera.project(gripper.to_world(geom["arrow"], R, t)), window_box)
    _dashed(draw, pa, pb, color, max(width - 4, 2), dash=22, gap=16, halo=True)
    d = pb - pa
    n = np.linalg.norm(d)
    if n > 1e-6:
        d = d / n
        perp = np.array([-d[1], d[0]])
        _haloed(draw, [tuple(pb), tuple(pb - d * 30 + perp * 15),
                       tuple(pb - d * 30 - perp * 15)], color, width, kind="poly")
    return _shift(camera.project(gripper.to_world(geom["segments"].reshape(-1, 3), R, t)), window_box)


def _target_mask(scene, target):
    if not target.instance_id:
        return np.zeros(scene.shape, bool)
    return scene.instances == target.instance_id


def _base_image(scene, mask, color, tint_alpha, outline_px):
    """Light tint plus a bold haloed contour: the object keeps its own colour,
    so it never blends into the method colour used for the gripper."""
    img = tint(scene.rgb, mask, color, alpha=tint_alpha)
    img = tint(img, mask_outline(mask, outline_px + 3), HALO, alpha=1.0)
    return tint(img, mask_outline(mask, outline_px), color, alpha=1.0)


def render_target(scene, target, gt_mask, out_path, window=None):
    """02: which object the method decided to grasp first."""
    color = METHOD_COLOR[target.method]
    mask = _target_mask(scene, target)
    window = window or compute_window(scene.rgb.shape[1::-1], [mask, gt_mask])
    pil = crop_window(_base_image(scene, mask, color, 0.20, 5), window)
    draw = ImageDraw.Draw(pil, "RGBA")
    draw_bbox_dashed(draw, gt_mask, GT_COLOR, 3, window, label="GT")
    if target.point:
        x, y = _shift([target.point], window)[0]
        draw.ellipse([x - 16, y - 16, x + 16, y + 16], fill=color, outline=HALO, width=4)
    text = "%s\ntarget id=%s  %s" % (METHOD_TITLE[target.method], target.instance_id, target.label)
    _label(draw, (24, 24), text, color, size=fit_font_size(draw, text, pil.size[0] - 64))
    pil.save(out_path)
    return pil


MIN_ZOOM = 1.15


def _zoom_inset(pil, centre, span_px, color, out_size=520, taken=()):
    """Magnified crop around the grasp.  At full-scene scale a small gripper is
    only a couple of hundred pixels wide, which is unreadable once panels are
    tiled.  A large gripper (GraspGen's Robotiq is 19.5 cm deep) already fills
    the frame, so the inset is skipped rather than zooming out."""
    w, h = pil.size
    src_half = int(np.clip(span_px * 0.9 + 60, 95, 460))
    if out_size / (2 * src_half) < MIN_ZOOM:
        return pil
    cx = int(np.clip(centre[0], src_half, w - src_half))
    cy = int(np.clip(centre[1], src_half, h - src_half))
    box = (cx - src_half, cy - src_half, cx + src_half, cy + src_half)
    crop = pil.crop(box).resize((out_size, out_size), Image.LANCZOS)

    # pick the destination corner before drawing anything, so a bail-out never
    # leaves an orphan source rectangle behind
    margin = 24
    free = [c for c in _corners((w, h), out_size, out_size, margin)
            if not any(abs(c[0] - t[0]) < out_size and abs(c[1] - t[1]) < out_size for t in taken)]
    if not free:
        return pil
    x0, y0 = max(free, key=lambda c: (c[0] + out_size / 2 - cx) ** 2
                                     + (c[1] + out_size / 2 - cy) ** 2)

    draw = ImageDraw.Draw(pil)
    src = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    for i in range(4):                                   # where the inset came from
        _dashed(draw, src[i], src[(i + 1) % 4], color, 3, dash=14, gap=10, halo=True)
    pil.paste(crop, (x0, y0))
    draw.rectangle([x0, y0, x0 + out_size - 1, y0 + out_size - 1], outline=color, width=6)
    _label(draw, (x0 + 14, y0 + 12), "zoom x%.1f" % (out_size / (2 * src_half)),
           color, size=25, pad=5)
    return pil


def render_grasp(scene, target, gt_mask, pose, stats, out_path, geom=None, window=None):
    """03: the grasp pose itself, drawn on the chosen object."""
    color = METHOD_COLOR[target.method]
    mask = _target_mask(scene, target)
    window = window or compute_window(scene.rgb.shape[1::-1], [mask, gt_mask])
    # a faint tint only -- the contour carries the identification, so a magenta
    # gripper never disappears into a magenta-tinted object
    pil = crop_window(_base_image(scene, mask, color, 0.10, 6), window)
    draw = ImageDraw.Draw(pil, "RGBA")
    draw_bbox_dashed(draw, gt_mask, GT_COLOR, 3, window, label="GT")

    projected = np.zeros((0, 2))
    if pose is not None:
        projected = draw_gripper_2d(draw, scene.camera, geom, color, window_box=window)
        aperture = pose.get("width", pose.get("gripper_aperture_m", 0.0))
        text = "%s\nid=%s %s\n%s  aperture=%.1f cm  score=%.3f" % (
            METHOD_TITLE[target.method], target.instance_id, target.label,
            pose.get("gripper", "graspnet gripper"), aperture * 100, pose["score"])
    else:
        text = "%s\nid=%s %s\nno grasp: %s" % (
            METHOD_TITLE[target.method], target.instance_id, target.label,
            stats.get("note", "n/a"))

    # keep the header off the gripper -- GraspGen's Robotiq spans most of the frame
    size = fit_font_size(draw, text, pil.size[0] - 64)
    tb = draw.multiline_textbbox((0, 0), text, font=font(size))
    label_wh = (min(tb[2] - tb[0] + 32, pil.size[0] - 48), tb[3] - tb[1] + 32)
    label_xy = _pick_corner(pil.size, label_wh[0], label_wh[1], projected)
    label_xy = (max(label_xy[0], 24), max(label_xy[1], 24))
    _label(draw, (label_xy[0] + 8, label_xy[1] + 8), text, color, size=size)

    if pose is not None:
        centre = _shift(scene.camera.project([geom["t"]]), window)[0]
        span = float(np.abs(projected - centre).max()) if len(projected) else 0.0
        _zoom_inset(pil, centre, span, color, taken=[label_xy])
    pil.save(out_path)
    return pil


def write_ply(path, points, colors):
    pts = np.asarray(points, dtype=np.float32)
    cols = (np.clip(np.asarray(colors), 0, 1) * 255).astype(np.uint8)
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(pts))
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for p, c in zip(pts, cols):
            fh.write("%f %f %f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))


def write_obj(path, verts, tris, color=(0.0, 1.0, 0.0)):
    """Gripper mesh, already transformed into camera coordinates."""
    mtl = os.path.splitext(path)[0] + ".mtl"
    with open(mtl, "w") as fh:
        fh.write("newmtl grasp\nKd %f %f %f\nKa 0 0 0\nKs 0 0 0\n" % color)
    with open(path, "w") as fh:
        fh.write("mtllib %s\nusemtl grasp\n" % os.path.basename(mtl))
        for v in np.asarray(verts):
            fh.write("v %f %f %f\n" % tuple(v))
        for t in np.asarray(tris):
            fh.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


def compose_compare(case, panels, out_path, panel_px=760):
    """One row of three method panels under a shared header."""
    order = [m for m in ("gemini_uoais_ref", "gemini_noref", "unograsp") if m in panels]
    head_h, pad, caption_h = 118, 18, 66
    w = panel_px * len(order) + pad * (len(order) + 1)
    h = head_h + panel_px + caption_h + pad * 2
    canvas = Image.new("RGB", (w, h), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)

    draw.text((pad + 6, 20), "%s  [%s]" % (case.key, case.difficulty),
              font=font(42), fill=(255, 255, 255))
    draw.text((pad + 6, 70), 'query: "%s"   GT top objects: %s   (dashed white = GT)'
              % (case.query, case.gt_top_ids), font=font(27), fill=(185, 185, 195))

    for i, method in enumerate(order):
        img, target, pose = panels[method]
        x = pad + i * (panel_px + pad)
        y = head_h + pad
        canvas.paste(img.resize((panel_px, panel_px), Image.LANCZOS), (x, y))
        edge = CORRECT_COLOR if target.correct else WRONG_COLOR
        draw.rectangle([x, y, x + panel_px - 1, y + panel_px - 1], outline=edge, width=7)
        verdict = "correct" if target.correct else "wrong"
        note = "grasp ok" if pose is not None else "no grasp"
        draw.text((x + 4, y + panel_px + 10),
                  "%s - id=%s (%s, %s)" % (METHOD_TITLE[method], target.instance_id, verdict, note),
                  font=font(29), fill=edge)
    canvas.save(out_path)


def draw_gripper_2d_points(camera, geom, include_arrow=True):
    """Where a gripper projects, without drawing it -- used to size the crop."""
    parts = [geom["segments"].reshape(-1, 3), geom["tips"]]
    if include_arrow:
        parts.append(geom["arrow"])
    return camera.project(gripper.to_world(np.concatenate(parts, axis=0), geom["R"], geom["t"]))


def render_clean(scene, geom, color, out_path, width=5, pad_margin=30):
    """FreeGrasp-style overlay: the original frame with nothing on it but the
    projected gripper wireframe.  No mask, no GT box, no caption.

    The frame is kept exactly as shot; it is only extended (never cropped) when
    a gripper reaches outside it, which GraspGen's 19.5 cm Robotiq often does."""
    h, w = scene.rgb.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h
    pts = None
    if geom is not None:
        pts = scene.camera.project(
            gripper.to_world(geom["segments"].reshape(-1, 3), geom["R"], geom["t"]))
        x0 = int(min(x0, np.floor(pts[:, 0].min()) - pad_margin))
        y0 = int(min(y0, np.floor(pts[:, 1].min()) - pad_margin))
        x1 = int(max(x1, np.ceil(pts[:, 0].max()) + pad_margin))
        y1 = int(max(y1, np.ceil(pts[:, 1].max()) + pad_margin))

    pil = crop_window(scene.rgb, (x0, y0, x1, y1))
    if geom is not None:
        draw = ImageDraw.Draw(pil)
        for a, b in geom["segments"]:
            pa, pb = _shift(scene.camera.project(
                gripper.to_world([a, b], geom["R"], geom["t"])), (x0, y0, x1, y1))
            draw.line([tuple(pa), tuple(pb)], fill=color, width=width)
    pil.save(out_path)
    return pil
