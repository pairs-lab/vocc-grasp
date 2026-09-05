"""FreeGrasp's grasp post-processing, reimplemented on plain numpy.

Order matches ``grasp_model.forward``:
    network -> keep grasps centred on the target mask -> drop colliding grasps
    -> keep top-down grasps only -> sort by score.
"""
import numpy as np

from . import gripper
from .fgc_infer import C_DEPTH, C_HEIGHT, C_ROT, C_SCORE, C_TRANS, C_WIDTH

# ModelFreeCollisionDetector's fixed gripper envelope
COLLISION_FINGER_WIDTH = 0.01
COLLISION_FINGER_LENGTH = 0.06
VOXEL_SIZE = 0.01
COLLISION_THRESH = 0.01
APPROACH_DIST = 0.05

TOP_DOWN_ANGLE = np.pi / 6      # check_grasp keeps grasps within 30 deg of vertical
TOP_DOWN_SCORE_FACTOR = 0.3


def rotations(gg):
    return gg[:, C_ROT].reshape(-1, 3, 3)


def translations(gg):
    return gg[:, C_TRANS]


def choose_in_mask(gg, camera, mask):
    """Keep grasps whose centre projects inside the target's mask."""
    if len(gg) == 0:
        return gg
    t = translations(gg)
    valid = np.abs(t[:, 2]) > 1e-9
    uv = np.rint(camera.project(t)).astype(int)
    h, w = mask.shape
    inside = valid & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    keep = np.zeros(len(gg), dtype=bool)
    idx = np.nonzero(inside)[0]
    keep[idx] = mask[uv[idx, 1], uv[idx, 0]]
    return gg[keep]


def voxel_downsample(points, voxel_size=VOXEL_SIZE):
    if len(points) == 0:
        return points
    keys = np.floor(np.asarray(points) / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.asarray(points)[np.sort(first)]


def collision_mask(gg, scene_points, voxel_size=VOXEL_SIZE,
                   approach_dist=APPROACH_DIST, collision_thresh=COLLISION_THRESH):
    """True where the gripper envelope would hit the scene cloud."""
    if len(gg) == 0:
        return np.zeros(0, dtype=bool)
    fw, fl = COLLISION_FINGER_WIDTH, COLLISION_FINGER_LENGTH
    approach_dist = max(approach_dist, fw)
    pts = voxel_downsample(scene_points, voxel_size)

    T = translations(gg)
    R = rotations(gg)
    heights = gg[:, C_HEIGHT][:, None]
    depths = gg[:, C_DEPTH][:, None]
    widths = gg[:, C_WIDTH][:, None]

    out = np.zeros(len(gg), dtype=bool)
    step = 64                                    # chunked: (M,N,3) gets large fast
    for s in range(0, len(gg), step):
        e = min(s + step, len(gg))
        tgt = pts[None, :, :] - T[s:e, None, :]
        tgt = np.matmul(tgt, R[s:e])             # into each gripper's local frame
        x, y, z = tgt[:, :, 0], tgt[:, :, 1], tgt[:, :, 2]
        hh, dd, ww = heights[s:e], depths[s:e], widths[s:e]

        in_height = (z > -hh / 2) & (z < hh / 2)
        in_fingers = (x > dd - fl) & (x < dd)
        outer_l = y > -(ww / 2 + fw)
        inner_l = y < -ww / 2
        outer_r = y < (ww / 2 + fw)
        inner_r = y > ww / 2
        in_bottom = (x <= dd - fl) & (x > dd - fl - fw)
        in_shift = (x <= dd - fl - fw) & (x > dd - fl - fw - approach_dist)

        left = in_height & in_fingers & outer_l & inner_l
        right = in_height & in_fingers & outer_r & inner_r
        bottom = in_height & outer_l & outer_r & in_bottom
        shift = in_height & outer_l & outer_r & in_shift
        hit = (left | right | bottom | shift).sum(axis=1)

        lr_vol = (hh * fl * fw / voxel_size ** 3).reshape(-1)
        bt_vol = (hh * (ww + 2 * fw) * fw / voxel_size ** 3).reshape(-1)
        sh_vol = (hh * (ww + 2 * fw) * approach_dist / voxel_size ** 3).reshape(-1)
        volume = lr_vol * 2 + bt_vol + sh_vol
        out[s:e] = (hit / (volume + 1e-6)) > collision_thresh
    return out


def check_top_down(gg, max_angle=TOP_DOWN_ANGLE, score_factor=TOP_DOWN_SCORE_FACTOR):
    """FreeGrasp's ``check_grasp``: keep grasps approaching from above, and only
    if the surviving score spread is wide enough to be meaningful."""
    if len(gg) == 0:
        return gg, "no grasps"
    R = rotations(gg)
    grasp_vec = np.einsum("nij,j->ni", R, np.array([-1.0, 0.0, 0.0]))
    angle = np.arccos(np.clip(grasp_vec @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))
    keep = angle <= max_angle
    if not keep.any():
        # report how far off the best candidate was: "rejected" and "rejected by
        # 3 degrees" call for very different responses, and the bare message
        # gave no way to tell them apart
        return gg[keep], ("no top-down grasp within %.0f deg (closest %.0f)"
                          % (np.degrees(max_angle), np.degrees(angle.min())))

    if score_factor is None:          # veto disabled outright
        return gg[keep], "select top-down"

    scores = gg[keep, C_SCORE]
    ref_value, ref_min = scores.max(), scores.min()
    shifted = scores - ref_min
    # NB: with a single surviving grasp shifted.max() is 0, so the veto still
    # fires even at score_factor=0 -- pass score_factor=None to truly bypass it.
    if shifted.max() > ref_value * score_factor:
        return gg[keep], "select top-down"
    return gg[np.zeros(len(gg), dtype=bool)], "no suitable grasp found"


_UNSET = object()


def best_grasp(gg, camera, mask, scene_points, debug=None, max_width=None,
               aperture_fn=None, max_aperture=None, top_down_angle=None,
               score_factor=_UNSET, pose_ok=None):
    """Full chain. Returns (best_row_or_None, stats dict).

    ``max_width`` (metres) drops grasps whose predicted width exceeds the jaws.
    ``aperture_fn`` + ``max_aperture`` apply the stricter, backend-neutral test
    instead: the measured object extent along the closing axis (see aperture.py).
    Either way the constraint has to be applied during selection -- checking the
    top-scoring grasp afterwards leaves most cases unexecutable even when a
    fitting candidate exists.
    """
    stats = {"raw": int(len(gg))}
    gg = choose_in_mask(gg, camera, mask)
    stats["in_mask"] = int(len(gg))

    if len(gg):
        gg = gg[~collision_mask(gg, scene_points)]
    stats["collision_free"] = int(len(gg))

    gg, note = check_top_down(
        gg,
        max_angle=TOP_DOWN_ANGLE if top_down_angle is None else top_down_angle,
        # None is a real value here -- it disables the veto -- so it cannot
        # double as "argument not given"; that collision silently restored the
        # default 0.3 and kept rejecting grasps callers had asked to allow.
        score_factor=(TOP_DOWN_SCORE_FACTOR if score_factor is _UNSET
                      else score_factor))
    stats["top_down"] = int(len(gg))
    stats["note"] = note

    if max_width is not None and len(gg):
        gg = gg[gg[:, C_WIDTH] <= max_width]
        stats["width_ok"] = int(len(gg))
        if len(gg) == 0:
            stats["note"] = "no grasp within %.0f mm jaw span" % (max_width * 1000)

    if len(gg) == 0:
        return None, stats
    gg = gg[np.argsort(-gg[:, C_SCORE])]
    if aperture_fn is not None:
        need = aperture_fn(gg)
        keep = need <= max_aperture
        stats["aperture_ok"] = int(keep.sum())
        if not keep.any():
            stats["note"] = ("no grasp fits %.0f mm jaws (closest %.0f mm)"
                             % (max_aperture * 1000, np.min(need) * 1000))
            return None, stats
        gg = gg[keep]
        stats["required_aperture"] = float(need[keep][0])

    if pose_ok is None:
        return gg[0], stats
    # Same lesson as the aperture constraint above: a caller-side veto applied
    # to the winner alone throws away cases where a lower-scoring candidate
    # satisfies it, and leaves the caller re-deciding identically every round.
    for i in range(len(gg)):
        ok, why = pose_ok(gg[i])
        if ok:
            stats["pose_ok_rank"] = i
            if aperture_fn is not None:
                # report the winner's aperture, not the top-ranked row's --
                # callers open the jaws to this number
                stats["required_aperture"] = float(need[keep][i])
            return gg[i], stats
        stats["note"] = why
    stats["pose_ok_rank"] = None
    return None, stats


def grasp_to_dict(row, backend):
    """Export in FreeGrasp's grasp_pose.json shape.

    ``rotation`` follows ``get_correct_pose`` (+z = approach); the raw GraspNet
    matrix (+x = approach) is kept alongside so the mesh can be rebuilt.
    """
    R = row[C_ROT].reshape(3, 3)
    t = row[C_TRANS]
    width = float(row[C_WIDTH])
    depth = float(row[C_DEPTH])
    R_ref, center = gripper.refined_pose(R, t, width, depth)
    return {
        "translation": center.tolist(),
        "rotation": R_ref.tolist(),
        "width": width,
        "depth": depth,
        "height": float(row[C_HEIGHT]),
        "score": float(row[C_SCORE]),
        "backend": backend,
        "graspnet_rotation": R.tolist(),
        "graspnet_translation": t.tolist(),
        "frame": "camera; rotation column 2 = approach, column 0 = closing axis",
    }
