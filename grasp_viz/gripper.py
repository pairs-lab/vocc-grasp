"""Parallel-jaw gripper geometry, written from the GraspNet convention.

Local frame (the one a GraspNet rotation matrix maps to camera coordinates):
    +x  approach direction (the way the gripper dives in)
    +y  closing axis (the fingers sit at y = +-width/2)
    +z  finger height

Dimensions match the ones FreeGrasp's ``get_correct_pose`` uses so the meshes
are directly comparable.
"""
import numpy as np

HEIGHT = 0.004
FINGER_WIDTH = 0.004
TAIL_LENGTH = 0.04
DEPTH_BASE = 0.02

# unit cube corners, spanning [0,1]^3
_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)
_CUBE_TRIS = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                       [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                       [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)


def _box(size, origin):
    return _CUBE * np.asarray(size, dtype=np.float64) + np.asarray(origin, dtype=np.float64)


def gripper_boxes(width, depth):
    """Four boxes -- two fingers, the bridge between them, and the tail.

    Returns (vertices (N,3), triangles (M,3)) in the local frame."""
    w, d = float(width), float(depth)
    boxes = [
        # left finger
        _box((d + DEPTH_BASE + FINGER_WIDTH, FINGER_WIDTH, HEIGHT),
             (-DEPTH_BASE - FINGER_WIDTH, -w / 2 - FINGER_WIDTH, -HEIGHT / 2)),
        # right finger
        _box((d + DEPTH_BASE + FINGER_WIDTH, FINGER_WIDTH, HEIGHT),
             (-DEPTH_BASE - FINGER_WIDTH, w / 2, -HEIGHT / 2)),
        # bridge joining the fingers
        _box((FINGER_WIDTH, w + 2 * FINGER_WIDTH, HEIGHT),
             (-DEPTH_BASE - FINGER_WIDTH, -w / 2 - FINGER_WIDTH, -HEIGHT / 2)),
        # tail / wrist
        _box((TAIL_LENGTH, FINGER_WIDTH, HEIGHT),
             (-DEPTH_BASE - FINGER_WIDTH - TAIL_LENGTH, -FINGER_WIDTH / 2, -HEIGHT / 2)),
    ]
    verts = np.concatenate(boxes, axis=0)
    tris = np.concatenate([_CUBE_TRIS + 8 * i for i in range(len(boxes))], axis=0)
    return verts, tris


def wireframe(width, depth):
    """Line segments (K,2,3) that read clearly once projected to 2D."""
    w, d = float(width), float(depth)
    back = -DEPTH_BASE
    tail = back - TAIL_LENGTH
    lt, rt = (d, -w / 2, 0.0), (d, w / 2, 0.0)          # finger tips
    lb, rb = (back, -w / 2, 0.0), (back, w / 2, 0.0)    # finger bases
    return np.array([
        [lt, lb],            # left finger
        [rt, rb],            # right finger
        [lb, rb],            # bridge
        [(back, 0.0, 0.0), (tail, 0.0, 0.0)],  # tail
    ], dtype=np.float64)


def to_world(local_pts, rotation, translation):
    """Local frame -> camera frame."""
    pts = np.asarray(local_pts, dtype=np.float64).reshape(-1, 3)
    out = pts @ np.asarray(rotation, dtype=np.float64).T + np.asarray(translation, dtype=np.float64)
    return out.reshape(np.asarray(local_pts).shape)


def refined_pose(rotation, translation, width, depth):
    """FreeGrasp's ``get_correct_pose``: re-derive a frame from the finger
    geometry.  The result has **+z as the approach direction** (the convention
    the exported grasp_pose.json uses), unlike the GraspNet matrix where the
    approach is +x."""
    w, d = float(width), float(depth)
    R = np.asarray(rotation, dtype=np.float64)
    c = np.asarray(translation, dtype=np.float64)

    bridge_mid = np.array([-DEPTH_BASE - FINGER_WIDTH / 2, 0.0, 0.0])
    left_tip = np.array([d, -w / 2 - FINGER_WIDTH / 2, 0.0])
    right_tip = np.array([d, w / 2 + FINGER_WIDTH / 2, 0.0])

    up_point = R @ bridge_mid + c
    left_point = R @ left_tip + c
    right_point = R @ right_tip + c
    center = (left_point + right_point) / 2.0

    vz = center - up_point
    vz /= np.linalg.norm(vz)
    vx = center - left_point
    vx /= np.linalg.norm(vx)
    vy = np.cross(vz, vx)
    vy /= np.linalg.norm(vy)
    return np.column_stack((vx, vy, vz)), center


def draw_geometry(pose):
    """Segments, finger tips and approach arrow in the camera frame, for the
    GraspNet-convention pose the FGC backend produces (+x = approach)."""
    R = np.asarray(pose["graspnet_rotation"], dtype=np.float64)
    t = np.asarray(pose["graspnet_translation"], dtype=np.float64)
    w, d = pose["width"], pose["depth"]
    return {
        "R": R,
        "t": t,
        "segments": wireframe(w, d),
        "tips": np.array([[d, -w / 2, 0.0], [d, w / 2, 0.0]]),
        "arrow": np.array([[-DEPTH_BASE - TAIL_LENGTH - 0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    }
