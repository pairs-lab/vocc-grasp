"""How wide the jaws must open to close on the target at a given pose.

Backend-neutral, which is the point: FGC predicts a per-grasp ``width`` that
carries ~22 mm of its own approach clearance, and GraspGen reports a Robotiq
2f-140 constant regardless of the object.  Neither number describes the object,
so neither can be compared against real hardware.  This measures the object
instead -- its extent along the grasp's closing axis, inside the finger
footprint -- and that is what the THO/PA1 jaws actually have to span.
"""
import numpy as np


def required_aperture(cloud_local):
    """Extent along x of points already expressed in the gripper frame."""
    if len(cloud_local) < 10:
        return np.inf
    return float(cloud_local[:, 0].max() - cloud_local[:, 0].min())


def object_points(scene, instance_id):
    obj = scene.cloud[scene.instances == int(instance_id)]
    return obj[np.isfinite(obj).all(axis=1)]


def measure(obj, R, t, finger_len, finger_width, tcp_z=0.0):
    """Required aperture for one pose.  ``R`` columns: x closing, z approach.

    ``tcp_z`` shifts the finger window along the approach axis -- FGC anchors a
    pose at the grasp centre, GraspGen at the gripper base one depth behind the
    tips, so the window has to be centred on the tool centre point either way.
    """
    if len(obj) == 0:
        return np.inf
    local = (obj - t) @ R
    inside = (np.abs(local[:, 1]) <= finger_width / 2.0) & \
             (np.abs(local[:, 2] - tcp_z) <= finger_len)
    return required_aperture(local[inside])


def measure_many(obj, Rs, ts, finger_len, finger_width, tcp_z=0.0):
    """Vectorised over poses; returns (N,) metres, inf where the pose misses."""
    out = np.full(len(Rs), np.inf)
    for i in range(len(Rs)):
        out[i] = measure(obj, Rs[i], ts[i], finger_len, finger_width, tcp_z)
    return out


def from_config(cfg):
    """-> (finger_len, finger_width, max_aperture) in metres."""
    f = cfg["fingers"]
    gf = cfg.get("grasp_filter", {})
    cap = float(gf.get("max_grasp_width", cfg["width"]))
    return (float(f["clamping_face_length"]), float(f["clamping_face_width"]),
            cap - float(gf.get("width_margin", 0.0)))
