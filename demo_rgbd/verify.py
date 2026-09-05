#!/usr/bin/env python3
"""Self-check for the demo bundle -- two independent levels.

    python verify.py          # LEVEL 1: data integrity   (numpy + opencv, ~5 s)
    python verify.py --run    # LEVEL 2: replay the grasp stage against expected/

LEVEL 1 needs no GPU, no model and no network. It opens every file and checks the
data is exactly what the README claims -- shape, dtype, value range, and that the
object ids in `target.json` really exist in `instances.png` -- then verifies sha256.

LEVEL 2 needs the FGC-GraspNet checkpoint and a CUDA GPU, but no API key: it takes
the target object straight from `expected/<case>.json` and replays the geometry half
of the pipeline (mask -> point cloud -> FGC -> candidate set), then checks the recorded
pose is in the candidate set within 1 mm / 1 deg.

It checks membership rather than the top-1 pose on purpose. `expected/` was produced by
the deployment package, whose gripper model differs from grasp_viz/gripper_tho_pa1.yaml,
so the two rank the surviving candidates differently. What is reproducible -- and what
this level asserts -- is that the same scene yields the same grasp candidates.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOL_MM = 1.0
TOL_DEG = 1.0
# A two-finger gripper is symmetric: rotating 180 deg about the approach axis is the
# same physical grasp with the fingers swapped. Without accounting for it, two
# identical poses can report a 180 deg error.
FLIP = np.diag([-1.0, -1.0, 1.0])


def cases():
    return sorted(d for d in os.listdir(HERE)
                  if d.startswith("img") and os.path.isdir(os.path.join(HERE, d)))


# ───────────────────────────────────────────────────────────────── LEVEL 1
def check_integrity():
    import cv2

    bad = []
    for c in cases():
        d = os.path.join(HERE, c)
        rgb = cv2.imread(os.path.join(d, "frame_rgb.png"))
        dep = cv2.imread(os.path.join(d, "frame_depth.png"), cv2.IMREAD_UNCHANGED)
        ins = cv2.imread(os.path.join(d, "instances.png"), cv2.IMREAD_UNCHANGED)
        k = json.load(open(os.path.join(d, "intrinsics.json")))
        t = json.load(open(os.path.join(d, "target.json")))

        def bad_if(cond, msg):
            if cond:
                bad.append("%s: %s" % (c, msg))

        bad_if(rgb is None or rgb.shape != (1200, 1200, 3), "frame_rgb is not 1200x1200x3")
        bad_if(dep is None or dep.dtype != np.uint16, "frame_depth is not uint16")
        bad_if(ins is None or ins.dtype != np.uint16, "instances is not uint16")
        if dep is None or ins is None:
            continue
        bad_if(dep.shape != (1200, 1200), "depth has the wrong shape")
        bad_if(ins.shape != (1200, 1200), "instances has the wrong shape")
        # depth 0 means a dead sensor pixel, NOT a surface at 0 m
        valid = dep[dep > 0]
        bad_if(valid.size == 0, "depth is all zero")
        if valid.size:
            bad_if(not (200 <= valid.min() and valid.max() <= 2000),
                   "depth outside 0.2-2.0 m (%d..%d mm)" % (valid.min(), valid.max()))
        ids = {int(x) for x in np.unique(ins)} - {0}
        bad_if(set(int(i) for i in t["instances"]) != ids,
               "target.json lists %s but instances.png holds %s"
               % (sorted(int(i) for i in t["instances"]), sorted(ids)))
        bad_if(t["reference_run"]["target_id"] not in ids,
               "target_id %s is absent from instances.png" % t["reference_run"]["target_id"])
        bad_if(not all(g in ids for g in t["ground_truth"]["top_ids"]),
               "gt top_ids are absent from instances.png")
        bad_if(k["cx"] <= 0 or k["fx"] <= 0, "implausible intrinsics")

    n_hash, bad_hash = 0, []
    path = os.path.join(HERE, "manifest.sha256")   # covers the DATA only, not the code
    if os.path.isfile(path):
        for line in open(path):
            want, rel = line.strip().split("  ", 1)
            f = os.path.join(HERE, rel)
            if not os.path.isfile(f):
                bad_hash.append("missing " + rel)
                continue
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            n_hash += 1
            if h.hexdigest() != want:
                bad_hash.append("content differs " + rel)

    print("LEVEL 1  data integrity")
    print("  %d cases checked for shape/dtype/range/labels -> %s"
          % (len(cases()), "PASS" if not bad else "%d FAILED" % len(bad)))
    for b in bad:
        print("    " + b)
    print("  %d files verified against sha256 -> %s"
          % (n_hash, "PASS" if not bad_hash else "%d FAILED" % len(bad_hash)))
    for b in bad_hash[:10]:
        print("    " + b)
    return not bad and not bad_hash


# ───────────────────────────────────────────────────────────────── LEVEL 2
def _ang(Ra, Rb):
    one = lambda A, B: float(np.degrees(np.arccos(
        np.clip((np.trace(A.T @ B) - 1) / 2.0, -1.0, 1.0))))
    return min(one(Ra, Rb), one(Ra, Rb @ FLIP))


def check_run(limit=None):
    """Replay mask -> point cloud -> FGC -> pose and compare with expected/."""
    sys.path.insert(0, REPO)
    import cv2
    from PIL import Image
    from grasp_viz import pipeline, scene as S

    if not os.path.isfile(os.path.join(REPO, "FreeGrasp_code/logs/checkpoint_fgc.tar")):
        print("LEVEL 2  skipped: FreeGrasp_code/logs/checkpoint_fgc.tar is missing\n"
              "         run scripts/download_checkpoints.sh fgc")
        return False
    from grasp_viz import fgc_infer

    todo = cases()[:limit] if limit else cases()
    ok = fail = skip = 0
    print("LEVEL 2  replaying the grasp stage on %d case(s)" % len(todo))
    for c in todo:
        d = os.path.join(HERE, c)
        exp_path = os.path.join(HERE, "expected", c + ".json")
        if not os.path.isfile(exp_path):
            skip += 1
            continue
        exp = json.load(open(exp_path))
        ref = exp.get("fgc_default") or {}
        if not ref.get("grasp_found"):
            skip += 1
            continue

        k = json.load(open(os.path.join(d, "intrinsics.json")))
        rgb = np.asarray(Image.open(os.path.join(d, "frame_rgb.png")).convert("RGB"))
        dep = cv2.imread(os.path.join(d, "frame_depth.png"), cv2.IMREAD_UNCHANGED)
        ins = cv2.imread(os.path.join(d, "instances.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        depth_m = dep.astype(np.float64) * float(k.get("depth_scale", 0.001))
        cam = S.Camera(width=depth_m.shape[1], height=depth_m.shape[0],
                       fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"])
        sc = S.Scene(0, rgb, depth_m, ins, cam, cloud=cam.backproject(depth_m))

        tid = int(exp["target_id"])
        mask = S.instance_mask(sc, tid)
        pts, _cols = S.sample_cloud(sc, S.crop_region(sc, mask))
        gg = fgc_infer.predict(pts, scene=sc, instance_id=tid)
        # the full surviving candidate set, ranked -- not just the winner
        keep = pipeline.choose_in_mask(gg, sc.camera, mask)
        keep = keep[~pipeline.collision_mask(keep, pts)]
        cand, _note = pipeline.check_top_down(keep)
        if len(cand) == 0:
            print("  %-16s FAIL  no candidate survived (expected one)" % c)
            fail += 1
            continue
        cand = cand[np.argsort(-cand[:, pipeline.C_SCORE])]

        want_t = np.array(ref["translation_cam"])
        want_R = np.array(ref["rotation_cam"])
        best, best_i, best_a = 1e9, -1, 1e9
        for i in range(len(cand)):
            pose = pipeline.grasp_to_dict(cand[i], "fgc")
            dt = 1000.0 * float(np.linalg.norm(np.array(pose["translation"]) - want_t))
            if dt < best:
                best, best_i = dt, i
                best_a = _ang(np.array(pose["rotation"]), want_R)
        good = best <= TOL_MM and best_a <= TOL_DEG
        print("  %-16s %s  %6.3f mm  %6.3f deg   rank %d of %d"
              % (c, "ok  " if good else "FAIL", best, best_a, best_i + 1, len(cand)))
        ok, fail = (ok + 1, fail) if good else (ok, fail + 1)

    print("  %d matched, %d failed, %d skipped (no reference grasp)" % (ok, fail, skip))
    return fail == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="also replay the grasp stage")
    ap.add_argument("--limit", type=int, help="only the first N cases in LEVEL 2")
    a = ap.parse_args()
    good = check_integrity()
    if a.run:
        print()
        good = check_run(a.limit) and good
    sys.exit(0 if good else 1)
