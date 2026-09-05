"""Render FGC-GraspNet grasp poses for the objects each reasoning method chose.

    python -m grasp_viz.run_grasp_viz --all
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

from . import cases as C
from . import gripper, paths, pipeline, real_data, render
from . import scene as S


class _Backend:
    """One grasp model: how to predict, how to serialise, how to draw."""

    def __init__(self, name="fgc", max_width=None, gripper_cfg=None):
        self.name = name
        self.max_width = max_width
        self.gripper_cfg = gripper_cfg
        self._ap = None
        if gripper_cfg is not None:
            from . import aperture
            self._ap = aperture.from_config(gripper_cfg)   # (finger_len, finger_width, cap)
        from . import fgc_infer
        self._infer = fgc_infer

    def best_grasp(self, scene, instance_id, mask, points):
        """-> (pose dict or None, stats)."""
        gg = self._infer.predict(points, scene=scene, instance_id=instance_id)
        ap_fn, cap = None, None
        if self._ap is not None:
            from . import aperture
            flen, fwid, cap = self._ap
            obj = aperture.object_points(scene, instance_id)
            ap_fn = lambda g: aperture.measure_many(
                obj, pipeline.rotations(g), g[:, pipeline.C_TRANS], flen, fwid)
        row, stats = pipeline.best_grasp(gg, scene.camera, mask, points,
                                         max_width=self.max_width,
                                         aperture_fn=ap_fn, max_aperture=cap)
        if row is None:
            return None, stats
        return pipeline.grasp_to_dict(row, self.name), stats

    def draw_geometry(self, pose):
        return gripper.draw_geometry(pose)

    def mesh(self, pose):
        """(vertices, triangles) of the gripper in camera coordinates."""
        verts, tris = gripper.gripper_boxes(pose["width"], pose["depth"])
        geom = self.draw_geometry(pose)
        return gripper.to_world(verts, geom["R"], geom["t"]), tris


def process_case(case, methods, backend, debug=False, dataset=paths.SYNTHETIC):
    sc = (real_data.load_scene(case.image_id) if dataset == paths.REAL
          else S.load_scene(case.image_id))
    gt_mask = np.isin(sc.instances, case.gt_top_ids)
    out_root = paths.out_dirs(backend.name, dataset)

    # pass 1: solve every method first, so all three panels can share one crop
    solved = {}
    for method in methods:
        target = case.targets[method]
        pose, geom, stats = None, None, {"note": "no object under predicted point"}
        pts = cols = None
        mask = render._target_mask(sc, target)
        if target.instance_id:
            region = S.crop_region(sc, mask)
            pts, cols = S.sample_cloud(sc, region)
            pose, stats = backend.best_grasp(sc, target.instance_id, mask, pts)
            if pose is not None:
                geom = backend.draw_geometry(pose)
        solved[method] = (target, mask, pose, geom, stats, pts, cols)
        if debug:
            print("   %-18s id=%-2s correct=%-5s %s" % (method, target.instance_id,
                                                        target.correct, stats))

    # one window covering the targets, the GT objects and every gripper, so the
    # panels line up and no gripper is clipped
    masks = [gt_mask] + [v[1] for v in solved.values()]
    extra = [render.draw_gripper_2d_points(sc.camera, v[3], include_arrow=False) for v in solved.values() if v[3]]
    extra = np.concatenate(extra, axis=0) if extra else []
    window = render.compute_window(sc.rgb.shape[1::-1], masks, extra)

    # pass 2: render
    panels, summary = {}, {}
    for method in methods:
        target, mask, pose, geom, stats, pts, cols = solved[method]
        out_dir = os.path.join(out_root[method], case.key)
        os.makedirs(out_dir, exist_ok=True)

        render.crop_window(sc.rgb, window).save(os.path.join(out_dir, "01_rgb.png"))
        render.render_target(sc, target, gt_mask,
                             os.path.join(out_dir, "02_target_mask.png"), window)
        if pts is not None:
            render.write_ply(os.path.join(out_dir, "cloud.ply"), pts, cols)
        if pose is not None:
            verts, tris = backend.mesh(pose)
            render.write_obj(os.path.join(out_dir, "grasp.obj"), verts, tris)

        img = render.render_grasp(sc, target, gt_mask, pose, stats,
                                  os.path.join(out_dir, "03_grasp_pose.png"), geom, window)
        if pose is not None:
            render.render_clean(sc, geom, render.METHOD_COLOR[method],
                                os.path.join(out_dir, "04_grasp_clean.png"))

        record = {
            "case": case.key, "image_id": case.image_id, "difficulty": case.difficulty,
            "query": case.query, "method": method, "target_id": target.instance_id,
            "target_label": target.label, "predicted_point": list(target.point or []),
            "gt_top_ids": case.gt_top_ids, "target_correct": target.correct,
            "grasp_found": pose is not None, "stats": stats,
        }
        if pose:
            record.update(pose)
        with open(os.path.join(out_dir, "grasp_pose.json"), "w") as fh:
            json.dump(record, fh, indent=2)

        panels[method] = (img, target, pose)
        summary[method] = {"target_id": target.instance_id, "correct": target.correct,
                           "grasp_found": pose is not None, **stats}

    compare = paths.compare_dir(backend.name, dataset)
    os.makedirs(compare, exist_ok=True)
    if len(panels) > 1:
        render.compose_compare(case, panels, os.path.join(compare, case.key + ".png"))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="all selected cases and methods")
    ap.add_argument("--cases", nargs="*", help="restrict to these case keys")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--backend", choices=["fgc"], default="fgc",
                    help="grasp model; only FGC-GraspNet is shipped")
    ap.add_argument("--dataset", choices=[paths.SYNTHETIC, paths.REAL],
                    default=paths.SYNTHETIC, help="which capture set to render")
    ap.add_argument("--n-medium", type=int, default=7, help="real dataset only")
    ap.add_argument("--n-hard", type=int, default=8, help="real dataset only")
    ap.add_argument("--max-width", type=float, default=None,
                    help="metres; drop grasps wider than the real jaws. "
                         "Pass 'config' to read grasp_viz/gripper_tho_pa1.yaml")
    ap.add_argument("--gripper-config", action="store_true",
                    help="take --max-width from the gripper config")
    ap.add_argument("--debug", action="store_true")
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
    if not selected:
        print("no cases matched", file=sys.stderr)
        return 1

    max_width = args.max_width
    gcfg = None
    if args.gripper_config:
        from . import gripper_config
        gcfg = gripper_config.load()
    backend = _Backend(args.backend, max_width=max_width, gripper_cfg=gcfg)
    summary = {}
    for case in selected:
        print("[%s] %s  query=%r" % (case.difficulty, case.key, case.query))
        summary[case.key] = {
            "difficulty": case.difficulty, "query": case.query,
            "gt_top_ids": case.gt_top_ids,
            "methods": process_case(case, args.methods, backend, args.debug,
                                    args.dataset),
        }

    compare = paths.compare_dir(args.backend, args.dataset)
    os.makedirs(compare, exist_ok=True)
    out = os.path.join(compare, "summary.json")
    with open(out, "w") as fh:
        json.dump({"backend": args.backend, "dataset": args.dataset,
                   "cases": summary}, fh, indent=2)
    print("\nwrote %s" % out)
    for m in args.methods:
        n_ok = sum(1 for c in summary.values() if c["methods"][m]["correct"])
        n_g = sum(1 for c in summary.values() if c["methods"][m]["grasp_found"])
        print("  %-18s target correct %d/%d   grasp found %d/%d"
              % (m, n_ok, len(summary), n_g, len(summary)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
