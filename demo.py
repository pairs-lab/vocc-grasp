#!/usr/bin/env python3
"""Full pipeline on one RGB-D frame: which object to clear first, and how to grasp it.

    python demo.py --case img000059_q4
    python demo.py --rgb my.png --depth my.png --intrinsics k.json --request "the white box"

Stages: Gemini points at every object -> UOAIS predicts amodal masks and scores the
occlusion geometry -> Gemini reasons over that evidence and names the object to remove
first -> FGC-GraspNet proposes a 6-DoF grasp on it -> everything is rendered.

Outputs match grasp_viz_real/gemini_uoais_ref: 01_rgb, 02_target_mask, 03_grasp_pose,
04_grasp_clean, 05_orbit.gif, cloud.ply, grasp.obj, grasp_pose.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_rgbd"

# uoais_obstruction thresholds the published runs used. run_gemini_uoais_ref.run()
# normally injects these; a direct caller has to, or edge scoring silently differs.
UOAIS_GLOBALS = dict(DEPTH_TOLERANCE_MM=90.0, CONTACT_DILATE_PX=8,
                     MIN_CONTACT_PIXELS=1, MIN_VALID_DEPTH_RATIO=0.5)
UOAIS_SCORE_THRESH = 0.35
UOAIS_NMS_THRESH = 0.7
MAX_NEAREST_PX = 120.0
MAX_REF_EDGES = 40
MIN_REF_CONTACT_PX = 1

EDGE_SCORING = dict(
    depth_norm_mode="percentile", edge_scoring="v2",
    v2_min_hidden_px=50, v2_min_explained_frac=0.05, v2_abs_contact_px=150,
    v2_keep_clearance=False, v2_depth_weight=0.5, v2_depth_sigma_mm=60.0,
    v2_depth_rescue_mm=30.0, v2_rescue_min_hidden_px=0,
)


def log(i, msg):
    print("\033[1;36m[%d/5]\033[0m %s" % (i, msg), flush=True)


# --------------------------------------------------------------------- input

def load_case(case):
    """One of the 28 bundled scenes."""
    d = DEMO / case
    if not d.is_dir():
        raise SystemExit("no such case: %s\navailable: %s" % (
            case, ", ".join(sorted(p.name for p in DEMO.glob("img*")))))
    tgt = json.loads((d / "target.json").read_text(encoding="utf-8"))
    return (d / "frame_rgb.png", d / "frame_depth.png", d / "intrinsics.json",
            tgt["request"], d / "instances.png", tgt)


def load_depth_mm(path, intr):
    """Depth in millimetres.

    A 16-bit PNG is raw sensor counts scaled by `depth_scale` metres per count, the
    convention RealSense and the MetaGraspNet captures use. .npy/.npz are taken as-is
    and unit-sniffed, since they carry no scale.
    """
    p = Path(path)
    if p.suffix in (".npy", ".npz"):
        d = np.load(p)
        if p.suffix == ".npz":
            d = d["depth"] if "depth" in d else d[list(d.keys())[0]]
        d = np.asarray(d, np.float32)
        hi = float(np.percentile(d[np.isfinite(d) & (d > 0)], 99))
        return d * (1000.0 if hi < 20 else 10.0 if hi < 500 else 1.0)
    d = np.asarray(Image.open(p)).astype(np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    return d * float(intr.get("depth_scale", 0.001)) * 1000.0


# --------------------------------------------------------------------- stages

def detect(rgb_path, out):
    import gemini_client as base
    import unobench_gemini_common as legacy
    pil = Image.open(rgb_path).convert("RGB")
    objects = legacy.detect_objects(Path(rgb_path).read_bytes(), pil.size)
    if not objects:
        raise SystemExit("Gemini returned no objects")
    json.dump(objects, open(out / "det_objects.json", "w", encoding="utf-8"), indent=2)
    return objects, base.draw_labeled_image(np.array(pil), objects, out / "labeled.png")


def segment(rgb, depth_mm, objects, predictor, cfg, args):
    """UOAIS masks, the badge->instance binding, and the occlusion edges.

    `instances` labels every pixel with the badge id covering it -- the same shape a
    dataset's instance annotation has, built from UOAIS so no ground truth is needed.
    """
    import cv2
    import run_gemini_uoais_ref as ref
    import uoais_obstruction as U

    U.set_depth_norm_mode(args.depth_norm_mode, depth_mm)
    uo = U.run_uoais(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_mm, predictor, cfg)
    id_to_idx, _d, unmapped = ref.map_detected_ids_to_uoais(
        objects, uo, max_nearest_px=MAX_NEAREST_PX)
    idx_to_id = {i: b for b, i in id_to_idx.items()}
    mapped = sorted(idx_to_id)

    instances = np.zeros(depth_mm.shape, np.int32)
    for i in mapped:
        instances[uo["visible_masks"][i]] = idx_to_id[i]

    edges = []
    if mapped:
        vis = np.stack([uo["visible_masks"][i] for i in mapped])
        am = np.stack([uo["amodal_masks"][i] for i in mapped])
        occ = np.array([uo["occluded"][i] for i in mapped])
        l2b = {l: idx_to_id[i] for l, i in enumerate(mapped)}
        for e in U.build_obstruction_edges(vis, am, depth_mm, occ, features=True):
            edges.append({
                "from": int(l2b[int(e["i"])]), "to": int(l2b[int(e["j"])]),
                "prob": round(float(e["conf"]), 3), "p_cv": round(float(e["p_cv"]), 3),
                "r_ij": round(float(e["r_ij"]), 3),
                "valid_depth_ratio": round(float(e["valid_depth_ratio"]), 3),
                "contact_px": int(e["contact_px"]),
                "direct_contact_px": int(e.get("direct_contact_px", e["contact_px"])),
                "clearance_contact_px": int(e.get("clearance_contact_px", 0)),
                "contact_mode": e.get("contact_mode", "hidden_overlap"),
                "occlusion_ratio": round(float(e["occlusion_ratio"]), 4),
                "amodal_px": int(e["amodal_px"]), "hidden_px": int(e["hidden_px"]),
                "depth_delta": round(float(e["depth_delta"]), 1),
                "accepted_reason": e.get("accepted_reason"),
            })
    return edges, instances, int(uo.get("n", 0)), unmapped


def resolve_request(objects, request):
    """Which badge the request names, so the geometry table can be ordered around it."""
    want = request.lower().strip()
    for pre in ("pick up ", "pick ", "grasp ", "grab ", "get ", "take ", "the "):
        if want.startswith(pre):
            want = want[len(pre):]
    labels = {int(o["id"]): (o.get("label") or "").lower() for o in objects}
    for b, l in labels.items():
        if l == want:
            return b
    hits = [b for b, l in labels.items() if l and (l in want or want in l)]
    if len(hits) == 1:
        return hits[0]
    tw = set(want.split())
    sc = sorted(((len(tw & set(l.split())), b) for b, l in labels.items() if l), reverse=True)
    if sc and sc[0][0] and (len(sc) == 1 or sc[0][0] > sc[1][0]):
        return sc[0][1]
    tied = hits or [b for n, b in sc if sc and n == sc[0][0] and n > 0]
    if len(tied) > 1:
        pos = {int(o["id"]): (o.get("x_px"), o.get("y_px")) for o in objects}
        for w, (ax, pick) in {"right": (0, max), "left": (0, min), "bottom": (1, max),
                              "top": (1, min), "front": (1, max), "back": (1, min)}.items():
            if w in tw:
                c = [b for b in tied if pos.get(b) and pos[b][ax] is not None]
                if c:
                    return pick(c, key=lambda b: pos[b][ax])
    return -1


def reason(labeled_b64, objects, request, edges, args):
    import gemini_uoais_lib as ref
    import run_gemini_uoais_ref as base
    scored = base.score_reference_edges(edges, args)
    badge = resolve_request(objects, request)
    ref_edges = ref.build_uoais_reference_edges(
        scored, target_id=badge, max_edges=MAX_REF_EDGES, min_contact_px=MIN_REF_CONTACT_PX)
    lines = ref.format_uoais_reference_lines(ref_edges)
    return ref.run_reasoning_once(labeled_b64, objects, request, lines,
                                  max_retries=2), ref_edges, lines, badge


# --------------------------------------------------------------------- grasping

def build_scene(rgb, depth_mm, instances, intr):
    from grasp_viz import scene as S
    h, w = depth_mm.shape
    dm = (np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0) / 1000.0).astype(np.float64)
    cam = S.Camera(width=w, height=h, fx=float(intr["fx"]), fy=float(intr["fy"]),
                   cx=float(intr["cx"]), cy=float(intr["cy"]))
    return S.Scene(0, rgb, dm, instances, cam, cloud=cam.backproject(dm))


def grasp(scene, badge, label, out, gripper_cfg):
    from grasp_viz import cases as C, paths as gp, real_data, run_grasp_viz as RGV
    target = C.Target("gemini_uoais_ref", None, label, int(badge), False, {})
    case = C.Case(key="scene", index=0, image_id=0, query_object=int(badge),
                  query=label, difficulty="", gt_top_ids=[],
                  targets={"gemini_uoais_ref": target})
    dirs = {m: str(out) for m in gp.METHOD_LOGS_REAL}
    gp.out_dirs = lambda b="fgc", dataset=None, _d=dirs: _d
    gp.compare_dir = lambda b="fgc", dataset=None, _o=str(out): _o
    real_data.load_scene = lambda _i, _s=scene: _s
    RGV.process_case(case, ["gemini_uoais_ref"],
                     RGV._Backend("fgc", gripper_cfg=gripper_cfg), dataset=gp.REAL)
    return out / "scene"


# --------------------------------------------------------------------- report

def write_outputs(out, request, target_badge, objects, parsed, gj, gt, elapsed):
    labels = {int(o["id"]): o.get("label") for o in objects}
    order = [int(i) for i in (parsed.get("chosen_to_remove_ids") or [])]
    first = order[0] if order else None
    plan = {
        "request": request,
        "request_badge_id": target_badge if target_badge > 0 else None,
        "grasp_first": first, "grasp_first_label": labels.get(first),
        "removal_order": order, "removal_order_labels": [labels.get(i) for i in order],
        "confidence": parsed.get("confidence"),
        "occlusion_chain": parsed.get("occlusion_chain") or [],
        "objects": [{"badge_id": int(o["id"]), "label": o.get("label"),
                     "point_px": [o.get("x_px"), o.get("y_px")]} for o in objects],
    }
    if gt:
        plan["ground_truth"] = gt.get("ground_truth")
        plan["reference_run"] = gt.get("reference_run")
    (out / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), "utf-8")

    L = ['Request     : "%s"' % request]
    L.append("Objects     : " + ", ".join("%s=%s" % (o["badge_id"], o["label"])
                                          for o in plan["objects"]))
    L.append("")
    chain = plan["occlusion_chain"]
    L.append("Occlusion chain (%d):" % len(chain))
    if not chain:
        L.append("  none -- the requested object is directly reachable")
    for e in chain:
        L.append("  %-2s %-20s occluded by  %-2s %-20s conf=%s" % (
            e.get("occluded_id"), e.get("occluded", ""),
            e.get("occluder_id"), e.get("occluder", ""), e.get("edge_confidence")))
    L.append("")
    L.append("Grasp first : id=%s (%s)" % (first, labels.get(first)))
    L.append("Confidence  : %s" % plan["confidence"])
    if gt:
        L.append("GT free ids : %s   (bundled answer, never shown to the model)"
                 % gt["ground_truth"]["top_ids"])
    L.append("")
    if gj and gj.get("grasp_found"):
        st = gj.get("stats", {})
        L.append("Grasp pose (camera frame), FGC-GraspNet:")
        L.append("  translation  %s m" % [round(v, 4) for v in gj["translation"]])
        L.append("  approach     %s  (rotation column 2)" %
                 [round(r[2], 3) for r in gj["rotation"]])
        L.append("  width %.1f cm   score %.3f" % (gj.get("width", 0) * 100, gj.get("score", 0)))
        L.append("  funnel: %s raw -> %s on target -> %s collision-free -> %s top-down%s"
                 % (st.get("raw"), st.get("in_mask"), st.get("collision_free"),
                    st.get("top_down"),
                    "" if st.get("aperture_ok") is None else " -> %s fit the jaws" % st["aperture_ok"]))
    else:
        L.append("Grasp pose  : none found (%s)"
                 % (gj or {}).get("stats", {}).get("note", "no candidate survived"))
    L.append("")
    L.append("Elapsed     : %.1f s" % elapsed)
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    return text


# --------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("input: either a bundled case, or your own frame")
    g.add_argument("--case", help="a scene from demo_rgbd/, e.g. img000059_q4")
    g.add_argument("--rgb", help="RGB image")
    g.add_argument("--depth", help="depth aligned to the RGB, same HxW")
    g.add_argument("--intrinsics", help="json with fx, fy, cx, cy (and depth_scale for 16-bit png)")
    g.add_argument("--request", help='what you want, in plain words, e.g. "the white box"')
    ap.add_argument("--masks", choices=["uoais", "provided"], default="uoais",
                    help="where the grasp mask comes from. 'uoais' is the deployable path "
                         "and needs no annotation; 'provided' uses demo_rgbd's instances.png "
                         "and reproduces the reference outputs exactly")
    ap.add_argument("--model", help="Gemini model id, e.g. gemini-robotics-er-2-preview")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gripper", default="robotiq_2f85",
                    help="gripper yaml in grasp_viz/, by name or path "
                         "(robotiq_2f85 = the one the UR5e sim drives; tho_pa1 = the other cell)")
    ap.add_argument("--no-gripper-config", action="store_true",
                    help="do not hold grasps to any jaw limit")
    ap.add_argument("--out", default=None, help="default: output/<case>")
    args = ap.parse_args(argv)
    for k, v in EDGE_SCORING.items():
        setattr(args, k, v)

    sys.path.insert(0, str(ROOT))
    gt = None
    if args.case:
        rgb_p, dep_p, intr_p, request, inst_p, gt = load_case(args.case)
        name = args.case
    else:
        miss = [f for f in ("rgb", "depth", "intrinsics", "request") if not getattr(args, f)]
        if miss:
            raise SystemExit("give --case, or all of --rgb --depth --intrinsics --request "
                             "(missing: %s)" % ", ".join("--" + m for m in miss))
        rgb_p, dep_p, intr_p, request, inst_p = (args.rgb, args.depth, args.intrinsics,
                                                 args.request, None)
        name = Path(args.rgb).stem
    out = Path(args.out) if args.out else ROOT / "output" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.model:
        import gemini_client as gc
        gc.MODEL = args.model
    import uoais_obstruction as U
    for k, v in UOAIS_GLOBALS.items():
        setattr(U, k, v)

    log(1, "loading %s" % name)
    rgb = np.asarray(Image.open(rgb_p).convert("RGB"))
    intr = json.loads(Path(intr_p).read_text(encoding="utf-8"))
    depth_mm = load_depth_mm(dep_p, intr)
    if depth_mm.shape != rgb.shape[:2]:
        raise SystemExit("depth %s does not match RGB %s" % (depth_mm.shape, rgb.shape[:2]))
    print('      %dx%d | request "%s" | fx=%.1f' % (rgb.shape[1], rgb.shape[0], request, intr["fx"]))

    log(2, "detecting objects  [Gemini]")
    objects, labeled_b64 = detect(rgb_p, out)
    print("      %d objects: %s" % (len(objects),
          ", ".join("%s=%s" % (o["id"], o.get("label")) for o in objects)))

    log(3, "amodal segmentation + occlusion geometry  [UOAIS]")
    predictor, cfg = U.load_uoais_predictor(U.CFG_RGBD, UOAIS_SCORE_THRESH,
                                            UOAIS_NMS_THRESH, "auto")
    edges, uoais_inst, n_inst, unmapped = segment(rgb, depth_mm, objects, predictor, cfg, args)
    print("      %d instances, %d occlusion edges%s" % (n_inst, len(edges),
          "" if not unmapped else ", badges %s unbound" % unmapped))

    log(4, "reasoning  [Gemini]")
    parsed, ref_edges, lines, badge = reason(labeled_b64, objects, request, edges, args)
    json.dump({"reasoning": parsed, "uoais_reference_edges": ref_edges,
               "uoais_reference_prompt_lines": lines},
              open(out / "reasoning.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    order = [int(i) for i in (parsed.get("chosen_to_remove_ids") or [])]
    if not order:
        raise SystemExit("the reasoner named no object to remove; see reasoning.json")
    first = order[0]
    labels = {int(o["id"]): o.get("label") for o in objects}
    print("      grasp first: id=%s (%s), chain %d step(s), confidence %s"
          % (first, labels.get(first), len(parsed.get("occlusion_chain") or []),
             parsed.get("confidence")))

    log(5, "grasp pose  [FGC-GraspNet]")
    if args.masks == "provided":
        if inst_p is None:
            raise SystemExit("--masks provided needs a bundled --case")
        # bind the chosen badge to the annotated instance through the badge's own pixel
        prov = np.asarray(Image.open(inst_p)).astype(np.int32)
        pt = next((o for o in objects if int(o["id"]) == first), None)
        iid = int(prov[int(pt["y_px"]), int(pt["x_px"])]) if pt else 0
        if iid == 0:
            raise SystemExit("badge %d does not land on any annotated instance" % first)
        instances, target_id = prov, iid
        print("      mask: instances.png, badge %d -> instance %d" % (first, iid))
    else:
        instances, target_id = uoais_inst, first
        print("      mask: UOAIS instance for badge %d" % first)

    gcfg = None
    if not args.no_gripper_config:
        from grasp_viz import gripper_config
        g = args.gripper
        if not os.path.exists(g):
            g = os.path.join(ROOT, "grasp_viz", "gripper_%s.yaml" % g.replace("gripper_", ""))
        gcfg = gripper_config.load(g)
        print("      gripper: %s, jaws <= %.0f mm"
              % (gcfg.get("name", g), 1000 * (gcfg["grasp_filter"]["max_grasp_width"]
                                              - gcfg["grasp_filter"]["width_margin"])))
    scene = build_scene(rgb, depth_mm, instances, intr)
    case_dir = grasp(scene, target_id, labels.get(first) or request, out, gcfg)
    gj = json.loads((case_dir / "grasp_pose.json").read_text()) \
        if (case_dir / "grasp_pose.json").exists() else {}
    print("      grasp_found=%s" % gj.get("grasp_found"))

    if not args.no_gif and gj.get("grasp_found"):
        from grasp_viz import make_orbit_gif as MOG
        MOG.build_gif(str(case_dir), str(case_dir / "05_orbit.gif"))

    for f in sorted(case_dir.iterdir()):
        if f.is_file():
            shutil.move(str(f), str(out / f.name))
    case_dir.rmdir()
    shutil.rmtree(out / "uoais_cache", ignore_errors=True)

    print("\n" + write_outputs(out, request, badge, objects, parsed, gj, gt, time.time() - t0))
    print("wrote %s/" % (out.relative_to(ROOT) if out.is_relative_to(ROOT) else out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
