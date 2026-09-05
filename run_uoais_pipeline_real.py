#!/usr/bin/env python3
"""
Run the UOAIS obstruction pipeline on the REAL data_ifl_real scenes, evaluating
against a GT file such as gt_from_occlusion_real_v1.json.

Unlike run_uoais_pipeline.py (which sources RGB/depth/object-points from the
synthetic UnoBench zips + flat image_id mapping), this runner reads everything
directly from the local scene folders:

    data_ifl_*/mnt/data1/data_ifl_real/scene{image_id}/
        3_rgb.png     -> RGB (BGR via cv2)
        3.npz         -> depth (mm after cm->mm), instances_objects (GT masks)

`image_id` in the GT file is the scene number. GT object points needed to map
each GT object id onto a UOAIS-detected instance are taken as an interior point
(distance-transform peak) of that object's visible mask in instances_objects.

Metrics (compute_summary) are computed exactly like run_uoais_pipeline.py:
predicted top-objects-to-remove vs GT top_objects.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import uoais_obstruction as U
from uoais_graph import (
    graph_from_uoais,
    map_object_ids_to_uoais,
    select_for_target,
    format_model_output,
)

ROOT = Path(__file__).resolve().parent
DATA_GLOB = "data_ifl_*/mnt/data1/data_ifl_real/scene*"


# ── helpers copied from run_uoais_pipeline.py (that file imports MAPPING_VERSION
#    from a newer uoais_graph than the one installed here, so it can't be
#    imported directly; these functions are pure and self-contained) ───────────

def prf(pred: set[int], truth: set[int]) -> tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    r = tp / len(truth) if truth else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def path_depth(row: dict) -> int:
    return max((len(p) - 1 for p in row.get("occlusion_paths", []) or [[row["query_object"]]]), default=0)


def make_path_bucket(depth: int) -> str:
    if depth <= 1:
        return "path_depth_1"
    if depth == 2:
        return "path_depth_2"
    return "path_depth_ge3"


def paths_to_triplets(paths: list[list[int]]) -> set[tuple[int, int]]:
    """Ordered consecutive pairs (occluder -> occluded) across all paths."""
    triplets = set()
    for p in paths:
        for i in range(len(p) - 1):
            triplets.add((int(p[i]), int(p[i + 1])))
    return triplets


def prf_triplets(pred: set, gt: set) -> tuple[float, float, float]:
    """Precision/recall/F1 over path edge-sets (evaluate_nlp.compute_prf)."""
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def _ned(seq1: list[int], seq2: list[int]) -> float:
    import editdistance
    return editdistance.eval(seq1, seq2) / max(len(seq1), len(seq2), 1)


def mp_ned(pred_paths: list[list[int]], gt_paths: list[list[int]], alpha: float = 1.0, beta: float = 1.0) -> float:
    """Matched-path normalized edit distance (Hungarian match), 0 = identical."""
    from scipy.optimize import linear_sum_assignment
    m, n = len(pred_paths), len(gt_paths)
    size = max(m, n)
    if size == 0:
        return 0.0
    C = np.zeros((size, size))
    for i in range(m):
        for j in range(n):
            C[i, j] = _ned(pred_paths[i], gt_paths[j])
    if m < n:
        C[m:size, :n] = alpha
    elif m > n:
        C[:m, n:size] = beta
    row_ind, col_ind = linear_sum_assignment(C)
    return float(C[row_ind, col_ind].sum() / size)


def compute_summary(results: list[dict], skipped: list[dict], requested: int) -> dict:
    buckets = defaultdict(lambda: {
        "n": 0, "exact": 0, "action": 0, "target_mapped": 0,
        "P": [], "R": [], "F1": [], "candidate_F1": [], "free_F1": [],
        # occlusion-path (occlusion reasoning) metrics
        "OP": [], "OR": [], "OF1": [], "MP_NED": [],
    })
    for row in results:
        truth = set(map(int, row["gt_top_objects"]))
        pred = set(map(int, row["chosen_to_remove_ids"]))
        cand = set(map(int, row["candidate_ids"]))
        free = set(map(int, row["free_object_ids"]))
        p, r, f1 = prf(pred, truth)
        _, _, cand_f1 = prf(cand, truth)
        _, _, free_f1 = prf(free, truth)
        pred_action = "pick object" if pred == {int(row["query_object"])} else "remove obstacle"
        gt_action = "pick object" if truth == {int(row["query_object"])} else "remove obstacle"

        gt_paths = row.get("gt_occlusion_paths", []) or [[int(row["query_object"])]]
        pred_paths = row.get("pred_paths", []) or [[int(row["query_object"])]]
        op, orr, of1 = prf_triplets(paths_to_triplets(pred_paths), paths_to_triplets(gt_paths))
        mpned = mp_ned(pred_paths, gt_paths)

        names = [
            "overall",
            f"new:{row['new_difficulty']}",
            f"old:{row['old_difficulty']}",
            row["path_bucket"],
            f"mapping:{row['target_mapping_status']}",
        ]
        for name in names:
            b = buckets[name]
            b["n"] += 1
            b["exact"] += int(pred == truth)
            b["action"] += int(pred_action == gt_action)
            b["target_mapped"] += int(row["target_mapped"])
            b["P"].append(p)
            b["R"].append(r)
            b["F1"].append(f1)
            b["candidate_F1"].append(cand_f1)
            b["free_F1"].append(free_f1)
            b["OP"].append(op)
            b["OR"].append(orr)
            b["OF1"].append(of1)
            b["MP_NED"].append(mpned)

    summary = {
        "requested_cases": requested,
        "evaluated_cases": len(results),
        "skipped_cases": len(skipped),
        "skipped_by_reason": dict(Counter(s["reason"] for s in skipped)),
    }
    for name, b in buckets.items():
        if not b["n"]:
            continue
        summary[name] = {
            "n": b["n"],
            "exact_success_rate": b["exact"] / b["n"],
            "action_accuracy": b["action"] / b["n"],
            "target_mapped_rate": b["target_mapped"] / b["n"],
            "SR_P": float(np.mean(b["P"])),
            "SR_R": float(np.mean(b["R"])),
            "SR_F1": float(np.mean(b["F1"])),
            "candidate_F1": float(np.mean(b["candidate_F1"])),
            "free_F1": float(np.mean(b["free_F1"])),
            # occlusion-path reasoning: precision/recall/F1 over path edges + MP_NED
            "occ_path_P": float(np.mean(b["OP"])),
            "occ_path_R": float(np.mean(b["OR"])),
            "occ_path_F1": float(np.mean(b["OF1"])),
            "MP_NED": float(np.mean(b["MP_NED"])),
        }
    return summary


def _explained_frac(edge: dict) -> float:
    return edge["direct_contact_px"] / max(edge["hidden_px"], 1)


# ── GT-free edge score `calib_conf` (report-only; predictions unchanged) ───────
# Raw p_cv confidence piles up at low scores (0.1-0.3) while r_ij is nicely
# right-skewed (more edges at high r_ij). Blend r_ij into the score so the
# distribution is right-skewed too: high score -> many edges & mostly correct,
# low score -> few edges. Weights are domain/shape choices (GT-free); GT is used
# only to verify separation. Never read by the gate/direction resolver, so SR /
# occlusion-path predictions stay identical.
CALIB_W_RIJ = 0.6      # weight on r_ij (drives the right-skewed shape)
CALIB_OCC_REF = 0.15   # occlusion_ratio reference level (domain prior)
CALIB_W_OCC = 0.5      # split of the strength term between occ_norm ...
CALIB_W_EXF = 0.5      # ... and explained_frac


def calibrated_confidence(edge: dict) -> float:
    """GT-free edge score in [0,1] = 0.6*r_ij + 0.4*strength, where
    strength = 0.5*min(occlusion_ratio/0.15,1) + 0.5*min(explained_frac,1).
    Convex combination of values already in [0,1] -> no sigmoid needed."""
    occ_norm = min(float(edge["occlusion_ratio"]) / CALIB_OCC_REF, 1.0)
    exf = min(_explained_frac(edge), 1.0)
    strength = CALIB_W_OCC * occ_norm + CALIB_W_EXF * exf
    return float(CALIB_W_RIJ * float(edge["r_ij"]) + (1.0 - CALIB_W_RIJ) * strength)


def annotate_calib_conf(edges: list[dict]) -> list[dict]:
    for e in edges:
        e["calib_conf"] = round(calibrated_confidence(e), 4)
    return edges


def _depth_delta(edge: dict) -> float:
    """depth_delta but NaN/None-safe.

    On real depth maps the contact region often has invalid/NaN pixels, so
    ``edge['depth_delta']`` comes back as NaN. A raw NaN silently corrupts both
    the direction resolver (``tanh(NaN)`` -> NaN makes every reciprocal-pair
    comparison False, so the *wrong* direction is kept) and the depth-rescue
    gate. Treating a missing depth delta as 0 (no depth evidence either way)
    lets ``_explained_frac`` decide the direction, which is the correct
    behaviour when depth is unreliable.
    """
    v = edge.get("depth_delta")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


def _direction_score(edge: dict, depth_weight: float, depth_sigma_mm: float) -> float:
    return _explained_frac(edge) + depth_weight * float(np.tanh(_depth_delta(edge) / max(depth_sigma_mm, 1e-6)))


def _resolve_directions_v2(edges: list[dict], depth_weight: float, depth_sigma_mm: float) -> list[dict]:
    edge_map = {(e["from"], e["to"]): e for e in edges}
    keep = set(edge_map)
    for i, j in list(edge_map):
        if i >= j or (j, i) not in edge_map:
            continue

        def rank(a: int, b: int):
            e = edge_map[(a, b)]
            return _direction_score(e, depth_weight, depth_sigma_mm), e["prob"]

        drop = (j, i) if rank(i, j) >= rank(j, i) else (i, j)
        keep.discard(drop)
    return [e for e in edges if (e["from"], e["to"]) in keep]


def _gate_v2(edge: dict, args) -> bool:
    depth_rescued = (
        args.v2_depth_rescue_mm > 0
        and _depth_delta(edge) >= args.v2_depth_rescue_mm
        and edge["hidden_px"] >= args.v2_rescue_min_hidden_px
    )
    if edge["contact_mode"] == "clearance_contact":
        return bool(args.v2_keep_clearance) or depth_rescued
    if edge["hidden_px"] < args.v2_min_hidden_px:
        return depth_rescued
    return (
        _explained_frac(edge) >= args.v2_min_explained_frac
        or edge["direct_contact_px"] >= args.v2_abs_contact_px
        or depth_rescued
    )


def rebuild_edges_v2(edges_all: list[dict], args) -> list[dict]:
    resolved = _resolve_directions_v2(edges_all, args.v2_depth_weight, args.v2_depth_sigma_mm)
    return sorted((e for e in resolved if _gate_v2(e, args)), key=lambda e: (e["to"], e["from"]))


def configure_uoais(args) -> None:
    U.DEPTH_TOLERANCE_MM = float(args.depth_tolerance_mm)
    U.CONTACT_DILATE_PX = int(args.contact_dilate_px)
    U.MIN_CONTACT_PIXELS = int(args.min_contact_pixels)
    U.MIN_VALID_DEPTH_RATIO = float(args.min_valid_depth_ratio)
    U.DEPTH_RELIABILITY_MODE = args.depth_reliability_mode
    U.DEPTH_CONSISTENCY_SIGMA_MM = float(args.depth_consistency_sigma_mm)
    U.CONTACT_SUPPORT_PX = float(args.contact_support_px)
    U.CV_OCCLUSION_O0 = float(args.cv_o0)
    U.CV_SIGMOID_KAPPA = float(args.cv_kappa)
    U.MIN_DIRECT_CONTACT_PX = int(args.min_direct_contact_px)
    U.MIN_EDGE_OCC_PIXELS = int(args.min_edge_occ_pixels)
    U.EDGE_CLEARANCE_FALLBACK = not args.no_clearance_edges


def build_scene_dirs() -> dict[int, Path]:
    """Map scene_id -> scene directory across all data_ifl_* buckets."""
    dirs: dict[int, Path] = {}
    for p in glob.glob(str(ROOT / DATA_GLOB)):
        m = re.search(r"scene(\d+)$", p)
        if m:
            dirs[int(m.group(1))] = Path(p)
    return dirs


def interior_point(mask: np.ndarray) -> tuple[int, int]:
    """Return an (x, y) point robustly inside `mask` (distance-transform peak)."""
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(dt.argmax()), dt.shape)
    return int(x), int(y)


def object_points_from_instances(inst: np.ndarray) -> dict[int, dict]:
    """Build {obj_id: {name, som_x, som_y}} from a GT instance-id map."""
    points: dict[int, dict] = {}
    for obj_id in np.unique(inst):
        oid = int(obj_id)
        if oid == 0:
            continue
        mask = inst == obj_id
        x, y = interior_point(mask)
        points[oid] = {"name": f"object {oid}", "som_x": x, "som_y": y}
    return points


def load_cases(gt_path: Path, difficulties: set[str], n_cases: int | None) -> list[dict]:
    rows = json.loads(gt_path.read_text(encoding="utf-8"))
    cases = []
    for idx, row in enumerate(rows):
        diff = row.get("difficulty")
        if diff not in difficulties:
            continue
        case = dict(row)
        case["case_index"] = idx
        case["old_difficulty"] = diff
        case["new_difficulty"] = diff
        case["selected_difficulty"] = diff
        case["query_object"] = int(case["query_object"])
        case["image_id"] = int(case["image_id"])
        case["top_objects"] = [int(x) for x in case.get("top_objects", [])]
        case["occlusion_paths"] = [[int(x) for x in p] for p in case.get("occlusion_paths", [])]
        cases.append(case)
        if n_cases is not None and len(cases) >= n_cases:
            break
    return cases


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / "image_graphs"
    graphs_dir.mkdir(exist_ok=True)

    gt_path = Path(args.gt_path)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path
    difficulties = set(args.difficulties)
    cases = load_cases(gt_path, difficulties, args.n_cases)
    by_image: dict[int, list] = defaultdict(list)
    for case in cases:
        by_image[int(case["image_id"])].append(case)

    scene_dirs = build_scene_dirs()

    configure_uoais(args)
    predictor, cfg = U.load_uoais_predictor(
        U.CFG_RGBD,
        score_thresh=args.uoais_score_thresh,
        nms_thresh=args.uoais_nms_thresh,
        device=args.device,
    )

    results: list[dict] = []
    skipped: list[dict] = []
    pred_path = out_dir / "predictions.jsonl"
    results_jsonl = out_dir / "results.jsonl"
    total_images = len(by_image)

    with pred_path.open("w", encoding="utf-8") as pred_f, results_jsonl.open("w", encoding="utf-8") as res_f:
        for image_rank, image_id in enumerate(sorted(by_image), start=1):
            scene_dir = scene_dirs.get(int(image_id))
            if scene_dir is None:
                for case in by_image[image_id]:
                    skipped.append({"image_id": image_id, "query_object": case["query_object"], "reason": "missing_scene_dir"})
                continue

            graph_path = graphs_dir / f"scene{image_id}.json"
            try:
                rgb = cv2.imread(str(scene_dir / "3_rgb.png"), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(str(scene_dir / "3_rgb.png"))
                npz = np.load(scene_dir / "3.npz", allow_pickle=True)
                depth_mm = npz["depth"].astype(np.float32)
                if np.nanmax(depth_mm) < 200:  # cm -> mm
                    depth_mm = depth_mm * 10.0
                object_points = object_points_from_instances(npz["instances_objects"])

                U.set_depth_norm_mode(args.depth_norm_mode, depth_mm)
                uoais = U.run_uoais(rgb, depth_mm, predictor, cfg)
                id_to_idx, mapping_details, unmapped = map_object_ids_to_uoais(
                    object_points, uoais, args.max_nearest_px
                )
                graph = graph_from_uoais(id_to_idx, uoais, depth_mm)
                annotate_calib_conf(graph["edges_all"])
                annotate_calib_conf(graph["edges"])
                graph_record = {
                    "image_id": int(image_id),
                    "scene_dir": str(scene_dir),
                    "n_uoais": int(uoais["n"]),
                    "id_to_uoais_idx": {str(k): int(v) for k, v in id_to_idx.items()},
                    "mapping_details": mapping_details,
                    "unmapped_object_ids": unmapped,
                    "object_points": object_points,
                    **graph,
                }
                with graph_path.open("w", encoding="utf-8") as f:
                    json.dump(graph_record, f, indent=2)
            except Exception as exc:
                for case in by_image[image_id]:
                    skipped.append({"image_id": image_id, "query_object": case["query_object"], "reason": "image_graph_failed", "error": str(exc)})
                continue

            if image_rank == 1 or image_rank % args.report_every_images == 0 or image_rank == total_images:
                print(f"[progress] images {image_rank}/{total_images} results={len(results)} skipped={len(skipped)}", flush=True)

            selection_edges = (
                rebuild_edges_v2(graph_record["edges_all"], args)
                if args.edge_scoring == "v2"
                else graph_record["edges"]
            )
            object_points_by_id = {int(k): v for k, v in graph_record["object_points"].items()}

            for case in by_image[image_id]:
                target = int(case["query_object"])
                target_mapping = next(
                    (d for d in graph_record.get("mapping_details", []) if int(d["object_id"]) == target),
                    {"status": "unmapped", "method": None, "distance_px": None},
                )
                if str(target) not in graph_record.get("id_to_uoais_idx", {}):
                    selection = {
                        "candidate_ids": [], "candidate_edges": [], "free_object_ids": [],
                        "chosen_to_remove_ids": [], "selection_reason": "target_unmapped_to_uoais", "pred_paths": [],
                    }
                    target_mapped = False
                else:
                    selection = select_for_target(target, selection_edges)
                    selection.setdefault("candidate_edges", [])
                    target_mapped = True

                model_output = format_model_output(
                    selection["pred_paths"] or [[target]],
                    selection["chosen_to_remove_ids"] or [],
                    object_points_by_id,
                )
                depth = path_depth(case)
                result = {
                    "case_index": int(case["case_index"]),
                    "image_id": int(image_id),
                    "query_object": target,
                    "difficulty": case["selected_difficulty"],
                    "old_difficulty": case["old_difficulty"],
                    "new_difficulty": case["new_difficulty"],
                    "path_depth": depth,
                    "path_bucket": make_path_bucket(depth),
                    "gt_top_objects": case["top_objects"],
                    "gt_occlusion_paths": case["occlusion_paths"],
                    "pred_paths": selection.get("pred_paths", []),
                    "target_mapped": target_mapped,
                    "target_mapping_status": target_mapping.get("status", "mapped" if target_mapped else "unmapped"),
                    "target_mapping_method": target_mapping.get("method"),
                    "target_mapping_distance_px": target_mapping.get("distance_px"),
                    "candidate_ids": selection["candidate_ids"],
                    "candidate_edges": selection.get("candidate_edges", []),
                    "free_object_ids": selection["free_object_ids"],
                    "chosen_to_remove_ids": selection["chosen_to_remove_ids"],
                    "selection_reason": selection["selection_reason"],
                    "model_output": model_output,
                }
                results.append(result)
                res_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                pred_f.write(json.dumps({
                    "image": str(scene_dir / "3_rgb.png"),
                    "image_id": int(image_id),
                    "query_object": target,
                    "model_output": model_output,
                }, ensure_ascii=False) + "\n")

    with (out_dir / "skipped_cases.json").open("w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = compute_summary(results, skipped, requested=len(cases))
    summary["edge_scoring"] = args.edge_scoring
    summary["gt_path"] = str(gt_path)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report = [
        "# UOAIS on REAL data_ifl_real",
        "",
        f"- gt path: `{gt_path}`",
        f"- edge scoring: `{args.edge_scoring}`",
        f"- requested cases: {len(cases)}",
        f"- evaluated cases: {len(results)}",
        f"- skipped cases: {len(skipped)}",
        f"- unique requested images (scenes): {len(by_image)}",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n===== Pipeline Summary (overall) =====")
    print(json.dumps(summary.get("overall", {}), indent=2))
    print(f"\nEvaluated {len(results)} / requested {len(cases)}; skipped {len(skipped)}.")
    print(f"Outputs in: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-path", default="gt_from_occlusion_real_v1.json")
    p.add_argument("--out", default="logs/uoais_pipeline_real")
    p.add_argument("--difficulties", nargs="+", default=["Easy", "Medium", "Hard"])
    p.add_argument("--n-cases", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--report-every-images", type=int, default=50)
    p.add_argument("--depth-norm-mode", choices=["uoais_default", "percentile"], default="uoais_default")
    p.add_argument("--uoais-score-thresh", type=float, default=0.35)
    p.add_argument("--uoais-nms-thresh", type=float, default=0.7)
    p.add_argument("--max-nearest-px", type=float, default=0.0)
    p.add_argument("--depth-tolerance-mm", type=float, default=90.0)
    p.add_argument("--contact-dilate-px", type=int, default=8)
    p.add_argument("--min-contact-pixels", type=int, default=1)
    p.add_argument("--min-valid-depth-ratio", type=float, default=0.5)
    p.add_argument("--depth-reliability-mode", choices=["valid_ratio", "soft"], default=U.DEPTH_RELIABILITY_MODE)
    p.add_argument("--depth-consistency-sigma-mm", type=float, default=U.DEPTH_CONSISTENCY_SIGMA_MM)
    p.add_argument("--contact-support-px", type=float, default=U.CONTACT_SUPPORT_PX)
    p.add_argument("--cv-o0", type=float, default=0.15)
    p.add_argument("--cv-kappa", type=float, default=12.0)
    p.add_argument("--min-direct-contact-px", type=int, default=0)
    p.add_argument("--min-edge-occ-pixels", type=int, default=0)
    p.add_argument("--no-clearance-edges", action="store_true")
    p.add_argument("--edge-scoring", choices=["legacy", "v2"], default="v2")
    p.add_argument("--v2-min-hidden-px", type=int, default=50)
    p.add_argument("--v2-min-explained-frac", type=float, default=0.05)
    p.add_argument("--v2-abs-contact-px", type=int, default=150)
    p.add_argument("--v2-keep-clearance", action="store_true")
    p.add_argument("--v2-depth-weight", type=float, default=0.5)
    p.add_argument("--v2-depth-sigma-mm", type=float, default=60.0)
    p.add_argument("--v2-depth-rescue-mm", type=float, default=30.0)
    p.add_argument("--v2-rescue-min-hidden-px", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())