#!/usr/bin/env python3
"""
UOAIS/3D obstruction-graph helpers shared by run_uoais_pipeline.py:
ID-to-instance alignment via SOM label points, occlusion-graph construction
from UOAIS masks + depth, target-centric candidate selection, and
model_output text formatting for evaluate_nlp.py.

Prediction uses only RGB-D, UOAIS masks, target object ID/name metadata, and SOM
label points for ID-to-instance alignment. GT occlusion relations and annotation
masks are used after prediction for evaluation.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import uoais_obstruction as U


ROOT = Path(__file__).resolve().parent
UNOBENCH = ROOT / "UnoBench"
# Bumped whenever ``map_object_ids_to_uoais`` changes semantics; graph caches key
# off it (see run_uoais_pipeline_v0.py) so stale caches are rebuilt instead of
# silently reused.
MAPPING_VERSION = 1
CHALLENGE_NLP = UNOBENCH / "challenge_only/test_nlp.jsonl"
CHALLENGE_SOM = UNOBENCH / "challenge_only/test_som.jsonl"
IMAGES_ZIP = UNOBENCH / "images.zip"
ANNOTATIONS_ZIP = UNOBENCH / "annotations.zip"
META_ZIP = UNOBENCH / "meta_data/annotations_meta.zip"
NAME_FOR_ALL = UNOBENCH / "meta_data/name_for_all.json"
OBS_INFO = UNOBENCH / "meta_data/occ_info/obs_information.json"
ANN_DIR = UNOBENCH / "annotations"
IMAGE_DIR = UNOBENCH / "_extracted/images"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_scene_key(key: str) -> tuple[int, int]:
    m = re.match(r"scene(\d+)/(\d+)_rgb\.png$", key)
    if not m:
        raise ValueError(f"Bad scene key: {key}")
    return int(m.group(1)), int(m.group(2))


def scene_key_to_npz_name(key: str) -> str:
    scene_id, view_id = parse_scene_key(key)
    return f"annotations_meta/ALL_NPZ/scene{scene_id}_view{view_id}.npz"


def obs_scene_key(scene_id: int, view_id: int) -> tuple[str, str]:
    return f"data_ifl_0/scene{scene_id}", str(view_id)


def load_challenge_cases(n_cases: int | None = None) -> list[dict]:
    nlp = read_jsonl(CHALLENGE_NLP)
    som_by_index = {int(r["test_index"]): r for r in read_jsonl(CHALLENGE_SOM)}
    cases = []
    for row in nlp:
        test_index = int(row["test_index"])
        som = som_by_index.get(test_index)
        if som is None:
            raise ValueError(f"Missing SOM row for test_index={test_index}")
        image_id = int(row["image_id"])
        if int(som["image_id"]) != image_id:
            raise ValueError(f"Image mismatch for test_index={test_index}")
        cases.append({
            "test_index": test_index,
            "image_id": image_id,
            "image": row["image"][0] if isinstance(row.get("image"), list) else row.get("image"),
            "query_object_name": row["query_object_name"],
            "query_object": int(som["query_object"]),
        })
    return cases if n_cases is None else cases[:n_cases]


def build_scene_index(name_for_all: dict) -> tuple[dict, dict]:
    inverted = defaultdict(set)
    obj_count = {}
    for key, objects in name_for_all.items():
        obj_count[key] = len(objects)
        for obj_id, info in objects.items():
            inverted[(int(obj_id), info["name"])].add(key)
    return inverted, obj_count


def load_annotation_from_zip(image_id: int, zann: zipfile.ZipFile) -> np.ndarray:
    with zann.open(f"image_{image_id:06d}.npy") as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=True).astype(np.int16)


def load_instances_from_meta(scene_key: str, zmeta: zipfile.ZipFile) -> np.ndarray:
    with zmeta.open(scene_key_to_npz_name(scene_key)) as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
        return data["instances_objects"].astype(np.int16)


def map_images_to_scenes(cases: list[dict], out_dir: Path, force: bool = False) -> dict[int, dict]:
    """Map challenge image IDs to scene/view keys via query names and exact mask match."""
    cache_path = out_dir / "challenge_scene_mapping.json"
    if cache_path.exists() and not force:
        return {int(k): v for k, v in load_json(cache_path).items()}

    name_for_all = load_json(NAME_FOR_ALL)
    inverted, obj_count = build_scene_index(name_for_all)
    by_image = defaultdict(list)
    for case in cases:
        by_image[int(case["image_id"])].append((int(case["query_object"]), case["query_object_name"]))

    mapping = {}
    failures = []
    image_ids = sorted(by_image)
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as zann, zipfile.ZipFile(META_ZIP) as zmeta:
        for rank, image_id in enumerate(image_ids, start=1):
            query_pairs = by_image[image_id]
            candidate_sets = [inverted.get(pair, set()) for pair in query_pairs]
            candidates = set.intersection(*candidate_sets) if candidate_sets else set()
            ann = load_annotation_from_zip(image_id, zann)
            n_objects = int(np.nanmax(ann))
            candidates = [k for k in candidates if obj_count.get(k) == n_objects]
            if len(candidates) == 1:
                matches = candidates
            else:
                matches = []
                for key in candidates:
                    try:
                        inst = load_instances_from_meta(key, zmeta)
                    except KeyError:
                        continue
                    if inst.shape == ann.shape and np.array_equal(inst, ann):
                        matches.append(key)
            if len(matches) != 1:
                failures.append({
                    "image_id": image_id,
                    "query_pairs": query_pairs,
                    "num_candidates": len(candidates),
                    "matches": matches[:10],
                })
                continue
            scene_id, view_id = parse_scene_key(matches[0])
            mapping[image_id] = {
                "scene_key": matches[0],
                "scene_id": scene_id,
                "view_id": view_id,
                "num_objects": n_objects,
                }

            if rank == 1 or rank % 300 == 0 or rank == len(image_ids):
                print(
                    f"[mapping] images {rank}/{len(image_ids)} mapped={len(mapping)} failures={len(failures)}",
                    flush=True,
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2)
    if failures:
        with (out_dir / "scene_mapping_failures.json").open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
    return mapping


def build_gt(cases: list[dict], scene_mapping: dict[int, dict], out_dir: Path) -> list[dict]:
    obs = load_json(OBS_INFO)
    edges_by_scene = defaultdict(list)
    ratios_by_edge = {}
    for row in obs:
        m = re.match(r"data_ifl_0/scene(\d+)$", row["scene_id"])
        if not m:
            continue
        scene_id = int(m.group(1))
        view_id = int(row["view_id"])
        blocker = int(row["obj1"])
        blocked = int(row["obj2"])
        edges_by_scene[(scene_id, view_id)].append((blocker, blocked))
        ratios_by_edge[(scene_id, view_id, blocker, blocked)] = float(row.get("mask_ratio", 0.0))

    gt_rows = []
    for case in cases:
        image_id = int(case["image_id"])
        target = int(case["query_object"])
        scene = scene_mapping.get(image_id)
        if scene is None:
            continue
        scene_id, view_id = int(scene["scene_id"]), int(scene["view_id"])
        blockers_of = defaultdict(list)
        for blocker, blocked in edges_by_scene.get((scene_id, view_id), []):
            blockers_of[int(blocked)].append(int(blocker))

        def dfs(node: int, visiting: set[int] | None = None) -> list[list[int]]:
            visiting = set() if visiting is None else set(visiting)
            if node in visiting:
                return [[node]]
            parents = sorted(set(blockers_of.get(node, [])))
            if not parents:
                return [[node]]
            visiting.add(node)
            paths = []
            for parent in parents:
                for upstream in dfs(parent, visiting):
                    paths.append(upstream + [node])
            return paths

        paths = dfs(target)
        # Drop exact duplicate paths while preserving order.
        dedup_paths = []
        seen = set()
        for p in paths:
            tup = tuple(p)
            if tup not in seen:
                seen.add(tup)
                dedup_paths.append(p)
        top_objects = sorted({int(p[0]) for p in dedup_paths})
        max_edges = max((len(p) - 1 for p in dedup_paths), default=0)
        if max_edges == 0:
            diff = "No-Occ"
        elif max_edges == 1 and len(dedup_paths) == 1:
            diff = "Easy"
        elif max_edges <= 2 and len(dedup_paths) <= 2:
            diff = "Medium"
        else:
            diff = "Hard"
        gt_rows.append({
            "test_index": int(case["test_index"]),
            "image_id": image_id,
            "query_object": target,
            "query_object_name": case["query_object_name"],
            "occlusion_paths": dedup_paths,
            "top_objects": top_objects,
            "difficulty": diff,
            "new_difficulty": diff,
            "path_depth": max_edges,
            "num_paths": len(dedup_paths),
            "scene_key": scene["scene_key"],
            "scene_id": scene_id,
            "view_id": view_id,
        })

    path = out_dir / "challenge_gt_from_occ_info.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(gt_rows, f, indent=2)
    return gt_rows


def ensure_image(image_id: int) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    dst = IMAGE_DIR / f"image_{image_id:06d}.png"
    if dst.exists():
        return dst
    with zipfile.ZipFile(IMAGES_ZIP) as z:
        with z.open(f"images/image_{image_id:06d}.png") as src, dst.open("wb") as f:
            shutil.copyfileobj(src, f)
    return dst


def ensure_annotation(image_id: int) -> Path:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    dst = ANN_DIR / f"image_{image_id:06d}.npy"
    if dst.exists():
        return dst
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as z:
        with z.open(f"image_{image_id:06d}.npy") as src, dst.open("wb") as f:
            shutil.copyfileobj(src, f)
    return dst


def load_depth_for_scene(scene_key: str, zmeta: zipfile.ZipFile) -> np.ndarray:
    with zmeta.open(scene_key_to_npz_name(scene_key)) as f:
        data = np.load(io.BytesIO(f.read()), allow_pickle=True)
        depth = data["depth"].astype(np.float32)
    if np.nanmax(depth) < 200:
        depth = depth * 10.0
    return depth


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def map_object_ids_to_uoais(object_points: dict[int, dict], uoais: dict, max_nearest_px: float) -> tuple[dict, list, list]:
    visible = uoais["visible_masks"]
    amodal = uoais["amodal_masks"]
    cents = [centroid(visible[i]) for i in range(len(visible))]
    proposals = []
    for obj_id, info in object_points.items():
        x, y = int(round(info["som_x"])), int(round(info["som_y"]))
        for idx in range(len(visible)):
            if 0 <= y < visible[idx].shape[0] and 0 <= x < visible[idx].shape[1] and visible[idx][y, x]:
                proposals.append((0, 0.0, obj_id, idx, "point_in_visible"))
            elif 0 <= y < amodal[idx].shape[0] and 0 <= x < amodal[idx].shape[1] and amodal[idx][y, x]:
                proposals.append((1, 0.0, obj_id, idx, "point_in_amodal"))
            elif cents[idx] is not None:
                cx, cy = cents[idx]
                dist = math.hypot(cx - x, cy - y)
                if dist <= max_nearest_px:
                    proposals.append((2, dist, obj_id, idx, "nearest_centroid"))

    mapping = {}
    used_obj, used_idx = set(), set()
    details = []
    for priority, dist, obj_id, idx, method in sorted(proposals):
        if obj_id in used_obj or idx in used_idx:
            continue
        mapping[int(obj_id)] = int(idx)
        used_obj.add(obj_id)
        used_idx.add(idx)
        details.append({
            "object_id": int(obj_id),
            "uoais_idx": int(idx),
            "method": method,
            "distance_px": round(float(dist), 1),
        })
    unmapped = [int(i) for i in object_points if int(i) not in mapping]
    return mapping, details, unmapped


def graph_from_uoais(id_to_idx: dict[int, int], uoais: dict, depth_mm: np.ndarray) -> dict:
    idx_to_id = {idx: obj_id for obj_id, idx in id_to_idx.items()}
    mapped_indices = sorted(idx_to_id)
    if not mapped_indices:
        return {"edges_all": [], "edges": [], "object_ids": []}

    vis = np.stack([uoais["visible_masks"][idx] for idx in mapped_indices])
    am = np.stack([uoais["amodal_masks"][idx] for idx in mapped_indices])
    occ = np.array([uoais["occluded"][idx] for idx in mapped_indices])
    local_to_obj = {local: idx_to_id[idx] for local, idx in enumerate(mapped_indices)}
    depths = [U.median_depth(depth_mm, vis[i]) for i in range(len(mapped_indices))]

    idx_edges = U.build_obstruction_edges(vis, am, depth_mm, occ, features=True)
    edge_by_pair = {}
    edges_all = []
    for edge in idx_edges:
        blocker = int(local_to_obj[int(edge["i"])])
        blocked = int(local_to_obj[int(edge["j"])])
        rec = {
            "from": blocker,
            "to": blocked,
            "prob": round(float(edge["conf"]), 4),
            "confidence": round(float(edge["conf"]), 4),
            "p_cv": round(float(edge["p_cv"]), 4),
            "r_ij": round(float(edge["r_ij"]), 4),
            "valid_depth_ratio": round(float(edge["valid_depth_ratio"]), 4),
            "depth_mad_mm": None if edge.get("depth_mad_mm") is None else round(float(edge["depth_mad_mm"]), 3),
            "depth_consistency": round(float(edge.get("depth_consistency", 0.0)), 4),
            "contact_support": round(float(edge.get("contact_support", 0.0)), 4),
            "contact_px": int(edge["contact_px"]),
            "direct_contact_px": int(edge.get("direct_contact_px", edge["contact_px"])),
            "clearance_contact_px": int(edge.get("clearance_contact_px", 0)),
            "contact_mode": edge.get("contact_mode", "hidden_overlap"),
            "occlusion_ratio": round(float(edge["occlusion_ratio"]), 4),
            "amodal_px": int(edge["amodal_px"]),
            "hidden_px": int(edge["hidden_px"]),
            "depth_delta": round(float(edge["depth_delta"]), 1),
            "accepted_reason": edge.get("accepted_reason"),
        }
        edges_all.append(rec)
        key = (blocker, blocked)
        if key not in edge_by_pair or rec["prob"] > edge_by_pair[key]["prob"]:
            edge_by_pair[key] = rec

    resolved = U.resolve_reciprocal_edges(
        [(int(e["i"]), int(e["j"]), float(e["conf"])) for e in idx_edges],
        depths,
    )
    edges = []
    for i, j, _ in resolved:
        key = (int(local_to_obj[int(i)]), int(local_to_obj[int(j)]))
        if key in edge_by_pair:
            edges.append(edge_by_pair[key])
    return {
        "object_ids": sorted(int(i) for i in id_to_idx),
        "edges_all": sorted(edges_all, key=lambda e: (e["to"], e["from"])),
        "edges": sorted(edges, key=lambda e: (e["to"], e["from"])),
    }


def select_for_target(target_id: int, edges: list[dict]) -> dict:
    blockers_of = defaultdict(list)
    blocked_by = defaultdict(list)
    for edge in edges:
        a, b = int(edge["from"]), int(edge["to"])
        blockers_of[b].append(a)
        blocked_by[a].append(b)

    reachable = set()
    stack = list(blockers_of.get(target_id, []))
    while stack:
        node = int(stack.pop())
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(blockers_of.get(node, []))

    if not reachable:
        return {
            "candidate_ids": [int(target_id)],
            "free_object_ids": [int(target_id)],
            "chosen_to_remove_ids": [int(target_id)],
            "selection_reason": "target_free_no_reachable_blockers",
            "pred_paths": [[int(target_id)]],
        }

    top_level = sorted(
        node for node in reachable
        if not any(parent in reachable for parent in blockers_of.get(node, []))
    )
    if not top_level:
        top_level = sorted(reachable)

    def paths_up(node: int, visiting: set[int] | None = None) -> list[list[int]]:
        visiting = set() if visiting is None else set(visiting)
        parents = [p for p in blockers_of.get(node, []) if p in reachable]
        if not parents or node in visiting:
            return [[node]]
        visiting.add(node)
        paths = []
        for parent in sorted(set(parents)):
            for p in paths_up(parent, visiting):
                paths.append(p + [node])
        return paths

    paths = paths_up(target_id)
    return {
        "candidate_ids": sorted(reachable),
        "free_object_ids": top_level,
        "chosen_to_remove_ids": top_level,
        "selection_reason": "top_level_reachable_blockers",
        "pred_paths": paths,
    }


def point_for_object(object_points: dict[int, dict], obj_id: int) -> tuple[int, int, str]:
    info = object_points.get(int(obj_id), {})
    return int(round(info.get("som_x", 0))), int(round(info.get("som_y", 0))), info.get("name", f"object {obj_id}")


def format_model_output(paths: list[list[int]], chosen: list[int], object_points: dict[int, dict]) -> str:
    think_parts = []
    for idx, path in enumerate(paths, start=1):
        if len(path) <= 1:
            x, y, name = point_for_object(object_points, path[0])
            think_parts.append(f"Path{idx}: {name} at ({x}, {y}) is free.")
            continue
        # Emit from target upward so evaluate_nlp reconstructs top->target after reversing.
        rels = []
        for child, parent in zip(reversed(path), reversed(path[:-1])):
            cx, cy, cname = point_for_object(object_points, child)
            px, py, pname = point_for_object(object_points, parent)
            rels.append(f"{cname} at ({cx}, {cy}) is occluded by {pname} at ({px}, {py})")
        think_parts.append(f"Path{idx}: " + ". ".join(rels) + ".")
    ans = []
    for obj_id in chosen:
        x, y, name = point_for_object(object_points, obj_id)
        ans.append(f"<points {x} {y}>{name}</points>")
    return f"<think>{' '.join(think_parts)}</think><answer>[{', '.join(ans)}]</answer>"


def evaluate_predictions(pred_path: Path, gt_path: Path, out_dir: Path) -> str:
    cmd = [
        sys.executable,
        str(ROOT / "evaluate_nlp.py"),
        "--pred_path", str(pred_path),
        "--gt_path", str(gt_path),
        "--npz_root", str(ANN_DIR),
        "--dataset_type", "synthetic",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    text = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
    (out_dir / "eval_results.txt").write_text(text, encoding="utf-8")
    return text


def compute_summary(results: list[dict], gt_rows: list[dict]) -> dict:
    gt_by_key = {(int(r["image_id"]), int(r["query_object"])): r for r in gt_rows}
    buckets = defaultdict(lambda: {"n": 0, "exact": 0, "P": [], "R": [], "F1": [], "path_depth": []})
    overall = buckets["overall"]
    for row in results:
        key = (int(row["image_id"]), int(row["query_object"]))
        gt = gt_by_key.get(key)
        if gt is None:
            continue
        pred = set(map(int, row["chosen_to_remove_ids"]))
        truth = set(map(int, gt["top_objects"]))
        tp = len(pred & truth)
        p = tp / len(pred) if pred else (1.0 if not truth else 0.0)
        r = tp / len(truth) if truth else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        for bucket_name in ["overall", gt["new_difficulty"], f"path_depth_{gt['path_depth']}"]:
            b = buckets[bucket_name]
            b["n"] += 1
            b["exact"] += int(pred == truth)
            b["P"].append(p)
            b["R"].append(r)
            b["F1"].append(f1)
            b["path_depth"].append(int(gt["path_depth"]))
    summary = {}
    for name, b in buckets.items():
        if b["n"] == 0:
            continue
        summary[name] = {
            "n": b["n"],
            "success_rate_exact": b["exact"] / b["n"],
            "SR_P": float(np.mean(b["P"])),
            "SR_R": float(np.mean(b["R"])),
            "SR_F1": float(np.mean(b["F1"])),
            "mean_path_depth": float(np.mean(b["path_depth"])),
        }
    return summary


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_challenge_cases(args.n_cases)
    scene_mapping = map_images_to_scenes(cases, out_dir, force=args.force_mapping)
    gt_rows = build_gt(cases, scene_mapping, out_dir)
    gt_path = out_dir / "challenge_gt_from_occ_info.json"

    name_for_all = load_json(NAME_FOR_ALL)
    by_image = defaultdict(list)
    for case in cases:
        if int(case["image_id"]) in scene_mapping:
            by_image[int(case["image_id"])].append(case)

    U.DEPTH_TOLERANCE_MM = float(args.depth_tolerance_mm)
    U.CONTACT_DILATE_PX = int(args.contact_dilate_px)
    U.MIN_CONTACT_PIXELS = int(args.min_contact_pixels)
    U.MIN_VALID_DEPTH_RATIO = float(args.min_valid_depth_ratio)
    U.DEPTH_RELIABILITY_MODE = args.depth_reliability_mode
    U.DEPTH_CONSISTENCY_SIGMA_MM = float(args.depth_consistency_sigma_mm)
    U.CONTACT_SUPPORT_PX = float(args.contact_support_px)
    U.CV_OCCLUSION_O0 = float(args.cv_o0)
    U.CV_SIGMOID_KAPPA = float(args.cv_kappa)
    predictor, cfg = U.load_uoais_predictor(
        U.CFG_RGBD,
        score_thresh=args.uoais_score_thresh,
        nms_thresh=args.uoais_nms_thresh,
        device=args.device,
    )

    graphs_dir = out_dir / "image_graphs"
    cases_dir = out_dir / "cases"
    graphs_dir.mkdir(exist_ok=True)
    cases_dir.mkdir(exist_ok=True)

    results = []
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as pred_f, zipfile.ZipFile(META_ZIP) as zmeta:
        total_images = len(by_image)
        for image_rank, image_id in enumerate(sorted(by_image), start=1):
            graph_path = graphs_dir / f"image_{image_id:06d}.json"
            scene = scene_mapping[image_id]
            object_points = {
                int(obj_id): {
                    "name": info["name"],
                    "som_x": int(info["som_x"]),
                    "som_y": int(info["som_y"]),
                }
                for obj_id, info in name_for_all[scene["scene_key"]].items()
            }
            if graph_path.exists() and not args.force:
                graph_record = load_json(graph_path)
            else:
                image_path = ensure_image(image_id)
                ensure_annotation(image_id)
                rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    raise FileNotFoundError(image_path)
                depth_mm = load_depth_for_scene(scene["scene_key"], zmeta)
                U.set_depth_norm_mode(args.depth_norm_mode, depth_mm)
                uoais = U.run_uoais(rgb, depth_mm, predictor, cfg)
                id_to_idx, mapping_details, unmapped = map_object_ids_to_uoais(
                    object_points,
                    uoais,
                    max_nearest_px=args.max_nearest_px,
                )
                graph = graph_from_uoais(id_to_idx, uoais, depth_mm)
                graph_record = {
                    "image_id": image_id,
                    "scene": scene,
                    "n_uoais": int(uoais["n"]),
                    "id_to_uoais_idx": {str(k): int(v) for k, v in id_to_idx.items()},
                    "mapping_details": mapping_details,
                    "unmapped_object_ids": unmapped,
                    "object_points": object_points,
                    **graph,
                }
                with graph_path.open("w", encoding="utf-8") as f:
                    json.dump(graph_record, f, indent=2)

            if image_rank == 1 or image_rank % args.report_every_images == 0 or image_rank == total_images:
                print(f"[progress] images {image_rank}/{total_images} processed, queries so far {len(results)}", flush=True)

            for case in by_image[image_id]:
                target = int(case["query_object"])
                if str(target) not in graph_record.get("id_to_uoais_idx", {}):
                    selection = {
                        "candidate_ids": [],
                        "free_object_ids": [],
                        "chosen_to_remove_ids": [],
                        "selection_reason": "target_unmapped_to_uoais",
                        "pred_paths": [],
                    }
                else:
                    selection = select_for_target(target, graph_record["edges"])
                model_output = format_model_output(
                    selection["pred_paths"] or [[target]],
                    selection["chosen_to_remove_ids"] or [],
                    object_points,
                )
                result = {
                    "test_index": int(case["test_index"]),
                    "image_id": image_id,
                    "query_object": target,
                    "query_object_name": case["query_object_name"],
                    "scene": scene,
                    "edges": graph_record["edges"],
                    "candidate_ids": selection["candidate_ids"],
                    "free_object_ids": selection["free_object_ids"],
                    "chosen_to_remove_ids": selection["chosen_to_remove_ids"],
                    "pred_object_ids": selection["chosen_to_remove_ids"],
                    "selection_reason": selection["selection_reason"],
                    "model_output": model_output,
                }
                results.append(result)
                pred_f.write(json.dumps({
                    "test_index": int(case["test_index"]),
                    "image": case["image"],
                    "image_id": image_id,
                    "query_object": target,
                    "model_output": model_output,
                }, ensure_ascii=False) + "\n")
                case_dir = cases_dir / f"test_{int(case['test_index']):05d}_image_{image_id:06d}_q{target}"
                case_dir.mkdir(exist_ok=True)
                with (case_dir / "result.json").open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = compute_summary(results, gt_rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    eval_text = evaluate_predictions(pred_path, gt_path, out_dir)

    report = [
        "# UOAIS UnoBench Challenge No-API Report",
        "",
        f"- queries: {len(results)} / {len(cases)}",
        f"- mapped images: {len(scene_mapping)}",
        f"- output: `{out_dir}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## evaluate_nlp.py",
        "",
        "```text",
        eval_text.strip(),
        "```",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(eval_text)
    print(json.dumps(summary.get("overall", {}), indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="logs/unobench_challenge_uoais_noapi")
    p.add_argument("--n-cases", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--force-mapping", action="store_true")
    p.add_argument("--report-every-images", type=int, default=300)
    p.add_argument("--depth-norm-mode", choices=["uoais_default", "percentile"], default="uoais_default")
    p.add_argument("--uoais-score-thresh", type=float, default=0.35)
    p.add_argument("--uoais-nms-thresh", type=float, default=0.7)
    p.add_argument("--max-nearest-px", type=float, default=120.0)
    p.add_argument("--depth-tolerance-mm", type=float, default=90.0)
    p.add_argument("--contact-dilate-px", type=int, default=8)
    p.add_argument("--min-contact-pixels", type=int, default=1)
    p.add_argument("--min-valid-depth-ratio", type=float, default=0.5)
    p.add_argument("--depth-reliability-mode", choices=["valid_ratio", "soft"], default=U.DEPTH_RELIABILITY_MODE)
    p.add_argument("--depth-consistency-sigma-mm", type=float, default=U.DEPTH_CONSISTENCY_SIGMA_MM)
    p.add_argument("--contact-support-px", type=float, default=U.CONTACT_SUPPORT_PX)
    p.add_argument("--cv-o0", type=float, default=0.15)
    p.add_argument("--cv-kappa", type=float, default=12.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
