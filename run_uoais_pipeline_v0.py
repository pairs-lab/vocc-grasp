#!/usr/bin/env python3
"""
Run UOAIS/3D on the full UnoBench GT split, excluding No-Occ cases.

This runner is intentionally separate from the FreeGrasp manifest runner, which is a
FreeGrasp manifest runner. It uses UnoBench metadata only for target object IDs
and SOM label points. GT top objects / paths are used after prediction for eval.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import uoais_obstruction as U
from uoais_graph import (
    MAPPING_VERSION,
    format_model_output,
    graph_from_uoais,
    map_object_ids_to_uoais,
    select_for_target,
)


ROOT = Path(__file__).resolve().parent
UNOBENCH = ROOT / "UnoBench"
GT_PATH = UNOBENCH / "gt_for_nlp.json"
NAME_FOR_ALL = UNOBENCH / "meta_data/name_for_all.json"
ID_MAP = UNOBENCH / "meta_data/image_id_scene_view_id_mapping.json"
IMAGES_ZIP = UNOBENCH / "images.zip"
ANNOTATIONS_ZIP = UNOBENCH / "annotations.zip"
META_ZIP = UNOBENCH / "meta_data/annotations_meta.zip"
IMAGE_DIR = UNOBENCH / "_extracted/images"
ANN_DIR = UNOBENCH / "annotations"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_key_from_mapping(row: dict) -> str:
    return f"scene{int(row['scene_id'])}/{int(row['view_id'])}_rgb.png"


def scene_npz_name(row: dict) -> str:
    stem = row.get("old_stem") or f"scene{int(row['scene_id'])}_view{int(row['view_id'])}"
    return f"annotations_meta/ALL_NPZ/{stem}.npz"


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


def load_depth_mm(map_row: dict, zmeta: zipfile.ZipFile) -> np.ndarray:
    with zmeta.open(scene_npz_name(map_row)) as f:
        npz = np.load(io.BytesIO(f.read()), allow_pickle=True)
        depth = npz["depth"].astype(np.float32)
    if np.nanmax(depth) < 200:
        depth = depth * 10.0
    return depth


def load_cases(gt_path: Path, difficulties: set[str], n_cases: int | None, difficulty_field: str) -> list[dict]:
    rows = load_json(gt_path)
    cases = []
    for idx, row in enumerate(rows):
        new_diff = row.get("new_difficulty") or row.get("difficulty")
        old_diff = row.get("difficulty")
        diff = new_diff if difficulty_field == "new_difficulty" else old_diff
        if diff not in difficulties:
            continue
        case = dict(row)
        case["case_index"] = idx
        case["old_difficulty"] = old_diff
        case["new_difficulty"] = new_diff
        case["selected_difficulty"] = diff
        case["query_object"] = int(case["query_object"])
        case["image_id"] = int(case["image_id"])
        case["top_objects"] = [int(x) for x in case.get("top_objects", [])]
        case["occlusion_paths"] = [[int(x) for x in p] for p in case.get("occlusion_paths", [])]
        cases.append(case)
        if n_cases is not None and len(cases) >= n_cases:
            break
    return cases


def balanced_smoke_cases(gt_path: Path, difficulty_field: str, n_per_diff: int = 10) -> list[dict]:
    rows = load_json(gt_path)
    buckets = defaultdict(list)
    for idx, row in enumerate(rows):
        new_diff = row.get("new_difficulty") or row.get("difficulty")
        old_diff = row.get("difficulty")
        diff = new_diff if difficulty_field == "new_difficulty" else old_diff
        allowed = {"No-Occ", "Easy", "Medium", "Hard"} if difficulty_field == "new_difficulty" else {"Easy", "Medium", "Hard"}
        if diff not in allowed:
            continue
        if len(buckets[diff]) >= n_per_diff:
            continue
        case = dict(row)
        case["case_index"] = idx
        case["old_difficulty"] = old_diff
        case["new_difficulty"] = new_diff
        case["selected_difficulty"] = diff
        case["query_object"] = int(case["query_object"])
        case["image_id"] = int(case["image_id"])
        case["top_objects"] = [int(x) for x in case.get("top_objects", [])]
        case["occlusion_paths"] = [[int(x) for x in p] for p in case.get("occlusion_paths", [])]
        buckets[diff].append(case)
        if all(len(buckets[d]) >= n_per_diff for d in sorted(allowed)):
            break
    out = []
    for diff in sorted(allowed):
        out.extend(buckets[diff])
    out.sort(key=lambda r: (int(r["image_id"]), int(r["query_object"])))
    return out


def object_points_for_scene(scene_key: str, name_for_all: dict) -> dict[int, dict] | None:
    objects = name_for_all.get(scene_key)
    if not objects:
        return None
    return {
        int(obj_id): {
            "name": info.get("name") or f"object {obj_id}",
            "som_x": int(info["som_x"]),
            "som_y": int(info["som_y"]),
        }
        for obj_id, info in objects.items()
    }


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


def compute_summary(results: list[dict], skipped: list[dict], requested: int) -> dict:
    buckets = defaultdict(lambda: {
        "n": 0,
        "exact": 0,
        "action": 0,
        "target_mapped": 0,
        "P": [],
        "R": [],
        "F1": [],
        "candidate_F1": [],
        "free_F1": [],
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
        }
    return summary


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


# ── v2 edge re-scoring (same rules as refilter_graph_edges_v2.py) ───────────────
# Applied at selection time on the cached ``edges_all`` features, so the graph
# cache itself stays scoring-agnostic and switching --edge-scoring never forces
# a UOAIS re-run.

def _explained_frac(edge: dict) -> float:
    return edge["direct_contact_px"] / max(edge["hidden_px"], 1)


def _direction_score(edge: dict, depth_weight: float, depth_sigma_mm: float) -> float:
    return _explained_frac(edge) + depth_weight * float(np.tanh(edge["depth_delta"] / max(depth_sigma_mm, 1e-6)))


def _resolve_directions_v2(edges: list[dict], depth_weight: float, depth_sigma_mm: float) -> list[dict]:
    """Keep one direction per reciprocal pair, ranked by geometric evidence
    instead of p_cv (which is biased by the blocked object's amodal area)."""
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
        and edge["depth_delta"] >= args.v2_depth_rescue_mm
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


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = Path(args.graph_cache_dir) if args.graph_cache_dir else out_dir / "image_graphs"
    graphs_dir.mkdir(exist_ok=True)

    difficulties = set(args.difficulties)
    gt_path = Path(args.gt_path)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path
    cases = (
        balanced_smoke_cases(gt_path, args.difficulty_field, args.smoke_per_difficulty)
        if args.smoke_per_difficulty
        else load_cases(gt_path, difficulties, args.n_cases, args.difficulty_field)
    )
    cases = [c for c in cases if c["selected_difficulty"] in difficulties]
    by_image = defaultdict(list)
    for case in cases:
        by_image[int(case["image_id"])].append(case)

    id_map_rows = {int(r["image_id"]): r for r in load_json(ID_MAP).get("mapping", [])}
    name_for_all = load_json(NAME_FOR_ALL)

    configure_uoais(args)
    predictor, cfg = U.load_uoais_predictor(
        U.CFG_RGBD,
        score_thresh=args.uoais_score_thresh,
        nms_thresh=args.uoais_nms_thresh,
        device=args.device,
    )

    results = []
    skipped = []
    gt_eval_rows = []
    pred_path = out_dir / "predictions.jsonl"
    results_jsonl = out_dir / "results.jsonl"

    with pred_path.open("w", encoding="utf-8") as pred_f, results_jsonl.open("w", encoding="utf-8") as res_f, zipfile.ZipFile(META_ZIP) as zmeta:
        total_images = len(by_image)
        for image_rank, image_id in enumerate(sorted(by_image), start=1):
            priority_object_ids = sorted({int(c["query_object"]) for c in by_image[image_id]})
            map_row = id_map_rows.get(int(image_id))
            if not map_row:
                for case in by_image[image_id]:
                    skipped.append({"image_id": image_id, "query_object": case["query_object"], "reason": "missing_image_id_mapping"})
                continue

            scene_key = scene_key_from_mapping(map_row)
            object_points = object_points_for_scene(scene_key, name_for_all)
            if object_points is None:
                for case in by_image[image_id]:
                    skipped.append({"image_id": image_id, "query_object": case["query_object"], "reason": "missing_name_for_all", "scene_key": scene_key})
                continue

            graph_path = graphs_dir / f"image_{image_id:06d}.json"
            try:
                graph_record = load_json(graph_path) if graph_path.exists() and not args.force else None
                edge_gate_params = {
                    "min_direct_contact_px": int(args.min_direct_contact_px),
                    "min_edge_occ_pixels": int(args.min_edge_occ_pixels),
                    "clearance_edges": not args.no_clearance_edges,
                }
                cache_is_current = (
                    graph_record is not None
                    and graph_record.get("mapping_version") == MAPPING_VERSION
                    and graph_record.get("priority_object_ids") == priority_object_ids
                    and graph_record.get("mapping_max_nearest_px") == float(args.max_nearest_px)
                    and graph_record.get("edge_gate_params", {
                        "min_direct_contact_px": 0,
                        "min_edge_occ_pixels": 0,
                        "clearance_edges": True,
                    }) == edge_gate_params
                )
                if not cache_is_current:
                    image_path = ensure_image(image_id)
                    ensure_annotation(image_id)
                    rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if rgb is None:
                        raise FileNotFoundError(str(image_path))
                    depth_mm = load_depth_mm(map_row, zmeta)
                    U.set_depth_norm_mode(args.depth_norm_mode, depth_mm)
                    uoais = U.run_uoais(rgb, depth_mm, predictor, cfg)
                    id_to_idx, mapping_details, unmapped = map_object_ids_to_uoais(
                        object_points,
                        uoais,
                        max_nearest_px=args.max_nearest_px,
                        priority_object_ids=set(priority_object_ids),
                    )
                    graph = graph_from_uoais(id_to_idx, uoais, depth_mm)
                    graph_record = {
                        "image_id": int(image_id),
                        "scene_key": scene_key,
                        "scene_id": int(map_row["scene_id"]),
                        "view_id": int(map_row["view_id"]),
                        "n_uoais": int(uoais["n"]),
                        "mapping_version": MAPPING_VERSION,
                        "priority_object_ids": priority_object_ids,
                        "mapping_max_nearest_px": float(args.max_nearest_px),
                        "edge_gate_params": edge_gate_params,
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
                print(
                    f"[progress] images {image_rank}/{total_images} results={len(results)} skipped={len(skipped)}",
                    flush=True,
                )

            selection_edges = (
                rebuild_edges_v2(graph_record["edges_all"], args)
                if args.edge_scoring == "v2"
                else graph_record["edges"]
            )
            # JSON round-trips object_points keys to strings on cache hits.
            object_points_by_id = {int(k): v for k, v in graph_record["object_points"].items()}

            for case in by_image[image_id]:
                target = int(case["query_object"])
                target_mapping = next(
                    (d for d in graph_record.get("mapping_details", []) if int(d["object_id"]) == target),
                    {"status": "unmapped", "method": None, "distance_px": None},
                )
                if str(target) not in graph_record.get("id_to_uoais_idx", {}):
                    selection = {
                        "candidate_ids": [],
                        "candidate_edges": [],
                        "free_object_ids": [],
                        "chosen_to_remove_ids": [],
                        "selection_reason": "target_unmapped_to_uoais",
                        "pred_paths": [],
                    }
                    target_mapped = False
                else:
                    selection = select_for_target(target, selection_edges)
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
                    "target_mapped": target_mapped,
                    "target_mapping_status": target_mapping["status"],
                    "target_mapping_method": target_mapping["method"],
                    "target_mapping_distance_px": target_mapping["distance_px"],
                    "candidate_ids": selection["candidate_ids"],
                    "candidate_edges": selection["candidate_edges"],
                    "free_object_ids": selection["free_object_ids"],
                    "chosen_to_remove_ids": selection["chosen_to_remove_ids"],
                    "selection_reason": selection["selection_reason"],
                    "model_output": model_output,
                }
                results.append(result)
                res_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                pred_f.write(json.dumps({
                    "image": f"images/image_{image_id:06d}.png",
                    "image_id": int(image_id),
                    "query_object": target,
                    "model_output": model_output,
                }, ensure_ascii=False) + "\n")
                gt_eval_rows.append({
                    "image_id": int(image_id),
                    "query_object": target,
                    "occlusion_paths": case["occlusion_paths"],
                    "top_objects": case["top_objects"],
                    "new_difficulty": case["new_difficulty"],
                })

    gt_eval_path = out_dir / "gt_eval.json"
    with gt_eval_path.open("w", encoding="utf-8") as f:
        json.dump(gt_eval_rows, f, indent=2)
    with (out_dir / "skipped_cases.json").open("w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)
    with (out_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary = compute_summary(results, skipped, requested=len(cases))
    summary["edge_scoring"] = args.edge_scoring
    if args.edge_scoring == "v2":
        summary["v2_params"] = {
            "min_hidden_px": args.v2_min_hidden_px,
            "min_explained_frac": args.v2_min_explained_frac,
            "abs_contact_px": args.v2_abs_contact_px,
            "keep_clearance": bool(args.v2_keep_clearance),
            "depth_weight": args.v2_depth_weight,
            "depth_sigma_mm": args.v2_depth_sigma_mm,
            "depth_rescue_mm": args.v2_depth_rescue_mm,
            "rescue_min_hidden_px": args.v2_rescue_min_hidden_px,
        }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    eval_text = evaluate_predictions(pred_path, gt_eval_path, out_dir) if results else ""
    gt_eval_old_path = out_dir / "gt_eval_old_difficulty.json"
    with gt_eval_old_path.open("w", encoding="utf-8") as f:
        json.dump([
            {
                "image_id": int(r["image_id"]),
                "query_object": int(r["query_object"]),
                "occlusion_paths": r["gt_occlusion_paths"],
                "top_objects": r["gt_top_objects"],
                "new_difficulty": r["old_difficulty"],
            }
            for r in results
        ], f, indent=2)
    eval_old_text = evaluate_predictions(pred_path, gt_eval_old_path, out_dir) if results else ""
    if results:
        (out_dir / "eval_results_new_difficulty.txt").write_text(eval_text, encoding="utf-8")
        (out_dir / "eval_results_old_difficulty.txt").write_text(eval_old_text, encoding="utf-8")
    report = [
        "# UOAIS Full UnoBench",
        "",
        f"- gt path: `{gt_path}`",
        f"- edge scoring: `{args.edge_scoring}`",
        f"- difficulty field for selection: `{args.difficulty_field}`",
        f"- requested cases: {len(cases)}",
        f"- evaluated cases: {len(results)}",
        f"- skipped cases: {len(skipped)}",
        f"- unique requested images: {len(by_image)}",
        f"- graph cache: `{graphs_dir}`",
        "",
        "## Summary",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## evaluate_nlp.py (new_difficulty)",
        "```text",
        eval_text.strip(),
        "```",
        "",
        "## evaluate_nlp.py (old difficulty)",
        "```text",
        eval_old_text.strip(),
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    if eval_text:
        print("\n===== evaluate_nlp.py (new_difficulty) =====")
        print(eval_text)
    if eval_old_text:
        print("\n===== evaluate_nlp.py (difficulty) =====")
        print(eval_old_text)
    print("\n===== Pipeline Summary (overall) =====")
    print(json.dumps(summary.get("overall", {}), indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="logs/uoais_pipeline")
    p.add_argument("--gt-path", default=str(GT_PATH))
    p.add_argument("--difficulty-field", choices=["new_difficulty", "difficulty"], default="new_difficulty")
    p.add_argument("--difficulties", nargs="+", default=["Easy", "Medium", "Hard"])
    p.add_argument("--graph-cache-dir", default=None)
    p.add_argument("--n-cases", type=int, default=None)
    p.add_argument("--smoke-per-difficulty", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--report-every-images", type=int, default=300)
    p.add_argument("--depth-norm-mode", choices=["uoais_default", "percentile"], default="uoais_default")
    p.add_argument("--uoais-score-thresh", type=float, default=0.35)
    p.add_argument("--uoais-nms-thresh", type=float, default=0.7)
    p.add_argument(
        "--max-nearest-px",
        type=float,
        default=0.0,
        help="Optional nearest-mask fallback radius; 0 disables the fallback (default)",
    )
    p.add_argument("--depth-tolerance-mm", type=float, default=90.0)
    p.add_argument("--contact-dilate-px", type=int, default=8)
    p.add_argument("--min-contact-pixels", type=int, default=1)
    p.add_argument("--min-valid-depth-ratio", type=float, default=0.5)
    p.add_argument("--depth-reliability-mode", choices=["valid_ratio", "soft"], default=U.DEPTH_RELIABILITY_MODE)
    p.add_argument("--depth-consistency-sigma-mm", type=float, default=U.DEPTH_CONSISTENCY_SIGMA_MM)
    p.add_argument("--contact-support-px", type=float, default=U.CONTACT_SUPPORT_PX)
    p.add_argument("--cv-o0", type=float, default=0.15)
    p.add_argument("--cv-kappa", type=float, default=12.0)
    p.add_argument(
        "--min-direct-contact-px",
        type=int,
        default=0,
        help="v1 gate: minimum direct hidden-overlap pixels for an occlusion edge (0 = legacy)",
    )
    p.add_argument(
        "--min-edge-occ-pixels",
        type=int,
        default=0,
        help="v1 gate: minimum hidden-region pixels of the blocked object (0 = legacy)",
    )
    p.add_argument(
        "--no-clearance-edges",
        action="store_true",
        help="v1 gate: drop clearance_contact fallback edges from the graph",
    )
    p.add_argument(
        "--edge-scoring",
        choices=["legacy", "v2"],
        default="v2",
        help="v2 (default) = evidence-based direction resolution + soft gate + depth "
             "rescue, applied on cached edges_all at selection time (no cache rebuild). "
             "legacy reproduces the pre-v2 baseline graphs.",
    )
    p.add_argument("--v2-min-hidden-px", type=int, default=50)
    p.add_argument("--v2-min-explained-frac", type=float, default=0.05)
    p.add_argument("--v2-abs-contact-px", type=int, default=150)
    p.add_argument("--v2-keep-clearance", action="store_true")
    p.add_argument("--v2-depth-weight", type=float, default=0.5)
    p.add_argument("--v2-depth-sigma-mm", type=float, default=60.0)
    p.add_argument("--v2-depth-rescue-mm", type=float, default=30.0,
                   help="Keep gated-out edges when depth_delta >= this margin (0 = off)")
    p.add_argument("--v2-rescue-min-hidden-px", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
