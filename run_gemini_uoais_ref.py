#!/usr/bin/env python3
"""
Gemini + UOAIS-reference pipeline on UnoBench cases from gt_for_nlp.json
(100/100/100 Easy/Medium/Hard by the 'difficulty' field by default).

This runner uses raw RGB images, runs Gemini pointing to create detected IDs and
labeled images, then runs the non-incremental UOAIS-reference reasoning prompt
from gemini_uoais_lib.py. GT masks/paths are used only for eval.

The UOAIS candidate rows handed to the reasoning prompt are scored with
run_uoais_pipeline_v0's edge scoring (``--edge-scoring v2``, the default) before
the reference table is built; ``--edge-scoring raw`` reproduces the previous
behaviour of passing every hypothesis through.
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

import uoais_obstruction as U
import unobench_gemini_common as legacy
import gemini_uoais_lib as ref
import gemini_client as base
import run_uoais_pipeline_v0 as pipeline_v0


ROOT = Path(__file__).resolve().parent
GT_FOR_NLP = ROOT / "UnoBench/gt_for_nlp.json"
NAME_FOR_ALL = ROOT / "UnoBench/meta_data/name_for_all.json"
ID_MAP = ROOT / "UnoBench/meta_data/image_id_scene_view_id_mapping.json"
IMAGE_DIR = ROOT / "UnoBench/_extracted/images"
ANN_DIR = ROOT / "UnoBench/annotations"

DEPTH_DIR = ROOT / "UnoBench/_extracted/depth"


def load_unobench_depth_mm(image_id):
    path = DEPTH_DIR / f"image_{int(image_id):06d}.npy"
    depth = np.load(path).astype(np.float32)
    # UnoBench extracted depth is usually in centimeters for synthetic data.
    if np.nanmax(depth) < 200:
        depth = depth * 10.0
    return depth


def centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def map_detected_ids_to_uoais(objects, uoais, max_nearest_px=120.0):
    visible = uoais["visible_masks"]
    amodal = uoais["amodal_masks"]
    cents = [centroid(visible[i]) for i in range(len(visible))]
    proposals = []

    for obj in objects:
        obj_id = int(obj["id"])
        x, y = int(obj["x_px"]), int(obj["y_px"])
        for idx in range(len(visible)):
            if 0 <= y < visible[idx].shape[0] and 0 <= x < visible[idx].shape[1] and visible[idx][y, x]:
                proposals.append((0, 0.0, obj_id, idx, "point_in_visible"))
            elif 0 <= y < amodal[idx].shape[0] and 0 <= x < amodal[idx].shape[1] and amodal[idx][y, x]:
                proposals.append((1, 0.0, obj_id, idx, "point_in_amodal"))
            elif cents[idx] is not None:
                cx, cy = cents[idx]
                dist = float(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5)
                if dist <= max_nearest_px:
                    proposals.append((2, dist, obj_id, idx, "nearest_centroid"))

    mapping, used_obj, used_idx = {}, set(), set()
    details = []
    for priority, dist, obj_id, idx, method in sorted(proposals):
        if obj_id in used_obj or idx in used_idx:
            continue
        mapping[obj_id] = idx
        used_obj.add(obj_id)
        used_idx.add(idx)
        details.append({
            "object_id": obj_id,
            "uoais_idx": int(idx),
            "method": method,
            "distance_px": round(float(dist), 1),
        })

    unmapped = [int(o["id"]) for o in objects if int(o["id"]) not in mapping]
    return mapping, details, unmapped



def load_json(path: Path):
    return json.load(open(path, "r", encoding="utf-8"))


def old_scene_key_by_image_id() -> dict[int, str]:
    raw = load_json(ID_MAP)
    out = {}
    for row in raw.get("mapping", []):
        if not isinstance(row, dict) or "image_id" not in row:
            continue
        out[int(row["image_id"])] = f"scene{row['scene_id']}/{row['view_id']}_rgb.png"
    return out


def resolve_object_name(image_id: int, obj_id: int, names: dict, id_to_scene_key: dict[int, str]) -> str | None:
    key = id_to_scene_key.get(int(image_id))
    if not key:
        return None
    rec = names.get(key, {}).get(str(int(obj_id)))
    if isinstance(rec, dict):
        return rec.get("name")
    return None


def make_case(row: dict, names: dict, id_to_scene_key: dict[int, str]) -> tuple[dict, dict | None]:
    image_id = int(row["image_id"])
    query_obj = int(row["query_object"])
    query_name = resolve_object_name(image_id, query_obj, names, id_to_scene_key)
    missing = None
    if not query_name:
        query_name = f"object {query_obj}"
        missing = {"image_id": image_id, "object_id": query_obj, "field": "query_object"}

    target_objects = []
    for obj_id in row.get("top_objects", []):
        obj_id = int(obj_id)
        name = resolve_object_name(image_id, obj_id, names, id_to_scene_key) or f"object {obj_id}"
        target_objects.append({"obj_id": obj_id, "object_name": name})

    case = {
        "case_name": f"img{image_id:06d}_q{query_obj}",
        "image_id": image_id,
        "query_object": {"obj_id": query_obj, "object_name": query_name},
        "target_objects": target_objects,
        "occlusion_paths": row.get("occlusion_paths", []),
        "difficulty": row.get("difficulty", "Unknown"),
        "new_difficulty": row.get("new_difficulty"),
        "k_min": row.get("k_min"),
        "num_paths": row.get("num_paths"),
        "source_gt": "UnoBench/gt_for_nlp.json",
    }
    return case, missing


def select_cases(limit_per_difficulty: int) -> tuple[list[dict], list[dict]]:
    rows = load_json(GT_FOR_NLP)
    names = load_json(NAME_FOR_ALL)
    id_to_scene_key = old_scene_key_by_image_id()
    selected, missing_names = [], []
    counts = Counter()
    for row in rows:
        diff = row.get("difficulty")
        if diff not in {"Easy", "Medium", "Hard"}:
            continue
        if counts[diff] >= limit_per_difficulty:
            continue
        case, missing = make_case(row, names, id_to_scene_key)
        selected.append(case)
        counts[diff] += 1
        if missing:
            missing_names.append(missing)
        if all(counts[d] >= limit_per_difficulty for d in ["Easy", "Medium", "Hard"]):
            break
    if any(counts[d] < limit_per_difficulty for d in ["Easy", "Medium", "Hard"]):
        raise ValueError(f"Could not select requested counts: {dict(counts)}")
    return selected, missing_names


def validate_selected(cases: list[dict]) -> list[dict]:
    missing = []
    for case in cases:
        image_path = IMAGE_DIR / f"image_{case['image_id']:06d}.png"
        ann_path = ANN_DIR / f"image_{case['image_id']:06d}.npy"
        if not image_path.exists():
            missing.append({"case": case["case_name"], "missing": str(image_path)})
        if not ann_path.exists():
            missing.append({"case": case["case_name"], "missing": str(ann_path)})
    return missing


def case_result_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = load_json(path)
    except Exception:
        return False
    return bool(data.get("raw_outputs")) and "pred_object_ids" in data


def run_uoais_reference_for_image(image_id: int, objects: list[dict], predictor, cfg, args, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"image_{int(image_id):06d}.json"
    if args.resume and cache_path.exists():
        return load_json(cache_path)

    rgb_path = IMAGE_DIR / f"image_{int(image_id):06d}.png"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Could not read RGB image: {rgb_path}")
    depth_mm = load_unobench_depth_mm(image_id)
    U.set_depth_norm_mode(args.depth_norm_mode, depth_mm)
    uoais = U.run_uoais(rgb, depth_mm, predictor, cfg)
    id_to_idx, mapping_details, unmapped_ids = map_detected_ids_to_uoais(
        objects,
        uoais,
        max_nearest_px=args.max_nearest_px,
    )

    idx_to_id = {idx: obj_id for obj_id, idx in id_to_idx.items()}
    mapped_indices = sorted(idx_to_id)
    edges_all = []
    if mapped_indices:
        vis = np.stack([uoais["visible_masks"][idx] for idx in mapped_indices])
        am = np.stack([uoais["amodal_masks"][idx] for idx in mapped_indices])
        occ = np.array([uoais["occluded"][idx] for idx in mapped_indices])
        local_to_obj = {local: idx_to_id[idx] for local, idx in enumerate(mapped_indices)}
        for edge in U.build_obstruction_edges(vis, am, depth_mm, occ, features=True):
            a = int(local_to_obj[int(edge["i"])])
            b = int(local_to_obj[int(edge["j"])])
            edges_all.append({
                "from": a,
                "to": b,
                "prob": round(float(edge["conf"]), 3),
                "p_cv": round(float(edge["p_cv"]), 3),
                "r_ij": round(float(edge["r_ij"]), 3),
                "valid_depth_ratio": round(float(edge["valid_depth_ratio"]), 3),
                "contact_px": int(edge["contact_px"]),
                "direct_contact_px": int(edge.get("direct_contact_px", edge["contact_px"])),
                "clearance_contact_px": int(edge.get("clearance_contact_px", 0)),
                "contact_mode": edge.get("contact_mode", "hidden_overlap"),
                "occlusion_ratio": round(float(edge["occlusion_ratio"]), 4),
                "amodal_px": int(edge["amodal_px"]),
                "hidden_px": int(edge["hidden_px"]),
                "depth_delta": round(float(edge["depth_delta"]), 1),
                "accepted_reason": edge.get("accepted_reason"),
            })

    payload = {
        "image_id": int(image_id),
        "n_uoais_instances": int(uoais.get("n", 0)),
        "id_mapping": mapping_details,
        "unmapped_detected_ids": unmapped_ids,
        "edges_all": edges_all,
    }
    json.dump(payload, open(cache_path, "w", encoding="utf-8"), indent=2)
    return payload


def score_reference_edges(edges_all: list[dict], args) -> list[dict]:
    """Edge scoring applied to the cached UOAIS edges before they reach Gemini.

    ``v2`` reuses run_uoais_pipeline_v0.rebuild_edges_v2: evidence-based direction
    resolution (explained_frac + depth) plus the soft gate with depth rescue, so
    each reciprocal pair contributes one hypothesis instead of two. ``raw`` keeps
    every candidate, which is what this runner did before.

    Like the pipeline, this runs at selection time on ``edges_all``; the UOAIS
    cache stays scoring-agnostic, so switching modes never forces a GPU re-run.
    """
    if getattr(args, "edge_scoring", "raw") == "raw":
        return edges_all
    return pipeline_v0.rebuild_edges_v2(edges_all, args)


def add_edge_scoring_args(ap: argparse.ArgumentParser) -> None:
    """v0 edge-scoring knobs; defaults mirror run_uoais_pipeline_v0.parse_args."""
    ap.add_argument(
        "--edge-scoring",
        choices=["raw", "v2"],
        default="v2",
        help="v2 (default) = run_uoais_pipeline_v0 edge scoring on the cached "
             "edges_all before building the Gemini reference table. raw = pass "
             "every UOAIS hypothesis through unfiltered (pre-v2 behaviour).",
    )
    ap.add_argument("--v2-min-hidden-px", type=int, default=50)
    ap.add_argument("--v2-min-explained-frac", type=float, default=0.05)
    ap.add_argument("--v2-abs-contact-px", type=int, default=150)
    ap.add_argument("--v2-keep-clearance", action="store_true")
    ap.add_argument("--v2-depth-weight", type=float, default=0.5)
    ap.add_argument("--v2-depth-sigma-mm", type=float, default=60.0)
    ap.add_argument("--v2-depth-rescue-mm", type=float, default=30.0,
                    help="Keep gated-out edges when depth_delta >= this margin (0 = off)")
    ap.add_argument("--v2-rescue-min-hidden-px", type=int, default=0)


def process_case(case: dict, args, out_root: Path, predictor, cfg) -> dict:
    case_name = case["case_name"]
    case_dir = out_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    result_path = case_dir / "result.json"
    if args.resume and case_result_valid(result_path):
        return load_json(result_path)

    image_path = IMAGE_DIR / f"image_{case['image_id']:06d}.png"
    image_bytes = image_path.read_bytes()
    image_pil = PILImage.open(image_path).convert("RGB")
    image_np = np.array(image_pil)

    det_path = case_dir / "det_objects.json"
    if args.resume and det_path.exists():
        objects = load_json(det_path)
    else:
        objects = legacy.detect_objects(image_bytes, image_pil.size)
        json.dump(objects, open(det_path, "w", encoding="utf-8"), indent=2)
    if not objects:
        raise RuntimeError("no_objects_detected")

    labeled_path = case_dir / "labeled.png"
    labeled_b64 = base.draw_labeled_image(image_np, objects, labeled_path)

    uoais_ref = run_uoais_reference_for_image(case["image_id"], objects, predictor, cfg, args, out_root / "uoais_cache")
    scored_edges = score_reference_edges(uoais_ref.get("edges_all", []), args)
    ref_edges = ref.build_uoais_reference_edges(
        scored_edges,
        target_id=int(case["query_object"]["obj_id"]),
        max_edges=args.max_ref_edges,
        min_contact_px=args.min_ref_contact_px,
    )
    ref_lines = ref.format_uoais_reference_lines(ref_edges)
    combined = ref.run_reasoning_once(
        labeled_b64,
        objects,
        case["query_object"]["object_name"],
        ref_lines,
        max_retries=args.max_reasoning_retries,
    )

    id_to_point = {
        int(o["id"]): {"label": o["label"], "x": int(o["x_px"]), "y": int(o["y_px"])}
        for o in objects
    }
    top_points = combined.get("top_points", [])
    top_points_with_ids = combined.get("top_points_with_ids", [])
    ev = ref.eval_result(case, top_points_with_ids)
    ann_mask = ref._annotation_mask(int(case["image_id"]))
    target_mapping = ref.map_badge_point_to_gt(ann_mask, combined.get("target_id"), id_to_point)
    target_gt_id = int(target_mapping.get("gt_id") or 0)
    query_obj_id = int(case["query_object"]["obj_id"])

    result = {
        "case": case_name,
        "image_id": int(case["image_id"]),
        "query": case["query_object"]["object_name"],
        "difficulty": case["difficulty"],
        "new_difficulty": case.get("new_difficulty"),
        "dataset": "unobench",
        "query_obj_id": None,
        "query_mask_id": query_obj_id,
        "gt_targets": [t["object_name"] for t in case["target_objects"]],
        "gt_action": ev["gt_action"],
        "pred_action": ev["action"],
        "pred_label": combined.get("pred_label", ""),
        "target_id": combined.get("target_id"),
        "target_label": combined.get("target_label", ""),
        "target_mapping": target_mapping,
        "target_grounding_correct": target_gt_id == query_obj_id,
        "pred_object_ids": sorted(ev["pred_ids_set"]),
        "gt_top_ids": sorted(ev["gt_top_ids"]),
        "confidence": combined.get("confidence"),
        "action_correct": ev["action_correct"],
        "target_matched": ev["target_matched"],
        "fully_correct": ev["fully_correct"],
        "occlusion_chain": combined.get("occlusion_chain", []),
        "candidates": combined.get("candidates", []),
        "free_objects": combined.get("free_objects", []),
        "chosen_to_remove": combined.get("chosen_to_remove", []),
        "candidate_ids": combined.get("candidate_ids", []),
        "free_object_ids": combined.get("free_object_ids", []),
        "chosen_to_remove_ids": combined.get("chosen_to_remove_ids", []),
        "invalid_ids": combined.get("invalid_ids", []),
        "id_to_point": id_to_point,
        "top_points": [[x, y, name] for x, y, name in top_points],
        "original_top_points": [[x, y, name] for x, y, name in top_points],
        "eval_top_points": ev["eval_top_points"],
        "top_points_with_ids": top_points_with_ids,
        "chosen_mapping": ev["chosen_mapping"],
        "mapping_policy": ref.MAPPING_POLICY,
        "mapping_radii_px": list(ref.MAPPING_RADII_PX),
        "id_space_note": ref.ID_SPACE_NOTE,
        "n_api_calls": 2,
        "raw_outputs": combined.get("raw_outputs", []),
        "det_objects": [
            {"id": o["id"], "label": o["label"], "x_px": o["x_px"], "y_px": o["y_px"]}
            for o in objects
        ],
        "adjacent_or_unclear": combined.get("adjacent_or_unclear", []),
        "uoais_reference_source": {
            "source": "computed_uoais_cache",
            "cache": str((out_root / "uoais_cache" / f"image_{int(case['image_id']):06d}.json").resolve()),
            "n_uoais_instances": uoais_ref.get("n_uoais_instances"),
            "id_mapping": uoais_ref.get("id_mapping", []),
            "unmapped_detected_ids": uoais_ref.get("unmapped_detected_ids", []),
            "edge_scoring": args.edge_scoring,
            "n_edges_all": len(uoais_ref.get("edges_all", [])),
            "n_edges_scored": len(scored_edges),
        },
        "uoais_reference_edges": ref_edges,
        "uoais_reference_prompt_lines": ref_lines.splitlines(),
        "uoais_supported_chosen_edges": ref.supported_edges(combined.get("occlusion_chain", []), ref_edges),
        "prompt_variant": ref.PROMPT_VARIANT,
        "prompt": combined.get("prompt", ""),
    }
    json.dump(result, open(result_path, "w", encoding="utf-8"), indent=2)
    return result


def build_gt_records(cases: list[dict]) -> list[dict]:
    return [
        {
            "image_id": int(case["image_id"]),
            "query_object": int(case["query_object"]["obj_id"]),
            "occlusion_paths": case.get("occlusion_paths", []),
            "top_objects": [int(t["obj_id"]) for t in case["target_objects"]],
            "new_difficulty": case["difficulty"],
        }
        for case in cases
    ]


def write_report(rows: list[dict], out_root: Path, args) -> None:
    summary = load_json(out_root / "summary.json")["summary"]
    eval_text = (out_root / "eval_results.txt").read_text(encoding="utf-8")
    lines = [
        "# Gemini + UOAIS Ref GT300 Pointing Report",
        "",
        f"- Cases: {len(rows)}",
        "- Selection: `UnoBench/gt_for_nlp.json`, old `difficulty`, first N per group.",
        "- Input: raw image -> Gemini pointing -> labeled image -> Gemini reasoning.",
        "- Prompt: non-incremental per case.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## evaluate_nlp.py",
        "",
        "```",
        eval_text.strip(),
        "```",
    ]
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    out_root = ROOT / args.out
    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases, missing_names = select_cases(args.limit_per_difficulty)
    counts = Counter(case["difficulty"] for case in cases)
    selection_manifest = {
        "source": str(GT_FOR_NLP),
        "difficulty_field": "difficulty",
        "requested_per_difficulty": args.limit_per_difficulty,
        "counts": dict(counts),
        "n": len(cases),
        "cases": [{"case": c["case_name"], "image_id": c["image_id"], "query_object": c["query_object"], "difficulty": c["difficulty"]} for c in cases],
    }
    json.dump(selection_manifest, open(out_root / "selection_manifest.json", "w", encoding="utf-8"), indent=2)
    json.dump(missing_names, open(out_root / "missing_names.json", "w", encoding="utf-8"), indent=2)

    legacy.ensure_images(cases, legacy.IMG_ZIP, IMAGE_DIR)
    legacy.ensure_depth(cases, legacy.DEPTH_DIR, legacy.META_ZIP, legacy.ID_MAP)
    legacy.ensure_annotations(cases, legacy.ANN_ZIP, ANN_DIR)
    missing_files = validate_selected(cases)
    if missing_files:
        json.dump(missing_files, open(out_root / "missing_files.json", "w", encoding="utf-8"), indent=2)
        raise FileNotFoundError(f"Missing {len(missing_files)} required files; see missing_files.json")

    if args.dry_run:
        print(json.dumps(selection_manifest, indent=2))
        print(f"missing_names={len(missing_names)}")
        return

    U.DEPTH_TOLERANCE_MM = args.depth_tolerance_mm
    U.CONTACT_DILATE_PX = args.contact_dilate_px
    U.MIN_CONTACT_PIXELS = args.min_contact_pixels
    U.MIN_VALID_DEPTH_RATIO = args.min_valid_depth_ratio
    predictor, cfg = U.load_uoais_predictor(U.CFG_RGBD, args.score_thresh, args.nms_thresh, args.device)

    rows, failed = [], []
    started = time.time()
    for idx, case in enumerate(cases, 1):
        try:
            if not args.compact_log:
                legacy.logger.info("[%03d/%03d] %s query=%r", idx, len(cases), case["case_name"], case["query_object"]["object_name"])
            result = process_case(case, args, out_root, predictor, cfg)
            rows.append(result)
        except Exception as exc:
            failed.append({"case": case["case_name"], "image_id": case["image_id"], "error": str(exc)})
            legacy.logger.exception("Failed case %s: %s", case["case_name"], exc)
        if args.batch_size > 0 and (idx % args.batch_size == 0 or idx == len(cases)):
            done = len(rows)
            legacy.logger.info("Batch checkpoint: processed=%d/%d succeeded=%d failed=%d elapsed=%.1fs", idx, len(cases), done, len(failed), time.time() - started)
            json.dump(failed, open(out_root / "failed_cases.json", "w", encoding="utf-8"), indent=2)

    json.dump(failed, open(out_root / "failed_cases.json", "w", encoding="utf-8"), indent=2)
    rows = []
    for case in cases:
        result_path = out_root / case["case_name"] / "result.json"
        if result_path.exists():
            rows.append(load_json(result_path))

    gt_records = build_gt_records(cases)
    ref.write_predictions_and_eval(rows, gt_records, out_root)
    summary = legacy.summarize(rows, out_root)
    write_report(rows, out_root, args)
    print(json.dumps(summary, indent=2))
    print(f"Wrote report: {out_root / 'report.md'}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/gemini_uoais_ref")
    ap.add_argument("--limit-per-difficulty", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--compact-log", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-ref-edges", type=int, default=40)
    ap.add_argument("--min-ref-contact-px", type=int, default=1)
    ap.add_argument("--max-reasoning-retries", type=int, default=2)
    ap.add_argument("--score-thresh", type=float, default=0.35)
    ap.add_argument("--nms-thresh", type=float, default=0.7)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--depth-norm-mode", choices=["uoais_default", "percentile"], default="percentile")
    ap.add_argument("--depth-tolerance-mm", type=float, default=90.0)
    ap.add_argument("--contact-dilate-px", type=int, default=8)
    ap.add_argument("--min-contact-pixels", type=int, default=1)
    ap.add_argument("--min-valid-depth-ratio", type=float, default=0.5)
    ap.add_argument("--max-nearest-px", type=float, default=120.0)
    add_edge_scoring_args(ap)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
