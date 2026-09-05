#!/usr/bin/env python3
"""
Calibrate the VLM and 3D obstruction-edge scores, fuse them in logit space, then
re-run graph reasoning and score the fused graph.

Inputs
------
- VLM log   : a run_gemini_uoais_ref[_batch].py output dir. Each case contributes
              its ``occlusion_chain`` edges; ``edge_confidence`` (0-100) is the raw
              edge probability and ``N_cand = len(candidates)`` the scene-context
              feature the VLM calibrators were trained with. Badge ids are mapped
              to GT-annotation-mask ids so both sources share one id space.
- 3D log    : a run_uoais_pipeline*.py output dir. Per image, the v2 edge set is
              rebuilt from the cached ``edges_all`` (identical to what the pipeline
              selects at run time), ``prob`` is the raw edge probability and
              ``N_cand = n_uoais`` the context feature the 3D calibrators used.

Calibration (calibration/): ``raw`` | ``global`` (Platt) | ``adaptive``
(N_cand-adaptive Platt). VLM edges use the VLM models, 3D edges the *_3d models.

Fusion, per case and per ordered pair (occluder -> occluded):

    logit(p_fuse) = logit(p_vlm_cal) + logit(p_3d_cal)      # edge in both sources
    p_fuse        = p_of_the_only_source                    # edge in one source

Graph reasoning runs on the fused edges kept by ``--threshold`` (a calibrated
probability has a meaningful 0.5 decision point); the reported edge/object
probabilities themselves are threshold-free.

Outputs (per config dir): predictions.jsonl, gt_eval.json, eval_results.txt,
edge_pairs_fused.json, object_pairs_fused.json, calibration_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import graph_eval_lib as G
import gemini_uoais_lib as ref
import run_uoais_pipeline_v0 as P
from uoais_graph import format_model_output, select_for_target

ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "calibration"
EPS = 1e-6

V2_DEFAULTS = argparse.Namespace(
    v2_min_hidden_px=50, v2_min_explained_frac=0.05, v2_abs_contact_px=150,
    v2_keep_clearance=False, v2_depth_weight=0.5, v2_depth_sigma_mm=60.0,
    v2_depth_rescue_mm=30.0, v2_rescue_min_hidden_px=0,
)


# ── calibration ────────────────────────────────────────────────────────────────

def _logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def load_models() -> dict:
    def rd(name):
        return json.loads((CALIB_DIR / name).read_text(encoding="utf-8"))
    return {
        ("vlm", "global"): rd("global_platt_model.json"),
        ("vlm", "adaptive"): rd("adaptive_platt_model.json"),
        ("3d", "global"): rd("global_platt_3d_model.json"),
        ("3d", "adaptive"): rd("adaptive_platt_3d_model.json"),
    }


def calibrate(prob: float, n_cand: int, model: dict | None) -> float:
    """Platt / N_cand-adaptive Platt, matching calibration/infer_*.py."""
    if model is None:
        return min(max(float(prob), EPS), 1.0 - EPS)
    p = min(max(float(prob), EPS), 1.0 - EPS)
    if model["model_type"] == "global_platt":
        return _sigmoid(float(model["a"]) * _logit(p) + float(model["c"]))
    phi = (math.log1p(max(int(n_cand), 0)) - float(model["mu_phi"])) / float(model["sigma_phi"])
    exponent = min(max(float(model["alpha0"]) + float(model["alphaN"]) * phi, -5.0), 5.0)
    return _sigmoid(math.exp(exponent) * _logit(p) + float(model["c0"]) + float(model["cN"]) * phi)


# ── edge extraction ────────────────────────────────────────────────────────────

def vlm_edges_by_case(vlm_log: Path, mask_loader) -> dict[tuple[int, int], dict]:
    """{(image_id, query_object): {(occluder_gt, occluded_gt): (p_raw, N_cand)}}."""
    per_case = json.loads((vlm_log / "summary.json").read_text(encoding="utf-8"))["per_case"]
    out: dict[tuple[int, int], dict] = {}
    unmapped = 0
    for row in per_case:
        image_id, target = int(row["image_id"]), int(row["query_mask_id"])
        n_cand = len(row.get("candidates") or [])
        id_to_point = {int(k): v for k, v in (row.get("id_to_point") or {}).items()}
        mask = mask_loader(image_id)
        edges: dict[tuple[int, int], tuple[float, int]] = {}
        for edge in row.get("occlusion_chain") or []:
            prob = edge.get("edge_confidence")
            if prob is None:
                continue
            src = G._to_int(edge.get("occluder_id"))
            dst = G._to_int(edge.get("occluded_id"))
            if src is None or dst is None:
                continue
            src_gt = int((ref.map_badge_point_to_gt(mask, src, id_to_point) or {}).get("gt_id") or 0)
            dst_gt = int((ref.map_badge_point_to_gt(mask, dst, id_to_point) or {}).get("gt_id") or 0)
            if src_gt <= 0 or dst_gt <= 0 or src_gt == dst_gt:
                unmapped += 1
                continue
            p = float(prob) / 100.0 if float(prob) > 1.5 else float(prob)
            prev = edges.get((src_gt, dst_gt))
            if prev is None or p > prev[0]:
                edges[(src_gt, dst_gt)] = (p, n_cand)
        out[(image_id, target)] = edges
    return out, unmapped


def uoais_edges_by_image(uoais_log: Path) -> dict[int, dict]:
    """{image_id: {"edges": {(from,to): p_raw}, "n_uoais": int, "object_points": {...}}}."""
    out = {}
    for path in sorted((uoais_log / "image_graphs").glob("image_*.json")):
        g = json.loads(path.read_text(encoding="utf-8"))
        out[int(g["image_id"])] = {
            "edges": {(int(e["from"]), int(e["to"])): float(e["prob"])
                      for e in P.rebuild_edges_v2(g["edges_all"], V2_DEFAULTS)},
            "n_uoais": int(g["n_uoais"]),
            "object_points": {int(k): v for k, v in (g.get("object_points") or {}).items()},
        }
    return out


# ── fusion + reasoning ─────────────────────────────────────────────────────────

def fuse(vlm: dict, uoais: dict, models: dict, vlm_cal: str, uoais_cal: str) -> dict:
    """{(occluder, occluded): {p_fuse, p_vlm, p_3d, source}}."""
    vlm_model = models.get(("vlm", vlm_cal))
    uoais_model = models.get(("3d", uoais_cal))
    fused = {}
    for pair in set(vlm) | set(uoais):
        p_v = calibrate(*vlm[pair], vlm_model) if pair in vlm else None
        p_3 = calibrate(uoais[pair][0], uoais[pair][1], uoais_model) if pair in uoais else None
        if p_v is not None and p_3 is not None:
            p, source = _sigmoid(_logit(p_v) + _logit(p_3)), "both"
        elif p_v is not None:
            p, source = p_v, "vlm_only"
        else:
            p, source = p_3, "3d_only"
        fused[pair] = {"p_fuse": p, "p_vlm": p_v, "p_3d": p_3, "source": source}
    return fused


def object_probabilities(target: int, fused: dict) -> dict[int, float]:
    """P(object must be removed) = best path reliability from it down to the target.

    Reliability of a path is the product of its fused edge probabilities; an
    object's score is the maximum over all its paths to the target. Threshold-free
    on purpose, so the reported object calibration does not depend on --threshold.
    """
    blockers_of = defaultdict(list)
    for (src, dst), rec in fused.items():
        blockers_of[dst].append((src, rec["p_fuse"]))

    best: dict[int, float] = {}
    stack = [(src, p) for src, p in blockers_of.get(target, [])]
    while stack:
        node, score = stack.pop()
        if score <= best.get(node, 0.0):
            continue
        best[node] = score
        for parent, p in blockers_of.get(node, []):
            if parent != node:
                stack.append((parent, score * p))
    return best


def run_case(case_gt: dict, fused: dict, object_points: dict, threshold: float) -> dict:
    target = int(case_gt["query_object"])
    edges = [{"from": a, "to": b} for (a, b), rec in sorted(fused.items()) if rec["p_fuse"] >= threshold]
    selection = select_for_target(target, edges)
    return {
        "pred_paths": selection["pred_paths"],
        "chosen_to_remove_ids": selection["chosen_to_remove_ids"],
        "candidate_ids": selection["candidate_ids"],
        "model_output": format_model_output(
            selection["pred_paths"] or [[target]],
            selection["chosen_to_remove_ids"] or [],
            object_points,
        ),
    }


# ── metrics ────────────────────────────────────────────────────────────────────

def calibration_metrics(pairs: list[tuple[float, int]], n_bins: int = 10) -> dict:
    if not pairs:
        return {"n": 0}
    n = len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    nll = sum(-(y * math.log(min(max(p, EPS), 1 - EPS)) + (1 - y) * math.log(1 - min(max(p, EPS), 1 - EPS)))
              for p, y in pairs) / n
    bins = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    ece = sum(len(b) / n * abs(sum(y for _, y in b) / len(b) - sum(p for p, _ in b) / len(b))
              for b in bins if b)
    return {"n": n, "pos_rate": sum(y for _, y in pairs) / n, "nll": nll, "brier": brier, "ece": ece}


def metrics_by_group(rows: list[dict]) -> dict:
    out = {"overall": calibration_metrics([(r["p"], r["label"]) for r in rows])}
    for group in sorted({r["difficulty"] for r in rows}):
        sel = [(r["p"], r["label"]) for r in rows if r["difficulty"] == group]
        out[group] = calibration_metrics(sel)
    return out


def run_evaluate_nlp(pred_path: Path, gt_path: Path, out_path: Path) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "evaluate_nlp.py"),
         "--pred_path", str(pred_path), "--gt_path", str(gt_path),
         "--npz_root", str(ROOT / "UnoBench/annotations"), "--dataset_type", "synthetic"],
        cwd=str(ROOT), text=True, capture_output=True)
    text = proc.stdout + ("\nSTDERR:\n" + proc.stderr if proc.stderr else "")
    out_path.write_text(text, encoding="utf-8")
    return text


# ── main ───────────────────────────────────────────────────────────────────────

def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = json.loads(Path(args.gt_path).read_text(encoding="utf-8"))
    models = load_models()
    mask_loader = G.MaskLoader()

    vlm_by_case, n_unmapped = vlm_edges_by_case(Path(args.vlm_log), mask_loader)
    uoais_by_image = uoais_edges_by_image(Path(args.uoais_log))
    print(f"[fuse] VLM cases={len(vlm_by_case)} (dropped {n_unmapped} unmappable badge edges) "
          f"| 3D images={len(uoais_by_image)}", flush=True)

    edge_rows, object_rows, gt_eval = [], [], []
    fused_by_case, points_by_case = {}, {}
    source_counts = defaultdict(int)
    for row in gt_rows:
        image_id, target = int(row["image_id"]), int(row["query_object"])
        difficulty = row.get(args.difficulty_field) or row.get("difficulty")
        image = uoais_by_image.get(image_id, {"edges": {}, "n_uoais": 0, "object_points": {}})
        vlm = vlm_by_case.get((image_id, target), {}) if args.sources in ("both", "vlm") else {}
        uoais = ({pair: (p, image["n_uoais"]) for pair, p in image["edges"].items()}
                 if args.sources in ("both", "3d") else {})
        fused = fuse(vlm, uoais, models, args.vlm_cal, args.uoais_cal)
        for rec in fused.values():
            source_counts[rec["source"]] += 1

        gt_edges = G.gt_edges_from_paths(row.get("occlusion_paths") or [])
        for (src, dst), rec in fused.items():
            edge_rows.append({
                "p": rec["p_fuse"],
                "label": int(G.Edge(occluder_id=src, occluded_id=dst) in gt_edges),
                "difficulty": difficulty, "image_id": image_id, "query_object": target,
                "occluder_id": src, "occluded_id": dst, "source": rec["source"],
                "p_vlm": rec["p_vlm"], "p_3d": rec["p_3d"],
            })
        gt_top = {int(x) for x in row.get("top_objects") or []}
        for obj, p in object_probabilities(target, fused).items():
            object_rows.append({
                "p": p, "label": int(obj in gt_top), "difficulty": difficulty,
                "image_id": image_id, "query_object": target, "object_id": obj,
            })

        fused_by_case[(image_id, target)] = fused
        points_by_case[(image_id, target)] = image["object_points"]
        gt_eval.append({"image_id": image_id, "query_object": target,
                        "occlusion_paths": row.get("occlusion_paths") or [],
                        "top_objects": sorted(gt_top), "new_difficulty": difficulty,
                        "difficulty": difficulty})

    gt_path = out_dir / "gt_eval.json"
    json.dump(gt_eval, gt_path.open("w", encoding="utf-8"), indent=2)
    json.dump(edge_rows, (out_dir / "edge_pairs_fused.json").open("w", encoding="utf-8"), indent=1)
    json.dump(object_rows, (out_dir / "object_pairs_fused.json").open("w", encoding="utf-8"), indent=1)

    metrics = {
        "config": {"vlm_cal": args.vlm_cal, "uoais_cal": args.uoais_cal,
                   "vlm_log": args.vlm_log, "uoais_log": args.uoais_log, "gt_path": args.gt_path},
        "edge_sources": dict(source_counts),
        "edge_level": metrics_by_group(edge_rows),
        "object_level": metrics_by_group(object_rows),
        "thresholds": {},
    }

    thresholds = [float(t) for t in str(args.thresholds).split(",")] if args.thresholds else [args.threshold]
    for threshold in thresholds:
        sub = out_dir if len(thresholds) == 1 else out_dir / f"thr_{threshold:.2f}"
        sub.mkdir(parents=True, exist_ok=True)
        pred_path = sub / "predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in gt_eval:
                key = (int(row["image_id"]), int(row["query_object"]))
                result = run_case(row, fused_by_case[key], points_by_case[key], threshold)
                f.write(json.dumps({"image": f"images/image_{key[0]:06d}.png", "image_id": key[0],
                                    "query_object": key[1], "model_output": result["model_output"]},
                                   ensure_ascii=False) + "\n")
        eval_text = run_evaluate_nlp(pred_path, gt_path, sub / "eval_results.txt")
        balanced = next((line.split("=")[-1].strip() for line in eval_text.splitlines()
                         if "Balanced SR-F1" in line), None)
        metrics["thresholds"][f"{threshold:.2f}"] = {"balanced_sr_f1": balanced,
                                                     "eval_dir": str(sub)}
        print(f"########## threshold={threshold:.2f}  balanced_sr_f1={balanced}")
        print(eval_text)

    json.dump(metrics, (out_dir / "calibration_metrics.json").open("w", encoding="utf-8"), indent=2)
    print(json.dumps({k: metrics[k] for k in ("edge_sources", "edge_level", "object_level")}, indent=2))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm-log", default="logs/gemini_uoais_ref_test_1800")
    ap.add_argument("--uoais-log", default="logs/uoais_1800_v2")
    ap.add_argument("--gt-path", default="UnoBench/subset_difficulty/test_GT_small_1800.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vlm-cal", choices=["raw", "global", "adaptive"], default="global")
    ap.add_argument("--uoais-cal", choices=["raw", "global", "adaptive"], default="global")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--thresholds", default=None,
                    help="comma-separated sweep, e.g. 0.1,0.2,0.3 (writes thr_XX/ subdirs)")
    ap.add_argument("--difficulty-field", choices=["difficulty", "new_difficulty"], default="difficulty")
    ap.add_argument("--sources", choices=["both", "vlm", "3d"], default="both",
                    help="ablation: restrict the fused graph to one source (same reasoning + threshold)")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
