#!/usr/bin/env python3
"""
Shared library for the Gemini + UOAIS-reference pipeline (prompts, UOAIS
reference-table formatting, reasoning call, point-to-mask mapping, and eval
output writing). The runnable entry point is run_gemini_uoais_ref.py.

This keeps the main result schema intact, but skips the
Gemini detection call by reusing cached detections/labeled images from a vanilla
run. UOAIS/3D is advisory only: pair geometry is added to the prompt as a compact
reference table; Gemini still decides target, edges, candidates, free objects,
and chosen_to_remove_ids.
"""
from __future__ import annotations

import base64
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from google.genai import types as gtypes

import unobench_gemini_common as legacy


ROOT = Path(__file__).resolve().parent
ANN_DIR = ROOT / "UnoBench/annotations"
MAPPING_RADII_PX = [3, 5, 10, 20, 40, 80]
MAPPING_POLICY = "point_exact_then_radius_fallback"
ID_SPACE_NOTE = (
    "gt_top_ids/query_mask_id/pred_object_ids are GT annotation mask-space; "
    "target_id/chosen_to_remove_ids/candidate_ids/free_object_ids are detected badge-space."
)


PROMPT_VARIANT = "gemini_uoais_ref_v3_direction_first_occlusion"


_SYSTEM_PROMPT = """\
You are a robotic grasp-planning expert reasoning about CAMERA-VIEW occlusion.

Objects in the image are marked with numbered ID badges. Identify every object
by its badge number. Do not use or output coordinates.

## Definition (strict, camera-view)
Object A OCCLUDES object B if, from the camera viewpoint, A is nearer the camera
than B AND A's visible region overlaps B's true (amodal) extent, so that part of
B's outline is hidden or cut by A. Direction is written B -> A.

This is VISUAL OCCLUSION, not "makes grasping harder":
- Two objects sitting side by side and touching, with neither in front of and
  over the other, do NOT occlude each other, even when close.
- Contact alone is not occlusion. Only front-overlap that hides part of B counts.
- Very small front-overlap can be real occlusion even when it is hard to see
  directly in the image.

## UOAIS geometry reference (hypotheses, not conclusions)
Each row is a CANDIDATE pair from a geometric pass. Treat every row as a
hypothesis about direction, not as a final conclusion. A row "A -> B" means:
UOAIS hypothesizes that badge A may occlude badge B. In the JSON occlusion_chain,
write that same relation as occluded_id=B, occluder_id=A.

Columns:
- occ = contact_px / amodal_px of the blocked object. Higher = more of B covered.
- hidden_cov = direct_contact_px / hidden_px. Higher = the candidate occluder
  sits directly over B's hidden region -> strong evidence of real front-overlap.
- clear_amodal = clearance-contact area / amodal_px. This is side/clearance
  contact, i.e. ADJACENCY. High clear_amodal with low occ and low hidden_cov
  means touching-not-occluding -> reject.
- p_cv = UOAIS mask probability; r = valid-depth reliability. Low r means depth
  is unreliable (transparent/reflective) -> LOWER confidence, never raise it.
- dz/depth_delta describes front-back consistency. Use it to decide direction:
  the occluder should be nearer the camera than the occluded object, allowing
  small noise.

Reading the geometry (label definition: ANY camera-view amodal overlap >=1%
counts as occlusion, no matter how small):
- A small occlusion_ratio (1-5%) is COMMON for true occlusions in this data.
  Never reject a pair merely because occ is small.
- Adjacency signature (reject): hidden_cov == 0 AND direct_contact_px is about 0,
  i.e. contact is clearance-only. This is the ONLY geometry-based reject.
- Direction test (the main check): an edge "B occluded by A" is plausible when
  depth_delta indicates A is nearer the camera than B at the contact region. If
  depth contradicts the stated direction, prefer the mirror direction if that
  mirror edge is present in the candidate table; otherwise mark the pair unclear.
- The image is used to arbitrate DIRECTION and identity, not to demand visible
  front-overlap: at 1-3% ratios the overlap is often too small to see. Absence
  of visible overlap is NOT evidence of absence when hidden_cov > 0 and depth
  supports the direction.

## Confidence calibration
For each accepted edge, output p_obstruct in [0,100]:
- 85-100: hidden_cov > 0, depth supports direction, AND overlap visible.
- 60-84: hidden_cov > 0 and depth supports direction; overlap too small to see.
- 35-59: geometry mixed (e.g. depth unreliable, r low) or direction ambiguous.
- 0-34: clearance-only contact, or depth clearly contradicts the direction.
Rules:
- Do NOT default to high values. Reserve 85-100 for visually verified occlusion.
- If hidden_cov > 0 and depth supports direction, do NOT cap the score merely
  because the visible overlap is hard to see.
- If r is low for the pair, reduce p_obstruct.
- Describe the direction/geometry evidence BEFORE committing to a number.
- Pairs you cannot resolve go in adjacent_or_unclear, not the occlusion chain.

## Output (JSON only, no markdown)
{
  "target_id": <detected object ID>,
  "occlusion_chain": [
    {
      "occluded_id": <detected object ID>,
      "occluder_id": <detected object ID>,
      "visual_evidence": "<where on the occluded object it is covered, and what visibly sits in front there>",
      "p_obstruct": <0-100>,
      "edge_factors": {
        "front_overlap_visible": <0-100>,
        "occluder_in_front": <0-100>,
        "outline_cut": <0-100>
      }
    }
  ],
  "adjacent_or_unclear": [
    {
      "occluded_id": <id>,
      "occluder_id": <id>,
      "reason_not_occlusion": "<why this UOAIS candidate is adjacency, unconfirmed, or badge_unreadable>"
    }
  ],
  "candidates": [
    {"id": <id>, "prob_free": <0-100>}
  ],
  "free_object_ids": [<detected object ID>, ...],
  "chosen_to_remove_ids": [<detected object ID>, ...]
}

## Rules
- Identify the exact target ID first. Use left/right/top/bottom/middle wording in
  your reasoning to disambiguate duplicate labels, but output only the ID.
- Every ID you output must be a badge number you can actually read in the image.
  If a UOAIS row references an object whose badge is unreadable or hidden, put the
  pair in adjacent_or_unclear with reason "badge_unreadable" rather than guessing.
- occlusion_chain contains only CONFIRMED occluders on a path to the target;
  direction occluded_id -> occluder_id.
- Every UOAIS candidate pair you were given must appear in EXACTLY ONE of
  occlusion_chain or adjacent_or_unclear.
- If the target has no confirmed occluder: occlusion_chain = [] and
  chosen_to_remove_ids = [target_id].
- If blocked: chosen_to_remove_ids = only the top-level, directly graspable
  occluders (occluders that are themselves not occluded) on a path to the target.
- Use only detected IDs listed in the user message. Do not output coordinates.
"""


_USER_PROMPT_TMPL = """\
Task: pick up "{query_name}"

Detected objects (badge_id: label):
{obj_list}

UOAIS geometry candidates for THIS scene. Each row is a hypothesis to confirm or
reject against the image:
{uoais_reference}

[image]
"""


def load_json(path: Path):
    return json.load(open(path, "r", encoding="utf-8"))


def object_list(objects):
    return "\n".join(
        f"{o['id']}: {o['label']}"
        for o in objects
    )








def _ref_priority(edge, target_id):
    direct_target = int(edge.get("to", -1)) == int(target_id)
    touches_target = direct_target or int(edge.get("from", -1)) == int(target_id)
    direct_mode = edge.get("contact_mode") == "hidden_overlap"
    return (
        1 if direct_target else 0,
        1 if touches_target else 0,
        1 if direct_mode else 0,
        float(edge.get("contact_px", 0)),
        float(edge.get("p_cv", edge.get("prob", 0))),
    )


def build_uoais_reference_edges(ref_edges, target_id, max_edges=40, min_contact_px=1):
    cleaned = []
    seen = set()
    for edge in ref_edges or []:
        src = legacy._to_int(edge.get("from"))
        dst = legacy._to_int(edge.get("to"))
        if src is None or dst is None or src == dst:
            continue
        contact_px = int(edge.get("contact_px") or 0)
        if contact_px < min_contact_px:
            continue
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({
            "from": src,
            "to": dst,
            "mode": edge.get("contact_mode", "unknown"),
            "occlusion_ratio": round(float(edge.get("occlusion_ratio", 0.0)), 4),
            "contact_px": contact_px,
            "direct_contact_px": int(edge.get("direct_contact_px") or 0),
            "clearance_contact_px": int(edge.get("clearance_contact_px") or 0),
            "amodal_px": int(edge.get("amodal_px") or 0),
            "hidden_px": int(edge.get("hidden_px") or 0),
            "p_cv": round(float(edge.get("p_cv", edge.get("prob", 0.0))), 3),
            "r_ij": round(float(edge.get("r_ij", 0.0)), 3),
            "depth_delta": round(float(edge.get("depth_delta", 0.0)), 1),
            "accepted_reason": edge.get("accepted_reason", ""),
        })
        last = cleaned[-1]
        last["hidden_cov"] = round(
            last["direct_contact_px"] / max(1, last["hidden_px"]),
            4,
        )
        last["contact_over_amodal"] = round(
            last["contact_px"] / max(1, last["amodal_px"]),
            4,
        )
        last["clearance_over_amodal"] = round(
            last["clearance_contact_px"] / max(1, last["amodal_px"]),
            4,
        )
    cleaned.sort(key=lambda e: _ref_priority(e, target_id), reverse=True)
    return cleaned[:max_edges]


def format_uoais_reference_lines(edges):
    if not edges:
        return "- none"
    lines = []
    for edge in edges:
        lines.append(
            "- {src} -> {dst} | mode={mode} | occ={occ:.4f} | hidden_cov={hidden_cov:.4f} "
            "| clear_amodal={clear_amodal:.4f} | contact={contact} | direct={direct} "
            "| clear={clear} | p_cv={pcv:.3f} | r={r:.3f} | dz={dz:.1f}mm".format(
                src=edge["from"],
                dst=edge["to"],
                mode=edge["mode"],
                occ=edge["occlusion_ratio"],
                hidden_cov=edge["hidden_cov"],
                clear_amodal=edge["clearance_over_amodal"],
                contact=edge["contact_px"],
                direct=edge["direct_contact_px"],
                clear=edge["clearance_contact_px"],
                pcv=edge["p_cv"],
                r=edge["r_ij"],
                dz=edge["depth_delta"],
            )
        )
    return "\n".join(lines)




def _json_obj_from_raw(raw):
    raw_clean = re.sub(r"^```[a-z]*\n?", "", (raw or "").strip(), flags=re.MULTILINE)
    raw_clean = re.sub(r"```$", "", raw_clean.strip())
    m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except Exception:
        return {}


def parse_reasoning_json(raw, objects):
    data = _json_obj_from_raw(raw)
    adjacent = data.get("adjacent_or_unclear") if isinstance(data, dict) else []
    if isinstance(data, dict) and data:
        for edge in data.get("occlusion_chain") or []:
            if not isinstance(edge, dict):
                continue
            if edge.get("edge_confidence") is None and edge.get("p_obstruct") is not None:
                edge["edge_confidence"] = edge.get("p_obstruct")
            occluded_id = legacy._to_int(edge.get("occluded_id"))
            occluder_id = legacy._to_int(edge.get("occluder_id"))
            if not edge.get("occluded") and occluded_id is not None:
                edge["occluded"] = legacy._object_label(objects, occluded_id)
            if not edge.get("occluder") and occluder_id is not None:
                edge["occluder"] = legacy._object_label(objects, occluder_id)
        if not data.get("candidate_ids"):
            data["candidate_ids"] = [
                legacy._to_int(c.get("id"))
                for c in data.get("candidates") or []
                if isinstance(c, dict) and legacy._to_int(c.get("id")) is not None
            ]
        parsed = legacy.parse_combined_json(json.dumps(data), objects)
    else:
        parsed = legacy.parse_combined_json(raw, objects)
    parsed["adjacent_or_unclear"] = adjacent if isinstance(adjacent, list) else []
    return parsed


def run_reasoning_once(labeled_b64, objects, query_name, ref_lines, max_retries=1):
    user_prompt = _USER_PROMPT_TMPL.format(
        obj_list=object_list(objects),
        query_name=query_name,
        uoais_reference=ref_lines,
    )
    contents = [
        _SYSTEM_PROMPT,
        user_prompt,
        gtypes.Part.from_bytes(data=base64.b64decode(labeled_b64), mime_type="image/png"),
    ]
    raw_outputs = []
    parsed = None
    for attempt in range(int(max_retries) + 1):
        try:
            resp = legacy.base.generate_content(
                model=legacy.base.MODEL,
                contents=contents,
                config=gtypes.GenerateContentConfig(temperature=0.0),
            )
            raw = (resp.text or "").strip()
            if raw:
                raw_outputs.append(raw)
                parsed = parse_reasoning_json(raw, objects)
                break
        except Exception as exc:
            if attempt >= int(max_retries):
                legacy.logger.warning("  [UOAIS-ref reasoning] failed: %s", exc)
            else:
                legacy.logger.warning("  [UOAIS-ref reasoning] retry after error: %s", exc)
    if parsed is None:
        parsed = {
            "pred_label": "",
            "confidence": None,
            "target_id": None,
            "target_label": "",
            "occlusion_chain": [],
            "candidates": [],
            "free_objects": [],
            "chosen_to_remove": [],
            "candidate_ids": [],
            "free_object_ids": [],
            "chosen_to_remove_ids": [],
            "invalid_ids": [],
            "top_points": [],
            "top_points_with_ids": [],
        }
    return {**parsed, "raw_outputs": raw_outputs, "prompt": _SYSTEM_PROMPT + "\n\n" + user_prompt}


def run_evaluate_nlp(pred_path, gt_path, out_path):
    """Run the packaged evaluate_nlp.py on a predictions file and save its report."""
    import subprocess, sys
    proc = subprocess.run(
        [
            sys.executable, str(ROOT / "evaluate_nlp.py"),
            "--pred_path", str(pred_path),
            "--gt_path", str(gt_path),
            "--npz_root", str(ANN_DIR),
            "--dataset_type", "synthetic",
        ],
        capture_output=True,
        text=True,
    )
    eval_text = proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
    out_path.write_text(eval_text, encoding="utf-8")


def points_to_answer(points):
    answer_parts = [f'<points {x} {y}>{name}</points>' for x, y, name in points]
    return f'<answer>[{", ".join(answer_parts)}]</answer>'


def build_think_from_chain(row, mask):
    """Synthesize a <think> block in the evaluator's path grammar from the
    VLM occlusion_chain, so occlusion-reasoning P/R/F1 and MP-NED are scored.

    Points are snapped to the nearest annotated pixel (same fallback as the
    answer points) so the evaluator's exact-pixel lookup lands on the object.
    """
    id_to_point = row.get("id_to_point") or {}
    pts = {}
    for badge, pt in id_to_point.items():
        badge = legacy._to_int(badge)
        if badge is None:
            continue
        mapped = _map_point_to_gt(mask, pt["x"], pt["y"])
        ex, ey = mapped["eval_point"]
        pts[badge] = (int(ex), int(ey), pt.get("label") or f"object {badge}")

    target_id = legacy._to_int(row.get("target_id"))
    adj = {}
    for edge in row.get("occlusion_chain") or []:
        occluded = legacy._to_int(edge.get("occluded_id"))
        occluder = legacy._to_int(edge.get("occluder_id"))
        if occluded in pts and occluder in pts and occluded != occluder:
            adj.setdefault(occluded, [])
            if occluder not in adj[occluded]:
                adj[occluded].append(occluder)

    if target_id not in pts:
        return ""

    def walk(node, visited):
        nexts = [n for n in adj.get(node, []) if n not in visited]
        if not nexts:
            yield [node]
            return
        for n in nexts:
            for rest in walk(n, visited | {n}):
                yield [node] + rest

    def fmt(badge):
        x, y, name = pts[badge]
        return f"{name} at ({x}, {y})"

    paths = [p for p in walk(target_id, {target_id}) if len(p) >= 2]
    if not paths:
        return f"<think>{fmt(target_id)} is not occluded.</think>"

    parts = []
    for i, path in enumerate(paths, 1):
        rels = ". ".join(
            f"{fmt(path[j])} is occluded by {fmt(path[j + 1])}"
            for j in range(len(path) - 1)
        )
        parts.append(f"Path{i}: {rels}.")
    return "<think>" + " ".join(parts) + "</think>"


def _annotation_mask(image_id: int) -> np.ndarray | None:
    path = ANN_DIR / f"image_{int(image_id):06d}.npy"
    if not path.exists():
        return None
    return np.load(path).astype(int)


def _map_point_to_gt(mask: np.ndarray | None, x, y, radii_px=MAPPING_RADII_PX):
    x_int, y_int = int(round(float(x))), int(round(float(y)))
    if mask is None:
        return {
            "gt_id": 0,
            "method": "annotation_missing",
            "radius_px": None,
            "point": [x_int, y_int],
            "eval_point": [x_int, y_int],
        }
    h, w = mask.shape
    if 0 <= x_int < w and 0 <= y_int < h:
        obj_id = int(mask[y_int, x_int])
        if obj_id > 0:
            return {
                "gt_id": obj_id,
                "method": "exact",
                "radius_px": 0,
                "point": [x_int, y_int],
                "eval_point": [x_int, y_int],
            }

    for radius in radii_px:
        y0, y1 = max(0, y_int - radius), min(h, y_int + radius + 1)
        x0, x1 = max(0, x_int - radius), min(w, x_int + radius + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        patch = mask[y0:y1, x0:x1]
        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            continue
        abs_x = xs + x0
        abs_y = ys + y0
        d2 = (abs_x - x_int) ** 2 + (abs_y - y_int) ** 2
        best_d2 = int(d2.min())
        candidate_indices = np.where(d2 == best_d2)[0]
        if len(candidate_indices) > 1:
            labels = patch[ys[candidate_indices], xs[candidate_indices]].astype(int)
            counts = Counter(labels.tolist())
            best_label = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            chosen_local = candidate_indices[np.where(labels == best_label)[0][0]]
        else:
            chosen_local = candidate_indices[0]
        eval_x, eval_y = int(abs_x[chosen_local]), int(abs_y[chosen_local])
        return {
            "gt_id": int(mask[eval_y, eval_x]),
            "method": "radius",
            "radius_px": int(radius),
            "point": [x_int, y_int],
            "eval_point": [eval_x, eval_y],
            "distance_px": round(float(best_d2 ** 0.5), 3),
        }

    return {
        "gt_id": 0,
        "method": "miss",
        "radius_px": None,
        "point": [x_int, y_int],
        "eval_point": [x_int, y_int],
    }


def map_badge_point_to_gt(mask: np.ndarray | None, badge_id, id_to_point: dict[int, dict]):
    badge_id = legacy._to_int(badge_id)
    if badge_id is None:
        return {"badge_id": None, "gt_id": 0, "method": "missing_badge_id"}
    point = id_to_point.get(int(badge_id))
    if not point:
        return {"badge_id": int(badge_id), "gt_id": 0, "method": "badge_point_missing"}
    mapped = _map_point_to_gt(mask, point["x"], point["y"])
    return {
        "badge_id": int(badge_id),
        "label": point.get("label", ""),
        **mapped,
    }


def map_top_points_to_gt(case, top_points_with_ids):
    mask = _annotation_mask(int(case["image_id"]))
    mapped_ids = []
    eval_top_points = []
    details = []
    for item in top_points_with_ids or []:
        if len(item) >= 4:
            x, y, name, badge_id = item[:4]
        else:
            x, y, name = item[:3]
            badge_id = None
        mapped = _map_point_to_gt(mask, x, y)
        gt_id = int(mapped["gt_id"])
        if gt_id > 0:
            mapped_ids.append(gt_id)
        eval_x, eval_y = mapped["eval_point"]
        eval_top_points.append([int(eval_x), int(eval_y), str(name)])
        details.append({
            "badge_id": None if badge_id is None else int(badge_id),
            "label": str(name),
            **mapped,
        })
    return set(mapped_ids), details, eval_top_points


def eval_result(case, top_points_with_ids):
    img_id = int(case["image_id"])
    query_obj_id = int(case["query_object"]["obj_id"])
    gt_top_ids = {int(t["obj_id"]) for t in case["target_objects"]}
    pred_ids_set, chosen_mapping, eval_top_points = map_top_points_to_gt(case, top_points_with_ids)
    non_query_ids = pred_ids_set - {query_obj_id}
    action = "remove obstacle" if non_query_ids else "pick object"
    # gt_action derives from whether the target actually has an occluder (GT
    # top objects other than the query itself), not a hardcoded "remove
    # obstacle" -- that previous hardcoding inverted correctness for every
    # No-Occ/Easy case, where "pick object" is the right answer.
    gt_action = "remove obstacle" if (gt_top_ids - {query_obj_id}) else "pick object"
    action_correct = action == gt_action
    target_matched = bool(pred_ids_set & gt_top_ids)
    return {
        "pred_ids_set": pred_ids_set,
        "gt_top_ids": gt_top_ids,
        "action": action,
        "gt_action": gt_action,
        "action_correct": action_correct,
        "target_matched": target_matched,
        "fully_correct": action_correct and target_matched,
        "chosen_mapping": chosen_mapping,
        "eval_top_points": eval_top_points,
    }


def supported_edges(chain, ref_edges):
    ref_pairs = {(int(e["to"]), int(e["from"])) for e in ref_edges}
    out = []
    for edge in chain or []:
        pair = (legacy._to_int(edge.get("occluded_id")), legacy._to_int(edge.get("occluder_id")))
        out.append({
            "occluded_id": pair[0],
            "occluder_id": pair[1],
            "supported_by_uoais": pair in ref_pairs,
        })
    return out


def write_predictions_and_eval(rows, gt_records, out_root):
    pred_path = out_root / "predictions.jsonl"
    gt_path = out_root / "gt_synthetic_eval.json"
    json.dump(gt_records, open(gt_path, "w", encoding="utf-8"), indent=2)
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            mask = _annotation_mask(int(row["image_id"]))
            think = build_think_from_chain(row, mask)
            answer = points_to_answer(row.get("eval_top_points") or row["top_points"])
            f.write(json.dumps({
                "image_id": row["image_id"],
                "query_object": row["query_mask_id"],
                "model_output": think + answer,
            }, ensure_ascii=False) + "\n")
    run_evaluate_nlp(pred_path, gt_path, out_root / "eval_results.txt")












