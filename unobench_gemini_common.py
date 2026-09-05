#!/usr/bin/env python3
"""
unobench_gemini_common.py — Gemini detection + UnoBench data extraction helpers (shared).

Pipeline per case:
  1. Detect   : Gemini bbox pointing → detected objects with pixel coordinates
  2. Labeled  : draw_labeled_image() → SoM-style labeled image
  3. Combined : 1 prompt × 1 run at temperature=0.0
                Model outputs pure JSON: {occlusion_chain[edge_factors], candidates[prob_free],
                free_objects, chosen_to_remove}. Code combines edge_factors → per-edge confidence.
  4. JSONL    : code builds <answer>[<points x y>name</points>]</answer> for evaluate_nlp.py
  5. Eval     : action_correct + target_matched + fully_correct

Usage:
  python run_unobench_gemini.py \
    --case-list data/unobench_eval/case_list_synthetic_100.json \
    --out logs/unobench_eval_synthetic \
    --n-cases 50
"""
import os, re, json, argparse, logging, base64, zipfile, subprocess
from pathlib import Path

import numpy as np
from PIL import Image as _PIL_Image

import gemini_client as base
from google.genai import types as gtypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("unobench")

ROOT    = Path(__file__).resolve().parent
EXTRACT = ROOT / "UnoBench/_extracted/images"
ANN_ZIP = ROOT / "UnoBench/annotations.zip"
ANN_DIR = ROOT / "UnoBench/_extracted/annotations"

# Reuse the official point-in-mask mapping so in-script scoring matches evaluate_nlp.py
import sys
sys.path.insert(0, str(ROOT))
import evaluate_nlp as eval_nlp

# ── object detection ──────────────────────────────────────────────────────────────
_BBOX_POINTING_PROMPT = (
    "Detect all distinct objects in the scene (exclude the background surface, "
    "bin walls, and any robotic arm/gripper). "
    "For each object return a JSON array:\n"
    '[{"id": 1, "label": "short descriptive name", '
    '"bbox": [y_min, x_min, y_max, x_max], "point": [y, x]}]\n'
    "where bbox values are normalized coordinates in [0, 1000] (0=top-left corner). "
    "The point must be well inside the visible pixels of that object, not merely the "
    "bbox center. For elongated, ring-shaped, thin, or partly occluded objects, choose "
    "a visible interior point that would map to the object's mask. "
    "Number objects 1, 2, 3, … in any order. "
    "Output ONLY the JSON array — no markdown, no extra text."
)


def detect_objects(image_bytes, image_size):
    """Gemini pointing + bbox. Returns [{id, label, x_px, y_px, bbox_px}]."""
    W, H = image_size
    try:
        text = base._call_gemini([
            gtypes.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            _BBOX_POINTING_PROMPT,
        ])
        text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
        text = re.sub(r"```$", "", text.strip())
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            raw = json.loads(m.group()) if m else []
    except Exception as e:
        logger.error(f"detect_objects failed: {e}")
        return []

    objects = []
    for obj in raw:
        obj_id = int(obj.get("id", len(objects) + 1))
        label  = str(obj.get("label", f"object_{obj_id}"))
        bbox   = obj.get("bbox")
        point  = obj.get("point") or obj.get("point_inside") or obj.get("center")
        if bbox is None and point is None:
            continue
        if bbox is not None and len(bbox) == 4:
            y1n, x1n, y2n, x2n = [float(v) for v in bbox]
            x1 = int(x1n / 1000 * W); y1 = int(y1n / 1000 * H)
            x2 = int(x2n / 1000 * W); y2 = int(y2n / 1000 * H)
        else:
            y_n, x_n = float(point[0]), float(point[1])
            x1 = int(x_n / 1000 * W); y1 = int(y_n / 1000 * H)
            r  = max(30, min(W, H) // 15)
            x1, y1, x2, y2 = x1 - r, y1 - r, x1 + r, y1 + r
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if point is not None and len(point) >= 2:
            y_n, x_n = float(point[0]), float(point[1])
            x_px = int(max(0, min(W - 1, x_n / 1000 * W)))
            y_px = int(max(0, min(H - 1, y_n / 1000 * H)))
        else:
            x_px = (x1 + x2) // 2
            y_px = (y1 + y2) // 2
        objects.append({
            "id": obj_id, "label": label,
            "x_px": x_px, "y_px": y_px,
            "bbox_px": (x1, y1, x2, y2),
        })
    logger.info(f"  detect_objects: {len(objects)} objects found")
    return objects


# ── coord extraction (no snapping) ────────────────────────────────────────────────
def _extract_name_coords(text):
    """Extract all 'name at (x, y)' patterns. Returns [(name, x, y), ...] with raw model coords."""
    pts = re.findall(
        r'([A-Za-z][^\n(]{0,60}?)\s+at\s+\((\d+\.?\d*),\s*(\d+\.?\d*)\)',
        text
    )
    result = []
    for name, x_str, y_str in pts:
        name = name.strip().rstrip(' -–,')
        result.append((name, int(float(x_str)), int(float(y_str))))
    return result


# ── combined prompt ────────────────────────────────────────────────────────────────
_COMBINED_PROMPT_TMPL = """\
You are a robotic manipulation expert specializing in obstruction reasoning for grasping.

Objects are labeled with numbered ID badges in the image.

Detected objects (ID: label):
{obj_list}

Task: pick up "{query_name}"

Analyze the obstruction situation for the target object by visually inspecting the image.
Use a target-centric grasping definition of obstruction, not only a visual stacking
definition. A parallel-jaw gripper needs about 8-10 cm of free clearance around the
target and along the approach path.

Before reasoning about obstruction, first identify the exact target instance ID.
Spatial words in the task are critical:
- left/right use image x-position among matching objects (left = smaller x, right = larger x).
- top/bottom use image y-position among matching objects (top = smaller y, bottom = larger y).
- middle/center mean the matching object closest to the middle of the matching group.
When several objects have the same category or label, compare the numbered ID badges and
object positions in the image. Do not choose a different duplicate just because it has the
same label.

Respond with JSON only — no markdown, no extra text:
{{
  "target_id": <detected object ID>,
  "target_label": "<label, optional>",
  "occlusion_chain": [
    {{
      "occluded_id": <detected object ID>,
      "occluder_id": <detected object ID>,
      "occluded": "<label, optional>",
      "occluder": "<label, optional>",
      "edge_factors": {{
        "contour_cut":       <0-100>,
        "depth_in_front":    <0-100>,
        "visibility_loss":   <0-100>,
        "overlap_certainty": <0-100>,
        "contact_or_boundary": <0-100>,
        "clearance_blocking": <0-100>
      }}
    }}
  ],
  "candidates": [
    {{"id": <detected object ID>, "label": "<label, optional>", "prob_free": <0-100>}}
  ],
  "candidate_ids": [<detected object ID>, ...],
  "free_object_ids": [<detected object ID>, ...],
  "chosen_to_remove_ids": [<detected object ID>, ...]
}}

Rules:
- occlusion_chain: the obstruction/access-blocking relationships from the target up to
  the directly removable blocker(s). Each entry is ONE directed edge: object "occluded"
  is blocked by object "occluder". Keep this key name exactly as "occlusion_chain" for JSON
  compatibility, but reason about obstruction broadly.
  Empty [] only if the target can be grasped directly with safe gripper clearance.
- target_id: the single detected ID that best matches the requested object instance. Use
  the spatial words in the task to disambiguate duplicate labels before deciding obstruction.
- Object A obstructs object B if ANY of these are true:
    1. stacking/support: A is on top of, resting on, covering, or visibly in front of B.
    2. contact/access: A touches B, shares a boundary/contact point, presses against B,
       or occupies the immediate access region needed to grasp B.
    3. clearance: A is adjacent/near enough to B that a parallel-jaw gripper with about
       8-10 cm width cannot approach, close, lift, or retreat safely without first
       removing A.
  IMPORTANT: physical contact or insufficient clearance is enough to count as obstruction,
  even when there is little or no visible overlap and nothing is literally on top.
  Be conservative for clearance-only edges: do NOT count objects that are merely nearby
  with a visible gap and no plausible gripper-corridor conflict. If there is no contact,
  no overlap, and no clear approach blockage, leave the edge out.
- IDs are authoritative. Use only the numbered ID badges shown in the image and listed above.
  Do NOT invent IDs. Do NOT output pixel coordinates in this reasoning JSON.
- edge_factors: for EACH edge, rate these visual-evidence factors INDEPENDENTLY from
  0 (no evidence) to 100 (certain) — they decide how strongly the occluder blocks grasping
  of the occluded object:
    * contour_cut       — the occluder's boundary visibly passes in front of / cuts the occluded object's outline.
    * depth_in_front    — the occluder clearly appears nearer the camera (in front of) the occluded object.
    * visibility_loss   — how much of the occluded object is hidden or access-blocked by THIS occluder.
    * overlap_certainty — certainty that the objects overlap OR touch/contact/share a blocking boundary.
    * contact_or_boundary — certainty that the objects physically touch, press together, share a boundary,
      or have a contact/access point that would impede grasping.
    * clearance_blocking — certainty that the occluder leaves insufficient 8-10 cm gripper clearance
      around the occluded object or along the approach/lift path.
  Give precise values — do NOT round to multiples of 5.
- candidates: EVERY distinct object that appears in occlusion_chain (INCLUDING the target
  itself). For each, prob_free (0-100) = probability the object is currently free — nothing
  on top of it AND enough contact/clearance space for direct grasping. Give precise values —
  do NOT round to multiples of 5.
- candidate_ids: the IDs of all candidate objects.
- free_object_ids: the subset of candidate_ids you believe are currently free (high prob_free).
- chosen_to_remove_ids: among free_object_ids, the object(s) to grasp/remove FIRST to make progress
  toward the target. If several objects are free, choose the best one(s) yourself.
  If the target itself is free, put target_id here. If the target is blocked, choose only
  top-level directly graspable blockers that lie on an obstruction path to target_id; do not
  choose a distractor object, a same-category duplicate, or an object that is itself still blocked.\
"""


# Per-EDGE visual-evidence sub-scores → continuous edge probability.
# Distinct weights make combinations of round sub-scores rarely collide, yielding a
# near-continuous distribution from a single generation (same trick, now applied per edge).
_EDGE_WEIGHTS = {
    "contour_cut":       0.25,
    "depth_in_front":    0.20,
    "visibility_loss":   0.20,
    "overlap_certainty": 0.15,
    "contact_or_boundary": 0.15,
    "clearance_blocking": 0.05,
}


def _weighted_combine(factors, weights):
    """Weighted mean of sub-scores → value in [0,100], or None if nothing usable."""
    if not factors:
        return None
    vals = [(w, factors.get(k)) for k, w in weights.items() if factors.get(k) is not None]
    if not vals:
        return None
    tot_w = sum(w for w, _ in vals)
    return round(sum(w * float(v) for w, v in vals) / tot_w, 1)


def _to_float(v):
    """Best-effort float, else None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v):
    """Best-effort int for model-returned IDs, else None."""
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _valid_ids(objects):
    return {int(o["id"]) for o in objects}


def _filter_valid_ids(values, objects):
    valid = _valid_ids(objects)
    out, invalid = [], []
    for v in values or []:
        oid = _to_int(v)
        if oid is None:
            continue
        if oid in valid:
            if oid not in out:
                out.append(oid)
        else:
            invalid.append(oid)
    return out, invalid


def _legacy_ids_from_strings(items, objects):
    """Map legacy '<name> at (x, y)' strings to nearest detected object center."""
    out = []
    for item in items or []:
        for _, x, y in _extract_name_coords(str(item)):
            if not objects:
                continue
            nearest = min(
                objects,
                key=lambda o: (int(o["x_px"]) - x) ** 2 + (int(o["y_px"]) - y) ** 2,
            )
            oid = int(nearest["id"])
            if oid not in out:
                out.append(oid)
    return out


def _object_label(objects, obj_id):
    for o in objects:
        if int(o["id"]) == int(obj_id):
            return str(o["label"])
    return f"object {obj_id}"


def _points_from_ids(ids, objects):
    by_id = {int(o["id"]): o for o in objects}
    pts = []
    for oid in ids:
        obj = by_id.get(int(oid))
        if obj is None:
            continue
        pts.append((int(obj["x_px"]), int(obj["y_px"]), str(obj["label"]), int(oid)))
    return pts


def _parse_chain(raw_chain, objects):
    """Normalize occlusion_chain → [{occluded, occluder, edge_factors, edge_confidence}].

    edge_confidence is computed in code from the per-edge visual-evidence sub-scores
    (weighted mean), so the distribution stays smooth. Tolerates a single edge_confidence
    number or legacy string entries. Malformed entries are skipped.
    """
    valid = _valid_ids(objects)
    out = []
    for e in raw_chain or []:
        if isinstance(e, dict):
            occluded_id = _to_int(e.get("occluded_id"))
            occluder_id = _to_int(e.get("occluder_id"))
            occluded = str(e.get("occluded", ""))
            occluder = str(e.get("occluder", ""))
            if occluded_id not in valid:
                occluded_id = None
            if occluder_id not in valid:
                occluder_id = None
            if not (occluded or occluder or occluded_id or occluder_id):
                continue
            factors = e.get("edge_factors") if isinstance(e.get("edge_factors"), dict) else None
            conf    = _weighted_combine(factors, _EDGE_WEIGHTS)
            if conf is None:                                  # fallback: direct number
                conf = _to_float(e.get("edge_confidence"))
            if not occluded and occluded_id is not None:
                occluded = _object_label(objects, occluded_id)
            if not occluder and occluder_id is not None:
                occluder = _object_label(objects, occluder_id)
            out.append({
                "occluded_id":      occluded_id,
                "occluder_id":      occluder_id,
                "occluded":        occluded,
                "occluder":        occluder,
                "edge_factors":    factors,
                "edge_confidence": conf,
            })
        elif isinstance(e, str) and e.strip():
            out.append({"occluded_id": None, "occluder_id": None,
                        "occluded": e.strip(), "occluder": "",
                        "edge_factors": None, "edge_confidence": None})
    return out


def _parse_candidates(raw_cands, objects):
    """Normalize candidates entries → [{object, prob_free}]. Malformed entries skipped."""
    valid = _valid_ids(objects)
    out = []
    for c in raw_cands or []:
        if not isinstance(c, dict):
            continue
        oid = _to_int(c.get("id") if c.get("id") is not None else c.get("object_id"))
        if oid not in valid:
            oid = None
        obj = str(c.get("object") or c.get("label") or "")
        if not obj and oid is not None:
            obj = _object_label(objects, oid)
        if obj or oid is not None:
            out.append({"id": oid, "object": obj, "prob_free": _to_float(c.get("prob_free"))})
    return out


def parse_combined_json(raw, objects):
    """Parse pure JSON output from model. Returns extracted data dict using raw model coords."""
    raw_clean = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
    raw_clean = re.sub(r"```$", "", raw_clean.strip())

    chain, candidates, free_objects, chosen, confidence = [], [], [], [], None
    target_id, target_label = None, ""
    candidate_ids, free_object_ids, chosen_to_remove_ids = [], [], []
    invalid_ids = []
    m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
    if m:
        try:
            data         = json.loads(m.group())
            target_id    = _to_int(data.get("target_id"))
            if target_id not in _valid_ids(objects):
                target_id = None
            target_label = str(data.get("target_label") or "")
            if not target_label and target_id is not None:
                target_label = _object_label(objects, target_id)
            chain        = _parse_chain(data.get("occlusion_chain"), objects)
            candidates   = _parse_candidates(data.get("candidates"), objects)
            free_objects = [str(s) for s in (data.get("free_objects") or [])]
            chosen       = [str(s) for s in (data.get("chosen_to_remove") or [])]
            candidate_ids, bad = _filter_valid_ids(data.get("candidate_ids"), objects)
            invalid_ids.extend(bad)
            free_object_ids, bad = _filter_valid_ids(data.get("free_object_ids"), objects)
            invalid_ids.extend(bad)
            chosen_to_remove_ids, bad = _filter_valid_ids(data.get("chosen_to_remove_ids"), objects)
            invalid_ids.extend(bad)

            if not candidate_ids:
                candidate_ids = [c["id"] for c in candidates if c.get("id") is not None]
            if not free_object_ids:
                free_object_ids = _legacy_ids_from_strings(free_objects, objects)
            if not chosen_to_remove_ids:
                chosen_to_remove_ids = _legacy_ids_from_strings(chosen, objects)

            # Case-level confidence = mean of per-edge confidences (smooth, derived in code).
            # If the target is free (no chain), use mean prob_free of candidates instead.
            edge_confs = [e["edge_confidence"] for e in chain if e["edge_confidence"] is not None]
            if edge_confs:
                confidence = round(sum(edge_confs) / len(edge_confs), 1)
            else:
                pfs = [c["prob_free"] for c in candidates if c["prob_free"] is not None]
                confidence = round(sum(pfs) / len(pfs), 1) if pfs else None
        except Exception:
            pass

    if not chosen_to_remove_ids:
        chosen_to_remove_ids = list(free_object_ids)
    if not chosen_to_remove_ids:
        chosen_to_remove_ids = _legacy_ids_from_strings([raw], objects)

    top_points_with_ids = _points_from_ids(chosen_to_remove_ids, objects)
    top_points = [(x, y, name) for x, y, name, _ in top_points_with_ids]
    first_label = top_points[0][2] if top_points else ""

    return {
        "pred_label":         first_label,
        "confidence":         confidence,
        "target_id":          target_id,
        "target_label":       target_label,
        "occlusion_chain":    chain,
        "candidates":         candidates,
        "free_objects":       free_objects,
        "chosen_to_remove":   chosen,
        "candidate_ids":      candidate_ids,
        "free_object_ids":    free_object_ids,
        "chosen_to_remove_ids": chosen_to_remove_ids,
        "invalid_ids":        sorted(set(invalid_ids)),
        "top_points":         top_points,
        "top_points_with_ids": [[x, y, name, oid] for x, y, name, oid in top_points_with_ids],
    }


def run_combined_once(labeled_b64, objects, query_name, temperature=0.0):
    """
    Run the combined reasoning prompt ONCE (single generation, saves tokens).
    Model outputs pure JSON with a multi-factor confidence rubric; we extract
    pred_label, occlusion_chain, and a continuous confidence combined in code.
    """
    obj_list = "\n".join(
        f"  ID {o['id']}: {o['label']}" for o in objects
    )
    prompt    = _COMBINED_PROMPT_TMPL.format(obj_list=obj_list, query_name=query_name)
    img_bytes = base64.b64decode(labeled_b64)
    contents  = [
        gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        prompt,
    ]

    raw_outputs, parsed = [], None
    try:
        resp = base.generate_content(
            model=base.MODEL,
            contents=contents,
            config=gtypes.GenerateContentConfig(temperature=temperature),
        )
        raw = (resp.text or "").strip()
        if raw:
            raw_outputs.append(raw)
            parsed = parse_combined_json(raw, objects)
    except Exception as e:
        logger.warning(f"  [Combined] failed: {e}")

    if parsed is None:
        parsed = {
            "pred_label": "", "confidence": None,
            "occlusion_chain": [], "candidates": [],
            "free_objects": [], "chosen_to_remove": [], "top_points": [],
        }

    pred_label = parsed.get("pred_label", "")
    action     = (
        "pick object"
        if (not pred_label or base._label_matches(pred_label, query_name))
        else "remove obstacle"
    )
    confidence = parsed.get("confidence")
    logger.info(
        f"  [Combined] pred={pred_label!r} action={action} conf={confidence}"
    )

    return {
        **parsed,
        "action":      action,
        "confidence":  confidence,
        "raw_outputs": raw_outputs,
    }


# ── annotation extraction ─────────────────────────────────────────────────────────
def ensure_annotations(cases, ann_zip, ann_dir):
    """Extract annotation .npy for each case from annotations.zip if not already present."""
    needed  = {c['image_id'] for c in cases}
    already = {int(p.stem.split('_')[1]) for p in ann_dir.glob('image_*.npy')}
    missing = needed - already
    if not missing:
        logger.info(f"  Annotations: all {len(needed)} files already present.")
        return
    logger.info(f"  Extracting {len(missing)} annotation masks from zip …")
    with zipfile.ZipFile(ann_zip) as z:
        names      = set(z.namelist())
        extracted  = 0
        for img_id in sorted(missing):
            zname = f"image_{img_id:06d}.npy"
            if zname in names:
                (ann_dir / zname).write_bytes(z.read(zname))
                extracted += 1
            else:
                logger.warning(f"  [WARN] annotation not in zip: {zname}")
    logger.info(f"  Extracted {extracted}/{len(missing)} annotation files.")


IMG_ZIP    = ROOT / "UnoBench/images.zip"
DEPTH_DIR  = ROOT / "UnoBench/_extracted/depth"
META_ZIP   = ROOT / "UnoBench/meta_data/annotations_meta.zip"
ID_MAP     = ROOT / "UnoBench/meta_data/image_id_scene_view_id_mapping.json"


def ensure_images(cases, img_zip, extract_dir):
    """Extract RGB image_*.png for each case from images.zip if not already present."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    needed  = {c['image_id'] for c in cases}
    already = {int(p.stem.split('_')[1]) for p in extract_dir.glob('image_*.png')}
    missing = needed - already
    if not missing:
        logger.info(f"  Images: all {len(needed)} files already present.")
        return
    logger.info(f"  Extracting {len(missing)} RGB images from zip …")
    with zipfile.ZipFile(img_zip) as z:
        names = set(z.namelist())
        for img_id in sorted(missing):
            for cand in (f"images/image_{img_id:06d}.png", f"image_{img_id:06d}.png"):
                if cand in names:
                    (extract_dir / f"image_{img_id:06d}.png").write_bytes(z.read(cand))
                    break
            else:
                logger.warning(f"  [WARN] image not in zip: image_{img_id:06d}.png")


def ensure_depth(cases, depth_dir, meta_zip, id_map_path):
    """Best-effort extract depth .npy for each case (depth is NOT used in scoring).

    Maps image_id → (scene, view) via the mapping json, reads the 'depth' array from
    meta_data/annotations_meta.zip::ALL_NPZ/scene{s}_view{v}.npz. Never raises.
    """
    try:
        depth_dir.mkdir(parents=True, exist_ok=True)
        if not (meta_zip.exists() and id_map_path.exists()):
            logger.info("  Depth: meta zip / mapping not found — skipping (depth unused).")
            return
        raw_map = json.load(open(id_map_path)).get("mapping", [])
        # mapping is a list of {image_id, scene_id, view_id, old_stem, ...}
        id_to_stem = {int(e["image_id"]): e.get("old_stem") or f"scene{e['scene_id']}_view{e['view_id']}"
                      for e in raw_map if isinstance(e, dict) and "image_id" in e}
        already = {int(p.stem.split('_')[1]) for p in depth_dir.glob('image_*.npy')}
        missing = {c['image_id'] for c in cases} - already
        if not missing:
            logger.info("  Depth: all requested files already present.")
            return
        import io
        ok = 0
        with zipfile.ZipFile(meta_zip) as z:
            names = set(z.namelist())
            for img_id in sorted(missing):
                stem = id_to_stem.get(img_id)
                if not stem:
                    continue
                npz_name = next((n for n in (
                    f"annotations_meta/ALL_NPZ/{stem}.npz", f"ALL_NPZ/{stem}.npz", f"{stem}.npz",
                ) if n in names), None)
                if npz_name is None:
                    continue
                with np.load(io.BytesIO(z.read(npz_name))) as npz:
                    if "depth" in npz:
                        np.save(depth_dir / f"image_{img_id:06d}.npy", npz["depth"])
                        ok += 1
        logger.info(f"  Depth: extracted {ok}/{len(missing)} depth maps (best-effort).")
    except Exception as e:
        logger.warning(f"  Depth extraction skipped (non-fatal): {e}")


# ── per-case processing ───────────────────────────────────────────────────────────
def _case_name(case):
    if case.get("case_name"):
        return str(case["case_name"])
    return f"img{case['image_id']:06d}_q{case['query_object']['obj_id']}"


def _case_image_path(case):
    if case.get("image_path_abs"):
        return Path(case["image_path_abs"])
    return EXTRACT / f"image_{case['image_id']:06d}.png"


def _case_npz_root(case):
    return Path(case.get("eval_npz_root") or ANN_DIR)


def _freegrasp_gt_action(query_obj_id, gt_top_ids):
    return "pick object" if gt_top_ids and all(i == int(query_obj_id) for i in gt_top_ids) else "remove obstacle"


def process_case(case, args, out_root, pred_fh):
    img_id       = case["image_id"]
    query_name   = case["query_object"]["object_name"]
    query_obj_id = case["query_object"]["obj_id"]
    gt_targets   = case["target_objects"]
    difficulty   = case["difficulty"]
    case_name    = _case_name(case)
    is_freegrasp = case.get("dataset") == "freegrasp"

    raw_path = _case_image_path(case)
    if not raw_path.exists():
        raise FileNotFoundError(f"Image not found: {raw_path}")

    image_bytes = raw_path.read_bytes()
    image_pil   = _PIL_Image.open(raw_path).convert("RGB")
    image_np    = np.array(image_pil)

    out_dir = out_root / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{case_name}] query='{query_name}'  "
        f"gt_targets={[t['object_name'] for t in gt_targets]}"
    )

    # Step 1: Detect objects
    objects = detect_objects(image_bytes, image_pil.size)
    if not objects:
        logger.error(f"[{case_name}] No objects detected — skipping")
        return {}

    # Step 2: Draw labeled image (SoM-style)
    label_path  = out_dir / "labeled.png"
    labeled_b64 = base.draw_labeled_image(image_np, objects, label_path)

    # Step 3: Combined reasoning + confidence (pure JSON, single call at temp=0.0)
    combined        = run_combined_once(labeled_b64, objects, query_name, temperature=0.0)
    pred_label      = combined["pred_label"]
    confidence      = combined["confidence"]
    target_id       = combined.get("target_id")
    target_label    = combined.get("target_label", "")
    occlusion_chain = combined.get("occlusion_chain", [])
    candidates      = combined.get("candidates", [])
    free_objects    = combined.get("free_objects", [])
    chosen_to_remove = combined.get("chosen_to_remove", [])
    candidate_ids   = combined.get("candidate_ids", [])
    free_object_ids = combined.get("free_object_ids", [])
    chosen_to_remove_ids = combined.get("chosen_to_remove_ids", [])
    invalid_ids     = combined.get("invalid_ids", [])
    top_points      = combined.get("top_points", [])
    id_to_point     = {
        int(o["id"]): {"label": o["label"], "x": int(o["x_px"]), "y": int(o["y_px"])}
        for o in objects
    }

    # Step 4: Build <answer> for evaluate_nlp.py (code-generated, not model output)
    answer_parts       = [f'<points {x} {y}>{name}</points>' for x, y, name in top_points]
    jsonl_model_output = f'<answer>[{", ".join(answer_parts)}]</answer>'

    pred_fh.write(json.dumps({
        "image_id":     img_id,
        "query_object": query_obj_id,
        "model_output": jsonl_model_output,
    }) + "\n")
    pred_fh.flush()

    # Step 5: Evaluate vs GT — purely geometric (point-in-mask), NO label/LLM matching.
    # Map each chosen point → object id via the instance mask (same logic as evaluate_nlp.py),
    # then compare ids against GT, instead of comparing object *names*.
    gt_top_ids       = {int(t["obj_id"]) for t in gt_targets}
    pred_ids, _, _   = eval_nlp.coords_to_object_ids(
        img_id, [(x, y, name) for x, y, name in top_points], str(_case_npz_root(case)), "synthetic",
    )
    pred_ids_set     = {i for i in pred_ids if i > 0}
    non_query_ids    = pred_ids_set - {int(query_obj_id)}

    # action: chose a non-query object → remove obstacle; only the query / a miss → pick object
    action         = "remove obstacle" if non_query_ids else "pick object"
    gt_action      = _freegrasp_gt_action(query_obj_id, gt_top_ids) if is_freegrasp else "remove obstacle"
    action_correct = (action == gt_action) if is_freegrasp else bool(non_query_ids)
    target_matched = bool(pred_ids_set & gt_top_ids)   # chosen point lands in a GT-target mask
    fully_correct  = action_correct and target_matched

    logger.info(
        f"  → action={action!r}  pred_label={pred_label!r}  "
        f"action_correct={action_correct}  target_matched={target_matched}  "
        f"conf={confidence}"
    )

    result = {
        "case":            case_name,
        "image_id":        img_id,
        "query":           query_name,
        "difficulty":      difficulty,
        "dataset":         case.get("dataset", "unobench"),
        "scene_id":        case.get("scene_id"),
        "query_obj_id":    case.get("freegrasp_query_obj_id"),
        "query_mask_id":   query_obj_id,
        "gt_targets":      [t["object_name"] for t in gt_targets],
        "gt_action":       gt_action,
        "pred_action":     action,
        "pred_label":      pred_label,
        "target_id":       target_id,
        "target_label":    target_label,
        "pred_object_ids": sorted(pred_ids_set),
        "gt_top_ids":      sorted(gt_top_ids),
        "ground_truth_obj_ids": case.get("ground_truth_obj_ids"),
        "ground_truth_mask_ids": case.get("ground_truth_mask_ids"),
        "confidence":      confidence,
        "action_correct":  action_correct,
        "target_matched":  target_matched,
        "fully_correct":   fully_correct,
        "occlusion_chain": occlusion_chain,
        "candidates":      candidates,
        "free_objects":    free_objects,
        "chosen_to_remove": chosen_to_remove,
        "candidate_ids":   candidate_ids,
        "free_object_ids": free_object_ids,
        "chosen_to_remove_ids": chosen_to_remove_ids,
        "invalid_ids":     invalid_ids,
        "id_to_point":     id_to_point,
        "top_points":      [[x, y, name] for x, y, name in top_points],
        "top_points_with_ids": combined.get("top_points_with_ids", []),
        "n_api_calls":     2,
        "raw_outputs":     combined.get("raw_outputs", []),
        "det_objects":     [
            {"id": o["id"], "label": o["label"], "x_px": o["x_px"], "y_px": o["y_px"]}
            for o in objects
        ],
    }
    json.dump(result, open(out_dir / "result.json", "w"), indent=2)
    return result


# ── aggregate summary ─────────────────────────────────────────────────────────────
def summarize(results, out_root):
    n = len(results)

    def acc(key):
        return round(sum(r[key] for r in results) / n, 3)

    confs     = [r["confidence"] for r in results if r.get("confidence") is not None]
    mean_conf = round(sum(confs) / len(confs), 1) if confs else None
    by_diff   = {}

    for d in ["Easy", "Medium", "Hard"]:
        rs = [r for r in results if r["difficulty"] == d]
        if not rs:
            continue
        cs = [r["confidence"] for r in rs if r.get("confidence") is not None]
        by_diff[d] = {
            "n":          len(rs),
            "action_acc": round(sum(r["action_correct"] for r in rs) / len(rs), 3),
            "target_acc": round(sum(r["target_matched"] for r in rs) / len(rs), 3),
            "full_acc":   round(sum(r["fully_correct"]  for r in rs) / len(rs), 3),
            "mean_conf":  round(sum(cs) / len(cs), 1) if cs else None,
        }

    conf_correct = [r["confidence"] for r in results if r.get("fully_correct") and r.get("confidence") is not None]
    conf_wrong   = [r["confidence"] for r in results if not r.get("fully_correct") and r.get("confidence") is not None]

    summary = {
        "n":                 n,
        "total_api_calls":   sum(r.get("n_api_calls", 2) for r in results),
        "action_acc":        acc("action_correct"),
        "target_acc":        acc("target_matched"),
        "full_acc":          acc("fully_correct"),
        "mean_conf":         mean_conf,
        "mean_conf_correct": round(sum(conf_correct) / len(conf_correct), 1) if conf_correct else None,
        "mean_conf_wrong":   round(sum(conf_wrong)   / len(conf_wrong),   1) if conf_wrong   else None,
        "by_difficulty":     by_diff,
    }

    json.dump({"summary": summary, "per_case": results},
              open(out_root / "summary.json", "w"), indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("UNOBENCH EVAL SUMMARY (Synthetic_train GT)")
    logger.info("=" * 60)
    logger.info(f"  cases={n}  api_calls={summary['total_api_calls']}")
    logger.info(f"  action_acc  (remove obstacle?) : {summary['action_acc']}")
    logger.info(f"  target_acc  (correct object?)  : {summary['target_acc']}")
    logger.info(f"  full_acc    (both correct?)     : {summary['full_acc']}")
    logger.info(f"  mean_conf   : {summary['mean_conf']}")
    logger.info(f"  mean_conf (correct) : {summary['mean_conf_correct']}")
    logger.info(f"  mean_conf (wrong)   : {summary['mean_conf_wrong']}")
    logger.info("  by difficulty:")
    for d, st in by_diff.items():
        logger.info(
            f"    {d:6}: n={st['n']:2}  action={st['action_acc']}  "
            f"target={st['target_acc']}  full={st['full_acc']}  conf={st['mean_conf']}"
        )
    return summary


# ── report ──────────────────────────────────────────────────────────────────────────
def write_report(results, summary, eval_text, out_root, args):
    """Write a human-readable markdown report (config + per-case + occlusion graph + eval)."""
    L = []
    L.append(f"# UnoBench eval report — {args.difficulty or 'all'} ({len(results)} cases)\n")
    L.append(f"- Model: `{base.MODEL}`  |  case-list: `{args.case_list}`  |  difficulty: `{args.difficulty}`")
    L.append(f"- Pipeline: 1 detect + 1 combined call/case (2 API calls), temp=0.0")
    L.append(f"- Output schema: ID-authoritative occlusion_chain + candidates[prob_free] + chosen_to_remove_ids")
    L.append(f"- Scoring: selected detected IDs → detection points → GT mask lookup via evaluate_nlp.coords_to_object_ids\n")

    L.append("## Aggregate metrics\n")
    L.append(f"| metric | value |\n|---|---|")
    L.append(f"| cases | {summary['n']} |")
    L.append(f"| total_api_calls | {summary['total_api_calls']} |")
    L.append(f"| action_acc | {summary['action_acc']} |")
    L.append(f"| target_acc | {summary['target_acc']} |")
    L.append(f"| full_acc | {summary['full_acc']} |")
    L.append(f"| mean_conf | {summary['mean_conf']} |")
    L.append(f"| mean_conf (correct) | {summary['mean_conf_correct']} |")
    L.append(f"| mean_conf (wrong) | {summary['mean_conf_wrong']} |\n")

    L.append("## Per-case summary\n")
    L.append("| case | query | gt_top | chosen_detect_ids | chosen→gt_ids | action✓ | target✓ | full✓ | conf |")
    L.append("|---|---|---|---|---|:---:|:---:|:---:|---|")
    for r in results:
        L.append(
            f"| {r['case']} | {r['query']} | {r['gt_top_ids']} | "
            f"{r.get('chosen_to_remove_ids', [])} | {r['pred_object_ids']} | "
            f"{'✅' if r['action_correct'] else '❌'} | "
            f"{'✅' if r['target_matched'] else '❌'} | "
            f"{'✅' if r['fully_correct'] else '❌'} | {r['confidence']} |"
        )

    L.append("\n## Occlusion graph per case\n")
    for r in results:
        L.append(f"### {r['case']} — query={r['query']!r}  ({'CORRECT' if r['fully_correct'] else 'WRONG'})")
        chain = r.get("occlusion_chain", [])
        if chain:
            L.append("**occlusion_chain (edges):**")
            for e in chain:
                L.append(f"- `{e.get('occluded','')}` ← occluded by `{e.get('occluder','')}`  (edge_confidence={e.get('edge_confidence')})")
        else:
            L.append("**occlusion_chain:** [] (target not blocked per model)")
        cands = r.get("candidates", [])
        if cands:
            L.append("\n**candidates (prob_free):** " +
                     ", ".join(f"ID {c.get('id')}:{c.get('object','')}={c.get('prob_free')}" for c in cands))
        L.append(f"\n**candidate_ids:** {r.get('candidate_ids', [])}")
        L.append(f"**free_object_ids:** {r.get('free_object_ids', [])}")
        L.append(f"**chosen_to_remove_ids:** {r.get('chosen_to_remove_ids', [])}")
        if r.get("invalid_ids"):
            L.append(f"**invalid_ids:** {r.get('invalid_ids', [])}")
        L.append(f"\n**free_objects:** {r.get('free_objects', [])}")
        L.append(f"**chosen_to_remove:** {r.get('chosen_to_remove', [])}\n")

    L.append("## evaluate_nlp.py output\n")
    L.append("```\n" + eval_text.strip() + "\n```")

    report_path = out_root / "report.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Report written → {report_path}")


# ── main ──────────────────────────────────────────────────────────────────────────
def select_freegrasp_cases(manifest_cases, n_cases, balanced):
    if not balanced:
        return manifest_cases[:n_cases]

    difficulties = ["Easy", "Medium", "Hard"]
    base_n = n_cases // len(difficulties)
    extra = n_cases % len(difficulties)
    selected = []
    for i, difficulty in enumerate(difficulties):
        count = base_n + (1 if i < extra else 0)
        group = [c for c in manifest_cases if c.get("difficulty") == difficulty]
        if len(group) < count:
            raise ValueError(f"Need {count} {difficulty} FreeGrasp cases, only {len(group)} available.")
        selected.extend(group[:count])
    selected.sort(key=lambda c: int(c.get("sample_index", 0)))
    return selected


def make_freegrasp_case(record, out_root, mask_dir):
    sample_dir = ROOT / record["sample_dir"]
    gt_path = sample_dir / "gt.json"
    gt = json.load(open(gt_path))
    scene_id = int(gt["scene_id"])
    sample_index = int(record.get("sample_index", -1))
    eval_image_id = 900000 + sample_index
    query_mask_id = int(gt["query_mask_id"])
    gt_mask_ids = [int(i) for i in gt["ground_truth_mask_ids"]]
    gt_obj_ids = [int(i) for i in gt["ground_truth_obj_ids"]]

    npz_path = ROOT / "data/npz_file" / f"{scene_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"FreeGrasp NPZ not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    if "instances_objects" not in npz:
        raise KeyError(f"instances_objects not found in {npz_path}; keys={list(npz.keys())}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"image_{eval_image_id:06d}.npy"
    if not mask_path.exists():
        np.save(mask_path, npz["instances_objects"].astype(np.int32))

    return {
        "dataset": "freegrasp",
        "case_name": f"fg_{sample_index:03d}_scene{scene_id:06d}_q{query_mask_id}",
        "image_id": eval_image_id,
        "eval_image_id": eval_image_id,
        "scene_id": scene_id,
        "image_path_abs": str(sample_dir / "image.png"),
        "eval_npz_root": str(mask_dir),
        "query_object": {
            "obj_id": query_mask_id,
            "object_name": str(gt["annotation"]),
        },
        "target_objects": [
            {"obj_id": mask_id, "object_name": f"mask_id_{mask_id}"}
            for mask_id in gt_mask_ids
        ],
        "occlusion_paths": [[mask_id, query_mask_id] for mask_id in gt_mask_ids],
        "difficulty": str(gt["difficulty"]),
        "freegrasp_query_obj_id": int(gt["query_obj_id"]),
        "ground_truth_obj_ids": gt_obj_ids,
        "ground_truth_mask_ids": gt_mask_ids,
        "sample_index": sample_index,
        "sample_dir": str(sample_dir),
    }


def run_cases(cases, out_root, args, gt_path, gt_records, npz_root):
    out_root.mkdir(parents=True, exist_ok=True)
    json.dump(gt_records, open(gt_path, "w"), indent=2)
    logger.info(f"GT file → {gt_path}  ({len(gt_records)} records)")

    pred_path = out_root / "predictions.jsonl"
    results_by_case = {}
    for case in cases:
        case_name = _case_name(case)
        result_path = out_root / case_name / "result.json"
        if result_path.exists():
            try:
                results_by_case[case_name] = json.load(open(result_path))
            except Exception as e:
                logger.warning(f"Could not load existing result {result_path}: {e}")

    remaining = [c for c in cases if _case_name(c) not in results_by_case]
    logger.info(
        f"Running {len(remaining)}/{len(cases)} remaining cases "
        f"({len(results_by_case)} existing results found) …"
    )

    with open(pred_path, "a") as pred_fh:
        for i, case in enumerate(cases, 1):
            case_name = _case_name(case)
            if case_name in results_by_case:
                logger.info(f"\n===== case {i}/{len(cases)}: {case_name} already done; skipping =====")
                continue
            logger.info(f"\n===== case {i}/{len(cases)} =====")
            try:
                r = process_case(case, args, out_root, pred_fh)
                if r:
                    results_by_case[case_name] = r
            except Exception as e:
                logger.error(f"  FAILED {case_name}: {e}", exc_info=True)

    results = [
        results_by_case[_case_name(case)]
        for case in cases
        if _case_name(case) in results_by_case
    ]
    summary = summarize(results, out_root) if results else None

    eval_script = ROOT / "evaluate_nlp.py"
    eval_out = out_root / "eval_results.txt"
    logger.info("\nRunning evaluate_nlp.py …")
    proc = subprocess.run(
        [
            "python", str(eval_script),
            "--pred_path", str(pred_path),
            "--gt_path", str(gt_path),
            "--npz_root", str(npz_root),
            "--dataset_type", "synthetic",
        ],
        capture_output=True, text=True,
    )
    eval_text = proc.stdout + ("\n" + proc.stderr if proc.stderr.strip() else "")
    eval_out.write_text(eval_text)
    logger.info(eval_text)
    logger.info(f"evaluate_nlp results saved → {eval_out}")

    if results and summary is not None:
        write_report(results, summary, eval_text, out_root, args)


def main_unobench(args):
    all_cases = json.load(open(ROOT / args.case_list))["cases"]
    if args.difficulty:
        all_cases = [c for c in all_cases if c.get("difficulty") == args.difficulty]
        logger.info(f"Filtered to difficulty={args.difficulty!r}: {len(all_cases)} cases available.")
    cases = all_cases[:args.n_cases]
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)

    ensure_images(cases, IMG_ZIP, EXTRACT)
    ensure_annotations(cases, ANN_ZIP, ANN_DIR)
    ensure_depth(cases, DEPTH_DIR, META_ZIP, ID_MAP)

    gt_path = out_root / "gt_synthetic_eval.json"
    gt_records = [
        {
            "image_id": case["image_id"],
            "query_object": case["query_object"]["obj_id"],
            "occlusion_paths": case.get("occlusion_paths", []),
            "top_objects": [t["obj_id"] for t in case["target_objects"]],
            "new_difficulty": case["difficulty"],
        }
        for case in cases
    ]
    run_cases(cases, out_root, args, gt_path, gt_records, ANN_DIR)


def main_freegrasp(args):
    freegrasp_root = ROOT / args.freegrasp_root
    manifest_path = freegrasp_root / "manifest.json"
    manifest = json.load(open(manifest_path))
    selected_records = select_freegrasp_cases(manifest["cases"], args.n_cases, args.balanced)

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    mask_dir = out_root / "freegrasp_masks"
    cases = [make_freegrasp_case(record, out_root, mask_dir) for record in selected_records]

    counts = {}
    for case in cases:
        counts[case["difficulty"]] = counts.get(case["difficulty"], 0) + 1
    logger.info(f"FreeGrasp selected cases: {len(cases)}  difficulty_counts={counts}")

    gt_path = out_root / "gt_freegrasp_eval.json"
    gt_records = [
        {
            "image_id": case["image_id"],
            "query_object": case["query_object"]["obj_id"],
            "occlusion_paths": case.get("occlusion_paths", []),
            "top_objects": [t["obj_id"] for t in case["target_objects"]],
            "new_difficulty": case["difficulty"],
        }
        for case in cases
    ]
    run_cases(cases, out_root, args, gt_path, gt_records, mask_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-list", default="data/unobench_eval/case_list_synthetic_100.json")
    ap.add_argument("--out",       default="logs/unobench_eval_synthetic")
    ap.add_argument("--n-cases",   type=int, default=50)
    ap.add_argument("--difficulty", default=None,
                    help="Filter cases by difficulty (e.g. Hard) before taking --n-cases.")
    ap.add_argument("--freegrasp-root", default=None,
                    help="Run on extracted FreeGrasp samples instead of UnoBench.")
    ap.add_argument("--balanced", action="store_true",
                    help="For FreeGrasp mode, select balanced Easy/Medium/Hard cases.")
    args = ap.parse_args()

    if args.freegrasp_root:
        main_freegrasp(args)
    else:
        main_unobench(args)


if __name__ == "__main__":
    main()
