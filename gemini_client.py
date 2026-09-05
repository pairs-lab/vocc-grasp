#!/usr/bin/env python3
"""
gemini_client.py — Gemini API client and helpers (pointing, labeled image, reasoning).

Both pointing (object detection) and reasoning (which object to pick/remove)
are handled by the same Gemini model, replacing Molmo + GPT-4o.

Pipeline per scene
──────────────────
1. Pointing  : Gemini detects all objects → list of {id, label, point[y,x] in [0,1000]}
               → convert to pixel coords → draw numbered labeled image
2. Reasoning : Gemini + (labeled image + object list + task) → [action, id, class]
3. Verify    : Gemini verifies the proposed action (same model as verifier)
4. Save      : per-scene JSON, labeled image, text log

Evaluation
──────────
Ground truth from obstruction_graph_v3.json:
  • target_class  — object to eventually grasp
  • edges         — if non-empty → target is occluded → expected "remove obstacle"
  • ancestors     — objects that must be removed
  • removal_order — first element = first obstacle to remove

Metrics computed
  • target_identified   : Gemini detected an object matching the target description
  • action_type_correct : "pick" vs "remove obstacle" matches ground truth
  • pointing_count      : number of objects Gemini detected

Usage:
    python gemini_client.py
    python gemini_client.py --path data/medium/sample_demo_0   # single scene
    python gemini_client.py --max-verify-retries 1
"""

import os
import re
import sys
import csv
import json
import time
import base64
import logging
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent

# ─── Gemini client ────────────────────────────────────────────────────────────

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_REQUEST_DELAY_SECONDS = float(os.getenv("GEMINI_REQUEST_DELAY_SECONDS", "0"))
GEMINI_HTTP_TIMEOUT_MS = float(os.getenv("GEMINI_HTTP_TIMEOUT_MS", "60000"))
MODEL = "gemini-robotics-er-1.6-preview"

import google.genai as genai
from google.genai import types as gtypes

_client = None
_last_gemini_request_at = 0.0


def _get_client():
    global _client
    if _client is None:
        if not GEMINI_KEY:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "in the environment or in a .env file next to the scripts."
            )
        _client = genai.Client(api_key=GEMINI_KEY)
    return _client


def _with_http_timeout(config=None):
    if GEMINI_HTTP_TIMEOUT_MS <= 0:
        return config
    http_options = gtypes.HttpOptions(timeout=GEMINI_HTTP_TIMEOUT_MS)
    if config is None:
        return gtypes.GenerateContentConfig(http_options=http_options)
    if getattr(config, "http_options", None) is None:
        config.http_options = http_options
    return config


def _wait_for_gemini_rate_delay():
    """Ensure Gemini requests are spaced out for free-tier rate limits."""
    global _last_gemini_request_at
    if GEMINI_REQUEST_DELAY_SECONDS <= 0:
        _last_gemini_request_at = time.monotonic()
        return
    elapsed = time.monotonic() - _last_gemini_request_at
    wait_s = GEMINI_REQUEST_DELAY_SECONDS - elapsed
    if wait_s > 0:
        logger.info(f"[Gemini] Waiting {wait_s:.1f}s before next request …")
        time.sleep(wait_s)
    _last_gemini_request_at = time.monotonic()


def generate_content(*, contents: list, model: str = None, config=None):
    """Gemini generate_content with optional inter-request delay."""
    _wait_for_gemini_rate_delay()
    kwargs = {"model": model or MODEL, "contents": contents}
    config = _with_http_timeout(config)
    if config is not None:
        kwargs["config"] = config
    return _get_client().models.generate_content(**kwargs)


def _call_gemini(contents: list, max_retries: int = 3, base_delay: float = 4.0) -> str:
    """Call Gemini with retry on transient errors. Returns text string."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = generate_content(model=MODEL, contents=contents)
            text = resp.text
            if text and text.strip():
                return text.strip()
            last_err = f"empty response on attempt {attempt+1}"
            logger.warning(f"[Gemini] {last_err}")
        except Exception as exc:
            last_err = str(exc)
            logger.warning(f"[Gemini] attempt {attempt+1} error: {last_err}")
        if attempt < max_retries:
            delay = base_delay * (2 ** attempt)
            logger.info(f"[Gemini] Retrying in {delay:.0f}s …")
            time.sleep(delay)
    raise RuntimeError(f"Gemini call failed after {max_retries+1} attempts. Last: {last_err}")


# ─── Pointing ─────────────────────────────────────────────────────────────────

_POINTING_PROMPT = (
    "Detect all distinct objects on the table (not the table surface, background, "
    "or robotic arm/gripper). For each object return a JSON array with format:\n"
    '[{"id": 1, "label": "...", "point": [y, x]}]\n'
    "where point [y, x] are normalized coordinates in range [0, 1000] "
    "(0=top-left, 1000=bottom-right). Number objects 1, 2, 3, … "
    "Output ONLY the JSON array, no markdown fences, no extra text."
)


def gemini_pointing(image_bytes: bytes, image_size: tuple) -> list:
    """
    Ask Gemini to detect all objects.
    Returns list of dicts: {id, label, x_px, y_px, y_norm, x_norm}
    """
    text = _call_gemini([
        gtypes.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        _POINTING_PROMPT,
    ])

    # Strip markdown fences if present
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip())

    W, H = image_size
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON array substring
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            logger.warning(f"[Pointing] Could not parse JSON from: {text[:200]}")
            return []
        raw = json.loads(m.group())

    objects = []
    for obj in raw:
        obj_id = int(obj.get("id", len(objects) + 1))
        label = str(obj.get("label", f"object_{obj_id}"))
        pt = obj.get("point", obj.get("bbox_2d", None))
        if pt is None:
            continue
        if len(pt) == 4:
            # bbox → use center
            y_norm = (pt[0] + pt[2]) / 2.0
            x_norm = (pt[1] + pt[3]) / 2.0
        else:
            y_norm, x_norm = float(pt[0]), float(pt[1])
        x_px = int(x_norm / 1000.0 * W)
        y_px = int(y_norm / 1000.0 * H)
        objects.append({
            "id": obj_id,
            "label": label,
            "x_px": max(0, min(W - 1, x_px)),
            "y_px": max(0, min(H - 1, y_px)),
            "y_norm": round(y_norm, 1),
            "x_norm": round(x_norm, 1),
        })

    logger.info(f"  [Pointing] Detected {len(objects)} objects")
    return objects


# ─── Labeled image ────────────────────────────────────────────────────────────

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
]


def draw_labeled_image(image_np: np.ndarray, objects: list,
                       save_path: Path) -> str:
    """Draw SoM-style labeled image; return base64 string."""
    H, W = image_np.shape[:2]
    dpi = 150
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image_np)
    ax.axis("off")

    for obj in objects:
        c = _COLORS[(obj["id"] - 1) % len(_COLORS)]
        x, y = obj["x_px"], obj["y_px"]
        ax.plot(x, y, "o", color=c, markersize=10, markeredgecolor="white",
                markeredgewidth=1.5)
        ax.text(x, y, str(obj["id"]),
                color="white", fontsize=8, fontweight="bold",
                ha="center", va="center",
                bbox=dict(facecolor=c, alpha=0.85, edgecolor="white",
                          linewidth=1.2, boxstyle="round,pad=0.2"))

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    with open(save_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ─── Reasoning ────────────────────────────────────────────────────────────────

_REASONING_SYSTEM = """\
You are a robotic bin-picking system using a parallel gripper.

Objects are numbered in the image with colored ID badges.

Rules:
1. If the target object has another object physically ON TOP of it (stacked, covering it,
   blocking gripper access from above), you must remove that obstacle first.
2. Objects placed NEXT TO each other (side by side) are NOT occluding each other.

Think step by step, then decide:
- REASONING: In 1-3 sentences, state (a) which numbered object is the target, and
  (b) whether any other object is physically resting ON TOP of / covering the target
  from above (name the object IDs), versus merely lying beside it. Be explicit about the
  3D spatial relationship (above / below / beside / overlapping) you observe.
- DECISION: On the FINAL line, output ONLY this exact format — nothing after it:
   [action, object_id, class_name]
   where action is either "pick object" or "remove obstacle".

Example:
   REASONING: The target is the red spray bottle (ID 3). It sits in the open with
   nothing on top of it; ID 5 is beside it, not above. So it can be grasped directly.
   [pick object, 3, red spray bottle]
"""


def gemini_reasoning(labeled_image_b64: str, objects: list, task: str,
                     extra_context: str = "") -> dict:
    """
    Ask Gemini to decide which action to take.
    Returns: {action, object_id, class_name, raw_output}
    """
    obj_list_str = "\n".join(
        f"  ID {o['id']}: {o['label']}" for o in objects
    )
    user_text = (
        f"{_REASONING_SYSTEM}\n\n"
        f"Detected objects:\n{obj_list_str}\n\n"
        f"Task: pick up \"{task}\"\n"
    )
    if extra_context:
        user_text += f"\n{extra_context}\n"
    user_text += "\nWhat should the robot do next?"

    img_bytes = base64.b64decode(labeled_image_b64)
    raw = _call_gemini([
        gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        user_text,
    ])

    result = _parse_action(raw)
    result["raw_output"] = raw
    logger.info(f"  [Reasoning] raw={raw!r} → parsed={result}")
    return result


def _parse_action(text: str) -> dict:
    """Parse [action, id, class] from raw text (use the LAST bracket = decision line)."""
    # Try: [pick object, 3, red box] or [remove obstacle, 7, blue box]
    matches = re.findall(
        r"\[(pick object|remove obstacle)[,\s]+(\d+)[,\s]+(.+?)\]",
        text, re.IGNORECASE
    )
    if matches:
        action, obj_id, class_name = matches[-1]
        return {
            "action": action.lower().strip(),
            "object_id": int(obj_id),
            "class_name": class_name.strip(),
        }
    # Fallback: just grab first number and action keyword
    action = "pick object"
    if re.search(r"remove obstacle", text, re.IGNORECASE):
        action = "remove obstacle"
    num = re.search(r"\b(\d+)\b", text)
    obj_id = int(num.group(1)) if num else 0
    return {"action": action, "object_id": obj_id, "class_name": "unknown"}


# ─── Verification ─────────────────────────────────────────────────────────────

_VERIFIER_PROMPT = """\
You are a quality-checker for a robotic bin-picking system.

Verify whether the proposed action is correct given the labeled image and task.

## Check
1. Correct target: does the chosen object match the task description?
2. Physical blocking: is another object clearly STACKED ON TOP of the target (not just next to it)?
3. If blocked: is the proposed obstacle actually physically on top of the target?

## Rules
- Side-by-side objects are NOT occluding each other.
- Only flag occlusion when an object is visibly ON TOP / resting on the target.
- When unsure, assume the object is free.

Respond ONLY with valid JSON:
{"is_correct": true/false, "reason": "...", "corrected_id": null_or_int, "corrected_class": null_or_str}
"""


def gemini_verify(labeled_image_b64: str, objects: list, task: str,
                  action_text: str) -> dict:
    """Returns {is_correct, reason, corrected_id, corrected_class}."""
    obj_list_str = "\n".join(f"  ID {o['id']}: {o['label']}" for o in objects)
    user_text = (
        f"{_VERIFIER_PROMPT}\n\n"
        f"Objects:\n{obj_list_str}\n\n"
        f"Task: \"{task}\"\n"
        f"Proposed action: {action_text}\n\n"
        "Is this action correct? Respond with JSON."
    )
    img_bytes = base64.b64decode(labeled_image_b64)
    try:
        raw = _call_gemini([
            gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            user_text,
        ])
        raw_clean = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw_clean = re.sub(r"```$", "", raw_clean.strip())
        m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
        result = json.loads(m.group()) if m else json.loads(raw_clean)
        return {
            "is_correct": bool(result.get("is_correct", True)),
            "reason": str(result.get("reason", "")),
            "corrected_id": result.get("corrected_id"),
            "corrected_class": result.get("corrected_class"),
        }
    except Exception as exc:
        logger.warning(f"  [Verify] failed: {exc} — assuming correct")
        return {"is_correct": True, "reason": f"verifier error: {exc}",
                "corrected_id": None, "corrected_class": None}


# ─── Confidence elicitation ───────────────────────────────────────────────────
# Adapted from the user-provided LLM_RESPONSE_PROMPT (confidence-score template).

_CONFIDENCE_PROMPT = """\
You are an intelligent assistant who is given a question. Your role is to provide accurate,
helpful, and well-reasoned responses based on your knowledge and capabilities.

Along with the question, you need to provide a confidence score for your answer. The
confidence score should be a number between 0 and 100, where:
- 0-25 indicates low confidence
- 26-75 indicates moderate confidence
- 76-100 indicates high confidence

Guidelines for providing answers:
1. Be direct and concise in your answer while ensuring completeness. Avoid unnecessary words or tangents.
2. If you are uncertain, provide a lower confidence score.
3. Base your confidence score on:
   - The reliability and recency of available information
   - Your knowledge of the specific domain

Here are some examples:

Example 1:
Question: What is the capital of France?
Answer: Paris
Confidence score: 91
(High confidence as this is a well-established fact)

Example 2:
Question: Which country has the best healthcare system?
Answer: It depends on the criteria used. Some rankings favor Switzerland, while others favor Sweden or Singapore.
Confidence score: 25
(There is no definitive answer, and the confidence is low due to the lack of a clear consensus.)

Example 3:
Question: Which state is between Washington and California?
Answer: Oregon
Confidence score: 87
(Maximum confidence as this is a clear geographic fact)

Example 4:
Question: What was Albert Einstein's favorite food?
Answer: There is no definitive record of his favorite food, but he reportedly liked pasta.
Confidence score: 25
(There are anecdotal mentions, but no verified records.)

Example 5:
Question: Is Irvine a city in California?
Answer: Yes
Confidence score: 81
(High confidence as this is a verifiable fact)

Example 6:
Question: What is the most popular programming language for AI development?
Answer: Python
Confidence score: 66
(Moderate-high confidence based on current trends, but this can change over time)

Here is a new example. Simply reply with your answer and confidence score.

Question: {question}

Provide your response in the following JSON format:
{{
  "answer": "Your answer here",
  "confidence_score": number between 0-100
}}
"""


def gemini_confidence(labeled_image_b64: str, task: str,
                      action: str, object_id: int, class_name: str) -> dict:
    """
    Ask the VLM for a confidence score (0-100) in its bin-picking decision,
    using the user-provided confidence-score prompt template.
    Returns {answer, confidence_score, raw}.
    """
    question = (
        "You are a robotic bin-picking system with a parallel gripper, looking at the "
        f"attached image where objects carry numbered ID badges. The goal is to grasp: "
        f"\"{task}\". An object is 'pick object' if it is clear/free from above, or "
        "'remove obstacle' if another object is resting ON TOP of the target and must be "
        "removed first (objects merely beside the target do NOT count). "
        f"The proposed next action is: \"{action}\" on object ID {object_id} ({class_name}). "
        "What is the correct next action for the target — \"pick object\" or "
        "\"remove obstacle\"? Answer with exactly one of those two phrases."
    )
    prompt = _CONFIDENCE_PROMPT.format(question=question)
    img_bytes = base64.b64decode(labeled_image_b64)
    try:
        raw = _call_gemini([
            gtypes.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            prompt,
        ])
        raw_clean = re.sub(r"^```[a-z]*\n?", "", raw.strip(), flags=re.MULTILINE)
        raw_clean = re.sub(r"```$", "", raw_clean.strip())
        m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
        obj = json.loads(m.group()) if m else json.loads(raw_clean)
        ans = str(obj.get("answer", "")).strip()
        score = obj.get("confidence_score", None)
        score = float(score) if score is not None else None
        # Normalize answer to canonical action phrase if present
        ans_norm = "remove obstacle" if re.search(r"remove", ans, re.IGNORECASE) else (
            "pick object" if re.search(r"pick", ans, re.IGNORECASE) else ans)
        logger.info(f"  [Confidence] answer={ans_norm!r}  score={score}")
        return {"answer": ans_norm, "confidence_score": score, "raw": raw}
    except Exception as exc:
        logger.warning(f"  [Confidence] failed: {exc}")
        return {"answer": None, "confidence_score": None, "raw": f"error: {exc}"}


# ─── Ground truth helpers ─────────────────────────────────────────────────────

def _load_gt(scene_path: Path) -> dict:
    """Load ground truth; tries v3 then falls back to base obstruction_graph.json."""
    for name in ("obstruction_graph_v3.json", "obstruction_graph.json"):
        gt_file = scene_path / name
        if gt_file.exists():
            return json.loads(gt_file.read_text())
    return {}


def _gt_expected_action(gt: dict) -> str:
    """'remove obstacle' if target is blocked, 'pick object' otherwise."""
    if gt.get("edges"):
        return "remove obstacle"
    return "pick object"


def _label_matches(detected_label: str, target_class: str) -> bool:
    """Fuzzy label match: check if any word from target_class appears in detected_label."""
    tgt_words = set(target_class.lower().split())
    det_words = set(detected_label.lower().split())
    overlap = tgt_words & det_words
    return len(overlap) >= 1


# ─── Per-scene runner ─────────────────────────────────────────────────────────

def run_scene(scene_path: Path, output_dir: Path,
              max_verify_retries: int = 2) -> dict:
    scene_path = scene_path.resolve()
    scene_name = f"{scene_path.parent.name}_{scene_path.name}"
    image_path = scene_path / "image.png"
    if not image_path.exists():
        logger.warning(f"[{scene_name}] No image.png — skipping")
        return {}

    task = (scene_path / "task.txt").read_text().strip().splitlines()[0] \
        if (scene_path / "task.txt").exists() else ""
    gt = _load_gt(scene_path)
    gt_target_class = gt.get("target_class", "")
    gt_action = _gt_expected_action(gt)
    gt_ancestors = gt.get("ancestors", [])
    gt_removal_order = gt.get("removal_order", [])

    out_scene = output_dir / scene_name
    out_scene.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{scene_name}]  task={task!r}  gt_action={gt_action!r}")

    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    image_bytes = image_path.read_bytes()

    # ── Step 1: Pointing ──────────────────────────────────────────────────────
    logger.info(f"  Step 1: Pointing …")
    objects = gemini_pointing(image_bytes, image_pil.size)

    # Save id mapping (same format as molmo_id.txt for compatibility)
    id_txt = "Gemini_ID X Y Label\n" + "\n".join(
        f"{o['id']} {o['x_px']} {o['y_px']} {o['label']}" for o in objects
    )
    (out_scene / "gemini_id.txt").write_text(id_txt)

    # Draw labeled image
    label_img_path = out_scene / "gemini_label.png"
    labeled_b64 = draw_labeled_image(image_np, objects, label_img_path)
    logger.info(f"  Labeled image → {label_img_path}")

    # ── Step 2: Reasoning (single pass, no verify) ───────────────────────────
    logger.info(f"  Step 2: Reasoning …")
    reason_result = gemini_reasoning(labeled_b64, objects, task)
    action_text = (
        f"[{reason_result['action']}, {reason_result['object_id']}, "
        f"{reason_result['class_name']}]"
    )
    feedback_log = []

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    final_action = reason_result["action"].lower().strip()
    final_id = reason_result["object_id"]
    final_class = reason_result["class_name"]

    # ── Confidence elicitation ────────────────────────────────────────────────
    logger.info(f"  Step 3: Confidence …")
    conf = gemini_confidence(labeled_b64, task, final_action, final_id, final_class)
    confidence_score = conf.get("confidence_score")
    confidence_answer = conf.get("answer")

    action_type_correct = (final_action == gt_action) if gt else None

    # Check if any detected object matches the target description
    target_obj = next(
        (o for o in objects if _label_matches(o["label"], task)), None
    )
    target_identified = target_obj is not None

    # If action is "pick object", check if it's actually picking the target
    picking_target = False
    if final_action == "pick object" and target_obj:
        picking_target = (final_id == target_obj["id"]) or \
                         _label_matches(final_class, gt_target_class or task)

    # ── Step 4: Write log ─────────────────────────────────────────────────────
    log_lines = [
        f"Scene: {scene_name}",
        f"Task: {task}",
        f"GT target class: {gt_target_class}",
        f"GT expected action: {gt_action}",
        f"GT ancestors: {gt_ancestors}",
        f"GT removal_order: {gt_removal_order}",
        "",
        f"Gemini detected {len(objects)} objects:",
    ] + [f"  ID {o['id']}: {o['label']}  ({o['x_px']}, {o['y_px']})" for o in objects] + [
        "",
        "=== VLM RAW REASONING ===",
        reason_result.get("raw_output", ""),
        "",
        f"=== CONFIDENCE === answer={confidence_answer!r}  score={confidence_score}",
        f"Confidence raw: {conf.get('raw','')}",
        "",
        f"Final action: {action_text}",
        f"Action type correct: {action_type_correct}",
        f"Target identified: {target_identified}",
        f"Picking correct target: {picking_target}",
        "",
        "=== Feedback Log ===",
    ] + [
        f"  Attempt {e['attempt']}: [{'OK' if e['is_correct'] else 'WRONG'}] "
        f"{e['action_text']}  reason={e['reason']!r}"
        for e in feedback_log
    ]
    (out_scene / "log_gemini.txt").write_text("\n".join(log_lines))

    result = {
        "scene": scene_name,
        "task": task,
        "gt_target_class": gt_target_class,
        "gt_action": gt_action,
        "gt_num_ancestors": len(gt_ancestors),
        "gemini_pointing_count": len(objects),
        "gemini_final_action": final_action,
        "gemini_final_id": final_id,
        "gemini_final_class": final_class,
        "vlm_reasoning": reason_result.get("raw_output", ""),
        "confidence_score": confidence_score,
        "confidence_answer": confidence_answer,
        "confidence_agrees_decision": (confidence_answer == final_action) if confidence_answer else None,
        "action_type_correct": action_type_correct,
        "target_identified": target_identified,
        "picking_target_correct": picking_target,
        "num_verify_attempts": 0,
        "final_verify_passed": None,
        "objects": objects,
        "feedback_log": feedback_log,
    }

    with open(out_scene / "gemini_result.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"  → action={final_action!r} id={final_id} class={final_class!r}  "
        f"action_correct={action_type_correct}  target_found={target_identified}"
    )
    return result


# ─── Batch runner ─────────────────────────────────────────────────────────────

def collect_scenes(data_root: Path, difficulty: str = "medium") -> list:
    folder = data_root / difficulty
    if not folder.exists():
        return []
    return sorted(
        [d for d in folder.iterdir() if d.is_dir() and (d / "image.png").exists()]
    )


def run_all(output_dir: Path, max_verify_retries: int = 2,
            difficulty: str = "medium") -> list:
    scenes = collect_scenes(ROOT / "data", difficulty)
    logger.info(f"Found {len(scenes)} {difficulty} scenes")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, sp in enumerate(scenes, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Scene {i}/{len(scenes)}: {sp.name}")
        logger.info(f"{'='*60}")
        try:
            result = run_scene(sp, output_dir, max_verify_retries)
            if result:
                all_results.append(result)
        except Exception as exc:
            logger.error(f"  FAILED: {exc}", exc_info=True)
            all_results.append({
                "scene": f"medium_{sp.name}",
                "task": "",
                "error": str(exc),
                "action_type_correct": None,
                "target_identified": False,
            })

    return all_results


# ─── Summary & evaluation ─────────────────────────────────────────────────────

def compute_summary(results: list, output_dir: Path) -> dict:
    valid = [r for r in results if "error" not in r]
    n = len(valid)
    if n == 0:
        logger.warning("No valid results.")
        return {}

    action_correct = [r for r in valid if r.get("action_type_correct") is True]
    target_found   = [r for r in valid if r.get("target_identified") is True]
    errors         = [r for r in results if "error" in r]

    # Breakdown by GT action type
    pick_scenes    = [r for r in valid if r.get("gt_action") == "pick object"]
    remove_scenes  = [r for r in valid if r.get("gt_action") == "remove obstacle"]
    pick_correct   = [r for r in pick_scenes if r.get("action_type_correct")]
    remove_correct = [r for r in remove_scenes if r.get("action_type_correct")]

    def pct(num, den): return round(100 * num / den, 1) if den > 0 else 0.0
    def mean_or_none(xs): return round(float(np.mean(xs)), 1) if xs else None

    # ── Confidence statistics ──────────────────────────────────────────────────
    conf_all     = [r["confidence_score"] for r in valid if r.get("confidence_score") is not None]
    conf_correct = [r["confidence_score"] for r in action_correct if r.get("confidence_score") is not None]
    conf_wrong   = [r["confidence_score"] for r in valid
                    if r.get("action_type_correct") is False and r.get("confidence_score") is not None]
    # Accuracy by confidence bucket
    buckets = {"0-25 (low)": (0, 25), "26-75 (moderate)": (26, 75), "76-100 (high)": (76, 100)}
    bucket_stats = {}
    for name, (lo, hi) in buckets.items():
        rows = [r for r in valid if r.get("confidence_score") is not None
                and lo <= r["confidence_score"] <= hi]
        acc = pct(sum(1 for r in rows if r.get("action_type_correct")), len(rows))
        bucket_stats[name] = {"n": len(rows), "accuracy": acc}
    # Does confidence-answer agree with the reasoning decision?
    conf_agree = [r for r in valid if r.get("confidence_agrees_decision") is True]

    summary = {
        "total_scenes": len(results),
        "valid_scenes": n,
        "errors": len(errors),
        "action_type_accuracy": pct(len(action_correct), n),
        "target_identification_rate": pct(len(target_found), n),
        "mean_pointing_count": round(np.mean([r["gemini_pointing_count"] for r in valid]), 1),
        "pick_scenes": len(pick_scenes),
        "pick_correct": len(pick_correct),
        "pick_accuracy": pct(len(pick_correct), len(pick_scenes)),
        "remove_scenes": len(remove_scenes),
        "remove_correct": len(remove_correct),
        "remove_accuracy": pct(len(remove_correct), len(remove_scenes)),
        "mean_verify_attempts": round(
            np.mean([r.get("num_verify_attempts", 1) for r in valid]), 2),
        "mean_confidence": mean_or_none(conf_all),
        "mean_confidence_when_correct": mean_or_none(conf_correct),
        "mean_confidence_when_wrong": mean_or_none(conf_wrong),
        "confidence_buckets": bucket_stats,
        "confidence_answer_agrees_decision": pct(len(conf_agree), n),
    }

    logger.info("\n" + "="*65)
    logger.info("GEMINI-ROBOTICS-ER-1.6 EVALUATION SUMMARY")
    logger.info("="*65)
    logger.info(f"  Scenes evaluated  : {n}/{len(results)}")
    logger.info(f"  Action type acc.  : {summary['action_type_accuracy']:.1f}%  "
                f"({len(action_correct)}/{n})")
    logger.info(f"    pick  : {summary['pick_accuracy']:.1f}%  "
                f"({len(pick_correct)}/{len(pick_scenes)})")
    logger.info(f"    remove: {summary['remove_accuracy']:.1f}%  "
                f"({len(remove_correct)}/{len(remove_scenes)})")
    logger.info(f"  Target found      : {summary['target_identification_rate']:.1f}%")
    logger.info(f"  Mean objects det. : {summary['mean_pointing_count']}")
    logger.info(f"  ── Confidence ──")
    logger.info(f"  Mean confidence       : {summary['mean_confidence']}")
    logger.info(f"  Mean conf (correct)   : {summary['mean_confidence_when_correct']}")
    logger.info(f"  Mean conf (wrong)     : {summary['mean_confidence_when_wrong']}")
    for name, st in bucket_stats.items():
        logger.info(f"    {name:18}: n={st['n']:2}  acc={st['accuracy']}%")
    logger.info(f"  Conf-answer agrees decision: {summary['confidence_answer_agrees_decision']}%")
    logger.info("="*65)

    # Per-scene table
    logger.info("\nPer-scene results (sorted by action_correct):")
    sorted_r = sorted(valid, key=lambda r: (
        not r.get("action_type_correct", False),
        r["scene"]
    ))
    for r in sorted_r:
        ok = "✓" if r.get("action_type_correct") else "✗"
        tf = "T+" if r.get("target_identified") else "T-"
        logger.info(
            f"  {ok} {tf}  {r['scene']:<40}  "
            f"gt={r.get('gt_action','?')[:6]}  "
            f"pred={r.get('gemini_final_action','?')[:6]}  "
            f"det={r.get('gemini_pointing_count',0)}obj  "
            f"task={r.get('task','')[:35]!r}"
        )

    # Save JSON summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"summary": summary, "per_scene": results}, f, indent=2)
    logger.info(f"\nSummary JSON → {summary_path}")

    # Save CSV
    csv_path = output_dir / "summary.csv"
    csv_fields = [
        "scene", "task", "gt_target_class", "gt_action", "gt_num_ancestors",
        "gemini_pointing_count", "gemini_final_action", "gemini_final_id",
        "gemini_final_class", "action_type_correct", "target_identified",
        "confidence_score", "confidence_answer", "confidence_agrees_decision",
        "num_verify_attempts", "final_verify_passed",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"Summary CSV  → {csv_path}")

    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run gemini-robotics-er-1.6-preview on 28 medium grasp scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--path", default=None,
                        help="Single scene directory (default: run all scenes)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: logs/gemini_robotics_eval/<difficulty>)")
    parser.add_argument("--difficulty", default="medium",
                        help="Data subfolder to evaluate: medium | hard (default: medium)")
    parser.add_argument("--max-verify-retries", type=int, default=2)
    args = parser.parse_args()

    out_dir = (Path(args.output_dir) if args.output_dir
               else ROOT / "logs" / "gemini_robotics_eval" / args.difficulty)

    if args.path:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = run_scene(Path(args.path), out_dir, args.max_verify_retries)
        if result:
            compute_summary([result], out_dir)
    else:
        results = run_all(out_dir, args.max_verify_retries, args.difficulty)
        compute_summary(results, out_dir)


if __name__ == "__main__":
    main()
