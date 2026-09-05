#!/usr/bin/env python3
"""
Batch-mode variant of run_gemini_uoais_ref.py using the Gemini Batch API
(50% cheaper than interactive calls). Reuses the exact same prompts, parsers,
UOAIS reference builder, and eval as the interactive runner — only the API
transport changes.

Edge scoring is shared with the interactive runner: UOAIS candidates go through
run_uoais_pipeline_v0's scoring (``--edge-scoring v2``, default) before the
reference table is built.

Differences vs run_gemini_uoais_ref.py:
- --gt-path: run on ANY GT json (all cases taken as-is; no per-difficulty quota).
- Phase A (pointing) is deduplicated per image and submitted as ONE batch job;
  Phase B (reasoning) is one batch job over all cases.
- Batch jobs are asynchronous: the script polls until completion (target within
  24h, usually much faster). Job names are persisted in --out, so Ctrl-C and
  re-running resumes polling / skips finished phases instead of paying again.

Flow:  pointing batch -> labeled images + UOAIS (local GPU) -> reasoning batch
       -> per-case result.json -> predictions/eval/report (same as interactive).

Usage:
    python run_gemini_uoais_ref_batch.py \
        --gt-path UnoBench/subset_difficulty/train_calib_GT_450.json \
        --out logs/gemini_uoais_ref_train_calib_450
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import gemini_client as base
import gemini_uoais_lib as ref
import unobench_gemini_common as legacy
import uoais_obstruction as U
from run_gemini_uoais_ref import (
    ANN_DIR,
    IMAGE_DIR,
    NAME_FOR_ALL,
    ID_MAP,
    add_edge_scoring_args,
    load_json,
    make_case,
    old_scene_key_by_image_id,
    run_uoais_reference_for_image,
    score_reference_edges,
    validate_selected,
)

ROOT = Path(__file__).resolve().parent
TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
CREATE_MAX_ATTEMPTS = 8


# ── case selection ──────────────────────────────────────────────────────────────

def select_all_cases(gt_path: Path) -> tuple[list[dict], list[dict]]:
    rows = load_json(gt_path)
    names = load_json(NAME_FOR_ALL)
    id_to_scene_key = old_scene_key_by_image_id()
    cases, missing = [], []
    for row in rows:
        case, miss = make_case(row, names, id_to_scene_key)
        case["source_gt"] = str(gt_path)
        cases.append(case)
        if miss:
            missing.append(miss)
    return cases, missing


# ── batch-job plumbing ──────────────────────────────────────────────────────────

def submit_batch(client, jsonl_path: Path, display_name: str, model: str, out_root: Path, phase: str) -> str:
    """Upload + create the batch job, checkpointing after each step so a crash
    mid-flight never forces re-uploading the (potentially large) requests file
    or re-creating a duplicate job on resume."""
    from google.genai import types as gtypes

    upload_state_path = out_root / f"batch_{phase}_upload.json"
    if upload_state_path.exists():
        uploaded_name = load_json(upload_state_path)["file_name"]
        legacy.logger.info("[batch] %s: reusing already-uploaded file %s", phase, uploaded_name)
    else:
        legacy.logger.info("[batch] %s: uploading %s ...", phase, jsonl_path.name)
        uploaded = client.files.upload(
            file=str(jsonl_path),
            config=gtypes.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
        )
        uploaded_name = uploaded.name
        json.dump({"file_name": uploaded_name}, upload_state_path.open("w", encoding="utf-8"), indent=2)
        legacy.logger.info("[batch] %s: uploaded as %s", phase, uploaded_name)

    legacy.logger.info("[batch] %s: creating batch job ...", phase)
    # A job whose enqueued tokens exceed the account's batch quota is rejected
    # with 429 RESOURCE_EXHAUSTED. That is often transient (an earlier job still
    # draining), so back off and retry instead of throwing away the phase.
    for attempt in range(1, CREATE_MAX_ATTEMPTS + 1):
        try:
            job = client.batches.create(
                model=model,
                src=uploaded_name,
                config=gtypes.CreateBatchJobConfig(display_name=display_name),
            )
            break
        except Exception as exc:
            if attempt >= CREATE_MAX_ATTEMPTS or "RESOURCE_EXHAUSTED" not in str(exc):
                raise
            wait_s = min(600.0, 60.0 * attempt)
            legacy.logger.warning("[batch] %s: create attempt %d/%d hit quota — retrying in %.0fs",
                                  phase, attempt, CREATE_MAX_ATTEMPTS, wait_s)
            time.sleep(wait_s)
    state_path = out_root / f"batch_{phase}_job.json"
    json.dump({"job_name": job.name}, state_path.open("w", encoding="utf-8"), indent=2)
    legacy.logger.info("[batch] submitted %s as %s (%d requests)", display_name, job.name,
                       sum(1 for _ in jsonl_path.open()))
    return job.name


def poll_batch(client, job_name: str, poll_seconds: float) -> dict[str, str]:
    """Wait for the job, then return {key: response_text} (missing keys = failed rows)."""
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name if job.state else "UNKNOWN"
        if state in TERMINAL_STATES:
            break
        legacy.logger.info("[batch] %s state=%s — waiting %.0fs", job_name, state, poll_seconds)
        time.sleep(poll_seconds)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job {job_name} ended in {state}: {getattr(job, 'error', None)}")

    dest = getattr(job.dest, "file_name", None) if job.dest else None
    if not dest:
        raise RuntimeError(f"Batch job {job_name} succeeded but has no result file")

    max_attempts = 12
    for attempt in range(1, max_attempts + 1):
        try:
            payload = client.files.download(file=dest)
            break
        except Exception as exc:
            if attempt >= max_attempts:
                raise
            wait_s = min(120.0, 5.0 * attempt)
            legacy.logger.warning("[batch] download attempt %d/%d failed (%s) — retrying in %.0fs",
                                  attempt, max_attempts, exc, wait_s)
            time.sleep(wait_s)
    text = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else str(payload)

    out: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key") or row.get("custom_id")
        resp = row.get("response") or {}
        parts = []
        for cand in resp.get("candidates") or []:
            for part in (cand.get("content") or {}).get("parts") or []:
                if part.get("text"):
                    parts.append(part["text"])
        if key and parts:
            out[key] = "".join(parts).strip()
    return out


def request_line(key: str, parts: list[dict], temperature: float | None = None) -> str:
    req: dict = {"contents": [{"role": "user", "parts": parts}]}
    if temperature is not None:
        req["generationConfig"] = {"temperature": temperature}
    return json.dumps({"key": key, "request": req}, ensure_ascii=False)


def image_part(png_bytes: bytes) -> dict:
    return {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png_bytes).decode("ascii")}}


def split_requests(jsonl_path: Path, out_root: Path, phase: str, max_rows: int) -> list[Path]:
    """Split the requests file into <=max_rows chunks, one batch job each.

    A single job carrying every request can exceed the account's batch quota
    (429 RESOURCE_EXHAUSTED at create time) even though the same requests are
    accepted as several smaller jobs. The split is checkpointed so a resume
    reuses the existing parts instead of rewriting gigabytes.
    """
    manifest_path = out_root / f"batch_{phase}_parts.json"
    if manifest_path.exists():
        parts = [out_root / name for name in load_json(manifest_path)["parts"]]
        if all(p.exists() for p in parts):
            legacy.logger.info("[batch] %s: reusing %d existing request parts", phase, len(parts))
            return parts

    parts, handle, rows = [], None, 0
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if handle is None or rows >= max_rows:
                if handle is not None:
                    handle.close()
                part = out_root / f"batch_{phase}_requests.part{len(parts):03d}.jsonl"
                parts.append(part)
                handle = part.open("w", encoding="utf-8")
                rows = 0
            handle.write(line)
            rows += 1
    if handle is not None:
        handle.close()
    json.dump({"parts": [p.name for p in parts]}, manifest_path.open("w", encoding="utf-8"), indent=2)
    legacy.logger.info("[batch] %s: split into %d parts of <=%d requests", phase, len(parts), max_rows)
    return parts


def run_phase(client, out_root: Path, phase: str, jsonl_builder, model: str, poll_seconds: float,
              max_requests_per_job: int = 0) -> dict[str, str]:
    """Submit (or resume) one batch phase and return its parsed responses.

    ``max_requests_per_job`` > 0 splits the phase across several sequential jobs;
    0 keeps the whole phase in one job.
    """
    responses_path = out_root / f"batch_{phase}_responses.json"
    if responses_path.exists():
        return load_json(responses_path)

    jsonl_path = out_root / f"batch_{phase}_requests.jsonl"
    requests_state_path = out_root / f"batch_{phase}_requests.done"
    single_job_state = out_root / f"batch_{phase}_job.json"
    if single_job_state.exists():
        # Phase was submitted as one job before any chunking was requested.
        job_name = load_json(single_job_state)["job_name"]
        legacy.logger.info("[batch] resuming %s job %s", phase, job_name)
        responses = poll_batch(client, job_name, poll_seconds)
        json.dump(responses, responses_path.open("w", encoding="utf-8"), indent=2)
        return responses

    if not requests_state_path.exists():
        legacy.logger.info("[batch] %s: building request file ...", phase)
        n = jsonl_builder(jsonl_path)
        requests_state_path.write_text(str(n), encoding="utf-8")
        legacy.logger.info("[batch] %s: request file built (%d rows)", phase, n)
    else:
        n = int(requests_state_path.read_text(encoding="utf-8").strip() or 0)
        legacy.logger.info("[batch] %s: reusing already-built request file (%d rows)", phase, n)
    if n == 0:
        json.dump({}, responses_path.open("w", encoding="utf-8"))
        return {}

    if max_requests_per_job and n > max_requests_per_job:
        parts = split_requests(jsonl_path, out_root, phase, max_requests_per_job)
    else:
        parts = [jsonl_path]

    responses: dict[str, str] = {}
    for k, part in enumerate(parts):
        sub_phase = phase if len(parts) == 1 else f"{phase}_p{k:03d}"
        part_responses_path = out_root / f"batch_{sub_phase}_responses.json"
        if part_responses_path.exists():
            legacy.logger.info("[batch] %s: part %d/%d already collected", phase, k + 1, len(parts))
            responses.update(load_json(part_responses_path))
            continue
        part_job_state = out_root / f"batch_{sub_phase}_job.json"
        if part_job_state.exists():
            job_name = load_json(part_job_state)["job_name"]
            legacy.logger.info("[batch] resuming %s job %s", sub_phase, job_name)
        else:
            job_name = submit_batch(client, part, f"{out_root.name}_{sub_phase}", model, out_root, sub_phase)
        legacy.logger.info("[batch] %s: polling part %d/%d (%s) ...", phase, k + 1, len(parts), job_name)
        part_responses = poll_batch(client, job_name, poll_seconds)
        json.dump(part_responses, part_responses_path.open("w", encoding="utf-8"), indent=2)
        responses.update(part_responses)

    json.dump(responses, responses_path.open("w", encoding="utf-8"), indent=2)
    return responses


# ── phase A: pointing (deduplicated per image) ──────────────────────────────────

def parse_pointing_text(text: str, image_size: tuple[int, int]) -> list[dict]:
    """Same parsing as gemini_client.gemini_pointing, applied to batch output."""
    import re
    text = re.sub(r"^```[a-z]*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```$", "", text.strip())
    W, H = image_size
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
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
    return objects


def phase_pointing(client, cases: list[dict], out_root: Path, args) -> dict[int, list[dict]]:
    det_dir = out_root / "det_by_image"
    det_dir.mkdir(exist_ok=True)
    image_ids = sorted({int(c["image_id"]) for c in cases})
    pending = [i for i in image_ids if not (det_dir / f"image_{i:06d}.json").exists()]

    def build(jsonl_path: Path) -> int:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for image_id in pending:
                png = (IMAGE_DIR / f"image_{image_id:06d}.png").read_bytes()
                f.write(request_line(
                    f"img_{image_id:06d}",
                    [image_part(png), {"text": base._POINTING_PROMPT}],
                ) + "\n")
        return len(pending)

    responses = run_phase(client, out_root, "pointing", build, args.model, args.poll_seconds,
                          args.max_requests_per_job)
    parse_failures = []
    for image_id in pending:
        text = responses.get(f"img_{image_id:06d}")
        if text is None:
            continue
        try:
            with PILImage.open(IMAGE_DIR / f"image_{image_id:06d}.png") as im:
                objects = parse_pointing_text(text, im.size)
        except Exception as exc:
            legacy.logger.warning("[batch] pointing: failed to parse response for image %06d (%s)", image_id, exc)
            parse_failures.append({"image_id": image_id, "error": str(exc)})
            objects = []
        json.dump(objects, (det_dir / f"image_{image_id:06d}.json").open("w", encoding="utf-8"), indent=2)
    if parse_failures:
        json.dump(parse_failures, (out_root / "pointing_parse_failures.json").open("w", encoding="utf-8"), indent=2)
        legacy.logger.info("[batch] pointing: %d/%d images failed to parse", len(parse_failures), len(pending))

    detections = {}
    for image_id in image_ids:
        path = det_dir / f"image_{image_id:06d}.json"
        if path.exists():
            objects = load_json(path)
            if objects:
                detections[image_id] = objects
    return detections


# ── phase B: reasoning ──────────────────────────────────────────────────────────

def prepare_case_inputs(case: dict, objects: list[dict], out_root: Path, predictor, cfg, args) -> dict:
    """Labeled image + UOAIS reference lines for one case (local, cached)."""
    case_dir = out_root / case["case_name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    json.dump(objects, (case_dir / "det_objects.json").open("w", encoding="utf-8"), indent=2)

    image_np = np.array(PILImage.open(IMAGE_DIR / f"image_{case['image_id']:06d}.png").convert("RGB"))
    labeled_b64 = base.draw_labeled_image(image_np, objects, case_dir / "labeled.png")

    uoais_ref = run_uoais_reference_for_image(
        case["image_id"], objects, predictor, cfg, args, out_root / "uoais_cache")
    scored_edges = score_reference_edges(uoais_ref.get("edges_all", []), args)
    ref_edges = ref.build_uoais_reference_edges(
        scored_edges,
        target_id=int(case["query_object"]["obj_id"]),
        max_edges=args.max_ref_edges,
        min_contact_px=args.min_ref_contact_px,
    )
    ref_lines = ref.format_uoais_reference_lines(ref_edges)
    user_prompt = ref._USER_PROMPT_TMPL.format(
        obj_list=ref.object_list(objects),
        query_name=case["query_object"]["object_name"],
        uoais_reference=ref_lines,
    )
    return {
        "labeled_b64": labeled_b64,
        "uoais_ref": uoais_ref,
        "ref_edges": ref_edges,
        "ref_lines": ref_lines,
        "user_prompt": user_prompt,
        "edge_scoring": args.edge_scoring,
        "n_edges_all": len(uoais_ref.get("edges_all", [])),
        "n_edges_scored": len(scored_edges),
    }


def assemble_result(case: dict, objects: list[dict], inputs: dict, combined: dict, out_root: Path) -> dict:
    """Mirror of run_gemini_uoais_ref.process_case result assembly."""
    id_to_point = {int(o["id"]): {"label": o["label"], "x": int(o["x_px"]), "y": int(o["y_px"])} for o in objects}
    top_points = combined.get("top_points", [])
    top_points_with_ids = combined.get("top_points_with_ids", [])
    ev = ref.eval_result(case, top_points_with_ids)
    ann_mask = ref._annotation_mask(int(case["image_id"]))
    target_mapping = ref.map_badge_point_to_gt(ann_mask, combined.get("target_id"), id_to_point)
    target_gt_id = int(target_mapping.get("gt_id") or 0)
    query_obj_id = int(case["query_object"]["obj_id"])
    uoais_ref = inputs["uoais_ref"]

    return {
        "case": case["case_name"],
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
        "api_mode": "batch",
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
            "edge_scoring": inputs.get("edge_scoring"),
            "n_edges_all": inputs.get("n_edges_all"),
            "n_edges_scored": inputs.get("n_edges_scored"),
        },
        "uoais_reference_edges": inputs["ref_edges"],
        "uoais_reference_prompt_lines": inputs["ref_lines"].splitlines(),
        "uoais_supported_chosen_edges": ref.supported_edges(combined.get("occlusion_chain", []), inputs["ref_edges"]),
        "prompt_variant": ref.PROMPT_VARIANT,
        "prompt": ref._SYSTEM_PROMPT + "\n\n" + inputs["user_prompt"],
    }


# ── main ────────────────────────────────────────────────────────────────────────

def run(args):
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    gt_path = Path(args.gt_path)
    if not gt_path.is_absolute():
        gt_path = ROOT / gt_path

    cases, missing_names = select_all_cases(gt_path)
    counts = Counter(c["difficulty"] for c in cases)
    json.dump({
        "source": str(gt_path),
        "mode": "batch_api",
        "counts": dict(counts),
        "n": len(cases),
        "cases": [{"case": c["case_name"], "image_id": c["image_id"], "query_object": c["query_object"], "difficulty": c["difficulty"]} for c in cases],
    }, (out_root / "selection_manifest.json").open("w", encoding="utf-8"), indent=2)
    json.dump(missing_names, (out_root / "missing_names.json").open("w", encoding="utf-8"), indent=2)
    legacy.logger.info("cases=%d images=%d difficulty=%s", len(cases), len({c["image_id"] for c in cases}), dict(counts))

    legacy.ensure_images(cases, legacy.IMG_ZIP, IMAGE_DIR)
    legacy.ensure_depth(cases, legacy.DEPTH_DIR, legacy.META_ZIP, legacy.ID_MAP)
    legacy.ensure_annotations(cases, legacy.ANN_ZIP, ANN_DIR)
    missing_files = validate_selected(cases)
    if missing_files:
        json.dump(missing_files, (out_root / "missing_files.json").open("w", encoding="utf-8"), indent=2)
        raise FileNotFoundError(f"Missing {len(missing_files)} required files; see missing_files.json")
    if args.dry_run:
        print(json.dumps({"cases": len(cases), "counts": dict(counts)}, indent=2))
        return

    client = base._get_client()

    # Phase A — pointing batch (one request per unique image).
    detections = phase_pointing(client, cases, out_root, args)
    legacy.logger.info("pointing done: %d/%d images detected", len(detections), len({c['image_id'] for c in cases}))

    # Local phase — labeled images + UOAIS reference (GPU).
    U.DEPTH_TOLERANCE_MM = args.depth_tolerance_mm
    U.CONTACT_DILATE_PX = args.contact_dilate_px
    U.MIN_CONTACT_PIXELS = args.min_contact_pixels
    U.MIN_VALID_DEPTH_RATIO = args.min_valid_depth_ratio
    predictor, cfg = U.load_uoais_predictor(U.CFG_RGBD, args.score_thresh, args.nms_thresh, args.device)

    failed = []
    case_inputs: dict[str, dict] = {}
    pending_cases = []
    for case in cases:
        result_path = out_root / case["case_name"] / "result.json"
        if result_path.exists():
            continue
        objects = detections.get(int(case["image_id"]))
        if not objects:
            failed.append({"case": case["case_name"], "image_id": case["image_id"], "error": "no_objects_detected"})
            continue
        try:
            case_inputs[case["case_name"]] = prepare_case_inputs(case, objects, out_root, predictor, cfg, args)
            pending_cases.append(case)
        except Exception as exc:
            failed.append({"case": case["case_name"], "image_id": case["image_id"], "error": str(exc)})
            legacy.logger.exception("prepare failed for %s", case["case_name"])

    # Phase B — reasoning batch (one request per case).
    def build_reasoning(jsonl_path: Path) -> int:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for case in pending_cases:
                inp = case_inputs[case["case_name"]]
                f.write(request_line(
                    case["case_name"],
                    [
                        {"text": ref._SYSTEM_PROMPT},
                        {"text": inp["user_prompt"]},
                        image_part(base64.b64decode(inp["labeled_b64"])),
                    ],
                    temperature=0.0,
                ) + "\n")
        return len(pending_cases)

    responses = run_phase(client, out_root, "reasoning", build_reasoning, args.model, args.poll_seconds,
                          args.max_requests_per_job)

    for case in pending_cases:
        raw = responses.get(case["case_name"])
        objects = detections[int(case["image_id"])]
        if raw is None:
            failed.append({"case": case["case_name"], "image_id": case["image_id"], "error": "batch_response_missing"})
            continue
        parsed = ref.parse_reasoning_json(raw, objects)
        combined = {**parsed, "raw_outputs": [raw]}
        result = assemble_result(case, objects, case_inputs[case["case_name"]], combined, out_root)
        json.dump(result, (out_root / case["case_name"] / "result.json").open("w", encoding="utf-8"), indent=2)

    json.dump(failed, (out_root / "failed_cases.json").open("w", encoding="utf-8"), indent=2)

    rows = []
    for case in cases:
        result_path = out_root / case["case_name"] / "result.json"
        if result_path.exists():
            rows.append(load_json(result_path))
    gt_records = [
        {
            "image_id": int(case["image_id"]),
            "query_object": int(case["query_object"]["obj_id"]),
            "occlusion_paths": case.get("occlusion_paths", []),
            "top_objects": [int(t["obj_id"]) for t in case["target_objects"]],
            "new_difficulty": case["difficulty"],
        }
        for case in cases
    ]
    ref.write_predictions_and_eval(rows, gt_records, out_root)
    summary = legacy.summarize(rows, out_root)
    print(json.dumps(summary, indent=2))
    print(f"done: {len(rows)}/{len(cases)} cases, {len(failed)} failed — outputs in {out_root}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="logs/gemini_uoais_ref_batch")
    ap.add_argument("--gt-path", default="UnoBench/gt_for_nlp.json")
    ap.add_argument("--model", default=base.MODEL, help="Batch model id (must support the Batch API)")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--max-requests-per-job", type=int, default=0,
                    help="Split each phase into sequential batch jobs of at most this many "
                         "requests (0 = one job per phase). Use it when a full-size job is "
                         "rejected with 429 RESOURCE_EXHAUSTED at create time.")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-ref-edges", type=int, default=40)
    ap.add_argument("--min-ref-contact-px", type=int, default=1)
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
