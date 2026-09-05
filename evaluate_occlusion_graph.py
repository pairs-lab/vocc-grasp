#!/usr/bin/env python3
"""
CLI: score a predicted occlusion graph (from any of the three pipeline log
schemas in this repo) against `occlusion_paths` ground truth, computing
per-case and dataset-level precision/recall/F1 (see graph_eval_lib.py for the
edge-direction convention, id-mapping adapters, and scoring formula).

Usage:
  python evaluate_occlusion_graph.py \
    --pred-log logs/gemini_uoais_ref_train_calib_450 \
    --gt-path UnoBench/gt_for_nlp.json \
    --out logs/gemini_uoais_ref_train_calib_450/graph_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import graph_eval_lib as g

ROOT = Path(__file__).resolve().parent


def detect_schema(pred_log: Path) -> str:
    results_jsonl = pred_log / "results.jsonl"
    if results_jsonl.exists():
        rows = g.read_jsonl(results_jsonl)
        if rows and "candidate_edges" in rows[0]:
            return "uoais_pipeline"

    summary_path = pred_log / "summary.json"
    if summary_path.exists():
        data = g.load_json(summary_path)
        per_case = data.get("per_case") if isinstance(data, dict) else None
        if per_case and "occlusion_chain" in per_case[0] and "id_to_point" in per_case[0]:
            return "gemini_uoais_ref"

    results_json = pred_log / "results.json"
    if results_json.exists():
        data = g.load_json(results_json)
        if isinstance(data, list) and data:
            if "candidate_edges" in data[0]:
                return "uoais_pipeline_list"
            if "model_output" in data[0]:
                return "model_output_text"

    raise ValueError(
        f"Could not auto-detect schema for {pred_log}; pass --schema explicitly "
        "(gemini_uoais_ref, uoais_pipeline, model_output_text)."
    )


def iter_gemini_uoais_ref(pred_log: Path):
    data = g.load_json(pred_log / "summary.json")
    for row in data.get("per_case", []):
        image_id = g._to_int(row.get("image_id"))
        query_object = g._to_int(row.get("query_mask_id"))
        yield image_id, query_object, row, row.get("case")


def iter_uoais_pipeline(pred_log: Path):
    path = pred_log / "results.jsonl"
    rows = g.read_jsonl(path) if path.exists() else g.load_json(pred_log / "results.json")
    for row in rows:
        image_id = g._to_int(row.get("image_id"))
        query_object = g._to_int(row.get("query_object"))
        case_name = f"img{image_id:06d}_q{query_object}" if image_id is not None and query_object is not None else None
        yield image_id, query_object, row, case_name


def iter_model_output_text(pred_log: Path):
    rows = g.load_json(pred_log / "results.json")
    for row in rows:
        image_id = g._to_int(row.get("image_id"))
        query_object = g._to_int(row.get("query_object"))
        case_name = f"img{image_id:06d}_q{query_object}" if image_id is not None and query_object is not None else None
        yield image_id, query_object, row, case_name


SCHEMA_ITERATORS = {
    "gemini_uoais_ref": iter_gemini_uoais_ref,
    "uoais_pipeline": iter_uoais_pipeline,
    "uoais_pipeline_list": iter_uoais_pipeline,
    "model_output_text": iter_model_output_text,
}


def score_row(schema: str, row: dict, mask) -> g.AdapterResult:
    if schema == "gemini_uoais_ref":
        return g.edges_from_gemini_uoais_chain(row.get("occlusion_chain") or [], row.get("id_to_point") or {}, mask)
    if schema in ("uoais_pipeline", "uoais_pipeline_list"):
        return g.edges_from_uoais_pipeline_candidates(row.get("candidate_edges") or [], mask)
    if schema == "model_output_text":
        return g.edges_from_model_output_text(row.get("model_output") or "", mask)
    raise ValueError(f"Unknown schema: {schema}")


def run(pred_log: Path, gt_path: Path, annotations_dir: Path, annotations_zip: Path,
        schema: str | None) -> dict:
    resolved_schema = schema or detect_schema(pred_log)
    gt = g.load_gt(gt_path)
    mask_loader = g.MaskLoader(annotations_dir, annotations_zip)

    iterator = SCHEMA_ITERATORS[resolved_schema]
    rows = []
    n_skipped_no_gt = 0
    for image_id, query_object, pred_row, case_name in iterator(pred_log):
        if image_id is None or query_object is None:
            n_skipped_no_gt += 1
            continue
        gt_row = gt.get((image_id, query_object))
        if gt_row is None:
            n_skipped_no_gt += 1
            continue
        mask = mask_loader(image_id)
        adapter_result = score_row(resolved_schema, pred_row, mask)
        scored = g.score_case(adapter_result.edges, adapter_result.n_invalid, gt_row["edges"])
        scored.update({
            "case": case_name,
            "image_id": image_id,
            "query_object": query_object,
            "difficulty": gt_row["difficulty"],
            "new_difficulty": gt_row["new_difficulty"],
            "path_depth_bucket": gt_row["path_depth_bucket"],
            "mapping_details": adapter_result.mapping_details,
        })
        rows.append(scored)

    aggregate = g.aggregate(rows)
    return {
        "schema": resolved_schema,
        "pred_log": str(pred_log),
        "gt_path": str(gt_path),
        "n_scored": len(rows),
        "n_skipped_no_gt_match": n_skipped_no_gt,
        "aggregate": aggregate,
        "rows": rows,
    }


def print_summary(result: dict) -> None:
    agg = result["aggregate"]
    print(f"schema={result['schema']}  pred_log={result['pred_log']}")
    print(f"scored={result['n_scored']}  skipped(no GT match)={result['n_skipped_no_gt_match']}\n")

    overall = agg["overall"]
    print("=== Overall ===")
    print(f"macro: P={overall['macro'].get('precision')} R={overall['macro'].get('recall')} "
          f"F1={overall['macro'].get('f1')} exact_rate={overall['macro'].get('exact_rate')} (n={overall['macro'].get('n')})")
    print(f"micro: P={overall['micro'].get('precision')} R={overall['micro'].get('recall')} "
          f"F1={overall['micro'].get('f1')} (tp={overall['micro'].get('tp')} fp={overall['micro'].get('fp')} fn={overall['micro'].get('fn')})\n")

    print("=== By difficulty ===")
    for key, val in agg["by_difficulty"].items():
        m = val["macro"]
        print(f"{key:<10} macro P={m.get('precision')} R={m.get('recall')} F1={m.get('f1')} (n={m.get('n')})")

    print("\n=== By path depth ===")
    for key, val in agg["by_path_depth"].items():
        m = val["macro"]
        print(f"{key:<16} macro P={m.get('precision')} R={m.get('recall')} F1={m.get('f1')} (n={m.get('n')})")

    print("\n=== Mapping diagnostics ===")
    print(json.dumps(agg["mapping_diagnostics"], indent=2, ensure_ascii=False))


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-log", required=True, type=Path)
    ap.add_argument("--gt-path", default=str(ROOT / "UnoBench/gt_for_nlp.json"), type=Path)
    ap.add_argument("--annotations-dir", default=str(g.DEFAULT_ANNOTATIONS_DIR), type=Path)
    ap.add_argument("--annotations-zip", default=str(g.DEFAULT_ANNOTATIONS_ZIP), type=Path)
    ap.add_argument("--schema", choices=["gemini_uoais_ref", "uoais_pipeline", "model_output_text"], default=None)
    ap.add_argument("--out", default=None, type=Path)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args.pred_log, args.gt_path, args.annotations_dir, args.annotations_zip, args.schema)
    print_summary(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
