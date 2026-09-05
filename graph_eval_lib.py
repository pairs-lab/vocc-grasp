#!/usr/bin/env python3
"""
Graph-level precision/recall/F1 evaluation of predicted occlusion graphs
against `occlusion_paths` ground truth (UnoBench/gt_for_nlp.json and every
file under UnoBench/subset_difficulty/ share this schema).

GT convention (confirmed against UnoBench/uoais_graph.py's GT builder and
evaluate_nlp.py's paths_to_triplets): for a path `p`, `p[i]` OCCLUDES `p[i+1]`
(`p[0]` is a top_objects member, `p[-1]` is query_object). Object ids in these
GT files are already GT-annotation-mask space (pixel value in
UnoBench/annotations/image_XXXXXX.npy == object id).

Three pipelines produce predicted graphs, each in a different id space:
  - run_gemini_uoais_ref.py: `occlusion_chain` edges in DETECTOR BADGE space,
    needs `id_to_point` + GT mask to resolve to GT ids.
  - run_uoais_pipeline.py: `candidate_edges` edges already in GT-mask space
    (mapped upstream via Hungarian assignment before being written out).
  - older runs with only a `model_output` <think> string: no ids at all, must
    regex-parse "A at (x,y) is occluded by B at (x,y)" and resolve pixels.

Edges are always represented with NAMED fields (occluder_id, occluded_id),
never a bare positional tuple: existing code in this repo disagrees on tuple
order (augment_gemini_uoais_training.gt_edge_set stores (occluded, occluder);
evaluate_nlp.paths_to_triplets stores (occluder, occluded)) and silently
reusing both would invert edge matching.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np

import gemini_uoais_lib as ref

ROOT = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS_DIR = ROOT / "UnoBench/annotations"
DEFAULT_ANNOTATIONS_ZIP = ROOT / "UnoBench/annotations.zip"

THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
THINK_EDGE_RE = re.compile(
    r"(.+?)\s+at\s+\(([\d.]+),\s*([\d.]+)\)\s+is occluded by\s+(.+?)\s+at\s+\(([\d.]+),\s*([\d.]+)\)",
    re.IGNORECASE,
)


class Edge(NamedTuple):
    occluder_id: int
    occluded_id: int


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def gt_edges_from_paths(occlusion_paths: list[list[int]]) -> set[Edge]:
    """path[i] occludes path[i+1]; union (deduped) across all paths."""
    edges: set[Edge] = set()
    for path in occlusion_paths or []:
        ids = [int(x) for x in path]
        for i in range(len(ids) - 1):
            occluder, occluded = ids[i], ids[i + 1]
            if occluder != occluded:
                edges.add(Edge(occluder_id=occluder, occluded_id=occluded))
    return edges


def path_depth_bucket(k_min) -> str:
    k = _to_int(k_min)
    if k is None:
        return "unknown"
    if k <= 0:
        return "path_depth_0"
    if k == 1:
        return "path_depth_1"
    if k == 2:
        return "path_depth_2"
    return "path_depth_ge3"


def load_gt(gt_path: Path) -> dict[tuple[int, int], dict]:
    """Load any occlusion_paths-shaped GT file, keyed by (image_id, query_object)."""
    rows = load_json(gt_path)
    out: dict[tuple[int, int], dict] = {}
    for row in rows:
        image_id = _to_int(row.get("image_id"))
        query_object = _to_int(row.get("query_object"))
        if image_id is None or query_object is None:
            continue
        paths = row.get("occlusion_paths") or [[query_object]]
        out[(image_id, query_object)] = {
            "image_id": image_id,
            "query_object": query_object,
            "occlusion_paths": paths,
            "edges": gt_edges_from_paths(paths),
            "top_objects": row.get("top_objects", [p[0] for p in paths]),
            "difficulty": row.get("difficulty"),
            "new_difficulty": row.get("new_difficulty"),
            "k_min": row.get("k_min"),
            "num_paths": row.get("num_paths"),
            "path_depth_bucket": path_depth_bucket(row.get("k_min")),
        }
    return out


# ---------------------------------------------------------------------------
# Annotation mask loading (disk, else zip fallback)
# ---------------------------------------------------------------------------

class MaskLoader:
    def __init__(self, annotations_dir: Path = DEFAULT_ANNOTATIONS_DIR,
                 annotations_zip: Path = DEFAULT_ANNOTATIONS_ZIP):
        self.annotations_dir = Path(annotations_dir)
        self.annotations_zip = Path(annotations_zip)
        self._cache: dict[int, np.ndarray | None] = {}

    def __call__(self, image_id: int) -> np.ndarray | None:
        image_id = int(image_id)
        if image_id in self._cache:
            return self._cache[image_id]
        name = f"image_{image_id:06d}.npy"
        path = self.annotations_dir / name
        mask = None
        if path.exists():
            mask = np.load(path).astype(int)
        elif self.annotations_zip.exists():
            try:
                with zipfile.ZipFile(self.annotations_zip) as archive:
                    with archive.open(name) as f:
                        mask = np.load(f).astype(int)
            except (FileNotFoundError, KeyError):
                mask = None
        self._cache[image_id] = mask
        return mask


# ---------------------------------------------------------------------------
# Predicted-edge adapters
# ---------------------------------------------------------------------------

class AdapterResult(NamedTuple):
    edges: set[Edge]
    n_invalid: int
    mapping_details: list[dict]


def edges_from_gemini_uoais_chain(occlusion_chain: list[dict], id_to_point: dict, mask) -> AdapterResult:
    """Pipeline 1 (run_gemini_uoais_ref.py): occlusion_chain edges are in
    detector BADGE space (occluded_id/occluder_id); resolve each endpoint via
    gemini_uoais_lib.map_badge_point_to_gt (exact pixel, else radius fallback).
    """
    points = {int(k): v for k, v in (id_to_point or {}).items()}
    edges: set[Edge] = set()
    n_invalid = 0
    details = []
    for edge in occlusion_chain or []:
        occluded_badge = _to_int(edge.get("occluded_id"))
        occluder_badge = _to_int(edge.get("occluder_id"))
        occluded_map = ref.map_badge_point_to_gt(mask, occluded_badge, points) if occluded_badge is not None else {"gt_id": 0, "method": "missing_badge_id"}
        occluder_map = ref.map_badge_point_to_gt(mask, occluder_badge, points) if occluder_badge is not None else {"gt_id": 0, "method": "missing_badge_id"}
        occluded_gt = int(occluded_map.get("gt_id") or 0)
        occluder_gt = int(occluder_map.get("gt_id") or 0)
        valid = occluded_gt > 0 and occluder_gt > 0 and occluded_gt != occluder_gt
        details.append({
            "occluded_badge_id": occluded_badge,
            "occluder_badge_id": occluder_badge,
            "occluded_gt_id": occluded_gt,
            "occluder_gt_id": occluder_gt,
            "occluded_method": occluded_map.get("method"),
            "occluder_method": occluder_map.get("method"),
            "valid": valid,
        })
        if valid:
            edges.add(Edge(occluder_id=occluder_gt, occluded_id=occluded_gt))
        else:
            n_invalid += 1
    return AdapterResult(edges=edges, n_invalid=n_invalid, mapping_details=details)


def edges_from_uoais_pipeline_candidates(candidate_edges: list[dict], mask=None) -> AdapterResult:
    """Pipeline 2 (run_uoais_pipeline.py): candidate_edges are already in
    GT-mask space (from=occluder, to=occluded), mapped upstream via Hungarian
    assignment. Still sanity-check each id against the mask's actual label
    set when available, since upstream target_mapping_status can be
    'merged'/'nearest' rather than 'exact'.
    """
    valid_labels = None
    if mask is not None:
        valid_labels = set(int(v) for v in np.unique(mask) if v > 0)
    edges: set[Edge] = set()
    n_invalid = 0
    details = []
    for edge in candidate_edges or []:
        occluder = _to_int(edge.get("from"))
        occluded = _to_int(edge.get("to"))
        ok = occluder is not None and occluded is not None and occluder != occluded
        if ok and valid_labels is not None:
            ok = occluder in valid_labels and occluded in valid_labels
        details.append({
            "occluder_gt_id": occluder,
            "occluded_gt_id": occluded,
            "valid": ok,
        })
        if ok:
            edges.add(Edge(occluder_id=occluder, occluded_id=occluded))
        else:
            n_invalid += 1
    return AdapterResult(edges=edges, n_invalid=n_invalid, mapping_details=details)


def _reconstruct_paths_from_think(think_text: str) -> list[list[tuple[float, float, str]]]:
    """Mirror evaluate_nlp.py's parse_think_paths_with_coords "concatenate +
    reverse" logic, but keep raw (x, y, name) tuples instead of resolving ids
    inline, so the caller can plug in a different point->id resolver.
    """
    parts = re.split(r"Path\d*:", think_text)
    paths = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        rels = THINK_EDGE_RE.findall(part)
        if rels:
            coords = []
            for child_name, x1, y1, parent_name, x2, y2 in rels:
                coords.append((float(x1), float(y1), child_name.strip()))
                coords.append((float(x2), float(y2), parent_name.strip()))
            if len(coords) >= 2:
                chain = [coords[0], coords[1]] + coords[3::2]
                chain.reverse()  # reverse direction: occluder -> occluded
                # de-dup consecutive duplicates while preserving order
                deduped = []
                for pt in chain:
                    if not deduped or deduped[-1] != pt:
                        deduped.append(pt)
                paths.append(deduped)
        else:
            m_single = re.search(r"at\s+\(([\d.]+),\s*([\d.]+)\)", part)
            if m_single:
                x, y = float(m_single.group(1)), float(m_single.group(2))
                paths.append([(x, y, "")])
    return paths


def edges_from_model_output_text(model_output: str, mask, radii_px=ref.MAPPING_RADII_PX) -> AdapterResult:
    """Pipeline 3 (older runs with no structured ids): regex-parse the
    <think> block for "A at (x,y) is occluded by B at (x,y)" relations (same
    pattern/reconstruction as evaluate_nlp.py's parse_think_paths_with_coords),
    then resolve each (x, y) via gemini_uoais_lib._map_point_to_gt (radius
    fallback, more lenient than evaluate_nlp.py's exact-pixel-only lookup).
    """
    m = THINK_RE.search(model_output or "")
    edges: set[Edge] = set()
    n_invalid = 0
    details = []
    if not m:
        return AdapterResult(edges=edges, n_invalid=0, mapping_details=details)
    for path in _reconstruct_paths_from_think(m.group(1).strip()):
        resolved = []
        for x, y, name in path:
            mapped = ref._map_point_to_gt(mask, x, y, radii_px=radii_px)
            gt_id = int(mapped.get("gt_id") or 0)
            resolved.append(gt_id)
            details.append({"x": x, "y": y, "name": name, "gt_id": gt_id, "method": mapped.get("method")})
        for i in range(len(resolved) - 1):
            occluder, occluded = resolved[i], resolved[i + 1]
            valid = occluder > 0 and occluded > 0 and occluder != occluded
            if valid:
                edges.add(Edge(occluder_id=occluder, occluded_id=occluded))
            else:
                n_invalid += 1
    return AdapterResult(edges=edges, n_invalid=n_invalid, mapping_details=details)


# ---------------------------------------------------------------------------
# Scoring (formula lifted from augment_gemini_uoais_training.chain_metrics)
# ---------------------------------------------------------------------------

def score_case(pred_edges: set[Edge], n_invalid: int, gt_edges: set[Edge]) -> dict:
    tp_edges = pred_edges & gt_edges
    tp = len(tp_edges)
    fp = len(pred_edges - gt_edges) + n_invalid
    fn = len(gt_edges - pred_edges)
    predicted_denominator = len(pred_edges) + n_invalid
    precision = tp / predicted_denominator if predicted_denominator else (1.0 if not gt_edges else 0.0)
    recall = tp / len(gt_edges) if gt_edges else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "predicted_edges": [e._asdict() for e in sorted(pred_edges)],
        "gt_edges": [e._asdict() for e in sorted(gt_edges)],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_invalid_predicted_edges": n_invalid,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "exact": n_invalid == 0 and pred_edges == gt_edges,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _macro(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "precision": round(sum(r["precision"] for r in rows) / len(rows), 6),
        "recall": round(sum(r["recall"] for r in rows) / len(rows), 6),
        "f1": round(sum(r["f1"] for r in rows) / len(rows), 6),
        "exact_rate": round(sum(bool(r["exact"]) for r in rows) / len(rows), 6),
    }


def _micro(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if not fn else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": len(rows),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _mapping_diagnostics(rows: list[dict]) -> dict:
    method_counts = Counter()
    n_invalid_total = 0
    for row in rows:
        n_invalid_total += row.get("n_invalid_predicted_edges", 0)
        for detail in row.get("mapping_details") or []:
            for key in ("method", "occluded_method", "occluder_method"):
                if key in detail and detail[key] is not None:
                    method_counts[detail[key]] += 1
    return {
        "n_invalid_predicted_edges_total": n_invalid_total,
        "mapping_method_counts": dict(method_counts),
    }


def aggregate(rows: list[dict]) -> dict:
    """rows: list of per-case dicts, each the output of score_case() merged
    with 'difficulty'/'new_difficulty'/'path_depth_bucket'/'mapping_details'.
    """
    by_difficulty = defaultdict(list)
    by_new_difficulty = defaultdict(list)
    by_path_depth = defaultdict(list)
    for row in rows:
        by_difficulty[str(row.get("difficulty"))].append(row)
        by_new_difficulty[str(row.get("new_difficulty"))].append(row)
        by_path_depth[str(row.get("path_depth_bucket"))].append(row)

    return {
        "overall": {"macro": _macro(rows), "micro": _micro(rows)},
        "by_difficulty": {k: {"macro": _macro(v), "micro": _micro(v)} for k, v in sorted(by_difficulty.items())},
        "by_new_difficulty": {k: {"macro": _macro(v), "micro": _micro(v)} for k, v in sorted(by_new_difficulty.items())},
        "by_path_depth": {k: {"macro": _macro(v), "micro": _micro(v)} for k, v in sorted(by_path_depth.items())},
        "mapping_diagnostics": _mapping_diagnostics(rows),
    }
