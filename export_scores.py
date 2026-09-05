#!/usr/bin/env python3
"""
Export each system's per-edge and per-object confidences as CSV, labelled from the GT.

    python export_scores.py                     # writes logs/final_reports/scores/

Formats (one row per assertion the system made; nothing is deduplicated):

    sample_id,edge,edge_confidence,label,difficulty_level      6,3->2,0.9500000000,1,med
    sample_id,object,object_confidence,label,difficulty_level  6,3,0.8200000000,1,med

`1->2` means object 1 sits on top of / occludes object 2, i.e. from=occluder,
to=occluded. `sample_id` indexes UnoBench/subset_difficulty/test_GT_small_1800.json.

Labels, both derived from that GT file's `occlusion_paths`:

    edge    1 if (occluder, occluded) is a GT occlusion edge of that sample.
    object  1 if the object sits at a free position of the sample's GT occlusion
            paths: `graph_ops.free_set(X, e*)` for any object other than the query,
            `graph_ops.target_is_free(X, e*)` for the query itself. That is exactly
            what every system's object score predicts - the VLMs emit `prob_free`
            ("probability the object is currently free") and the math stack the
            free-set marginal q_o - and it is the same rule
            the object-level calibration scores against, so the numbers here
            reproduce it.

Score sources
-------------
    UOAIS 3D        results.json[].candidate_edges[].prob, already in GT id space.
    VLM logs        summary.json per_case[].occlusion_chain[].edge_confidence and
                    [].candidates[].prob_free, both in detector-badge id space and
                    mapped to GT ids through the annotation mask.
    math stack      the calibrated fused edge scores fed INTO the stack (the prior
                    shift only moves the MAP cut-point, so the shifted posteriors
                    would measure the cut-point rather than the model), plus the
                    free-set marginals q_o the stack itself produces.

D3G stores `candidate_edges` without any score and UnoGrasp only stores rendered
text, so neither produces a CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import graph_eval_lib as G           # noqa: E402
import gemini_uoais_lib as ref       # noqa: E402

sys.path.insert(0, str(ROOT / "math"))
from freeset.inference import graph_ops as OPS  # noqa: E402  (free_set / target_is_free)

LEVEL = {"Easy": "easy", "Medium": "med", "Hard": "hard", "No-Occ": "no-occ"}
EDGE_FIELDS = ["sample_id", "edge", "edge_confidence", "label", "difficulty_level"]
OBJ_FIELDS = ["sample_id", "object", "object_confidence", "label", "difficulty_level"]

UOAIS_LOGS = {"uoais_pipeline_1800": "UOAIS 3D (filtered edges)",
              "uoais_1800_legacy": "UOAIS 3D (unfiltered edges, legacy)"}
VLM_LOGS = {
    "gemini_noref_test_1800": "Gemini (no 3D ref)",
    "gemini_uoais_ref_test_1800": "Gemini + 3D ref",
    "gpt4o_noref_test_1800": "GPT-4o (no 3D ref)",
    "gpt4o_uoaisref_test_1800": "GPT-4o + 3D ref",
    "internvl3_5_14b_4bit_noref_test_1800": "InternVL3.5-14B-4bit (no 3D ref)",
    "internvl3_5_14b_4bit_uoaisref_test_1800": "InternVL3.5-14B-4bit + 3D ref",
    "qwen3_5_9b_4bit_noref_test_1800": "Qwen3.5-9B-4bit (no 3D ref)",
    "qwen3_5_9b_4bit_uoaisref_test_1800": "Qwen3.5-9B-4bit + 3D ref",
}
# {display name: (file stem, fused CSV fed into the stack)}. Every entry uses the
# same recipe - the VLM's +3D-ref log fused with logs/uoais_1800_v2 under adaptive
# Platt and weights (0.5, 0.5, -1) - and is decoded at tau_edge=0.09.
MATH_SYSTEMS = {
    "VLM + math stack, t=0 (no edge filtering)":
        ("math_stack", "logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv",
         "gemini_uoais_ref_test_1800"),
    "GPT-4o + 3D fused + math stack":
        ("math_stack_gpt4o", "logs/fused_gpt4o_uoaisref/fused_adaptive_w0.5_0.5_m1.csv",
         "gpt4o_uoaisref_test_1800"),
    "Qwen3.5-9B-4bit + 3D fused + math stack":
        ("math_stack_qwen", "logs/fused_qwen_uoaisref/fused_adaptive_w0.5_0.5_m1.csv",
         "qwen3_5_9b_4bit_uoaisref_test_1800"),
    "InternVL3.5-14B-4bit + 3D fused + math stack":
        ("math_stack_internvl", "logs/fused_internvl_uoaisref/fused_adaptive_w0.5_0.5_m1.csv",
         "internvl3_5_14b_4bit_uoaisref_test_1800"),
}


class GtIndex:
    """Sample lookup plus the two label rules, straight from the GT file."""

    def __init__(self, gt_path: Path):
        self.rows = json.loads(gt_path.read_text(encoding="utf-8"))
        self.by_case = {(int(r["image_id"]), int(r["query_object"])): i
                        for i, r in enumerate(self.rows)}
        self.edges, self.free, self.target, self.target_free, self.level = {}, {}, {}, {}, {}
        for i, r in enumerate(self.rows):
            paths = r.get("occlusion_paths") or []
            target = int(r["query_object"])
            # graph_ops orients edges (blocked, blocker), the reverse of a GT path
            # step p[i] -> p[i+1] ("p[i] occludes p[i+1]").
            e_star = sorted({(v, u) for p in paths for u, v in zip(p, p[1:])})
            self.edges[i] = G.gt_edges_from_paths(paths)
            self.free[i] = set(OPS.free_set(target, e_star))
            self.target[i] = target
            self.target_free[i] = bool(OPS.target_is_free(target, e_star))
            self.level[i] = LEVEL[r["difficulty"]]

    def edge_label(self, sample: int, src: int, dst: int) -> int:
        return int(G.Edge(occluder_id=src, occluded_id=dst) in self.edges[sample])

    def object_label(self, sample: int, obj: int) -> int:
        """1 = the object is free in the GT occlusion graph of this sample."""
        if obj == self.target[sample]:
            return int(self.target_free[sample])
        return int(obj in self.free[sample])


def write(path: Path, rows: list[dict], fields: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ── extractors ─────────────────────────────────────────────────────────────────

def from_uoais(log: Path, gt: GtIndex) -> tuple[list[dict], list[dict]]:
    """candidate_edges carry `prob`; there is no per-object score."""
    rows = []
    for r in json.loads((log / "results.json").read_text(encoding="utf-8")):
        sample = gt.by_case.get((int(r["image_id"]), int(r["query_object"])))
        if sample is None:
            continue
        for e in r.get("candidate_edges") or []:
            src, dst = int(e["from"]), int(e["to"])
            rows.append({"sample_id": sample, "edge": f"{src}->{dst}",
                         "edge_confidence": f"{float(e['prob']):.10f}",
                         "label": gt.edge_label(sample, src, dst),
                         "difficulty_level": gt.level[sample]})
    return rows, []


def from_vlm(log: Path, gt: GtIndex, mask_loader) -> tuple[list[dict], list[dict]]:
    """occlusion_chain -> edge rows, candidates -> object rows, badge ids resolved
    to GT ids through the annotation mask (same mapping export_edge_csvs_full.py
    uses). Rows whose badge lands on no object (id 0) or collapses to a self-loop
    are kept: they are real assertions and they are simply wrong, so label 0."""
    per_case = json.loads((log / "summary.json").read_text(encoding="utf-8"))["per_case"]
    edges, objects = [], []
    for case in per_case:
        sample = gt.by_case.get((int(case["image_id"]), int(case["query_mask_id"])))
        if sample is None:
            continue
        id_to_point = {int(k): v for k, v in (case.get("id_to_point") or {}).items()}
        mask = mask_loader(int(case["image_id"]))

        def to_gt(badge) -> int:
            badge = G._to_int(badge)
            if badge is None:
                return 0
            return int((ref.map_badge_point_to_gt(mask, badge, id_to_point) or {}).get("gt_id") or 0)

        for e in case.get("occlusion_chain") or []:
            p = e.get("edge_confidence")
            if p is None:
                continue
            src, dst = to_gt(e.get("occluder_id")), to_gt(e.get("occluded_id"))
            p = float(p) / 100.0 if float(p) > 1.5 else float(p)
            edges.append({"sample_id": sample, "edge": f"{src}->{dst}",
                          "edge_confidence": f"{p:.10f}",
                          "label": gt.edge_label(sample, src, dst),
                          "difficulty_level": gt.level[sample]})
        for c in case.get("candidates") or []:
            p = c.get("prob_free")
            if p is None:
                continue
            obj = to_gt(c.get("id"))
            p = float(p) / 100.0 if float(p) > 1.5 else float(p)
            objects.append({"sample_id": sample, "object": obj,
                            "object_confidence": f"{p:.10f}",
                            "label": gt.object_label(sample, obj),
                            "difficulty_level": gt.level[sample]})
    return edges, objects


def from_math(gt: GtIndex, csv_path: str, tau_edge: float,
              anchor_edges: list[dict], anchor_objects: list[dict]
              ) -> tuple[list[dict], list[dict]]:
    """Edge scores as handed TO the math stack; object scores as produced BY it.

    RAW-VLM ANCHOR. Both levels are restricted to the identities this backbone's own
    raw VLM produced (`anchor_edges` / `anchor_objects`, i.e. the `*_edges.csv` and
    `*_objects.csv` written next to these files). Fusion pulls in 3D-only candidates -
    5280 of 6854 edges for InternVL, 5370 of 6754 for Qwen - and scoring those would
    measure 3D's candidate recall rather than what calibration+fusion did to the VLM's
    own predictions. The graph inference still runs on the FULL fused scene, so 3D-only
    structure keeps influencing the anchored objects' marginals; only the rows written
    out are filtered.

    The two levels deliberately come from different places. `tau_edge` only moves the
    MAP cut-point - `p' = sigma(logit p - logit t)` - so
    the shifted posteriors carry no extra information while multiplying every edge
    probability upward (mean 0.84 against a 0.27 base rate). Reporting them would
    measure the cut-point, not the model, so the edge rows are the calibrated fused
    scores the stack consumes. The object rows are the free-set marginals q_o, which
    genuinely are the stack's own output.
    """
    sys.path.insert(0, str(ROOT / "math"))
    from freeset.io.csv_loader import load_scenes            # noqa: E402
    from freeset.pipeline import FreeSetPipeline, PipelineConfig  # noqa: E402

    keep = {(int(a["sample_id"]), a["edge"]) for a in anchor_edges}
    edges = []
    for r in csv.DictReader(Path(csv_path).open(encoding="utf-8")):
        sample = int(r["sample_id"])
        if (sample, r["edge"]) not in keep:
            continue                       # 3D-only / fusion-added: not a VLM prediction
        src, dst = (int(x) for x in r["edge"].split("->"))
        label = gt.edge_label(sample, src, dst)
        assert label == int(r["label"]), (
            f"label rule disagrees with {csv_path} at sample {sample} edge {r['edge']}")
        edges.append({"sample_id": sample, "edge": r["edge"],
                      "edge_confidence": f"{float(r['edge_confidence']):.10f}",
                      "label": label, "difficulty_level": gt.level[sample]})

    targets = {i: int(r["query_object"]) for i, r in enumerate(gt.rows)}
    scenes, _ = load_scenes(csv_path, targets, tau_edge=tau_edge)
    pipe = FreeSetPipeline(PipelineConfig())
    # inference over the full fused scene, then read off only the anchor objects
    q_by_sample = {int(s.sample_id): pipe.run(s).q for s in scenes}
    objects = []
    for a in anchor_objects:
        sample, obj = int(a["sample_id"]), int(a["object"])
        # q_o = 0 for an anchor object the fused graph never reached: the repo's own
        # convention (rocc_task3_object_level) - a confident miss, not a dropped row.
        q = q_by_sample.get(sample, {})
        objects.append({"sample_id": sample, "object": obj,
                        "object_confidence": f"{float(q.get(obj, 0.0)):.10f}",
                        "label": gt.object_label(sample, obj),
                        "difficulty_level": gt.level[sample]})
    assert len(edges) == len(anchor_edges) and len(objects) == len(anchor_objects), (
        f"{csv_path}: anchor {len(anchor_edges)}/{len(anchor_objects)} vs "
        f"kept {len(edges)}/{len(objects)}")
    return edges, objects


# ── main ───────────────────────────────────────────────────────────────────────

def run(args) -> None:
    gt = GtIndex(Path(args.gt_path))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs)
    manifest = {}

    for dirname, name in UOAIS_LOGS.items():
        e, o = from_uoais(logs / dirname, gt)
        manifest[name] = {"edges": write(out_dir / f"{dirname}_edges.csv", e, EDGE_FIELDS),
                          "objects": 0}
        print(f"{name:46} edges={len(e):6} objects=   -")

    mask_loader = G.MaskLoader()
    raw_vlm = {}
    for dirname, name in VLM_LOGS.items():
        e, o = from_vlm(logs / dirname, gt, mask_loader)
        raw_vlm[dirname] = (e, o)
        write(out_dir / f"{dirname}_edges.csv", e, EDGE_FIELDS)
        write(out_dir / f"{dirname}_objects.csv", o, OBJ_FIELDS)
        manifest[name] = {"edges": len(e), "objects": len(o)}
        print(f"{name:46} edges={len(e):6} objects={len(o):6}")

    for name, (stem, fused_csv, anchor_log) in MATH_SYSTEMS.items():
        a_e, a_o = raw_vlm[anchor_log]
        e, o = from_math(gt, fused_csv, args.tau_edge, a_e, a_o)
        # Named for the stage they belong to: the edge scores are the calibrated,
        # fused ones handed TO the stack, so `_edges.csv` alone would read as if
        # they came out of it. The objects really are the stack's own output.
        write(out_dir / f"{stem}_edges_calib_fused.csv", e, EDGE_FIELDS)
        write(out_dir / f"{stem}_objects.csv", o, OBJ_FIELDS)
        manifest[name] = {"edges": len(e), "objects": len(o)}
        print(f"{name:46} edges={len(e):6} objects={len(o):6}")

    stems = {name: dirname for dirname, name in {**UOAIS_LOGS, **VLM_LOGS}.items()}
    stems.update({name: stem for name, (stem, _, _) in MATH_SYSTEMS.items()})
    json.dump({"stems": stems, "counts": manifest},
              (out_dir / "manifest.json").open("w", encoding="utf-8"), indent=2)
    print(f"\nwrote {len(manifest)} systems to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-path", default="UnoBench/subset_difficulty/test_GT_small_1800.json")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="logs/final_reports/scores")
    ap.add_argument("--tau-edge", type=float, default=0.09)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
