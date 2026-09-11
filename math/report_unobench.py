#!/usr/bin/env python3
"""Report the post-fusion stack in the standard UnoBench SR / OR / MP_NED format.

The three metric functions are IMPORTED from the repo's `evaluate_nlp.py`, not
reimplemented, so the numbers are computed by exactly the same code that produced
`logs/uoais_pipeline_1800/eval_results_*.txt`.

Mapping this stack's outputs onto the three metrics
---------------------------------------------------
`evaluate_nlp.py` scores a VLM that emits an answer set and reasoning paths. This stack
emits a posterior over DAGs, so each metric is fed the corresponding decoded quantity:

  SR                  pred_ans = {X} if q_X > tau_set else F_hat = {o : q_o > tau_set}
                      (eq:freeset-pred). The {X} branch is required: for every one of the
                      600 No-Occ scenes the GT `top_objects` is exactly [query_object],
                      and Free(X,e) excludes X by definition, so F_hat alone would score 0.
                      Scored against GT `top_objects`, as evaluate_nlp.py does.

  Occlusion reasoning decoded MAP graph e^(1) (Top-1 MAP over D), turned back into CSV
                      orientation and compared as a directed edge set against
                      paths_to_triplets(GT paths).

  MP_NED              all simple paths of that same MAP graph running from a source down
                      to X, which mirrors how GT occlusion_paths are built.

Grouping. The template groups by the ORIGINAL `difficulty` field (Easy/Medium/Hard,
600 each). The 600 old-`Easy` records are exactly the 600 No-Occ records, so the Easy
group has no GT edges and its Occlusion-reasoning line is suppressed -- the same
suppression evaluate_nlp.py applies to its No-Occ group.

Coverage. 245 of the 1800 GT records have no candidate edge at all from the perception
front end. evaluate_nlp.py skips absent predictions outright; that would silently drop
205 No-Occ scenes this stack answers correctly. Both readings are therefore printed:

  --coverage full     (default) absent record => "no occlusion found", pred = ({X}, [[X]]).
                      Gives the 600/600/600 counts of the template.
  --coverage scored   skip absent records, exactly as evaluate_nlp.py does.

    python report_unobench.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Repository root from the file's own location, so the cwd does not matter.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from evaluate_nlp import compute_prf, mp_ned, paths_to_triplets  # noqa: E402

from freeset.decision.policy import DecisionConfig  # noqa: E402
from freeset.inference import graph_ops as G  # noqa: E402
from freeset.io.csv_loader import load_scenes  # noqa: E402
from freeset.pipeline import FreeSetPipeline, PipelineConfig  # noqa: E402

GT = os.path.join(_ROOT, "UnoBench/subset_difficulty/test_GT_small_1800.json")
CSV = os.path.join(_ROOT, "logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv")
MAX_PATHS = 64          # guard against path blow-up in a dense decoded DAG


def decode_paths(target: int, active) -> tuple[list[list[int]], int]:
    """All simple paths from a source down to X, in CSV orientation.

    The framework arc (i, j) means "i is blocked by j", so walking outward from X follows
    obstruction upward; reversing each walk yields a GT-style path
    [outermost, ..., X]. Returns ([[X]], 0) when X is unobstructed, matching the GT
    convention `occlusion_paths = [[query_object]]` for No-Occ scenes.
    """
    adj = G.adjacency(active)
    out: list[list[int]] = []
    truncated = 0

    def walk(node: int, seen: list[int]) -> None:
        nonlocal truncated
        if len(out) >= MAX_PATHS:
            truncated = 1
            return
        nxt = [v for v in adj.get(node, ()) if v not in seen]
        if not nxt:                                  # sink: nothing blocks `node`
            out.append(list(reversed(seen)))         # -> CSV orientation
            return
        for v in nxt:
            walk(v, seen + [v])

    walk(target, [target])
    return (out or [[target]]), truncated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--coverage", choices=["full", "scored"], default="full")
    ap.add_argument("--c-fp", type=float, default=1.0)
    ap.add_argument("--c-fn", type=float, default=1.0)
    ap.add_argument("--lambda-defer", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tau-edge", type=float, default=None,
                    help="MAP edge cut-point; loader default is 0.1. "
                         "Pass 0.5 for the unmodified MAP.")
    a = ap.parse_args()
    tau = {} if a.tau_edge is None else {"tau_edge": a.tau_edge}

    gt = json.load(open(a.gt))
    targets = {i: int(e["query_object"]) for i, e in enumerate(gt)}
    scenes, rep = load_scenes(a.csv, targets, **tau)
    by_id = {s.sample_id: s for s in scenes}

    dcfg = DecisionConfig(c_fp=a.c_fp, c_fn=a.c_fn, lambda_defer=a.lambda_defer)
    pipe = FreeSetPipeline(PipelineConfig(decision=dcfg))
    tau_set = dcfg.tau_set

    res = defaultdict(lambda: {"SR-P": [], "SR-R": [], "SR-F1": [],
                               "OP": [], "OR": [], "F1": [], "MP_NED": []})
    gt_edges = defaultdict(int)     # GT triplets per group; 0 => suppress the OR line
    n_absent = n_trunc = 0

    for idx, entry in enumerate(gt):
        X = int(entry["query_object"])
        gt_paths = entry["occlusion_paths"] or [[X]]
        gt_top = set(entry["top_objects"])
        diff = entry["difficulty"]

        s = by_id.get(idx)
        if s is None:
            n_absent += 1
            if a.coverage == "scored":
                continue
            pred_ans, pred_paths = {X}, [[X]]        # no candidate edge => no occlusion
        else:
            d = pipe.run(s)
            pred_ans = set(d.free_set)
            if d.q.get(X, 0.0) > tau_set:            # X itself is the graspable object
                pred_ans |= {X}
            modes, _ = pipe.modes(s)
            pred_paths, t = decode_paths(X, modes[0].active)   # MAP e^(1)
            n_trunc += t

        p, r, f1 = compute_prf(pred_ans, gt_top)
        res[diff]["SR-P"].append(p); res[diff]["SR-R"].append(r); res[diff]["SR-F1"].append(f1)

        gt_trip = paths_to_triplets(gt_paths)
        gt_edges[diff] += len(gt_trip)
        op, orr, f1t = compute_prf(paths_to_triplets(pred_paths), gt_trip)
        res[diff]["OP"].append(op); res[diff]["OR"].append(orr); res[diff]["F1"].append(f1t)
        res[diff]["MP_NED"].append(mp_ned(pred_paths, gt_paths))

    # ---- output ----------------------------------------------------------------------
    lines = []
    def emit(s=""):
        lines.append(s); print(s)

    emit("========== Evaluation Summary ==========")
    emit(f"Scenes loaded from CSV      : {rep.n_scenes}  (targets guessed {rep.guessed_targets})")
    emit(f"GT records with no candidate: {n_absent}")
    emit(f"Coverage mode               : {a.coverage}"
         + ("  (absent => 'no occlusion', pred = ({X}, [[X]]))" if a.coverage == "full"
            else "  (absent records skipped, as evaluate_nlp.py does)"))
    emit(f"tau_set={tau_set:.2f}  tau_act={dcfg.tau_act:.2f}"
         + (f"   [paths truncated at {MAX_PATHS} on {n_trunc} scenes]" if n_trunc else ""))
    emit("========================================")
    emit()

    for diff in ["Easy", "Medium", "Hard"]:
        if not res[diff]["MP_NED"]:
            continue
        emit(f"=== {diff} ===")
        emit(f"SR: P={np.mean(res[diff]['SR-P']):.4f}, "
             f"R={np.mean(res[diff]['SR-R']):.4f}, "
             f"F1={np.mean(res[diff]['SR-F1']):.4f} "
             f"({len(res[diff]['SR-P'])} samples)")
        if gt_edges[diff]:                       # No-Occ group has no GT edges at all
            emit(f"Occlusion reasoning: "
                 f"P={np.mean(res[diff]['OP']):.4f}, "
                 f"R={np.mean(res[diff]['OR']):.4f}, "
                 f"F1={np.mean(res[diff]['F1']):.4f}")
        emit(f"MP_NED: {np.mean(res[diff]['MP_NED']):.4f}")
        emit()

    grp = [np.mean(res[d]["SR-F1"]) for d in ["Easy", "Medium", "Hard"] if res[d]["SR-F1"]]
    emit("=== Overall (Group-weighted) ===")
    emit(f"Balanced SR-F1 (Group-weighted) = {np.mean(grp):.3f}")

    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
