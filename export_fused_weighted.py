#!/usr/bin/env python3
"""
Export the fused adaptive edge scores under explicit fusion weights.

    logit p_fuse = w_vlm·logit p_vlm_adaptive
                 + w_3d ·logit p_3d_adaptive
                 + w_prior·logit P(y)          P(y) = 0.270914 (fixed)

One CSV per weight triple, over the full fused candidate pool (VLM edges UNION 3D
edges, 6467 rows), with the same five columns as the *_new files:
`sample_id, edge, edge_confidence, label, difficulty_level`.

Weights apply only where both sources see the edge. An edge seen by one source
keeps that source's calibrated probability (K = 1: nothing to combine, no prior
to remove) - the same convention as export_edge_csvs_full.py, so two weight
triples differ only on the rows where fusion actually happens.

The (1, 1, -1) output is asserted to reproduce 5_fused_adaptive.csv row for row.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "calibration"))
import infer_adaptive_platt as IA    # noqa: E402

import graph_eval_lib as G           # noqa: E402
import gemini_uoais_lib as ref       # noqa: E402
import run_uoais_pipeline_v0 as P    # noqa: E402
from fuse_calibrate_eval import V2_DEFAULTS  # noqa: E402

CSV_DIR = ROOT / "logs/edge_scores_csv_test1800_full"
PRIOR_GEMINI = 0.270914   # base rate of the Gemini pool, kept for the anchor check
EPS = 1e-6
LEVEL = {"Easy": "easy", "Medium": "med", "Hard": "hard", "No-Occ": "no-occ"}
WEIGHTS = [(1.0, 1.0, -1.0), (0.5, 0.5, -1.0)]


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def metrics(pairs, n_bins: int = 10):
    n = len(pairs)
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    nll = sum(-(y * math.log(min(max(p, EPS), 1 - EPS)) + (1 - y) * math.log(1 - min(max(p, EPS), 1 - EPS)))
              for p, y in pairs) / n
    bins = defaultdict(list)
    for p, y in pairs:
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    ece = sum(len(b) / n * abs(sum(y for _, y in b) / len(b) - sum(p for p, _ in b) / len(b))
              for b in bins.values())
    return n, sum(y for _, y in pairs) / n, nll, brier, ece


def build_pool(args) -> list[dict]:
    """Full fused pool: one row per (sample, edge), with both calibrated sources."""
    gt_rows = json.loads(Path(args.gt_path).read_text(encoding="utf-8"))
    sample_id = {(int(r["image_id"]), int(r["query_object"])): i for i, r in enumerate(gt_rows)}
    gt_edges = {(int(r["image_id"]), int(r["query_object"])):
                G.gt_edges_from_paths(r.get("occlusion_paths") or []) for r in gt_rows}
    level = {(int(r["image_id"]), int(r["query_object"])): LEVEL[r["difficulty"]] for r in gt_rows}

    m_vlm = json.loads((ROOT / "calibration/adaptive_platt_model.json").read_text())
    m_3d = json.loads((ROOT / "calibration/adaptive_platt_3d_model.json").read_text())

    uoais = {}
    for path in (Path(args.uoais_log) / "image_graphs").glob("image_*.json"):
        g = json.loads(path.read_text(encoding="utf-8"))
        uoais[int(g["image_id"])] = {
            "n_uoais": int(g["n_uoais"]),
            "edges": {(int(e["from"]), int(e["to"])): float(e["prob"])
                      for e in P.rebuild_edges_v2(g["edges_all"], V2_DEFAULTS)},
        }

    per_case = json.loads((Path(args.vlm_log) / "summary.json").read_text(encoding="utf-8"))["per_case"]
    mask_loader = G.MaskLoader()
    vlm_by_case = defaultdict(list)
    for case_row in per_case:
        key = (int(case_row["image_id"]), int(case_row["query_mask_id"]))
        if key not in sample_id:
            continue
        n_cand = len(case_row.get("candidates") or [])
        id_to_point = {int(k): v for k, v in (case_row.get("id_to_point") or {}).items()}
        mask = mask_loader(key[0])
        for edge in case_row.get("occlusion_chain") or []:
            prob = edge.get("edge_confidence")
            if prob is None:
                continue
            src = G._to_int(edge.get("occluder_id"))
            dst = G._to_int(edge.get("occluded_id"))
            src_gt = int((ref.map_badge_point_to_gt(mask, src, id_to_point) or {}).get("gt_id") or 0) if src is not None else 0
            dst_gt = int((ref.map_badge_point_to_gt(mask, dst, id_to_point) or {}).get("gt_id") or 0) if dst is not None else 0
            p_raw = float(prob) / 100.0 if float(prob) > 1.5 else float(prob)
            vlm_by_case[key].append({"src": src_gt, "dst": dst_gt,
                                     "p_vlm": IA.calibrate(p_raw, n_cand, m_vlm)})

    pool = []
    for case in sorted(sample_id, key=lambda k: sample_id[k]):
        image = uoais.get(case[0], {"edges": {}, "n_uoais": 0})
        cal_3d = {pair: IA.calibrate(p, image["n_uoais"], m_3d) for pair, p in image["edges"].items()}
        used = set()
        for r in vlm_by_case.get(case, []):
            pair = (r["src"], r["dst"])
            used.add(pair)
            pool.append({"case": case, "src": pair[0], "dst": pair[1],
                         "p_vlm": r["p_vlm"], "p_3d": cal_3d.get(pair)})
        for pair, p_3 in sorted(cal_3d.items()):
            if pair not in used:
                pool.append({"case": case, "src": pair[0], "dst": pair[1],
                             "p_vlm": None, "p_3d": p_3})
    for r in pool:
        r["sample_id"] = sample_id[r["case"]]
        r["label"] = int(G.Edge(occluder_id=r["src"], occluded_id=r["dst"]) in gt_edges[r["case"]])
        r["difficulty_level"] = level[r["case"]]
    return pool


def fuse(row: dict, w_vlm: float, w_3d: float, w_prior: float,
         prior: float = PRIOR_GEMINI) -> float:
    """Default prior keeps callers written against the Gemini pool working unchanged
    (rocc_task4_edge_level.py); run() always passes this pool's own base rate."""
    if row["p_vlm"] is None:
        return row["p_3d"]
    if row["p_3d"] is None:
        return row["p_vlm"]
    return sigmoid(w_vlm * logit(row["p_vlm"]) + w_3d * logit(row["p_3d"]) + w_prior * logit(prior))


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = build_pool(args)
    both = sum(1 for r in pool if r["p_vlm"] is not None and r["p_3d"] is not None)
    print(f"[pool] {len(pool)} edges | {both} seen by both sources (weights act here) | "
          f"{len(pool) - both} by one source (that source's probability is kept)")

    # P(y) is a property of THIS candidate pool, so it has to be re-estimated for
    # every VLM branch; hard-coding the Gemini value would mis-shift the others.
    if args.prior:
        prior, src = float(args.prior), "--prior"
    else:
        pos = sum(r["label"] for r in pool)
        prior, src = pos / len(pool), f"pool base rate ({pos}/{len(pool)})"
    print(f"[prior] P(y) = {prior:.6f}  [{src}]\n")

    weights = ([tuple(float(x) for x in w.split(",")) for w in args.weights.split(";")]
               if args.weights else WEIGHTS)
    fields = ["sample_id", "edge", "edge_confidence", "label", "difficulty_level"]
    for w_vlm, w_3d, w_prior in weights:
        tag = f"w{w_vlm:g}_{w_3d:g}_{w_prior:g}".replace("-", "m")
        out_path = out_dir / f"fused_adaptive_{tag}.csv"
        rows = []
        for r in pool:
            rows.append({"sample_id": r["sample_id"], "edge": f"{r['src']}->{r['dst']}",
                         "edge_confidence": f"{fuse(r, w_vlm, w_3d, w_prior, prior):.10f}",
                         "label": r["label"], "difficulty_level": r["difficulty_level"]})
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        # (1, 1, -1) on the Gemini pool must reproduce the file already on disk.
        if args.anchor and (w_vlm, w_3d, w_prior) == (1.0, 1.0, -1.0):
            ref_rows = list(csv.DictReader((CSV_DIR / "5_fused_adaptive.csv").open(encoding="utf-8")))
            assert len(ref_rows) == len(rows), f"{len(ref_rows)} vs {len(rows)} rows"
            worst = max(abs(float(a["edge_confidence"]) - float(b["edge_confidence"]))
                        for a, b in zip(ref_rows, rows))
            # 5_fused_adaptive.csv was written with the exact pool base rate
            # 1752/6467 = 0.27091387...; PRIOR here is that value rounded to six
            # decimals as specified, which moves every fused row by ~1.6e-7.
            assert worst < 5e-6, f"(1,1,-1) differs from 5_fused_adaptive.csv, max |delta| = {worst}"
            print(f"[check] {out_path.name} matches 5_fused_adaptive.csv, max |delta| = {worst:.1e} "
                  f"(only from rounding P(y) {PRIOR_GEMINI:.6f} vs {1752 / 6467:.10f})")

        counts = Counter(r["difficulty_level"] for r in rows)
        print(f"\n=== {out_path.name}  (w_vlm={w_vlm:g}, w_3d={w_3d:g}, w_prior={w_prior:g})  "
              f"{len(rows)} rows  " + "  ".join(f"{k}={counts[k]}" for k in ("easy", "med", "hard")))
        print(f"{'group':9}{'n':>7}{'pos':>8}{'NLL':>10}{'Brier':>10}{'ECE':>10}")
        for group in ("overall", "med", "hard", "easy"):
            sel = rows if group == "overall" else [r for r in rows if r["difficulty_level"] == group]
            n, pos, nll, brier, ece = metrics([(float(r["edge_confidence"]), r["label"]) for r in sel])
            print(f"{group:9}{n:>7}{pos:>8.3f}{nll:>10.4f}{brier:>10.4f}{ece:>10.4f}")
    print(f"\nwrote {len(WEIGHTS)} CSVs to {out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm-log", default="logs/gemini_uoais_ref_test_1800")
    ap.add_argument("--uoais-log", default="logs/uoais_1800_v2")
    ap.add_argument("--gt-path", default="UnoBench/subset_difficulty/test_GT_small_1800.json")
    ap.add_argument("--out", default=str(CSV_DIR))
    ap.add_argument("--prior", type=float, default=None,
                    help="P(y) for the naive-Bayes term; default = base rate of this pool")
    ap.add_argument("--weights", default=None,
                    help='semicolon-separated triples, e.g. "0.5,0.5,-1"')
    ap.add_argument("--anchor", action="store_true",
                    help="check the (1,1,-1) output against 5_fused_adaptive.csv (Gemini only)")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
