#!/usr/bin/env python3
"""Emit the real target map {sample_id: X} from the UnoBench GT list.

The fused CSV carries no target column, and X is not recoverable from the edges --
`csv_loader._guess_target` is a documented heuristic that has to guess every scene.
But `sample_id` is the 0-based index of the case in the GT list, so the GT field
`query_object` *is* X. Supplying it removes the heuristic entirely.

The index alignment is asserted here, not assumed: it is the single point where the
offset-0 correspondence is relied upon.

    python make_targets.py --out ../logs/math_test1800/targets.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import pandas as pd

GT = "../UnoBench/subset_difficulty/test_GT_small_1800.json"
CSV = "../logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv"

# GT `difficulty` -> the CSV's `difficulty_level` spelling.
DIFF = {"Easy": "easy", "Medium": "med", "Hard": "hard"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--out", default="../logs/math_test1800/targets.json")
    a = ap.parse_args()

    gt = json.load(open(a.gt))
    df = pd.read_csv(a.csv)
    sids = sorted(df.sample_id.unique())

    # ---- alignment assertions -----------------------------------------------------
    assert len(gt) == 1800, f"GT list must hold 1800 cases, got {len(gt)}"
    assert max(sids) < len(gt), f"sample_id {max(sids)} out of range for GT"
    bad = [s for s in sids
           if DIFF.get(gt[int(s)]["difficulty"]) != str(df[df.sample_id == s]
                                                        .difficulty_level.iloc[0])]
    assert not bad, f"difficulty mismatch on {len(bad)} scenes, e.g. {bad[:5]}"
    print(f"alignment OK: sample_id indexes the GT list at offset 0 "
          f"({len(sids)}/{len(sids)} difficulty match)")

    targets = {int(s): int(gt[int(s)]["query_object"]) for s in sids}
    json.dump(targets, open(a.out, "w"), indent=0)
    print(f"wrote {a.out}  ({len(targets)} targets)")

    # ---- report the scenes where X has no candidate edge at all -------------------
    pe = [e.split("->") for e in df.edge]
    df["_A"] = [int(x) for x, _ in pe]
    df["_B"] = [int(y) for _, y in pe]
    d = df[df._A != df._B]
    absent = Counter()
    for sid, g in d.groupby("sample_id"):
        nodes = set(g._A) | set(g._B)
        if targets[int(sid)] not in nodes:
            absent[gt[int(sid)]["new_difficulty"]] += 1
    tot = sum(absent.values())
    print(f"\nX absent from the scene's own candidate edges: {tot} scenes")
    print("  by GT difficulty:", dict(absent))
    print(f"  -> {absent.get('No-Occ', 0)} are genuinely No-Occ: X isolated, q_X=1,")
    print("     F={} and 'grasp target' is the CORRECT answer.")
    print(f"  -> {tot - absent.get('No-Occ', 0)} are upstream perception recall misses:")
    print("     the stack will say 'grasp target' but X is really occluded.")


if __name__ == "__main__":
    main()
