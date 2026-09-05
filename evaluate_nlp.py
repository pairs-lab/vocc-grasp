#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Occlusion-reasoning evaluation (final version)
Supports <points x y>object name</points> outputs.
Maps coordinates to object IDs via instance masks ('instances_objects'),
then computes SR / OR / MP-NED and detailed hit statistics.
"""

import argparse
import json, re, os
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


# ============================================================
# Regex template
# ============================================================

# Format in <answer>: <points x y>object name</points>
ANSWER_POINTS_RE = re.compile(r"<points\s+([\d\.]+)\s+([\d\.]+)>(.*?)</points>", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)

# Format in <think>: (x, y)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
THINK_COORD_RE = re.compile(r"\(([\d\.]+),\s*([\d\.]+)\)", re.IGNORECASE)
IMAGE_ID_RE = re.compile(r"image_(\d+)", re.IGNORECASE)

REAL_CROP_WIDTH = 1200
REAL_CROP_HEIGHT = 1200
REAL_ORI_WIDTH = 1944
REAL_ORI_HEIGHT = 1200

DATASET_TYPES = ("synthetic", "real")


# ============================================================
# Parsing Tools
# ============================================================
def extract_points_from_answer(text):
    """Extract <points x y>object name</points> from <answer> tag"""
    m = ANSWER_TAG_RE.search(text or "")
    if not m:
        return []
    pts = ANSWER_POINTS_RE.findall(m.group(1))
    # Here it is (x, y)
    return [(float(x), float(y), name.strip()) for x, y, name in pts]



def parse_image_id_from_path(img_path):
    m = IMAGE_ID_RE.search(img_path or "")
    if not m:
        return None
    return int(m.group(1))


def get_image_id(record):
    image_id = record.get("image_id")
    if image_id is None:
        image_id = parse_image_id_from_path(record.get("image", ""))
    if image_id is None:
        return None
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return image_id


def get_crop_offsets(dataset_type):
    if dataset_type == "real":
        return ((REAL_ORI_WIDTH - REAL_CROP_WIDTH) / 2,
                (REAL_ORI_HEIGHT - REAL_CROP_HEIGHT) / 2)
    return 0.0, 0.0


def coords_to_object_ids(image_id, coords_list, npz_root, dataset_type="synthetic"):
    """Return object IDs and count hits / misses."""
    if image_id is None:
        print("[WARN] Missing image_id; cannot locate instance mask.")
        return [0 for _ in coords_list], 0, len(coords_list)

    if isinstance(image_id, int):
        npz_name = f"image_{image_id:06d}.npy"
    else:
        npz_name = f"image_{image_id}.npy"
    npz_path = os.path.join(npz_root, npz_name)

    if not os.path.exists(npz_path):
        print(f"[WARN] NPZ not found: {npz_path}")
        return [0 for _ in coords_list], 0, len(coords_list)

    mask = np.load(npz_path).astype(int)
    H, W = mask.shape
    offset_x, offset_y = get_crop_offsets(dataset_type)

    ids, hits, misses = [], 0, 0
    for (x, y, _) in coords_list:
        x += offset_x
        y += offset_y
        x_int, y_int = int(round(x)), int(round(y))

        if 0 <= x_int < W and 0 <= y_int < H:
            obj_id = int(mask[y_int, x_int])
        else:
            obj_id = 0

        if obj_id > 0:
            hits += 1
        else:
            misses += 1

        ids.append(obj_id)

    return ids, hits, misses



def extract_answer_object_ids(model_output, image_id, npz_root, dataset_type="synthetic"):
    coords = extract_points_from_answer(model_output)
    if not coords:
        return [], 0, 0
    ids, hits, misses = coords_to_object_ids(image_id, coords, npz_root, dataset_type)
    return [i for i in ids if i > 0], hits, misses


def parse_think_paths_with_coords(model_output, image_id, npz_root, dataset_type="synthetic"):
    """
    Extract 'A at (x, y) is occluded by B at (x, y)' structure from <think>,
    convert to object_id path list ordered by occlusion relationship (occluder -> occluded).
    Build paths using "concatenate + reverse" logic.
    """
    m = THINK_RE.search(model_output or "")
    if not m:
        return [], 0, 0
    think_text = m.group(1).strip()
    if not think_text:
        return [], 0, 0

    parts = re.split(r"Path\d*:", think_text)
    all_paths = []
    total_hits = total_misses = 0

    for p in parts:
        p = p.strip()
        if not p:
            continue

        rels = re.findall(
            r"(.+?)\s+at\s+\(([\d\.]+),\s*([\d\.]+)\)\s+is occluded by\s+(.+?)\s+at\s+\(([\d\.]+),\s*([\d\.]+)\)",
            p, re.IGNORECASE)

        if rels:
            coords_list = []
            for (child_name, x1, y1, parent_name, x2, y2) in rels:
                coords_list.append((float(x1), float(y1), child_name.strip()))
                coords_list.append((float(x2), float(y2), parent_name.strip()))

            ids, hits, misses = coords_to_object_ids(image_id, coords_list, npz_root, dataset_type)
            total_hits += hits
            total_misses += misses

            # Concatenate and reverse
            # ids format: [child1, parent1, child2, parent2, ...]
            # We want to get parent-to-child order chain
            if len(ids) >= 2:
                # Take the first child->parent pair
                tmp_list = [ids[0], ids[1]]
                # Append the parent of each subsequent relationship
                #tmp_list += [ids[i + 1] for i in range(2, len(ids) - 1, 2)]
                tmp_list+=ids[3::2]
                tmp_list = [x for x in tmp_list if x is not None]
                tmp_list.reverse()  # Reverse direction: occluder -> occluded
                tmp_list = [int(x) for x in tmp_list if isinstance(x, (int, float))]
               #  if len(ids)>=4:
               #      if ids[1:-1:2]!=ids[2::2]:
               #          tmp_list=[]
                
                if tmp_list:
                    all_paths.append(list(dict.fromkeys(tmp_list)))  # Remove duplicates while preserving order

        else:
            m_single = re.search(r"at\s+\(([\d\.]+),\s*([\d\.]+)\)", p)
            if m_single:
                x, y = float(m_single.group(1)), float(m_single.group(2))
                ids, hits, misses = coords_to_object_ids(image_id, [(x, y, "")], npz_root, dataset_type)
                total_hits += hits
                total_misses += misses
                if ids and ids[0] > 0:
                    all_paths.append([ids[0]])

    return all_paths, total_hits, total_misses

    



# ============================================================
# Metric Functions
# ============================================================
def ned(seq1, seq2):
    import editdistance
    return editdistance.eval(seq1, seq2) / max(len(seq1), len(seq2))


def mp_ned(pred_paths, gt_paths, alpha=1.0, beta=1.0):
    m, n = len(pred_paths), len(gt_paths)
    size = max(m, n)
    if size == 0:
        return 0.0
    C = np.zeros((size, size))
    for i in range(m):
        for j in range(n):
            C[i, j] = ned(pred_paths[i], gt_paths[j])
    if m < n:
        for i in range(m, size):
            C[i, :n] = alpha
    elif m > n:
        for j in range(n, size):
            C[:m, j] = beta
    row_ind, col_ind = linear_sum_assignment(C)
    return C[row_ind, col_ind].sum() / size



def paths_to_triplets(paths):
    triplets = set()
    for p in paths:
        for i in range(len(p) - 1):
            triplets.add((p[i], p[i + 1]))
    return triplets


def compute_prf(pred, gt):
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


# ============================================================
# Main Evaluation Pipeline
# ============================================================
def main(pred_path, gt_path, npz_root, dataset_type="synthetic", difficulty_field="new_difficulty"):
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    if not os.path.isdir(npz_root):
        raise FileNotFoundError(f"NPZ root directory not found: {npz_root}")

    preds = {}
    total_hits = total_misses = total_points = 0

    # ---------- Parse Predictions ----------
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            image_id = get_image_id(it)
            if image_id is None:
                continue

            ans_ids, h1, m1 = extract_answer_object_ids(
                it["model_output"], image_id, npz_root, dataset_type)
            think_paths, h2, m2 = parse_think_paths_with_coords(
                it["model_output"], image_id, npz_root, dataset_type)
            total_hits += h1 + h2
            total_misses += m1 + m2
            total_points += h1 + h2 + m1 + m2
            if "query_object" not in it:
                print("[WARN] Prediction missing query_object; skipping record.")
                continue
            tgt = int(it["query_object"])
            preds[(image_id, tgt)] = {"answer": ans_ids, "paths": think_paths}

    # ---------- Load GT ----------
    gts = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        arr = json.load(f)
        for it in arr:
            image_id = get_image_id(it)
            if image_id is None:
                continue
            tgt = int(it["query_object"])
            paths = it.get("occlusion_paths", []) or [[tgt]]
            diff = "No-Occ" if all(len(p) == 1 for p in paths) else it.get(difficulty_field, "Easy")
            gts[(image_id, tgt)] = {
                "paths": paths,
                "tops": it.get("top_objects", [p[0] for p in paths]),
                "difficulty": diff,
            }

    # ---------- Compute Metrics ----------
    res = defaultdict(lambda: {"SR-P": [], "SR-R": [], "SR-F1": [],
                               "OP": [], "OR": [], "F1": [], "MP_NED": []})
    for key, gt in gts.items():
        if key not in preds:
            continue
        pred = preds[key]
        gt_top = set(gt["tops"])
        pred_ans = set(pred["answer"])
        p, r, f1 = compute_prf(pred_ans, gt_top)
        res[gt["difficulty"]]["SR-P"].append(p)
        res[gt["difficulty"]]["SR-R"].append(r)
        res[gt["difficulty"]]["SR-F1"].append(f1)

        gt_trip = paths_to_triplets(gt["paths"])
        pred_trip = paths_to_triplets(pred["paths"])
        op, orr, f1t = compute_prf(pred_trip, gt_trip)
        res[gt["difficulty"]]["OP"].append(op)
        res[gt["difficulty"]]["OR"].append(orr)
        res[gt["difficulty"]]["F1"].append(f1t)
        res[gt["difficulty"]]["MP_NED"].append(mp_ned(pred["paths"], gt["paths"]))

    # ---------- Output ----------
    print("\n========== Evaluation Summary ==========")
    print(f"Total parsed coordinates : {total_points}")
    print(f"  Valid hits (object_id>0): {total_hits}")
    print(f"  Miss / empty (object_id=0): {total_misses}")
    print(f"  Hit ratio: {100*total_hits/(total_points+1e-8):.2f}%")
    print("========================================\n")

    for diff in ["No-Occ", "Easy", "Medium", "Hard"]:
        if not res[diff]["MP_NED"]:
            continue
        print(f"=== {diff} ===")
        print(f"SR: P={np.mean(res[diff]['SR-P']):.4f}, "
              f"R={np.mean(res[diff]['SR-R']):.4f}, "
              f"F1={np.mean(res[diff]['SR-F1']):.4f} "
              f"({len(res[diff]['SR-P'])} samples)")
        if diff != "No-Occ":
            print(f"Occlusion reasoning: "
                  f"P={np.mean(res[diff]['OP']):.4f}, "
                  f"R={np.mean(res[diff]['OR']):.4f}, "
                  f"F1={np.mean(res[diff]['F1']):.4f}")
        print(f"MP_NED: {np.mean(res[diff]['MP_NED']):.4f}\n")


    # Overall (group-weighted): equal weight for No-Occ, Easy, Medium, and Hard.
    def mean_or_zero(lst):
        return float(np.mean(lst)) if lst else 0.0

    sr_no_occ_val = mean_or_zero(res["No-Occ"]["SR-F1"])
    srf1_easy_val = mean_or_zero(res["Easy"]["SR-F1"])
    srf1_medium_val = mean_or_zero(res["Medium"]["SR-F1"])
    srf1_hard_val = mean_or_zero(res["Hard"]["SR-F1"])
    overall_group_srf1 = 0.25 * (sr_no_occ_val + srf1_easy_val + srf1_medium_val + srf1_hard_val)

    print("=== Overall (Group-weighted) ===")
    print(f"Balanced SR-F1 (Group-weighted) = {overall_group_srf1:.3f}")


# ============================================================
# Entry Point
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate NLP predictions against instance masks and GT occlusion paths.")
    parser.add_argument("--pred_path", required=True,
                        help="Path to predictions jsonl file.")
    parser.add_argument("--gt_path", required=True,
                        help="Path to GT json file.")
    parser.add_argument("--npz_root", required=True,
                        help="Path to instance mask npy directory.")
    parser.add_argument("--dataset_type", choices=DATASET_TYPES,
                        default="synthetic",
                        help="Dataset type: use 'real' for cropped real images.")
    parser.add_argument("--difficulty_field", choices=("difficulty", "new_difficulty"),
                        default="new_difficulty",
                        help="GT field to use for Easy/Medium/Hard bucketing.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.pred_path, args.gt_path, args.npz_root, args.dataset_type, args.difficulty_field)

