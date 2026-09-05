#!/usr/bin/env python3
"""
Chay run_gemini_uoais_ref_batch.py tren du lieu REAL (data_ifl_real) thay vi
UnoBench synthetic.

Pipeline goc gia dinh moi thu nam trong UnoBench/: anh, depth .npy, mask .npy,
va bang ten vat name_for_all.json, voi image_id la chi so phang cua UnoBench.
Ban real thi khac o 4 diem, va file nay va vao dung 4 cho do roi goi lai
``run_gemini_uoais_ref_batch.run`` — khong sua mot dong nao cua pipeline goc:

  1. anh RGB   : images/image_XXXXXX.png (crop giua 1200x1200 cua 1944x1200)
  2. mask GT   : masks_npy_real_crop/image_XXXXXX.npy (cung crop 1200x1200)
  3. depth     : data_ifl_*/.../scene{N}/3.npz['depth'], crop cung kieu, cm->mm
  4. ten vat   : real_object_names.json (rut tu test_nlp_reasoning.jsonl)

image_id -> scene{N} lay tu real_world_mapping_fixed.json.

Usage:
    python run_gemini_uoais_ref_batch_real.py \
        --gt-path test_GT_subset_hardall_easy300_medium300.json \
        --id-map real_world_mapping_fixed.json \
        --out logs/gemini_uoais_ref_real_subset
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CROP_LEFT = (1944 - 1200) // 2      # = 372, giong images/resize.py
CROP_W = CROP_H = 1200


def scene_dirs() -> dict[int, Path]:
    dirs: dict[int, Path] = {}
    for p in glob.glob(str(ROOT / "data_ifl_*/mnt/data1/data_ifl_real/scene*")):
        m = re.search(r"scene(\d+)$", p)
        if m:
            dirs[int(m.group(1))] = Path(p)
    return dirs


def load_id_map(path: Path) -> dict[int, int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for image_id, scene_name in raw.items():
        m = re.search(r"scene(\d+)", str(scene_name))
        if m:
            out[int(image_id)] = int(m.group(1))
    return out


def patch(id_map_path: Path, image_dir: Path, ann_dir: Path, names_path: Path) -> None:
    """Tro pipeline goc sang du lieu real."""
    import gemini_uoais_lib as ref
    import run_gemini_uoais_ref as base_ref
    import run_gemini_uoais_ref_batch as batch
    import unobench_gemini_common as legacy

    id_map = load_id_map(id_map_path)
    dirs = scene_dirs()
    names = json.loads(names_path.read_text(encoding="utf-8"))

    for mod in (base_ref, batch):
        mod.IMAGE_DIR = image_dir
        mod.ANN_DIR = ann_dir
    ref.ANN_DIR = ann_dir

    def load_real_depth_mm(image_id):
        """Depth cua scene tuong ung, crop giong anh RGB, doi cm -> mm."""
        scene = id_map[int(image_id)]
        depth = np.load(dirs[scene] / "3.npz", allow_pickle=True)["depth"].astype(np.float32)
        depth = depth[:CROP_H, CROP_LEFT:CROP_LEFT + CROP_W]
        if np.nanmax(depth) < 200:      # cm -> mm, cung quy uoc voi ban synthetic
            depth = depth * 10.0
        return depth

    base_ref.load_unobench_depth_mm = load_real_depth_mm
    batch.run_uoais_reference_for_image.__globals__["load_unobench_depth_mm"] = load_real_depth_mm
    batch.run_uoais_reference_for_image.__globals__["IMAGE_DIR"] = image_dir

    def resolve_real_name(image_id, obj_id, _names, _id_to_scene_key):
        return names.get(str(int(image_id)), {}).get(str(int(obj_id)))

    base_ref.resolve_object_name = resolve_real_name
    base_ref.make_case.__globals__["resolve_object_name"] = resolve_real_name
    base_ref.old_scene_key_by_image_id = lambda: {i: str(i) for i in id_map}
    batch.old_scene_key_by_image_id = base_ref.old_scene_key_by_image_id
    batch.select_all_cases.__globals__["NAME_FOR_ALL"] = names_path

    # Anh/depth/mask real da nam san tren dia -> khong giai nen tu zip UnoBench.
    legacy.ensure_images = lambda *a, **k: None
    legacy.ensure_depth = lambda *a, **k: None
    legacy.ensure_annotations = lambda *a, **k: None

    return batch


def main() -> None:
    argv = sys.argv[1:]
    extra = {"--id-map": ROOT / "real_world_mapping_fixed.json",
             "--image-dir": ROOT / "images",
             "--ann-dir": ROOT / "masks_npy_real_crop",
             "--names": ROOT / "real_object_names.json"}
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] in extra:
            extra[argv[i]] = Path(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1

    batch = patch(extra["--id-map"], extra["--image-dir"], extra["--ann-dir"], extra["--names"])
    sys.argv = [sys.argv[0]] + rest
    batch.run(batch.parse_args())


if __name__ == "__main__":
    main()
