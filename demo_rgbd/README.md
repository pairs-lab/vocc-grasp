# demo_rgbd — 28 RGB-D scenes for running the pipeline end to end

The minimum data needed to **run and self-check** the chain

```
RGB-D → segmentation → pick the blocking object → FGC-GraspNet → 6-DoF pose
```

with no camera, no robot, and no 28 GB dataset download. Exactly the 28 scenes used in
`grasp_viz_real/gemini_uoais_ref/`. **63 MB.**

```bash
python verify.py              # LEVEL 1: data integrity        (numpy + opencv, ~5 s)
python verify.py --run        # LEVEL 2: replay the grasp stage (needs FGC + GPU)
```

## What a case holds

```
img000059_q4/
  frame_rgb.png      1200×1200×3  uint8
  frame_depth.png    1200×1200    uint16, MILLIMETRES  (0 = dead pixel, not 0 m)
  instances.png      1200×1200    uint16, object ids   (0 = background)
  intrinsics.json    fx fy cx cy width height depth_scale
  labels.json        {id: name}
  target.json        the request, the answer, and what the reference run did
```

`target.json`:

```jsonc
{
  "case": "img000059_q4", "image_id": 59, "difficulty": "Medium",
  "request": "white box",                       // what the user asks for
  "instances": {"1": "lemon", "2": "red bowl", "3": "orange sponge", "4": "white box"},
  "ground_truth": {"top_ids": [3]},             // objects that are NOT occluded
  "reference_run": {                            // what Gemini + UOAIS reference chose
    "method": "gemini_uoais_ref", "target_id": 3,
    "target_label": "sponge", "target_correct": true
  }
}
```

The user asks for the **white box** (id 4), but the **orange sponge** (id 3) is on top of
it. The reasoner has to answer *clear the sponge first*. `ground_truth.top_ids` is the
answer key — used for **scoring only**, never fed to the model.

At the directory level:

| | |
|---|---|
| `cases.json` | index of the 28 cases: image id, difficulty, request, object count |
| `extrinsics.json` | `T_BASE_CAM` **for this image set only** — see below |
| `expected/<case>.json` | reference results, produced **on the files in this repo** |
| `manifest.sha256` | 198 data files |

## The case set

28 scenes — **21 Hard, 7 Medium** — 2 to 10 objects each, stacked and overlapping.

## Self-check

**LEVEL 1** opens every file and checks shape, dtype, value range, and that the object ids
in `target.json` really exist in `instances.png`, then verifies sha256. No GPU, no model,
no network.

**LEVEL 2** replays only the geometry half — mask → point cloud → FGC-GraspNet → pose —
taking the target object from `expected/<case>.json`, so it needs the FGC checkpoint and a
GPU but **no API key**. Matching the recorded pose within 1 mm / 1° passes. A two-finger
gripper is symmetric about its approach axis, so a 180° flip counts as the same grasp.

## Running the full pipeline

From the repository root, with a Gemini key in `.env`:

```bash
python demo.py --case img000059_q4 --masks provided
```

`--masks provided` uses `instances.png` and reproduces `expected/`. `--masks uoais` runs
UOAIS for the mask instead, which is the deployable path and needs no annotation.

## Coordinate frame

`extrinsics.json` places the table at z = 0 and the bin 0.55 m in front of the robot base
so the geometry chain runs end to end and produces something you can look at. These are
real captures with no robot frame attached — **on a real cell, replace it with your own
hand-eye calibration.**

## Provenance

Cropped from MetaGraspNet-V2 (`data_ifl_real`), 1200×1200 centre crop of the original
1944×1200 frames. Subject to that dataset's own terms; included here for demonstration.
