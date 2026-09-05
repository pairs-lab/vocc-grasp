# VOCC-Grasp: Calibrated occlusion reasoning for grasping in clutter

This is the repository for VOCC-Grasp, a pipeline that answers **which object to move first**
when the object a user asks for is buried in a cluttered bin, and returns a 6-DoF grasp for it.

A VLM proposes the occlusion structure of the bin and an amodal segmenter scores the same
structure from geometry. Both edge sources are calibrated and fused into one probability per
edge. The posterior is then taken over **acyclic** occlusion graphs — not a single graph — and
the resulting free-set marginals carry an approximation certificate, so the decision to act or
defer is made under a bound rather than a guess.

```
RGB-D
  ├── Gemini ──────────► occlusion chain, p(edge)
  └── UOAIS  ──────────► amodal masks, p(edge) from contact / hidden area / depth order
                              │
                    adaptive Platt per source, then logit fusion          configs/pipeline.yaml
                              │
                    Top-K MAP over the acyclic support D
                    exact enumeration ≤ 20 edges, else ILP with
                    lazy acyclicity + no-good cuts (CBC)
                              │
                    free-set marginals q_o, q_X  +  certificate ε_K
                    adaptive stopping: first K with ε_K ≤ ε_target,
                    or an action margin already certified
                              │
                    τ_set = c_FP/(c_FP+c_FN),  τ_act = 1 − λ_defer  →  act or defer
                              │
  FGC-GraspNet on the chosen object ──────────────────────────────────► 6-DoF pose
```

The certificate is tail-model-free: `Z̄ = Π(1 − p_ij p_ji) ≥ Z` bounds the partition function
from above, so `ε_K = (Z̄ − Z_K)/Z̄` bounds the marginal error without assuming anything about
the unenumerated tail. [`math/Mathematical.md`](math/Mathematical.md) derives it.

The calibration is fit **once**, on a scene-disjoint slice of the synthetic training split, and
used unchanged for synthetic evaluation, real evaluation and deployment. Nothing is refit per
domain and nothing is retrained.

## Setup

### Installation Requirements

- Torch 2.7.0, Torchvision 0.22.0
- CUDA 12.8
- A CUDA GPU for the demo and for grasp inference; the reports run on CPU
- Tested on an RTX 5060 Ti (sm_120)

### Installation Step

1. **Create the environment and build everything**
   ```bash
   ./scripts/install.sh uoais
   conda activate vocc
   ```
   This creates the conda env `vocc` (python 3.10), installs torch cu128 and
   `requirements.txt`, compiles the FGC-GraspNet CUDA extensions, and installs
   detectron2 + AdelaiDet. The `uoais` argument is required for the demo.

1. **Download the FGC-GraspNet checkpoint**
   ```bash
   ./scripts/download_checkpoints.sh
   ```
   Fetches `checkpoint_fgc.tar` (13 MB) into `FreeGrasp_code/logs/`.

1. **Download the UOAIS weights**

   These need the upstream terms accepted and cannot be redistributed here. Get
   `R50_rgbdconcat_mlc_occatmask_hom_concat` from
   [gist-ailab/uoais](https://github.com/gist-ailab/uoais) and place it at:
   ```
   uoais-ft/output/R50_rgbdconcat_mlc_occatmask_hom_concat/model_final.pth
   ```

1. **Set your Gemini API key**
   ```bash
   cp .env.example .env      # then fill in GEMINI_API_KEY
   ```

Without the `uoais` argument and the UOAIS weights you can still run the reports and the data
self-check, but not the demo.

### Potential Issues of Installation

1. `fatal error: cusparse.h: No such file or directory` when building the FGC extensions

- **Cause**: `torch.utils.cpp_extension` looks for the CUDA headers only under `$CUDA_HOME/include`.
- **Solution**: `scripts/build_fgc_extensions.sh` symlinks them there before compiling. Run it
  through `scripts/install.sh` rather than calling `setup.py` directly.

2. detectron2 or AdelaiDet fails to build

- **Solution**: they must be built against the torch already installed. Follow
  [gist-ailab/uoais](https://github.com/gist-ailab/uoais), which pins the versions that work
  together. `uoais-ft/` here is that repository with a local C++/CUDA fix and this project's
  fine-tuning.

## Dataset

The demo needs no dataset — 28 real RGB-D scenes ship in [`demo_rgbd/`](demo_rgbd/). The
evaluation splits are already in `UnoBench/subset_difficulty/`; only the image data and
`gt_for_nlp.json` are downloaded separately.

- **UnoBench (synthetic)**: [chiencn/vocc_synthetic](https://huggingface.co/datasets/chiencn/vocc_synthetic)
  — the `test_GT_small_1800` split, 1800 cases over 1400 images (~2.3 GB download, ~16 GB extracted)
- **MetaGraspNet-V2 (real)**: `<placeholder: filtered subset link>`

**Download the synthetic split**

```bash
pip install huggingface_hub
hf download chiencn/vocc_synthetic --repo-type dataset --local-dir /tmp/unobench_dl

mkdir -p UnoBench/_extracted
for f in images depth annotations; do
    mkdir -p UnoBench/_extracted/$f
    tar -xzf /tmp/unobench_dl/$f.tar.gz -C UnoBench/_extracted/$f
done
cp /tmp/unobench_dl/meta/gt_for_nlp.json UnoBench/
```

The result is the layout the code expects:

```
vocc-grasp
└── UnoBench
    ├── gt_for_nlp.json
    ├── subset_difficulty/          (already in this repository)
    └── _extracted
        ├── images/image_000003.png
        ├── depth/image_000003.npy
        └── annotations/image_000003.npy
```

UnoBench depth is in **centimetres** and ships no intrinsics; a pinhole is synthesised from a
60° vertical FOV.

## Running demo

One RGB-D frame in, a grasp pose out.

1. **Run the demo**
   ```bash
   conda activate vocc
   python demo.py --case img000071_q1
   ```

   The user asks for the **top pouch**; an **orange** sits on it. The pipeline says clear the
   orange first and returns a pose for it.

   ```
   [2/5] detecting objects  [Gemini]
         5 objects: 1=soap refill pouch, 2=orange, 3=deodorant roll-on, 4=food can, 5=oil filter
   [3/5] amodal segmentation + occlusion geometry  [UOAIS]
         11 instances, 1 occlusion edge
   [4/5] reasoning  [Gemini]
         grasp first: id=2 (orange), chain 1 step(s), confidence 90.0
   [5/5] grasp pose  [FGC-GraspNet]
         gripper: robotiq_2f85, jaws <= 80 mm
         grasp_found=True
   ```

   `output/img000071_q1/` holds the annotated renders, the orbit GIF, `grasp_pose.json`
   (6-DoF, camera frame) and `plan.json` (the removal order).

1. **Reproduce the bundled reference exactly**
   ```bash
   python demo.py --case img000071_q1 --masks provided
   ```

   Object names, badge numbers and the confidence come from the VLM and differ between runs;
   which object gets picked does not. Running against the bundled instance masks instead of
   UOAIS's reproduces [`demo_rgbd/expected/img000071_q1.json`](demo_rgbd/expected/):
   translation `[0.0561, -0.0230, 0.6495]` m, score `0.642`, funnel 262 → 200 → 135 → 61.

1. **Run on your own frame**
   ```bash
   python demo.py --rgb scene.png --depth depth.png --intrinsics k.json --request "the white box"
   ```

**Input RGB-D and instruction (take the top pouch). Output grasp pose:**

<table align="center">
  <tr>
    <td align="center"><img src="assets/pipeline_example.png" width="240px"><br><b>Reasoning</b></td>
    <td align="center"><img src="assets/grasp_orbit.gif" width="240px"><br><b>Point cloud</b></td>
  </tr>
</table>

For more scenes please check the folder `demo_rgbd/`.

## Running on dataset

Both commands run from the shipped edge scores, so neither needs a GPU or an API key.

1. **Synthetic — UnoBench `test_GT_small_1800`**
   ```bash
   python math/report_unobench.py \
       --csv logs/edge_scores_csv_test1800_full/fused_adaptive_w0.5_0.5_m1.csv \
       --tau-edge 0.09
   ```

1. **Real — the 838-case MetaGraspNet-V2 subset**
   ```bash
   python math/report_unobench.py \
       --csv logs/edge_scores_csv_real_subset/fused_adaptive_w0.5_0.5_m1.csv \
       --gt test_GT_subset_hardall_easy300_medium300.json \
       --tau-edge 0.09
   ```

To regenerate those CSVs from raw predictions rather than using the shipped ones, run
`export_scores.py` then `export_fused_weighted.py`.

## Self-check

```bash
python demo_rgbd/verify.py         # data integrity — no GPU, no key, no weights
python demo_rgbd/verify.py --run   # replay the grasp stage — needs a GPU and the FGC weights
```

28/28 scenes, recorded pose recovered within 0.001 mm and 0.044°.

## Configuration

Every threshold the pipeline runs on lives in **[`configs/pipeline.yaml`](configs/pipeline.yaml)**
— detector thresholds, edge-scoring rules, fusion weights, Top-K solver limits, decision
thresholds, grasp filters. `demo.py` loads it; the batch drivers take the same values as
defaults. The reported numbers were produced with that file unchanged.

| | file |
|---|---|
| Pipeline parameters | [`configs/pipeline.yaml`](configs/pipeline.yaml) |
| Calibration coefficients | [`calibration/adaptive_platt_model.json`](calibration/) (VLM), `adaptive_platt_3d_model.json` (3D) |
| Fit report: splits, ECE before/after | [`calibration/README.md`](calibration/README.md) |
| Gripper geometry | [`grasp_viz/gripper_robotiq_2f85.yaml`](grasp_viz/gripper_robotiq_2f85.yaml) |

## Structure

```
demo.py                     RGB-D → grasp pose
run_gemini_uoais_ref*.py    the model: VLM reasoning over a UOAIS geometry table
export_fused_weighted.py    calibrate + fuse → the edge CSVs the math stack reads
math/                       marginalisation, decision policy, reporting
calibration/                frozen Platt parameters and their fit reports
grasp_viz/                  segmentation → point cloud → FGC → renders
configs/pipeline.yaml       every threshold, in one place
demo_rgbd/                  28 real RGB-D scenes + expected results + verify.py
uoais-ft/                   UOAIS, fine-tuned      github.com/gist-ailab/uoais
FreeGrasp_code/             FGC-GraspNet           github.com/luyh20/FGC-GraspNet
```

---

# License

MIT for the code written here. `FreeGrasp_code/` and `uoais-ft/` keep their upstream licences;
`demo_rgbd/` follows MetaGraspNet-V2's terms.
