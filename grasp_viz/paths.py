"""Filesystem layout. Everything is derived from the package root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS = os.path.join(ROOT, "logs")

SYNTHETIC = "synthetic"
REAL = "real"
UNOBENCH = os.path.join(ROOT, "UnoBench", "_extracted")

IMAGES = os.path.join(UNOBENCH, "images")
DEPTHS = os.path.join(UNOBENCH, "depth")
ANNOTS = os.path.join(UNOBENCH, "annotations")

# reasoning logs -> the three methods being compared
METHOD_LOGS = {
    "gemini_uoais_ref": os.path.join(LOGS, "gemini_uoais_ref_test_1800"),
    "gemini_noref": os.path.join(LOGS, "gemini_noref_test_1800"),
    "unograsp": os.path.join(LOGS, "unograsp_test_1800"),
}

# the ground truth list every log is index-aligned to
GT_JSON = os.path.join(METHOD_LOGS["gemini_uoais_ref"], "gt_synthetic_eval.json")

# outputs live outside logs/, one set per grasp backend
OUT_DIRS = {m: os.path.join(ROOT, "grasp_viz_" + m) for m in METHOD_LOGS}
COMPARE_DIR = os.path.join(ROOT, "grasp_viz_compare")



def out_dirs(backend="fgc", dataset=SYNTHETIC):
    return OUT_DIRS_REAL if dataset == REAL else OUT_DIRS


def compare_dir(backend="fgc", dataset=SYNTHETIC):
    return COMPARE_DIR_REAL if dataset == REAL else COMPARE_DIR



def image_path(image_id):
    return os.path.join(IMAGES, "image_%06d.png" % image_id)


def depth_path(image_id):
    return os.path.join(DEPTHS, "image_%06d.npy" % image_id)


def annot_path(image_id):
    return os.path.join(ANNOTS, "image_%06d.npy" % image_id)


FGC_ROOT = os.path.join(ROOT, "FreeGrasp_code", "models", "FGC_graspnet")
FGC_CHECKPOINT = os.path.join(ROOT, "FreeGrasp_code", "logs", "checkpoint_fgc.tar")

# ---------------------------------------------------------------- real dataset
# MetaGraspNetV2 real: the 1200x1200 RGB/mask files here are a centred crop of
# the 1944x1200 captures under data_ifl_*/, which is where depth and the real
# intrinsics still live.  See real_data.py for how the two are stitched.
IMAGES_REAL = os.path.join(ROOT, "images")
MASKS_REAL = os.path.join(ROOT, "masks_npy_real_crop")
MAPPING_JSON = os.path.join(ROOT, "real_world_mapping_fixed.json")
DATA_IFL_GLOB = os.path.join(ROOT, "data_ifl_*", "mnt", "data1", "data_ifl_real")
GT_JSON_REAL = os.path.join(ROOT, "test_GT_subset_hardall_easy300_medium300.json")

METHOD_LOGS_REAL = {
    "gemini_uoais_ref": os.path.join(LOGS, "gemini_uoais_ref_real_subset"),
    "gemini_noref": os.path.join(LOGS, "gemini_noref_real_subset"),
    "unograsp": os.path.join(LOGS, "unograsp_subset_hardall_e300_m300"),
}

OUT_ROOT_REAL = os.path.join(ROOT, "grasp_viz_real")
OUT_DIRS_REAL = {m: os.path.join(OUT_ROOT_REAL, m) for m in METHOD_LOGS_REAL}
COMPARE_DIR_REAL = os.path.join(OUT_ROOT_REAL, "_compare")




def method_logs(dataset=SYNTHETIC):
    return METHOD_LOGS_REAL if dataset == REAL else METHOD_LOGS


def image_path_real(image_id):
    return os.path.join(IMAGES_REAL, "image_%06d.png" % image_id)


def annot_path_real(image_id):
    return os.path.join(MASKS_REAL, "image_%06d.npy" % image_id)
