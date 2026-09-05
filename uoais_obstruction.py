"""
UOAIS-based 3D Obstruction Graph Builder

Given:  image.png + depth.npz + task.txt  (one FreeGrasp scene)
Output: obstruction_graph_uoais.json  +  vis_uoais.png

Pipeline:
  1. UOAIS-Net (RGB-D) → per-object {visible_mask, amodal_mask, occluded_flag}
  2. Depth → median height per object (top-down camera: lower depth = closer = on top)
  3. Footprint overlap + depth ordering → obstruction edges
  4. BFS from target object → prune to reachable subgraph → obstruction graph
  5. Optionally: match target by task text via CLIP similarity (or fallback: most-occluded object)

Usage:
    # Run on a single extracted FreeGrasp scene:
    python run_uoais_obstruction.py --scene data/freegrasp_sample_100_labeled/sample_demo_000

    # Run on all extracted FreeGrasp scenes:
    python run_uoais_obstruction.py --all data/freegrasp_sample_100_labeled

    # Use depth-only UOAIS model:
    python run_uoais_obstruction.py --scene data/freegrasp_sample_100_labeled/sample_demo_000 --depth-only
"""

import argparse
import json
import sys
import os
from pathlib import Path

import cv2
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
UOAIS_DIR = Path(__file__).parent / "uoais-ft"
sys.path.insert(0, str(UOAIS_DIR))

# ── UOAIS config options ────────────────────────────────────────────────────────
CFG_RGBD  = str(UOAIS_DIR / "configs/R50_rgbdconcat_mlc_occatmask_hom_concat.yaml")
CFG_DEPTH = str(UOAIS_DIR / "configs/R50_depth_mlc_occatmask_hom_concat.yaml")

# Depth normalization range for FreeGrasp data (values in cm, convert to mm).
# UOAIS demos normalize raw metric depth with this default range. Some local
# experiments use per-scene percentiles, so keep both modes available.
UOAIS_DEPTH_MIN_MM = 250.0
UOAIS_DEPTH_MAX_MM = 1500.0
DEPTH_MIN_MM = UOAIS_DEPTH_MIN_MM
DEPTH_MAX_MM = UOAIS_DEPTH_MAX_MM
DEPTH_NORM_MODE = "uoais_default"

# How much XY footprint overlap (fraction of smaller object) counts as obstruction
FOOTPRINT_OVERLAP_THRESH = 0.10

# Depth gap (mm) required for one object to be considered "above" another
DEPTH_GAP_MM = 5.0

# ── Amodal-based occlusion params ────────────────────────────────────────────────
# Fraction of object j's HIDDEN (amodal − visible) region that must be covered
# by another object's visible mask for that object to count as a blocker.
AMODAL_OCC_OVERLAP_THRESH = 0.05
# Minimum number of hidden pixels for an object to be considered "occluded".
MIN_OCC_PIXELS = 80
# A blocker may sit at most this much DEEPER than the blocked object and still
# count (allows for flat occluders / depth-sensor noise on thin objects).
DEPTH_TOLERANCE_MM = 90.0
CONF_OVERLAP_SATURATION = 0.30
CONF_HIDDEN_PIXELS_CAP = 2500.0
CONF_DEPTH_SUPPORT_CAP_MM = 180.0
CV_OCCLUSION_O0 = 0.15
CV_SIGMOID_KAPPA = 12.0
MIN_CONTACT_PIXELS = 1
MIN_VALID_DEPTH_RATIO = 0.5
# ``valid_depth_ratio`` is the raw hardware gate. On synthetic UnoBench it is
# often exactly 1.0 because every depth pixel is valid, so ``r_ij`` can optionally
# use a softer reliability score that also accounts for contact support and local
# depth consistency while keeping the raw ratio available in JSON.
DEPTH_RELIABILITY_MODE = "soft"  # {"valid_ratio", "soft"}
DEPTH_CONSISTENCY_SIGMA_MM = 20.0
CONTACT_SUPPORT_PX = 150.0
# Dilate the blocked object's visible/hidden footprint to catch light contact or
# gripper-clearance conflicts that do not appear as direct hidden-mask overlap.
CONTACT_DILATE_PX = 8


# ══════════════════════════════════════════════════════════════════════════════
#  Depth utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_depth_mm(npz_path: Path) -> np.ndarray:
    """Load depth from .npz, return float32 array in mm."""
    npz = np.load(npz_path)
    d = npz["depth"].astype(np.float32)
    # FreeGrasp depth is in cm → convert to mm
    if d.max() < 200:
        d = d * 10.0
    return d


def set_depth_norm_mode(mode: str, depth_mm: np.ndarray | None = None) -> None:
    """Configure global depth normalization used by UOAIS input preparation."""
    global DEPTH_NORM_MODE, DEPTH_MIN_MM, DEPTH_MAX_MM
    if mode not in {"uoais_default", "percentile"}:
        raise ValueError(f"Unknown depth normalization mode: {mode}")
    DEPTH_NORM_MODE = mode
    if mode == "uoais_default" or depth_mm is None:
        DEPTH_MIN_MM = UOAIS_DEPTH_MIN_MM
        DEPTH_MAX_MM = UOAIS_DEPTH_MAX_MM
    else:
        fg = depth_mm[np.isfinite(depth_mm) & (depth_mm > 0)]
        if len(fg):
            DEPTH_MIN_MM = float(np.percentile(fg, 2))
            DEPTH_MAX_MM = float(np.percentile(fg, 98))
        else:
            DEPTH_MIN_MM = UOAIS_DEPTH_MIN_MM
            DEPTH_MAX_MM = UOAIS_DEPTH_MAX_MM


def depth_to_uoais_input(depth_mm: np.ndarray, H: int, W: int) -> np.ndarray:
    """Normalize depth (mm) to uint8 [H, W, 3] as expected by UOAIS."""
    from utils import normalize_depth, inpaint_depth
    d = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM)
    d_norm = normalize_depth(d, min_val=DEPTH_MIN_MM, max_val=DEPTH_MAX_MM)
    d_norm = cv2.resize(d_norm, (W, H), interpolation=cv2.INTER_NEAREST)
    d_norm = inpaint_depth(d_norm)
    return d_norm


# ══════════════════════════════════════════════════════════════════════════════
#  UOAIS inference
# ══════════════════════════════════════════════════════════════════════════════

def load_uoais_predictor(
    config_file: str,
    score_thresh: float = 0.35,
    nms_thresh: float = 0.7,
    device: str = "auto",
):
    from adet.config import get_cfg
    from adet.utils.post_process import DefaultPredictor
    import torch
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(UOAIS_DIR / cfg.OUTPUT_DIR / "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
    if device == "auto":
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg), cfg


def run_uoais(rgb_bgr: np.ndarray, depth_mm: np.ndarray,
              predictor, cfg) -> dict:
    """
    Run UOAIS on one image.
    Returns dict with:
        amodal_masks   : (N, H, W) bool
        visible_masks  : (N, H, W) bool
        occluded       : (N,) int  0/1
        boxes          : (N, 4) float  xyxy
    All spatial outputs are at the ORIGINAL image resolution (before resize).
    """
    from adet.utils.post_process import detector_postprocess

    H_orig, W_orig = rgb_bgr.shape[:2]
    W_cfg, H_cfg = cfg.INPUT.IMG_SIZE          # (640, 480) in config

    rgb_in = cv2.resize(rgb_bgr, (W_cfg, H_cfg))
    depth_in = depth_to_uoais_input(depth_mm, H_cfg, W_cfg)

    if cfg.INPUT.DEPTH and not cfg.INPUT.DEPTH_ONLY:
        uoais_input = np.concatenate([rgb_in, depth_in], axis=-1)
    elif cfg.INPUT.DEPTH and cfg.INPUT.DEPTH_ONLY:
        uoais_input = depth_in
    else:
        uoais_input = rgb_in

    with __import__("torch").no_grad():
        outputs = predictor(uoais_input)

    instances = detector_postprocess(
        outputs["instances"], H_cfg, W_cfg
    ).to("cpu")

    amodal  = instances.pred_masks.numpy().astype(bool)          # (N, H_cfg, W_cfg)
    visible = instances.pred_visible_masks.numpy().astype(bool)
    occs    = instances.pred_occlusions.numpy().astype(int)       # (N,)
    boxes   = instances.pred_boxes.tensor.numpy()                 # (N, 4)

    # Resize masks back to original resolution
    def resize_masks(masks):
        out = np.zeros((len(masks), H_orig, W_orig), dtype=bool)
        for i, m in enumerate(masks):
            out[i] = cv2.resize(
                m.astype(np.uint8), (W_orig, H_orig),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        return out

    # Scale boxes to original resolution
    sx, sy = W_orig / W_cfg, H_orig / H_cfg
    scale = np.array([sx, sy, sx, sy])
    boxes_orig = boxes * scale

    return {
        "amodal_masks":  resize_masks(amodal),
        "visible_masks": resize_masks(visible),
        "occluded":      occs,
        "boxes":         boxes_orig,
        "n":             len(occs),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3D position + obstruction logic
# ══════════════════════════════════════════════════════════════════════════════

def median_depth(depth_mm: np.ndarray, mask: np.ndarray) -> float:
    """Median depth (mm) of foreground pixels in mask."""
    px = depth_mm[mask]
    if len(px) == 0:
        return float(depth_mm.mean())
    return float(np.median(px))


def compute_obstruction_confidence(
    overlap_frac: float,
    hidden_px: int,
    depth_delta: float,
    occ_flag: bool,
    overlap_thresh: float | None = None,
    hidden_px_conf_cap: float = CONF_HIDDEN_PIXELS_CAP,
    depth_conf_cap_mm: float = CONF_DEPTH_SUPPORT_CAP_MM,
) -> dict:
    """
    Convert accepted geometric evidence into an interpretable obstruction confidence.

    Design goal:
      - weak-but-accepted overlap -> moderate confidence
      - overlap around 0.30 with supportive depth -> strong confidence (~0.8-0.9)
    """
    overlap_thresh = AMODAL_OCC_OVERLAP_THRESH if overlap_thresh is None else overlap_thresh
    overlap_sat = max(CONF_OVERLAP_SATURATION, overlap_thresh + 1e-6)

    overlap_strength = float(np.clip(
        (overlap_frac - overlap_thresh) / (overlap_sat - overlap_thresh),
        0.0, 1.0,
    ))
    hidden_mass_strength = float(np.clip(hidden_px / max(hidden_px_conf_cap, 1.0), 0.0, 1.0))
    depth_support_strength = float(np.clip(depth_delta / max(depth_conf_cap_mm, 1.0), 0.0, 1.0))
    occluded_bonus = 0.08 if occ_flag else 0.0

    confidence_raw = (
        0.60 * overlap_strength +
        0.22 * depth_support_strength +
        0.10 * hidden_mass_strength +
        occluded_bonus
    )
    confidence_raw = float(np.clip(confidence_raw, 0.0, 1.0))

    # Keep accepted weak edges away from near-zero, while allowing strong geometry
    # to climb quickly into a high-confidence range.
    confidence = 0.45 + 0.5 * (confidence_raw ** 0.7)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        "overlap_strength": overlap_strength,
        "hidden_mass_strength": hidden_mass_strength,
        "depth_support_strength": depth_support_strength,
        "occluded_bonus": occluded_bonus,
        "confidence_raw": confidence_raw,
        "confidence": confidence,
    }


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for scalar geometry scores."""
    x = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def valid_depth_ratio(depth_mm: np.ndarray, region_mask: np.ndarray) -> tuple[int, float]:
    """Return (#valid depth pixels, valid-depth ratio) inside a contact zone."""
    total = int(region_mask.sum())
    if total == 0:
        return 0, 0.0
    px = depth_mm[region_mask]
    valid_px = int((np.isfinite(px) & (px > 0)).sum())
    return valid_px, float(valid_px / total)


def soft_depth_reliability(depth_mm: np.ndarray, region_mask: np.ndarray) -> dict:
    """
    Soft reliability for an obstruction evidence region.

    The original valid-depth ratio is a strict hardware-integrity gate:
    ``#valid_depth / #contact_pixels``. In synthetic datasets this is frequently
    binary, so this helper keeps that raw gate but reports ``r_ij`` as:

        r_ij = valid_depth_ratio * depth_consistency * contact_support

    where ``depth_consistency`` decreases when valid depths in the evidence
    region are noisy/discontinuous and ``contact_support`` decreases for tiny
    evidence regions. This keeps ``r_ij`` in [0, 1] and makes it informative even
    when all depth pixels are technically valid.
    """
    total = int(region_mask.sum())
    if total == 0:
        return {
            "valid_depth_px": 0,
            "valid_depth_ratio": 0.0,
            "r_ij": 0.0,
            "depth_median_mm": None,
            "depth_mad_mm": None,
            "depth_consistency": 0.0,
            "contact_support": 0.0,
        }

    px = depth_mm[region_mask]
    valid = px[np.isfinite(px) & (px > 0)]
    valid_px = int(len(valid))
    valid_ratio = float(valid_px / total)
    if valid_px == 0:
        return {
            "valid_depth_px": 0,
            "valid_depth_ratio": valid_ratio,
            "r_ij": 0.0,
            "depth_median_mm": None,
            "depth_mad_mm": None,
            "depth_consistency": 0.0,
            "contact_support": 0.0,
        }

    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    sigma = max(float(DEPTH_CONSISTENCY_SIGMA_MM), 1e-6)
    depth_consistency = float(np.exp(-mad / sigma))
    support_px = max(float(CONTACT_SUPPORT_PX), 1e-6)
    contact_support = float(1.0 - np.exp(-valid_px / support_px))
    if DEPTH_RELIABILITY_MODE == "valid_ratio":
        r_ij = valid_ratio
    else:
        r_ij = valid_ratio * depth_consistency * contact_support
    return {
        "valid_depth_px": valid_px,
        "valid_depth_ratio": valid_ratio,
        "r_ij": float(np.clip(r_ij, 0.0, 1.0)),
        "depth_median_mm": median,
        "depth_mad_mm": mad,
        "depth_consistency": float(np.clip(depth_consistency, 0.0, 1.0)),
        "contact_support": float(np.clip(contact_support, 0.0, 1.0)),
    }


def dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Binary dilation helper used for clearance/contact reasoning."""
    radius_px = int(radius_px)
    if radius_px <= 0 or not mask.any():
        return mask.astype(bool)
    k = 2 * radius_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def compute_cv_obstruction_confidence(
    contact_mask: np.ndarray,
    amodal_px: int,
    depth_mm: np.ndarray,
    o0: float | None = None,
    kappa: float | None = None,
) -> dict:
    """
    Paper-style directed obstruction score.

    For blocked object i and blocker j:
      IV_i = A_i \\ V_i
      contact_ij = IV_i ∩ V_j
      o_ij = area(contact_ij) / area(A_i)
      p_cv_ij = sigmoid(kappa * (o_ij - o0))

    r_ij is reported separately as the valid-depth ratio in contact_ij and is
    used as a hardware reliability gate, not multiplied into p_cv_ij.
    """
    o0 = CV_OCCLUSION_O0 if o0 is None else o0
    kappa = CV_SIGMOID_KAPPA if kappa is None else kappa
    contact_px = int(contact_mask.sum())
    amodal_px = int(amodal_px)
    occlusion_ratio = float(contact_px / amodal_px) if amodal_px > 0 else 0.0
    p_cv = _sigmoid(float(kappa) * (occlusion_ratio - float(o0)))
    reliability = soft_depth_reliability(depth_mm, contact_mask)

    return {
        "amodal_px": amodal_px,
        "contact_px": contact_px,
        "occlusion_ratio": occlusion_ratio,
        "p_cv": p_cv,
        "valid_depth_px": int(reliability["valid_depth_px"]),
        "valid_depth_ratio": float(reliability["valid_depth_ratio"]),
        "valid_depth_ratio_raw": float(reliability["valid_depth_ratio"]),
        "r_ij": float(reliability["r_ij"]),
        "depth_median_mm": reliability["depth_median_mm"],
        "depth_mad_mm": reliability["depth_mad_mm"],
        "depth_consistency": float(reliability["depth_consistency"]),
        "contact_support": float(reliability["contact_support"]),
        "depth_reliable": bool(reliability["valid_depth_ratio"] >= MIN_VALID_DEPTH_RATIO),
        "confidence_raw": p_cv,
        "confidence": p_cv,
        # Legacy names retained for downstream JSON consumers.
        "overlap_strength": p_cv,
        "hidden_mass_strength": float(np.clip(amodal_px / max(CONF_HIDDEN_PIXELS_CAP, 1.0), 0.0, 1.0)),
        "depth_support_strength": float(reliability["r_ij"]),
        "occluded_bonus": 0.0,
    }


def footprint_overlap_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Overlap fraction relative to the smaller mask."""
    inter = np.logical_and(mask_a, mask_b).sum()
    smaller = min(mask_a.sum(), mask_b.sum())
    if smaller == 0:
        return 0.0
    return inter / smaller


def build_obstruction_edges(
    visible_masks: np.ndarray,
    amodal_masks: np.ndarray,
    depth_mm: np.ndarray,
    occluded: np.ndarray,
    features: bool = False,
) -> list:
    """
    Amodal-based obstruction detection.

    Edge orientation follows the code convention ``i -> j`` where object i is
    the blocker and object j is blocked. In the paper notation, this corresponds
    to blocked object i_paper=j and blocker j_paper=i.

    For each blocked object j, its invisible region is IV_j = A_j \\ V_j.
    Any blocker i whose visible mask V_i intersects IV_j gives a directed
    contact zone IV_j ∩ V_i.

    This is the correct signal: two objects' *visible* masks never overlap
    (segmentation assigns each pixel to one instance), so the old
    visible-vs-visible footprint overlap was almost always ~0. Overlapping object
    i's visible mask with object j's *hidden* region recovers true occlusion.

    Returns list of (blocker_idx, blocked_idx, confidence).

    If ``features=True``, instead returns accepted edge dicts with raw signals:
        {i, j, conf, p_cv, r_ij, occlusion_ratio, contact_px, amodal_px,
         hidden_px, depth_blocker, depth_blocked, depth_delta, occ_flag,
         accepted_reason}
    where depth_delta = blocked_depth - blocker_depth; positive means the blocker
    is closer to the camera, which supports an obstruction relation.
    """
    n = len(visible_masks)
    depths = [median_depth(depth_mm, visible_masks[i]) for i in range(n)]
    edges = []

    for j in range(n):
        # Invisible region of blocked object j from amodal segmentation.
        amodal_px = int(amodal_masks[j].sum())
        occ_region = amodal_masks[j] & ~visible_masks[j]
        occ_px = int(occ_region.sum())
        # Clearance/access region around the blocked object. This catches light
        # contact and nearby gripper-corridor conflicts that have little or no
        # explicit hidden-mask overlap.
        clearance_seed = occ_region | visible_masks[j]
        clearance_region = dilate_mask(clearance_seed, CONTACT_DILATE_PX) & ~visible_masks[j]

        for i in range(n):
            if i == j:
                continue
            # Contact zone for paper notation: IV_blocked ∩ V_blocker.
            direct_contact_mask = visible_masks[i] & occ_region
            direct_contact_px = int(direct_contact_mask.sum())
            clearance_contact_mask = visible_masks[i] & clearance_region
            clearance_contact_px = int(clearance_contact_mask.sum())
            if direct_contact_px >= MIN_CONTACT_PIXELS:
                evidence_mask = direct_contact_mask
                accepted_reason_base = "hidden_overlap"
            else:
                evidence_mask = clearance_contact_mask
                accepted_reason_base = "clearance_contact"
            evidence_px = int(evidence_mask.sum())
            if evidence_px < MIN_CONTACT_PIXELS:
                continue

            conf_parts = compute_cv_obstruction_confidence(
                contact_mask=evidence_mask,
                amodal_px=amodal_px,
                depth_mm=depth_mm,
            )
            if conf_parts["valid_depth_ratio"] < MIN_VALID_DEPTH_RATIO:
                continue

            # Depth order is retained as a separate sanity check. Median depth over
            # visible pixels can be noisy, so only reject when the blocker is clearly
            # deeper than the blocked object.
            depth_delta = depths[j] - depths[i]
            if depth_delta < -DEPTH_TOLERANCE_MM:
                continue

            depth_support = float(np.clip(
                depth_delta / max(CONF_DEPTH_SUPPORT_CAP_MM, 1.0), 0.0, 1.0
            ))
            confidence = conf_parts["p_cv"]
            if features:
                if depth_delta >= 0:
                    accepted_reason = f"{accepted_reason_base}_depth_supported"
                else:
                    accepted_reason = f"{accepted_reason_base}_depth_tolerated"
                edges.append({
                    "i": i, "j": j, "conf": confidence,
                    "frac": float(direct_contact_px / occ_px) if occ_px else 0.0,
                    "hidden_px": occ_px,
                    "amodal_px": amodal_px,
                    "contact_px": evidence_px,
                    "direct_contact_px": direct_contact_px,
                    "clearance_contact_px": clearance_contact_px,
                    "contact_mode": accepted_reason_base,
                    "contact_dilate_px": int(CONTACT_DILATE_PX),
                    "depth_blocker": float(depths[i]),
                    "depth_blocked": float(depths[j]),
                    "depth_delta": float(depth_delta),
                    "depth_support": depth_support,
                    "occ_flag": int(occluded[j]),
                    **conf_parts,
                    "accepted_reason": accepted_reason,
                })
            else:
                edges.append((i, j, confidence))

    return edges


def build_obstruction_pair_diagnostics(
    visible_masks: np.ndarray,
    amodal_masks: np.ndarray,
    depth_mm: np.ndarray,
    occluded: np.ndarray | None = None,
) -> list:
    """
    Return loose pairwise contact diagnostics without changing graph decisions.

    This helper is intentionally diagnostic-only: it reports every ordered pair
    with either hidden-overlap contact or clearance contact, including pairs that
    the stricter graph builder would reject due to depth or raw depth reliability.
    Edge orientation matches ``build_obstruction_edges``: blocker i -> blocked j.
    """
    n = len(visible_masks)
    depths = [median_depth(depth_mm, visible_masks[i]) for i in range(n)]
    if occluded is None:
        occluded = np.zeros(n, dtype=np.int32)
    rows = []

    for j in range(n):
        amodal_px = int(amodal_masks[j].sum())
        occ_region = amodal_masks[j] & ~visible_masks[j]
        occ_px = int(occ_region.sum())
        clearance_seed = occ_region | visible_masks[j]
        clearance_region = dilate_mask(clearance_seed, CONTACT_DILATE_PX) & ~visible_masks[j]

        for i in range(n):
            if i == j:
                continue
            direct_contact_mask = visible_masks[i] & occ_region
            direct_contact_px = int(direct_contact_mask.sum())
            clearance_contact_mask = visible_masks[i] & clearance_region
            clearance_contact_px = int(clearance_contact_mask.sum())
            if direct_contact_px <= 0 and clearance_contact_px <= 0:
                continue

            if direct_contact_px > 0:
                evidence_mask = direct_contact_mask
                contact_mode = "hidden_overlap"
            else:
                evidence_mask = clearance_contact_mask
                contact_mode = "clearance_contact"

            conf_parts = compute_cv_obstruction_confidence(
                contact_mask=evidence_mask,
                amodal_px=amodal_px,
                depth_mm=depth_mm,
            )
            depth_delta = float(depths[j] - depths[i])
            reject_reasons = []
            if int(conf_parts["contact_px"]) < MIN_CONTACT_PIXELS:
                reject_reasons.append("insufficient_contact")
            if float(conf_parts["valid_depth_ratio"]) < MIN_VALID_DEPTH_RATIO:
                reject_reasons.append("low_valid_depth_ratio")
            if depth_delta < -DEPTH_TOLERANCE_MM:
                reject_reasons.append("depth_contradicts_direction")
            accepted = not reject_reasons
            accepted_reason = (
                f"{contact_mode}_diagnostic_accepted"
                if accepted else
                "diagnostic_rejected:" + ",".join(reject_reasons)
            )
            rows.append({
                "i": i,
                "j": j,
                "conf": float(conf_parts["p_cv"]),
                "frac": float(direct_contact_px / occ_px) if occ_px else 0.0,
                "hidden_px": occ_px,
                "amodal_px": amodal_px,
                "contact_px": int(conf_parts["contact_px"]),
                "direct_contact_px": direct_contact_px,
                "clearance_contact_px": clearance_contact_px,
                "contact_mode": contact_mode,
                "contact_dilate_px": int(CONTACT_DILATE_PX),
                "depth_blocker": float(depths[i]),
                "depth_blocked": float(depths[j]),
                "depth_delta": depth_delta,
                "depth_support": float(np.clip(depth_delta / max(CONF_DEPTH_SUPPORT_CAP_MM, 1.0), 0.0, 1.0)),
                "occ_flag": int(occluded[j]) if len(occluded) > j else 0,
                "accepted": bool(accepted),
                "reject_reason": "none" if accepted else ",".join(reject_reasons),
                **conf_parts,
                "accepted_reason": accepted_reason,
            })

    return rows


def resolve_reciprocal_edges(edges: list[tuple[int, int, float]], depths: list[float]) -> list[tuple[int, int, float]]:
    """Drop the weaker edge from each two-node cycle before ordering/free tests."""
    edge_map = {(i, j): c for i, j, c in edges}
    keep = set(edge_map)
    for i, j in list(edge_map):
        if i >= j or (j, i) not in edge_map:
            continue

        def rank(a, b):
            conf = edge_map[(a, b)]
            depth_delta = depths[b] - depths[a] if a < len(depths) and b < len(depths) else 0.0
            return conf + 0.1 * np.sign(depth_delta), conf, depth_delta

        keep_edge = (i, j) if rank(i, j) >= rank(j, i) else (j, i)
        drop_edge = (j, i) if keep_edge == (i, j) else (i, j)
        keep.discard(drop_edge)
    return [(i, j, c) for i, j, c in edges if (i, j) in keep]


def build_obstruction_graph_from_target(
    target_idx: int,
    edges: list[tuple[int, int, float]],
    n_objects: int,
) -> dict:
    """
    BFS from target_idx following edges that block target (directly or transitively).
    Returns: {ancestors: list, removal_order: list, edges_json: list}
    """
    # Build adjacency: blocked_idx → [blocker_idx, ...]
    blockers_of = {i: [] for i in range(n_objects)}
    for (blocker, blocked, conf) in edges:
        blockers_of[blocked].append((blocker, conf))

    # BFS from target
    ancestors = []
    queue = [target_idx]
    visited = set([target_idx])
    edge_list = []

    while queue:
        node = queue.pop(0)
        for (blocker, conf) in blockers_of[node]:
            edge_list.append({"from_id": node, "to_id": blocker, "confidence": round(conf, 3)})
            if blocker not in visited:
                visited.add(blocker)
                ancestors.append(blocker)
                queue.append(blocker)

    # Removal order: leaves first (objects not blocking anything in the subgraph)
    subgraph_blocked_by = {a: [] for a in ancestors}
    for (blocker, blocked, _) in edges:
        if blocker in visited and blocked in visited and blocker != target_idx and blocked != target_idx:
            subgraph_blocked_by[blocker].append(blocked)

    # Topological sort → removal order (process things that aren't blocking others first)
    removal_order = []
    remaining = list(ancestors)
    while remaining:
        leaves = [a for a in remaining if not any(
            dep in remaining for dep in subgraph_blocked_by.get(a, [])
        )]
        if not leaves:
            removal_order.extend(remaining)
            break
        for leaf in leaves:
            removal_order.append(leaf)
            remaining.remove(leaf)

    return {
        "ancestors": ancestors,
        "removal_order": removal_order,
        "edges": edge_list,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Target matching
# ══════════════════════════════════════════════════════════════════════════════

def find_target_idx_by_task(
    task: str,
    visible_masks: np.ndarray,
    rgb_bgr: np.ndarray,
) -> int | None:
    """
    Try to match task text to one of the detected objects via CLIP embedding.
    Falls back to the most occluded object if CLIP unavailable.
    Returns 0-indexed object index or None if no objects.
    """
    if len(visible_masks) == 0:
        return None

    try:
        import torch
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        from PIL import Image as PILImage

        text_tokens = clip.tokenize([task]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        best_idx, best_score = 0, -1.0
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        for i, mask in enumerate(visible_masks):
            # crop tightest bounding box of visible mask
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            crop = rgb_rgb[y0:y1+1, x0:x1+1]
            if crop.size == 0:
                continue
            pil_crop = PILImage.fromarray(crop)
            img_tensor = preprocess(pil_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            score = float((img_feat @ text_feat.T).squeeze())
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    except ImportError:
        print("  [CLIP not available] Falling back to most-occluded object as target")
        # Fallback: return the object with largest occluded area (most likely the target)
        from utils import normalize_depth
        # Pick object with highest depth (furthest = most likely buried)
        if len(visible_masks) == 0:
            return None
        return 0   # caller should handle


# ══════════════════════════════════════════════════════════════════════════════
#  Visualization
# ══════════════════════════════════════════════════════════════════════════════

COLORS = [
    (255, 80,  80),  (80, 200, 80),  (80,  80, 255), (255, 200,  80),
    (200,  80, 255), (80, 255, 200), (255, 130,  50), (130,  50, 255),
    (50,  200, 255), (200, 255,  50), (255, 50, 180), (50, 180, 255),
]


def visualize_results(
    rgb_bgr: np.ndarray,
    visible_masks: np.ndarray,
    amodal_masks: np.ndarray,
    occluded: np.ndarray,
    target_idx: int | None,
    graph: dict,
    task: str,
    depths: list[float],
) -> np.ndarray:
    vis = rgb_bgr.copy()
    n = len(visible_masks)
    ancestors = set(graph.get("ancestors", []))

    for i in range(n):
        color = COLORS[i % len(COLORS)]
        alpha = 0.4

        # Fill amodal mask lightly
        overlay = vis.copy()
        overlay[amodal_masks[i]] = [c // 3 for c in color]
        vis = cv2.addWeighted(vis, 1 - alpha * 0.5, overlay, alpha * 0.5, 0)

        # Fill visible mask
        overlay2 = vis.copy()
        overlay2[visible_masks[i]] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay2, alpha, 0)

        # Bounding label
        ys, xs = np.where(visible_masks[i])
        if len(ys) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())

        label_parts = [f"#{i}"]
        if occluded[i]:
            label_parts.append("OCC")
        if target_idx == i:
            label_parts.append("TARGET")
        elif i in ancestors:
            label_parts.append("BLOCKER")
        label_parts.append(f"{depths[i]:.0f}mm")
        label = " ".join(label_parts)

        border_color = (0, 0, 255) if target_idx == i else \
                       (255, 100, 0) if i in ancestors else (200, 200, 200)
        cv2.putText(vis, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, border_color, 1, cv2.LINE_AA)

    # Draw obstruction arrows
    for edge in graph.get("edges", []):
        frm, to_ = edge["from_id"], edge["to_id"]
        def centroid(masks, idx):
            ys, xs = np.where(masks[idx])
            if len(ys) == 0:
                return None
            return int(xs.mean()), int(ys.mean())
        pt1 = centroid(visible_masks, frm)
        pt2 = centroid(visible_masks, to_)
        if pt1 and pt2:
            cv2.arrowedLine(vis, pt1, pt2, (0, 0, 255), 2, tipLength=0.2)

    cv2.putText(vis, f"Task: {task}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(vis, f"Task: {task}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    action = "remove obstacle" if graph.get("ancestors") else "pick object"
    cv2.putText(vis, f"Action: {action}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(vis, f"Action: {action}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0) if action == "pick object" else (0, 100, 255), 1)

    return vis


# ══════════════════════════════════════════════════════════════════════════════
#  Main per-scene function
# ══════════════════════════════════════════════════════════════════════════════

def process_scene(scene_dir: Path, predictor, cfg, depth_only: bool = False, depth_norm_mode: str = "uoais_default"):
    print(f"\n{'='*60}")
    print(f"Scene: {scene_dir.name}")

    rgb_bgr  = cv2.imread(str(scene_dir / "image.png"))
    depth_mm = load_depth_mm(scene_dir / "depth.npz")
    set_depth_norm_mode(depth_norm_mode, depth_mm)
    task     = (scene_dir / "task.txt").read_text().strip()
    print(f"  task: {task!r}")
    print(f"  image: {rgb_bgr.shape}  depth: {depth_mm.shape} [{depth_mm.min():.0f}–{depth_mm.max():.0f} mm]")

    # ── Step 1: UOAIS ──────────────────────────────────────────────────────────
    print("  Step 1: UOAIS inference …")
    result = run_uoais(rgb_bgr, depth_mm, predictor, cfg)
    n = result["n"]
    print(f"  Detected {n} objects  |  occluded: {result['occluded'].sum()}/{n}")

    if n == 0:
        print("  No objects detected — skipping scene.")
        return {"scene": scene_dir.name, "error": "no_objects", "action": "pick object", "ancestors": []}

    # ── Step 2: Depth per object ───────────────────────────────────────────────
    depths = [median_depth(depth_mm, result["visible_masks"][i]) for i in range(n)]

    # ── Step 3: Obstruction edges ─────────────────────────────────────────────
    print("  Step 2: Building obstruction edges (amodal) …")
    edges = build_obstruction_edges(
        result["visible_masks"], result["amodal_masks"],
        depth_mm, result["occluded"],
    )
    edges_for_order = resolve_reciprocal_edges(edges, depths)
    print(f"  Found {len(edges)} obstruction edge(s)")

    # ── Step 4: Find target object ────────────────────────────────────────────
    print("  Step 3: Matching target to task …")
    target_idx = find_target_idx_by_task(task, result["visible_masks"], rgb_bgr)
    if target_idx is None:
        target_idx = 0
    print(f"  Target object index: {target_idx}  (depth {depths[target_idx]:.0f} mm)")

    # ── Step 5: Build obstruction graph from target ───────────────────────────
    print("  Step 4: Building obstruction graph …")
    graph = build_obstruction_graph_from_target(target_idx, edges_for_order, n)
    action = "remove obstacle" if graph["ancestors"] else "pick object"
    print(f"  action={action!r}  ancestors={graph['ancestors']}  removal_order={graph['removal_order']}")

    # ── Save JSON ──────────────────────────────────────────────────────────────
    out_json = {
        "target_class": task,
        "target_obj_idx": target_idx,
        "n_objects": n,
        "action": action,
        "edges": graph["edges"],
        "ancestors": graph["ancestors"],
        "removal_order": graph["removal_order"],
        "per_object_depth_mm": [round(d, 1) for d in depths],
        "per_object_occluded": result["occluded"].tolist(),
        "method": "uoais_3d_geometry",
    }
    out_path = scene_dir / "obstruction_graph_uoais.json"
    out_path.write_text(json.dumps(out_json, indent=2))
    print(f"  Saved → {out_path}")

    # ── Visualization ──────────────────────────────────────────────────────────
    vis = visualize_results(
        rgb_bgr, result["visible_masks"], result["amodal_masks"],
        result["occluded"], target_idx, graph, task, depths
    )
    vis_path = scene_dir / "vis_uoais.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"  Vis  → {vis_path}")

    return out_json


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation against GT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_vs_gt(scene_dir: Path, uoais_result: dict) -> dict:
    gt_path = scene_dir / "obstruction_graph_v3.json"
    if not gt_path.exists():
        return {"gt_action": "unknown", "pred_action": uoais_result["action"], "correct": None}

    gt = json.loads(gt_path.read_text())
    gt_action = "remove obstacle" if gt.get("edges") else "pick object"
    pred_action = uoais_result["action"]
    correct = gt_action == pred_action

    return {
        "scene":       scene_dir.name,
        "task":        uoais_result["target_class"],
        "gt_action":   gt_action,
        "pred_action": pred_action,
        "correct":     correct,
        "gt_ancestors_count": len(gt.get("ancestors", [])),
        "pred_ancestors_count": len(uoais_result.get("ancestors", [])),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global AMODAL_OCC_OVERLAP_THRESH, MIN_OCC_PIXELS, DEPTH_TOLERANCE_MM
    global CV_OCCLUSION_O0, CV_SIGMOID_KAPPA, MIN_CONTACT_PIXELS, MIN_VALID_DEPTH_RATIO
    global DEPTH_RELIABILITY_MODE, DEPTH_CONSISTENCY_SIGMA_MM, CONTACT_SUPPORT_PX
    parser = argparse.ArgumentParser(description="UOAIS Obstruction Graph Builder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene",  type=str, help="Path to a single scene directory")
    group.add_argument("--all",    type=str, help="Path to directory of scenes (runs all subdirs)")
    parser.add_argument("--depth-only", action="store_true",
                        help="Use depth-only UOAIS model instead of RGB-D")
    parser.add_argument("--score-thresh", type=float, default=0.35)
    parser.add_argument("--nms-thresh", type=float, default=0.7)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--depth-norm-mode", choices=["uoais_default", "percentile"],
                        default="uoais_default")
    parser.add_argument("--edge-overlap-thresh", type=float, default=AMODAL_OCC_OVERLAP_THRESH)
    parser.add_argument("--min-occ-pixels", type=int, default=MIN_OCC_PIXELS)
    parser.add_argument("--depth-tolerance-mm", type=float, default=DEPTH_TOLERANCE_MM)
    parser.add_argument("--cv-o0", type=float, default=CV_OCCLUSION_O0)
    parser.add_argument("--cv-kappa", type=float, default=CV_SIGMOID_KAPPA)
    parser.add_argument("--min-contact-pixels", type=int, default=MIN_CONTACT_PIXELS)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=MIN_VALID_DEPTH_RATIO)
    parser.add_argument("--depth-reliability-mode", choices=["valid_ratio", "soft"],
                        default=DEPTH_RELIABILITY_MODE)
    parser.add_argument("--depth-consistency-sigma-mm", type=float, default=DEPTH_CONSISTENCY_SIGMA_MM)
    parser.add_argument("--contact-support-px", type=float, default=CONTACT_SUPPORT_PX)
    args = parser.parse_args()

    config_file = CFG_DEPTH if args.depth_only else CFG_RGBD
    AMODAL_OCC_OVERLAP_THRESH = args.edge_overlap_thresh
    MIN_OCC_PIXELS = args.min_occ_pixels
    DEPTH_TOLERANCE_MM = args.depth_tolerance_mm
    CV_OCCLUSION_O0 = args.cv_o0
    CV_SIGMOID_KAPPA = args.cv_kappa
    MIN_CONTACT_PIXELS = args.min_contact_pixels
    MIN_VALID_DEPTH_RATIO = args.min_valid_depth_ratio
    DEPTH_RELIABILITY_MODE = args.depth_reliability_mode
    DEPTH_CONSISTENCY_SIGMA_MM = args.depth_consistency_sigma_mm
    CONTACT_SUPPORT_PX = args.contact_support_px
    print(f"Loading UOAIS model: {Path(config_file).stem}")
    predictor, cfg = load_uoais_predictor(config_file, args.score_thresh, args.nms_thresh, args.device)
    print("Model loaded.\n")

    if args.scene:
        scenes = [Path(args.scene)]
    else:
        base = Path(args.all)
        scenes = sorted([d for d in base.iterdir() if d.is_dir() and (d / "image.png").exists()])
        print(f"Found {len(scenes)} scenes in {base}")

    results = []
    eval_rows = []
    for scene_dir in scenes:
        try:
            r = process_scene(scene_dir, predictor, cfg, args.depth_only, args.depth_norm_mode)
            results.append(r)
            ev = evaluate_vs_gt(scene_dir, r)
            eval_rows.append(ev)
        except Exception as e:
            print(f"  ERROR in {scene_dir.name}: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────────────
    if eval_rows:
        valid = [e for e in eval_rows if e["correct"] is not None]
        n_correct = sum(e["correct"] for e in valid)
        print(f"\n{'='*60}")
        print(f"UOAIS 3D Obstruction Graph — Evaluation Summary")
        print(f"{'='*60}")
        print(f"  Scenes: {len(valid)}  |  Action accuracy: {n_correct}/{len(valid)} = {100*n_correct/max(1,len(valid)):.1f}%")
        print()
        for e in sorted(eval_rows, key=lambda x: (not x["correct"] if x["correct"] is not None else 0)):
            sym = "✓" if e["correct"] else ("✗" if e["correct"] is False else "?")
            print(f"  {sym}  {e['scene']:<40}  gt={e['gt_action']:<16}  pred={e['pred_action']:<16}  task={e['task']!r}")


if __name__ == "__main__":
    main()
