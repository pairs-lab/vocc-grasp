"""Case selection and per-method target resolution.

All three logs are index-aligned 1:1 with ``gt_synthetic_eval.json``; the case
key is ``img{image_id:06d}_q{query_object}``.  Every method ultimately gives a
pixel in the 1200x1200 image, so the grasp target is resolved uniformly:
predicted point -> instance id via the annotation map.  That keeps the three
methods in one id space instead of trusting each log's own numbering.
"""
import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from . import paths

POINT_RE = re.compile(r"<points?\s+(\d+)\s+(\d+)\s*>([^<]*)")
MEDIUM = "Medium"
HARD = "Hard"
N_PER_DIFFICULTY = 5


def case_key(image_id, query_object):
    return "img%06d_q%d" % (image_id, query_object)


def _read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _first_point(model_output):
    """Return (x, y, label) of the first <points x y>label</points> tag."""
    if not model_output:
        return None
    m = POINT_RE.search(model_output)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3).strip()


@dataclass
class Target:
    """What one method decided to grasp first, for one case."""
    method: str
    point: tuple | None          # (x, y) in pixels, or None if unparseable
    label: str                   # the method's own name for the object
    instance_id: int             # id in the annotation map (0 = background)
    correct: bool                # instance_id is one of the GT top objects
    raw: dict = field(default_factory=dict)


@dataclass
class Case:
    key: str
    index: int
    image_id: int
    query_object: int
    query: str
    difficulty: str
    gt_top_ids: list
    targets: dict                # method -> Target


def _uoais_point(log_dir, key):
    with open(os.path.join(log_dir, key, "result.json")) as fh:
        res = json.load(fh)
    pts = res.get("eval_top_points") or res.get("top_points") or []
    point, label = None, ""
    if pts:
        x, y, label = pts[0][0], pts[0][1], (pts[0][2] if len(pts[0]) > 2 else "")
        point = (int(x), int(y))
    return point, label, res


def load_cases(methods=None, difficulties=(MEDIUM, HARD), n_per=N_PER_DIFFICULTY):
    """Deterministic pick: first ``n_per`` cases per difficulty, sorted by key."""
    methods = list(methods or paths.METHOD_LOGS)
    with open(paths.GT_JSON) as fh:
        gt = json.load(fh)

    uoais_dir = paths.METHOD_LOGS["gemini_uoais_ref"]
    noref = _read_jsonl(os.path.join(paths.METHOD_LOGS["gemini_noref"], "predictions.jsonl"))
    unograsp = _read_jsonl(os.path.join(paths.METHOD_LOGS["unograsp"], "predictions.jsonl"))

    rows = []
    for i, g in enumerate(gt):
        key = case_key(g["image_id"], g["query_object"])
        with open(os.path.join(uoais_dir, key, "result.json")) as fh:
            res = json.load(fh)
        rows.append((key, res["difficulty"], i, g, res))

    picked = []
    for diff in difficulties:
        same = sorted((r for r in rows if r[1] == diff), key=lambda r: r[0])
        picked.extend(same[:n_per])

    cases = []
    for key, diff, i, g, res in picked:
        annot = np.load(paths.annot_path(g["image_id"]))
        gt_top = [int(v) for v in g["top_objects"]]

        def make(method, point, label, raw):
            iid = 0
            if point is not None:
                x, y = point
                if 0 <= y < annot.shape[0] and 0 <= x < annot.shape[1]:
                    iid = int(annot[y, x])
            return Target(method, point, label, iid, iid in gt_top, raw)

        targets = {}
        if "gemini_uoais_ref" in methods:
            p, lb, raw = _uoais_point(uoais_dir, key)
            targets["gemini_uoais_ref"] = make("gemini_uoais_ref", p, lb, raw)
        if "gemini_noref" in methods:
            hit = _first_point(noref[i].get("model_output"))
            targets["gemini_noref"] = make(
                "gemini_noref", hit[:2] if hit else None, hit[2] if hit else "", noref[i])
        if "unograsp" in methods:
            hit = _first_point(unograsp[i].get("model_output"))
            targets["unograsp"] = make(
                "unograsp", hit[:2] if hit else None, hit[2] if hit else "", unograsp[i])

        cases.append(Case(key=key, index=i, image_id=g["image_id"],
                          query_object=g["query_object"], query=res.get("query", ""),
                          difficulty=diff, gt_top_ids=gt_top, targets=targets))
    return cases


# Cases dropped from the real pick, with why.  FGC finds grasps on the target
# but none survive the <=30 degree top-down filter, so all three methods render
# an empty panel -- the next case by key is used instead.
REAL_EXCLUDED = {
    "img000062_q2": "no top-down grasp (raw 47, in_mask 19, collision_free 12, top_down 0)",
}


def _resolve_targets(methods, annot, gt_top, uoais_dir, key, noref_row, uno_row):
    """predicted pixel -> instance id -> Target, uniformly for every method."""
    def make(method, point, label, raw):
        iid = 0
        if point is not None:
            x, y = point
            if 0 <= y < annot.shape[0] and 0 <= x < annot.shape[1]:
                iid = int(annot[y, x])
        return Target(method, point, label, iid, iid > 0 and iid in gt_top, raw)

    targets = {}
    if "gemini_uoais_ref" in methods:
        p, lb, raw = _uoais_point(uoais_dir, key)
        targets["gemini_uoais_ref"] = make("gemini_uoais_ref", p, lb, raw)
    for method, row in (("gemini_noref", noref_row), ("unograsp", uno_row)):
        if method in methods:
            hit = _first_point(row.get("model_output"))
            targets[method] = make(method, hit[:2] if hit else None,
                                   hit[2] if hit else "", row)
    return targets


def load_cases_real(methods=None, picks=(("Medium", 7), ("Hard", 8)),
                    all_correct=True, keys=None):
    """Cases from the real subset GT, index-aligned to the three real logs.

    Unlike the synthetic loader the difficulty is read straight off the GT entry
    (``test_GT_subset_hardall_easy300_medium300.json`` carries it), so a case
    only needs ``result.json`` for the human-readable query.

    With ``all_correct`` the pick is restricted to cases where all three methods
    land on a GT top object -- those show the grasp poses themselves rather than
    a reasoning failure.

    ``keys`` names cases explicitly and bypasses both filters, so asking for one
    case returns that case whatever ``methods`` is set to -- otherwise narrowing
    ``methods`` would silently change which cases pass ``all_correct`` and so
    which ones land in the top-N.
    """
    methods = list(methods or paths.METHOD_LOGS_REAL)
    with open(paths.GT_JSON_REAL) as fh:
        gt = json.load(fh)

    logs = paths.METHOD_LOGS_REAL
    uoais_dir = logs["gemini_uoais_ref"]
    noref = _read_jsonl(os.path.join(logs["gemini_noref"], "predictions.jsonl"))
    unograsp = _read_jsonl(os.path.join(logs["unograsp"], "predictions.jsonl"))
    if not (len(gt) == len(noref) == len(unograsp)):
        raise ValueError("logs are not index-aligned with %s" % paths.GT_JSON_REAL)

    keys = set(keys) if keys else None
    wanted = {d for d, _ in picks}
    built = []
    for i, g in enumerate(gt):
        key = case_key(g["image_id"], g["query_object"])
        if keys is not None:
            if key not in keys:
                continue
        elif g["difficulty"] not in wanted:
            continue
        result_path = os.path.join(uoais_dir, key, "result.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path) as fh:
            res = json.load(fh)

        annot = np.load(paths.annot_path_real(g["image_id"]))
        gt_top = [int(v) for v in g["top_objects"]]
        targets = _resolve_targets(methods, annot, gt_top, uoais_dir, key,
                                   noref[i], unograsp[i])
        if keys is None:
            if key in REAL_EXCLUDED:
                continue
            if all_correct and not all(t.correct for t in targets.values()):
                continue
        built.append(Case(key=key, index=i, image_id=g["image_id"],
                          query_object=g["query_object"], query=res.get("query", ""),
                          difficulty=g["difficulty"], gt_top_ids=gt_top,
                          targets=targets))

    if keys is not None:
        return sorted(built, key=lambda c: c.key)

    cases = []
    for difficulty, n in picks:
        same = sorted((c for c in built if c.difficulty == difficulty),
                      key=lambda c: c.key)
        if len(same) < n:
            raise ValueError("only %d %s cases available, asked for %d"
                             % (len(same), difficulty, n))
        cases.extend(same[:n])
    return cases
