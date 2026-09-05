"""Load the deployment gripper's config (grasp_viz/gripper_tho_pa1.yaml).

Kept separate from ``gripper.py`` (which draws the FGC gripper wireframe) so the
hardware limits live in one place and both backends can consult them.
"""
import os

import yaml

from . import paths

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "gripper_tho_pa1.yaml")

_CACHE = {}


def load(path=None):
    path = path or DEFAULT_PATH
    if path not in _CACHE:
        with open(path) as fh:
            _CACHE[path] = yaml.safe_load(fh)
    return _CACHE[path]


def max_grasp_width(path=None, override=None):
    """Widest grasp the jaws can span, minus the approach margin.

    ``override`` (metres) lets a caller sweep the aperture without editing the
    yaml -- the stroke is still unconfirmed, see the file's ``unresolved``.
    """
    cfg = load(path)
    gf = cfg.get("grasp_filter", {})
    cap = float(override if override is not None else gf.get("max_grasp_width", cfg["width"]))
    return cap - float(gf.get("width_margin", 0.0))
