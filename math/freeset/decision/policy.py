"""Risk-aware decision layer.

Two thresholds with distinct origins (remark in sec:decision) and frozen separately:

    tau_act = 1 - lambda_d                      action / defer  (eq:defer)
    tau_set = c_FP / (c_FP + c_FN)              free-set        (eq:setrule)

Action space is V_t, i.e. the direct-target action X competes with every blocker o.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.models import Certificate, Decision, Scene


@dataclass(frozen=True)
class DecisionConfig:
    c_fp: float = 1.0
    c_fn: float = 1.0
    lambda_defer: float = 0.5

    @property
    def tau_set(self) -> float:
        d = self.c_fp + self.c_fn
        if d <= 0.0:
            raise ValueError(f"c_fp + c_fn must be > 0, got {d}")
        return self.c_fp / d

    @property
    def tau_act(self) -> float:
        if not (0.0 < self.lambda_defer < 1.0):
            raise ValueError(f"lambda_defer must be in (0,1), got {self.lambda_defer}")
        return 1.0 - self.lambda_defer


def decide(scene: Scene, scores: dict[int, float], margin: float,
           cert: Certificate | None, cfg: DecisionConfig, n_modes: int) -> Decision:
    """Apply eq:defer for the action and eq:setrule for the free set."""
    tau_a, tau_s = cfg.tau_act, cfg.tau_set

    # free set: blockers only, strictly above tau_set
    free = frozenset(o for o, q in scores.items()
                     if o != scene.target and q > tau_s)

    # action: argmax over V_t (target included); tie broken by lowest id, then defer
    if scores:
        best = max(sorted(scores), key=lambda a: scores[a])
        act_ok = scores[best] > tau_a          # equality -> defer, by convention
    else:
        best, act_ok = None, False

    certified = (cert is not None and act_ok and margin > 2.0 * cert.eps_K)
    return Decision(
        sample_id=scene.sample_id,
        q=scores,
        free_set=free,
        action=best if act_ok else None,
        deferred=not act_ok,
        margin=margin,
        arg_certified=certified,
        certificate=cert,
        n_modes=n_modes,
    )
