"""Orchestrator: Scene -> Top-K modes -> marginals -> certificate -> decision.

Back-end selection is an engineering choice only; the mathematics is identical:
  m <= exact_max_edges   ->  full enumeration of D  (Z_K = Z, certificate tight)
  otherwise              ->  ILP Top-K with lazy cuts, adaptive stopping on eps_K.

Adaptive stopping (sec:adaptive): enumerate up to K_max and stop at the first K whose
certificate satisfies eps_K <= eps_target, or whose action margin is already certified
(margin > 2 eps_K). Falling back to K_max is expected and does not by itself trigger
defer -- certification concerns approximation error, defer concerns posterior risk.
"""
from __future__ import annotations

from dataclasses import dataclass

from .core.models import Decision, Mode, Scene
from .decision.policy import DecisionConfig, decide
from .inference import marginalization as M
from .inference.topk import ILPTopKSolver, enumerate_exact


@dataclass(frozen=True)
class PipelineConfig:
    exact_max_edges: int = 20          # 2^20 configurations is still fast
    K_max: int = 256
    eps_target: float = 0.05
    zeta: float = 1e-9
    decision: DecisionConfig = DecisionConfig()


class FreeSetPipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self.solver = ILPTopKSolver()

    # ---- top-K stage -------------------------------------------------------------
    def modes(self, scene: Scene) -> tuple[list[Mode], bool]:
        """Return (modes, exact) where `exact` marks a complete enumeration of D."""
        if scene.m <= self.cfg.exact_max_edges:
            return enumerate_exact(scene, self.cfg.zeta), True
        return self.solver.solve(scene, self.cfg.K_max, self.cfg.zeta), False

    # ---- adaptive stopping -------------------------------------------------------
    def _stop_at(self, modes: list[Mode], scene: Scene, exact: bool) -> int:
        if exact:
            return len(modes)
        for K in range(1, len(modes) + 1):
            cert = M.certificate(modes[:K], scene, exact=False, zeta=self.cfg.zeta)
            if cert.eps_K <= self.cfg.eps_target:
                return K
            sc = M.action_scores(modes[:K], scene)
            if M.margin(sc) > 2.0 * cert.eps_K:
                return K
        return len(modes)

    # ---- full run ----------------------------------------------------------------
    def run(self, scene: Scene) -> Decision:
        all_modes, exact = self.modes(scene)
        if not all_modes:                                  # D empty: safe defer
            return Decision(sample_id=scene.sample_id, q={a: 0.0 for a in scene.nodes},
                            free_set=frozenset(), action=None, deferred=True,
                            margin=0.0, arg_certified=False, certificate=None, n_modes=0)
        K = self._stop_at(all_modes, scene, exact)
        modes = all_modes[:K]
        scores = M.action_scores(modes, scene)
        cert = M.certificate(modes, scene, exact=exact, zeta=self.cfg.zeta)
        return decide(scene, scores, M.margin(scores), cert,
                      self.cfg.decision, n_modes=len(modes))
