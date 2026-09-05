"""Top-K MAP over the acyclic support D.

Two interchangeable back-ends producing the same object (an ordered list of modes):

  * `enumerate_exact`  -- full enumeration of D. Exact by construction, so Z_K = Z and
    the certificate becomes tight. Cost 2^m, used when m <= exact_max_edges.
  * `ILPTopKSolver`    -- eq:ilp-obj with lazy acyclicity cuts (eq:dag-cut) and no-good
    cuts (eq:nogood), solved by CBC. Used when exact enumeration is too large.

Both return modes sorted by decreasing W, i.e. e^(1), e^(2), ...
"""
from __future__ import annotations

import itertools

from ..core.logmath import log_bernoulli, logit
from ..core.models import EdgeKey, Mode, Scene
from . import graph_ops as G


# ----------------------------------------------------------------------------------
def _log_weight(active: frozenset[EdgeKey], w: dict[EdgeKey, float]) -> float:
    """l(e) = sum_ij [e_ij log p_ij + (1-e_ij) log(1-p_ij)]  (eq:log-weight)."""
    return sum(log_bernoulli(k in active, wk) for k, wk in w.items())


def _mode(rank: int, active: frozenset[EdgeKey], w, target: int) -> Mode:
    return Mode(
        rank=rank,
        active=active,
        log_weight=_log_weight(active, w),
        free_set=G.free_set(target, active),
        target_free=G.target_is_free(target, active),
    )


# ----------------------------------------------------------------------------------
def enumerate_exact(scene: Scene, zeta: float = 1e-9, K: int | None = None) -> list[Mode]:
    """Enumerate every e in D exactly, sorted by decreasing W."""
    keys = sorted(scene.edges)
    w = {k: logit(scene.edges[k], zeta) for k in keys}
    out: list[Mode] = []
    for bits in itertools.product((0, 1), repeat=len(keys)):
        active = frozenset(k for k, b in zip(keys, bits) if b)
        if not G.is_acyclic(active):
            continue
        out.append(_mode(0, active, w, scene.target))
    out.sort(key=lambda m: -m.log_weight)
    if K is not None:
        out = out[:K]
    return [Mode(rank=i + 1, active=m.active, log_weight=m.log_weight,
                 free_set=m.free_set, target_free=m.target_free)
            for i, m in enumerate(out)]


# ----------------------------------------------------------------------------------
class ILPTopKSolver:
    """Sequential Top-K MAP: max sum w_ij z_ij subject to acyclicity + no-good cuts."""

    def __init__(self, threads: int = 1, seed: int = 0, time_limit_s: float = 30.0):
        self.threads, self.seed, self.time_limit_s = threads, seed, time_limit_s

    def _solver(self):
        import pulp
        return pulp.PULP_CBC_CMD(
            msg=0, threads=self.threads, timeLimit=self.time_limit_s,
            options=["randomSeed", str(self.seed), "randomCbcSeed", str(self.seed)],
        )

    def solve(self, scene: Scene, K: int, zeta: float = 1e-9) -> list[Mode]:
        import pulp
        keys = sorted(scene.edges)
        w = {k: logit(scene.edges[k], zeta) for k in keys}

        prob = pulp.LpProblem("topk_dag", pulp.LpMaximize)
        z = {k: pulp.LpVariable(f"z_{k[0]}_{k[1]}", cat="Binary") for k in keys}
        prob += pulp.lpSum(w[k] * z[k] for k in keys)          # eq:ilp-obj

        solver = self._solver()
        modes: list[Mode] = []
        for rank in range(1, K + 1):
            active = self._solve_acyclic(prob, z, solver, keys)
            if active is None:
                break                                          # D exhausted
            fs = frozenset(active)
            modes.append(_mode(rank, fs, w, scene.target))
            S = set(active)                                    # eq:nogood
            prob += (pulp.lpSum(1 - z[k] for k in keys if k in S)
                     + pulp.lpSum(z[k] for k in keys if k not in S) >= 1)
        return modes

    @staticmethod
    def _solve_acyclic(prob, z, solver, keys):
        """Solve, adding a lazy cycle cut (eq:dag-cut) until the incumbent is acyclic."""
        import pulp
        while True:
            prob.solve(solver)
            if pulp.LpStatus[prob.status] != "Optimal":
                return None
            active = [k for k in keys if z[k].value() is not None and z[k].value() > 0.5]
            cyc = G.find_cycle(active)
            if cyc is None:
                return active
            prob += pulp.lpSum(z[k] for k in cyc) <= len(cyc) - 1
