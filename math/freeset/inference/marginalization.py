"""Marginalization and the tail-model-free certificate.

    q_o        = P(o in Free(X,e) | Omega, e in D)                    (eq:blocker-risk)
    q_X        = P(deg_X^+(e) = 0 | Omega, e in D)                    (eq:target-risk)
    q_hat^(K)  = sum_{k<=K} 1[.] W^(k) / Z_K                          (eq:qo-topk)
    Zbar       = prod_{i<j} (1 - p_ij p_ji)   >= Z                    (prop:zbar)
    eps_K      = (Zbar - Z_K)/Zbar,  |q - q_hat^(K)| <= eps_K         (thm:cert)
    mu         = 1 - Z  >= 1 - Zbar                                   (def:mu)

Everything is accumulated in log-space; Z_K is never formed as a linear product.
"""
from __future__ import annotations

import math

from ..core.logmath import NEG_INF, clip_prob, log1mexp, logsumexp
from ..core.models import Certificate, Mode, Scene


def log_partition_K(modes: list[Mode]) -> float:
    """log Z_K = logsumexp_k l^(k)."""
    return logsumexp(m.log_weight for m in modes)


def action_scores(modes: list[Mode], scene: Scene) -> dict[int, float]:
    """s_a for every a in V_t: q_X at the target key, q_o elsewhere (eq:action-score)."""
    log_ZK = log_partition_K(modes)
    if log_ZK == NEG_INF:
        return {a: 0.0 for a in scene.nodes}
    out: dict[int, float] = {}
    for a in scene.nodes:
        if a == scene.target:
            sel = [m.log_weight for m in modes if m.target_free]
        else:
            sel = [m.log_weight for m in modes if a in m.free_set]
        num = logsumexp(sel)
        out[a] = 0.0 if num == NEG_INF else math.exp(num - log_ZK)
    return out


# ---- certificate -----------------------------------------------------------------
def log_Zbar(scene: Scene, zeta: float = 1e-9) -> float:
    """log Zbar = sum_{i<j} log(1 - p_ij p_ji)  (eq:zbar).

    p_ij = 0 for directed pairs outside P_cand, so those unordered pairs contribute 0.
    """
    p = {k: clip_prob(v, zeta) for k, v in scene.edges.items()}
    ids = sorted(scene.nodes)
    total = 0.0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            prod = p.get((i, j), 0.0) * p.get((j, i), 0.0)
            if prod > 0.0:
                total += log1mexp(math.log(prod))
    return total


def certificate(modes: list[Mode], scene: Scene, exact: bool,
                zeta: float = 1e-9) -> Certificate:
    if not modes:
        raise ValueError("certificate needs at least one enumerated mode")
    log_ZK = log_partition_K(modes)
    log_Zb = log_Zbar(scene, zeta)
    ratio = math.exp(min(0.0, log_ZK - log_Zb))          # Z_K / Zbar in (0,1]
    eps = max(0.0, min(1.0, 1.0 - ratio))
    # mu = 1 - Z. Exact when D was fully enumerated (then Z_K = Z); otherwise the
    # computable lower bound 1 - Zbar is reported.
    mu = 1.0 - math.exp(log_ZK) if exact else 1.0 - math.exp(log_Zb)
    return Certificate(K=len(modes), log_ZK=log_ZK, log_Zbar=log_Zb,
                       eps_K=eps, mu_lower=mu, exact=exact)


def margin(scores: dict[int, float]) -> float:
    """Approximate action margin s_(1) - s_(2)  (eq:approx-action-margin)."""
    v = sorted(scores.values(), reverse=True)
    if not v:
        return 0.0
    return v[0] - v[1] if len(v) > 1 else v[0]
