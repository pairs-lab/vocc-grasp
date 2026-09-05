"""Log-space primitives.

The whole stack accumulates in log-space: Z_K is never formed as a linear product, and
W(e) = exp(l(e)) is only materialised for reporting. These are the four numerically
delicate operations that make that possible.

    clip_prob      p -> [zeta, 1-zeta], so logit is finite and Z > 0 (rmk:cert)
    logit          w_ij = log(p/(1-p))                                (eq:wij)
    log_bernoulli  one term of l(e), parameterised by the *logit*     (eq:log-weight)
    log1mexp       log(1-exp(x)) without catastrophic cancellation    (eq:zbar)
    logsumexp      log sum exp, the only way Z_K is ever accumulated  (thm:cert)
"""
from __future__ import annotations

import math

NEG_INF = float("-inf")

_LOG2 = math.log(2.0)


def clip_prob(p: float, zeta: float = 1e-9) -> float:
    """Clip p into [zeta, 1-zeta]  (rmk:cert).

    Keeps logit finite and guarantees the empty graph has strictly positive weight,
    hence Z > 0 and the feasible set D is never degenerate.
    """
    if zeta <= 0.0 or zeta >= 0.5:
        raise ValueError(f"zeta must lie in (0, 0.5), got {zeta}")
    return min(max(p, zeta), 1.0 - zeta)


def logit(p: float, zeta: float = 1e-9) -> float:
    """w_ij = log(p/(1-p))  (eq:wij). Clipped first, so the result is always finite."""
    p = clip_prob(p, zeta)
    return math.log(p) - math.log1p(-p)


def _softplus(x: float) -> float:
    """log(1 + exp(x)), overflow-safe for large |x|."""
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def log_bernoulli(active: bool, w: float) -> float:
    """One term of l(e), where `w` is the LOGIT of p -- not p itself.

    `topk._log_weight` sums this over w = {k: logit(p_k)} to build

        l(e) = sum_ij [ e_ij log p_ij + (1 - e_ij) log(1 - p_ij) ]      (eq:log-weight)

    With w = logit(p) we have p = sigma(w), so

        log p     = log sigma(w)  = -softplus(-w)
        log(1-p)  = log sigma(-w) = -softplus(+w)

    which is exact and free of the underflow that log(1 - exp(...)) would introduce.
    """
    return -_softplus(-w) if active else -_softplus(w)


def log1mexp(x: float) -> float:
    """log(1 - exp(x)) for x <= 0, split at -log 2 to avoid cancellation.

    Used for log Zbar = sum_{i<j} log(1 - p_ij p_ji)  (eq:zbar), where the product
    p_ij p_ji is typically tiny and 1 - p_ij p_ji would round to exactly 1.
    """
    if x > 0.0:
        raise ValueError(f"log1mexp requires x <= 0, got {x}")
    if x == 0.0:
        return NEG_INF
    if x > -_LOG2:
        return math.log(-math.expm1(x))     # x near 0: expm1 keeps the precision
    return math.log1p(-math.exp(x))         # x very negative: exp(x) is tiny


def logsumexp(vals) -> float:
    """log sum_k exp(v_k), shifted by the max. NEG_INF on an empty or all -inf input.

    Accepts any iterable, including a generator (`marginalization.log_partition_K`
    passes one directly).
    """
    finite = [v for v in vals if v > NEG_INF]
    if not finite:
        return NEG_INF
    hi = max(finite)
    if hi == float("inf"):
        return hi
    return hi + math.log(math.fsum(math.exp(v - hi) for v in finite))
