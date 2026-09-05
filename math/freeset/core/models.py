"""Immutable value objects carried between the stages of the stack.

    Scene        one scene: V_t, X, the fused p_ij over P_cand, and the GT labels
    Mode         one enumerated e in D, with its log-weight and Free(X,e)
    Certificate  Z_K, Zbar, eps_K, mu  -- the tail-model-free stopping certificate
    Decision     the final action, free set, and certification flags

All four are frozen: a Scene is never mutated in place, and a Mode is never re-scored.
They hold dicts and are therefore unhashable in practice; nothing hashes them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

EdgeKey = tuple[int, int]
"""A directed framework arc (i, j): "i is directly blocked by j".

Note the orientation. The CSV writes `A->B` for "A blocks B", so the loader stores that
row as (B, A). The reversal happens once, in `io/csv_loader.py`, and nowhere else.
"""


@dataclass(frozen=True)
class Scene:
    """One scene. `edges` maps P_cand -> fused p_ij; `labels` maps it -> GT edge label."""

    sample_id: int
    nodes: tuple[int, ...]
    target: int
    edges: dict[EdgeKey, float]
    labels: dict[EdgeKey, int] = field(default_factory=dict)
    difficulty: str = ""

    @property
    def n(self) -> int:
        """|V_t|."""
        return len(self.nodes)

    @property
    def m(self) -> int:
        """|P_cand|. Drives the back-end choice: m <= exact_max_edges -> enumerate D."""
        return len(self.edges)


@dataclass(frozen=True)
class Mode:
    """One feasible configuration e^(k) in D, in decreasing order of W."""

    rank: int
    active: frozenset[EdgeKey]
    log_weight: float                       # l(e)          (eq:log-weight)
    free_set: frozenset[int]                # Free(X, e)    (sec:compute-qo)
    target_free: bool                       # deg_X^+(e) == 0


@dataclass(frozen=True)
class Certificate:
    """Tail-model-free stopping certificate (thm:cert).

    eps_K = (Zbar - Z_K)/Zbar bounds |q - q_hat^(K)| uniformly over every marginal.
    `mu_lower` is the exact 1 - Z when D was fully enumerated, else the computable
    lower bound 1 - Zbar (def:mu).
    """

    K: int
    log_ZK: float
    log_Zbar: float
    eps_K: float
    mu_lower: float
    exact: bool


@dataclass(frozen=True)
class Decision:
    """Output of the decision layer: eq:policy for the action, eq:freeset-pred for F."""

    sample_id: int
    q: dict[int, float]                     # s_a for every a in V_t
    free_set: frozenset[int]                # F_hat = {o : q_o > tau_set}
    action: int | None                      # None iff deferred
    deferred: bool
    margin: float                           # s_(1) - s_(2)  (eq:approx-action-margin)
    arg_certified: bool                     # margin > 2 eps_K  (eq:carg)
    certificate: Certificate | None
    n_modes: int
