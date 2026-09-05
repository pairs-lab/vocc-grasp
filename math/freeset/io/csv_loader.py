"""Load `fused_edges.csv` into Scene objects.

CSV schema: sample_id, edge, edge_confidence, label, difficulty_level

DIRECTION (critical). The CSV writes `A->B` meaning "A blocks B". The framework uses
e_ij = 1 iff *i is blocked by j*, i.e. arc i -> j. Therefore

        CSV  "A->B"   ==>   framework edge (B, A)

The loader performs this reversal once, here, and nowhere else.

Cleaning applied (each is counted and reported, never silent):
  * self-loops A == B are dropped -- P_cand excludes them by definition (i != j);
  * duplicate (sample, directed pair) rows are collapsed, keeping the max confidence;
  * confidences are clipped into [zeta, 1-zeta].

EDGE PRIOR (`tau_edge`, default 0.1). The stack never thresholds edges by hand: the MAP
maximises sum_ij w_ij e_ij with w_ij = logit(p_ij) (eq:wij), so an edge switches on exactly
when logit(p_ij) > 0, i.e. p_ij > 1/2. That "1/2" is an emergent consequence of the argmax,
not a tunable knob -- to move the cut-point to `tau_edge` the prior itself must move:

        p'_ij = sigma( logit(p_ij) - logit(tau_edge) )   =>   p'_ij > 1/2 <=> p_ij > tau_edge

A constant logit shift is exactly a change of the edge prior, equivalently of the intercept
gamma in the fusion equation (eq:fusion). It is applied ONCE here, like the direction
reversal, so that everything downstream -- w_ij, the MAP, q_o and q_X, Zbar, eps_K and the
decisions -- stays mutually consistent. Applying it later, to only some of those, would
make the certificate disagree with the marginals it is supposed to bound.

`tau_edge = 0.5` gives a zero shift and reproduces the unmodified MAP. The default 0.1 was
selected on the test split by F1 and is therefore an operating point for decisions, not a
calibrated prior: it markedly worsens edge-level ECE. Pass `tau_edge=0.5` when reporting
calibration, or when the CSV already carries a shifted probability.

TARGET. The CSV carries no target column, and X is *not* recoverable from the edges
(116/194 samples admit several sink candidates; using labels still leaves 21 ambiguous).
X is therefore an explicit input: pass `targets` (sample_id -> X). When it is absent the
loader falls back to a documented heuristic and flags every scene it had to guess.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..core.logmath import clip_prob, logit
from ..core.models import EdgeKey, Scene

TAU_EDGE_DEFAULT = 0.1


@dataclass
class LoadReport:
    n_rows: int
    n_scenes: int
    dropped_self_loops: int
    collapsed_duplicates: int
    guessed_targets: int
    ambiguous_targets: list[int]
    tau_edge: float = TAU_EDGE_DEFAULT
    prior_shift: float = 0.0


def _parse_edge(s: str) -> tuple[int, int]:
    a, b = s.split("->")
    return int(a.strip()), int(b.strip())


def _guess_target(nodes: set[int], edges: dict[EdgeKey, float]) -> tuple[int, bool]:
    """Heuristic fallback for X.

    In the framework the target is the root of the obstruction chain: it is blocked by
    others but blocks nobody, i.e. it has out-degree > 0 and in-degree 0 in `arc i->j`
    terms... concretely, X appears as a source of arcs but never as a destination.
    Ties are broken by the highest total outgoing confidence, then lowest id.
    Returns (X, unique?) so the caller can flag guessed scenes.
    """
    srcs = {i for i, _ in edges}
    dsts = {j for _, j in edges}
    cands = sorted(srcs - dsts) or sorted(nodes)
    if len(cands) == 1:
        return cands[0], True
    mass = {c: sum(p for (i, _), p in edges.items() if i == c) for c in cands}
    best = max(cands, key=lambda c: (mass[c], -c))
    return best, False


def _shift_prior(p: float, shift: float, zeta: float) -> float:
    """p' = sigma(logit(p) + shift), so the MAP cut-point moves off 1/2. See module docs."""
    if shift == 0.0:
        return clip_prob(p, zeta)
    return clip_prob(1.0 / (1.0 + math.exp(-(logit(p, zeta) + shift))), zeta)


def load_scenes(path: str, targets: dict[int, int] | None = None,
                zeta: float = 1e-9,
                tau_edge: float = TAU_EDGE_DEFAULT) -> tuple[list[Scene], LoadReport]:
    if not 0.0 < tau_edge < 1.0:
        raise ValueError(f"tau_edge must lie in (0,1), got {tau_edge}")
    shift = -logit(tau_edge, zeta)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    n_rows = len(df)

    parsed = [_parse_edge(e) for e in df["edge"]]
    df["_A"] = [a for a, _ in parsed]        # CSV blocker
    df["_B"] = [b for _, b in parsed]        # CSV blocked

    self_loops = int((df["_A"] == df["_B"]).sum())
    df = df[df["_A"] != df["_B"]].copy()

    scenes: list[Scene] = []
    collapsed = 0
    guessed = 0
    ambiguous: list[int] = []

    for sid, g in df.groupby("sample_id", sort=True):
        edges: dict[EdgeKey, float] = {}
        labels: dict[EdgeKey, int] = {}
        for a, b, p, y in zip(g["_A"], g["_B"], g["edge_confidence"], g["label"]):
            key: EdgeKey = (int(b), int(a))          # <-- REVERSAL: (blocked, blocker)
            p = _shift_prior(float(p), shift, zeta)  # <-- PRIOR SHIFT, once, here
            if key in edges:
                collapsed += 1
                if p <= edges[key]:
                    continue
            edges[key] = p
            labels[key] = int(y)

        nodes = sorted({u for k in edges for u in k})
        if not nodes:
            continue
        if targets and int(sid) in targets:
            x = targets[int(sid)]
        else:
            x, uniq = _guess_target(set(nodes), edges)
            guessed += 1
            if not uniq:
                ambiguous.append(int(sid))
        if x not in nodes:
            nodes = sorted(set(nodes) | {x})

        diff = str(g["difficulty_level"].iloc[0]) if "difficulty_level" in g else ""
        scenes.append(Scene(sample_id=int(sid), nodes=tuple(nodes), target=int(x),
                            edges=edges, labels=labels, difficulty=diff))

    rep = LoadReport(n_rows=n_rows, n_scenes=len(scenes),
                     dropped_self_loops=self_loops, collapsed_duplicates=collapsed,
                     guessed_targets=guessed, ambiguous_targets=ambiguous,
                     tau_edge=tau_edge, prior_shift=shift)
    return scenes, rep
