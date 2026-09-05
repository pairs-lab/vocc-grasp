"""Directed-graph primitives.

Convention (mathematical_proving.md, Notation): e_ij = 1 iff object i is directly
blocked by j; the arc i -> j means "j must be removed before i".

    Reach(X, e)     nodes reachable from X along obstruction arcs (includes X)
    Obstruct(X, e)  Reach(X, e) \\ {X}
    deg_v^+(e)      out-degree of v
    Free(X, e)      {v in Obstruct(X,e) : deg_v^+(e) = 0}   -- excludes X by definition
"""
from __future__ import annotations
from typing import Iterable, Optional

EdgeKey = tuple[int, int]


def adjacency(active: Iterable[EdgeKey]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {}
    for i, j in active:
        adj.setdefault(i, []).append(j)
    for i in adj:
        adj[i].sort()                       # deterministic traversal
    return adj


def out_degree(active: Iterable[EdgeKey]) -> dict[int, int]:
    deg: dict[int, int] = {}
    for i, _ in active:
        deg[i] = deg.get(i, 0) + 1
    return deg


def reach(target: int, active: Iterable[EdgeKey]) -> set[int]:
    """Reach(X, e), including X itself."""
    adj = adjacency(active)
    seen = {target}
    stack = [target]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def find_cycle(active: Iterable[EdgeKey]) -> Optional[list[EdgeKey]]:
    """Return one directed cycle as an edge list, or None if acyclic.

    Deterministic iterative 3-colour DFS (WHITE=0, GREY=1, BLACK=2); roots and
    neighbours visited in sorted order so the same cut is produced every run.
    """
    active = list(active)
    adj = adjacency(active)
    nodes = sorted({u for e in active for u in e})
    colour = {v: 0 for v in nodes}
    parent: dict[int, int] = {}

    def walk(start: int) -> Optional[list[EdgeKey]]:
        colour[start] = 1
        stack = [(start, iter(adj.get(start, ())))]
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                c = colour.get(v, 0)
                if c == 0:
                    colour[v] = 1
                    parent[v] = u
                    stack.append((v, iter(adj.get(v, ()))))
                    advanced = True
                    break
                if c == 1:                                  # back edge -> cycle
                    cyc = [(u, v)]
                    x = u
                    while x != v:
                        p = parent[x]
                        cyc.append((p, x))
                        x = p
                    cyc.reverse()
                    return cyc
            if not advanced:
                colour[u] = 2
                stack.pop()
        return None

    for s in nodes:
        if colour[s] == 0:
            c = walk(s)
            if c is not None:
                return c
    return None


def is_acyclic(active: Iterable[EdgeKey]) -> bool:
    """e in D  <=>  G(e) acyclic. No reachability constraint (eq:feasible-dag)."""
    return find_cycle(active) is None


def free_set(target: int, active: Iterable[EdgeKey]) -> frozenset[int]:
    """Free(X, e) = reachable blockers with zero out-degree; X itself excluded."""
    active = list(active)
    r = reach(target, active)
    deg = out_degree(active)
    return frozenset(v for v in r if v != target and deg.get(v, 0) == 0)


def target_is_free(target: int, active: Iterable[EdgeKey]) -> bool:
    """deg_X^+(e) == 0: the target has no remaining blocker (eq:target-risk)."""
    return out_degree(active).get(target, 0) == 0
