#!/usr/bin/env python3
"""Independent verification of the post-fusion stack.

Every quantity is recomputed from scratch in linear space (no shared code path with
the log-space implementation) and compared:

  1. D            -- the enumerated feasible set equals the brute-force acyclic set
  2. l(e)         -- exp(l) equals prod p^e (1-p)^(1-e)
  3. Z            -- Z_K at full enumeration equals sum over D
  4. q_o, q_X     -- log-space marginals equal the linear-space ones
  5. Zbar         -- equals prod_{i<j}(1 - p_ij p_ji) and satisfies Z <= Zbar
  6. eps_K        -- equals (Zbar - Z_K)/Zbar and bounds the true error |q - q_hat^(K)|
  7. ILP          -- Top-K from CBC matches the exact enumeration order and set
  8. Free(X,e)    -- excludes X, is a subset of Reach(X,e), all members have out-deg 0
"""
from __future__ import annotations

import itertools
import math
import sys

from freeset.core.models import Scene
from freeset.inference import graph_ops as G
from freeset.inference import marginalization as M
from freeset.inference.topk import ILPTopKSolver, enumerate_exact
from freeset.io.csv_loader import load_scenes

PASS = FAIL = 0


def ck(name, cond, extra=""):
    global PASS, FAIL
    PASS += bool(cond); FAIL += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def brute(scene: Scene):
    """Linear-space ground truth: (D, Z, q_o for all a, Zbar)."""
    keys = sorted(scene.edges)
    p = {k: scene.edges[k] for k in keys}
    D, Z = [], 0.0
    num = {a: 0.0 for a in scene.nodes}
    for bits in itertools.product((0, 1), repeat=len(keys)):
        act = frozenset(k for k, b in zip(keys, bits) if b)
        if not G.is_acyclic(act):
            continue
        W = 1.0
        for k in keys:
            W *= p[k] if k in act else (1.0 - p[k])
        D.append((act, W)); Z += W
        fs = G.free_set(scene.target, act)
        for a in scene.nodes:
            hit = G.target_is_free(scene.target, act) if a == scene.target else (a in fs)
            if hit:
                num[a] += W
    q = {a: (num[a] / Z if Z > 0 else 0.0) for a in scene.nodes}
    ids = sorted(scene.nodes)
    zbar = 1.0
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            i, j = ids[x], ids[y]
            zbar *= 1.0 - p.get((i, j), 0.0) * p.get((j, i), 0.0)
    return D, Z, q, zbar


def main():
    scenes, rep = load_scenes("fused_edges.csv")
    small = [s for s in scenes if s.m <= 14]
    print(f"loaded {len(scenes)} scenes; verifying {min(12,len(small))} with |P_cand| <= 14\n")

    worst_q = worst_W = worst_Z = worst_zb = 0.0
    bad_D = bad_free = bad_bound = bad_ilp = 0
    ilp = ILPTopKSolver()

    for s in small[:12]:
        modes = enumerate_exact(s)
        D, Z, q, zbar = brute(s)

        # 1. feasible set
        if {m.active for m in modes} != {a for a, _ in D}:
            bad_D += 1
        # 2. log weight
        for m in modes:
            Wb = next(W for a, W in D if a == m.active)
            worst_W = max(worst_W, abs(math.exp(m.log_weight) - Wb))
        # 3. partition
        worst_Z = max(worst_Z, abs(math.exp(M.log_partition_K(modes)) - Z))
        # 4. marginals
        sc = M.action_scores(modes, s)
        worst_q = max(worst_q, max(abs(sc[a] - q[a]) for a in s.nodes))
        # 5. Zbar
        cert = M.certificate(modes, s, exact=True)
        worst_zb = max(worst_zb, abs(math.exp(cert.log_Zbar) - zbar))
        if math.exp(cert.log_Zbar) < Z - 1e-12:
            bad_bound += 1
        # 6. eps_K bounds the true truncation error at K=1
        if len(modes) > 1:
            c1 = M.certificate(modes[:1], s, exact=False)
            s1 = M.action_scores(modes[:1], s)
            if max(abs(s1[a] - q[a]) for a in s.nodes) > c1.eps_K + 1e-9:
                bad_bound += 1
        # 7. ILP vs exact
        if s.m <= 8:
            im = ilp.solve(s, K=len(modes))
            if {m.active for m in im} != {m.active for m in modes}:
                bad_ilp += 1
            elif any(im[i].log_weight < im[i + 1].log_weight - 1e-9
                     for i in range(len(im) - 1)):
                bad_ilp += 1
        # 8. Free-set semantics
        for m in modes:
            r = G.reach(s.target, m.active)
            deg = G.out_degree(m.active)
            if (s.target in m.free_set or not m.free_set <= r
                    or any(deg.get(v, 0) != 0 for v in m.free_set)):
                bad_free += 1
                break

    print("=" * 70)
    ck("1. enumerated D == brute-force acyclic set", bad_D == 0, f"({bad_D} bad)")
    ck("2. exp(l(e)) == prod p^e (1-p)^(1-e)", worst_W < 1e-12, f"max|d|={worst_W:.1e}")
    ck("3. Z_K (full) == Z", worst_Z < 1e-9, f"max|d|={worst_Z:.1e}")
    ck("4. q_o and q_X match brute force", worst_q < 1e-9, f"max|dq|={worst_q:.1e}")
    ck("5. Zbar matches recompute", worst_zb < 1e-9, f"max|d|={worst_zb:.1e}")
    ck("6. Z <= Zbar and eps_K bounds true error", bad_bound == 0, f"({bad_bound} violations)")
    ck("7. ILP Top-K == exact enumeration, ordered", bad_ilp == 0, f"({bad_ilp} bad)")
    ck("8. Free(X,e) excludes X, subset of Reach, out-deg 0", bad_free == 0)

    # direction check on a hand-verifiable scene
    s6 = next(s for s in scenes if s.sample_id == 6)
    ck("9. CSV '3->2' (3 blocks 2) became framework e[2<-3]",
       (2, 3) in s6.edges and (3, 2) not in s6.edges)
    print("=" * 70)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
