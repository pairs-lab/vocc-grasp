#!/usr/bin/env python3
"""Run the post-fusion stack on `fused_edges.csv`.

    python run_csv.py fused_edges.csv                 # summary over all scenes
    python run_csv.py fused_edges.csv --show 5        # print 5 scenes in detail
    python run_csv.py fused_edges.csv --sample 6      # one scene, full trace
    python run_csv.py fused_edges.csv --targets t.json --out results.csv

`--targets` is a JSON mapping {"sample_id": target_object_id}. Without it the loader
falls back to a heuristic and reports how many targets it had to guess.
"""
from __future__ import annotations

import argparse
import json
import math

from freeset.core.models import Scene
from freeset.decision.policy import DecisionConfig
from freeset.inference import graph_ops as G
from freeset.io.csv_loader import load_scenes
from freeset.pipeline import FreeSetPipeline, PipelineConfig


def show_scene(scene: Scene, dec, pipe: FreeSetPipeline) -> None:
    print("=" * 74)
    print(f" sample {scene.sample_id}   difficulty={scene.difficulty}   "
          f"|V_t|={scene.n}  |P_cand|={scene.m}   X={scene.target}")
    print("=" * 74)
    print("  fused edges  (framework (i,j) = 'i blocked by j'; CSV was reversed)")
    for (i, j), p in sorted(scene.edges.items()):
        y = scene.labels.get((i, j))
        print(f"    e[{i}<-{j}]  p={p:.4f}   w=logit={math.log(p/(1-p)):+.4f}"
              f"   (csv {j}->{i}, label={y})")
    modes, exact = pipe.modes(scene)
    print(f"\n  |D| enumerated = {len(modes)}   exact={exact}")
    m0 = modes[0]
    print(f"  MAP e^(1): active={sorted(m0.active) if m0.active else '{}'}  "
          f"l={m0.log_weight:.4f}")
    print(f"             Free(X,e)={set(m0.free_set) or '{}'}   deg_X^+=0? {m0.target_free}")
    c = dec.certificate
    print(f"\n  Z_K={math.exp(c.log_ZK):.6f}   Zbar={math.exp(c.log_Zbar):.6f}   "
          f"eps_K={c.eps_K:.6f}")
    print(f"  mu = {c.mu_lower:.6f} ({'exact 1-Z' if c.exact else 'lower bound 1-Zbar'})")
    print("\n  action scores s_a  (a = X uses q_X, else q_o):")
    for a in sorted(dec.q, key=lambda k: -dec.q[k]):
        tag = "  [X]" if a == scene.target else ""
        star = "  <- argmax" if a == dec.action else ""
        inF = " in F_hat" if a in dec.free_set else ""
        print(f"    s[{a}] = {dec.q[a]:.6f}{tag}{inF}{star}")
    print(f"\n  F_hat = {{{', '.join(map(str, sorted(dec.free_set)))}}}"
          f"   (tau_set={pipe.cfg.decision.tau_set:.2f})")
    if dec.deferred:
        print(f"  ACTION: DEFER   (max s = {max(dec.q.values()):.4f} "
              f"<= tau_act={pipe.cfg.decision.tau_act:.2f})")
    else:
        print(f"  ACTION: {'grasp TARGET ' + str(dec.action) if dec.action == scene.target else 'remove blocker ' + str(dec.action)}"
              f"   (s={dec.q[dec.action]:.4f} > tau_act={pipe.cfg.decision.tau_act:.2f})")
    print(f"  margin={dec.margin:.4f}   argmax certified (margin>2*eps_K)? {dec.arg_certified}")


def _tau(a) -> dict:
    """Only override the loader default when --tau-edge was actually given."""
    return {} if a.tau_edge is None else {"tau_edge": a.tau_edge}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--targets", help="JSON {sample_id: target_id}")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--c-fp", type=float, default=1.0)
    ap.add_argument("--c-fn", type=float, default=1.0)
    ap.add_argument("--lambda-defer", type=float, default=0.5)
    ap.add_argument("--eps-target", type=float, default=0.05)
    ap.add_argument("--tau-edge", type=float, default=None,
                    help="MAP edge cut-point; default comes from the loader (0.1). "
                         "Pass 0.5 for the unmodified MAP.")
    a = ap.parse_args()

    targets = None
    if a.targets:
        targets = {int(k): int(v) for k, v in json.load(open(a.targets)).items()}

    scenes, rep = load_scenes(a.csv, targets, **_tau(a))
    dcfg = DecisionConfig(c_fp=a.c_fp, c_fn=a.c_fn, lambda_defer=a.lambda_defer)
    pipe = FreeSetPipeline(PipelineConfig(eps_target=a.eps_target, decision=dcfg))

    print("=" * 74)
    print(" LOAD REPORT")
    print("=" * 74)
    print(f"  rows={rep.n_rows}  scenes={rep.n_scenes}")
    print(f"  dropped self-loops   : {rep.dropped_self_loops}")
    print(f"  collapsed duplicates : {rep.collapsed_duplicates}")
    print(f"  targets guessed      : {rep.guessed_targets}"
          + ("  (pass --targets to supply them)" if rep.guessed_targets else ""))
    print(f"  ...of which ambiguous: {len(rep.ambiguous_targets)}")
    print(f"  tau_act={dcfg.tau_act:.2f}  tau_set={dcfg.tau_set:.2f}")
    print(f"  tau_edge={rep.tau_edge:.3f}  (prior shift {rep.prior_shift:+.4f}; "
          f"0.5 = unmodified MAP)")

    if a.sample is not None:
        s = next(x for x in scenes if x.sample_id == a.sample)
        show_scene(s, pipe.run(s), pipe)
        return

    rows = []
    n_defer = n_cert = n_target = 0
    for s in scenes:
        d = pipe.run(s)
        n_defer += d.deferred
        n_cert += d.arg_certified
        n_target += (d.action == s.target)
        rows.append(dict(
            sample_id=s.sample_id, difficulty=s.difficulty, n_nodes=s.n, n_edges=s.m,
            target=s.target, n_modes=d.n_modes,
            action=("DEFER" if d.deferred else d.action),
            action_is_target=(d.action == s.target),
            free_set="|".join(map(str, sorted(d.free_set))),
            q_max=max(d.q.values()) if d.q else 0.0,
            q_X=d.q.get(s.target, 0.0), margin=d.margin,
            eps_K=(d.certificate.eps_K if d.certificate else 1.0),
            mu=(d.certificate.mu_lower if d.certificate else 1.0),
            arg_certified=d.arg_certified,
        ))

    for s, d in zip(scenes[:a.show], (pipe.run(x) for x in scenes[:a.show])):
        show_scene(s, d, pipe)

    import pandas as pd
    out = pd.DataFrame(rows)
    print("\n" + "=" * 74)
    print(" SUMMARY")
    print("=" * 74)
    print(f"  scenes                : {len(out)}")
    print(f"  deferred              : {n_defer} ({100*n_defer/len(out):.1f}%)")
    print(f"  action = grasp target : {n_target} ({100*n_target/len(out):.1f}%)")
    print(f"  argmax certified      : {n_cert} ({100*n_cert/len(out):.1f}%)")
    print(f"  mean |F_hat|          : {out.free_set.apply(lambda s: 0 if not s else len(s.split('|'))).mean():.2f}")
    print(f"  mean eps_K            : {out.eps_K.mean():.4f}")
    print(f"  mean mu               : {out.mu.mean():.4f}")
    print(f"  mean modes |D|        : {out.n_modes.mean():.1f}")
    print("\n  by difficulty:")
    print(out.groupby("difficulty").agg(
        n=("sample_id", "size"), defer=("action", lambda s: (s == "DEFER").mean()),
        eps=("eps_K", "mean"), mu=("mu", "mean"),
        cert=("arg_certified", "mean")).to_string())
    if a.out:
        out.to_csv(a.out, index=False)
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
