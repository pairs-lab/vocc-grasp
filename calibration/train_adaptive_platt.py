#!/usr/bin/env python3
"""
Train N_cand-adaptive monotonic Platt calibration.

Formula:
    phi_s = (log(1 + N_cand_s) - mu_train) / sigma_train
    a_s   = exp(alpha0 + alphaN * phi_s) > 0
    c_s   = c0 + cN * phi_s
    p_cal = sigmoid(a_s * logit(p_raw) + c_s)

The positive exponential slope guarantees within-scene rank preservation.
Adaptive terms are L2-regularized. Lambda is selected only inside the training
set by grouped cross-validation.

Example:
    python train_adaptive_platt.py \
        --train-csv train_adaptive.csv \
        --valid-csv valid_adaptive.csv \
        --model-out adaptive_platt_model.json \
        --report-out adaptive_platt_report.json

Dependencies:
    numpy, scipy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def load_csv(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "prob_edge", "N_cand", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            p = float(row["prob_edge"])
            if p > 1.5:
                p /= 100.0
            n_cand_float = float(row["N_cand"])
            n_cand = int(n_cand_float)
            y = int(float(row["label"]))
            if not 0.0 <= p <= 1.0 or n_cand < 0 or n_cand_float != n_cand or y not in (0, 1):
                raise ValueError(f"{path}:{line_no}: invalid p/N_cand/label")
            out.append({
                "image": str(row["image"]),
                "p": min(max(p, EPS), 1.0 - EPS),
                "n_cand": n_cand,
                "y": y,
            })
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def arrays(
    rows: list[dict],
    mu_phi: float | None = None,
    sigma_phi: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    p = np.asarray([r["p"] for r in rows], dtype=float)
    x = np.log(p / (1.0 - p))
    y = np.asarray([r["y"] for r in rows], dtype=float)
    logn = np.log1p(np.asarray([r["n_cand"] for r in rows], dtype=float))

    if mu_phi is None:
        mu_phi = float(np.mean(logn))
    if sigma_phi is None:
        sigma_phi = float(np.std(logn))
        if sigma_phi < 1e-8:
            sigma_phi = 1.0

    phi = (logn - mu_phi) / sigma_phi
    return p, x, y, phi, float(mu_phi), float(sigma_phi)


def binary_nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for i in range(n_bins):
        upper = edges[i + 1] + (1e-12 if i == n_bins - 1 else 0.0)
        mask = (p >= edges[i]) & (p < upper)
        if np.any(mask):
            value += float(np.mean(mask)) * abs(float(np.mean(p[mask]) - np.mean(y[mask])))
    return value


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"nll": binary_nll(y, p), "brier": brier(y, p), "ece10": ece(y, p, 10)}


def make_group_folds(rows: list[dict], k: int, seed: int) -> list[set[str]]:
    scenes = sorted({r["image"] for r in rows})
    if len(scenes) < k:
        raise ValueError(f"Only {len(scenes)} scenes; cannot create {k} grouped folds")
    rng = random.Random(seed)
    rng.shuffle(scenes)
    return [set(scenes[i::k]) for i in range(k)]


def fit_model(
    rows: list[dict],
    regularization: float,
    starts: int = 5,
    seed: int = 0,
) -> dict:
    _, x, y, phi, mu_phi, sigma_phi = arrays(rows)

    def objective(theta: np.ndarray) -> float:
        alpha0, alpha_n, c0, c_n = [float(v) for v in theta]
        a_scene = np.exp(np.clip(alpha0 + alpha_n * phi, -5.0, 5.0))
        q = sigmoid(a_scene * x + c0 + c_n * phi)
        penalty = 0.5 * regularization * (alpha_n**2 + c_n**2)
        return binary_nll(y, q) + penalty

    initial = [
        np.asarray([math.log(0.70), 0.0, -0.40, 0.0]),
        np.asarray([math.log(1.00), 0.0, 0.00, 0.0]),
    ]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, starts - len(initial))):
        initial.append(np.asarray([
            float(rng.uniform(math.log(0.35), math.log(1.30))),
            float(rng.normal(0.0, 0.10)),
            float(rng.normal(-0.35, 0.50)),
            float(rng.normal(0.0, 0.15)),
        ]))

    bounds = [(-5.0, 3.0), (-2.0, 2.0), (None, None), (-3.0, 3.0)]
    best = None
    for x0 in initial:
        result = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not best.success:
        raise RuntimeError(f"Optimization failed: {getattr(best, 'message', 'unknown error')}")

    alpha0, alpha_n, c0, c_n = [float(v) for v in best.x]
    return {
        "model_type": "ncand_adaptive_platt",
        "formula": (
            "phi=(log1p(N_cand)-mu_phi)/sigma_phi; "
            "a_scene=exp(alpha0+alphaN*phi); "
            "p_cal=sigmoid(a_scene*logit(p_raw)+c0+cN*phi)"
        ),
        "alpha0": alpha0,
        "alphaN": alpha_n,
        "c0": c0,
        "cN": c_n,
        "mu_phi": mu_phi,
        "sigma_phi": sigma_phi,
        "regularization": float(regularization),
        "eps": EPS,
        "penalized_train_objective": float(best.fun),
    }


def predict(rows: list[dict], model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p, x, y, phi, _, _ = arrays(
        rows, float(model["mu_phi"]), float(model["sigma_phi"])
    )
    a_scene = np.exp(
        np.clip(float(model["alpha0"]) + float(model["alphaN"]) * phi, -5.0, 5.0)
    )
    q = sigmoid(a_scene * x + float(model["c0"]) + float(model["cN"]) * phi)
    return y, p, q, a_scene


def select_regularization(
    train_rows: list[dict],
    candidates: list[float],
    folds: int,
    seed: int,
) -> tuple[float, dict]:
    group_folds = make_group_folds(train_rows, folds, seed)
    results = {}

    for lam in candidates:
        fold_nll = []
        for fold_idx, heldout_scenes in enumerate(group_folds):
            inner_train = [r for r in train_rows if r["image"] not in heldout_scenes]
            inner_valid = [r for r in train_rows if r["image"] in heldout_scenes]
            model = fit_model(
                inner_train, regularization=lam, starts=4, seed=seed + 100 * fold_idx
            )
            y, _, q, _ = predict(inner_valid, model)
            fold_nll.append(binary_nll(y, q))
        results[str(lam)] = {
            "mean_nll": float(np.mean(fold_nll)),
            "std_nll": float(np.std(fold_nll, ddof=1)),
            "fold_nll": [float(v) for v in fold_nll],
        }

    best = min(
        candidates,
        key=lambda lam: (results[str(lam)]["mean_nll"], -lam),
    )
    return float(best), results


def monotonicity_check(model: dict, observed_n_cand: Iterable[int]) -> dict:
    grid = np.linspace(EPS, 1.0 - EPS, 10001)
    x = np.log(grid / (1.0 - grid))
    minimum_increment = float("inf")
    slope_values = []

    for n_cand in sorted(set(int(v) for v in observed_n_cand)):
        phi = (
            math.log1p(n_cand) - float(model["mu_phi"])
        ) / float(model["sigma_phi"])
        a_scene = math.exp(
            float(np.clip(
                float(model["alpha0"]) + float(model["alphaN"]) * phi,
                -5.0,
                5.0,
            ))
        )
        q = sigmoid(a_scene * x + float(model["c0"]) + float(model["cN"]) * phi)
        minimum_increment = min(minimum_increment, float(np.min(np.diff(q))))
        slope_values.append(a_scene)

    return {
        "passed_within_scene": bool(minimum_increment > 0.0),
        "minimum_increment": minimum_increment,
        "minimum_scene_slope": float(min(slope_values)),
        "maximum_scene_slope": float(max(slope_values)),
    }


def scene_bootstrap(
    rows: list[dict],
    model: dict,
    repeats: int,
    seed: int,
) -> dict:
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_scene[r["image"]].append(r)
    scenes = sorted(by_scene)
    rng = np.random.default_rng(seed)

    deltas = {"nll": [], "brier": [], "ece10": []}
    for _ in range(repeats):
        sampled = rng.choice(scenes, size=len(scenes), replace=True)
        boot = [r for s in sampled for r in by_scene[str(s)]]
        y, raw, cal, _ = predict(boot, model)
        m_raw, m_cal = metrics(y, raw), metrics(y, cal)
        for key in deltas:
            deltas[key].append(m_cal[key] - m_raw[key])

    result = {}
    for key, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        result[key] = {
            "mean_delta_cal_minus_raw": float(np.mean(arr)),
            "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        }
    return result


def synthetic_recovery(
    train_rows: list[dict],
    fitted_model: dict,
    repeats: int,
    seed: int,
) -> dict:
    """
    Numerical stress test only. It is not additional empirical evidence and is
    never used to remove real samples or inspect the real validation labels.
    """
    if repeats <= 0:
        return {"repeats": 0}

    p_emp = np.asarray([r["p"] for r in train_rows], dtype=float)
    n_emp = np.asarray([r["n_cand"] for r in train_rows], dtype=int)
    rng = np.random.default_rng(seed)
    n_train = len(train_rows)
    n_test = max(300, n_train // 2)

    recovered = {"alpha0": [], "alphaN": [], "c0": [], "cN": [], "nll": []}

    def true_prob(p: np.ndarray, n: np.ndarray) -> np.ndarray:
        phi = (
            np.log1p(n.astype(float)) - float(fitted_model["mu_phi"])
        ) / float(fitted_model["sigma_phi"])
        a_scene = np.exp(
            np.clip(
                float(fitted_model["alpha0"]) + float(fitted_model["alphaN"]) * phi,
                -5.0,
                5.0,
            )
        )
        x = np.log(p / (1.0 - p))
        return sigmoid(
            a_scene * x + float(fitted_model["c0"]) + float(fitted_model["cN"]) * phi
        )

    for rep in range(repeats):
        idx = rng.integers(0, len(p_emp), size=n_train)
        p_tr, n_tr = p_emp[idx], n_emp[idx]
        y_tr = rng.binomial(1, true_prob(p_tr, n_tr))
        synthetic_rows = [
            {
                "image": f"syn_{rep}_{i}",
                "p": float(p_tr[i]),
                "n_cand": int(n_tr[i]),
                "y": int(y_tr[i]),
            }
            for i in range(n_train)
        ]
        model = fit_model(
            synthetic_rows,
            regularization=float(fitted_model["regularization"]),
            starts=3,
            seed=seed + rep + 1,
        )

        idx_te = rng.integers(0, len(p_emp), size=n_test)
        p_te, n_te = p_emp[idx_te], n_emp[idx_te]
        y_te = rng.binomial(1, true_prob(p_te, n_te))
        test_rows = [
            {
                "image": f"test_{rep}_{i}",
                "p": float(p_te[i]),
                "n_cand": int(n_te[i]),
                "y": int(y_te[i]),
            }
            for i in range(n_test)
        ]
        y, _, q, _ = predict(test_rows, model)

        for key in ("alpha0", "alphaN", "c0", "cN"):
            recovered[key].append(float(model[key]))
        recovered["nll"].append(binary_nll(y, q))

    def summary(values: Iterable[float]) -> dict:
        a = np.asarray(list(values), dtype=float)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "q05": float(np.quantile(a, 0.05)),
            "q95": float(np.quantile(a, 0.95)),
        }

    return {
        "repeats": repeats,
        "note": "Parametric numerical recovery test; not used to filter or tune the real validation set.",
        **{key: summary(values) for key, values in recovered.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--model-out", default="adaptive_platt_model.json")
    parser.add_argument("--report-out", default="adaptive_platt_report.json")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument(
        "--lambda-grid",
        default="0,0.001,0.01,0.1,1,10,100",
        help="Comma-separated L2 penalties for alphaN and cN",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--stress-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    train = load_csv(args.train_csv)
    valid = load_csv(args.valid_csv)
    overlap = {r["image"] for r in train} & {r["image"] for r in valid}
    if overlap:
        raise ValueError(f"Scene leakage: {len(overlap)} image IDs occur in train and validation")

    lambda_candidates = [float(v.strip()) for v in args.lambda_grid.split(",") if v.strip()]
    chosen_lambda, lambda_cv = select_regularization(
        train, lambda_candidates, args.inner_folds, args.seed
    )
    model = fit_model(
        train, regularization=chosen_lambda, starts=7, seed=args.seed + 50
    )

    y_tr, raw_tr, cal_tr, slope_tr = predict(train, model)
    y_va, raw_va, cal_va, slope_va = predict(valid, model)

    report = {
        "model": model,
        "data": {
            "train_edges": len(train),
            "valid_edges": len(valid),
            "train_scenes": len({r["image"] for r in train}),
            "valid_scenes": len({r["image"] for r in valid}),
        },
        "regularization_selection": lambda_cv,
        "train": {"raw": metrics(y_tr, raw_tr), "calibrated": metrics(y_tr, cal_tr)},
        "validation": {"raw": metrics(y_va, raw_va), "calibrated": metrics(y_va, cal_va)},
        "generalization_gap": {
            "nll_valid_minus_train": binary_nll(y_va, cal_va) - binary_nll(y_tr, cal_tr),
            "brier_valid_minus_train": brier(y_va, cal_va) - brier(y_tr, cal_tr),
        },
        "scene_slope_summary": {
            "train_min": float(np.min(slope_tr)),
            "train_max": float(np.max(slope_tr)),
            "valid_min": float(np.min(slope_va)),
            "valid_max": float(np.max(slope_va)),
        },
        "monotonicity": monotonicity_check(
            model, [r["n_cand"] for r in train + valid]
        ),
        "validation_scene_bootstrap": scene_bootstrap(
            valid, model, args.bootstrap_repeats, args.seed + 101
        ),
        "synthetic_recovery": synthetic_recovery(
            train, model, args.stress_repeats, args.seed + 202
        ),
    }

    Path(args.model_out).write_text(json.dumps(model, indent=2), encoding="utf-8")
    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
