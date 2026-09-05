#!/usr/bin/env python3
"""
Train global monotonic Platt calibration.

Formula:
    p_cal = sigmoid(a * logit(p_raw) + c),  a > 0

Example:
    python train_global_platt.py \
        --train-csv train_global.csv \
        --valid-csv valid_global.csv \
        --model-out global_platt_model.json \
        --report-out global_platt_report.json

Dependencies:
    numpy, scipy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

EPS = 1e-6


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def softplus(x: float) -> float:
    return float(np.logaddexp(0.0, x))


def inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def load_csv(path: str) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "prob_edge", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            p = float(row["prob_edge"])
            if p > 1.5:
                p /= 100.0
            y = int(float(row["label"]))
            if not 0.0 <= p <= 1.0 or y not in (0, 1):
                raise ValueError(f"{path}:{line_no}: invalid p/label")
            out.append({
                "image": str(row["image"]),
                "p": min(max(p, EPS), 1.0 - EPS),
                "y": y,
            })
    if not out:
        raise ValueError(f"{path}: no rows")
    return out


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray([r["p"] for r in rows], dtype=float)
    x = np.log(p / (1.0 - p))
    y = np.asarray([r["y"] for r in rows], dtype=float)
    return p, x, y


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


def fit_model(rows: list[dict], starts: int = 5, seed: int = 0) -> dict:
    _, x, y = arrays(rows)

    def objective(theta: np.ndarray) -> float:
        a = softplus(float(theta[0])) + EPS
        c = float(theta[1])
        return binary_nll(y, sigmoid(a * x + c))

    initial = [
        np.asarray([inv_softplus(0.70), -0.40]),
        np.asarray([inv_softplus(1.00), 0.00]),
    ]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, starts - len(initial))):
        initial.append(np.asarray([
            inv_softplus(float(rng.uniform(0.35, 1.30))),
            float(rng.normal(0.0, 0.7)),
        ]))

    best = None
    for x0 in initial:
        result = minimize(objective, x0=x0, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not best.success:
        raise RuntimeError(f"Optimization failed: {getattr(best, 'message', 'unknown error')}")

    return {
        "model_type": "global_platt",
        "formula": "sigmoid(a * logit(p_raw) + c)",
        "a": softplus(float(best.x[0])) + EPS,
        "c": float(best.x[1]),
        "eps": EPS,
        "train_nll": float(best.fun),
    }


def predict(rows: list[dict], model: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, x, y = arrays(rows)
    q = sigmoid(float(model["a"]) * x + float(model["c"]))
    return y, p, q


def monotonicity_check(model: dict) -> dict:
    grid = np.linspace(EPS, 1.0 - EPS, 20001)
    x = np.log(grid / (1.0 - grid))
    q = sigmoid(float(model["a"]) * x + float(model["c"]))
    diffs = np.diff(q)
    return {
        "passed": bool(np.all(diffs > 0.0)),
        "minimum_increment": float(np.min(diffs)),
        "output_min": float(np.min(q)),
        "output_max": float(np.max(q)),
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
        y, raw, cal = predict(boot, model)
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
    Numerical stress test only. It is not additional empirical evidence.
    It resamples the observed p distribution and generates labels from the fitted model.
    """
    if repeats <= 0:
        return {"repeats": 0}

    p_emp = np.asarray([r["p"] for r in train_rows], dtype=float)
    rng = np.random.default_rng(seed)
    fitted_a, fitted_c = float(fitted_model["a"]), float(fitted_model["c"])
    recovered_a, recovered_c, heldout_nll = [], [], []

    n_train = len(train_rows)
    n_test = max(300, n_train // 2)

    for rep in range(repeats):
        p_tr = rng.choice(p_emp, size=n_train, replace=True)
        x_tr = np.log(p_tr / (1.0 - p_tr))
        q_tr = sigmoid(fitted_a * x_tr + fitted_c)
        y_tr = rng.binomial(1, q_tr)

        synthetic_rows = [
            {"image": f"syn_{rep}_{i}", "p": float(p_tr[i]), "y": int(y_tr[i])}
            for i in range(n_train)
        ]
        model = fit_model(synthetic_rows, starts=3, seed=seed + rep + 1)

        p_te = rng.choice(p_emp, size=n_test, replace=True)
        x_te = np.log(p_te / (1.0 - p_te))
        q_true = sigmoid(fitted_a * x_te + fitted_c)
        y_te = rng.binomial(1, q_true)
        q_hat = sigmoid(float(model["a"]) * x_te + float(model["c"]))

        recovered_a.append(float(model["a"]))
        recovered_c.append(float(model["c"]))
        heldout_nll.append(binary_nll(y_te.astype(float), q_hat))

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
        "a": summary(recovered_a),
        "c": summary(recovered_c),
        "synthetic_holdout_nll": summary(heldout_nll),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--model-out", default="global_platt_model.json")
    parser.add_argument("--report-out", default="global_platt_report.json")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--stress-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    train = load_csv(args.train_csv)
    valid = load_csv(args.valid_csv)
    overlap = {r["image"] for r in train} & {r["image"] for r in valid}
    if overlap:
        raise ValueError(f"Scene leakage: {len(overlap)} image IDs occur in train and validation")

    model = fit_model(train, starts=7, seed=args.seed)
    y_tr, raw_tr, cal_tr = predict(train, model)
    y_va, raw_va, cal_va = predict(valid, model)

    report = {
        "model": model,
        "data": {
            "train_edges": len(train),
            "valid_edges": len(valid),
            "train_scenes": len({r["image"] for r in train}),
            "valid_scenes": len({r["image"] for r in valid}),
        },
        "train": {"raw": metrics(y_tr, raw_tr), "calibrated": metrics(y_tr, cal_tr)},
        "validation": {"raw": metrics(y_va, raw_va), "calibrated": metrics(y_va, cal_va)},
        "generalization_gap": {
            "nll_valid_minus_train": binary_nll(y_va, cal_va) - binary_nll(y_tr, cal_tr),
            "brier_valid_minus_train": brier(y_va, cal_va) - brier(y_tr, cal_tr),
        },
        "monotonicity": monotonicity_check(model),
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
