#!/usr/bin/env python3
"""
Inference for a trained N_cand-adaptive Platt calibrator.

Single value:
    python infer_adaptive_platt.py \
        --model adaptive_platt_model.json \
        --prob 0.95 \
        --n-cand 5

CSV:
    python infer_adaptive_platt.py \
        --model adaptive_platt_model.json \
        --input-csv valid_adaptive.csv \
        --output-csv valid_adaptive_calibrated.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

EPS_DEFAULT = 1e-6


def sigmoid(z: float) -> float:
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def calibrate(prob: float, n_cand: int, model: dict) -> float:
    if prob > 1.5:
        prob /= 100.0
    eps = float(model.get("eps", EPS_DEFAULT))
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0,1] or percentage form; got {prob}")
    if n_cand < 0:
        raise ValueError(f"N_cand must be nonnegative; got {n_cand}")

    p = min(max(prob, eps), 1.0 - eps)
    phi = (
        math.log1p(n_cand) - float(model["mu_phi"])
    ) / float(model["sigma_phi"])
    exponent = float(model["alpha0"]) + float(model["alphaN"]) * phi
    exponent = min(max(exponent, -5.0), 5.0)
    a_scene = math.exp(exponent)
    logit = math.log(p / (1.0 - p))
    z = a_scene * logit + float(model["c0"]) + float(model["cN"]) * phi
    return sigmoid(z)


def process_csv(
    input_csv: str,
    output_csv: str,
    model: dict,
    prob_column: str,
    ncand_column: str,
) -> None:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {prob_column, ncand_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    output_column = "p_adaptive_cal"
    if output_column not in fields:
        fields.append(output_column)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            p_cal = calibrate(
                float(row[prob_column]), int(float(row[ncand_column])), model
            )
            row[output_column] = f"{p_cal:.10f}"
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prob", type=float)
    group.add_argument("--input-csv")
    parser.add_argument("--n-cand", type=int)
    parser.add_argument("--output-csv")
    parser.add_argument("--prob-column", default="prob_edge")
    parser.add_argument("--ncand-column", default="N_cand")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if model.get("model_type") != "ncand_adaptive_platt":
        raise ValueError("The supplied model is not an ncand_adaptive_platt model")

    if args.prob is not None:
        if args.n_cand is None:
            raise ValueError("--n-cand is required with --prob")
        print(f"{calibrate(args.prob, args.n_cand, model):.10f}")
    else:
        if not args.output_csv:
            raise ValueError("--output-csv is required with --input-csv")
        process_csv(
            args.input_csv,
            args.output_csv,
            model,
            args.prob_column,
            args.ncand_column,
        )
        print(args.output_csv)


if __name__ == "__main__":
    main()
