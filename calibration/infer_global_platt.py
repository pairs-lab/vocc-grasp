#!/usr/bin/env python3
"""
Inference for a trained global Platt calibrator.

Single value:
    python infer_global_platt.py --model global_platt_model.json --prob 0.95

CSV:
    python infer_global_platt.py \
        --model global_platt_model.json \
        --input-csv valid_global.csv \
        --output-csv valid_global_calibrated.csv
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


def calibrate(prob: float, model: dict) -> float:
    if prob > 1.5:
        prob /= 100.0
    eps = float(model.get("eps", EPS_DEFAULT))
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0,1] or percentage form; got {prob}")
    p = min(max(prob, eps), 1.0 - eps)
    logit = math.log(p / (1.0 - p))
    return sigmoid(float(model["a"]) * logit + float(model["c"]))


def process_csv(input_csv: str, output_csv: str, model: dict, prob_column: str) -> None:
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if prob_column not in (reader.fieldnames or []):
            raise ValueError(f"Missing probability column: {prob_column}")
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    output_column = "p_global_cal"
    if output_column not in fields:
        fields.append(output_column)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row[output_column] = f"{calibrate(float(row[prob_column]), model):.10f}"
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prob", type=float)
    group.add_argument("--input-csv")
    parser.add_argument("--output-csv")
    parser.add_argument("--prob-column", default="prob_edge")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if model.get("model_type") != "global_platt":
        raise ValueError("The supplied model is not a global_platt model")

    if args.prob is not None:
        print(f"{calibrate(args.prob, model):.10f}")
    else:
        if not args.output_csv:
            raise ValueError("--output-csv is required with --input-csv")
        process_csv(args.input_csv, args.output_csv, model, args.prob_column)
        print(args.output_csv)


if __name__ == "__main__":
    main()
