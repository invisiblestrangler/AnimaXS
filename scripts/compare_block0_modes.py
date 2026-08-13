#!/usr/bin/env python3
"""Compare Metal block-0 branch updates with S/Q/E oracle outputs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SHAPE = (1024, 2048)


def load(path):
    value = np.fromfile(path, dtype="<f4")
    if value.size != SHAPE[0] * SHAPE[1]:
        raise ValueError(f"{path}: unexpected element count {value.size}")
    return value.reshape(SHAPE)


def metrics(actual, reference):
    a = actual.reshape(-1).astype(np.float64)
    b = reference.reshape(-1).astype(np.float64)
    error = a - b
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "cosine": float(np.dot(a, b) / (an * bn)),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs": float(np.max(np.abs(error))),
        "relative_l2": float(np.linalg.norm(error) / bn),
        "actual_norm": float(an), "reference_norm": float(bn),
        "norm_ratio": float(an / bn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--emulation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, default=0)
    args = parser.parse_args()
    capture = Path(args.capture)
    prefix = f"w8-step0-block{args.block}"
    x0 = load(capture / f"{prefix}-input.f32")
    metal = {
        "self": load(capture / f"{prefix}-after-self.f32"),
        "cross": load(capture / f"{prefix}-after-cross.f32"),
        "mlp": load(capture / f"{prefix}-after-mlp.f32"),
    }
    modes = {}
    for label, root in (("S", args.source), ("Q", args.quant), ("E", args.emulation)):
        root = Path(root)
        modes[label] = {branch: load(root / f"block0_{name}.f32") for branch, name in
                        (("self", "x1"), ("cross", "x2"), ("mlp", "x3"))}
    previous_m = x0
    previous = {label: x0 for label in modes}
    rows = []
    summary = {}
    for branch in ("self", "cross", "mlp"):
        delta_m = metal[branch] - previous_m
        deltas = {label: values[branch] - previous[label] for label, values in modes.items()}
        comparisons = {
            "M_Q": metrics(delta_m, deltas["Q"]),
            "M_E": metrics(delta_m, deltas["E"]),
            "E_Q": metrics(deltas["E"], deltas["Q"]),
            "Q_S": metrics(deltas["Q"], deltas["S"]),
        }
        summary[branch] = comparisons
        for comparison, values in comparisons.items():
            rows.append({"branch": branch, "comparison": comparison, **values})
        previous_m = metal[branch]
        for label in modes: previous[label] = modes[label][branch]
    for branch, values in summary.items():
        runtime_gap = values["M_Q"]["relative_l2"]
        quant_gap = values["Q_S"]["relative_l2"]
        backend_gap = values["M_E"]["relative_l2"]
        if runtime_gap >= .005 and runtime_gap >= 2 * quant_gap:
            classification = "runtime-dominant"
        elif quant_gap >= .005 and quant_gap >= 2 * runtime_gap and backend_gap <= .005:
            classification = "quantization-dominant"
        else:
            classification = "inconclusive"
        values["classification"] = classification
        print(f"BLOCK={args.block} BRANCH={branch} CLASSIFICATION={classification} "
              f"RUNTIME_GAP={runtime_gap:.9g} QUANT_GAP={quant_gap:.9g} BACKEND_GAP={backend_gap:.9g}")
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    with open(output / "w8_block0_branch_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    with open(output / "w8_block0_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
