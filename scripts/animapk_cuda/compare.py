"""Metrics, stage tables, plots, and artifact writers for the ladder."""

from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np
import torch


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    cos = float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    diff = a - b
    rmse = float(diff.pow(2).mean().sqrt())
    rel_l2 = float(diff.norm() / (b.norm() + 1e-30))
    maxabs = float(diff.abs().max())
    return {
        "cosine": cos,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "max_abs": maxabs,
        "a_norm": float(a.norm()),
        "b_norm": float(b.norm()),
        "a_finite": bool(torch.isfinite(a).all()),
        "b_finite": bool(torch.isfinite(b).all()),
        "a_sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16],
        "b_sha256": hashlib.sha256(b.numpy().tobytes()).hexdigest()[:16],
    }


def tensor_norms(t: torch.Tensor) -> dict:
    t = t.detach().float()
    return {
        "l2": float(t.norm()),
        "max_abs": float(t.abs().max()),
        "mean": float(t.mean()),
        "std": float(t.std()),
        "finite": bool(torch.isfinite(t).all()),
    }


def stage_table(stages: list[str], cmp: dict[str, dict]) -> str:
    """Render a markdown stage parity table for a comparison bundle."""
    lines = ["| stage | cosine | RMSE | rel L2 | maxAbs | a_norm | b_norm | finite |", "|---|---|---|---|---|---|---|---|"]
    for s in stages:
        m = cmp.get(s, {})
        lines.append(
            f"| {s} | {m.get('cosine', float('nan')):.6f} | {m.get('rmse', float('nan')):.6e} "
            f"| {m.get('rel_l2', float('nan')):.4f} | {m.get('max_abs', float('nan')):.4f} "
            f"| {m.get('a_norm', float('nan')):.4f} | {m.get('b_norm', float('nan')):.4f} "
            f"| {m.get('a_finite', True) and m.get('b_finite', True)} |"
        )
    return "\n".join(lines)


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_md(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


def plot_ladder(out_png: str, stages: list[str], series: dict[str, dict[str, float]],
                title: str, ylabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = list(range(len(stages)))
    for label, vals in series.items():
        ys = [vals.get(s, float("nan")) for s in stages]
        ax.plot(xs, ys, marker="o", label=label, linewidth=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels(stages, rotation=90, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
