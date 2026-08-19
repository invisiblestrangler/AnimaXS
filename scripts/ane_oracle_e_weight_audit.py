#!/usr/bin/env python3
"""Audit ANE-native W8 projection quantization against legacy group64 W8.

The audit is intentionally streaming/bounded-memory: it reads a source
safetensors tensor in row chunks and applies the exact packer quantizers used
by the current AnimaXS repo.

Outputs:
  - JSON with per-tensor + aggregate metrics
  - CSV with one row per ANE-native tensor
  - Markdown summary with worst tensors and family/block aggregates
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from safetensors import safe_open

from scripts.pack_anima import (
    ANE_PROJECTION_SUFFIXES,
    _quantize_ane_row_chunk,
    _quantize_chunk,
    is_ane_projection,
)

BLOCK_RE = re.compile(r"^model\.diffusion_model\.blocks\.(\d+)\.(.+)$")


def blank_stats() -> dict[str, float]:
    return {
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "sum_squared_error": 0.0,
        "sum_absolute_error": 0.0,
        "max_absolute_error": 0.0,
        "element_count": 0.0,
        "q_zero_count": 0.0,
        "q_max_count": 0.0,
    }


def add_stats(dst: dict[str, float], src: dict[str, float]) -> None:
    for key in ("sum_x2", "sum_y2", "sum_xy", "sum_squared_error",
                "sum_absolute_error", "element_count",
                "q_zero_count", "q_max_count"):
        dst[key] += float(src[key])
    dst["max_absolute_error"] = max(
        float(dst["max_absolute_error"]), float(src["max_absolute_error"])
    )


def finalize(stats: dict[str, float]) -> dict[str, float]:
    n = max(float(stats["element_count"]), 1.0)
    x2 = max(float(stats["sum_x2"]), 0.0)
    y2 = max(float(stats["sum_y2"]), 0.0)
    denom = math.sqrt(x2 * y2)
    cosine = float(stats["sum_xy"]) / denom if denom > 0 else 1.0
    rel_l2 = math.sqrt(float(stats["sum_squared_error"]) / x2) if x2 > 0 else 0.0
    rmse = math.sqrt(float(stats["sum_squared_error"]) / n)
    return {
        **{k: float(v) for k, v in stats.items()},
        "cosine": cosine,
        "relative_l2": rel_l2,
        "rmse": rmse,
        "mean_absolute_error": float(stats["sum_absolute_error"]) / n,
        "q_zero_fraction": float(stats["q_zero_count"]) / n,
        "q_max_fraction": float(stats["q_max_count"]) / n,
        "q_saturation_fraction": (
            float(stats["q_zero_count"]) + float(stats["q_max_count"])
        ) / n,
    }


def tensor_family(name: str) -> tuple[int, str]:
    m = BLOCK_RE.match(name)
    if not m:
        raise ValueError(f"not a DiT block tensor: {name}")
    return int(m.group(1)), m.group(2)


def iter_native_names(handle: Any) -> list[str]:
    names = [name for name in handle.keys() if is_ane_projection(name)]
    names.sort()
    return names


def audit_tensor(handle: Any, name: str, row_chunk: int) -> dict[str, Any]:
    block, family = tensor_family(name)
    sl = handle.get_slice(name)
    shape = list(sl.get_shape())
    if len(shape) != 2:
        raise ValueError(f"ANE projection is not rank-2: {name} shape={shape}")
    rows, columns = int(shape[0]), int(shape[1])

    ane_stats = blank_stats()
    legacy_stats = blank_stats()

    for row0 in range(0, rows, row_chunk):
        row1 = min(rows, row0 + row_chunk)
        values = sl[row0:row1]
        if hasattr(values, "float"):
            values = values.float().cpu().numpy()
        else:
            values = np.asarray(values, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if values.shape != (row1 - row0, columns):
            raise RuntimeError(
                f"unexpected slice shape for {name}: {values.shape}, "
                f"expected {(row1 - row0, columns)}"
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"source tensor contains non-finite values: {name}")

        _, _, _, a = _quantize_ane_row_chunk(values)
        _, _, _, l = _quantize_chunk(values, bits=8, group=64, optimize_w4=False)
        add_stats(ane_stats, a)
        add_stats(legacy_stats, l)

    return {
        "name": name,
        "block": block,
        "family": family,
        "shape": [rows, columns],
        "elements": rows * columns,
        "ane_native": finalize(ane_stats),
        "legacy_group64": finalize(legacy_stats),
    }


def aggregate(records: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {"ane_native": blank_stats(), "legacy_group64": blank_stats()}
    )
    for rec in records:
        group = str(rec[key])
        for lane in ("ane_native", "legacy_group64"):
            add_stats(groups[group][lane], rec[lane])

    out: dict[str, Any] = {}
    for group, lanes in sorted(groups.items()):
        out[group] = {lane: finalize(stats) for lane, stats in lanes.items()}
    return out


def lane_summary(records: Iterable[dict[str, Any]], lane: str) -> dict[str, float]:
    stats = blank_stats()
    for rec in records:
        add_stats(stats, rec[lane])
    return finalize(stats)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "name", "block", "family", "rows", "columns", "elements",
        "ane_cosine", "ane_relative_l2", "ane_rmse", "ane_mae",
        "ane_max_abs_error", "ane_saturation_fraction",
        "legacy_cosine", "legacy_relative_l2", "legacy_rmse", "legacy_mae",
        "legacy_max_abs_error", "legacy_saturation_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rec in records:
            a = rec["ane_native"]
            l = rec["legacy_group64"]
            w.writerow({
                "name": rec["name"],
                "block": rec["block"],
                "family": rec["family"],
                "rows": rec["shape"][0],
                "columns": rec["shape"][1],
                "elements": rec["elements"],
                "ane_cosine": a["cosine"],
                "ane_relative_l2": a["relative_l2"],
                "ane_rmse": a["rmse"],
                "ane_mae": a["mean_absolute_error"],
                "ane_max_abs_error": a["max_absolute_error"],
                "ane_saturation_fraction": a["q_saturation_fraction"],
                "legacy_cosine": l["cosine"],
                "legacy_relative_l2": l["relative_l2"],
                "legacy_rmse": l["rmse"],
                "legacy_mae": l["mean_absolute_error"],
                "legacy_max_abs_error": l["max_absolute_error"],
                "legacy_saturation_fraction": l["q_saturation_fraction"],
            })


def pct(x: float) -> str:
    return f"{x * 100.0:.5f}%"


def fmt(x: float) -> str:
    if abs(x) >= 1e-3:
        return f"{x:.8f}"
    return f"{x:.4e}"


def write_markdown(path: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    worst_ane = sorted(records, key=lambda r: r["ane_native"]["relative_l2"], reverse=True)[:20]
    worst_legacy = sorted(records, key=lambda r: r["legacy_group64"]["relative_l2"], reverse=True)[:20]

    lines = [
        "# ANE Oracle E — Weight Reconstruction Audit",
        "",
        "This report compares the exact current AnimaXS ANE-native projection quantizer against the exact current legacy group64 W8 quantizer, both starting from the same pinned source weights.",
        "",
        "## Global reconstruction",
        "",
        "| Lane | Cosine | Relative L2 | RMSE | MAE | Max abs error | U8 saturation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, label in (
        ("ane_native", "ANE per-row U8 + FP32 scale/bias"),
        ("legacy_group64", "Legacy group64 U8 + FP16 scale/zero"),
    ):
        s = summary["global"][lane]
        lines.append(
            f"| {label} | {fmt(s['cosine'])} | {fmt(s['relative_l2'])} | "
            f"{fmt(s['rmse'])} | {fmt(s['mean_absolute_error'])} | "
            f"{fmt(s['max_absolute_error'])} | {pct(s['q_saturation_fraction'])} |"
        )

    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "These are **weight reconstruction metrics only**. They do not establish end-to-end image quality or runtime correctness. Oracle E step-0 must use the ANE-native reconstructed weights plus the actual ANE FP16 activation handoffs before device-vs-oracle conclusions are made.",
        "",
        "## Worst 20 ANE-native tensors by relative L2",
        "",
        "| Tensor | rel-L2 | cosine | max abs | saturation |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in worst_ane:
        s = rec["ane_native"]
        lines.append(
            f"| `{rec['name']}` | {fmt(s['relative_l2'])} | {fmt(s['cosine'])} | "
            f"{fmt(s['max_absolute_error'])} | {pct(s['q_saturation_fraction'])} |"
        )

    lines += [
        "",
        "## Worst 20 legacy-W8 tensors by relative L2",
        "",
        "| Tensor | rel-L2 | cosine | max abs | saturation |",
        "|---|---:|---:|---:|---:|",
    ]
    for rec in worst_legacy:
        s = rec["legacy_group64"]
        lines.append(
            f"| `{rec['name']}` | {fmt(s['relative_l2'])} | {fmt(s['cosine'])} | "
            f"{fmt(s['max_absolute_error'])} | {pct(s['q_saturation_fraction'])} |"
        )

    lines += [
        "",
        "## Per-family aggregates",
        "",
        "| Family | ANE rel-L2 | Legacy rel-L2 | ANE cosine | Legacy cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for family, lanes in summary["by_family"].items():
        a = lanes["ane_native"]
        l = lanes["legacy_group64"]
        lines.append(
            f"| `{family}` | {fmt(a['relative_l2'])} | {fmt(l['relative_l2'])} | "
            f"{fmt(a['cosine'])} | {fmt(l['cosine'])} |"
        )

    lines += [
        "",
        "## Per-block aggregates",
        "",
        "| Block | ANE rel-L2 | Legacy rel-L2 | ANE cosine | Legacy cosine |",
        "|---:|---:|---:|---:|---:|",
    ]
    for block, lanes in summary["by_block"].items():
        a = lanes["ane_native"]
        l = lanes["legacy_group64"]
        lines.append(
            f"| {block} | {fmt(a['relative_l2'])} | {fmt(l['relative_l2'])} | "
            f"{fmt(a['cosine'])} | {fmt(l['cosine'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--row-chunk", type=int, default=128)
    ap.add_argument("--source-sha256")
    ap.add_argument("--source-repo")
    ap.add_argument("--source-revision")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with safe_open(args.source, framework="pt", device="cpu") as handle:
        names = iter_native_names(handle)
        expected = 28 * len(ANE_PROJECTION_SUFFIXES)
        if len(names) != expected:
            raise RuntimeError(f"expected {expected} ANE-native projections, found {len(names)}")
        records: list[dict[str, Any]] = []
        for index, name in enumerate(names, 1):
            print(f"[{index:03d}/{len(names)}] {name}", flush=True)
            records.append(audit_tensor(handle, name, args.row_chunk))

    summary = {
        "schema": 1,
        "source": {
            "path": str(args.source),
            "sha256": args.source_sha256,
            "repo": args.source_repo,
            "revision": args.source_revision,
        },
        "tensor_count": len(records),
        "global": {
            "ane_native": lane_summary(records, "ane_native"),
            "legacy_group64": lane_summary(records, "legacy_group64"),
        },
        "by_family": aggregate(records, "family"),
        "by_block": aggregate(records, "block"),
        "records": records,
    }

    (out / "ane_oracle_e_weight_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(out / "ane_oracle_e_weight_audit.csv", records)
    write_markdown(out / "ANE_ORACLE_E_WEIGHT_AUDIT.md", records, summary)

    global_a = summary["global"]["ane_native"]
    global_l = summary["global"]["legacy_group64"]
    print(
        "ANE global: "
        f"cos={global_a['cosine']:.10f} relL2={global_a['relative_l2']:.10f} "
        f"sat={global_a['q_saturation_fraction']:.8%}"
    )
    print(
        "Legacy global: "
        f"cos={global_l['cosine']:.10f} relL2={global_l['relative_l2']:.10f} "
        f"sat={global_l['q_saturation_fraction']:.8%}"
    )


if __name__ == "__main__":
    main()
