#!/usr/bin/env python3
"""Compare two Oracle-E compatible 84-checkpoint sample manifests.

Typical use:
  python -m scripts.ane_oracle_e_compare \
    --reference /oracle/E2/oracle_e_checkpoints.json \
    --candidate /device/unpacked/device_manifest.json \
    --out-dir /tmp/parity

No CUDA/Metal bit identity is assumed.  The report localizes the first
meaningful error jump using relative L2, cosine similarity, and max-absolute
error while preserving all 84 checkpoint metrics for planner review.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

BRANCHES = ("self", "cross", "mlp")
EXPECTED = 28 * len(BRANCHES)


def _records(manifest_path: Path) -> tuple[dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = doc.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{manifest_path} has no records list")
    indexed: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        if int(record.get("step", 0)) != 0:
            continue
        key = (int(record["block"]), str(record["branch"]))
        if key[0] not in range(28) or key[1] not in BRANCHES:
            raise ValueError(f"invalid checkpoint key {key} in {manifest_path}")
        if key in indexed:
            raise ValueError(f"duplicate checkpoint key {key} in {manifest_path}")
        indexed[key] = record
    if len(indexed) != EXPECTED:
        missing = [(b, br) for b in range(28) for br in BRANCHES if (b, br) not in indexed]
        raise ValueError(
            f"{manifest_path} has {len(indexed)}/{EXPECTED} step-0 checkpoints; missing={missing[:8]}")
    return doc, indexed


def _sample_path(manifest_path: Path, record: dict[str, Any]) -> Path:
    name = record.get("sample_file")
    if not isinstance(name, str) or not name:
        raise ValueError(f"checkpoint record in {manifest_path} lacks sample_file")
    path = manifest_path.parent / name
    if not path.is_file():
        raise ValueError(f"missing sample file: {path}")
    return path


def _load_sample(manifest_path: Path, record: dict[str, Any]) -> np.ndarray:
    sample = np.fromfile(_sample_path(manifest_path, record), dtype="<f4")
    expected = int(record.get("sample_count", record.get("sampleCount", sample.size)))
    if sample.size != expected:
        raise ValueError(
            f"sample count mismatch for block={record['block']} branch={record['branch']}: "
            f"file={sample.size} manifest={expected}")
    return sample.astype(np.float64, copy=False)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | int | bool]:
    if reference.shape != candidate.shape:
        raise ValueError(f"sample shape mismatch: {reference.shape} vs {candidate.shape}")
    ref_finite = np.isfinite(reference)
    cand_finite = np.isfinite(candidate)
    finite_pair = ref_finite & cand_finite
    nonfinite_mismatch = int(np.count_nonzero(ref_finite != cand_finite))
    both_nonfinite = int(np.count_nonzero(~ref_finite & ~cand_finite))
    if not finite_pair.any():
        return {
            "relative_rmse": math.inf,
            "cosine": math.nan,
            "max_abs_error": math.inf,
            "rmse": math.inf,
            "reference_l2": 0.0,
            "candidate_l2": 0.0,
            "finite_pairs": 0,
            "nonfinite_mismatch": nonfinite_mismatch,
            "both_nonfinite": both_nonfinite,
            "all_finite": False,
        }
    a = reference[finite_pair]
    b = candidate[finite_pair]
    diff = b - a
    diff_sq = float(np.dot(diff, diff))
    ref_sq = float(np.dot(a, a))
    cand_sq = float(np.dot(b, b))
    rmse = math.sqrt(diff_sq / a.size)
    relative_rmse = math.sqrt(diff_sq / max(ref_sq, np.finfo(np.float64).tiny))
    denom = math.sqrt(ref_sq * cand_sq)
    cosine = float(np.dot(a, b)) / denom if denom else (1.0 if diff_sq == 0 else 0.0)
    return {
        "relative_rmse": relative_rmse,
        "cosine": cosine,
        "max_abs_error": float(np.max(np.abs(diff), initial=0.0)),
        "rmse": rmse,
        "reference_l2": math.sqrt(ref_sq),
        "candidate_l2": math.sqrt(cand_sq),
        "finite_pairs": int(a.size),
        "nonfinite_mismatch": nonfinite_mismatch,
        "both_nonfinite": both_nonfinite,
        "all_finite": bool(ref_finite.all() and cand_finite.all()),
    }


def compare(
    reference_manifest: Path,
    candidate_manifest: Path,
    *,
    rel_threshold: float,
    cosine_threshold: float,
) -> dict[str, Any]:
    ref_doc, ref_records = _records(reference_manifest)
    cand_doc, cand_records = _records(candidate_manifest)
    rows: list[dict[str, Any]] = []
    previous_rel: float | None = None
    previous_key: str | None = None

    for block in range(28):
        for branch in BRANCHES:
            key = (block, branch)
            rr = ref_records[key]
            cr = cand_records[key]
            ref_stride = int(rr.get("sample_stride", rr.get("sampleStride", -1)))
            cand_stride = int(cr.get("sample_stride", cr.get("sampleStride", -1)))
            if ref_stride != cand_stride:
                raise ValueError(
                    f"sampling stride mismatch at block {block} {branch}: {ref_stride} vs {cand_stride}")
            reference = _load_sample(reference_manifest, rr)
            candidate = _load_sample(candidate_manifest, cr)
            m = _metrics(reference, candidate)
            rel = float(m["relative_rmse"])
            if previous_rel is None or not math.isfinite(rel) or not math.isfinite(previous_rel):
                delta = math.nan if previous_rel is None else math.inf
                ratio = math.nan if previous_rel is None else math.inf
            else:
                delta = rel - previous_rel
                ratio = rel / max(previous_rel, 1e-12)
            checkpoint = f"b{block:02d}.{branch}"
            row = {
                "ordinal": len(rows),
                "checkpoint": checkpoint,
                "block": block,
                "branch": branch,
                "sample_stride": ref_stride,
                **m,
                "relrmse_delta_from_previous": delta,
                "relrmse_ratio_to_previous": ratio,
                "previous_checkpoint": previous_key,
                "threshold_crossed": (
                    nonzero_bad := (
                        (math.isfinite(rel) and rel >= rel_threshold)
                        or (math.isfinite(float(m["cosine"])) and float(m["cosine"]) <= cosine_threshold)
                        or int(m["nonfinite_mismatch"]) > 0
                    )
                ),
            }
            # Keep a concrete bool; the assignment expression above exists only
            # to make the threshold definition visually indivisible.
            row["threshold_crossed"] = bool(nonzero_bad)
            rows.append(row)
            previous_rel = rel
            previous_key = checkpoint

    first_crossing = next((row for row in rows if row["threshold_crossed"]), None)
    finite_deltas = [
        row for row in rows[1:]
        if math.isfinite(float(row["relrmse_delta_from_previous"]))
    ]
    largest_jump = max(
        finite_deltas,
        key=lambda row: float(row["relrmse_delta_from_previous"]),
        default=None,
    )
    finite_ratios = [
        row for row in rows[1:]
        if math.isfinite(float(row["relrmse_ratio_to_previous"]))
    ]
    largest_ratio = max(
        finite_ratios,
        key=lambda row: float(row["relrmse_ratio_to_previous"]),
        default=None,
    )
    worst_rel = max(rows, key=lambda row: float(row["relative_rmse"]))
    finite_cos = [row for row in rows if math.isfinite(float(row["cosine"]))]
    worst_cos = min(finite_cos, key=lambda row: float(row["cosine"]), default=None)

    return {
        "schema": 1,
        "reference_manifest": str(reference_manifest.resolve()),
        "candidate_manifest": str(candidate_manifest.resolve()),
        "reference_mode": ref_doc.get("mode") or ref_doc.get("producer"),
        "candidate_mode": cand_doc.get("mode") or cand_doc.get("producer") or cand_doc.get("linear_backend"),
        "thresholds": {
            "relative_rmse_gte": rel_threshold,
            "cosine_lte": cosine_threshold,
            "nonfinite_mismatch_gt": 0,
            "note": "thresholds are localization heuristics, not CUDA/Metal bit-identity requirements",
        },
        "checkpoint_count": len(rows),
        "first_threshold_crossing": first_crossing,
        "largest_relrmse_absolute_jump": largest_jump,
        "largest_relrmse_ratio_jump": largest_ratio,
        "worst_relative_rmse": worst_rel,
        "worst_cosine": worst_cos,
        "rows": rows,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def write_report(result: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(result)
    (out_dir / "oracle_e_parity.json").write_text(
        json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "ordinal", "checkpoint", "block", "branch", "sample_stride",
        "relative_rmse", "cosine", "max_abs_error", "rmse",
        "reference_l2", "candidate_l2", "finite_pairs", "nonfinite_mismatch",
        "both_nonfinite", "all_finite", "relrmse_delta_from_previous",
        "relrmse_ratio_to_previous", "previous_checkpoint", "threshold_crossed",
    ]
    with (out_dir / "oracle_e_parity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])

    def checkpoint_summary(row: dict[str, Any] | None) -> str:
        if not row:
            return "none"
        return (
            f"`{row['checkpoint']}` — relRMSE={float(row['relative_rmse']):.6g}, "
            f"cos={float(row['cosine']):.8g}, maxAbs={float(row['max_abs_error']):.6g}"
        )

    lines = [
        "# ANE Oracle E parity report",
        "",
        f"- Reference: `{result['reference_manifest']}`",
        f"- Candidate: `{result['candidate_manifest']}`",
        f"- Checkpoints compared: **{result['checkpoint_count']} / {EXPECTED}**",
        "- Bit identity is **not** required across CUDA and A12. Metrics are used to localize where error begins/grows.",
        "",
        "## Localization",
        "",
        f"- First heuristic threshold crossing: {checkpoint_summary(result['first_threshold_crossing'])}",
        f"- Largest absolute relRMSE jump: {checkpoint_summary(result['largest_relrmse_absolute_jump'])}",
        f"- Largest relRMSE ratio jump: {checkpoint_summary(result['largest_relrmse_ratio_jump'])}",
        f"- Worst relRMSE: {checkpoint_summary(result['worst_relative_rmse'])}",
        f"- Worst cosine: {checkpoint_summary(result['worst_cosine'])}",
        "",
        "## All step-0 branch checkpoints",
        "",
        "| # | checkpoint | relRMSE | cosine | maxAbs | Δ relRMSE | ratio | nonfinite mismatch | threshold |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in result["rows"]:
        delta = row["relrmse_delta_from_previous"]
        ratio = row["relrmse_ratio_to_previous"]
        lines.append(
            f"| {row['ordinal']} | `{row['checkpoint']}` | {float(row['relative_rmse']):.6g} | "
            f"{float(row['cosine']):.8g} | {float(row['max_abs_error']):.6g} | "
            f"{float(delta):.6g} | {float(ratio):.6g} | {row['nonfinite_mismatch']} | "
            f"{'YES' if row['threshold_crossed'] else ''} |"
        )
    (out_dir / "ORACLE_E_PARITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--rel-threshold", type=float, default=0.10)
    ap.add_argument("--cosine-threshold", type=float, default=0.99)
    args = ap.parse_args()
    if args.rel_threshold < 0:
        raise SystemExit("--rel-threshold must be non-negative")
    if not -1 <= args.cosine_threshold <= 1:
        raise SystemExit("--cosine-threshold must be in [-1, 1]")
    result = compare(
        args.reference,
        args.candidate,
        rel_threshold=args.rel_threshold,
        cosine_threshold=args.cosine_threshold,
    )
    write_report(result, args.out_dir)
    first = result["first_threshold_crossing"]
    print(json.dumps({
        "checkpoint_count": result["checkpoint_count"],
        "first_threshold_crossing": first["checkpoint"] if first else None,
        "report": str(args.out_dir / "ORACLE_E_PARITY.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
